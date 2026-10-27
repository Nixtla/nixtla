__all__ = ["MLForecast"]


import copy
import warnings
import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)

import cloudpickle
import fsspec
import narwhals as nw
import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from sklearn.base import BaseEstimator, clone
from utilsforecast.compat import DataFrame, DFType, pl, pl_DataFrame, pl_Series

from mlforecast.core import (
    DateFeature,
    Freq,
    Lags,
    LagTransforms,
    Models,
    TargetTransform,
    TimeSeries,
    _get_model_name,
    _name_models,
    _validate_horizon_params,
)

from .grouped_array import GroupedArray

if TYPE_CHECKING:
    from mlforecast.lgb_cv import LightGBMCV
from .data_validation import _validate_no_null_times, validate_df
from .compat import CatBoostRegressor
from .target_transforms import _BaseGroupedArrayTargetTransform
from .utils import _resolve_num_threads
from .conformal_prediction import (
    Calibration,
    PredictionIntervals,
    ErrorScaling,
    Recalibration,
    TransferMethod,
    TransferResult,
    TransferState,
    _add_transfer_intervals,
    _apply_scale_alignment,
    _floor_series_scales,
    _robust_scale_ratio,
    _raw_series_scales,
    _rescale_interval_columns,
    get_conformal_method,
    compute_conformity_scores,
)

_get_conformal_method = get_conformal_method  # backward compat


def _ensure_h_int64(res):
    """Cast the ``h`` column to Int64, preserving the input backend.

    Deep-copies the pandas result so callers can mutate it without writing
    through to the cached ``fcst_fitted_values_`` (narwhals' ``with_columns``
    shares column buffers on pandas without copy-on-write).
    """
    res = (
        nw.from_native(res, eager_only=True)
        .with_columns(nw.col("h").cast(nw.Int64))
        .to_native()
    )
    return ufp.copy_if_pandas(res, deep=True)


def _train_ends(ts: TimeSeries) -> DataFrame:
    df_constructor = pl_DataFrame if isinstance(ts.uids, pl_Series) else pd.DataFrame
    return df_constructor({ts.id_col: ts.uids, "train_end": ts.last_dates})


_REFIT_FIX_H = (
    "pass PredictionIntervals(h=h) to the `prediction_intervals` "
    "argument when refitting the model"
)


def _check_scores_horizon(cs_h: int, h: int, fix_h: str) -> None:
    if cs_h != 1 and cs_h < h:
        raise ValueError(
            "The `h` of the conformity scores should be equal to one or "
            f"greater or equal to `h`. To fix it, {fix_h}."
        )
    if cs_h == 1 and h > 1:
        warnings.warn(
            "Prediction intervals are calculated using 1-step ahead cross-validation, "
            "with a constant width for all horizons. To vary the error by horizon, "
            f"{fix_h}.",
            UserWarning,
        )


def _check_out_of_sample(
    cutoffs: DataFrame, train_ends: DataFrame, id_col: str
) -> None:
    # compared in python because the ids of `df` may have another dtype
    ends = dict(zip(train_ends[id_col].to_list(), train_ends["train_end"].to_list()))
    details = "; ".join(
        f"'{uid}' was trained up to {ends[uid]} but its first window starts after {cutoff}"
        for uid, cutoff in zip(cutoffs[id_col].to_list(), cutoffs["cutoff"].to_list())
        if uid in ends and cutoff < ends[uid]
    )
    if details:
        raise ValueError(
            "The backtest windows of series seen during training must start after "
            "the end of their training data, otherwise the intervals are computed "
            "on in-sample forecasts and are too narrow. Pass more observations after "
            "the training end or reduce the `n_windows` or `step_size` of the transfer "
            "method, or the `h` of `Recalibration`. Series are matched to the training series by id, "
            "so rename the ids of unrelated series that reuse them. "
            f"Series with in-sample windows: {details}."
        )


def _frozen_backtest(
    fcst: "MLForecast",
    new_df: DFType,
    n_windows: int,
    h: int,
    step_size: int,
    train_ends: Optional[DataFrame],
) -> DFType:
    """Run inference-only backtesting windows using already-fitted models.

    Unlike cross_validation, never calls fit — uses source-trained models throughout.
    step_size=1 is safe here (no refitting means no leakage from overlapping windows).
    Windows are computed per series (``ufp.backtest_splits``) so that series with
    unaligned ends each get cutoffs relative to their own last timestamp.
    If `train_ends` is given, the windows of the series in it must start after
    their training end.
    """
    ts = fcst.ts
    id_col, time_col, target_col = ts.id_col, ts.time_col, ts.target_col
    # future values of dynamic inputs come from each validation window
    future_cols = ts._future_input_cols
    missing = [c for c in future_cols if c not in nw.from_native(new_df).columns]
    if missing:
        raise ValueError(
            f"`df` must contain the future values of {missing} "
            "to compute transfer intervals."
        )
    max_lag = ts.keep_last_n or (max(ts.lags) if ts.lags else 0)
    required = h + (n_windows - 1) * step_size + max_lag + 1
    counts_df = ufp.counts_by_id(new_df, id_col)
    nw_counts = nw.from_native(counts_df)
    min_count = int(nw_counts["counts"].min())
    if min_count < required:
        bad = nw_counts.filter(nw_counts["counts"] < required)
        details = "; ".join(
            f"'{uid}' has {cnt} time steps"
            for uid, cnt in zip(bad[id_col].to_list(), bad["counts"].to_list())
        )
        raise ValueError(
            "Adapting the prediction intervals to new series with a frozen-model "
            "backtest (the `Recalibration` and `ErrorScaling` methods of `transfer`) "
            f"requires >= {required} time steps per series "
            f"(h={h}, n_windows={n_windows}, step_size={step_size}, max_lag={max_lag}). "
            f"Series too short: {details}. For point forecasts only, use "
            "`transfer(df)` without a method."
        )

    all_results = []
    splits = ufp.backtest_splits(
        new_df,
        n_windows=n_windows,
        h=h,
        id_col=id_col,
        time_col=time_col,
        freq=fcst.freq,
        step_size=step_size,
    )
    for i, (cutoffs, train, valid) in enumerate(splits):
        if i == 0 and train_ends is not None:
            _check_out_of_sample(cutoffs, train_ends, id_col)
        X_df = valid[[id_col, time_col, *future_cols]] if future_cols else None
        preds = fcst._transfer(train, None).predict(h=h, X_df=X_df)
        preds = ufp.join(preds, cutoffs, on=id_col, how="left")
        joined = ufp.join(
            valid[[id_col, time_col, target_col]],
            preds,
            on=[id_col, time_col],
        )
        expected_rows = h * len(nw.from_native(cutoffs))
        if len(nw.from_native(joined)) != expected_rows:
            raise ValueError(
                "Frozen-model backtest predictions did not align with the "
                "validation windows. This usually means some series "
                "have gaps in their time index; please fill the gaps and retry."
            )
        sort_idxs = ufp.maybe_compute_sort_indices(joined, id_col, time_col)
        if sort_idxs is not None:
            joined = ufp.take_rows(joined, sort_idxs)
        all_results.append(joined)

    return nw.to_native(nw.concat([nw.from_native(r) for r in all_results]))


class MLForecast:
    # Calibration state, set by ``fit``, ``load`` and ``transfer``. The defaults
    # are what an instance never fit with intervals holds.
    prediction_intervals: Optional[PredictionIntervals] = None
    _calibration_: Optional[Calibration] = None
    _train_ends_: Optional[DataFrame] = None
    _transfer_: Optional[TransferState] = None

    def __init__(
        self,
        models: Models,
        freq: Freq,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[TargetTransform]] = None,
        lag_transforms_namer: Optional[Callable] = None,
        date_features_as_dummies: bool = False,
        drop_auxiliary_columns: Union[bool, Sequence[str]] = True,
    ):
        """Forecasting pipeline

        Args:
            models (regressor or list of regressors): Models that will be trained and used to compute the forecasts.
            freq (str or int or pd.offsets.BaseOffset): Pandas offset, pandas offset alias, e.g. 'D', 'W-THU' or integer denoting the frequency of the series.
            lags (list of int, optional): Lags of the target to use as features. Defaults to None.
            lag_transforms (dict of int to list of functions, optional): Mapping of target lags to their transformations. Defaults to None.
            date_features (list of str, callable or CalendarFeature, optional): Features computed from the dates. Can be calendar features from `mlforecast.date_features` (listed by `mlforecast.date_features.available()`), date attributes of the dataframe backend or functions that will take the dates as input. Defaults to None.
            num_threads (int): Number of threads to use when computing the features. Use -1 to use all available CPU cores. Defaults to 1.
            target_transforms (list of transformers, optional): Transformations that will be applied to the target before computing the features and restored after the forecasting step. Defaults to None.
            lag_transforms_namer (callable, optional): Function that takes a transformation (either function or class), a lag and extra arguments and produces a name. Defaults to None.
            date_features_as_dummies (bool): If True, date features with a known finite range (e.g. `mlforecast.date_features.month`, 'dayofweek') are expanded into binary indicator columns named '{feature}_{value}' instead of being kept as ordinal integers. Defaults to False.
            drop_auxiliary_columns (bool or list of str): Controls which columns used solely for grouping are excluded from the model feature matrix. True (default) drops all columns referenced in any groupby transform. False keeps all columns. A list of strings drops only the named columns explicitly. Changed in v1.0.4: default changed from False (keep all columns) to True (auto-drop groupby columns).
        """
        if not isinstance(models, dict) and not isinstance(models, list):
            models = [models]
        if isinstance(models, list):
            model_names = _name_models([_get_model_name(m) for m in models])
            models_with_names = dict(zip(model_names, models))
        else:
            models_with_names = models
        self.models = models_with_names
        num_threads = _resolve_num_threads(num_threads)
        self.ts = TimeSeries(
            freq=freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=date_features,
            num_threads=num_threads,
            target_transforms=target_transforms,
            lag_transforms_namer=lag_transforms_namer,
            date_features_as_dummies=date_features_as_dummies,
            drop_auxiliary_columns=drop_auxiliary_columns,
        )

    @property
    def horizon_features_(self) -> Dict[int, List[str]]:
        return self.ts.horizon_features_

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(models=[{', '.join(self.models.keys())}], "
            f"freq={self.freq}, "
            f"lag_features={list(self.ts.transforms.keys())}, "
            f"date_features={self.ts.date_features}, "
            f"num_threads={self.ts.num_threads})"
        )

    @property
    def freq(self):
        return self.ts.freq

    @classmethod
    def from_cv(cls, cv: "LightGBMCV") -> "MLForecast":
        """Create a forecaster from a fitted `LightGBMCV`.

        The forecaster uses the same features and `params` as the CV object, with the number of estimators set to its best iteration
        and the same `categorical_feature`, which `fit` uses unless `models_fit_kwargs` provides one for the model.
        Feature names in `categorical_feature` need the features as a DataFrame, so they can't be combined with `as_numpy=True`.

        Args:
            cv (LightGBMCV): Fitted CV object.

        Returns:
            MLForecast: Forecaster with a single `LGBMRegressor` model, named 'LGBMRegressor'.
        """
        if not hasattr(cv, "best_iteration_"):
            raise ValueError("LightGBMCV object must be fitted first.")
        from mlforecast.lgb_cv import _LGBMRegressor

        model = _LGBMRegressor(**{**cv.params, "n_estimators": cv.best_iteration_})
        model.categorical_feature = copy.copy(getattr(cv, "categorical_feature", None))
        return cls._from_ts(copy.deepcopy(cv.ts), models={"LGBMRegressor": model})

    @classmethod
    def _from_ts(
        cls,
        ts: TimeSeries,
        models: Models,
        *,
        models_: Optional[Dict[str, Any]] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
        calibration: Optional[Calibration] = None,
        train_ends: Optional[DataFrame] = None,
        transfer_: Optional[TransferState] = None,
    ) -> "MLForecast":
        """A forecaster around an already-built `TimeSeries`."""
        fcst = cls(models=models, freq=ts.freq)
        fcst.ts = ts
        if models_ is not None:
            fcst.models_ = models_
        fcst.prediction_intervals = prediction_intervals
        fcst._calibration_ = calibration
        fcst._train_ends_ = train_ends
        fcst._transfer_ = transfer_
        return fcst

    def _with_ts(self, ts: Optional[TimeSeries] = None) -> "MLForecast":
        """Shallow copy, optionally around `ts`.

        Whatever a nested fit, preprocess or predict leaves behind lands on the
        copy instead of on this instance.
        """
        out = copy.copy(self)
        if ts is not None:
            out.ts = ts
        return out

    def _validate_data(
        self,
        df: DataFrame,
        id_col: str,
        time_col: str,
    ) -> None:
        validate_df(df, id_col, time_col, self.freq)

    def _validate_data_or_warn(
        self,
        df: DataFrame,
        id_col: str,
        time_col: str,
        validate_data: bool,
    ) -> None:
        _validate_no_null_times(df, time_col)
        if validate_data:
            self._validate_data(df, id_col, time_col)
        else:
            has_pooled = any(
                getattr(v, "global_", False)
                or getattr(v, "groupby", None)
                or getattr(v, "partition_by", None)
                for v in self.ts.transforms.values()
            )
            if has_pooled:
                warnings.warn(
                    "Pooled (global/groupby/partition_by) lag transforms assume "
                    "a continuous, gap-free time grid. Data validation has been "
                    "disabled (validate_data=False), so timestamp gaps may "
                    "silently produce incorrect feature values. Consider "
                    "enabling validation or pre-validating your data.",
                    UserWarning,
                    stacklevel=3,
                )

    def _get_dynamic_exog_cols(
        self,
        df_columns: List[str],
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        weight_col: Optional[str] = None,
    ) -> List[str]:
        if static_features is None:
            static_cols = set(c for c in df_columns if c not in (time_col, target_col))
        else:
            static_cols = set(static_features)
            static_cols.add(id_col)
        if weight_col is not None:
            static_cols.discard(weight_col)
        lag_cols = set(self.ts.transforms.keys())
        date_cols = set(self.ts._date_feature_names)
        exclude = static_cols | lag_cols | date_cols | {id_col, time_col, target_col}
        if weight_col is not None:
            exclude.add(weight_col)
        return [c for c in df_columns if c not in exclude]

    def _resolve_horizon_features(
        self,
        df: DFType,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]],
        max_horizon: Optional[int],
        horizons: Optional[List[int]],
        horizon_features: Optional[Dict[int, List[str]]],
        horizon_feature_templates: Optional[List[str]],
        weight_col: Optional[str],
    ) -> Dict[int, List[str]]:
        if horizon_features is not None and horizon_feature_templates is not None:
            raise ValueError(
                "Only one of `horizon_features` and `horizon_feature_templates` can be provided."
            )
        if horizon_features is None and horizon_feature_templates is None:
            return {}
        if max_horizon is None and horizons is None:
            arg_name = (
                "`horizon_features`"
                if horizon_features is not None
                else "`horizon_feature_templates`"
            )
            raise ValueError(
                f"{arg_name} is only supported when using `max_horizon` or `horizons`."
            )
        _, effective_max_horizon = _validate_horizon_params(max_horizon, horizons)
        assert effective_max_horizon is not None
        exog_cols = self._get_dynamic_exog_cols(
            list(df.columns),
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            weight_col=weight_col,
        )
        exog_col_set = set(exog_cols)
        if horizon_features is not None:
            if not isinstance(horizon_features, dict):
                raise ValueError(
                    "`horizon_features` must be a dictionary that maps horizons to feature lists."
                )
            resolved = {}
            for horizon, cols in horizon_features.items():
                if not isinstance(horizon, int) or horizon <= 0:
                    raise ValueError(
                        "`horizon_features` keys must be positive integers."
                    )
                if horizon > effective_max_horizon:
                    raise ValueError(
                        f"`horizon_features` includes horizon {horizon}, but the maximum configured horizon is {effective_max_horizon}."
                    )
                if not isinstance(cols, list) or any(
                    not isinstance(col, str) for col in cols
                ):
                    raise ValueError(
                        "Each value in `horizon_features` must be a list of column names."
                    )
                if not cols:
                    warnings.warn(
                        f"`horizon_features` includes an empty list for horizon {horizon}. "
                        "This entry has no effect and can be removed.",
                        UserWarning,
                        stacklevel=2,
                    )
                resolved[horizon] = list(dict.fromkeys(cols))
        else:
            templates = list(horizon_feature_templates or [])
            compiled_patterns = []
            for template in templates:
                if not isinstance(template, str):
                    raise ValueError(
                        "All entries in `horizon_feature_templates` must be strings."
                    )
                if template.count("{h}") != 1:
                    raise ValueError(
                        "Each template in `horizon_feature_templates` must include exactly one '{h}' placeholder."
                    )
                pattern_str = (
                    "^"
                    + re.escape(template).replace(r"\{h\}", r"(?P<h>[1-9]\d*)")
                    + "$"
                )
                compiled_patterns.append(re.compile(pattern_str))
            resolved = {}
            matched_cols = set()
            for col in exog_cols:
                matched_horizon = None
                for compiled_pattern in compiled_patterns:
                    match = compiled_pattern.match(col)
                    if match is None:
                        continue
                    horizon = int(match.group("h"))
                    if horizon > effective_max_horizon:
                        raise ValueError(
                            f"Column '{col}' maps to horizon {horizon}, but the maximum configured horizon is {effective_max_horizon}."
                        )
                    if matched_horizon is not None and matched_horizon != horizon:
                        raise ValueError(
                            f"Column '{col}' matches multiple horizon templates with conflicting horizons."
                        )
                    matched_horizon = horizon
                if matched_horizon is not None:
                    resolved.setdefault(matched_horizon, []).append(col)
                    matched_cols.add(col)
            if templates and not matched_cols:
                raise ValueError(
                    "No dynamic exogenous columns matched `horizon_feature_templates`. "
                    "Please verify the template patterns."
                )
        unknown_cols = sorted(
            {
                col
                for cols in resolved.values()
                for col in cols
                if col not in exog_col_set
            }
        )
        if unknown_cols:
            raise ValueError(
                "The following `horizon_features` columns were not found among the dynamic exogenous features: "
                f"{unknown_cols}."
            )
        return resolved

    def preprocess(
        self,
        df: DFType,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        horizon_features: Optional[Dict[int, List[str]]] = None,
        horizon_feature_templates: Optional[List[str]] = None,
        return_X_y: bool = False,
        as_numpy: bool = False,
        weight_col: Optional[str] = None,
        validate_data: bool = True,
    ) -> Union[DFType, Tuple[DFType, np.ndarray]]:
        """Add the features to `data`.

        Args:
            df (pandas DataFrame): Series data in long format.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            static_features (list of str, optional): Names of the features that are static and will be repeated when forecasting. Defaults to None.
            dropna (bool): Drop rows with missing values produced by the transformations. Defaults to True.
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. Pooled lag transforms (global_/groupby/partition_by) with a window wider than this keep that wider window instead, since their shared aggregates have no per-series buffer to trim below it. Defaults to None.
            max_horizon (int, optional): Train this many models, where each model will predict a specific horizon. Defaults to None.
            horizons (list of int, optional): Train models only for specific horizons (1-indexed). Mutually exclusive with max_horizon. Defaults to None.
            horizon_features (dict of int to list of str, optional): Explicit mapping of 1-indexed horizons to dynamic exogenous columns.
                Only supported when using `max_horizon` or `horizons`. Defaults to None.
            horizon_feature_templates (list of str, optional): Template patterns for horizon-specific dynamic exogenous features.
                Each template must include exactly one '{h}' placeholder (1-indexed), for example: ['feature_h{h}'].
                Acts as shorthand for `horizon_features` and is only supported when using `max_horizon` or `horizons`. Defaults to None.
            return_X_y (bool): Return a tuple with the features and the target. If False will return a single dataframe. Defaults to False.
            as_numpy (bool): Cast features to numpy array. Only works for `return_X_y=True`. Defaults to False.
            weight_col (str, optional): Column that contains the sample weights. Defaults to None.
            validate_data (bool): Run data quality validations before preprocessing. Warns about missing dates and raises on duplicate rows. Defaults to True.

        Returns:
            DataFrame or tuple of pandas Dataframe and a numpy array: `df` plus added features and target(s).
        """
        # Run data validations if requested
        self._validate_data_or_warn(df, id_col, time_col, validate_data)
        resolved_horizon_features = self._resolve_horizon_features(
            df=df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            max_horizon=max_horizon,
            horizons=horizons,
            horizon_features=horizon_features,
            horizon_feature_templates=horizon_feature_templates,
            weight_col=weight_col,
        )

        prep = self.ts.fit_transform(
            df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            max_horizon=max_horizon,
            horizons=horizons,
            return_X_y=return_X_y,
            as_numpy=as_numpy,
            weight_col=weight_col,
            horizon_features=resolved_horizon_features,
        )
        self._drop_transfer_intervals()
        return prep

    def history_warmup(
        self,
        df: DFType,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        keep_last_n: Optional[int] = None,
        weight_col: Optional[str] = None,
        max_horizon: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        horizon_features: Optional[Dict[int, List[str]]] = None,
        horizon_feature_templates: Optional[List[str]] = None,
        as_numpy: Optional[bool] = None,
        validate_data: bool = True,
    ) -> "MLForecast":
        """Build the internal state from `df` without computing the features.

        Runs the same state-building steps as `preprocess` (series storage,
        target transforms, lag-transform buffers, pooled aggregates, feature
        order) but skips materializing the features dataframe, which makes it
        much cheaper when the features themselves aren't needed. Use it to warm
        an instance whose models are already trained -- e.g. after unpickling,
        `MLForecast.load`, or assigning externally trained models to `models_`
        -- so that `update` and `predict` can be used without calling `fit` or
        `preprocess` first.

        History is trimmed like in `fit`: `keep_last_n` (explicit or inferred
        from the transforms) is applied after the transform states are warmed,
        so only the observations required for updates are kept.

        Args:
            df (pandas or polars DataFrame): Series data in long format.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            static_features (list of str, optional): Names of the features that are static and will be repeated when forecasting. Defaults to None.
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Inferred from the transforms when None. Defaults to None.
            weight_col (str, optional): Column that contains the sample weights. Defaults to None.
            max_horizon (int, optional): Horizon the models were trained for when using one model per horizon. Leave None for recursive models; when None, any value from a previous fit is preserved. Defaults to None.
            horizons (list of int, optional): Specific 1-indexed horizons the models were trained for. Mutually exclusive with max_horizon. Defaults to None.
            horizon_features (dict of int to list of str, optional): Explicit mapping of 1-indexed horizons to dynamic exogenous columns the models were trained with. When None, any mapping from a previous fit is preserved. Defaults to None.
            horizon_feature_templates (list of str, optional): Template patterns for horizon-specific dynamic exogenous features, e.g. ['feature_h{h}']. Defaults to None.
            as_numpy (bool, optional): Whether prediction passes a numpy array to the model instead of a dataframe. When None, any value from a previous fit is preserved (False for a fresh instance). Defaults to None.
            validate_data (bool): Run data quality validations before warming up. Warns about missing dates and raises on duplicate rows. Defaults to True.

        Returns:
            MLForecast: Forecast object with the internal state built from `df`.
        """
        self._validate_data_or_warn(df, id_col, time_col, validate_data)
        resolved_horizon_features = None
        if horizon_features is not None or horizon_feature_templates is not None:
            effective_max_horizon = max_horizon
            if effective_max_horizon is None and horizons is None:
                # preserve model-shape metadata from a previous fit (e.g. an
                # unpickled instance being re-warmed), same as TimeSeries.history_warmup
                effective_max_horizon = self.ts.max_horizon
            resolved_horizon_features = self._resolve_horizon_features(
                df=df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                max_horizon=effective_max_horizon,
                horizons=horizons,
                horizon_features=horizon_features,
                horizon_feature_templates=horizon_feature_templates,
                weight_col=weight_col,
            )
        elif self.ts.horizon_features_:
            # Preserving the mapping from a previous fit (the explicit branch
            # above re-validates it). `_fit` rebuilds `features_order_` from
            # `df.columns`, so a preserved mapping referencing exog columns
            # absent from `df` would otherwise surface as a raw ``KeyError``
            # deep inside ``predict``. Validate it eagerly here instead.
            exog_cols = set(
                self._get_dynamic_exog_cols(
                    list(df.columns),
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_features=static_features,
                    weight_col=weight_col,
                )
            )
            missing = {
                col for cols in self.ts.horizon_features_.values() for col in cols
            } - exog_cols
            if missing:
                raise ValueError(
                    "Preserved `horizon_features` reference columns missing from "
                    f"the warmup data's dynamic exogenous features: {sorted(missing)}. "
                    "Pass `horizon_features` explicitly to reconfigure the model "
                    "shape, or include the columns in `df`."
                )
        self.ts.history_warmup(
            df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            keep_last_n=keep_last_n,
            weight_col=weight_col,
            max_horizon=max_horizon,
            horizons=horizons,
            as_numpy=as_numpy,
            horizon_features=resolved_horizon_features,
        )
        self._drop_transfer_intervals()
        return self

    def _drop_transfer_intervals(self) -> None:
        # they were computed for the previous series
        if self._transfer_ is not None:
            self._transfer_ = TransferState()

    def fit_models(
        self,
        X: Union[DataFrame, np.ndarray, None] = None,
        y: Union[np.ndarray, None] = None,
        models_fit_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        generator_factory: Optional[Callable[[], Iterator]] = None,
    ) -> "MLForecast":
        """Manually train models. Use this if you called `MLForecast.preprocess` beforehand.

        Args:
            X (pandas or polars DataFrame or numpy array, optional): Features (for recursive forecasting).
            y (numpy array, optional): Target (for recursive forecasting).
            models_fit_kwargs (dict, optional): Keyword arguments for each model's fit method.
            generator_factory (callable, optional): Factory function that returns an iterator yielding (h, X_h, y_h) tuples per horizon.

        Returns:
            MLForecast: Forecast object with trained models.
        """
        models_fit_kwargs = models_fit_kwargs or {}
        # Check that all keys in models_fit_kwargs match known models
        unknown_models = set(models_fit_kwargs) - set(self.models)
        if unknown_models:
            raise ValueError(
                f"The following model names in `model_fit_kwargs` are not recognized: {unknown_models}. "
                f"Valid model names are: {list(self.models)}."
            )

        def fit_model(
            model, X, y, weight_col, model_fit_kwargs: Optional[dict[str, Any]]
        ):
            fit_kwargs = model_fit_kwargs or {}
            if weight_col is not None:
                if isinstance(X, np.ndarray):
                    fit_kwargs["sample_weight"] = X[:, 0]
                    X = X[:, 1:]
                else:
                    fit_kwargs["sample_weight"] = X[weight_col]
                    X = ufp.drop_columns(X, weight_col)
            if isinstance(model, CatBoostRegressor):
                # CatBoost consumes pandas, not polars; convert at the model input boundary
                if isinstance(X, pl_DataFrame):
                    X = X.to_pandas()
                sample_weight = fit_kwargs.get("sample_weight")
                if isinstance(sample_weight, pl_Series):
                    fit_kwargs["sample_weight"] = sample_weight.to_numpy()
            return clone(model).fit(X, y, **fit_kwargs)

        self.models_: Dict[str, Union[BaseEstimator, Dict[int, BaseEstimator]]] = {}

        if generator_factory is not None:
            # Direct forecasting with generator factory
            # Models are stored as dict keyed by horizon index (0-indexed)
            for name, model in self.models.items():
                model_fit_kwargs = models_fit_kwargs.get(name, None)
                self.models_[name] = {}
                # Create fresh generator for each model (generators are single-use)
                horizon_gen = generator_factory()
                for h, X_h, y_h in horizon_gen:
                    fitted = fit_model(
                        model, X_h, y_h, self.ts.weight_col, model_fit_kwargs
                    )
                    self.models_[name][h] = fitted
        else:
            # Recursive forecasting (unchanged)
            assert X is not None and y is not None, (
                "X and y are required when generator_factory is not provided"
            )
            for name, model in self.models.items():
                model_fit_kwargs = models_fit_kwargs.get(name, None)
                self.models_[name] = fit_model(
                    model, X, y, self.ts.weight_col, model_fit_kwargs
                )
        self._train_ends_ = _train_ends(self.ts)
        self._transfer_ = None
        return self

    def _conformity_scores(
        self,
        df: DFType,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        horizon_features: Optional[Dict[int, List[str]]] = None,
        horizon_feature_templates: Optional[List[str]] = None,
        n_windows: int = 2,
        h: int = 1,
        as_numpy: bool = False,
    ) -> DFType:
        """Compute conformity scores.

        We need at least two cross validation errors to compute
        quantiles for prediction intervals (`n_windows=2`).

        The exception is raised by the PredictionIntervals data class.

        In this simplest case, we assume the width of the interval
        is the same for all the forecasting horizon (`h=1`).
        """
        min_size = ufp.counts_by_id(df, id_col)["counts"].min()
        min_samples = h * n_windows + 1
        if min_size < min_samples:
            raise ValueError(
                "Minimum required samples in each serie for the prediction intervals "
                f"settings are: {min_samples}, shortest serie has: {min_size}. "
                "Please reduce the number of windows, horizon or remove those series."
            )
        # cross-validation fits the TimeSeries it runs on and leaves CV state
        # behind; run it on a copy so `self` only carries what `fit` sets afterwards
        scratch = self._with_ts(self.ts._clone_cold())
        cv_results = scratch.cross_validation(
            df=df,
            n_windows=n_windows,
            h=h,
            refit=False,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            dropna=dropna,
            keep_last_n=keep_last_n,
            max_horizon=max_horizon,
            horizons=horizons,
            horizon_features=horizon_features,
            horizon_feature_templates=horizon_feature_templates,
            prediction_intervals=None,
            as_numpy=as_numpy,
        )
        return compute_conformity_scores(
            cv_results, list(self.models.keys()), target_col
        )

    def _invert_transforms_fitted(self, df: DFType) -> DFType:
        if self.ts.target_transforms is None:
            return df
        if any(
            isinstance(tfm, _BaseGroupedArrayTargetTransform)
            for tfm in self.ts.target_transforms
        ):
            model_cols = [
                c for c in df.columns if c not in (self.ts.id_col, self.ts.time_col)
            ]
            id_counts = ufp.counts_by_id(df, self.ts.id_col)
            sizes = id_counts["counts"].to_numpy().astype(np.int64)
            indptr = np.append(0, sizes.cumsum())
        for tfm in self.ts.target_transforms[::-1]:
            if isinstance(tfm, _BaseGroupedArrayTargetTransform):
                if self.ts._dropped_series is not None:
                    idxs = np.delete(
                        np.arange(self.ts.ga.n_groups), self.ts._dropped_series
                    )
                    tfm = tfm.take(idxs)
                for col in model_cols:
                    ga = GroupedArray(df[col].to_numpy(), indptr)
                    ga = tfm.inverse_transform_fitted(ga)
                    df = ufp.assign_columns(df, col, ga.data)
            else:
                df = tfm.inverse_transform(df)
        return df

    def _extract_X_y(
        self,
        prep: DFType,
        target_col: str,
        weight_col: Optional[str],
    ) -> Tuple[Union[DFType, np.ndarray], np.ndarray]:
        x_cols = self.ts.features_order_
        if weight_col is not None:
            x_cols = [weight_col, *x_cols]
        X = prep[x_cols]
        targets = [c for c in prep.columns if re.match(rf"^{target_col}\d*$", c)]
        if len(targets) == 1:
            targets = targets[0]
        y = prep[targets].to_numpy()
        return X, y

    def _compute_fitted_values(
        self,
        base: DFType,
        X: Union[DFType, np.ndarray],
        y: np.ndarray,
        id_col: str,
        time_col: str,
        target_col: str,
        max_horizon: Optional[int],
        weight_col: Optional[str],
        original_df: Optional[DFType] = None,
    ) -> DFType:
        if weight_col is not None:
            if isinstance(X, np.ndarray):
                X = X[:, 1:]
            else:
                X = ufp.drop_columns(X, weight_col)
        base = ufp.copy_if_pandas(base, deep=False)
        sort_idxs = ufp.maybe_compute_sort_indices(base, id_col, time_col)
        if sort_idxs is not None:
            base = ufp.take_rows(base, sort_idxs)
            X = ufp.take_rows(X, sort_idxs)
            y = y[sort_idxs]
        if max_horizon is None:
            fitted_values = ufp.assign_columns(base, target_col, y)
            for name, model in self.models_.items():
                assert not isinstance(model, dict)  # mypy
                preds = model.predict(X)
                fitted_values = ufp.assign_columns(fitted_values, name, preds)
            fitted_values = self._invert_transforms_fitted(fitted_values)
        else:
            # Use the passed y array which has shape (n_rows, max_horizon)
            # y was already extracted from prep by _extract_X_y and contains expanded targets
            exog_cols = self.ts._get_dynamic_exog_cols(self.ts.features_order_)
            common_exog_cols, horizon_exog_map = self.ts._split_horizon_exog_cols(
                exog_cols, self.horizon_features_
            )

            # Only allocate entries for trained horizons (sparse or dense).
            trained_horizons = sorted({h for hm in self.models_.values() for h in hm})
            horizon_fitted_values = {}
            for h in trained_horizons:
                y_h = y[:, h] if y.ndim == 2 else y
                horizon_base = ufp.copy_if_pandas(base, deep=True)
                horizon_base = ufp.assign_columns(horizon_base, target_col, y_h)
                horizon_fitted_values[h] = horizon_base

            # Check if models were trained with numpy arrays
            models_trained_with_numpy = self.ts.as_numpy

            # Precompute per-horizon aligned features and validity masks once,
            # shared across all models to avoid redundant joins per model.
            x_cols: Optional[List[str]] = (
                list(X.columns) if hasattr(X, "columns") else None
            )
            feature_idx = {c: i for i, c in enumerate(self.ts.features_order_)}
            horizon_cache = {}  # h -> (X_h, valid_mask)
            for h in trained_horizons:
                y_h = y[:, h] if y.ndim == 2 else y
                valid_target = ~np.isnan(y_h)
                x_cols_h: Optional[List[str]]
                if self.horizon_features_:
                    x_cols_h = self.ts._get_cols_for_horizon(
                        h, common_exog_cols, horizon_exog_map, exog_cols
                    )
                else:
                    x_cols_h = x_cols
                exog_cols_h = (
                    [c for c in x_cols_h if c in exog_cols]
                    if x_cols_h is not None
                    else exog_cols
                )
                if not hasattr(X, "columns"):
                    if x_cols_h is not None:
                        col_idx = np.array(
                            [feature_idx[c] for c in x_cols_h], dtype=np.int32
                        )
                        X_h = X[:, col_idx]
                    else:
                        X_h = X
                    horizon_cache[h] = (X_h, valid_target)
                    continue

                if h == 0 or not exog_cols_h or original_df is None:
                    if x_cols_h is not None:
                        X_h = X[x_cols_h]
                    else:
                        X_h = X
                    horizon_cache[h] = (X_h, valid_target)
                else:
                    non_exog_cols = (
                        [c for c in x_cols_h if c not in exog_cols]
                        if x_cols_h is not None
                        else None
                    )
                    offset_times = ufp.offset_times(base[time_col], self.ts.freq, h)

                    if isinstance(base, pl_DataFrame):
                        lookup_df = (
                            base[[id_col]]
                            .with_columns(pl_Series("_offset_time", offset_times))
                            .with_row_index("_row_idx")
                        )
                        exog_source = original_df[[id_col, time_col] + exog_cols_h]
                        exog_renamed = exog_source.rename({time_col: "_offset_time"})
                        merged = lookup_df.join(
                            exog_renamed, on=[id_col, "_offset_time"], how="left"
                        )
                        merged = merged.sort("_row_idx")
                        if non_exog_cols:
                            X_h = X[non_exog_cols]
                            for col in exog_cols_h:
                                X_h = X_h.with_columns(merged[col].alias(col))
                            X_h = X_h[x_cols_h]
                        else:
                            X_h = X
                    else:
                        lookup_df = base[[id_col]].copy()
                        lookup_df["_offset_time"] = offset_times
                        lookup_df["_row_idx"] = np.arange(len(base))
                        exog_source = original_df[[id_col, time_col] + exog_cols_h]
                        exog_renamed = exog_source.rename(
                            columns={time_col: "_offset_time"}
                        )
                        merged = lookup_df.merge(
                            exog_renamed, on=[id_col, "_offset_time"], how="left"
                        )
                        merged = merged.sort_values("_row_idx")
                        if non_exog_cols:
                            X_h = X[non_exog_cols].copy()
                            for col in exog_cols_h:
                                X_h[col] = merged[col].values
                            X_h = X_h[x_cols_h]
                        else:
                            X_h = X

                    valid = valid_target.copy()
                    for col in exog_cols_h:
                        valid &= ~ufp.is_nan_or_none(X_h[col]).to_numpy()
                    horizon_cache[h] = (X_h, valid)

            for name, horizon_models in self.models_.items():
                for h, model in horizon_models.items():
                    X_h, valid = horizon_cache[h]
                    X_valid = ufp.filter_with_mask(X_h, valid)
                    X_pred = (
                        ufp.to_numpy(X_valid) if models_trained_with_numpy else X_valid
                    )
                    preds_valid = model.predict(X_pred)
                    preds = np.full(len(X_h), np.nan)
                    preds[valid] = preds_valid
                    horizon_fitted_values[h] = ufp.assign_columns(
                        horizon_fitted_values[h], name, preds
                    )

            for h, horizon_df in horizon_fitted_values.items():
                _, keep_mask = horizon_cache[h]
                horizon_df = ufp.filter_with_mask(horizon_df, keep_mask)
                horizon_df = ufp.copy_if_pandas(horizon_df, deep=True)
                horizon_df = self._invert_transforms_fitted(horizon_df)
                horizon_df = ufp.assign_columns(horizon_df, "h", np.int64(h + 1))
                horizon_fitted_values[h] = horizon_df
            fitted_values = ufp.vertical_concat(
                list(horizon_fitted_values.values()), match_categories=False
            )
        if self.ts.target_transforms is not None:
            for tfm in self.ts.target_transforms[::-1]:
                if hasattr(tfm, "store_fitted"):
                    tfm.store_fitted = False
                if hasattr(tfm, "fitted_"):
                    tfm.fitted_ = []
        return fitted_values

    def _compute_recursive_fitted_values_on_demand(
        self,
        h: int,
        train_df: Optional[DFType] = None,
    ) -> pd.DataFrame:
        # `h` and transform constraints are validated in `forecast_fitted_values`.
        # This intentionally mirrors the existing recursive prediction engine:
        # feature updates are vectorized across series at each step, but origins
        # are advanced one at a time to preserve parity with `predict`. A fully
        # origin-batched rollout would require a separate recursive engine.
        if train_df is None:
            if not hasattr(self, "_fitted_train_df_"):
                raise ValueError(
                    "Training data for multi-step fitted values was not cached. "
                    "Pass `train_df` to `forecast_fitted_values` or refit with "
                    "`cache_train_df=True`."
                )
            train_df = self._fitted_train_df_

        # pandas-only algorithm; converts to pandas at its boundary
        if isinstance(train_df, pd.DataFrame):
            train_pd = train_df.copy()
        else:
            train_pd = train_df.to_pandas()

        one_step = self.fcst_fitted_values_
        # pandas-only algorithm; converts to pandas at its boundary
        if isinstance(one_step, pd.DataFrame):
            one_step_pd = one_step[[self.ts.id_col, self.ts.time_col]].copy()
        else:
            one_step_pd = one_step.select(
                [self.ts.id_col, self.ts.time_col]
            ).to_pandas()

        id_col = self.ts.id_col
        time_col = self.ts.time_col
        target_col = self.ts.target_col
        train_pd = train_pd.sort_values([id_col, time_col]).reset_index(drop=True)

        valid_one_step_times = {
            uid: set(group[time_col].tolist())
            for uid, group in one_step_pd.groupby(id_col, observed=True)
        }
        static = [c for c in self.ts.static_features_.columns if c != id_col]
        exclude = {id_col, time_col, target_col, *static}
        if self.ts.weight_col is not None:
            exclude.add(self.ts.weight_col)
        dynamic = [c for c in train_pd.columns if c not in exclude]
        model_names = list(self.models_.keys())
        # pandas-only algorithm; converts to pandas at its boundary
        if isinstance(self.ts.static_features_, pd.DataFrame):
            static_features_pd = self.ts.static_features_.copy(deep=True)
        else:
            static_features_pd = self.ts.static_features_.to_pandas()

        rows = []
        for uid, group in train_pd.groupby(id_col, observed=True):
            group = group.sort_values(time_col).reset_index(drop=True)
            valid_uid_times = valid_one_step_times.get(uid, set())
            if not valid_uid_times:
                continue

            valid_origins = []
            for origin_idx in range(len(group) - h):
                first_fcst_time = group.iloc[origin_idx + 1][time_col]
                if first_fcst_time not in valid_uid_times:
                    continue
                valid_origins.append(origin_idx)
            if not valid_origins:
                continue

            # Fit once on the first valid origin and then move through origins with updates.
            first_origin = valid_origins[0]
            hist = group.iloc[: first_origin + 1]
            hist = hist[[id_col, time_col, target_col, *dynamic]]
            # a one-series instance wearing the parent's statics and feature
            # order, since `hist` carries neither
            temp_ts = self.ts._clone_warm(hist, static_features=[id_col])
            temp_ts.static_features_ = static_features_pd[
                static_features_pd[id_col].eq(uid)
            ].reset_index(drop=True)
            temp_ts.static_features = self.ts.static_features
            temp_ts.features_order_ = list(self.ts.features_order_)

            current_origin = first_origin
            for origin_idx in valid_origins:
                if origin_idx > current_origin:
                    # Advance the state to the current origin with observed values.
                    for update_idx in range(current_origin + 1, origin_idx + 1):
                        obs = group.iloc[[update_idx]][[id_col, time_col, target_col]]
                        temp_ts.update(obs)
                    current_origin = origin_idx

                future = group.iloc[origin_idx + 1 : origin_idx + h + 1]
                if future.shape[0] < h:
                    continue
                X_df = None
                if dynamic:
                    X_df = future[[id_col, time_col, *dynamic]]
                preds = temp_ts.predict(models=self.models_, horizon=h, X_df=X_df)
                pred_last = preds.iloc[-1]
                target_idx = origin_idx + h
                row = {
                    id_col: group.iloc[target_idx, group.columns.get_loc(id_col)],
                    time_col: group.iloc[target_idx, group.columns.get_loc(time_col)],
                    target_col: group.iloc[
                        target_idx, group.columns.get_loc(target_col)
                    ],
                    "h": h,
                }
                for model_name in model_names:
                    row[model_name] = pred_last[model_name]
                rows.append(row)

        cols = [id_col, time_col, target_col, "h", *model_names]
        if not rows:
            return pd.DataFrame(columns=cols)
        result = pd.DataFrame(rows)[cols]
        result = result.sort_values([id_col, time_col]).reset_index(drop=True)
        return result

    def fit(
        self,
        df: DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        horizon_features: Optional[Dict[int, List[str]]] = None,
        horizon_feature_templates: Optional[List[str]] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
        fitted: bool = False,
        as_numpy: bool = False,
        weight_col: Optional[str] = None,
        models_fit_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        validate_data: bool = True,
        cache_train_df: bool = True,
    ) -> "MLForecast":
        """Apply the feature engineering and train the models.

        Args:
            df (pandas or polars DataFrame): Series data in long format.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            static_features (list of str, optional): Names of the features that are static and will be repeated when forecasting. If `None`, will consider all columns (except id_col and time_col) as static. Defaults to None.
            dropna (bool): Drop rows with missing values produced by the transformations. Defaults to True.
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. Pooled lag transforms (global_/groupby/partition_by) with a window wider than this keep that wider window instead, since their shared aggregates have no per-series buffer to trim below it. Defaults to None.
            max_horizon (int, optional): Train this many models, where each model will predict a specific horizon. Defaults to None.
            horizons (list of int, optional): Train models only for specific horizons (1-indexed). For example, `horizons=[7, 14]` trains models only for steps 7 and 14. Mutually exclusive with max_horizon. Defaults to None.
            horizon_features (dict of int to list of str, optional): Explicit mapping of 1-indexed horizons to dynamic exogenous columns.
                Only supported when using `max_horizon` or `horizons`. Defaults to None.
            horizon_feature_templates (list of str, optional): Template patterns for horizon-specific dynamic exogenous features.
                Each template must include exactly one '{h}' placeholder (1-indexed), for example: ['feature_h{h}'].
                Acts as shorthand for `horizon_features` and is only supported when using `max_horizon` or `horizons`. Defaults to None.
            prediction_intervals (PredictionIntervals, optional): Configuration to calibrate prediction intervals (Conformal Prediction). Defaults to None.
            fitted (bool): Save in-sample predictions. Defaults to False.
            as_numpy (bool): Cast features to numpy array. Defaults to False.
            weight_col (str, optional): Column that contains the sample weights. Defaults to None.
            models_fit_kwargs (dict, optional): Keyword arguments for each model's fit method. Defaults to None.
            validate_data (bool): Run data quality validations before fitting. Warns about missing dates and raises on duplicate rows. Defaults to True.
            cache_train_df (bool): Cache a copy of the training data when `fitted=True` so
                `forecast_fitted_values(h>1)` can be called later for recursive models without
                passing `train_df`. Disable this to avoid the memory overhead and pass
                `train_df` directly to `forecast_fitted_values` when needed. Defaults to True.

        Returns:
            MLForecast: Forecast object with series values and trained models.
        """
        if fitted and self.ts.target_transforms is not None:
            for tfm in self.ts.target_transforms:
                if hasattr(tfm, "store_fitted"):
                    tfm.store_fitted = True
        self.prediction_intervals = prediction_intervals
        self._calibration_ = None
        if prediction_intervals is not None:
            scores = self._conformity_scores(
                df=df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                dropna=dropna,
                keep_last_n=keep_last_n,
                max_horizon=max_horizon,
                horizons=horizons,
                horizon_features=horizon_features,
                horizon_feature_templates=horizon_feature_templates,
                n_windows=prediction_intervals.n_windows,
                h=prediction_intervals.h,
                as_numpy=as_numpy,
            )
            source_scales = _raw_series_scales(
                df=df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                methods=("mad", "std"),
            )
            self._calibration_ = Calibration(scores=scores, source_scales=source_scales)
        # When max_horizon or horizons is set, use generator factory approach
        if max_horizon is not None or horizons is not None:
            # Preprocess to get DataFrame with expanded targets
            prep = self.preprocess(
                df=df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                dropna=dropna,
                keep_last_n=keep_last_n,
                max_horizon=max_horizon,
                horizons=horizons,
                horizon_features=horizon_features,
                horizon_feature_templates=horizon_feature_templates,
                return_X_y=False,
                as_numpy=as_numpy,
                weight_col=weight_col,
                validate_data=validate_data,
            )

            # Get the effective max horizon and internal horizons from preprocessing
            effective_max_horizon = self.ts.max_horizon
            internal_horizons = self.ts._horizons
            assert internal_horizons is not None

            # Store original df for exog lookup in generator
            original_df = df

            # Factory function - creates fresh generator for each model
            def generator_factory():
                return self.ts._transform_per_horizon(
                    prep, original_df, internal_horizons, target_col, as_numpy
                )

            # Train models using generator factory
            self.fit_models(
                generator_factory=generator_factory, models_fit_kwargs=models_fit_kwargs
            )

            if fitted:
                assert not isinstance(
                    prep, tuple
                )  # return_X_y=False ensures prep is a DataFrame
                base = prep[[id_col, time_col]]
                X, y = self._extract_X_y(prep, target_col, weight_col)
                # Keep X as DataFrame for exog alignment in _compute_fitted_values
                # (don't convert to numpy even if as_numpy=True)
                fitted_values = self._compute_fitted_values(
                    base=base,
                    X=X,
                    y=y,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    max_horizon=effective_max_horizon,
                    weight_col=self.ts.weight_col,
                    original_df=original_df,
                )
                fitted_values = ufp.drop_index_if_pandas(fitted_values)
                self.fcst_fitted_values_ = fitted_values
                if hasattr(self, "_fitted_train_df_"):
                    delattr(self, "_fitted_train_df_")
        else:
            # Standard recursive path (unchanged)
            prep = self.preprocess(
                df=df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=static_features,
                dropna=dropna,
                keep_last_n=keep_last_n,
                max_horizon=max_horizon,
                horizon_features=horizon_features,
                horizon_feature_templates=horizon_feature_templates,
                return_X_y=not fitted,
                as_numpy=as_numpy,
                weight_col=weight_col,
                validate_data=validate_data,
            )
            if isinstance(prep, tuple):
                X, y = prep
            else:
                base = prep[[id_col, time_col]]
                X, y = self._extract_X_y(prep, target_col, weight_col)
                if as_numpy:
                    X = ufp.to_numpy(X)
                del prep
            self.fit_models(X, y, models_fit_kwargs)
            if fitted:
                fitted_values = self._compute_fitted_values(
                    base=base,
                    X=X,
                    y=y,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    max_horizon=max_horizon,
                    weight_col=self.ts.weight_col,
                )
                fitted_values = ufp.drop_index_if_pandas(fitted_values)
                self.fcst_fitted_values_ = fitted_values
                if cache_train_df:
                    self._fitted_train_df_ = ufp.copy_if_pandas(df, deep=True)
                elif hasattr(self, "_fitted_train_df_"):
                    delattr(self, "_fitted_train_df_")
        return self

    def forecast_fitted_values(
        self,
        level: Optional[List[Union[int, float]]] = None,
        *,
        h: int = 1,
        train_df: Optional[DFType] = None,
    ) -> DataFrame:
        """Access in-sample predictions.

        Args:
            level (list of ints or floats, optional): Confidence levels between 0 and 100 for prediction intervals. Defaults to None.
            h (int): Forecast horizon for fitted values. Defaults to 1.
                For recursive models and ``h>1``, values are computed on demand and are not available
                when global or grouped lag transforms are configured.
            train_df (pandas or polars DataFrame, optional): Training data to use when computing
                recursive fitted values for ``h>1`` on demand. Pass this to avoid caching a full
                copy of the training data on the fitted object. Defaults to None.

        Returns:
            pandas or polars DataFrame: DataFrame with the following columns:

            - ``id_col``: series identifier.
            - ``time_col``: timestamp of the predicted observation.
            - ``target_col``: actual observed value.
            - ``h``: number of steps ahead the prediction was made. For recursive models
              this equals the ``h`` argument. For direct models (``max_horizon``) it
              reflects the specific horizon step (1-indexed) at which each row was
              predicted, ranging from 1 to ``max_horizon``.
            - One column per model with the fitted (in-sample) predictions.
            - If ``level`` is provided, additional columns with the lower and upper
              bounds of the prediction intervals for each model and confidence level.
        """
        if not hasattr(self, "fcst_fitted_values_"):
            raise Exception("Please run the `fit` method using `fitted=True`")
        if not isinstance(h, int) or h < 1:
            raise ValueError("`h` must be a positive integer.")
        if not self.models_:
            if h != 1:
                raise ValueError(
                    "No fitted models available to compute multi-step fitted values."
                )
            res = self.fcst_fitted_values_
            if "h" not in res.columns:
                res = ufp.assign_columns(res, "h", np.int64(h))
            return _ensure_h_int64(res)

        first_model = next(iter(self.models_.values()))
        is_direct = isinstance(first_model, dict)
        if is_direct:
            res = self.fcst_fitted_values_
            if "h" not in res.columns:
                raise ValueError(
                    "Direct fitted values are missing horizon information. "
                    "Please refit the model with the current version."
                )
            res = ufp.drop_index_if_pandas(
                nw.to_native(
                    nw.from_native(res, eager_only=True).filter(nw.col("h") == h)
                )
            )
            available = sorted(
                nw.from_native(self.fcst_fitted_values_, eager_only=True)["h"]
                .unique()
                .to_list()
            )
            if len(res) == 0:
                raise ValueError(
                    f"No fitted values found for h={h}. Available horizons: {available}."
                )
        else:
            if h == 1:
                res = self.fcst_fitted_values_
            else:
                has_nonlocal = any(
                    mode != "local" for mode, _, _ in self.ts._pooled_states
                )
                if has_nonlocal:
                    raise ValueError(
                        "On-demand recursive fitted values for `h>1` are not supported when using "
                        "global or grouped lag transforms."
                    )
                warnings.warn(
                    "Computing recursive fitted values for h>1 on demand can be slow.",
                    UserWarning,
                    stacklevel=2,
                )
                res_pd = self._compute_recursive_fitted_values_on_demand(
                    h, train_df=train_df
                )
                # convert the pandas-only recursive result back to the origin backend
                if isinstance(self.fcst_fitted_values_, pd.DataFrame):
                    res = res_pd
                else:
                    res = pl.from_pandas(res_pd)
        if "h" not in res.columns:
            res = ufp.assign_columns(res, "h", np.int64(h))
        res = _ensure_h_int64(res)
        if level is not None:
            res = ufp.add_insample_levels(
                res,
                models=list(self.models_.keys()),
                level=level,
                id_col=self.ts.id_col,
                target_col=self.ts.target_col,
            )
        return res

    def make_future_dataframe(self, h: int) -> DataFrame:
        """Create a dataframe with all ids and future times in the forecasting horizon.

        Args:
            h (int): Number of periods to predict.

        Returns:
            pandas or polars DataFrame: DataFrame with expected ids and future times
        """
        if not hasattr(self.ts, "id_col"):
            raise ValueError("You must call fit first")
        return ufp.make_future_dataframe(
            uids=self.ts.uids,
            last_times=self.ts.last_dates,
            freq=self.freq,
            h=h,
            id_col=self.ts.id_col,
            time_col=self.ts.time_col,
        )

    def get_missing_future(self, h: int, X_df: DFType) -> DFType:
        """Get the missing id and time combinations in `X_df`.

        Args:
            h (int): Number of periods to predict.
            X_df (pandas or polars DataFrame): Dataframe with the future exogenous features. Should have the id column and the time column.

        Returns:
            pandas or polars DataFrame: DataFrame with expected ids and future times missing in `X_df`
        """
        expected = self.make_future_dataframe(h=h)
        ids = [self.ts.id_col, self.ts.time_col]
        return ufp.anti_join(expected, X_df[ids], on=ids)

    def _check_fitted(self) -> None:
        if not hasattr(self, "models_"):
            raise ValueError(
                "No fitted models found. You have to call fit or preprocess + fit_models. "
                "If you used cross_validation before please fit again."
            )
        first_model = next(iter(self.models_.values()))
        # Models are stored as dict for direct forecasting
        first_model_is_multi = isinstance(first_model, dict)
        max_horizon = self.ts.max_horizon
        if first_model_is_multi and max_horizon is None:
            raise ValueError(
                "Found one model per horizon but `max_horizon` is None. "
                "If you ran preprocess after fit please run fit again."
            )
        elif not first_model_is_multi and max_horizon is not None:
            raise ValueError(
                "Found a single model for all horizons "
                f"but `max_horizon` is {max_horizon}. "
                "If you ran preprocess after fit please run fit again."
            )

    def transfer(
        self,
        df: DFType,
        method: Optional[TransferMethod] = None,
    ) -> "MLForecast":
        """Forecaster that uses the trained models to continue the series in `df`.

        Args:
            df (pandas or polars DataFrame): Series data to continue from. Should have the same structure as the one used to fit the model, including any features.
            method (Recalibration, ErrorScaling or ScaleAlignment, optional): How to compute the prediction intervals of the series in `df`, from `df` when `transfer` is called. The backtest windows of series seen during training (matched by id) must start after the end of their training data. Defaults to None (no prediction intervals).

        Returns:
            MLForecast: Forecast object for the series in `df` that shares the trained models. This instance is not modified.
        """
        if method is not None and not isinstance(method, get_args(TransferMethod)):
            hint = (
                f" Pass an instance, e.g. `{method.__name__}()`."
                if isinstance(method, type)
                else ""
            )
            raise TypeError(
                "`method` must be a `Recalibration`, `ErrorScaling` or `ScaleAlignment` "
                f"from `mlforecast.conformal_prediction`, got {method!r}.{hint}"
            )
        return self._transfer(df, method)

    def _transfer(
        self,
        df: DFType,
        method: Optional[TransferMethod],
    ) -> "MLForecast":
        self._check_fitted()
        result = None if method is None else self._adapt(df, method)
        out = self._with_ts(self.ts._clone_warm(df))
        out.models_ = dict(self.models_)
        out._transfer_ = TransferState(result=result)
        return out

    def _adapt(self, df: DFType, method: TransferMethod) -> TransferResult:
        """Adapt the source calibration to `df`."""
        pi = self.prediction_intervals
        train_ends = self._train_ends_
        model_names = list(self.models.keys())
        if isinstance(method, Recalibration):
            if method.h is not None:
                h = method.h
            elif pi is not None:
                h = pi.h
            else:
                raise ValueError(
                    "`Recalibration` requires that the model was fit with "
                    "`prediction_intervals=PredictionIntervals(...)` or an explicit "
                    "`Recalibration(h=...)`."
                )
            n_windows = method.n_windows or (pi.n_windows if pi is not None else 2)
            backtest_results = _frozen_backtest(
                self, df, n_windows, h, method.step_size, train_ends
            )
            return TransferResult(
                cs_df=compute_conformity_scores(
                    backtest_results, model_names, self.ts.target_col, signed=True
                ),
                h=h,
                signed=True,
            )
        calibration = self._calibration_
        if pi is None or calibration is None:
            raise ValueError(
                f"`{type(method).__name__}` requires that the model was fit with "
                "`prediction_intervals=PredictionIntervals(...)`."
            )
        if isinstance(method, ErrorScaling):
            backtest_results = _frozen_backtest(
                self,
                df,
                method.n_windows or pi.n_windows,
                pi.h,
                method.step_size,
                train_ends,
            )
            target_cs_df = compute_conformity_scores(
                backtest_results, model_names, self.ts.target_col
            )
            scaled = ufp.copy_if_pandas(calibration.scores, deep=False)
            for model in model_names:
                src = calibration.scores[model].to_numpy().astype(float)
                tgt = target_cs_df[model].to_numpy().astype(float)
                scaled = ufp.assign_columns(
                    scaled, model, src * _robust_scale_ratio(src, tgt)
                )
            return TransferResult(cs_df=scaled, h=pi.h)
        estimator = method.scale_estimator
        raw_target = _raw_series_scales(
            df, self.ts.id_col, self.ts.time_col, self.ts.target_col, (estimator,)
        )
        target_scales = _floor_series_scales(
            raw_target[estimator], floor_factor=1e-3, stacklevel=5
        )
        source_scales = _floor_series_scales(
            calibration.source_scales[estimator],
            floor_factor=1e-3,
            stacklevel=5,
            kind="Training series",
        )
        return TransferResult(
            cs_df=_apply_scale_alignment(
                calibration.scores, model_names, self.ts.id_col, source_scales
            ),
            h=pi.h,
            target_scales=target_scales,
        )

    def predict(
        self,
        h: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        level: Optional[List[Union[int, float]]] = None,
        X_df: Optional[DFType] = None,
        ids: Optional[List[str]] = None,
    ) -> DFType:
        """Compute the predictions for the next `h` steps.

        To forecast other series with the trained models use `transfer`.

        Args:
            h (int): Number of periods to predict.
            before_predict_callback (callable, optional): Function to call on the features before computing the predictions. This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure. The series identifier is on the index. Defaults to None.
            after_predict_callback (callable, optional): Function to call on the predictions before updating the targets. This function will take a pandas Series with the predictions and should return another one with the same structure. The series identifier is on the index. Defaults to None.
            level (list of ints or floats, optional): Confidence levels between 0 and 100 for prediction intervals. Defaults to None.
            X_df (pandas or polars DataFrame, optional): Dataframe with the future exogenous features. Should have the id column and the time column. Defaults to None.
            ids (list of str, optional): List with subset of ids seen during training for which the forecasts should be computed. Defaults to None.

        Returns:
            pandas or polars DataFrame: Predictions for each serie and timestep, with one column per model.
        """
        self._check_fitted()
        state = self._transfer_
        if level is not None and state is not None and state.result is None:
            raise ValueError(
                "This transferred model has no prediction intervals for its "
                "current series. Call `transfer(df, method)` with a `Recalibration`, "
                "`ErrorScaling` or `ScaleAlignment` method to compute them."
            )
        forecasts = self.ts.predict(
            models=self.models_,
            horizon=h,
            before_predict_callback=before_predict_callback,
            after_predict_callback=after_predict_callback,
            X_df=X_df,
            ids=ids,
        )
        if level is not None:
            if state is not None:
                assert state.result is not None
                forecasts = self._add_intervals_from_transfer(
                    forecasts, state.result, sorted(level), h
                )
            elif self._calibration_ is None or self.prediction_intervals is None:
                warn_msg = (
                    "Please rerun the `fit` method passing a proper value "
                    "to prediction intervals to compute them."
                )
                warnings.warn(warn_msg, UserWarning)
            else:
                forecasts = self._add_intervals_from_fit(
                    forecasts, sorted(level), h, ids
                )
        return forecasts

    def _add_intervals_from_transfer(
        self,
        forecasts: DFType,
        result: TransferResult,
        level: List[Union[int, float]],
        h: int,
    ) -> DFType:
        if result.signed:
            fix_h = "pass `Recalibration(h=h)` to `transfer`"
            method = "signed"
        else:
            assert self.prediction_intervals is not None
            fix_h = _REFIT_FIX_H
            method = self.prediction_intervals.method
        _check_scores_horizon(result.h, h, fix_h)
        model_names = list(self.models.keys())
        # the scores are pooled over the calibration set, so `ids` only selects
        # which forecasts to return
        forecasts = _add_transfer_intervals(
            forecasts,
            result.cs_df,
            model_names=model_names,
            level=level,
            horizon=h,
            cs_h=result.h,
            method=method,
        )
        if result.target_scales is None:
            return forecasts
        # Per-series σ_tgt scaling: multiply interval half-widths by σ_tgt_j.
        # Scores are already normalized by σ_src_i, so quantiles are on the
        # normalized scale; rescaling by σ_tgt_j gives per-series correct widths.
        codes, uniques = pd.factorize(forecasts[self.ts.id_col].to_numpy())
        missing = [uid for uid in uniques if uid not in result.target_scales]
        if missing:
            raise ValueError(
                f"Series {missing} were added after `transfer`, so they "
                "have no scale estimate. Call `transfer` with their "
                "history to compute their prediction intervals."
            )
        sigma_tgt = np.array(
            [result.target_scales[uid] for uid in uniques], dtype=float
        )[codes]
        return _rescale_interval_columns(forecasts, model_names, level, sigma_tgt)

    def _add_intervals_from_fit(
        self,
        forecasts: DFType,
        level: List[Union[int, float]],
        h: int,
        ids: Optional[List[str]],
    ) -> DFType:
        calibration = self._calibration_
        pi = self.prediction_intervals
        assert calibration is not None and pi is not None
        scores = calibration.scores
        cs_ids = set(
            nw.from_native(scores, eager_only=True)[self.ts.id_col].unique().to_list()
        )
        if ids is None:
            if cs_ids != set(self.ts.uids):
                raise ValueError(
                    "Prediction intervals were calibrated on a different set of series "
                    "than the current forecasting state. Please rerun `fit` before "
                    "requesting intervals."
                )
            n_series = self.ts.ga.n_groups
        else:
            missing_ids = set(ids) - cs_ids
            if missing_ids:
                raise ValueError(
                    "Prediction intervals are only available for series seen during "
                    f"interval calibration. Missing ids: {missing_ids}."
                )
            scores = ufp.filter_with_mask(
                scores, ufp.is_in(scores[self.ts.id_col], ids)
            )
            n_series = len(ids)
        _check_scores_horizon(pi.h, h, _REFIT_FIX_H)
        return _get_conformal_method(pi.method)(
            forecasts,
            scores,
            model_names=list(self.models.keys()),
            level=level,
            cs_h=pi.h,
            cs_n_windows=pi.n_windows,
            n_series=n_series,
            horizon=h,
        )

    def _add_intervals_to_calibrated(
        self, forecasts: DFType, level: List[Union[int, float]], h: int
    ) -> Tuple[DFType, List[Any]]:
        """Add the fit's intervals to the series it calibrated, leave the rest missing.

        Also returns the ids of the series without intervals."""
        assert self._calibration_ is not None
        id_col = self.ts.id_col
        scores_ids = self._calibration_.scores[id_col]
        cs_ids = nw.from_native(scores_ids, series_only=True).unique().to_list()
        calibrated = ufp.is_in(forecasts[id_col], cs_ids)
        seen = ufp.filter_with_mask(forecasts, calibrated)
        ids = nw.from_native(seen[id_col], series_only=True).unique(maintain_order=True)
        seen = self._add_intervals_from_fit(seen, level, h, ids.to_list())
        unseen = ufp.filter_with_mask(forecasts, ~calibrated)
        unseen_ids = nw.from_native(unseen[id_col], series_only=True)
        out = nw.concat([nw.from_native(seen), nw.from_native(unseen)], how="diagonal")
        return out.to_native(), unseen_ids.unique(maintain_order=True).to_list()

    def cross_validation(
        self,
        df: DFType,
        n_windows: int,
        h: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        step_size: Optional[int] = None,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        refit: Union[bool, int] = True,
        max_horizon: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        horizon_features: Optional[Dict[int, List[str]]] = None,
        horizon_feature_templates: Optional[List[str]] = None,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
        level: Optional[List[Union[int, float]]] = None,
        input_size: Optional[int] = None,
        fitted: bool = False,
        as_numpy: bool = False,
        weight_col: Optional[str] = None,
        validate_data: bool = True,
    ) -> DFType:
        """Perform time series cross validation.
        Creates `n_windows` splits where each window has `h` test periods,
        trains the models, computes the predictions and merges the actuals.

        Args:
            df (pandas or polars DataFrame): Series data in long format.
            n_windows (int): Number of windows to evaluate.
            h (int): Forecast horizon.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            step_size (int, optional): Step size between each cross validation window. If None it will be equal to `h`. Defaults to None.
            static_features (list of str, optional): Names of the features that are static and will be repeated when forecasting. Defaults to None.
            dropna (bool): Drop rows with missing values produced by the transformations. Defaults to True.
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. Pooled lag transforms (global_/groupby/partition_by) with a window wider than this keep that wider window instead, since their shared aggregates have no per-series buffer to trim below it. Defaults to None.
            max_horizon (int, optional): Train this many models, where each model will predict a specific horizon. Defaults to None.
            horizons (list of int, optional): Train models only for specific horizons (1-indexed). Mutually exclusive with max_horizon. Defaults to None.
            horizon_features (dict of int to list of str, optional): Explicit mapping of 1-indexed horizons to dynamic exogenous columns.
                Only supported when using `max_horizon` or `horizons`. Defaults to None.
            horizon_feature_templates (list of str, optional): Template patterns for horizon-specific dynamic exogenous features.
                Each template must include exactly one '{h}' placeholder (1-indexed), for example: ['feature_h{h}'].
                Acts as shorthand for `horizon_features` and is only supported when using `max_horizon` or `horizons`. Defaults to None.
            refit (bool or int): Retrain model for each cross validation window. If False, the models are trained at the beginning and then used to predict each window. If positive int, the models are retrained every `refit` windows. The windows that aren't retrained use the prediction intervals of the last fit. Defaults to True.
            before_predict_callback (callable, optional): Function to call on the features before computing the predictions. This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure. The series identifier is on the index. Defaults to None.
            after_predict_callback (callable, optional): Function to call on the predictions before updating the targets. This function will take a pandas Series with the predictions and should return another one with the same structure. The series identifier is on the index. Defaults to None.
            prediction_intervals (PredictionIntervals, optional): Configuration to calibrate prediction intervals (Conformal Prediction). Defaults to None.
            level (list of ints or floats, optional): Confidence levels between 0 and 100 for prediction intervals. Defaults to None.
            input_size (int, optional): Maximum training samples per serie in each window. If None, will use an expanding window. Defaults to None.
            fitted (bool): Store the in-sample predictions. Defaults to False.
            as_numpy (bool): Cast features to numpy array. Defaults to False.
            weight_col (str, optional): Column that contains the sample weights. Defaults to None.
            validate_data (bool): Run data quality validations on the full dataset before cross-validation. Warns about missing dates and raises on duplicate rows. Defaults to True.

        Returns:
            pandas or polars DataFrame: Predictions for each window with the series id, timestamp, last train date, target value and predictions from each model.
        """
        _validate_no_null_times(df, time_col)
        # Run data validations once on full dataset if requested
        if validate_data:
            self._validate_data(df, id_col, time_col)

        results = []
        cv_models = []
        cv_fitted_values = []
        uncalibrated: Dict[Any, None] = {}
        splits = ufp.backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            freq=self.freq,
            step_size=step_size,
            input_size=input_size,
        )
        for i_window, (cutoffs, train, valid) in enumerate(splits):
            should_fit = i_window == 0 or (refit > 0 and i_window % refit == 0)
            if should_fit:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Pooled.*validate_data",
                        category=UserWarning,
                    )
                    self.fit(
                        train,
                        id_col=id_col,
                        time_col=time_col,
                        target_col=target_col,
                        static_features=static_features,
                        dropna=dropna,
                        keep_last_n=keep_last_n,
                        max_horizon=max_horizon,
                        horizons=horizons,
                        horizon_features=horizon_features,
                        horizon_feature_templates=horizon_feature_templates,
                        prediction_intervals=prediction_intervals,
                        fitted=fitted,
                        as_numpy=as_numpy,
                        weight_col=weight_col,
                        validate_data=False,
                    )
                cv_models.append(self.models_)
                if fitted:
                    cv_fitted_values.append(
                        ufp.assign_columns(self.fcst_fitted_values_, "fold", i_window)
                    )
            if fitted and not should_fit:
                if self.ts.target_transforms is not None:
                    for tfm in self.ts.target_transforms:
                        if hasattr(tfm, "store_fitted"):
                            tfm.store_fitted = True
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Pooled.*validate_data",
                        category=UserWarning,
                    )
                    prep = self.preprocess(
                        train,
                        id_col=id_col,
                        time_col=time_col,
                        target_col=target_col,
                        static_features=static_features,
                        dropna=dropna,
                        keep_last_n=keep_last_n,
                        max_horizon=max_horizon,
                        horizons=horizons,
                        horizon_features=horizon_features,
                        horizon_feature_templates=horizon_feature_templates,
                        return_X_y=False,
                        weight_col=weight_col,
                        validate_data=False,
                    )
                assert not isinstance(prep, tuple)
                effective_max_horizon = self.ts.max_horizon
                base = prep[[id_col, time_col]]
                train_X, train_y = self._extract_X_y(prep, target_col, weight_col)
                if as_numpy:
                    train_X = ufp.to_numpy(train_X)
                del prep
                fitted_values = self._compute_fitted_values(
                    base=base,
                    X=train_X,
                    y=train_y,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    max_horizon=effective_max_horizon,
                    weight_col=weight_col,
                    original_df=train if effective_max_horizon is not None else None,
                )
                fitted_values = ufp.assign_columns(fitted_values, "fold", i_window)
                cv_fitted_values.append(fitted_values)
            static = [c for c in self.ts.static_features_.columns if c != id_col]
            dynamic = [
                c
                for c in valid.columns
                if c not in static + [id_col, time_col, target_col]
            ]
            if dynamic:
                X_df: Optional[DataFrame] = ufp.drop_columns(
                    valid, static + [target_col]
                )
            else:
                X_df = None
            if should_fit:
                fcst = self
            else:
                fcst = self._with_ts(self.ts._clone_warm(train))
            frozen_intervals = (
                not should_fit and level is not None and self._calibration_ is not None
            )
            with warnings.catch_warnings():
                if not should_fit:
                    # the window that was fit already emitted these warnings
                    for message in (
                        "Prediction intervals are calculated using 1-step",
                        "Please rerun the `fit` method",
                    ):
                        warnings.filterwarnings(
                            "ignore", message=message, category=UserWarning
                        )
                y_pred = fcst.predict(
                    h=h,
                    before_predict_callback=before_predict_callback,
                    after_predict_callback=after_predict_callback,
                    level=None if frozen_intervals else level,
                    X_df=X_df,
                )
                if frozen_intervals:
                    assert level is not None
                    y_pred, unseen_ids = fcst._add_intervals_to_calibrated(
                        y_pred, sorted(level), h
                    )
                    uncalibrated.update(dict.fromkeys(unseen_ids))
            y_pred = ufp.join(y_pred, cutoffs, on=id_col, how="left")
            result = ufp.join(
                valid[[id_col, time_col, target_col]],
                y_pred,
                on=[id_col, time_col],
            )
            sort_idxs = ufp.maybe_compute_sort_indices(result, id_col, time_col)
            if sort_idxs is not None:
                result = ufp.take_rows(result, sort_idxs)
            # Calculate expected rows accounting for sparse horizons
            internal_horizons = self.ts._horizons
            full_range = (
                list(range(self.ts.max_horizon)) if self.ts.max_horizon else None
            )
            is_sparse = (
                internal_horizons is not None and internal_horizons != full_range
            )
            if is_sparse:
                # Sparse horizons: expect only predictions for trained horizons <= h
                assert internal_horizons is not None
                n_trained_horizons = sum(th < h for th in internal_horizons)
                n_series = nw.from_native(valid[id_col], series_only=True).n_unique()
                expected_rows = n_trained_horizons * n_series
            else:
                expected_rows = valid.shape[0]
            if result.shape[0] < expected_rows:
                raise ValueError(
                    "Cross validation result produced less results than expected. "
                    "Please verify that the frequency set on the MLForecast constructor matches your series' "
                    "and that there aren't any missing periods."
                )
            results.append(result)
        if uncalibrated:
            warnings.warn(
                "The prediction intervals of the windows that aren't refit use the "
                "conformity scores of the last fit, which didn't calibrate series "
                f"{list(uncalibrated)}. Their intervals are missing.",
                UserWarning,
            )
        del self.models_
        self.cv_models_ = cv_models
        self.cv_fitted_values_ = cv_fitted_values
        out = ufp.vertical_concat(results, match_categories=False)
        out = ufp.drop_index_if_pandas(out)
        first_out_cols = [id_col, time_col, "cutoff", target_col]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]
        return out[first_out_cols + remaining_cols]

    def cross_validation_fitted_values(self):
        if not getattr(self, "cv_fitted_values_", []):
            raise ValueError("Please run cross_validation with fitted=True first.")
        out = ufp.vertical_concat(self.cv_fitted_values_, match_categories=False)
        first_out_cols = [self.ts.id_col, self.ts.time_col, "fold", self.ts.target_col]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]
        out = ufp.drop_index_if_pandas(out)
        return out[first_out_cols + remaining_cols]

    def save(self, path: Union[str, Path]) -> None:
        """Save forecast object

        Args:
            path (str or pathlib.Path): Directory where artifacts will be stored.
        """
        self.ts.save(f"{path}/ts.pkl")
        with fsspec.open(f"{path}/models.pkl", "wb") as f:
            cloudpickle.dump(self.models_, f)
        with fsspec.open(f"{path}/intervals.pkl", "wb") as f:
            cloudpickle.dump(
                {
                    "settings": self.prediction_intervals,
                    "calibration": self._calibration_,
                    "train_ends": self._train_ends_,
                    "transfer": self._transfer_,
                },
                f,
            )

    @staticmethod
    def load(path: Union[str, Path]) -> "MLForecast":
        """Load forecast object

        Args:
            path (str or pathlib.Path): Directory with saved artifacts.
        """
        ts = TimeSeries.load(f"{path}/ts.pkl")
        with fsspec.open(f"{path}/models.pkl", "rb") as f:
            models = cloudpickle.load(f)
        try:
            with fsspec.open(f"{path}/intervals.pkl", "rb") as f:
                intervals = cloudpickle.load(f)
        except FileNotFoundError:
            intervals = {}
        if "train_ends" not in intervals:
            raise ValueError(
                f"{path} was saved by an older version of mlforecast. Refit the "
                "model and save it again."
            )
        return MLForecast._from_ts(
            ts,
            models=models,
            models_=models,
            prediction_intervals=intervals["settings"],
            calibration=intervals["calibration"],
            train_ends=intervals["train_ends"],
            transfer_=intervals["transfer"],
        )

    def update(self, df: DataFrame, validate_new_data: bool = False) -> None:
        """Update the values of the stored series.

        Args:
            df (pandas or polars DataFrame): Dataframe with new observations.
            validate_new_data (bool): If True, validate continuity, start dates, and frequency.
        """
        self.ts.update(df, validate_new_data)
