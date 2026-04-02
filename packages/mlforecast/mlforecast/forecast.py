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
    Tuple,
    Union,
)

import cloudpickle
import fsspec
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
from .data_validation import validate_df
from .target_transforms import _BaseGroupedArrayTargetTransform
from .utils import PredictionIntervals, _resolve_num_threads


def _add_conformal_distribution_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
) -> DFType:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This strategy creates forecasts paths
    based on errors and calculate quantiles using those paths.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        mean = fcst_df[model].to_numpy().reshape(1, n_series, -1)
        scores = np.vstack([mean - scores, mean + scores])
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1).T
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, out_cols, quantiles)
    return fcst_df


def _add_conformal_error_intervals(
    fcst_df: DFType,
    cs_df: DFType,
    model_names: List[str],
    level: List[Union[int, float]],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
) -> DFType:
    """
    Adds conformal intervals to a `fcst_df` based on conformal scores `cs_df`.
    `level` should be already sorted. This startegy creates prediction intervals
    based on the absolute errors.
    """
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        # restrict scores to horizon
        scores = scores[:, :, :horizon]
        quantiles = np.quantile(
            scores,
            cuts,
            axis=0,
        )
        quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        columns = lo_cols + hi_cols
        fcst_df = ufp.assign_columns(fcst_df, columns, quantiles)
    return fcst_df


def _get_conformal_method(method: str):
    available_methods = {
        "conformal_distribution": _add_conformal_distribution_intervals,
        "conformal_error": _add_conformal_error_intervals,
    }
    if method not in available_methods.keys():
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(available_methods.keys())}"
        )
    return available_methods[method]


class MLForecast:
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
    ):
        """Forecasting pipeline

        Args:
            models (regressor or list of regressors): Models that will be trained and used to compute the forecasts.
            freq (str or int or pd.offsets.BaseOffset): Pandas offset, pandas offset alias, e.g. 'D', 'W-THU' or integer denoting the frequency of the series.
            lags (list of int, optional): Lags of the target to use as features. Defaults to None.
            lag_transforms (dict of int to list of functions, optional): Mapping of target lags to their transformations. Defaults to None.
            date_features (list of str or callable, optional): Features computed from the dates. Can be pandas date attributes or functions that will take the dates as input. Defaults to None.
            num_threads (int): Number of threads to use when computing the features. Use -1 to use all available CPU cores. Defaults to 1.
            target_transforms (list of transformers, optional): Transformations that will be applied to the target before computing the features and restored after the forecasting step. Defaults to None.
            lag_transforms_namer (callable, optional): Function that takes a transformation (either function or class), a lag and extra arguments and produces a name. Defaults to None.
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
        if not hasattr(cv, "best_iteration_"):
            raise ValueError("LightGBMCV object must be fitted first.")
        import lightgbm as lgb

        fcst = cls(
            models=lgb.LGBMRegressor(
                **{**cv.params, "n_estimators": cv.best_iteration_}
            ),
            freq=cv.ts.freq,
        )
        fcst.ts = copy.deepcopy(cv.ts)
        return fcst

    def _validate_data(
        self,
        df: DataFrame,
        id_col: str,
        time_col: str,
    ) -> None:
        validate_df(df, id_col, time_col, self.freq)

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
                    raise ValueError("`horizon_features` keys must be positive integers.")
                if horizon > effective_max_horizon:
                    raise ValueError(
                        f"`horizon_features` includes horizon {horizon}, but the maximum configured horizon is {effective_max_horizon}."
                    )
                if not isinstance(cols, list) or any(not isinstance(col, str) for col in cols):
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
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. Defaults to None.
            max_horizon (int, optional): Train this many models, where each model will predict a specific horizon. Defaults to None.
            horizons (list of int, optional): Train models only for specific horizons (1-indexed). Mutually exclusive with max_horizon. Defaults to None.
            horizon_features (dict of int to list of str, optional): Explicit mapping of 1-indexed horizons to dynamic exogenous columns.
                Only supported when using `max_horizon` or `horizons`. Defaults to None.
            horizon_feature_templates (list of str, optional): Template patterns for horizon-specific dynamic exogenous features.
                Each template must include exactly one '{h}' placeholder (1-indexed), for example: ['feature_h{h}'].
                Acts as shorthand for `horizon_features` and is only supported when using `max_horizon` or `horizons`. Defaults to None.
            return_X_y (bool): Return a tuple with the features and the target. If False will return a single dataframe. Defaults to False.
            as_numpy (bool): Cast features to numpy array. Only works for `return_X_y=True`. Defaults to True.
            weight_col (str, optional): Column that contains the sample weights. Defaults to None.
            validate_data (bool): Run data quality validations before preprocessing. Warns about missing dates and raises on duplicate rows. Defaults to True.

        Returns:
            DataFrame or tuple of pandas Dataframe and a numpy array: `df` plus added features and target(s).
        """
        # Run data validations if requested
        if validate_data:
            self._validate_data(df, id_col, time_col)
        self.ts.horizon_features_ = self._resolve_horizon_features(
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

        return self.ts.fit_transform(
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
        )

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
                    fitted = fit_model(model, X_h, y_h, self.ts.weight_col, model_fit_kwargs)
                    self.models_[name][h] = fitted
        else:
            # Recursive forecasting (unchanged)
            assert X is not None and y is not None, "X and y are required when generator_factory is not provided"
            for name, model in self.models.items():
                model_fit_kwargs = models_fit_kwargs.get(name, None)
                self.models_[name] = fit_model(
                    model, X, y, self.ts.weight_col, model_fit_kwargs
                )
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
        cv_results = self.cross_validation(
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
        # conformity score for each model
        for model in self.models.keys():
            # compute absolute error for each model
            abs_err = abs(cv_results[model] - cv_results[target_col])
            cv_results = ufp.assign_columns(cv_results, model, abs_err)
        return ufp.drop_columns(cv_results, target_col)

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
            x_cols: Optional[List[str]] = list(X.columns) if hasattr(X, "columns") else None
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
                        col_idx = np.array([feature_idx[c] for c in x_cols_h], dtype=np.int32)
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
                            .with_columns(pl_Series('_offset_time', offset_times))
                            .with_row_index('_row_idx')
                        )
                        exog_source = original_df[[id_col, time_col] + exog_cols_h]
                        exog_renamed = exog_source.rename({time_col: '_offset_time'})
                        merged = lookup_df.join(exog_renamed, on=[id_col, '_offset_time'], how='left')
                        merged = merged.sort('_row_idx')
                        if non_exog_cols:
                            X_h = X[non_exog_cols]
                            for col in exog_cols_h:
                                X_h = X_h.with_columns(merged[col].alias(col))
                            X_h = X_h[x_cols_h]
                        else:
                            X_h = X
                    else:
                        lookup_df = base[[id_col]].copy()
                        lookup_df['_offset_time'] = offset_times
                        lookup_df['_row_idx'] = np.arange(len(base))
                        exog_source = original_df[[id_col, time_col] + exog_cols_h]
                        exog_renamed = exog_source.rename(columns={time_col: '_offset_time'})
                        merged = lookup_df.merge(exog_renamed, on=[id_col, '_offset_time'], how='left')
                        merged = merged.sort_values('_row_idx')
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
                    X_pred = ufp.to_numpy(X_valid) if models_trained_with_numpy else X_valid
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

        if isinstance(train_df, pd.DataFrame):
            train_pd = train_df.copy()
        else:
            train_pd = train_df.to_pandas()

        one_step = self.fcst_fitted_values_
        if isinstance(one_step, pd.DataFrame):
            one_step_pd = one_step[[self.ts.id_col, self.ts.time_col]].copy()
        else:
            one_step_pd = one_step.select([self.ts.id_col, self.ts.time_col]).to_pandas()

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
            temp_ts = TimeSeries(
                freq=self.ts.freq,
                lags=self.ts.lags,
                lag_transforms=self.ts.lag_transforms,
                date_features=self.ts.date_features,
                num_threads=self.ts.num_threads,
                target_transforms=copy.deepcopy(self.ts.target_transforms),
                lag_transforms_namer=self.ts.lag_transforms_namer,
            )
            temp_ts._fit(
                hist,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static_features=[id_col],
                keep_last_n=self.ts.keep_last_n,
                weight_col=self.ts.weight_col,
            )
            temp_ts.static_features_ = static_features_pd[
                static_features_pd[id_col].eq(uid)
            ].reset_index(drop=True)
            temp_ts.static_features = self.ts.static_features
            temp_ts.features_order_ = list(self.ts.features_order_)
            core_tfms = temp_ts._get_core_lag_tfms()
            if core_tfms:
                temp_ts._compute_transforms(core_tfms, updates_only=False)
            temp_ts.max_horizon = self.ts.max_horizon
            temp_ts._horizons = self.ts._horizons
            temp_ts.as_numpy = self.ts.as_numpy

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
                    target_col: group.iloc[target_idx, group.columns.get_loc(target_col)],
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
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. Defaults to None.
            max_horizon (int, optional): Train this many models, where each model will predict a specific horizon. Defaults to None.
            horizons (list of int, optional): Train models only for specific horizons (1-indexed). For example, `horizons=[7, 14]` trains models only for steps 7 and 14. Mutually exclusive with max_horizon. Defaults to None.
            horizon_features (dict of int to list of str, optional): Explicit mapping of 1-indexed horizons to dynamic exogenous columns.
                Only supported when using `max_horizon` or `horizons`. Defaults to None.
            horizon_feature_templates (list of str, optional): Template patterns for horizon-specific dynamic exogenous features.
                Each template must include exactly one '{h}' placeholder (1-indexed), for example: ['feature_h{h}'].
                Acts as shorthand for `horizon_features` and is only supported when using `max_horizon` or `horizons`. Defaults to None.
            prediction_intervals (PredictionIntervals, optional): Configuration to calibrate prediction intervals (Conformal Prediction). Defaults to None.
            fitted (bool): Save in-sample predictions. Defaults to False.
            as_numpy (bool): Cast features to numpy array. Defaults to True.
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
        self._cs_df: Optional[DataFrame] = None
        if prediction_intervals is not None:
            self.prediction_intervals = prediction_intervals
            self._cs_df = self._conformity_scores(
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
                as_numpy=False,
                weight_col=weight_col,
                validate_data=validate_data,
            )
            # Restore the as_numpy setting for prediction
            self.ts.as_numpy = as_numpy

            # Get the effective max horizon and internal horizons from preprocessing
            effective_max_horizon = self.ts.max_horizon
            internal_horizons = self.ts._horizons

            # Store original df for exog lookup in generator
            original_df = df

            # Factory function - creates fresh generator for each model
            def generator_factory():
                return self.ts._transform_per_horizon(
                    prep, original_df, internal_horizons, target_col, as_numpy
                )

            # Train models using generator factory
            self.fit_models(generator_factory=generator_factory, models_fit_kwargs=models_fit_kwargs)

            if fitted:
                assert not isinstance(prep, tuple)  # return_X_y=False ensures prep is a DataFrame
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
                raise ValueError("No fitted models available to compute multi-step fitted values.")
            res = self.fcst_fitted_values_
            if "h" not in res.columns:
                res = ufp.assign_columns(res, "h", np.int64(h))
            if isinstance(res, pd.DataFrame):
                res = res.copy()
                res["h"] = res["h"].astype(np.int64)
            else:
                res = res.with_columns(pl.col("h").cast(pl.Int64))
            return res

        first_model = next(iter(self.models_.values()))
        is_direct = isinstance(first_model, dict)
        if is_direct:
            res = self.fcst_fitted_values_
            if "h" not in res.columns:
                raise ValueError(
                    "Direct fitted values are missing horizon information. "
                    "Please refit the model with the current version."
                )
            if isinstance(res, pd.DataFrame):
                res = res[res["h"].eq(h)].reset_index(drop=True)
                available = sorted(self.fcst_fitted_values_["h"].unique().tolist())
            else:
                res = res.filter(pl.col("h") == h)
                available = sorted(self.fcst_fitted_values_["h"].unique().to_list())
            if len(res) == 0:
                raise ValueError(
                    f"No fitted values found for h={h}. Available horizons: {available}."
                )
        else:
            if h == 1:
                res = self.fcst_fitted_values_
            else:
                if self.ts._get_global_tfms() or self.ts._get_group_tfms():
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
                if isinstance(self.fcst_fitted_values_, pd.DataFrame):
                    res = res_pd
                else:
                    res = pl.from_pandas(res_pd)
        if "h" not in res.columns:
            res = ufp.assign_columns(res, "h", np.int64(h))
        if isinstance(res, pd.DataFrame):
            res = res.copy()
            res["h"] = res["h"].astype(np.int64)
        else:
            res = res.with_columns(pl.col("h").cast(pl.Int64))
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

    def predict(
        self,
        h: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        new_df: Optional[DFType] = None,
        level: Optional[List[Union[int, float]]] = None,
        X_df: Optional[DFType] = None,
        ids: Optional[List[str]] = None,
    ) -> DFType:
        """Compute the predictions for the next `h` steps.

        Args:
            h (int): Number of periods to predict.
            before_predict_callback (callable, optional): Function to call on the features before computing the predictions. This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure. The series identifier is on the index. Defaults to None.
            after_predict_callback (callable, optional): Function to call on the predictions before updating the targets. This function will take a pandas Series with the predictions and should return another one with the same structure. The series identifier is on the index. Defaults to None.
            new_df (pandas or polars DataFrame, optional): Series data of new observations for which forecasts are to be generated. This dataframe should have the same structure as the one used to fit the model, including any features and time series data. If `new_df` is not None, the method will generate forecasts for the new observations. Defaults to None.
            level (list of ints or floats, optional): Confidence levels between 0 and 100 for prediction intervals. Defaults to None.
            X_df (pandas or polars DataFrame, optional): Dataframe with the future exogenous features. Should have the id column and the time column. Defaults to None.
            ids (list of str, optional): List with subset of ids seen during training for which the forecasts should be computed. Defaults to None.

        Returns:
            pandas or polars DataFrame: Predictions for each serie and timestep, with one column per model.
        """
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

        new_ts: Optional[TimeSeries] = None
        if new_df is not None:
            if level is not None:
                raise ValueError(
                    "Prediction intervals are not supported in transfer learning."
                )
            new_ts = TimeSeries(
                freq=self.ts.freq,
                lags=self.ts.lags,
                lag_transforms=self.ts.lag_transforms,
                date_features=self.ts.date_features,
                num_threads=self.ts.num_threads,
                target_transforms=self.ts.target_transforms,
                lag_transforms_namer=self.ts.lag_transforms_namer,
            )
            new_ts._fit(
                new_df,
                id_col=self.ts.id_col,
                time_col=self.ts.time_col,
                target_col=self.ts.target_col,
                static_features=self.ts.static_features,
                keep_last_n=self.ts.keep_last_n,
                weight_col=self.ts.weight_col,
            )
            core_tfms = new_ts._get_core_lag_tfms()
            if core_tfms:
                # populate the stats needed for the updates
                new_ts._compute_transforms(core_tfms, updates_only=False)
            new_ts.max_horizon = self.ts.max_horizon
            new_ts._horizons = self.ts._horizons
            new_ts.as_numpy = self.ts.as_numpy
            new_ts.horizon_features_ = copy.deepcopy(self.ts.horizon_features_)
            ts = new_ts
        else:
            ts = self.ts
        forecasts = ts.predict(
            models=self.models_,
            horizon=h,
            before_predict_callback=before_predict_callback,
            after_predict_callback=after_predict_callback,
            X_df=X_df,
            ids=ids,
        )
        if new_ts is not None:
            # Persist transfer-learning state only after a successful prediction.
            self.ts = new_ts
        if level is not None:
            if self._cs_df is None:
                warn_msg = (
                    "Please rerun the `fit` method passing a proper value "
                    "to prediction intervals to compute them."
                )
                warnings.warn(warn_msg, UserWarning)
            else:
                if isinstance(self._cs_df, pl_DataFrame):
                    cs_ids = set(self._cs_df[self.ts.id_col].unique().to_list())
                else:
                    cs_ids = set(self._cs_df[self.ts.id_col].unique().tolist())
                if ids is None:
                    active_ids = set(self.ts.uids)
                    if cs_ids != active_ids:
                        raise ValueError(
                            "Prediction intervals were calibrated on a different set of series "
                            "than the current forecasting state. Please rerun `fit` before "
                            "requesting intervals."
                        )
                else:
                    missing_ids = set(ids) - cs_ids
                    if missing_ids:
                        raise ValueError(
                            "Prediction intervals are only available for series seen during "
                            "interval calibration. Missing ids: "
                            f"{missing_ids}."
                        )
                if (self.prediction_intervals.h != 1) and (
                    self.prediction_intervals.h < h
                ):
                    raise ValueError(
                        "The `h` argument of PredictionIntervals "
                        "should be equal to one or greater or equal to `h`. "
                        "Please rerun the `fit` method passing a proper value "
                        "to prediction intervals."
                    )
                if self.prediction_intervals.h == 1 and h > 1:
                    warn_msg = (
                        "Prediction intervals are calculated using 1-step ahead cross-validation, "
                        "with a constant width for all horizons. To vary the error by horizon, "
                        "pass PredictionIntervals(h=h) to the `prediction_intervals` "
                        "argument when refitting the model."
                    )
                    warnings.warn(warn_msg, UserWarning)
                level_ = sorted(level)
                model_names = self.models.keys()
                conformal_method = _get_conformal_method(
                    self.prediction_intervals.method
                )
                if ids is not None:
                    ids_mask = ufp.is_in(self._cs_df[self.ts.id_col], ids)
                    cs_df = ufp.filter_with_mask(self._cs_df, ids_mask)
                    n_series = len(ids)
                else:
                    cs_df = self._cs_df
                    n_series = self.ts.ga.n_groups
                forecasts = conformal_method(
                    forecasts,
                    cs_df,
                    model_names=list(model_names),
                    level=level_,
                    cs_h=self.prediction_intervals.h,
                    cs_n_windows=self.prediction_intervals.n_windows,
                    n_series=n_series,
                    horizon=h,
                )
        return forecasts

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
            keep_last_n (int, optional): Keep only these many records from each serie for the forecasting step. Can save time and memory if your features allow it. Defaults to None.
            max_horizon (int, optional): Train this many models, where each model will predict a specific horizon. Defaults to None.
            horizons (list of int, optional): Train models only for specific horizons (1-indexed). Mutually exclusive with max_horizon. Defaults to None.
            horizon_features (dict of int to list of str, optional): Explicit mapping of 1-indexed horizons to dynamic exogenous columns.
                Only supported when using `max_horizon` or `horizons`. Defaults to None.
            horizon_feature_templates (list of str, optional): Template patterns for horizon-specific dynamic exogenous features.
                Each template must include exactly one '{h}' placeholder (1-indexed), for example: ['feature_h{h}'].
                Acts as shorthand for `horizon_features` and is only supported when using `max_horizon` or `horizons`. Defaults to None.
            refit (bool or int): Retrain model for each cross validation window. If False, the models are trained at the beginning and then used to predict each window. If positive int, the models are retrained every `refit` windows. Defaults to True.
            before_predict_callback (callable, optional): Function to call on the features before computing the predictions. This function will take the input dataframe that will be passed to the model for predicting and should return a dataframe with the same structure. The series identifier is on the index. Defaults to None.
            after_predict_callback (callable, optional): Function to call on the predictions before updating the targets. This function will take a pandas Series with the predictions and should return another one with the same structure. The series identifier is on the index. Defaults to None.
            prediction_intervals (PredictionIntervals, optional): Configuration to calibrate prediction intervals (Conformal Prediction). Defaults to None.
            level (list of ints or floats, optional): Confidence levels between 0 and 100 for prediction intervals. Defaults to None.
            input_size (int, optional): Maximum training samples per serie in each window. If None, will use an expanding window. Defaults to None.
            fitted (bool): Store the in-sample predictions. Defaults to False.
            as_numpy (bool): Cast features to numpy array. Defaults to True.
            weight_col (str, optional): Column that contains the sample weights. Defaults to None.
            validate_data (bool): Run data quality validations on the full dataset before cross-validation. Warns about missing dates and raises on duplicate rows. Defaults to True.

        Returns:
            pandas or polars DataFrame: Predictions for each window with the series id, timestamp, last train date, target value and predictions from each model.
        """
        # Run data validations once on full dataset if requested
        if validate_data:
            self._validate_data(df, id_col, time_col)

        results = []
        cv_models = []
        cv_fitted_values = []
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
            y_pred = self.predict(
                h=h,
                before_predict_callback=before_predict_callback,
                after_predict_callback=after_predict_callback,
                new_df=train if not should_fit else None,
                level=level,
                X_df=X_df,
            )
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
            internal_horizons = getattr(self.ts, "_horizons", None)
            full_range = list(range(self.ts.max_horizon)) if self.ts.max_horizon else None
            is_sparse = internal_horizons is not None and internal_horizons != full_range
            if is_sparse:
                # Sparse horizons: expect only predictions for trained horizons <= h
                assert internal_horizons is not None
                n_trained_horizons = sum(th < h for th in internal_horizons)
                n_series = valid[id_col].nunique()
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
        if self._cs_df is not None:
            with fsspec.open(f"{path}/intervals.pkl", "wb") as f:
                cloudpickle.dump(
                    {"scores": self._cs_df, "settings": self.prediction_intervals}, f
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
            intervals = None
        fcst = MLForecast(models=models, freq=ts.freq)
        fcst.ts = ts
        fcst.models_ = models
        if intervals is not None:
            fcst.prediction_intervals = intervals["settings"]
            fcst._cs_df = intervals["scores"]
        return fcst

    def update(self, df: DataFrame, validate_new_data: bool = False) -> None:
        """Update the values of the stored series.

        Args:
            df (pandas or polars DataFrame): Dataframe with new observations.
            validate_new_data (bool): If True, validate continuity, start dates, and frequency.
        """
        self.ts.update(df, validate_new_data)
