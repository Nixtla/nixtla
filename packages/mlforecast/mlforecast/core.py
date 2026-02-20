__all__ = ['TimeSeries']


import copy
import inspect
import reprlib
import warnings
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import cloudpickle
import fsspec
import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from utilsforecast.compat import (
    DataFrame,
    DFType,
    Series,
    pl,
    pl_DataFrame,
    pl_Series,
)
from utilsforecast.validation import validate_format, validate_freq

from mlforecast.target_transforms import (
    BaseTargetTransform,
    _BaseGroupedArrayTargetTransform,
)

from .grouped_array import GroupedArray
from .lag_transforms import Lag, _BaseLagTransform
from .utils import _ShortSeriesException, _resolve_num_threads

date_features_dtypes = {
    "year": np.uint16,
    "month": np.uint8,
    "day": np.uint8,
    "hour": np.uint8,
    "minute": np.uint8,
    "second": np.uint8,
    "dayofyear": np.uint16,
    "day_of_year": np.uint16,
    "weekofyear": np.uint8,
    "week": np.uint8,
    "dayofweek": np.uint8,
    "day_of_week": np.uint8,
    "weekday": np.uint8,
    "quarter": np.uint8,
    "daysinmonth": np.uint8,
    "is_month_start": np.uint8,
    "is_month_end": np.uint8,
    "is_quarter_start": np.uint8,
    "is_quarter_end": np.uint8,
    "is_year_start": np.uint8,
    "is_year_end": np.uint8,
}


def _build_function_transform_name(tfm: Callable, lag: int, *args) -> str:
    """Creates a name for a transformation based on `lag`, the name of the function and its arguments."""
    tfm_name = f"{tfm.__name__}_lag{lag}"
    func_params = inspect.signature(tfm).parameters
    func_args = list(func_params.items())[1:]  # remove input array argument
    changed_params = [
        f"{name}{value}"
        for value, (name, arg) in zip(args, func_args)
        if arg.default != value
    ]
    if changed_params:
        tfm_name += "_" + "_".join(changed_params)
    return tfm_name


def _build_lag_transform_name(tfm: _BaseLagTransform, lag: int) -> str:
    return tfm._get_name(lag)


def _build_transform_name(
    tfm: Union[Callable, _BaseLagTransform], lag: int, *args
) -> str:
    if callable(tfm):
        name = _build_function_transform_name(tfm, lag, *args)
    else:
        name = _build_lag_transform_name(tfm, lag)
    return name


def _get_model_name(model) -> str:
    if isinstance(model, Pipeline):
        return _get_model_name(model.steps[-1][1])
    return model.__class__.__name__


def _name_models(current_names):
    ctr = Counter(current_names)
    if not ctr:
        return []
    if max(ctr.values()) < 2:
        return current_names
    names = current_names.copy()
    for i, x in enumerate(reversed(current_names), start=1):
        count = ctr[x]
        if count > 1:
            name = f"{x}{count}"
            ctr[x] -= 1
        else:
            name = x
        names[-i] = name
    return names


def _as_tuple(x):
    """Return a tuple from the input."""
    if isinstance(x, tuple):
        return x
    return (x,)


Freq = Union[int, str]
Lags = Iterable[int]
LagTransform = Union[Callable, Tuple[Callable, Any]]
LagTransforms = Dict[int, List[LagTransform]]
DateFeature = Union[str, Callable]
Models = Union[BaseEstimator, List[BaseEstimator], Dict[str, BaseEstimator]]
TargetTransform = Union[BaseTargetTransform, _BaseGroupedArrayTargetTransform]
Transforms = Dict[str, Union[Tuple[Any, ...], _BaseLagTransform]]


def _validate_horizon_params(
    max_horizon: Optional[int], horizons: Optional[List[int]]
) -> Tuple[Optional[List[int]], Optional[int]]:
    """Validate and normalize horizon parameters.

    Args:
        max_horizon: Train models for all horizons 1 to max_horizon.
        horizons: Train models only for specific horizons (1-indexed).

    Returns:
        Tuple of (internal_horizons, effective_max_horizon):
        - internal_horizons: 0-indexed list of horizons, or None for recursive mode
        - effective_max_horizon: Maximum horizon value (for target expansion)
    """
    if max_horizon is not None and horizons is not None:
        raise ValueError("Cannot specify both 'max_horizon' and 'horizons'")

    if horizons is not None:
        if not horizons:
            raise ValueError("'horizons' cannot be empty")
        if not all(isinstance(h, int) and h > 0 for h in horizons):
            raise ValueError("All horizons must be positive integers")
        horizons = sorted(set(horizons))  # dedupe and sort
        return [h - 1 for h in horizons], max(horizons)  # 0-indexed, max

    if max_horizon is not None:
        return list(range(max_horizon)), max_horizon

    return None, None


def _parse_transforms(
    lags: Lags,
    lag_transforms: LagTransforms,
    namer: Optional[Callable] = None,
) -> Transforms:
    transforms: Transforms = OrderedDict()
    if namer is None:
        namer = _build_transform_name
    for lag in lags:
        transforms[f"lag{lag}"] = Lag(lag)
    for lag in lag_transforms.keys():
        for tfm in lag_transforms[lag]:
            if isinstance(tfm, _BaseLagTransform):
                tfm_name = namer(tfm, lag)
                transforms[tfm_name] = copy.deepcopy(tfm)._set_core_tfm(lag)
            else:
                tfm, *args = _as_tuple(tfm)
                assert callable(tfm)
                tfm_name = namer(tfm, lag, *args)
                transforms[tfm_name] = (lag, tfm, *args)
    return transforms


class TimeSeries:
    """Utility class for storing and transforming time series data."""

    def __init__(
        self,
        freq: Freq,
        lags: Optional[Lags] = None,
        lag_transforms: Optional[LagTransforms] = None,
        date_features: Optional[Iterable[DateFeature]] = None,
        num_threads: int = 1,
        target_transforms: Optional[List[TargetTransform]] = None,
        lag_transforms_namer: Optional[Callable] = None,
    ):
        self.freq = freq
        num_threads = _resolve_num_threads(num_threads)
        if not isinstance(num_threads, int) or num_threads < 1:
            warnings.warn("Setting num_threads to 1.")
            num_threads = 1
        self.lags = [] if lags is None else list(lags)
        for lag in self.lags:
            if lag <= 0 or not isinstance(lag, int):
                raise ValueError("lags must be positive integers.")
        self.lag_transforms = {} if lag_transforms is None else lag_transforms
        for lag in self.lag_transforms.keys():
            if lag <= 0 or not isinstance(lag, int):
                raise ValueError("keys of lag_transforms must be positive integers.")
        self.date_features = [] if date_features is None else list(date_features)
        self.num_threads = num_threads
        self.target_transforms = target_transforms
        if self.target_transforms is not None:
            for tfm in self.target_transforms:
                if isinstance(tfm, _BaseGroupedArrayTargetTransform):
                    tfm.set_num_threads(num_threads)
        for feature in self.date_features:
            if callable(feature) and feature.__name__ == "<lambda>":
                raise ValueError(
                    "Can't use a lambda as a date feature because the function name gets used as the feature name."
                )
        self.lag_transforms_namer = lag_transforms_namer
        self.transforms = _parse_transforms(
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            namer=lag_transforms_namer,
        )
        self.ga: GroupedArray

    def _get_core_lag_tfms(self) -> Dict[str, _BaseLagTransform]:
        return {
            k: v for k, v in self.transforms.items() if isinstance(v, _BaseLagTransform)
        }

    def _get_global_tfms(self) -> Dict[str, _BaseLagTransform]:
        return {
            k: v
            for k, v in self.transforms.items()
            if isinstance(v, _BaseLagTransform) and getattr(v, "global_", False)
        }

    def _get_group_tfms(self) -> Dict[Tuple[str, ...], Dict[str, _BaseLagTransform]]:
        grouped: Dict[Tuple[str, ...], Dict[str, _BaseLagTransform]] = {}
        for name, tfm in self.transforms.items():
            if not isinstance(tfm, _BaseLagTransform):
                continue
            groupby = getattr(tfm, "groupby", None)
            if not groupby:
                continue
            key = tuple(groupby)
            grouped.setdefault(key, {})[name] = tfm
        return grouped

    def _get_local_tfms(
        self,
        transforms: Mapping[str, Union[Tuple[Any, ...], _BaseLagTransform]],
    ) -> Dict[str, Union[Tuple[Any, ...], _BaseLagTransform]]:
        local = {}
        for name, tfm in transforms.items():
            if isinstance(tfm, _BaseLagTransform) and getattr(tfm, "global_", False):
                continue
            if isinstance(tfm, _BaseLagTransform) and getattr(tfm, "groupby", None):
                continue
            local[name] = tfm
        return local

    def _check_aligned_ends(self) -> None:
        """Check that all series end at the same timestamp when using global/group transforms."""
        if not (self._get_global_tfms() or self._get_group_tfms()):
            return
        if isinstance(self.last_dates, pd.Index):
            aligned = self.last_dates.nunique() == 1
        else:
            aligned = self.last_dates.n_unique() == 1
        if not aligned:
            raise ValueError(
                "Global and group lag transforms require all series to end at the same timestamp."
            )

    @property
    def _date_feature_names(self):
        return [f.__name__ if callable(f) else f for f in self.date_features]

    @property
    def features(self) -> List[str]:
        """Names of all computed features."""
        return list(self.transforms.keys()) + self._date_feature_names

    def _get_dynamic_exog_cols(self, df_columns: List[str]) -> List[str]:
        """Identify user-provided exogenous columns that need time-alignment."""
        static_cols = set(self.static_features_.columns)
        lag_cols = set(self.transforms.keys())
        date_cols = {f.__name__ if callable(f) else f for f in self.date_features}
        exclude = static_cols | lag_cols | date_cols | {self.id_col, self.time_col, self.target_col}
        if self.weight_col is not None:
            exclude.add(self.weight_col)
        return [c for c in df_columns if c not in exclude]

    def __repr__(self):
        return (
            f"TimeSeries(freq={self.freq}, "
            f"transforms={list(self.transforms.keys())}, "
            f"date_features={self._date_feature_names}, "
            f"num_threads={self.num_threads})"
        )

    def _fit(
        self,
        df: DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        keep_last_n: Optional[int] = None,
        weight_col: Optional[str] = None,
    ) -> "TimeSeries":
        """Save the series values, ids and last dates."""
        validate_format(df, id_col, time_col, target_col)
        validate_freq(df[time_col], self.freq)
        if ufp.is_nan_or_none(df[target_col]).any():
            raise ValueError(f"{target_col} column contains null values.")
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        self.weight_col = weight_col
        self.keep_last_n = keep_last_n
        self.static_features = static_features
        sorted_df = df[[id_col, time_col, target_col]]
        sorted_df = ufp.copy_if_pandas(sorted_df, deep=False)
        uids, times, data, indptr, self._sort_idxs = ufp.process_df(
            df=sorted_df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        if data.ndim == 2:
            data = data[:, 0]
        ga = GroupedArray(data, indptr)
        if isinstance(df, pd.DataFrame):
            self.uids = pd.Index(uids)
            self.last_dates = pd.Index(times)
        else:
            self.uids = uids
            self.last_dates = pl_Series(times)
        self._check_aligned_ends()
        if self._sort_idxs is not None:
            self._restore_idxs: Optional[np.ndarray] = np.empty(
                df.shape[0], dtype=np.int32
            )
            self._restore_idxs[self._sort_idxs] = np.arange(df.shape[0])
            sorted_df = ufp.take_rows(sorted_df, self._sort_idxs)
        else:
            self._restore_idxs = None
        if self.target_transforms is not None:
            for tfm in self.target_transforms:
                if isinstance(tfm, _BaseGroupedArrayTargetTransform):
                    try:
                        ga = tfm.fit_transform(ga)
                    except _ShortSeriesException as exc:
                        tfm_name = tfm.__class__.__name__
                        uids = reprlib.repr(list(self.uids[exc.args]))
                        raise ValueError(
                            f"The following series are too short for the '{tfm_name}' transformation: {uids}."
                        ) from None
                    sorted_df = ufp.assign_columns(sorted_df, target_col, ga.data)
                else:
                    tfm.set_column_names(id_col, time_col, target_col)
                    sorted_df = tfm.fit_transform(sorted_df)
                    ga.data = sorted_df[target_col].to_numpy()
        self._global_ga: Optional[GroupedArray] = None
        self._global_times: Optional[Union[Series, pd.Index]] = None
        if self._get_global_tfms():
            global_df = ufp.group_by_agg(
                sorted_df[[time_col, target_col]],
                time_col,
                {target_col: "sum"},
                maintain_order=True,
            )
            global_df = ufp.sort(global_df, by=time_col)
            global_values = global_df[target_col].to_numpy().astype(ga.data.dtype)
            self._global_ga = GroupedArray(
                global_values, np.array([0, global_values.size], dtype=np.int32)
            )
            if isinstance(global_df, pd.DataFrame):
                self._global_times = pd.Index(global_df[time_col])
            else:
                self._global_times = global_df[time_col]
        to_drop = [id_col, time_col, target_col]
        if static_features is None:
            static_features = [c for c in df.columns if c not in [time_col, target_col]]
        elif id_col not in static_features:
            static_features = [id_col, *static_features]
        else:  # static_features defined and contain id_col
            to_drop = [time_col, target_col]
        if weight_col is not None:
            to_drop.append(weight_col)
            static_features = [f for f in static_features if f != weight_col]
        self.ga = ga
        series_starts = ga.indptr[:-1]
        series_ends = ga.indptr[1:] - 1
        if self._sort_idxs is not None:
            series_starts = self._sort_idxs[series_starts]
            series_ends = self._sort_idxs[series_ends]
        statics_on_starts = ufp.drop_index_if_pandas(
            ufp.take_rows(df, series_starts)[static_features]
        )
        statics_on_ends = ufp.drop_index_if_pandas(
            ufp.take_rows(df, series_ends)[static_features]
        )
        for feat in static_features:
            if (statics_on_starts[feat] != statics_on_ends[feat]).any():
                raise ValueError(
                    f"{feat} is declared as a static feature but its values change "
                    "over time. Please set the `static_features` argument to "
                    "indicate which features are static.\nIf all of your features "
                    "are dynamic please set `static_features=[]`."
                )
        self.static_features_ = statics_on_ends
        self.features_order_ = [c for c in df.columns if c not in to_drop] + [
            f for f in self.features if f not in df.columns
        ]
        self._group_states: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        group_tfms = self._get_group_tfms()
        if group_tfms:
            if self.target_transforms is not None:
                transformed_target = ga.data
                if self._restore_idxs is not None:
                    transformed_target = transformed_target[self._restore_idxs]
                df_for_group = ufp.assign_columns(df, target_col, transformed_target)
            else:
                df_for_group = df

            def _add_group_id(data, cols):
                if isinstance(data, pd.DataFrame):
                    groups = data[cols].drop_duplicates().reset_index(drop=True)
                    groups["_group_id"] = np.arange(len(groups), dtype=np.int64)
                    data = data.merge(groups, on=cols, how="left")
                else:
                    groups = data.select(cols).unique(maintain_order=True)
                    groups = groups.with_row_index(name="_group_id")
                    data = data.join(groups, on=cols, how="left")
                return data, groups

            def _map_group_id(data, groups, cols):
                if isinstance(data, pd.DataFrame):
                    joined = data[cols].merge(groups, on=cols, how="left")
                    return joined["_group_id"].to_numpy()
                joined = data.select(cols).join(groups, on=cols, how="left")
                return joined["_group_id"].to_numpy()

            for group_cols, tfms in group_tfms.items():
                for col in group_cols:
                    if col not in df.columns:
                        raise ValueError(f"Groupby column '{col}' not found in dataframe.")
                group_cols_list = list(group_cols)
                missing = [c for c in group_cols_list if c not in self.static_features_.columns]
                if missing:
                    raise ValueError(
                        "Groupby columns must be static features. "
                        f"Missing from static_features: {missing}."
                    )
                group_df = ufp.group_by_agg(
                    df_for_group[group_cols_list + [time_col, target_col]],
                    group_cols_list + [time_col],
                    {target_col: "sum"},
                    maintain_order=True,
                )
                group_df = ufp.sort(group_df, by=group_cols_list + [time_col])
                group_df, groups = _add_group_id(group_df, group_cols_list)
                group_df = ufp.drop_index_if_pandas(group_df)
                groups = ufp.drop_index_if_pandas(groups)
                if isinstance(group_df, pd.DataFrame):
                    process_df = group_df[["_group_id", time_col, target_col]]
                else:
                    process_df = group_df.select(["_group_id", time_col, target_col])
                processed = ufp.process_df(
                    process_df,
                    id_col="_group_id",
                    time_col=time_col,
                    target_col=target_col,
                )
                if processed.sort_idxs is not None:
                    group_df = ufp.take_rows(group_df, processed.sort_idxs)
                group_df = ufp.drop_index_if_pandas(group_df)
                group_values = processed.data[:, 0]
                ga = GroupedArray(group_values, processed.indptr)
                group_uids = processed.uids
                series_group_id = _map_group_id(
                    self.static_features_, groups, group_cols_list
                )
                group_idx = series_group_id.astype(np.int64, copy=False)
                self._group_states[group_cols] = {
                    "ga": ga,
                    "df": group_df,
                    "group_cols": group_cols_list,
                    "groups": groups,
                    "group_uids": group_uids,
                    "group_idx": group_idx,
                }
        return self

    def _compute_transforms(
        self,
        transforms: Mapping[str, Union[Tuple[Any, ...], _BaseLagTransform]],
        updates_only: bool,
    ) -> Dict[str, np.ndarray]:
        """Compute the transformations defined in the constructor.

        If `self.num_threads > 1` these are computed using multithreading."""
        transforms = self._get_local_tfms(transforms)
        if not transforms:
            return {}
        if self.num_threads == 1 or len(transforms) == 1:
            out = self.ga.apply_transforms(
                transforms=transforms, updates_only=updates_only
            )
        else:
            out = self.ga.apply_multithreaded_transforms(
                transforms=transforms,
                num_threads=self.num_threads,
                updates_only=updates_only,
            )
        return out

    def _compute_date_feature(self, dates, feature):
        if callable(feature):
            feat_name = feature.__name__
            feat_vals = feature(dates)
        else:
            feat_name = feature
            if isinstance(dates, pd.DatetimeIndex):
                if feature in ("week", "weekofyear"):
                    dates = dates.isocalendar()
                feat_vals = getattr(dates, feature)
            else:
                feat_vals = getattr(dates.dt, feature)()
        if isinstance(feat_vals, (pd.Index, pd.Series)):
            feat_vals = np.asarray(feat_vals)
            feat_dtype = date_features_dtypes.get(feature)
            if feat_dtype is not None:
                feat_vals = feat_vals.astype(feat_dtype)
        return feat_name, feat_vals

    def _transform(
        self,
        df: DFType,
        dropna: bool = True,
        max_horizon: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        return_X_y: bool = False,
        as_numpy: bool = False,
    ) -> DFType:
        """Add the features to `df`.

        if `dropna=True` then all the null rows are dropped.

        Args:
            df: Input dataframe
            dropna: Drop rows with missing values
            max_horizon: Train models for all horizons 1 to max_horizon
            horizons: Train models only for specific horizons (1-indexed)
            return_X_y: Return tuple of (X, y) instead of dataframe
            as_numpy: Convert X to numpy array
        """
        # Validate and normalize horizon parameters
        self._horizons, effective_max_horizon = _validate_horizon_params(
            max_horizon, horizons
        )

        # we need to compute all transformations in case they save state
        features = self._compute_transforms(
            transforms=self.transforms, updates_only=False
        )
        global_tfms = self._get_global_tfms()
        if global_tfms:
            assert self._global_ga is not None
            assert self._global_times is not None
            global_vals = self._global_ga.apply_transforms(
                transforms=global_tfms, updates_only=False
            )
            if isinstance(df, pd.DataFrame):
                for name, vals in global_vals.items():
                    mapped = pd.Series(vals, index=self._global_times)
                    features[name] = df[self.time_col].map(mapped).to_numpy()
            else:
                global_df = pl_DataFrame({self.time_col: self._global_times})
                for name, vals in global_vals.items():
                    global_df = global_df.with_columns(
                        pl.Series(name=name, values=vals)
                    )
                joined = df.select(self.time_col).join(
                    global_df, on=self.time_col, how="left"
                )
                for name in global_vals.keys():
                    features[name] = joined[name].to_numpy()
        group_tfms = self._get_group_tfms()
        if group_tfms:
            for group_cols, tfms in group_tfms.items():
                state = self._group_states[group_cols]
                group_df = state["df"]
                ga = state["ga"]
                group_vals = ga.apply_transforms(transforms=tfms, updates_only=False)
                group_cols_list = state["group_cols"]
                feature_cols = list(group_vals.keys())
                if isinstance(df, pd.DataFrame):
                    join_df = group_df[group_cols_list + [self.time_col]].copy()
                    for name, vals in group_vals.items():
                        join_df[name] = vals
                    joined = df[group_cols_list + [self.time_col]].merge(
                        join_df, on=group_cols_list + [self.time_col], how="left"
                    )
                    for name in feature_cols:
                        features[name] = joined[name].to_numpy()
                else:
                    join_df = group_df.select(group_cols_list + [self.time_col])
                    for name, vals in group_vals.items():
                        join_df = join_df.with_columns(pl.Series(name=name, values=vals))
                    joined = df.select(group_cols_list + [self.time_col]).join(
                        join_df, on=group_cols_list + [self.time_col], how="left"
                    )
                    for name in feature_cols:
                        features[name] = joined[name].to_numpy()
        # filter out the features that already exist in df to avoid overwriting them
        features = {k: v for k, v in features.items() if k not in df}
        if self._restore_idxs is not None:
            for k, v in features.items():
                features[k] = v[self._restore_idxs]

        # target
        self.max_horizon = effective_max_horizon
        if effective_max_horizon is None:
            target = self.ga.data
        else:
            target = self.ga.expand_target(effective_max_horizon)
        if self._restore_idxs is not None:
            target = target[self._restore_idxs]

        # determine rows to keep
        target_nulls = np.isnan(target)
        if target_nulls.ndim == 2:
            # target nulls for each horizon are dropped in MLForecast.fit_models
            # we just drop rows here for which all the target values are null
            target_nulls = target_nulls.all(axis=1)
        if dropna:
            feature_nulls = np.full(df.shape[0], False)
            for feature_vals in features.values():
                feature_nulls |= np.isnan(feature_vals)
            keep_rows = ~(feature_nulls | target_nulls)
        else:
            # we always want to drop rows with nulls in the target
            keep_rows = ~target_nulls

        self._dropped_series: Optional[np.ndarray] = None
        if not keep_rows.all():
            # remove rows with nulls
            for k, v in features.items():
                features[k] = v[keep_rows]
            target = target[keep_rows]
            df = ufp.filter_with_mask(df, keep_rows)
            df = ufp.copy_if_pandas(df, deep=False)
            last_idxs = self.ga.indptr[1:] - 1
            if self._sort_idxs is not None:
                last_idxs = self._sort_idxs[last_idxs]
            last_vals_nan = ~keep_rows[last_idxs]
            if last_vals_nan.any():
                self._dropped_series = np.where(last_vals_nan)[0]
                dropped_ids = reprlib.repr(list(self.uids[self._dropped_series]))
                warnings.warn(
                    "The following series were dropped completely "
                    f"due to the transformations and features: {dropped_ids}.\n"
                    "These series won't show up if you use `MLForecast.forecast_fitted_values()`.\n"
                    "You can set `dropna=False` or use transformations that require less samples to mitigate this"
                )
        elif isinstance(df, pd.DataFrame):
            # we'll be assigning columns below, so we need to copy
            df = df.copy(deep=False)

        # once we've computed the features and target we can slice the series
        update_samples = [
            getattr(tfm, "update_samples", -1) for tfm in self.transforms.values()
        ]
        if (
            self.keep_last_n is None
            and update_samples
            and all(samples > 0 for samples in update_samples)
        ):
            # user didn't set keep_last_n and we can infer it from the transforms
            self.keep_last_n = max(update_samples)
        if self.keep_last_n is not None:
            self.ga = self.ga.take_from_groups(slice(-self.keep_last_n, None))
        del self._restore_idxs, self._sort_idxs

        # lag transforms
        for feat in self.transforms.keys():
            if feat in features:
                df = ufp.assign_columns(df, feat, features[feat])

        # date features
        names = [f.__name__ if callable(f) else f for f in self.date_features]
        date_features = [
            f for f, name in zip(self.date_features, names) if name not in df
        ]
        if date_features:
            unique_dates = df[self.time_col].unique()
            if isinstance(df, pd.DataFrame):
                # all kinds of trickery to make this fast
                unique_dates = pd.Index(unique_dates)
                date2pos = {date: i for i, date in enumerate(unique_dates)}
                restore_idxs = df[self.time_col].map(date2pos)
                for feature in date_features:
                    feat_name, feat_vals = self._compute_date_feature(
                        unique_dates, feature
                    )
                    df[feat_name] = feat_vals[restore_idxs]
            elif isinstance(df, pl_DataFrame):
                exprs = []
                for feat in date_features:  # type: ignore
                    name, vals = self._compute_date_feature(pl.col(self.time_col), feat)
                    exprs.append(vals.alias(name))
                feats = unique_dates.to_frame().with_columns(*exprs)
                df = df.join(feats, on=self.time_col, how="left")

        # assemble return
        if return_X_y:
            if self.weight_col is not None:
                x_cols = [self.weight_col, *self.features_order_]
            else:
                x_cols = self.features_order_
            X = df[x_cols]
            if as_numpy:
                X = ufp.to_numpy(X)
            return X, target
        if effective_max_horizon is not None:
            # remove original target
            out_cols = [c for c in df.columns if c != self.target_col]
            df = df[out_cols]
            target_names = [f"{self.target_col}{i}" for i in range(effective_max_horizon)]
            df = ufp.assign_columns(df, target_names, target)
        else:
            df = ufp.copy_if_pandas(df, deep=False)
            df = ufp.assign_columns(df, self.target_col, target)
        return df

    def _transform_per_horizon(
        self,
        prep: DFType,
        original_df: DFType,
        horizons: List[int],
        target_col: str,
        as_numpy: bool = False,
    ) -> Iterator[Tuple[int, Union[DFType, np.ndarray], np.ndarray]]:
        """Generator that yields (h, X, y) tuples for each horizon.

        For horizon h:
        - Dynamic exogenous features are aligned to predict h steps ahead
        - Target at each horizon is taken from the preprocessed dataframe
        - Lag/date features remain unchanged (computed from current time)

        Args:
            prep: Preprocessed dataframe with expanded targets (y0, y1, ..., y{max_horizon-1})
            original_df: Original input dataframe for exog feature lookup
            horizons: List of horizons to process (0-indexed)
            target_col: Name of target column
            as_numpy: Whether to convert X to numpy array

        Yields:
            Tuple of (horizon_index, X, y) where horizon_index is 0-indexed
        """
        exog_cols = self._get_dynamic_exog_cols(list(original_df.columns))

        # Get feature columns (excluding target columns)
        if self.weight_col is not None:
            x_cols = [self.weight_col, *self.features_order_]
        else:
            x_cols = self.features_order_

        # Non-exog feature columns (lags, date features, static)
        non_exog_cols = [c for c in x_cols if c not in exog_cols]

        # Build exog lookup dictionary from original_df for efficient lookups
        # Key: (id, time) -> exog values
        if exog_cols:
            # Create a lookup dataframe indexed by (id, time)
            exog_lookup = original_df[[self.id_col, self.time_col] + exog_cols]

        for h in horizons:
            # Target column name for this horizon
            target_col_h = f"{target_col}{h}"

            # Get target for this horizon
            y_h = prep[target_col_h].to_numpy()

            if h == 0 or not exog_cols:
                # No offset needed for horizon 0 or if no exog cols
                X_h = prep[x_cols]
            else:
                # Start with the non-exog features from prep
                X_h = ufp.copy_if_pandas(prep[non_exog_cols], deep=True)

                # Offset timestamps by h to get aligned exogenous features
                offset_times = ufp.offset_times(prep[self.time_col], self.freq, h)

                # Create lookup dataframe with row index to maintain order
                if isinstance(prep, pl_DataFrame):
                    # Polars - use with_row_index()
                    lookup_df = (
                        prep[[self.id_col]]
                        .with_columns(pl_Series('_offset_time', offset_times))
                        .with_row_index('_row_idx')
                    )
                    exog_renamed = exog_lookup.rename({self.time_col: '_offset_time'})
                    merged = lookup_df.join(exog_renamed, on=[self.id_col, '_offset_time'], how='left')
                    merged = merged.sort('_row_idx')

                    # Assign exog columns to X_h
                    for col in exog_cols:
                        X_h = X_h.with_columns(merged[col].alias(col))
                else:
                    # Pandas
                    lookup_df = prep[[self.id_col]].copy()
                    lookup_df['_offset_time'] = offset_times
                    lookup_df['_row_idx'] = np.arange(len(prep))
                    exog_renamed = exog_lookup.rename(columns={self.time_col: '_offset_time'})
                    merged = lookup_df.merge(exog_renamed, on=[self.id_col, '_offset_time'], how='left')
                    # Sort by original row order
                    merged = merged.sort_values('_row_idx')

                    # Assign exog columns to X_h
                    for col in exog_cols:
                        X_h[col] = merged[col].values

                # Reorder columns to match x_cols
                X_h = X_h[x_cols]

            # Filter valid rows (non-NaN target and exog)
            valid = ~np.isnan(y_h)
            if exog_cols and h > 0:
                for col in exog_cols:
                    valid &= ~ufp.is_nan_or_none(X_h[col]).to_numpy()

            X_h = ufp.filter_with_mask(X_h, valid)
            y_h = y_h[valid]

            if as_numpy:
                X_h = ufp.to_numpy(X_h)

            yield h, X_h, y_h

    def fit_transform(
        self,
        data: DFType,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: Optional[List[str]] = None,
        dropna: bool = True,
        keep_last_n: Optional[int] = None,
        max_horizon: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        return_X_y: bool = False,
        as_numpy: bool = False,
        weight_col: Optional[str] = None,
    ) -> Union[DFType, Tuple[DFType, np.ndarray]]:
        """Add the features to `data` and save the required information for the predictions step.

        If not all features are static, specify which ones are in `static_features`.
        If you don't want to drop rows with null values after the transformations set `dropna=False`
        If `keep_last_n` is not None then that number of observations is kept across all series for updates.

        Args:
            max_horizon: Train models for all horizons 1 to max_horizon.
            horizons: Train models only for specific horizons (1-indexed).
                      Mutually exclusive with max_horizon.
        """
        self.dropna = dropna
        self.as_numpy = as_numpy
        self._fit(
            df=data,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            keep_last_n=keep_last_n,
            weight_col=weight_col,
        )
        return self._transform(
            df=data,
            dropna=dropna,
            max_horizon=max_horizon,
            horizons=horizons,
            return_X_y=return_X_y,
            as_numpy=as_numpy,
        )

    def _update_y(self, new: np.ndarray) -> None:
        """Appends the elements of `new` to every time serie.

        These values are used to update the transformations and are stored as predictions.
        """
        if not hasattr(self, "y_pred"):
            self.y_pred = []
        self.y_pred.append(new)
        new_arr = np.asarray(new)
        self.ga = self.ga.append(new_arr)
        global_tfms = self._get_global_tfms()
        if global_tfms:
            assert self._global_ga is not None
            new_val = np.array([new_arr.sum()], dtype=self.ga.data.dtype)
            combined = np.concatenate([self._global_ga.data, new_val])
            indptr = np.array([0, combined.size], dtype=np.int32)
            self._global_ga = GroupedArray(combined, indptr)
            if self._global_times is not None:
                if isinstance(self._global_times, pl_Series):
                    self._global_times = pl.concat(
                        [self._global_times, pl.Series([self.curr_dates[0]])]
                    )
                else:
                    self._global_times = self._global_times.append(
                        pd.Index([self.curr_dates[0]])
                    )
        group_tfms = self._get_group_tfms()
        if group_tfms:
            for group_cols, _tfms in group_tfms.items():
                state = self._group_states[group_cols]
                group_idx = state["group_idx"]
                n_groups = len(state["group_uids"])
                group_sums = np.zeros(n_groups, dtype=self.ga.data.dtype)
                np.add.at(group_sums, group_idx, new_arr)
                state["ga"] = state["ga"].append(group_sums)

    def _update_features(self) -> DataFrame:
        """Compute the current values of all the features using the latest values of the time series."""
        self.curr_dates: Union[pd.Index, pl_Series] = ufp.offset_times(
            self.curr_dates, self.freq, 1
        )
        self.test_dates.append(self.curr_dates)

        features = self._compute_transforms(self.transforms, updates_only=True)
        global_tfms = self._get_global_tfms()
        if global_tfms:
            assert self._global_ga is not None
            global_updates = self._global_ga.apply_transforms(
                transforms=global_tfms, updates_only=True
            )
            n_series = len(self.uids)
            for name, vals in global_updates.items():
                features[name] = np.full(n_series, vals[0])
        group_tfms = self._get_group_tfms()
        if group_tfms:
            for group_cols, tfms in group_tfms.items():
                state = self._group_states[group_cols]
                updates = state["ga"].apply_transforms(transforms=tfms, updates_only=True)
                group_idx = state["group_idx"]
                for name, vals in updates.items():
                    features[name] = vals[group_idx]

        for feature in self.date_features:
            feat_name, feat_vals = self._compute_date_feature(self.curr_dates, feature)
            features[feat_name] = feat_vals

        if isinstance(self.last_dates, pl_Series):
            df_constructor = pl_DataFrame
        else:
            df_constructor = pd.DataFrame
        features_df = df_constructor(features)[self.features]
        return ufp.horizontal_concat([self.static_features_, features_df])

    def _get_raw_predictions(self) -> np.ndarray:
        return np.array(self.y_pred).ravel("F")

    def _get_future_ids(self, h: int):
        if isinstance(self.uids, pl_Series):
            uids = pl.concat([self.uids for _ in range(h)]).sort()
        else:
            uids = pd.Series(
                np.repeat(self.uids, h), name=self.id_col, dtype=self.uids.dtype
            )
        return uids

    def _get_predictions(self) -> DataFrame:
        """Get all the predicted values with their corresponding ids and datestamps."""
        h = len(self.y_pred)
        if isinstance(self.uids, pl_Series):
            df_constructor = pl_DataFrame
        else:
            df_constructor = pd.DataFrame
        uids = self._get_future_ids(h)
        df = df_constructor(
            {
                self.id_col: uids,
                self.time_col: np.array(self.test_dates).ravel("F"),
                f"{self.target_col}_pred": self._get_raw_predictions(),
            },
        )
        return df

    def _get_features_for_next_step(self, X_df=None):
        new_x = self._update_features()
        if X_df is not None:
            n_series = len(self.uids)
            h = X_df.shape[0] // n_series  # how many timestamps per series
            # Use min to cap at last available row if self._h exceeds available data
            row_offset = min(self._h, h - 1)
            rows = np.arange(row_offset, X_df.shape[0], h)
            X = ufp.take_rows(X_df, rows)
            X = ufp.drop_index_if_pandas(X)
            new_x = ufp.horizontal_concat([new_x, X])
        if isinstance(new_x, pd.DataFrame):
            nulls = new_x.isnull().any()
            cols_with_nulls = nulls[nulls].index.tolist()
        else:
            nulls = new_x.select(pl.all().is_null().any())
            cols_with_nulls = [k for k, v in nulls.to_dicts()[0].items() if v]
        if cols_with_nulls:
            warnings.warn(f'Found null values in {", ".join(cols_with_nulls)}.')
        self._h += 1
        new_x = new_x[self.features_order_]
        if self.as_numpy:
            new_x = ufp.to_numpy(new_x)
        return new_x

    @contextmanager
    def _backup(self) -> Iterator[None]:
        # this gets modified during predict because the predictions are appended
        ga = copy.copy(self.ga)
        # if these save state (like ExpandingMean) they'll get modified by the updates
        lag_tfms = copy.deepcopy(self.transforms)
        group_states = copy.deepcopy(getattr(self, "_group_states", {}))
        global_ga = copy.copy(getattr(self, "_global_ga", None))
        global_times = copy.copy(getattr(self, "_global_times", None))
        try:
            yield
        finally:
            self.ga = ga
            self.transforms = lag_tfms
            if global_ga is not None or global_times is not None:
                self._global_ga = global_ga
                self._global_times = global_times
            if group_states:
                self._group_states = group_states

    def _predict_setup(self) -> None:
        # TODO: move to utils
        if isinstance(self.last_dates, pl_Series):
            self.curr_dates = self.last_dates.clone()
        else:
            self.curr_dates = self.last_dates.copy()
        self.test_dates: List[Union[pd.Index, pl_Series]] = []
        self.y_pred = []
        self._h = 0

    def _predict_recursive(
        self,
        models: Dict[str, BaseEstimator],
        horizon: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        X_df: Optional[DFType] = None,
    ) -> DFType:
        """Use `model` to predict the next `horizon` timesteps."""
        for i, (name, model) in enumerate(models.items()):
            with self._backup():
                self._predict_setup()
                for _ in range(horizon):
                    new_x = self._get_features_for_next_step(X_df)
                    if before_predict_callback is not None:
                        new_x = before_predict_callback(new_x)
                    predictions = model.predict(new_x)
                    if after_predict_callback is not None:
                        predictions = after_predict_callback(predictions)
                    self._update_y(predictions)
                if i == 0:
                    preds = self._get_predictions()
                    rename_dict = {f"{self.target_col}_pred": name}
                    preds = ufp.rename(preds, rename_dict)
                else:
                    raw_preds = self._get_raw_predictions()
                    preds = ufp.assign_columns(preds, name, raw_preds)
        return preds

    def _predict_multi(
        self,
        models: Dict[str, Dict[int, BaseEstimator]],
        horizon: int,
        before_predict_callback: Optional[Callable] = None,
        X_df: Optional[DFType] = None,
    ) -> DFType:
        assert self.max_horizon is not None
        if horizon > self.max_horizon:
            raise ValueError(
                f"horizon must be at most max_horizon ({self.max_horizon})"
            )

        # Determine horizons to predict based on _horizons (sparse) or all up to horizon
        internal_horizons = getattr(self, "_horizons", None)

        # Check if horizons are sparse (not a contiguous range from 0)
        full_range = list(range(self.max_horizon))
        is_sparse = internal_horizons is not None and internal_horizons != full_range

        if is_sparse:
            # Sparse horizons: filter to those <= requested horizon (0-indexed)
            assert internal_horizons is not None  # mypy: guaranteed by is_sparse check
            horizons_to_predict = [h for h in internal_horizons if h < horizon]
            if not horizons_to_predict:
                raise ValueError(
                    f"No trained horizons available for prediction up to h={horizon}. "
                    f"Trained horizons (1-indexed): {[h + 1 for h in internal_horizons]}"
                )
            output_horizon = len(horizons_to_predict)
        else:
            # Full horizons: predict all from 0 to horizon-1
            horizons_to_predict = list(range(horizon))
            output_horizon = horizon

        self._predict_setup()
        uids = self._get_future_ids(output_horizon)

        if is_sparse:
            # Generate dates only for the specific horizons we're predicting
            # uids structure is [s0, s0, ..., s0 (output_horizon times), s1, s1, ..., s1, ...]
            # So dates need to be [s0_h0, s0_h1, ..., s1_h0, s1_h1, ...]
            if isinstance(self.curr_dates, pl_Series):
                df_constructor = pl_DataFrame
                # Compute dates for all horizons, then stack and flatten
                dates_per_horizon = [ufp.offset_times(self.curr_dates, self.freq, h + 1) for h in horizons_to_predict]
                # Stack: each row is a series, each col is a horizon
                dates_matrix = pl.DataFrame(dates_per_horizon).transpose()
                # Flatten row by row: [s0_h0, s0_h1, ..., s1_h0, s1_h1, ...]
                dates = dates_matrix.to_numpy().ravel()
                dates = pl.Series(dates)
            else:
                df_constructor = pd.DataFrame
                # Compute dates for all horizons, then stack and flatten
                dates_per_horizon = [ufp.offset_times(self.curr_dates, self.freq, h + 1) for h in horizons_to_predict]
                # Stack: each row is a series, each col is a horizon
                dates_matrix = np.column_stack(dates_per_horizon)
                # Flatten row by row: [s0_h0, s0_h1, ..., s1_h0, s1_h1, ...]
                dates = dates_matrix.ravel()
        else:
            # Original behavior: generate contiguous date range
            starts = ufp.offset_times(self.curr_dates, self.freq, 1)
            dates = ufp.time_ranges(starts, self.freq, periods=horizon)
            if isinstance(self.curr_dates, pl_Series):
                df_constructor = pl_DataFrame
            else:
                df_constructor = pd.DataFrame

        result = df_constructor({self.id_col: uids, self.time_col: dates})

        for name, model in models.items():
            with self._backup():
                self._predict_setup()
                predictions = np.empty((len(self.uids), output_horizon))

                for out_idx, h in enumerate(horizons_to_predict):
                    # Advance features to the correct horizon step
                    # We need to step through all horizons up to h to maintain state
                    while self._h <= h:
                        new_x = self._get_features_for_next_step(X_df)

                    if before_predict_callback is not None:
                        new_x = before_predict_callback(new_x)

                    horizon_model = model[h]
                    preds = horizon_model.predict(new_x)
                    if len(preds) != len(self.uids):
                        raise ValueError(f"Model returned {len(preds)} predictions but expected {len(self.uids)}")
                    predictions[:, out_idx] = preds

                raw_preds = predictions.ravel()
                result = ufp.assign_columns(result, name, raw_preds)
        return result

    def _has_ga_target_tfms(self):
        return any(
            isinstance(tfm, _BaseGroupedArrayTargetTransform)
            for tfm in self.target_transforms
        )

    @contextmanager
    def _maybe_subset(self, idxs: Optional[np.ndarray]) -> Iterator[None]:
        # save original
        ga = self.ga
        uids = self.uids
        statics = self.static_features_
        last_dates = self.last_dates
        targ_tfms = copy.copy(self.target_transforms)
        lag_tfms = copy.deepcopy(self.transforms)

        if idxs is not None:
            # assign subsets
            self.ga = self.ga.take(idxs)
            self.uids = uids[idxs]
            self.static_features_ = ufp.take_rows(statics, idxs)
            self.static_features_ = ufp.drop_index_if_pandas(self.static_features_)
            self.last_dates = last_dates[idxs]
            if self.target_transforms is not None:
                for i, tfm in enumerate(self.target_transforms):
                    if isinstance(tfm, _BaseGroupedArrayTargetTransform):
                        self.target_transforms[i] = tfm.take(idxs)
            for name, lag_tfm in self.transforms.items():
                if isinstance(lag_tfm, _BaseLagTransform):
                    lag_tfm = lag_tfm.take(idxs)
                self.transforms[name] = lag_tfm
        try:
            yield
        finally:
            self.ga = ga
            self.uids = uids
            self.static_features_ = statics
            self.last_dates = last_dates
            self.target_transforms = targ_tfms
            self.transforms = lag_tfms

    def predict(
        self,
        models: Dict[str, Union[BaseEstimator, Dict[int, BaseEstimator]]],
        horizon: int,
        before_predict_callback: Optional[Callable] = None,
        after_predict_callback: Optional[Callable] = None,
        X_df: Optional[DFType] = None,
        ids: Optional[List[str]] = None,
    ) -> DFType:
        if ids is not None and (self._get_global_tfms() or self._get_group_tfms()):
            raise ValueError(
                "Cannot use `ids` with global or group lag transforms. "
                "These transforms require forecasting all series together."
            )
        self._check_aligned_ends()
        if ids is not None:
            unseen = set(ids) - set(self.uids)
            if unseen:
                raise ValueError(
                    f"The following ids weren't seen during training and thus can't be forecasted: {unseen}"
                )
            idxs: Optional[np.ndarray] = np.where(ufp.is_in(self.uids, ids))[0]
        else:
            idxs = None
        with self._maybe_subset(idxs):
            if X_df is not None:
                if self.id_col not in X_df or self.time_col not in X_df:
                    raise ValueError(
                        f"X_df must have '{self.id_col}' and '{self.time_col}' columns."
                    )
                if X_df.shape[1] < 3:
                    raise ValueError("Found no exogenous features in `X_df`.")
                statics = [c for c in self.static_features_.columns if c != self.id_col]
                dynamics = [
                    c for c in X_df.columns if c not in [self.id_col, self.time_col]
                ]
                common = [c for c in dynamics if c in statics]
                if common:
                    raise ValueError(
                        f"The following features were provided through `X_df` but were considered as static during fit: {common}.\n"
                        "Please re-run the fit step using the `static_features` argument to indicate which features are static. "
                        "If all your features are dynamic please provide an empty list (static_features=[])."
                    )
                starts = ufp.offset_times(self.last_dates, self.freq, 1)
                ends = ufp.offset_times(self.last_dates, self.freq, horizon)
                expected_rows_X = len(self.uids) * horizon
                dates_validation = type(X_df)(
                    {
                        self.id_col: self.uids,
                        "_start": starts,
                        "_end": ends,
                    }
                )
                X_df = ufp.join(X_df, dates_validation, on=self.id_col)
                mask = ufp.between(X_df[self.time_col], X_df["_start"], X_df["_end"])
                X_df = ufp.filter_with_mask(X_df, mask)
                if X_df.shape[0] < len(self.uids):
                    msg = (
                        "Found missing inputs in X_df. "
                        "It should have at least one row per id.\n"
                        "You can get the expected structure by running `MLForecast.make_future_dataframe(h)` "
                        "or get the missing combinations in your current `X_df` by running `MLForecast.get_missing_future(h, X_df)`."
                    )
                    raise ValueError(msg)
                # Warn if partial features provided
                if X_df.shape[0] < expected_rows_X:
                    warnings.warn(
                        f"X_df has {X_df.shape[0]} rows but {expected_rows_X} expected for horizon {horizon}. "
                        "Features will be reused for missing horizon steps. "
                        "Use `make_future_dataframe(h)` or `get_missing_future(h, X_df)` to generate complete features."
                    )
                drop_cols = [self.id_col, self.time_col, "_start", "_end"]
                X_df = ufp.sort(X_df, [self.id_col, self.time_col])
                X_df = ufp.drop_columns(X_df, drop_cols)
            if getattr(self, "max_horizon", None) is None:
                preds = self._predict_recursive(
                    models=models,
                    horizon=horizon,
                    before_predict_callback=before_predict_callback,
                    after_predict_callback=after_predict_callback,
                    X_df=X_df,
                )
            else:
                preds = self._predict_multi(
                    models=models,
                    horizon=horizon,
                    before_predict_callback=before_predict_callback,
                    X_df=X_df,
                )
            if self.target_transforms is not None:
                if self._has_ga_target_tfms():
                    model_cols = [
                        c
                        for c in preds.columns
                        if c not in (self.id_col, self.time_col)
                    ]
                    # Calculate actual predictions per series (handles sparse horizons)
                    preds_per_series = len(preds) // len(self.uids)
                    indptr = np.arange(0, preds_per_series * (len(self.uids) + 1), preds_per_series)
                for tfm in self.target_transforms[::-1]:
                    if isinstance(tfm, _BaseGroupedArrayTargetTransform):
                        for col in model_cols:
                            ga = GroupedArray(
                                preds[col].to_numpy().astype(self.ga.data.dtype), indptr
                            )
                            ga = tfm.inverse_transform(ga)
                            preds = ufp.assign_columns(preds, col, ga.data)
                    else:
                        preds = tfm.inverse_transform(preds)
        return preds

    def save(self, path: Union[str, Path]) -> None:
        with fsspec.open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path], protocol: Optional[str] = None) -> "TimeSeries":
        with fsspec.open(path, "rb", protocol=protocol) as f:
            ts = cloudpickle.load(f)
        return ts
    
    def _validate_new_df(self, df: DataFrame) -> None:
        if isinstance(df, pl.DataFrame):
            df = df.sort([self.id_col, self.time_col])
            stats = (
                df.group_by(self.id_col)
                .agg(
                    pl.col(self.time_col).min().alias("_min"),
                    pl.col(self.time_col).max().alias("_max"),
                    pl.len().alias("_size"),
                )
                .sort(self.id_col)
            )
            last_dates_df = pl_DataFrame(
                {self.id_col: self.uids, "_last": self.last_dates}
            )
            expected_start = ufp.offset_times(last_dates_df["_last"], self.freq, 1)
            expected_df = last_dates_df.with_columns(
                pl.Series(name="_expected_start", values=expected_start)
            ).select([self.id_col, "_expected_start"])
            stats = stats.with_columns(pl.col(self.id_col).cast(pl.Utf8))
            expected_df = expected_df.with_columns(pl.col(self.id_col).cast(pl.Utf8))
            stats = stats.join(expected_df, on=self.id_col, how="left")
            bad_starts = stats.filter(
                pl.col("_expected_start").is_not_null()
                & (pl.col("_min") != pl.col("_expected_start"))
            )
            if bad_starts.height:
                bad_ids = bad_starts[self.id_col].to_list()
                raise ValueError(
                    "Series have invalid start dates. "
                    f"Expected start at last_date + freq for: {bad_ids}."
                )
            expected_next = ufp.offset_times(df[self.time_col], self.freq, 1)
            df_check = df.with_columns(
                pl.Series(name="_expected_next", values=expected_next)
            ).with_columns(
                pl.col(self.time_col).shift(-1).over(self.id_col).alias("_next")
            )
            gaps = df_check.filter(
                pl.col("_next").is_not_null()
                & (pl.col("_expected_next") != pl.col("_next"))
            )
            if gaps.height:
                bad_ids = gaps[self.id_col].unique().to_list()
                raise ValueError(
                    "Found gaps or duplicate timestamps in the update for: "
                    f"{bad_ids}."
                )
            return
        df = df.sort_values([self.id_col, self.time_col])
        stats = (
            df.groupby(self.id_col, observed=True)[self.time_col]
            .agg(["min", "max", "size"])
            .rename(columns={"min": "_min", "max": "_max", "size": "_size"})
            .reset_index()
        )
        last_dates_df = pd.DataFrame(
            {self.id_col: self.uids, "_last": self.last_dates}
        )
        expected_start = ufp.offset_times(last_dates_df["_last"], self.freq, 1)
        expected_df = pd.DataFrame(
            {self.id_col: last_dates_df[self.id_col], "_expected_start": expected_start}
        )
        stats[self.id_col] = stats[self.id_col].astype(str)
        expected_df[self.id_col] = expected_df[self.id_col].astype(str)
        stats = stats.merge(expected_df, on=self.id_col, how="left")
        start_mismatch = stats["_expected_start"].notna() & (
            stats["_min"] != stats["_expected_start"]
        )
        if start_mismatch.any():
            bad_ids = stats.loc[start_mismatch, self.id_col].tolist()
            raise ValueError(
                "Series have invalid start dates. "
                f"Expected start at last_date + freq for: {bad_ids}."
            )
        expected_next = ufp.offset_times(df[self.time_col], self.freq, 1)
        next_time = df.groupby(self.id_col, observed=True)[self.time_col].shift(-1)
        gaps = next_time.notna() & (expected_next != next_time)
        if gaps.any():
            bad_ids = df.loc[gaps, self.id_col].unique().tolist()
            raise ValueError(
                "Found gaps or duplicate timestamps in the update for: "
                f"{bad_ids}."
            )

    def update(self, df: DataFrame, validate_input: bool = False) -> None:
        """Update the values of the stored series.

        Args:
            df: New observations to append.
            validate_input: If True, validate continuity, start dates, and frequency.
        """
        validate_format(df, self.id_col, self.time_col, self.target_col)
        uids = self.uids
        if isinstance(uids, pd.Index):
            uids = pd.Series(uids)
        uids, new_ids = ufp.match_if_categorical(uids, df[self.id_col])
        df = ufp.copy_if_pandas(df, deep=False)
        df = ufp.assign_columns(df, self.id_col, new_ids)
        df = ufp.sort(df, by=[self.id_col, self.time_col])
        values = df[self.target_col].to_numpy()
        values = values.astype(self.ga.data.dtype, copy=False)
        self._check_aligned_ends()
        if self._get_global_tfms() or self._get_group_tfms():
            if isinstance(df, pd.DataFrame):
                expected_ids = pd.Index(uids).union(pd.Index(new_ids))
                expected_count = len(expected_ids)
                counts = df.groupby(self.time_col, observed=True)[self.id_col].nunique()
                bad_times = counts[counts != expected_count]
                if not bad_times.empty:
                    raise ValueError(
                        "Global and group lag transforms require updates to include all series for each timestamp."
                    )
            else:
                expected_ids = pl.concat([pl.Series(uids), pl.Series(new_ids)]).unique()
                expected_count = expected_ids.len()
                counts = (
                    df.group_by(self.time_col)
                    .agg(pl.col(self.id_col).n_unique().alias("_n_ids"))
                )
                bad_times = counts.filter(pl.col("_n_ids") != expected_count)
                if bad_times.height:
                    raise ValueError(
                        "Global and group lag transforms require updates to include all series for each timestamp."
                    )
        if validate_input:   
            self._validate_new_df(df=df) 
        id_counts = ufp.counts_by_id(df, self.id_col)
        try:
            sizes = ufp.join(uids, id_counts, on=self.id_col, how="outer_coalesce")
        except (KeyError, ValueError):
            # pandas raises key error, polars before coalesce raises value error
            sizes = ufp.join(uids, id_counts, on=self.id_col, how="outer")
        sizes = ufp.fill_null(sizes, {"counts": 0})
        sizes = ufp.sort(sizes, by=self.id_col)
        new_groups = ~ufp.is_in(sizes[self.id_col], uids)
        last_dates = ufp.group_by_agg(df, self.id_col, {self.time_col: "max"})
        last_dates = ufp.join(sizes, last_dates, on=self.id_col, how="left")
        curr_last_dates = type(df)({self.id_col: uids, "_curr": self.last_dates})
        last_dates = ufp.join(last_dates, curr_last_dates, on=self.id_col, how="left")
        last_dates = ufp.fill_null(last_dates, {self.time_col: last_dates["_curr"]})
        last_dates = ufp.sort(last_dates, by=self.id_col)
        self.last_dates = ufp.cast(last_dates[self.time_col], self.last_dates.dtype)
        self.uids = ufp.sort(sizes[self.id_col])
        if isinstance(df, pd.DataFrame):
            self.uids = pd.Index(self.uids)
            self.last_dates = pd.Index(self.last_dates)
        if new_groups.any():
            if self.target_transforms is not None:
                raise ValueError("Can not update target_transforms with new series.")
            new_ids = ufp.filter_with_mask(sizes[self.id_col], new_groups)
            new_ids_df = ufp.filter_with_mask(df, ufp.is_in(df[self.id_col], new_ids))
            new_ids_counts = ufp.counts_by_id(new_ids_df, self.id_col)
            new_statics = ufp.take_rows(
                df, new_ids_counts["counts"].to_numpy().cumsum() - 1
            )
            new_statics = new_statics[self.static_features_.columns]
            self.static_features_ = ufp.vertical_concat(
                [self.static_features_, new_statics]
            )
            self.static_features_ = ufp.sort(self.static_features_, self.id_col)
        if self.target_transforms is not None:
            if self._has_ga_target_tfms():
                indptr = np.append(0, id_counts["counts"]).cumsum()
            for tfm in self.target_transforms:
                if isinstance(tfm, _BaseGroupedArrayTargetTransform):
                    ga = GroupedArray(values, indptr)
                    ga = tfm.update(ga)
                    df = ufp.assign_columns(df, self.target_col, ga.data)
                else:
                    df = tfm.update(df)
                values = df[self.target_col].to_numpy()
        self.ga = self.ga.append_several(
            new_sizes=sizes["counts"].to_numpy().astype(np.int32),
            new_values=values,
            new_groups=new_groups.to_numpy(),
        )
        global_tfms = self._get_global_tfms()
        if global_tfms:
            global_df = ufp.group_by_agg(
                df[[self.time_col, self.target_col]],
                self.time_col,
                {self.target_col: "sum"},
                maintain_order=True,
            )
            global_df = ufp.sort(global_df, by=self.time_col)
            global_values = global_df[self.target_col].to_numpy().astype(
                self.ga.data.dtype
            )
            if self._global_ga is None:
                self._global_ga = GroupedArray(
                    global_values,
                    np.array([0, global_values.size], dtype=np.int32),
                )
            else:
                combined = np.concatenate([self._global_ga.data, global_values])
                indptr = np.array([0, combined.size], dtype=np.int32)
                self._global_ga = GroupedArray(combined, indptr)
            if self._global_times is None:
                if isinstance(global_df, pd.DataFrame):
                    self._global_times = pd.Index(global_df[self.time_col])
                else:
                    self._global_times = global_df[self.time_col]
            else:
                if isinstance(self._global_times, pl_Series):
                    self._global_times = pl.concat(
                        [self._global_times, global_df[self.time_col]]
                    )
                else:
                    self._global_times = self._global_times.append(
                        pd.Index(global_df[self.time_col])
                    )
        group_tfms = self._get_group_tfms()
        if group_tfms:
            def _attach_group_id(data, groups, cols):
                if isinstance(data, pd.DataFrame):
                    return data.merge(groups, on=cols, how="left")
                return data.join(groups, on=cols, how="left")

            for group_cols in group_tfms.keys():
                state = self._group_states[group_cols]
                group_cols_list = state["group_cols"]
                group_df = ufp.group_by_agg(
                    df[group_cols_list + [self.time_col, self.target_col]],
                    group_cols_list + [self.time_col],
                    {self.target_col: "sum"},
                    maintain_order=True,
                )
                group_df = ufp.sort(group_df, by=group_cols_list + [self.time_col])
                groups = state["groups"]
                group_df = _attach_group_id(group_df, groups, group_cols_list)
                if isinstance(group_df, pd.DataFrame):
                    missing = group_df["_group_id"].isna()
                    if missing.any():
                        new_groups = group_df.loc[missing, group_cols_list].drop_duplicates()
                        new_groups = new_groups.reset_index(drop=True)
                        start = len(groups)
                        new_groups["_group_id"] = np.arange(
                            start, start + len(new_groups), dtype=np.int64
                        )
                        groups = pd.concat([groups, new_groups], ignore_index=True)
                        group_df = group_df.drop(columns="_group_id").merge(
                            groups, on=group_cols_list, how="left"
                        )
                else:
                    missing = group_df["_group_id"].is_null()
                    if missing.any():
                        new_groups = (
                            group_df.filter(missing)
                            .select(group_cols_list)
                            .unique(maintain_order=True)
                        )
                        start = groups.height
                        new_groups = new_groups.with_row_index(
                            name="_group_id", offset=start
                        )
                        groups = pl.concat([groups, new_groups], how="vertical")
                        group_df = group_df.drop("_group_id").join(
                            groups, on=group_cols_list, how="left"
                        )
                state["groups"] = groups
                id_counts = ufp.counts_by_id(group_df, "_group_id")
                uids = state["group_uids"]
                if isinstance(uids, pd.Index):
                    uids = pd.Series(uids)
                uids, new_ids = ufp.match_if_categorical(uids, group_df["_group_id"])
                group_df = ufp.assign_columns(group_df, "_group_id", new_ids)
                group_df = ufp.sort(group_df, by=["_group_id", self.time_col])
                values = group_df[self.target_col].to_numpy().astype(self.ga.data.dtype, copy=False)
                try:
                    sizes = ufp.join(uids, id_counts, on="_group_id", how="outer_coalesce")
                except (KeyError, ValueError):
                    sizes = ufp.join(uids, id_counts, on="_group_id", how="outer")
                sizes = ufp.fill_null(sizes, {"counts": 0})
                sizes = ufp.sort(sizes, by="_group_id")
                new_groups = ~ufp.is_in(sizes["_group_id"], uids)
                state["ga"] = state["ga"].append_several(
                    new_sizes=sizes["counts"].to_numpy().astype(np.int32),
                    new_values=values,
                    new_groups=new_groups.to_numpy(),
                )
                state["group_uids"] = ufp.sort(sizes["_group_id"])
                if isinstance(self.static_features_, pd.DataFrame):
                    series_group_id = self.static_features_[group_cols_list].merge(
                        groups, on=group_cols_list, how="left"
                    )["_group_id"].to_numpy()
                    state["group_idx"] = series_group_id.astype(np.int64, copy=False)
                else:
                    series_group_id = (
                        self.static_features_
                        .select(group_cols_list)
                        .join(groups, on=group_cols_list, how="left")["_group_id"]
                        .to_numpy()
                    )
                    state["group_idx"] = series_group_id.astype(np.int64, copy=False)
