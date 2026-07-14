__all__ = ["TimeSeries"]


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
    Sequence,
    Tuple,
    Union,
)

import cloudpickle
import fsspec
import narwhals as nw
import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from utilsforecast.compat import (
    DataFrame,
    DFType,
    pl,
    pl_DataFrame,
    pl_Series,
)
from utilsforecast.validation import validate_format, validate_freq

from mlforecast.target_transforms import (
    BaseTargetTransform,
    _BaseGroupedArrayTargetTransform,
)

from .compat import CatBoostRegressor
from .data_validation import _index_to_series
from .grouped_array import GroupedArray
from .lag_transforms import Lag, _BaseLagTransform
from .pooled import (
    PooledState,
    _order_preserving_left_join,
    compute_pooled_features,
)
from .utils import (
    _DUMMY_FEATURE_VALUES,
    _ShortSeriesException,
    _compute_date_dummies,
    _resolve_num_threads,
)

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


def _dedupe_preserve_order(items: Iterable[str]) -> list:
    return list(dict.fromkeys(items))


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


def _static_feature_changes_over_time(start_series, end_series) -> bool:
    """Whether a static feature differs between each series' start and end.

    Backend-agnostic (narwhals) and null-safe: two missing values count as equal
    (so an all-null static feature, e.g. a null groupby key, is not flagged as
    changing), while a missing-vs-present pair counts as a change. Mirrors the
    dtype-aware "missing" rule used for pooled bucket keys (null, plus NaN for
    float dtypes).
    """
    a = nw.from_native(start_series, series_only=True)
    b = nw.from_native(end_series, series_only=True)
    a_missing = a.is_null()
    b_missing = b.is_null()
    if a.dtype.is_float():
        a_missing = a_missing | a.is_nan()
    if b.dtype.is_float():
        b_missing = b_missing | b.is_nan()
    both_missing = a_missing & b_missing
    both_present_equal = (~a_missing) & (~b_missing) & (a == b)
    changed = ~(both_missing | both_present_equal)
    return bool(changed.fill_null(False).any())


def _to_native_index(values, *, df):
    """Wrap ids/dates into the backend-native container mlforecast exposes as
    ``TimeSeries.uids`` / ``TimeSeries.last_dates``.

    pandas -> ``pd.Index``, polars -> ``pl.Series``. Centralized here because these
    attributes are observable API (tests rely on ``.tolist()`` and fancy indexing),
    so their native type must be produced in exactly one place.
    """
    if isinstance(df, pd.DataFrame):
        return pd.Index(values)
    return pl_Series(values)


class TimeSeries:
    """Utility class for storing and transforming time series data."""

    # Per-predict state (set in ``_predict_setup``, used in the horizon loop).
    # Declared here so mypy has a type regardless of method processing order.
    # _uniform_dates defaults to False so the date-feature broadcast fast
    # path stays off if the feature computation runs without a prior
    # ``_predict_setup`` (the non-broadcast path is always correct).
    _uniform_dates: bool = False
    _feature_null_cols: List[str]
    _xdf_null_cols: Optional[List[str]]

    def __init__(
        self,
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
        self.freq = freq
        self.date_features_as_dummies = date_features_as_dummies
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
        self.drop_auxiliary_columns = drop_auxiliary_columns
        self.transforms = _parse_transforms(
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            namer=lag_transforms_namer,
        )
        self.horizon_features_: Dict[int, List[str]] = {}
        self.ga: GroupedArray

    def _get_core_lag_tfms(self) -> Dict[str, _BaseLagTransform]:
        return {
            k: v for k, v in self.transforms.items() if isinstance(v, _BaseLagTransform)
        }

    def _get_pooled_tfms(self) -> Dict[Tuple, Dict[str, _BaseLagTransform]]:
        """Group all nonlocal transforms by their pooled key.

        Key structure: ``(mode, group_cols_tuple, partition_cols_tuple)``

        Examples::

            ("global", (), ())                        -- pure global
            ("groupby", ("brand",), ())               -- pure groupby
            ("local", (), ("promo",))                 -- local partition
            ("nonlocal", (), ("promo",))              -- global+partition
            ("nonlocal", ("brand",), ("promo",))      -- groupby+partition
        """
        pooled: Dict[Tuple, Dict[str, _BaseLagTransform]] = {}
        for name, tfm in self.transforms.items():
            if not isinstance(tfm, _BaseLagTransform):
                continue
            is_global = getattr(tfm, "global_", False)
            groupby = getattr(tfm, "groupby", None)
            partition_by = getattr(tfm, "partition_by", None)
            if not is_global and not groupby and not partition_by:
                continue
            if partition_by:
                mode = "local" if (not is_global and not groupby) else "nonlocal"
            elif is_global:
                mode = "global"
            else:
                mode = "groupby"
            group_cols = tuple(groupby) if groupby else ()
            part_cols = tuple(partition_by) if partition_by else ()
            key = (mode, group_cols, part_cols)
            pooled.setdefault(key, {})[name] = tfm
        return pooled

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
            if isinstance(tfm, _BaseLagTransform) and getattr(
                tfm, "partition_by", None
            ):
                continue
            local[name] = tfm
        return local

    def _trim_pooled_states(self) -> None:
        """Trim each pooled state's history under ``keep_last_n``.

        Parity with the ``self.ga`` trim: a pooled state whose transforms are
        *all* finite-window drops its unused history prefix, while a state
        containing any Expanding*/EWM transform keeps full history (pooled has
        no carried accumulator -- it recomputes over the full aggregate vectors
        at predict, so trimming those would move predictions).

        Retention is ``max(keep_last_n, W_state)`` ordinals, where ``W_state``
        is the state's largest finite window. The floor is required and is where
        pooled legitimately diverges from the local coreforecast path: local
        rolling survives an undersized explicit ``keep_last_n`` because
        coreforecast carries a per-transform window buffer; pooled has none (the
        aggregates *are* the buffer), so trimming below ``W_state`` would compute
        windows off a truncated prefix. When ``keep_last_n`` is inferred it
        already equals the global max window, so the floor is then a no-op.
        """
        if self.keep_last_n is None:
            return
        for key, tfms in self._get_pooled_tfms().items():
            state = self._pooled_states.get(key)
            if state is None:
                continue
            tfm_list = list(tfms.values())
            if not all(tfm._is_finite_window for tfm in tfm_list):
                continue
            w_state = max(tfm.update_samples for tfm in tfm_list)
            state.trim_to_last(max(self.keep_last_n, w_state))

    def _initialize_lag_transform_states(self) -> None:
        """Materialize lag transform state for subsequent update-based prediction.

        This is needed when a new ``TimeSeries`` instance is created from historical
        data right before calling ``predict(new_df=...)``. Local (coreforecast)
        transforms need a full ``transform`` pass so stateful transforms like
        ``ExpandingMean`` can initialize their internal buffers before the first
        ``update(...)`` call. Pooled transforms keep their state in ``_ts_aggs``
        (built at construction), so they need no warm-up pass.
        """
        core_tfms = self._get_core_lag_tfms()
        if core_tfms:
            self._compute_transforms(core_tfms, updates_only=False)

    def _check_aligned_ends(self) -> None:
        """Check that all series end at the same timestamp when using pooled lag transforms."""
        if not self._get_pooled_tfms():
            return
        last_dates = _index_to_series(self.last_dates)
        aligned = nw.from_native(last_dates, series_only=True).n_unique() == 1
        if not aligned:
            raise ValueError(
                "Pooled lag transforms require all series to end at the same timestamp "
                "(recursive prediction advances all series in lockstep)."
            )

    @property
    def _date_feature_names(self) -> List[str]:
        names: List[str] = []
        for f in self.date_features:
            if (
                self.date_features_as_dummies
                and isinstance(f, str)
                and f in _DUMMY_FEATURE_VALUES
            ):
                names.extend(f"{f}_{v}" for v in _DUMMY_FEATURE_VALUES[f])
            elif callable(f):
                names.append(f.__name__)
            else:
                names.append(f)
        return names

    @property
    def features(self) -> List[str]:
        """Names of all computed features."""
        return list(self.transforms.keys()) + self._date_feature_names

    def _get_dynamic_exog_cols(self, df_columns: List[str]) -> List[str]:
        """Identify user-provided exogenous columns that need time-alignment."""
        static_cols = set(self.static_features_.columns)
        lag_cols = set(self.transforms.keys())
        date_cols = set(self._date_feature_names) | {
            f
            for f in self.date_features
            if self.date_features_as_dummies
            and isinstance(f, str)
            and f in _DUMMY_FEATURE_VALUES
        }
        exclude = (
            static_cols
            | lag_cols
            | date_cols
            | {self.id_col, self.time_col, self.target_col}
        )
        if self.weight_col is not None:
            exclude.add(self.weight_col)
        return [c for c in df_columns if c not in exclude]

    def _split_horizon_exog_cols(
        self,
        exog_cols: List[str],
        horizon_features: Dict[int, List[str]],
    ) -> Tuple[List[str], Dict[int, List[str]]]:
        """Split exogenous columns into common and horizon-specific sets."""
        if not horizon_features:
            return exog_cols, {}
        matched_cols = {
            col
            for cols in horizon_features.values()
            for col in cols
            if col in exog_cols
        }
        common_exog = [c for c in exog_cols if c not in matched_cols]
        return common_exog, horizon_features

    def _get_cols_for_horizon(
        self,
        h: int,
        common_exog: List[str],
        horizon_exog_map: Dict[int, List[str]],
        exog_cols: List[str],
    ) -> List[str]:
        """Return the ordered feature columns to use for 0-indexed horizon h.

        ``horizon_exog_map`` uses 1-indexed keys (matching the user-facing
        ``horizon_features`` dict), so we convert with ``h + 1``.
        """
        # Internal horizons are 0-indexed; user-facing horizon_features keys are
        # 1-indexed, hence the +1 conversion here.
        allowed_exog = common_exog + horizon_exog_map.get(h + 1, [])
        return [
            c for c in self.features_order_ if c not in exog_cols or c in allowed_exog
        ]

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
        self.uids = _to_native_index(uids, df=df)
        self.last_dates = _to_native_index(times, df=df)
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
        to_drop = [id_col, time_col, target_col]
        if static_features is None:
            partition_cols = {
                col
                for tfm in self.transforms.values()
                if isinstance(tfm, _BaseLagTransform)
                for col in (getattr(tfm, "partition_by", None) or [])
            }
            self._partition_cols = partition_cols
            static_features = [
                c
                for c in df.columns
                if c not in [time_col, target_col] and c not in partition_cols
            ]
        else:
            self._partition_cols = {
                col
                for tfm in self.transforms.values()
                if isinstance(tfm, _BaseLagTransform)
                for col in (getattr(tfm, "partition_by", None) or [])
            }
            if id_col not in static_features:
                static_features = [id_col, *static_features]
            else:
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
            if _static_feature_changes_over_time(
                statics_on_starts[feat], statics_on_ends[feat]
            ):
                raise ValueError(
                    f"{feat} is declared as a static feature but its values change "
                    "over time. Please set the `static_features` argument to "
                    "indicate which features are static.\nIf all of your features "
                    "are dynamic please set `static_features=[]`."
                )
        self.static_features_ = statics_on_ends
        raw_date_sources = {
            f
            for f in self.date_features
            if self.date_features_as_dummies
            and isinstance(f, str)
            and f in _DUMMY_FEATURE_VALUES
        }
        self.features_order_ = [
            c for c in df.columns if c not in to_drop and c not in raw_date_sources
        ] + [f for f in self.features if f not in df.columns]
        if self.drop_auxiliary_columns is True:
            to_exclude = set()
            for lag_tfm in self.transforms.values():
                if not isinstance(lag_tfm, _BaseLagTransform):
                    continue
                for col in getattr(lag_tfm, "groupby", None) or []:
                    to_exclude.add(col)
                for col in getattr(lag_tfm, "partition_by", None) or []:
                    to_exclude.add(col)
        elif self.drop_auxiliary_columns is False:
            to_exclude = set()
        else:
            to_exclude = set(self.drop_auxiliary_columns)
            unknown = [f for f in to_exclude if f not in self.features_order_]
            if unknown:
                warnings.warn(
                    f"The following drop_auxiliary_columns were not found in the feature set: {unknown}",
                    UserWarning,
                )
        self.features_order_ = [f for f in self.features_order_ if f not in to_exclude]
        self._pooled_states: Dict[Tuple, PooledState] = {}
        pooled_tfms = self._get_pooled_tfms()
        if pooled_tfms:
            if self.target_transforms is not None:
                transformed_target = ga.data
                if self._restore_idxs is not None:
                    transformed_target = transformed_target[self._restore_idxs]
                df_for_pooled = ufp.assign_columns(df, target_col, transformed_target)
            else:
                df_for_pooled = df
            for key, tfms in pooled_tfms.items():
                mode, group_cols_t, part_cols_t = key
                if mode == "global" and not part_cols_t:
                    self._pooled_states[key] = PooledState.from_global(
                        sorted_df,
                        id_col=id_col,
                        time_col=time_col,
                        target_col=target_col,
                        ga_data_dtype=ga.data.dtype,
                        n_series=len(ga.indptr) - 1,
                    )
                elif mode == "groupby" and not part_cols_t:
                    for col in group_cols_t:
                        if col not in df.columns:
                            raise ValueError(
                                f"Groupby column '{col}' not found in dataframe."
                            )
                    group_cols_list = list(group_cols_t)
                    missing = [
                        c
                        for c in group_cols_list
                        if c not in self.static_features_.columns
                    ]
                    if missing:
                        raise ValueError(
                            "Groupby columns must be static features. "
                            f"Missing from static_features: {missing}."
                        )
                    self._pooled_states[key] = PooledState.from_groupby(
                        df_for_pooled,
                        group_cols_list=group_cols_list,
                        id_col=id_col,
                        time_col=time_col,
                        target_col=target_col,
                        ga_data_dtype=ga.data.dtype,
                        static_features=self.static_features_,
                    )
                else:
                    all_cols = list(group_cols_t) + list(part_cols_t)
                    for col in all_cols:
                        if col not in df.columns and col != id_col:
                            raise ValueError(
                                f"partition_by/groupby column '{col}' not found in dataframe."
                            )
                    part_group_cols = list(group_cols_t) if group_cols_t else None
                    self._pooled_states[key] = PooledState.from_partition(
                        df_for_pooled,
                        mode=mode,
                        group_cols_list=part_group_cols,
                        partition_cols_list=list(part_cols_t),
                        id_col=id_col,
                        time_col=time_col,
                        target_col=target_col,
                        ga_data_dtype=ga.data.dtype,
                        static_features=self.static_features_,
                        n_series=len(ga.indptr) - 1,
                    )
            for key, state in self._pooled_states.items():
                if state.groups is not None:
                    from .pooled import _compute_idsorted_to_bucket_pos

                    state._idsorted_to_bucket_pos = _compute_idsorted_to_bucket_pos(
                        state.bucket_df,
                        id_col,
                        time_col,
                    )
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

    def _join_bucket_features(
        self,
        features: Dict[str, np.ndarray],
        df: DFType,
        bucket_df: DFType,
        bucket_vals: Dict[str, np.ndarray],
        join_cols: list,
    ) -> None:
        feature_cols = list(bucket_vals.keys())
        if not feature_cols:
            return
        nw_bucket = nw.from_native(bucket_df, eager_only=True)
        backend = nw.get_native_namespace(nw_bucket)
        join_df = nw_bucket.select(join_cols)
        for name, vals in bucket_vals.items():
            join_df = join_df.with_columns(nw.new_series(name, vals, backend=backend))
        join_df = join_df.to_native()
        left = nw.from_native(df, eager_only=True).select(join_cols).to_native()
        joined = nw.to_native(
            _order_preserving_left_join(
                nw.from_native(left), nw.from_native(join_df), on=join_cols
            )
        )
        for name in feature_cols:
            features[name] = joined[name].to_numpy()

    def _compute_date_feature(self, dates, feature) -> Dict[str, Any]:
        """Compute date feature(s) and return as a ``{col_name: values}`` dict."""
        if (
            self.date_features_as_dummies
            and isinstance(feature, str)
            and feature in _DUMMY_FEATURE_VALUES
        ):
            return _compute_date_dummies(dates, feature)

        if callable(feature):
            feat_name = feature.__name__
            feat_vals = feature(dates)
            # callable-returned native types have no narwhals equivalent
            if isinstance(feat_vals, pd.DataFrame):
                return {col: np.asarray(feat_vals[col]) for col in feat_vals.columns}
            if isinstance(feat_vals, (pd.Index, pd.Series)):
                feat_vals = np.asarray(feat_vals)
            return {feat_name: feat_vals}

        # regular string feature
        feat_name = feature
        # DatetimeIndex date-attribute access has no narwhals equivalent
        if isinstance(dates, pd.DatetimeIndex):
            if feature in ("week", "weekofyear"):
                dates = dates.isocalendar()
            feat_vals = getattr(dates, feature)
            if isinstance(feat_vals, (pd.Index, pd.Series)):
                feat_vals = np.asarray(feat_vals)
                feat_dtype = date_features_dtypes.get(feature)
                if feat_dtype is not None:
                    feat_vals = feat_vals.astype(feat_dtype)
        else:
            feat_vals = getattr(dates.dt, feature)()
        return {feat_name: feat_vals}

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
        pooled_tfms = self._get_pooled_tfms()
        if pooled_tfms:
            if self._sort_idxs is not None:
                df_sorted = ufp.take_rows(df, self._sort_idxs)
            else:
                df_sorted = df
            for key, tfms in pooled_tfms.items():
                state = self._pooled_states[key]
                fast_features: Dict[str, Any] = {}
                slow_tfms: Dict[str, _BaseLagTransform] = {}
                for name, tfm in tfms.items():
                    ts_vals = tfm._compute_ts_level_from_aggs(state._ts_aggs)
                    if ts_vals is not None:
                        fast_features[name] = ts_vals
                    else:
                        slow_tfms[name] = tfm
                if fast_features:
                    if state.groups is None:
                        unique_times = np.unique(state.time)
                        time_vals = df_sorted[self.time_col].to_numpy()
                        row_ords = np.searchsorted(unique_times, time_vals)
                        for name, ts_vals_by_bucket in fast_features.items():
                            features[name] = ts_vals_by_bucket[0][row_ords]
                    elif state._idsorted_to_bucket_pos is not None:
                        pos = state._idsorted_to_bucket_pos
                        bid_df = state.bucket_id[pos]
                        ord_df = state.time_index[pos]
                        for name, ts_vals_by_bucket in fast_features.items():
                            out = np.full(len(pos), np.nan)
                            for bid, ts_vals in ts_vals_by_bucket.items():
                                mask = bid_df == bid
                                bucket_ords = ord_df[mask]
                                agg = state._ts_aggs[bid]
                                dense_pos = np.searchsorted(
                                    agg.unique_times, bucket_ords
                                )
                                out[mask] = ts_vals[dense_pos]
                            features[name] = out
                    else:
                        slow_tfms.update({n: tfms[n] for n in fast_features})
                if slow_tfms:
                    bucket_vals = compute_pooled_features(state, slow_tfms)
                    self._join_bucket_features(
                        features,
                        df_sorted,
                        state.bucket_df,
                        bucket_vals,
                        state.join_cols,
                    )
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
        else:
            # we'll be assigning columns below; copy_if_pandas is a no-op on polars (immutable)
            df = ufp.copy_if_pandas(df, deep=False)

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
            self._trim_pooled_states()
        del self._restore_idxs, self._sort_idxs

        # lag transforms
        for feat in self.transforms.keys():
            if feat in features:
                df = ufp.assign_columns(df, feat, features[feat])

        # date features
        def _feature_in_df(f, cols):
            if (
                self.date_features_as_dummies
                and isinstance(f, str)
                and f in _DUMMY_FEATURE_VALUES
            ):
                return all(f"{f}_{v}" in cols for v in _DUMMY_FEATURE_VALUES[f])
            name = f.__name__ if callable(f) else f
            return name in cols

        df_cols = set(df.columns)
        date_features = [
            f for f in self.date_features if not _feature_in_df(f, df_cols)
        ]
        if date_features:
            unique_dates = df[self.time_col].unique()
            # pandas positional-map fast path; narwhals rewrite would regress this hot path
            if isinstance(df, pd.DataFrame):
                # all kinds of trickery to make this fast
                unique_dates = pd.Index(unique_dates)
                date2pos = {date: i for i, date in enumerate(unique_dates)}
                restore_idxs = df[self.time_col].map(date2pos)
                for feature in date_features:
                    for feat_name, feat_vals in self._compute_date_feature(
                        unique_dates, feature
                    ).items():
                        df[feat_name] = feat_vals[restore_idxs]
            # backend-specific date-feature fast path (pandas positional-map + polars expr build); kept for performance
            elif isinstance(df, pl_DataFrame):
                exprs = []
                nw_feats: Dict[str, Any] = {}
                for feat in date_features:  # type: ignore
                    if (
                        self.date_features_as_dummies
                        and isinstance(feat, str)
                        and feat in _DUMMY_FEATURE_VALUES
                    ):
                        nw_feats.update(_compute_date_dummies(unique_dates, feat))
                    else:
                        for name, vals in self._compute_date_feature(
                            pl.col(self.time_col), feat
                        ).items():
                            exprs.append(vals.alias(name))
                feats = unique_dates.to_frame()
                if exprs:
                    feats = feats.with_columns(*exprs)
                for col_name, col_vals in nw_feats.items():
                    feats = feats.with_columns(
                        pl.Series(name=col_name, values=col_vals)
                    )
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
            target_names = [
                f"{self.target_col}{i}" for i in range(effective_max_horizon)
            ]
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
        exog_cols = self._get_dynamic_exog_cols(self.features_order_)
        exog_cols_set = set(exog_cols)
        common_exog_cols, horizon_exog_map = self._split_horizon_exog_cols(
            exog_cols, self.horizon_features_
        )

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
            h_cols = self._get_cols_for_horizon(
                h, common_exog_cols, horizon_exog_map, exog_cols
            )
            h_cols_set = set(h_cols)
            # exog subset for this horizon — used for time-aligned joining and NaN filtering
            horizon_exog = [c for c in h_cols if c in exog_cols_set]
            # weight_col lives in x_cols but not in features_order_ (and thus not in
            # h_cols); keep any x_col that is non-exog (weight, lags, dates, static)
            # or is an allowed exog for this horizon.
            x_cols_h = [c for c in x_cols if c not in exog_cols_set or c in h_cols_set]

            # Target column name for this horizon
            target_col_h = f"{target_col}{h}"

            # Get target for this horizon
            y_h = prep[target_col_h].to_numpy()

            if h == 0 or not horizon_exog:
                # No offset needed for horizon 0 or if no exog cols
                X_h = prep[x_cols_h]
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
                        .with_columns(pl_Series("_offset_time", offset_times))
                        .with_row_index("_row_idx")
                    )
                    exog_renamed = exog_lookup.rename({self.time_col: "_offset_time"})
                    merged = lookup_df.join(
                        exog_renamed, on=[self.id_col, "_offset_time"], how="left"
                    )
                    merged = merged.sort("_row_idx")

                    # Assign exog columns to X_h
                    for col in horizon_exog:
                        X_h = X_h.with_columns(merged[col].alias(col))
                else:
                    # Pandas
                    lookup_df = prep[[self.id_col]].copy()
                    lookup_df["_offset_time"] = offset_times
                    lookup_df["_row_idx"] = np.arange(len(prep))
                    exog_renamed = exog_lookup.rename(
                        columns={self.time_col: "_offset_time"}
                    )
                    merged = lookup_df.merge(
                        exog_renamed, on=[self.id_col, "_offset_time"], how="left"
                    )
                    # Sort by original row order
                    merged = merged.sort_values("_row_idx")

                    # Assign exog columns to X_h
                    for col in horizon_exog:
                        X_h[col] = merged[col].values

                # Reorder columns to match x_cols
                X_h = X_h[x_cols_h]

            # Filter valid rows: rows where any horizon-specific exog is NaN/null
            # are dropped — they cannot be used for this horizon's model even if
            # the target itself is valid.
            valid = ~np.isnan(y_h)
            if horizon_exog and h > 0:
                for col in horizon_exog:
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
        for state in self._pooled_states.values():
            state.append_predictions(self.curr_dates, new_arr, len(new_arr))

    def _update_features(self) -> DataFrame:
        """Compute the current values of all the features using the latest values of the time series.

        Expects the per-predict state from ``_predict_setup`` (``curr_dates``,
        ``static_features_``, pooled states); ``_uniform_dates`` safely
        defaults to False if setup hasn't run.
        """
        features_df = self._compute_features_df()
        return ufp.horizontal_concat([self.static_features_, features_df])

    def _compute_features_df(self) -> DataFrame:
        """Advance ``curr_dates`` one step and compute the features frame (without statics)."""
        self.curr_dates: Union[pd.Index, pl_Series] = ufp.offset_times(
            self.curr_dates, self.freq, 1
        )
        self.test_dates.append(self.curr_dates)

        features = self._compute_transforms(self.transforms, updates_only=True)
        pooled_tfms = self._get_pooled_tfms()
        for key, tfms in pooled_tfms.items():
            state = self._pooled_states[key]
            n_series = len(self.uids)
            slow_tfms: Dict[str, _BaseLagTransform] = {}
            for name, tfm in tfms.items():
                latest = tfm._compute_latest_from_aggs(
                    state._ts_aggs,
                    state.next_time_index_by_bucket,
                )
                if latest is not None:
                    if state.groups is None:
                        features[name] = np.full(n_series, latest[0])
                    else:
                        max_bid = max(
                            max(latest.keys(), default=-1),
                            int(state.series_bucket_id.max()),
                        )
                        lookup = np.full(max_bid + 1, np.nan)
                        for bid, val in latest.items():
                            lookup[bid] = val
                        features[name] = lookup[state.series_bucket_id]
                else:
                    slow_tfms[name] = tfm
            if slow_tfms:
                query = state.build_query_arrays(self.curr_dates, n_series)
                bucket_vals = compute_pooled_features(
                    state,
                    slow_tfms,
                    query_arrays=query,
                )
                if state.groups is None:
                    for name, vals in bucket_vals.items():
                        features[name] = np.full(n_series, vals[-1])
                else:
                    tmp_bid = query[0]
                    n_orig = len(state.y)
                    for name, vals in bucket_vals.items():
                        new_vals = vals[n_orig:]
                        new_bid_vals = tmp_bid[n_orig:]
                        val_map = {}
                        for bv, v in zip(new_bid_vals, new_vals):
                            val_map[bv] = v
                        max_bid = max(
                            max(val_map.keys(), default=-1),
                            int(state.series_bucket_id.max()),
                        )
                        lookup = np.full(max_bid + 1, np.nan)
                        for bid, val in val_map.items():
                            lookup[bid] = val
                        features[name] = lookup[state.series_bucket_id]

        if self._uniform_dates and self.date_features:
            # all series are on the same timestamp, so date features are
            # constant across series: compute them on a single date and
            # broadcast instead of recomputing on n_series identical dates.
            # restricted to non-callables, which are guaranteed elementwise
            broadcast_idx = np.zeros(len(self.uids), dtype=np.int64)
        for feature in self.date_features:
            if self._uniform_dates and not callable(feature):
                for feat_name, feat_vals in self._compute_date_feature(
                    self.curr_dates[:1], feature
                ).items():
                    features[feat_name] = feat_vals[broadcast_idx]
            else:
                for feat_name, feat_vals in self._compute_date_feature(
                    self.curr_dates, feature
                ).items():
                    features[feat_name] = feat_vals

        # native-container boundary — uids/df constructor selected per backend
        if isinstance(self.last_dates, pl_Series):
            df_constructor = pl_DataFrame
        else:
            df_constructor = pd.DataFrame
        self._feature_null_cols = self._null_cols_in_feature_values(
            features, nan_is_null=df_constructor is pd.DataFrame
        )
        features_df = df_constructor(features)
        if list(features_df.columns) != self.features:
            features_df = features_df[self.features]
        return features_df

    def _null_cols_in_feature_values(
        self, features: Dict[str, Any], nan_is_null: bool
    ) -> List[str]:
        """Names of computed features containing nulls, in ``self.features`` order.

        Runs once per horizon step, so it checks the raw feature values
        (numpy arrays or native series) before the frame is assembled instead
        of scanning the assembled frame, keeping the per-step null diagnostic
        vectorized. NaN counts as null only when ``nan_is_null`` (the pandas
        backend), matching each backend's null semantics.
        """
        null_names = set()
        for name, vals in features.items():
            if isinstance(vals, np.ndarray):
                kind = vals.dtype.kind
                if kind == "f":
                    has_null = nan_is_null and bool(np.isnan(vals).any())
                elif kind in "Mm":
                    has_null = bool(np.isnat(vals).any())
                elif kind == "O":
                    has_null = bool(pd.isnull(vals).any())
                else:
                    # int/uint/bool arrays can't hold nulls
                    has_null = False
            else:
                try:
                    has_null = bool(
                        nw.from_native(vals, series_only=True).is_null().any()
                    )
                except TypeError:
                    has_null = bool(pd.isnull(np.asarray(vals)).any())
            if has_null:
                null_names.add(name)
        return [c for c in self.features if c in null_names]

    def _get_raw_predictions(self) -> np.ndarray:
        return np.array(self.y_pred).ravel("F")

    def _get_future_ids(self, h: int):
        # native-container boundary — uids/df constructor selected per backend
        if isinstance(self.uids, pl_Series):
            idxs = np.repeat(np.arange(len(self.uids)), h)
            return self.uids.gather(idxs).sort()
        repeated = np.repeat(np.asarray(self.uids), h)
        return pd.Series(repeated, name=self.id_col, dtype=self.uids.dtype)

    def _get_predictions(self) -> DataFrame:
        """Get all the predicted values with their corresponding ids and datestamps."""
        h = len(self.y_pred)
        # native-container boundary — uids/df constructor selected per backend
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

    def _current_step_rows(self, X_df):
        """One row per series from ``X_df`` for the current horizon step (``self._h``)."""
        n_series = len(self.uids)
        h = X_df.shape[0] // n_series
        row_offset = min(self._h, h - 1)
        rows = np.arange(row_offset, X_df.shape[0], h)
        return ufp.drop_index_if_pandas(ufp.take_rows(X_df, rows))

    def _update_partition_assignments(self, X_df):
        """Update partition state bucket assignments from current X_df row.

        Returns the sliced X_row (one row per series for the current step)
        so ``_get_features_for_next_step`` can reuse it without re-slicing.
        Returns ``None`` when there are no partition columns.
        """
        if not getattr(self, "_partition_cols", None):
            return None
        X_row = self._current_step_rows(X_df)
        # Both frames hold one row per series in uid order, so selecting key
        # columns from each and concatenating keeps the context aligned.
        X_row_nw = nw.from_native(X_row, eager_only=True)
        statics_nw = nw.from_native(self.static_features_, eager_only=True)
        x_cols = set(X_row_nw.columns)
        sf_cols = set(statics_nw.columns)
        for key, state in self._pooled_states.items():
            _mode, _group_cols, part_cols = key
            if not part_cols or state.key_cols is None:
                continue
            from_statics = [self.id_col]
            from_x = []
            missing_keys = []
            for col in state.key_cols:
                if col == self.id_col:
                    continue
                if col in x_cols:
                    from_x.append(col)
                elif col in sf_cols:
                    from_statics.append(col)
                else:
                    missing_keys.append(col)
            if missing_keys:
                raise ValueError(
                    f"Partition/group key column(s) {missing_keys} not found "
                    f"in X_df or static_features. Provide these columns in "
                    f"X_df for prediction."
                )
            context_nw = statics_nw.select(from_statics)
            if from_x:
                context_nw = nw.concat(
                    [context_nw, X_row_nw.select(from_x)], how="horizontal"
                )
            state.update_series_bucket_id(context_nw.to_native(), self.id_col)
        return X_row

    def _get_features_for_next_step(self, X_df=None):
        X_row = None
        if X_df is not None:
            X_row = self._update_partition_assignments(X_df)
        # same frame _update_features builds, but with the statics trimmed to
        # model features (_statics_keep) so columns dropped by the
        # features_order_ selection below aren't carried through every step
        features_df = self._compute_features_df()
        new_x = ufp.horizontal_concat([self._statics_keep, features_df])
        # statics were scanned in _predict_setup and computed features while
        # building them in _update_features; assembled here in new_x column
        # order (statics, features, X_row) so the warning matches a full scan
        cols_with_nulls = self._static_null_cols + self._feature_null_cols
        if X_df is not None:
            if X_row is None:
                X_row = self._current_step_rows(X_df)
            if self._xdf_null_cols is None:
                # X_df is fixed for the whole predict; scan it once so the
                # common no-nulls case skips the per-step check entirely
                xdf_nw = nw.from_native(X_df, eager_only=True)
                if xdf_nw.columns:
                    xdf_null_counts = xdf_nw.null_count().row(0)
                    self._xdf_null_cols = [
                        c for c, n in zip(xdf_nw.columns, xdf_null_counts) if n
                    ]
                else:
                    self._xdf_null_cols = []
            if self._xdf_null_cols:
                row_nw = nw.from_native(X_row, eager_only=True).select(
                    self._xdf_null_cols
                )
                row_null_counts = row_nw.null_count().row(0)
                cols_with_nulls += [
                    c for c, n in zip(self._xdf_null_cols, row_null_counts) if n
                ]
            new_x = ufp.horizontal_concat([new_x, X_row])
        if cols_with_nulls:
            warnings.warn(f"Found null values in {', '.join(cols_with_nulls)}.")
        self._h += 1
        if list(new_x.columns) != self.features_order_:
            new_x = new_x[self.features_order_]
        if self.as_numpy:
            new_x = ufp.to_numpy(new_x)
        return new_x

    @contextmanager
    def _backup(self) -> Iterator[None]:
        ga = copy.copy(self.ga)
        lag_tfms = copy.deepcopy(self.transforms)
        # Pooled states are only appended to during prediction, so a cheap
        # structural snapshot (references + shallow container copies) restores
        # them faithfully without deep-copying every aggregate array per model.
        pooled_states = getattr(self, "_pooled_states", {})
        pooled_snaps = {key: state.snapshot() for key, state in pooled_states.items()}
        try:
            yield
        finally:
            self.ga = ga
            self.transforms = lag_tfms
            for key, snap in pooled_snaps.items():
                self._pooled_states[key].restore(snap)

    def _predict_setup(self) -> None:
        # TODO: move to utils
        # native-container boundary — clone/copy method selected per backend
        if isinstance(self.last_dates, pl_Series):
            self.curr_dates = self.last_dates.clone()
        else:
            self.curr_dates = self.last_dates.copy()
        self.test_dates: List[Union[pd.Index, pl_Series]] = []
        self.y_pred = []
        self._h = 0
        # Null-diagnostic state for _get_features_for_next_step. Statics are
        # fixed for the whole predict, so scan them once here (boundary)
        # instead of once per horizon step (hot loop).
        self._feature_null_cols = []
        self._xdf_null_cols = None
        # when all series end at the same timestamp, per-step date features
        # are constant across series and can be computed on a single date
        last_dates = _index_to_series(self.last_dates)
        self._uniform_dates = bool(
            nw.from_native(last_dates, series_only=True).n_unique() == 1
        )
        # _predict_setup runs once per model; the statics-derived state below
        # is fixed for the whole predict, so reuse it across models
        # (TimeSeries.predict clears _static_null_src per call)
        if getattr(self, "_static_null_src", None) is self.static_features_:
            return
        self._static_null_src = self.static_features_
        self._static_null_cols: List[str] = []
        statics_nw = nw.from_native(self.static_features_, eager_only=True)
        if statics_nw.columns:
            static_null_counts = statics_nw.null_count().row(0)
            self._static_null_cols = [
                c for c, n in zip(statics_nw.columns, static_null_counts) if n
            ]
        # cache the statics restricted to actual model features (usually all
        # but the id column, which can itself be a feature when the user lists
        # it in static_features) so the per-step feature assembly doesn't
        # carry columns that the features_order_ selection drops at the end.
        # when no static column is a feature keep the full frame: it provides
        # the row count of new_x (e.g. no computed features at all)
        feature_set = set(self.features_order_)
        static_cols = [c for c in statics_nw.columns if c in feature_set]
        if static_cols and static_cols != statics_nw.columns:
            self._statics_keep = self.static_features_[static_cols]
        else:
            self._statics_keep = self.static_features_

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
                    model_x = new_x
                    # sklearn models consume pandas/numpy; native conversion at the model boundary
                    if isinstance(model, CatBoostRegressor) and isinstance(
                        new_x, pl_DataFrame
                    ):
                        model_x = new_x.to_pandas()
                    predictions = model.predict(model_x)
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
                # Compute dates for all horizons, then stack and flatten.
                # horizons_to_predict is 0-indexed; offset_times(dates, freq, n) gives
                # dates + n*freq, so h + 1 converts 0-indexed h to a 1-step-ahead offset.
                dates_per_horizon = [
                    ufp.offset_times(self.curr_dates, self.freq, h + 1)
                    for h in horizons_to_predict
                ]
                # backend-specific date-matrix build at prediction assembly
                # Stack: each row is a series, each col is a horizon
                dates_matrix = pl.DataFrame(dates_per_horizon).transpose()
                # Flatten row by row: [s0_h0, s0_h1, ..., s1_h0, s1_h1, ...]
                dates = dates_matrix.to_numpy().ravel()
                dates = pl.Series(dates)
            else:
                df_constructor = pd.DataFrame
                # Compute dates for all horizons, then stack and flatten.
                # horizons_to_predict is 0-indexed; offset_times(dates, freq, n) gives
                # dates + n*freq, so h + 1 converts 0-indexed h to a 1-step-ahead offset.
                dates_per_horizon = [
                    ufp.offset_times(self.curr_dates, self.freq, h + 1)
                    for h in horizons_to_predict
                ]
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
        exog_cols = self._get_dynamic_exog_cols(self.features_order_)
        common_exog_cols, horizon_exog_map = self._split_horizon_exog_cols(
            exog_cols, self.horizon_features_
        )
        feature_idx = {c: i for i, c in enumerate(self.features_order_)}
        horizon_feature_indices = {}
        if self.horizon_features_:
            for h in horizons_to_predict:
                h_cols = self._get_cols_for_horizon(
                    h, common_exog_cols, horizon_exog_map, exog_cols
                )
                horizon_feature_indices[h] = np.array(
                    [feature_idx[c] for c in h_cols], dtype=np.int32
                )

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

                    model_x = new_x
                    if self.horizon_features_:
                        if isinstance(new_x, np.ndarray):
                            col_idx = horizon_feature_indices.get(h)
                            if col_idx is not None:
                                model_x = new_x[:, col_idx]
                        else:
                            h_cols = self._get_cols_for_horizon(
                                h, common_exog_cols, horizon_exog_map, exog_cols
                            )
                            model_x = new_x[h_cols]
                    horizon_model = model[h]
                    # sklearn models consume pandas/numpy; native conversion at the model boundary
                    if isinstance(horizon_model, CatBoostRegressor) and isinstance(
                        model_x, pl_DataFrame
                    ):
                        model_x = model_x.to_pandas()
                    preds = horizon_model.predict(model_x)
                    if len(preds) != len(self.uids):
                        raise ValueError(
                            f"Model returned {len(preds)} predictions but expected {len(self.uids)}"
                        )
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
        if ids is not None:
            has_nonlocal = any(mode != "local" for mode, _, _ in self._pooled_states)
            if has_nonlocal:
                raise ValueError(
                    "Cannot use `ids` with global, group, or nonlocal partition lag transforms. "
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
        if X_df is None:
            required_future_cols = set(
                self._get_dynamic_exog_cols(self.features_order_)
            )
            required_future_cols.update(getattr(self, "_partition_cols", set()))
            if required_future_cols:
                raise ValueError(
                    "X_df is required for prediction because future values are needed "
                    "for feature generation or model inputs used during training: "
                    f"{sorted(required_future_cols)}."
                )
        with self._maybe_subset(idxs):
            # invalidate the per-predict statics cache in _predict_setup
            self._static_null_src = None
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
                    warnings.warn(
                        "The following columns were provided through X_df but were considered "
                        f"static during fit and will be ignored: {common}. "
                        "If any of these columns should vary over the forecast horizon, refit "
                        "with static_features=[] or exclude them from static_features.",
                        UserWarning,
                        stacklevel=2,
                    )
                required_future_cols = set(
                    self._get_dynamic_exog_cols(self.features_order_)
                )
                required_future_cols.update(getattr(self, "_partition_cols", set()))
                missing = sorted(required_future_cols - set(dynamics))
                if missing:
                    raise ValueError(
                        "X_df is missing future values required for feature generation or "
                        f"model inputs used during training: {missing}."
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
                drop_cols = [self.id_col, self.time_col, "_start", "_end"] + common
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
                    indptr = np.arange(
                        0, preds_per_series * (len(self.uids) + 1), preds_per_series
                    )
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
        from .data_validation import validate_update_df

        validate_update_df(
            df, self.id_col, self.time_col, self.uids, self.last_dates, self.freq
        )

    def update(self, df: DataFrame, validate_new_data: bool = False) -> None:
        """Update the values of the stored series.

        Args:
            df: New observations to append.
            validate_new_data: If True, validate continuity, start dates, and frequency.
        """
        validate_format(df, self.id_col, self.time_col, self.target_col)
        uids = _index_to_series(self.uids)
        uids, new_ids = ufp.match_if_categorical(uids, df[self.id_col])
        df = ufp.copy_if_pandas(df, deep=False)
        df = ufp.assign_columns(df, self.id_col, new_ids)
        df = ufp.sort(df, by=[self.id_col, self.time_col])
        values = df[self.target_col].to_numpy()
        values = values.astype(self.ga.data.dtype, copy=False)
        self._check_aligned_ends()
        if self._pooled_states:
            uids_nw = nw.from_native(_index_to_series(uids), series_only=True).alias(
                "_uid"
            )
            new_ids_nw = nw.from_native(
                _index_to_series(new_ids), series_only=True
            ).alias("_uid")
            expected_count = nw.concat(
                [uids_nw.to_frame(), new_ids_nw.to_frame()], how="vertical"
            )["_uid"].n_unique()
            counts = (
                nw.from_native(df, eager_only=True)
                .group_by(self.time_col)
                .agg(nw.col(self.id_col).n_unique().alias("_n_ids"))
            )
            if counts.filter(nw.col("_n_ids") != expected_count).shape[0] > 0:
                raise ValueError(
                    "Pooled lag transforms require updates to include all series for each timestamp."
                )
        if validate_new_data:
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
        self.uids = _to_native_index(self.uids, df=df)
        self.last_dates = _to_native_index(self.last_dates, df=df)
        if new_groups.any():
            if self.target_transforms is not None:
                raise ValueError("Can not update target_transforms with new series.")
            new_ids = ufp.filter_with_mask(sizes[self.id_col], new_groups)
            new_ids_df = ufp.filter_with_mask(df, ufp.is_in(df[self.id_col], new_ids))
            new_ids_counts = ufp.counts_by_id(new_ids_df, self.id_col)
            new_statics = ufp.take_rows(
                new_ids_df, new_ids_counts["counts"].to_numpy().cumsum() - 1
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
        for state in self._pooled_states.values():
            state.append_observations(
                df,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
                ga_data_dtype=self.ga.data.dtype,
                static_features=self.static_features_,
            )
