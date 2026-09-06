__all__ = [
    "RollingMean",
    "RollingStd",
    "RollingMin",
    "RollingMax",
    "RollingQuantile",
    "SeasonalRollingMean",
    "SeasonalRollingStd",
    "SeasonalRollingMin",
    "SeasonalRollingMax",
    "SeasonalRollingQuantile",
    "ExpandingMean",
    "ExpandingStd",
    "ExpandingMin",
    "ExpandingMax",
    "ExpandingQuantile",
    "ExponentiallyWeightedMean",
    "LookupLag",
    "Offset",
    "Combine",
]


import copy
import inspect
import re
import warnings
from typing import Callable, List, Optional, Sequence

import coreforecast.lag_transforms as core_tfms
import numpy as np
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray
from sklearn.base import BaseEstimator


def _pascal2camel(pascal_str: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_str).lower()


def _normalize_columns(columns):
    if columns is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)
    if not columns:
        return None
    return list(dict.fromkeys(columns))


# Allowed per-timestamp aggregations for the pooled ``time_agg`` option. Defined
# here (not in pooled.py) so ``_validate_time_agg`` can run at construction time
# without importing pooled.py, which imports this module (avoids a cycle).
_TIME_AGGS = ("sum", "count", "mean", "min", "max")


def _validate_time_agg(time_agg, global_, groupby, *, allow_none=True, scope_exempt=()):
    """Validate a ``time_agg`` value at construction time.

    ``allow_none`` and ``scope_exempt`` encode per-transform policy:
    ``ExponentiallyWeightedMean`` rejects ``None`` (its update rule is
    inherently a per-timestamp bucket-mean pass) and exempts ``"mean"`` from
    the pooled-scope requirement.
    """
    if time_agg is None:
        if allow_none:
            return
        raise ValueError(
            "This transform does not accept time_agg=None; use "
            'time_agg="mean" for its bucket-mean update rule.'
        )
    if time_agg not in _TIME_AGGS:
        allowed = f"one of {_TIME_AGGS}" + (" or None" if allow_none else "")
        raise ValueError(f"time_agg must be {allowed}; got {time_agg!r}.")
    if time_agg not in scope_exempt and not (global_ or groupby):
        raise ValueError(
            "time_agg requires a pooled aggregation scope: set global_=True or "
            "groupby=[...] (optionally combined with partition_by). In local or "
            "partition_by-only mode each (bucket, timestamp) has a single row, so "
            "time_agg would be a no-op."
        )


class _BaseLagTransform(BaseEstimator):
    # Bucket scope and per-timestamp pre-aggregation, redefined as instance
    # attributes by every transform that accepts them. The class-level defaults
    # give a uniform answer for the transforms that don't (Lag) and for the
    # wrappers (Offset/Combine), so the pooled properties are always safe to
    # read.
    global_: bool = False
    groupby: Optional[Sequence[str]] = None
    partition_by: Optional[Sequence[str]] = None
    time_agg: Optional[str] = None

    def _get_init_signature(self):
        return {
            k: v
            for k, v in inspect.signature(self.__class__.__init__).parameters.items()
            if k != "self"
            and v.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

    def _set_core_tfm(self, lag: int) -> "_BaseLagTransform":
        init_args = {k: getattr(self, k) for k in self._get_init_signature()}
        init_args.pop("global_", None)
        init_args.pop("global", None)
        init_args.pop("groupby", None)
        init_args.pop("partition_by", None)
        init_args.pop("time_agg", None)
        # resolved along the class hierarchy, so a subclass keeps its parent's
        # coreforecast counterpart
        for cls in type(self).__mro__:
            core_cls = getattr(core_tfms, cls.__name__, None)
            if core_cls is not None:
                break
        else:
            raise AttributeError(
                f"coreforecast has no transform for {type(self).__name__!r}"
            )
        self._core_tfm = core_cls(lag=lag, **init_args)
        return self

    def _get_name(self, lag: int) -> str:
        init_params = self._get_init_signature()
        prefix = ""
        groupby = getattr(self, "groupby", None)
        partition_by = getattr(self, "partition_by", None)
        if getattr(self, "global_", False):
            prefix = "global_"
        elif groupby:
            group_str = "__".join(groupby)
            prefix = f"groupby_{group_str}_"
        if partition_by:
            part_str = "__".join(partition_by)
            prefix += f"partby_{part_str}_"
        result = f"{prefix}{_pascal2camel(self.__class__.__name__)}_lag{lag}"
        changed_params = [
            f"{name}{getattr(self, name)}"
            for name, arg in init_params.items()
            if arg.default != getattr(self, name)
            and name not in {"global_", "groupby", "partition_by"}
        ]
        if changed_params:
            result += "_" + "_".join(changed_params)
        return result

    @property
    def _gb_cols(self) -> List[str]:
        return list(self.groupby or [])

    @property
    def _pt_cols(self) -> List[str]:
        return list(self.partition_by or [])

    @property
    def _is_pooled(self) -> bool:
        return bool(self.global_ or self._gb_cols or self._pt_cols)

    @property
    def _pooled_mode(self) -> str:
        if self._pt_cols:
            return "local" if not (self.global_ or self._gb_cols) else "nonlocal"
        return "global" if self.global_ else "groupby"

    @property
    def _pooled_key(self):
        """Leaves sharing this key share one `PooledState`.

        `time_agg` is deliberately absent: it changes how a bucket's rows are
        summarised, not which rows are in it, so it is a cached view over the
        same store rather than a separate one.
        """
        return (self._pooled_mode, tuple(self._gb_cols), tuple(self._pt_cols))

    def _pooled_leaves(self) -> List["_BaseLagTransform"]:
        """The pooled transforms this one is built from (itself, if pooled)."""
        return [self] if self._is_pooled else []

    def _pooled_eval(self, eval_leaf):
        """Evaluate by resolving pooled leaves.

        Only called on trees with at least one pooled leaf, and `Combine`
        requires both children to share a bucket scope, so every leaf reached
        here is pooled.
        """
        return eval_leaf(self)

    def _get_configured_lag(self) -> int:
        return self._core_tfm.lag

    def transform(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.transform(ga)

    def update(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.update(ga)

    def take(self, idxs: np.ndarray) -> "_BaseLagTransform":
        out = copy.deepcopy(self)
        if self._is_pooled:
            # pooled state is keyed by bucket, lives in `_pooled_inner` and the
            # shared `PooledState`, and stays whole under a subset forecast;
            # only `lag` is ever read off `_core_tfm`
            return out
        out._core_tfm = self._core_tfm.take(idxs)
        return out

    @staticmethod
    def stack(transforms: Sequence["_BaseLagTransform"]) -> "_BaseLagTransform":
        out = copy.deepcopy(transforms[0])
        out._core_tfm = transforms[0]._core_tfm.stack(
            [tfm._core_tfm for tfm in transforms]
        )
        return out

    @property
    def _lag(self):
        return self._core_tfm.lag - 1

    @property
    def update_samples(self) -> int:
        return -1

    @property
    def _pooled_retention(self) -> Optional[int]:
        """Trailing calendar columns ``update`` still reads after priming.

        A pooled state may be trimmed under ``keep_last_n`` only if *every* one
        of its transforms declares a finite retention; ``None`` means unbounded
        and blocks the trim. Transforms carrying an accumulator (Expanding*,
        EWM) need only a short tail, because the prefix they dropped is already
        folded into that accumulator; the ones that re-gather from ordinal 0
        (``ExpandingQuantile``, ``LookupLag``) need everything.

        Distinct from ``update_samples``, which sizes the ``self.ga`` trim and is
        wrong here: it reports 1 for Expanding*/EWM, which read ``lag`` columns.

        Defaults to ``None`` so an unknown/custom transform is never silently
        trimmed (correctness over the perf win).
        """
        return None


class Lag(_BaseLagTransform):
    def __init__(self, lag: int):
        self.lag = lag
        self._core_tfm = core_tfms.Lag(lag=lag)

    def _set_core_tfm(self, _lag: int) -> "Lag":
        return self

    def _get_name(self, lag: int) -> str:
        return f"lag{lag}"

    def __eq__(self, other):
        return isinstance(other, Lag) and self.lag == other.lag

    @property
    def update_samples(self) -> int:
        return self.lag

    @property
    def _pooled_retention(self) -> Optional[int]:
        return self.lag


class LookupLag(_BaseLagTransform):
    """Look up the target from a previous matching occurrence.

    The lag value is provided by the ``lag_transforms`` dictionary key. For
    example, ``lag_transforms={1: [LookupLag(partition_by=["holiday_name"])]}``
    returns the previous target value observed within each
    ``(unique_id, holiday_name)`` bucket.

    ``partition_by`` is required: it defines the matching buckets and is what
    makes this a lookup rather than a plain :class:`Lag`. Like other pooled
    transforms, the partition columns may vary over time and must be supplied
    via ``X_df`` at prediction.

    Args:
        partition_by (Sequence[str]): Dynamic column names used to define the
            matching buckets within each series. Required.
    """

    def __init__(
        self,
        partition_by: Optional[Sequence[str]] = None,
    ):
        self.partition_by = _normalize_columns(partition_by)
        if self.partition_by is None:
            raise ValueError(
                "LookupLag requires `partition_by`; it defines the buckets "
                "used for the occurrence lookup."
            )
        self._core_tfm = None

    def _set_core_tfm(self, lag: int) -> "LookupLag":
        self._core_tfm = core_tfms.Lag(lag=lag)
        return self

    def _get_name(self, lag: int) -> str:
        prefix = ""
        if self.partition_by:
            part_str = "__".join(self.partition_by)
            prefix = f"partby_{part_str}_"
        return f"{prefix}lookup_lag{lag}"

    @property
    def update_samples(self) -> int:
        if self._core_tfm is None:
            return -1
        # LookupLag's pooled state is never trimmed under ``keep_last_n`` (it is
        # not finite-window; see ``_pooled_retention``), so it keeps full bucket
        # history at predict. This value only feeds the ``self.ga`` keep_last_n
        # inference and the regular-``ga`` core-``Lag`` output it governs -- which
        # the pooled result overwrites; pooled trimming ignores it. ``lag`` is the
        # minimal safe value.
        return self._core_tfm.lag

    @property
    def _pooled_retention(self) -> Optional[int]:
        # A matching occurrence can be arbitrarily far back, so LookupLag needs
        # unbounded history; its pooled state must never be trimmed.
        return None


class _RollingBase(_BaseLagTransform):
    "Rolling statistic"

    def __init__(
        self,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        time_agg: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            window_size (int): Number of samples in the window.
            min_samples (int, optional): Minimum samples required to output the statistic.
                If `None`, will be set to `window_size`, except in local partition mode
                (``partition_by`` without ``global_``/``groupby``), where it will be set
                to 1. Defaults to None.
                In local (per-series) mode, ``min_samples`` is capped at ``window_size``
                by coreforecast.  In pooled mode (``global_=True``, ``groupby`` or
                ``partition_by``), ``min_samples`` counts total non-NaN observations
                across **all series** in the bucket within the rolling window, with no
                capping.  For example, ``RollingMean(window_size=1, min_samples=2,
                groupby=["brand"])`` produces a non-null result at timestamps where at
                least 2 series in the brand group contribute observations.
                With ``partition_by``, the window spans ``window_size`` parent-calendar
                steps while only same-partition observations count toward
                ``min_samples``, so requiring a full window is rarely attainable in
                local partition mode; its default of 1 matches SQL RANGE-window
                semantics (NULL only for empty windows). When ``partition_by`` is
                combined with ``global_`` or ``groupby``, the default remains
                ``window_size``, counted across all series in the (group, partition)
                bucket.
                When ``time_agg`` is set, ``min_samples`` instead counts observed
                **timestamps** in the window (each contributes at most one aggregated
                value), not rows.
            global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
                Requires all series to end at the same timestamp. Defaults to False.
            groupby (Sequence[str], optional): Column names to group by before computing the statistic.
                Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
            partition_by (Sequence[str], optional): Column names to partition by.
                Each unique combination of partition values creates a separate bucket.
                Unlike ``groupby``, partition columns may vary over time and must be
                supplied via ``X_df`` at prediction. Composes with ``global_`` (cross-series
                aggregates within each partition), ``groupby`` (group aggregates within each
                partition), or stands alone (per-(id, partition) buckets, *local* mode).
                See the Pooled lag transforms guide for details. Defaults to None.
            time_agg (str, optional): Pre-aggregate all rows sharing a timestamp within
                each bucket into a single value before applying the transform, e.g.
                ``RollingMean(window_size=7, groupby=["category"], time_agg="sum")`` is a
                rolling mean of the category's daily sums. One of ``"sum"``, ``"count"``,
                ``"mean"``, ``"min"``, ``"max"``. Requires ``global_`` or ``groupby``
                (raises ``ValueError`` otherwise, since local/partition-only modes have a
                single row per (bucket, timestamp) and the aggregation would be an
                identity). Defaults to None, which treats each row as an individual
                pooled sample.
        """
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if "partition_by" in kwargs:
            partition_by = kwargs.pop("partition_by")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.window_size = window_size
        self.min_samples = min_samples
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        self.partition_by = _normalize_columns(partition_by)
        self.time_agg = time_agg
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")
        _validate_time_agg(time_agg, self.global_, self.groupby)
        if (
            min_samples is not None
            and min_samples == 0
            and (self.global_ or self.groupby or self.partition_by)
        ):
            warnings.warn(
                "min_samples=0 with pooled transforms (global_/groupby/partition_by) "
                "produces NaN for timestamps with no observations in the window.",
                stacklevel=2,
            )

    @property
    def update_samples(self) -> int:
        return self._lag + self.window_size

    @property
    def _pooled_retention(self) -> Optional[int]:
        # coreforecast's rolling update reads ``lag-1 + window_size`` trailing
        # values, which is also the floor the pooled ``k`` factor needs
        return self._lag + self.window_size


class RollingMean(_RollingBase): ...


class RollingStd(_RollingBase): ...


class RollingMin(_RollingBase): ...


class RollingMax(_RollingBase): ...


class RollingQuantile(_RollingBase):
    """Rolling quantile.

    Note:
        In pooled modes (``global_``/``groupby``/``partition_by``) this
        transform has no aggregate-cache fast path: it falls back to a
        row-level pass whose cost grows with ``unique timestamps x bucket
        rows`` at fit, and aggregates are rebuilt at every recursive
        prediction step. Can be slow on large panels.
    """

    def __init__(
        self,
        p: float,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        time_agg: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            min_samples=min_samples,
            global_=global_,
            groupby=groupby,
            partition_by=partition_by,
            time_agg=time_agg,
            **kwargs,
        )
        self.p = p

    def _set_core_tfm(self, lag: int):
        self._core_tfm = core_tfms.RollingQuantile(
            lag=lag,
            p=self.p,
            window_size=self.window_size,
            min_samples=self.min_samples,
        )
        return self


class _Seasonal_RollingBase(_BaseLagTransform):
    """Rolling statistic over seasonal periods"""

    def __init__(
        self,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        time_agg: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            season_length (int): Periodicity of the seasonal period.
            window_size (int): Number of samples in the window.
            min_samples (int, optional): Minimum samples required to output the statistic.
                If `None`, will be set to `window_size`, except in local partition mode
                (``partition_by`` without ``global_``/``groupby``), where it will be set
                to 1. Defaults to None.
                In local (per-series) mode, ``min_samples`` is capped at ``window_size``
                by coreforecast.  In pooled mode (``global_=True``, ``groupby`` or
                ``partition_by``), ``min_samples`` counts total non-NaN observations
                across **all series** in the bucket within the rolling window, with no
                capping.  For example, ``SeasonalRollingMean(season_length=7,
                window_size=1, min_samples=2, groupby=["brand"])`` produces a non-null
                result at the target seasonal timestamp when at least 2 series in the
                brand group contribute observations.
                With ``partition_by``, the window targets ``window_size`` seasonal
                steps of the parent calendar while only same-partition observations
                count toward ``min_samples``, so requiring a full window is rarely
                attainable in local partition mode; its default of 1 matches SQL
                RANGE-window semantics (NULL only for empty windows). When
                ``partition_by`` is combined with ``global_`` or ``groupby``, the
                default remains ``window_size``, counted across all series in the
                (group, partition) bucket.
                When ``time_agg`` is set, ``min_samples`` instead counts observed
                **timestamps** in the window, not rows.
            global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
                Requires all series to end at the same timestamp. Defaults to False.
            groupby (Sequence[str], optional): Column names to group by before computing the statistic.
                Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
            partition_by (Sequence[str], optional): Column names to partition by.
                Each unique combination of partition values creates a separate bucket.
                Unlike ``groupby``, partition columns may vary over time and must be
                supplied via ``X_df`` at prediction. Composes with ``global_`` (cross-series
                aggregates within each partition), ``groupby`` (group aggregates within each
                partition), or stands alone (per-(id, partition) buckets, *local* mode).
                See the Pooled lag transforms guide for details. Defaults to None.
            time_agg (str, optional): Pre-aggregate all rows sharing a timestamp within
                each bucket into a single value before applying the transform. One of
                ``"sum"``, ``"count"``, ``"mean"``, ``"min"``, ``"max"``. Requires
                ``global_`` or ``groupby``. Defaults to None.
        """
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if "partition_by" in kwargs:
            partition_by = kwargs.pop("partition_by")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.season_length = season_length
        self.window_size = window_size
        self.min_samples = min_samples
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        self.partition_by = _normalize_columns(partition_by)
        self.time_agg = time_agg
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")
        _validate_time_agg(time_agg, self.global_, self.groupby)
        if (
            min_samples is not None
            and min_samples == 0
            and (self.global_ or self.groupby or self.partition_by)
        ):
            warnings.warn(
                "min_samples=0 with pooled transforms (global_/groupby/partition_by) "
                "produces NaN for timestamps with no observations in the window.",
                stacklevel=2,
            )

    @property
    def update_samples(self) -> int:
        return self._lag + self.season_length * self.window_size

    @property
    def _pooled_retention(self) -> Optional[int]:
        # the strided update touches ``lag-1 + 1 + (window_size-1)*season_length``
        # trailing values, i.e. exactly the ``window_size`` seasonal cells
        return self._lag + (self.window_size - 1) * self.season_length + 1


class SeasonalRollingMean(_Seasonal_RollingBase): ...


class SeasonalRollingStd(_Seasonal_RollingBase): ...


class SeasonalRollingMin(_Seasonal_RollingBase): ...


class SeasonalRollingMax(_Seasonal_RollingBase): ...


class SeasonalRollingQuantile(_Seasonal_RollingBase):
    """Seasonal rolling quantile.

    Note:
        In pooled modes (``global_``/``groupby``/``partition_by``) this
        transform has no aggregate-cache fast path: a quantile can't be
        recovered from the cached channels, so it falls back to a row-level
        pass whose cost grows with ``unique timestamps x bucket rows``. Can be
        slow on large panels.
    """

    def __init__(
        self,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        time_agg: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
            global_=global_,
            groupby=groupby,
            partition_by=partition_by,
            time_agg=time_agg,
            **kwargs,
        )
        self.p = p


class _ExpandingBase(_BaseLagTransform):
    """Expanding statistic

    Args:
        global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
            Requires all series to end at the same timestamp. Defaults to False.
        groupby (Sequence[str], optional): Column names to group by before computing the statistic.
            Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
        partition_by (Sequence[str], optional): Column names to partition by.
            Each unique combination of partition values creates a separate bucket.
            Unlike ``groupby``, partition columns may vary over time and must be
            supplied via ``X_df`` at prediction. Composes with ``global_`` (cross-series
            aggregates within each partition), ``groupby`` (group aggregates within each
            partition), or stands alone (per-(id, partition) buckets, *local* mode).
            See the Pooled lag transforms guide for details. Defaults to None.
        time_agg (str, optional): Pre-aggregate all rows sharing a timestamp within each
            bucket into a single value before applying the transform. One of ``"sum"``,
            ``"count"``, ``"mean"``, ``"min"``, ``"max"``. Requires ``global_`` or
            ``groupby``. Defaults to None.
    """

    def __init__(
        self,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        time_agg: Optional[str] = None,
        **kwargs,
    ):
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if "partition_by" in kwargs:
            partition_by = kwargs.pop("partition_by")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        self.partition_by = _normalize_columns(partition_by)
        self.time_agg = time_agg
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")
        _validate_time_agg(time_agg, self.global_, self.groupby)

    @property
    def update_samples(self) -> int:
        return 1

    @property
    def _pooled_retention(self) -> Optional[int]:
        # The inner coreforecast transform carries the running accumulator in
        # ``stats_`` and its update reads a single value at ``lag`` from the end,
        # while ``window_cells`` derives ``k`` from the absolute ordinal, so the
        # dropped prefix stays fully represented.
        return self._lag + 1


class ExpandingMean(_ExpandingBase): ...


class ExpandingStd(_ExpandingBase): ...


class ExpandingMin(_ExpandingBase): ...


class ExpandingMax(_ExpandingBase): ...


class ExpandingQuantile(_ExpandingBase):
    """Expanding quantile.

    Note:
        In pooled modes (``global_``/``groupby``/``partition_by``) this
        transform has no aggregate-cache fast path: it falls back to a
        row-level pass whose cost grows with ``unique timestamps x bucket
        rows`` at fit, and aggregates are rebuilt at every recursive
        prediction step. Can be slow on large panels.
    """

    def __init__(
        self,
        p: float,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        time_agg: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            global_=global_,
            groupby=groupby,
            partition_by=partition_by,
            time_agg=time_agg,
            **kwargs,
        )
        self.p = p

    @property
    def update_samples(self) -> int:
        return -1

    @property
    def _pooled_retention(self) -> Optional[int]:
        # unlike its siblings this is a row kernel: it re-gathers every
        # observation from ordinal 0, so nothing may be dropped
        return None


class ExponentiallyWeightedMean(_BaseLagTransform):
    """Exponentially weighted average

    Args:
        alpha (float): Smoothing factor.
        global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
            Requires all series to end at the same timestamp. Defaults to False.
        groupby (Sequence[str], optional): Column names to group by before computing the statistic.
            Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
        partition_by (Sequence[str], optional): Column names to partition by.
            Each unique combination of partition values creates a separate bucket.
            Unlike ``groupby``, partition columns may vary over time and must be
            supplied via ``X_df`` at prediction. Composes with ``global_`` (cross-series
            aggregates within each partition), ``groupby`` (group aggregates within each
            partition), or stands alone (per-(id, partition) buckets, *local* mode).
            See the Pooled lag transforms guide for details. Defaults to None.
        time_agg (str): Pre-aggregate all rows sharing a timestamp within each
            bucket into a single value before applying the transform. One of ``"sum"``,
            ``"count"``, ``"mean"``, ``"min"``, ``"max"``. Values other than
            ``"mean"`` require ``global_`` or ``groupby``. Defaults to ``"mean"``,
            which matches EWM's bucket-mean update rule: each timestamp contributes
            its bucket aggregate mean exactly once, regardless of how many rows
            aggregated there. ``None`` is not accepted.
    """

    def __init__(
        self,
        alpha: float,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        partition_by: Optional[Sequence[str]] = None,
        time_agg: str = "mean",
        **kwargs,
    ):
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if "partition_by" in kwargs:
            partition_by = kwargs.pop("partition_by")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.alpha = alpha
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        self.partition_by = _normalize_columns(partition_by)
        self.time_agg = time_agg
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")
        _validate_time_agg(
            time_agg,
            self.global_,
            self.groupby,
            allow_none=False,
            scope_exempt=("mean",),
        )
        if self.partition_by:
            warnings.warn(
                "Partitioned EWM skips timestamps where the partition bucket "
                "has no observations and applies decay only across observed "
                "bucket aggregates. Each observed timestamp contributes its "
                "aggregate mean once, regardless of how many rows were "
                "aggregated at that timestamp.",
                stacklevel=2,
            )

    @property
    def update_samples(self) -> int:
        return 1

    @property
    def _pooled_retention(self) -> Optional[int]:
        # ``EwmK`` carries the running mean and the calendar cursor, folding one
        # new cell per step; the oldest column it reads is ``lag`` from the end.
        return self._lag + 1


class Offset(_BaseLagTransform):
    """Shift series before computing transformation

    Args:
        tfm (LagTransform): Transformation to be applied
        n (int): Number of positions to shift (lag) series before applying the transformation
    """

    def __init__(self, tfm: _BaseLagTransform, n: int):
        self.tfm = tfm
        self.n = n
        self.global_ = getattr(tfm, "global_", False)
        self.groupby = getattr(tfm, "groupby", None)
        self.partition_by = getattr(tfm, "partition_by", None)
        # time_agg is intentionally not mirrored (unlike the mode attributes
        # above, nothing reads it on the wrapper): the delegated hooks apply
        # the inner transform's own re-aggregation.

    def _get_name(self, lag: int) -> str:
        return self.tfm._get_name(lag + self.n)

    def _set_core_tfm(self, lag: int) -> "Offset":
        if lag + self.n < 1:
            raise ValueError(
                f"Offset(n={self.n}) applied to lag {lag} produces an "
                f"effective lag of {lag + self.n}; the effective lag must be "
                "at least 1."
            )
        self.tfm = copy.deepcopy(self.tfm)._set_core_tfm(lag + self.n)
        self._core_tfm = self.tfm._core_tfm
        return self

    def _get_configured_lag(self) -> int:
        return self.tfm._get_configured_lag() - self.n

    def _pooled_leaves(self):
        return self.tfm._pooled_leaves()

    def _pooled_eval(self, eval_leaf):
        # `_set_core_tfm` already primed the inner transform at `lag + n`, so
        # the offset is baked into its core transform
        return self.tfm._pooled_eval(eval_leaf)

    @property
    def update_samples(self) -> int:
        return self.tfm.update_samples + self.n

    @property
    def _pooled_retention(self) -> Optional[int]:
        # no ``+ self.n``: ``_set_core_tfm`` already baked the offset into the
        # inner transform's lag
        return self.tfm._pooled_retention


class Combine(_BaseLagTransform):
    """Combine two lag transformations using an operator

    Args:
        tfm1 (LagTransform): First transformation.
        tfm2 (LagTransform): Second transformation.
        operator (callable): Binary operator that defines how to combine the two transformations.
    """

    def __init__(
        self, tfm1: _BaseLagTransform, tfm2: _BaseLagTransform, operator: Callable
    ):
        self.tfm1 = tfm1
        self.tfm2 = tfm2
        self.operator = operator
        global_1 = getattr(tfm1, "global_", False)
        global_2 = getattr(tfm2, "global_", False)
        groupby_1 = getattr(tfm1, "groupby", None)
        groupby_2 = getattr(tfm2, "groupby", None)
        if global_1 != global_2:
            raise ValueError(
                "Can't combine transforms with different global_ settings."
            )
        if (groupby_1 or groupby_2) and groupby_1 != groupby_2:
            raise ValueError(
                "Can't combine transforms with different groupby settings."
            )
        self.global_ = global_1
        self.groupby = groupby_1
        partition_by_1 = getattr(tfm1, "partition_by", None)
        partition_by_2 = getattr(tfm2, "partition_by", None)
        if (partition_by_1 or partition_by_2) and partition_by_1 != partition_by_2:
            raise ValueError(
                "Can't combine transforms with different partition_by settings."
            )
        self.partition_by = partition_by_1
        # time_agg needs no reconciliation: it doesn't affect the pooled mode key,
        # and each inner transform applies its own re-aggregation at hook entry, so
        # mixing (e.g. rolling mean of sums / rolling mean of means) is intentional.

    def _set_core_tfm(self, lag: int) -> "Combine":
        self.tfm1 = copy.deepcopy(self.tfm1)._set_core_tfm(lag)
        self.tfm2 = copy.deepcopy(self.tfm2)._set_core_tfm(lag)
        return self

    def _get_name(self, lag: int) -> str:
        lag1 = getattr(self.tfm1, "lag", lag)
        lag2 = getattr(self.tfm2, "lag", lag)
        return f"{self.tfm1._get_name(lag1)}_{self.operator.__name__}_{self.tfm2._get_name(lag2)}"

    def transform(self, ga: CoreGroupedArray) -> np.ndarray:
        return self.operator(self.tfm1.transform(ga), self.tfm2.transform(ga))

    def update(self, ga: CoreGroupedArray) -> np.ndarray:
        return self.operator(self.tfm1.update(ga), self.tfm2.update(ga))

    @property
    def update_samples(self):
        return max(self.tfm1.update_samples, self.tfm2.update_samples)

    @property
    def _pooled_retention(self) -> Optional[int]:
        r1 = self.tfm1._pooled_retention
        r2 = self.tfm2._pooled_retention
        if r1 is None or r2 is None:
            return None
        return max(r1, r2)

    def _get_configured_lag(self) -> int:
        lag1 = self.tfm1._get_configured_lag()
        lag2 = self.tfm2._get_configured_lag()
        if lag1 != lag2:
            raise ValueError("Combined transforms must share the same configured lag.")
        return lag1

    def _pooled_leaves(self):
        return self.tfm1._pooled_leaves() + self.tfm2._pooled_leaves()

    def _pooled_eval(self, eval_leaf):
        return self.operator(
            self.tfm1._pooled_eval(eval_leaf),
            self.tfm2._pooled_eval(eval_leaf),
        )

    def take(self, idxs: np.ndarray) -> "Combine":
        out = copy.deepcopy(self)
        out.tfm1 = self.tfm1.take(idxs)
        out.tfm2 = self.tfm2.take(idxs)
        return out

    @staticmethod
    def stack(transforms: Sequence["Combine"]) -> "Combine":
        out = copy.copy(transforms[0])
        out.tfm1 = transforms[0].tfm1.stack([tfm.tfm1 for tfm in transforms])
        out.tfm2 = transforms[0].tfm2.stack([tfm.tfm2 for tfm in transforms])
        return out
