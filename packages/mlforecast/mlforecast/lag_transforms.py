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
    "Offset",
    "Combine",
]


import copy
import inspect
import re
import warnings
from typing import Callable, Dict, Optional, Sequence

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


def _build_sparse_table(arr, op):
    n = len(arr)
    if n == 0:
        return np.empty((1, 0))
    K = max(1, int(np.log2(n)) + 1)
    table = np.full((K, n), np.nan)
    table[0] = arr
    for j in range(1, K):
        step = 1 << (j - 1)
        valid = n - (1 << j) + 1
        if valid <= 0:
            break
        table[j, :valid] = op(table[j-1, :valid], table[j-1, step:step+valid])
    return table


def _query_sparse_table(table, lefts, rights, op):
    lengths = rights - lefts + 1
    valid = lengths > 0
    k = np.zeros_like(lengths)
    k[valid] = np.floor(np.log2(lengths[valid])).astype(int)
    result = np.full(len(lefts), np.nan)
    v = valid & (lefts >= 0)
    if v.any():
        left_vals = table[k[v], lefts[v]]
        right_idx = rights[v] - (1 << k[v]) + 1
        right_vals = table[k[v], right_idx]
        result[v] = op(left_vals, right_vals)
    return result


class _BaseLagTransform(BaseEstimator):
    def _get_init_signature(self):
        return {
            k: v
            for k, v in inspect.signature(self.__class__.__init__).parameters.items()
            if k != "self" and v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

    def _set_core_tfm(self, lag: int) -> "_BaseLagTransform":
        init_args = {k: getattr(self, k) for k in self._get_init_signature()}
        init_args.pop("global_", None)
        init_args.pop("global", None)
        init_args.pop("groupby", None)
        self._core_tfm = getattr(core_tfms, self.__class__.__name__)(
            lag=lag, **init_args
        )
        return self

    def _get_name(self, lag: int) -> str:
        init_params = self._get_init_signature()
        prefix = ""
        groupby = getattr(self, "groupby", None)
        if getattr(self, "global_", False):
            prefix = "global_"
        elif groupby:
            group_str = "__".join(groupby)
            prefix = f"groupby_{group_str}_"
        result = f"{prefix}{_pascal2camel(self.__class__.__name__)}_lag{lag}"
        changed_params = [
            f"{name}{getattr(self, name)}"
            for name, arg in init_params.items()
            if arg.default != getattr(self, name)
            and name not in {"global_", "groupby"}
        ]
        if changed_params:
            result += "_" + "_".join(changed_params)
        return result

    def _compute_bucket_feature(
        self,
        _bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        _y_arr: np.ndarray,
        _ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> Optional[np.ndarray]:
        return None

    def _compute_latest_from_aggs(
        self, _ts_aggs, _target_ords: Dict[int, int],
    ) -> Optional[Dict[int, float]]:
        """Compute feature value at the target timestamp per bucket from cached aggregates.

        ``_target_ords`` maps bucket_id to the time-index ordinal at which to
        evaluate the statistic (typically ``next_time_index_by_bucket``).
        Returns None if this transform doesn't support the fast path.
        """
        return None

    def _compute_ts_level_from_aggs(self, _ts_aggs):
        return None

    def _get_configured_lag(self) -> int:
        return self._core_tfm.lag

    def transform(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.transform(ga)

    def update(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.update(ga)

    def take(self, idxs: np.ndarray) -> "_BaseLagTransform":
        out = copy.deepcopy(self)
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


class _RollingBase(_BaseLagTransform):
    "Rolling statistic"

    def __init__(
        self,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """
        Args:
            window_size (int): Number of samples in the window.
            min_samples (int, optional): Minimum samples required to output the statistic.
                If `None`, will be set to `window_size`. Defaults to None.
                In local (per-series) mode, ``min_samples`` is capped at ``window_size``
                by coreforecast.  In pooled mode (``global_=True`` or ``groupby``),
                ``min_samples`` counts total non-NaN observations across **all series**
                in the bucket within the rolling window, with no capping.  For example,
                ``RollingMean(window_size=1, min_samples=2, groupby=["brand"])`` produces
                a non-null result at timestamps where at least 2 series in the brand
                group contribute observations.
            global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
                Requires all series to end at the same timestamp. Defaults to False.
            groupby (Sequence[str], optional): Column names to group by before computing the statistic.
                Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
        """
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.window_size = window_size
        self.min_samples = min_samples
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")
        if min_samples is not None and min_samples == 0 and (self.global_ or self.groupby):
            warnings.warn(
                "min_samples=0 with pooled transforms (global_/groupby) "
                "produces NaN for timestamps with no observations in the window.",
                stacklevel=2,
            )

    @property
    def update_samples(self) -> int:
        return self._lag + self.window_size

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        # TODO: use _ts_aggs fast path (needs per-timestamp aggregates
        # specific to each subclass stat: sum_sq for Std, min/max for Min/Max)
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid in np.unique(bid_arr):
            idxs = np.where(bid_arr == bid)[0]
            ord_b = ord_arr[idxs]
            y_b = y_arr[idxs]
            unique_ord = np.unique(ord_b)
            inv = np.searchsorted(unique_ord, ord_b)
            feat_u = np.full(len(unique_ord), np.nan)
            for k, o in enumerate(unique_ord):
                upper = o - lag
                lower = o - lag - w + 1
                mask = (ord_b >= lower) & (ord_b <= upper)
                vals = y_b[mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) >= min_samples and len(vals) > 0:
                    feat_u[k] = self._window_stat(vals)
            result[idxs] = feat_u[inv]
        return result

    def _window_stat(self, vals: np.ndarray) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `_window_stat`."
        )


def _rolling_mean_from_agg(agg, lag, window_size, min_samples):
    cum_sum = np.cumsum(agg.sums)
    cum_cnt = np.cumsum(agg.counts)
    upper_ord = agg.unique_times - lag
    lower_ord = agg.unique_times - lag - window_size
    upper_idxs = np.searchsorted(agg.unique_times, upper_ord, side="right") - 1
    lower_idxs = np.searchsorted(agg.unique_times, lower_ord, side="right") - 1
    upper_sum = np.where(upper_idxs >= 0, cum_sum[upper_idxs], 0.0)
    upper_cnt = np.where(upper_idxs >= 0, cum_cnt[upper_idxs], 0.0)
    lower_sum = np.where(lower_idxs >= 0, cum_sum[lower_idxs], 0.0)
    lower_cnt = np.where(lower_idxs >= 0, cum_cnt[lower_idxs], 0.0)
    win_sum = upper_sum - lower_sum
    win_cnt = upper_cnt - lower_cnt
    safe_cnt = np.where(win_cnt > 0, win_cnt, 1.0)
    return np.where((win_cnt >= min_samples) & (win_cnt > 0), win_sum / safe_cnt, np.nan)


class RollingMean(_RollingBase):
    def _compute_latest_from_aggs(
        self, ts_aggs, target_ords: Dict[int, int],
    ) -> Optional[Dict[int, float]]:
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            cum_sum = np.cumsum(agg.sums)
            cum_cnt = np.cumsum(agg.counts)
            t = target_ords[bid]
            upper = t - lag
            lower = t - lag - w
            ui = int(np.searchsorted(agg.unique_times, upper, side="right")) - 1
            li = int(np.searchsorted(agg.unique_times, lower, side="right")) - 1
            s = (cum_sum[ui] if ui >= 0 else 0.0) - (cum_sum[li] if li >= 0 else 0.0)
            c = (cum_cnt[ui] if ui >= 0 else 0.0) - (cum_cnt[li] if li >= 0 else 0.0)
            result[bid] = s / c if c >= min_samples and c > 0 else float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        return {
            bid: _rolling_mean_from_agg(agg, lag, w, min_samples)
            for bid, agg in ts_aggs.items()
        }

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        return self._compute_row_level(bid_arr, y_arr, ord_arr)

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _rolling_mean_from_agg(agg, lag, w, min_samples)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_row_level(self, bid_arr, y_arr, ord_arr):
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid in np.unique(bid_arr):
            idxs = np.where(bid_arr == bid)[0]
            ord_b = ord_arr[idxs]
            y = y_arr[idxs]
            valid = ~np.isnan(y)
            y_clean = np.where(valid, y, 0.0)
            unique_ord, inv = np.unique(ord_b, return_inverse=True)
            m = len(unique_ord)
            ord_sums = np.bincount(inv, weights=y_clean, minlength=m)
            valid_counts = np.bincount(inv, weights=valid.astype(float), minlength=m)
            cum_sum = np.cumsum(ord_sums)
            cum_cnt = np.cumsum(valid_counts)
            upper_ord = unique_ord - lag
            lower_ord = unique_ord - lag - w
            upper_idxs = np.searchsorted(unique_ord, upper_ord, side="right") - 1
            lower_idxs = np.searchsorted(unique_ord, lower_ord, side="right") - 1
            upper_sum = np.where(upper_idxs >= 0, cum_sum[upper_idxs], 0.0)
            upper_cnt = np.where(upper_idxs >= 0, cum_cnt[upper_idxs], 0.0)
            lower_sum = np.where(lower_idxs >= 0, cum_sum[lower_idxs], 0.0)
            lower_cnt = np.where(lower_idxs >= 0, cum_cnt[lower_idxs], 0.0)
            win_sum = upper_sum - lower_sum
            win_cnt = upper_cnt - lower_cnt
            safe_cnt = np.where(win_cnt > 0, win_cnt, 1.0)
            feat_u = np.where((win_cnt >= min_samples) & (win_cnt > 0), win_sum / safe_cnt, np.nan)
            result[idxs] = feat_u[inv]
        return result


def _rolling_std_from_agg(agg, lag, window_size, min_samples):
    cum_sum = np.cumsum(agg.sums)
    cum_cnt = np.cumsum(agg.counts)
    cum_sum_sq = np.cumsum(agg.sum_sq)
    upper_ord = agg.unique_times - lag
    lower_ord = agg.unique_times - lag - window_size
    upper_idxs = np.searchsorted(agg.unique_times, upper_ord, side="right") - 1
    lower_idxs = np.searchsorted(agg.unique_times, lower_ord, side="right") - 1
    upper_s = np.where(upper_idxs >= 0, cum_sum[upper_idxs], 0.0)
    lower_s = np.where(lower_idxs >= 0, cum_sum[lower_idxs], 0.0)
    upper_sq = np.where(upper_idxs >= 0, cum_sum_sq[upper_idxs], 0.0)
    lower_sq = np.where(lower_idxs >= 0, cum_sum_sq[lower_idxs], 0.0)
    upper_c = np.where(upper_idxs >= 0, cum_cnt[upper_idxs], 0.0)
    lower_c = np.where(lower_idxs >= 0, cum_cnt[lower_idxs], 0.0)
    win_sum = upper_s - lower_s
    win_sq = upper_sq - lower_sq
    win_cnt = upper_c - lower_c
    safe_n = np.where(win_cnt > 0, win_cnt, 1.0)
    den = np.where(win_cnt > 1, win_cnt - 1, 1.0)
    var = (win_sq - win_sum**2 / safe_n) / den
    var = np.maximum(var, 0.0)
    return np.where((win_cnt >= min_samples) & (win_cnt > 1), np.sqrt(var), np.nan)


class RollingStd(_RollingBase):
    def _window_stat(self, vals: np.ndarray) -> float:
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        return super()._compute_bucket_feature(bid_arr, _ts_arr, y_arr, ord_arr)

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _rolling_std_from_agg(agg, lag, w, min_samples)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_latest_from_aggs(
        self, ts_aggs, target_ords: Dict[int, int],
    ) -> Optional[Dict[int, float]]:
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            cum_sum = np.cumsum(agg.sums)
            cum_cnt = np.cumsum(agg.counts)
            cum_sum_sq = np.cumsum(agg.sum_sq)
            t = target_ords[bid]
            upper = t - lag
            lower = t - lag - w
            ui = int(np.searchsorted(agg.unique_times, upper, side="right")) - 1
            li = int(np.searchsorted(agg.unique_times, lower, side="right")) - 1
            s = (cum_sum[ui] if ui >= 0 else 0.0) - (cum_sum[li] if li >= 0 else 0.0)
            sq = (cum_sum_sq[ui] if ui >= 0 else 0.0) - (cum_sum_sq[li] if li >= 0 else 0.0)
            c = (cum_cnt[ui] if ui >= 0 else 0.0) - (cum_cnt[li] if li >= 0 else 0.0)
            if c >= min_samples and c > 1:
                var = (sq - s**2 / c) / (c - 1)
                var = max(var, 0.0)
                result[bid] = float(np.sqrt(var))
            else:
                result[bid] = float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        return {
            bid: _rolling_std_from_agg(agg, lag, w, min_samples)
            for bid, agg in ts_aggs.items()
        }


def _rolling_min_from_agg(agg, lag, window_size, min_samples):
    sparse = _build_sparse_table(agg.mins, np.fmin)
    cum_cnt = np.cumsum(agg.counts)
    upper_ord = agg.unique_times - lag
    lower_ord = agg.unique_times - lag - window_size + 1
    upper_idxs = np.searchsorted(agg.unique_times, upper_ord, side="right") - 1
    lower_idxs = np.searchsorted(agg.unique_times, lower_ord, side="left")
    upper_cnt = np.where(upper_idxs >= 0, cum_cnt[upper_idxs], 0.0)
    lower_cnt = np.where(lower_idxs > 0, cum_cnt[lower_idxs - 1], 0.0)
    win_cnt = upper_cnt - lower_cnt
    result = _query_sparse_table(sparse, lower_idxs, upper_idxs, np.fmin)
    return np.where((win_cnt >= min_samples) & (win_cnt > 0), result, np.nan)


def _rolling_max_from_agg(agg, lag, window_size, min_samples):
    sparse = _build_sparse_table(agg.maxs, np.fmax)
    cum_cnt = np.cumsum(agg.counts)
    upper_ord = agg.unique_times - lag
    lower_ord = agg.unique_times - lag - window_size + 1
    upper_idxs = np.searchsorted(agg.unique_times, upper_ord, side="right") - 1
    lower_idxs = np.searchsorted(agg.unique_times, lower_ord, side="left")
    upper_cnt = np.where(upper_idxs >= 0, cum_cnt[upper_idxs], 0.0)
    lower_cnt = np.where(lower_idxs > 0, cum_cnt[lower_idxs - 1], 0.0)
    win_cnt = upper_cnt - lower_cnt
    result = _query_sparse_table(sparse, lower_idxs, upper_idxs, np.fmax)
    return np.where((win_cnt >= min_samples) & (win_cnt > 0), result, np.nan)


class RollingMin(_RollingBase):
    def _window_stat(self, vals: np.ndarray) -> float:
        return float(np.min(vals))

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        return super()._compute_bucket_feature(bid_arr, _ts_arr, y_arr, ord_arr)

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.full(n, np.nan)
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _rolling_min_from_agg(agg, lag, w, min_samples)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_latest_from_aggs(
        self, ts_aggs, target_ords: Dict[int, int],
    ) -> Optional[Dict[int, float]]:
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            sparse = _build_sparse_table(agg.mins, np.fmin)
            cum_cnt = np.cumsum(agg.counts)
            t = target_ords[bid]
            upper = t - lag
            lower = t - lag - w + 1
            ui = int(np.searchsorted(agg.unique_times, upper, side="right")) - 1
            li = int(np.searchsorted(agg.unique_times, lower, side="left"))
            c = (cum_cnt[ui] if ui >= 0 else 0.0) - (cum_cnt[li - 1] if li > 0 else 0.0)
            if c >= min_samples and c > 0:
                val = _query_sparse_table(sparse, np.array([li]), np.array([ui]), np.fmin)
                result[bid] = float(val[0])
            else:
                result[bid] = float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        return {
            bid: _rolling_min_from_agg(agg, lag, w, min_samples)
            for bid, agg in ts_aggs.items()
        }


class RollingMax(_RollingBase):
    def _window_stat(self, vals: np.ndarray) -> float:
        return float(np.max(vals))

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        return super()._compute_bucket_feature(bid_arr, _ts_arr, y_arr, ord_arr)

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.full(n, np.nan)
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _rolling_max_from_agg(agg, lag, w, min_samples)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_latest_from_aggs(
        self, ts_aggs, target_ords: Dict[int, int],
    ) -> Optional[Dict[int, float]]:
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            sparse = _build_sparse_table(agg.maxs, np.fmax)
            cum_cnt = np.cumsum(agg.counts)
            t = target_ords[bid]
            upper = t - lag
            lower = t - lag - w + 1
            ui = int(np.searchsorted(agg.unique_times, upper, side="right")) - 1
            li = int(np.searchsorted(agg.unique_times, lower, side="left"))
            c = (cum_cnt[ui] if ui >= 0 else 0.0) - (cum_cnt[li - 1] if li > 0 else 0.0)
            if c >= min_samples and c > 0:
                val = _query_sparse_table(sparse, np.array([li]), np.array([ui]), np.fmax)
                result[bid] = float(val[0])
            else:
                result[bid] = float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        return {
            bid: _rolling_max_from_agg(agg, lag, w, min_samples)
            for bid, agg in ts_aggs.items()
        }


class RollingQuantile(_RollingBase):
    def __init__(
        self,
        p: float,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            min_samples=min_samples,
            global_=global_,
            groupby=groupby,
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

    def _window_stat(self, vals: np.ndarray) -> float:
        return float(np.quantile(vals, self.p))


class _Seasonal_RollingBase(_BaseLagTransform):
    """Rolling statistic over seasonal periods"""

    def __init__(
        self,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """
        Args:
            season_length (int): Periodicity of the seasonal period.
            window_size (int): Number of samples in the window.
            min_samples (int, optional): Minimum samples required to output the statistic.
                If `None`, will be set to `window_size`. Defaults to None.
                In local (per-series) mode, ``min_samples`` is capped at ``window_size``
                by coreforecast.  In pooled mode (``global_=True`` or ``groupby``),
                ``min_samples`` counts total non-NaN observations across **all series**
                in the bucket within the rolling window, with no capping.  For example,
                ``SeasonalRollingMean(season_length=7, window_size=1, min_samples=2,
                groupby=["brand"])`` produces a non-null result at the target seasonal
                timestamp when at least 2 series in the brand group contribute
                observations.
            global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
                Requires all series to end at the same timestamp. Defaults to False.
            groupby (Sequence[str], optional): Column names to group by before computing the statistic.
                Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
        """
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.season_length = season_length
        self.window_size = window_size
        self.min_samples = min_samples
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")
        if min_samples is not None and min_samples == 0 and (self.global_ or self.groupby):
            warnings.warn(
                "min_samples=0 with pooled transforms (global_/groupby) "
                "produces NaN for timestamps with no observations in the window.",
                stacklevel=2,
            )

    @property
    def update_samples(self) -> int:
        return self._lag + self.season_length * self.window_size

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        # TODO: use _ts_aggs fast path (needs per-season aggregation)
        lag = self._core_tfm.lag
        sl = self.season_length
        w = self.window_size
        min_samples = self.min_samples if self.min_samples is not None else w
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid in np.unique(bid_arr):
            idxs = np.where(bid_arr == bid)[0]
            ord_b = ord_arr[idxs]
            y_b = y_arr[idxs]
            unique_ord = np.unique(ord_b)
            inv = np.searchsorted(unique_ord, ord_b)
            feat_u = np.full(len(unique_ord), np.nan)
            for k, o in enumerate(unique_ord):
                target_ords = [o - lag - i * sl for i in range(w)]
                target_ords = [t for t in target_ords if t >= 0]
                if not target_ords:
                    continue
                mask = np.isin(ord_b, target_ords)
                vals = y_b[mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) >= min_samples and len(vals) > 0:
                    feat_u[k] = self._seasonal_stat(vals)
            result[idxs] = feat_u[inv]
        return result

    def _seasonal_stat(self, vals: np.ndarray) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `_seasonal_stat`."
        )


class SeasonalRollingMean(_Seasonal_RollingBase):
    def _seasonal_stat(self, vals: np.ndarray) -> float:
        return float(np.mean(vals))


class SeasonalRollingStd(_Seasonal_RollingBase):
    def _seasonal_stat(self, vals: np.ndarray) -> float:
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan


class SeasonalRollingMin(_Seasonal_RollingBase):
    def _seasonal_stat(self, vals: np.ndarray) -> float:
        return float(np.min(vals))


class SeasonalRollingMax(_Seasonal_RollingBase):
    def _seasonal_stat(self, vals: np.ndarray) -> float:
        return float(np.max(vals))


class SeasonalRollingQuantile(_Seasonal_RollingBase):
    def __init__(
        self,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
            global_=global_,
            groupby=groupby,
            **kwargs,
        )
        self.p = p

    def _seasonal_stat(self, vals: np.ndarray) -> float:
        return float(np.quantile(vals, self.p))


class _ExpandingBase(_BaseLagTransform):
    """Expanding statistic

    Args:
        global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
            Requires all series to end at the same timestamp. Defaults to False.
        groupby (Sequence[str], optional): Column names to group by before computing the statistic.
            Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
    """

    def __init__(
        self,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")

    @property
    def update_samples(self) -> int:
        return 1

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        # TODO: use _ts_aggs fast path (cumsum of sums/counts)
        lag = self._core_tfm.lag
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid in np.unique(bid_arr):
            idxs = np.where(bid_arr == bid)[0]
            ord_b = ord_arr[idxs]
            y_b = y_arr[idxs]
            unique_ord = np.unique(ord_b)
            inv = np.searchsorted(unique_ord, ord_b)
            feat_u = np.full(len(unique_ord), np.nan)
            for k, o in enumerate(unique_ord):
                upper = o - lag
                if upper < 0:
                    continue
                mask = ord_b <= upper
                vals = y_b[mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    feat_u[k] = self._expanding_stat(vals)
            result[idxs] = feat_u[inv]
        return result

    def _expanding_stat(self, vals: np.ndarray) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `_expanding_stat`."
        )


def _expanding_mean_from_agg(agg, lag):
    cum_sum = np.cumsum(agg.sums)
    cum_cnt = np.cumsum(agg.counts)
    upper_ord = agg.unique_times - lag
    upper_idxs = np.searchsorted(agg.unique_times, upper_ord, side="right") - 1
    win_sum = np.where(upper_idxs >= 0, cum_sum[upper_idxs], 0.0)
    win_cnt = np.where(upper_idxs >= 0, cum_cnt[upper_idxs], 0.0)
    safe_cnt = np.where(win_cnt > 0, win_cnt, 1.0)
    return np.where(win_cnt > 0, win_sum / safe_cnt, np.nan)


class ExpandingMean(_ExpandingBase):
    def _expanding_stat(self, vals: np.ndarray) -> float:
        return float(np.mean(vals))

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        return super()._compute_bucket_feature(bid_arr, _ts_arr, y_arr, ord_arr)

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        n = len(bid_arr)
        result = np.full(n, np.nan)
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _expanding_mean_from_agg(agg, lag)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_latest_from_aggs(self, ts_aggs, target_ords):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            cum_sum = np.cumsum(agg.sums)
            cum_cnt = np.cumsum(agg.counts)
            t = target_ords[bid]
            upper = t - lag
            ui = int(np.searchsorted(agg.unique_times, upper, side="right")) - 1
            s = cum_sum[ui] if ui >= 0 else 0.0
            c = cum_cnt[ui] if ui >= 0 else 0.0
            result[bid] = s / c if c > 0 else float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        return {
            bid: _expanding_mean_from_agg(agg, lag)
            for bid, agg in ts_aggs.items()
        }


def _expanding_std_from_agg(agg, lag):
    cum_sum = np.cumsum(agg.sums)
    cum_cnt = np.cumsum(agg.counts)
    cum_sum_sq = np.cumsum(agg.sum_sq)
    upper_ord = agg.unique_times - lag
    upper_idxs = np.searchsorted(agg.unique_times, upper_ord, side="right") - 1
    win_sum = np.where(upper_idxs >= 0, cum_sum[upper_idxs], 0.0)
    win_sq = np.where(upper_idxs >= 0, cum_sum_sq[upper_idxs], 0.0)
    win_cnt = np.where(upper_idxs >= 0, cum_cnt[upper_idxs], 0.0)
    safe_n = np.where(win_cnt > 0, win_cnt, 1.0)
    den = np.where(win_cnt > 1, win_cnt - 1, 1.0)
    var = (win_sq - win_sum**2 / safe_n) / den
    var = np.maximum(var, 0.0)
    return np.where((win_cnt > 1), np.sqrt(var), np.nan)


class ExpandingStd(_ExpandingBase):
    def _expanding_stat(self, vals: np.ndarray) -> float:
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        return super()._compute_bucket_feature(bid_arr, _ts_arr, y_arr, ord_arr)

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        n = len(bid_arr)
        result = np.full(n, np.nan)
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _expanding_std_from_agg(agg, lag)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_latest_from_aggs(self, ts_aggs, target_ords):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            cum_sum = np.cumsum(agg.sums)
            cum_cnt = np.cumsum(agg.counts)
            cum_sum_sq = np.cumsum(agg.sum_sq)
            t = target_ords[bid]
            upper = t - lag
            ui = int(np.searchsorted(agg.unique_times, upper, side="right")) - 1
            s = cum_sum[ui] if ui >= 0 else 0.0
            c = cum_cnt[ui] if ui >= 0 else 0.0
            sq = cum_sum_sq[ui] if ui >= 0 else 0.0
            if c > 1:
                var = (sq - s**2 / c) / (c - 1)
                var = max(var, 0.0)
                result[bid] = float(np.sqrt(var))
            else:
                result[bid] = float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        return {
            bid: _expanding_std_from_agg(agg, lag)
            for bid, agg in ts_aggs.items()
        }


def _expanding_min_from_agg(agg, lag):
    prefix_min = np.fmin.accumulate(agg.mins)
    upper_ord = agg.unique_times - lag
    upper_idxs = np.searchsorted(agg.unique_times, upper_ord, side="right") - 1
    return np.where(upper_idxs >= 0, prefix_min[upper_idxs], np.nan)


def _expanding_max_from_agg(agg, lag):
    prefix_max = np.fmax.accumulate(agg.maxs)
    upper_ord = agg.unique_times - lag
    upper_idxs = np.searchsorted(agg.unique_times, upper_ord, side="right") - 1
    return np.where(upper_idxs >= 0, prefix_max[upper_idxs], np.nan)


class ExpandingMin(_ExpandingBase):
    def _expanding_stat(self, vals: np.ndarray) -> float:
        return float(np.min(vals))

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        return super()._compute_bucket_feature(bid_arr, _ts_arr, y_arr, ord_arr)

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        n = len(bid_arr)
        result = np.full(n, np.nan)
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _expanding_min_from_agg(agg, lag)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_latest_from_aggs(self, ts_aggs, target_ords):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            prefix_min = np.fmin.accumulate(agg.mins)
            t = target_ords[bid]
            upper = t - lag
            ui = int(np.searchsorted(agg.unique_times, upper, side="right")) - 1
            result[bid] = float(prefix_min[ui]) if ui >= 0 else float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        return {
            bid: _expanding_min_from_agg(agg, lag)
            for bid, agg in ts_aggs.items()
        }


class ExpandingMax(_ExpandingBase):
    def _expanding_stat(self, vals: np.ndarray) -> float:
        return float(np.max(vals))

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        return super()._compute_bucket_feature(bid_arr, _ts_arr, y_arr, ord_arr)

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        n = len(bid_arr)
        result = np.full(n, np.nan)
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _expanding_max_from_agg(agg, lag)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_latest_from_aggs(self, ts_aggs, target_ords):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            prefix_max = np.fmax.accumulate(agg.maxs)
            t = target_ords[bid]
            upper = t - lag
            ui = int(np.searchsorted(agg.unique_times, upper, side="right")) - 1
            result[bid] = float(prefix_max[ui]) if ui >= 0 else float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        return {
            bid: _expanding_max_from_agg(agg, lag)
            for bid, agg in ts_aggs.items()
        }


class ExpandingQuantile(_ExpandingBase):
    def __init__(
        self,
        p: float,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(global_=global_, groupby=groupby, **kwargs)
        self.p = p

    @property
    def update_samples(self) -> int:
        return -1

    def _expanding_stat(self, vals: np.ndarray) -> float:
        return float(np.quantile(vals, self.p))


def _ewm_from_agg(agg, lag, alpha):
    mean_per_ord = np.full(len(agg.counts), np.nan)
    np.divide(agg.sums, agg.counts, out=mean_per_ord, where=agg.counts > 0)
    feat_u = np.full(len(agg.unique_times), np.nan)
    ewm = np.nan
    consume_idx = 0
    for k in range(len(agg.unique_times)):
        upper = agg.unique_times[k] - lag
        while consume_idx < len(agg.unique_times) and agg.unique_times[consume_idx] <= upper:
            if not np.isnan(mean_per_ord[consume_idx]):
                ewm = mean_per_ord[consume_idx] if np.isnan(ewm) else alpha * mean_per_ord[consume_idx] + (1 - alpha) * ewm
            consume_idx += 1
        feat_u[k] = ewm
    return feat_u


class ExponentiallyWeightedMean(_BaseLagTransform):
    """Exponentially weighted average

    Args:
        alpha (float): Smoothing factor.
        global_ (bool): If True, compute the statistic across all series aggregated by timestamp.
            Requires all series to end at the same timestamp. Defaults to False.
        groupby (Sequence[str], optional): Column names to group by before computing the statistic.
            Columns must be static features. Mutually exclusive with `global_`. Defaults to None.
    """

    def __init__(
        self,
        alpha: float,
        global_: bool = False,
        groupby: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        if "global" in kwargs:
            global_ = kwargs.pop("global")
        if "groupby" in kwargs:
            groupby = kwargs.pop("groupby")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
        self.alpha = alpha
        self.global_ = global_
        self.groupby = _normalize_columns(groupby)
        if self.global_ and self.groupby:
            raise ValueError("`global_` and `groupby` can't be used together.")

    @property
    def update_samples(self) -> int:
        return 1

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        _ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> np.ndarray:
        if _ts_aggs:
            return self._compute_from_aggregates(bid_arr, ord_arr, _ts_aggs)
        lag = self._core_tfm.lag
        alpha = self.alpha
        n = len(bid_arr)
        result = np.empty(n)
        result[:] = np.nan
        for bid in np.unique(bid_arr):
            idxs = np.where(bid_arr == bid)[0]
            ord_b = ord_arr[idxs]
            y_b = y_arr[idxs]
            unique_ord = np.unique(ord_b)
            inv = np.searchsorted(unique_ord, ord_b)
            mean_per_ord = np.full(len(unique_ord), np.nan)
            for k, o in enumerate(unique_ord):
                mask = ord_b == o
                vals = y_b[mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    mean_per_ord[k] = np.mean(vals)
            feat_u = np.full(len(unique_ord), np.nan)
            ewm = np.nan
            consume_idx = 0
            for k in range(len(unique_ord)):
                upper = unique_ord[k] - lag
                while consume_idx < len(unique_ord) and unique_ord[consume_idx] <= upper:
                    if not np.isnan(mean_per_ord[consume_idx]):
                        ewm = mean_per_ord[consume_idx] if np.isnan(ewm) else alpha * mean_per_ord[consume_idx] + (1 - alpha) * ewm
                    consume_idx += 1
                feat_u[k] = ewm
            result[idxs] = feat_u[inv]
        return result

    def _compute_from_aggregates(self, bid_arr, ord_arr, ts_aggs):
        lag = self._core_tfm.lag
        alpha = self.alpha
        n = len(bid_arr)
        result = np.full(n, np.nan)
        for bid, agg in ts_aggs.items():
            idxs = np.where(bid_arr == bid)[0]
            feat_u = _ewm_from_agg(agg, lag, alpha)
            inv = np.searchsorted(agg.unique_times, ord_arr[idxs])
            result[idxs] = feat_u[inv]
        return result

    def _compute_latest_from_aggs(self, ts_aggs, target_ords):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        alpha = self.alpha
        result: Dict[int, float] = {}
        for bid, agg in ts_aggs.items():
            if len(agg.unique_times) == 0:
                result[bid] = float("nan")
                continue
            t = target_ords[bid]
            mean_per_ord = np.full(len(agg.counts), np.nan)
            np.divide(agg.sums, agg.counts, out=mean_per_ord, where=agg.counts > 0)
            ewm = np.nan
            upper = t - lag
            for k in range(len(agg.unique_times)):
                if agg.unique_times[k] > upper:
                    break
                if not np.isnan(mean_per_ord[k]):
                    ewm = mean_per_ord[k] if np.isnan(ewm) else alpha * mean_per_ord[k] + (1 - alpha) * ewm
            result[bid] = float(ewm) if not np.isnan(ewm) else float("nan")
        return result

    def _compute_ts_level_from_aggs(self, ts_aggs):
        if not ts_aggs:
            return None
        lag = self._core_tfm.lag
        alpha = self.alpha
        return {
            bid: _ewm_from_agg(agg, lag, alpha)
            for bid, agg in ts_aggs.items()
        }


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

    def _get_name(self, lag: int) -> str:
        return self.tfm._get_name(lag + self.n)

    def _set_core_tfm(self, lag: int) -> "Offset":
        self.tfm = copy.deepcopy(self.tfm)._set_core_tfm(lag + self.n)
        self._core_tfm = self.tfm._core_tfm
        return self

    def _get_configured_lag(self) -> int:
        return self.tfm._get_configured_lag() - self.n

    @property
    def update_samples(self) -> int:
        return self.tfm.update_samples + self.n

    def _compute_ts_level_from_aggs(self, ts_aggs):
        return self.tfm._compute_ts_level_from_aggs(ts_aggs)

    def _compute_latest_from_aggs(self, ts_aggs, target_ords):
        return self.tfm._compute_latest_from_aggs(ts_aggs, target_ords)

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> Optional[np.ndarray]:
        return self.tfm._compute_bucket_feature(
            bid_arr, ts_arr, y_arr, ord_arr, _ts_aggs=_ts_aggs,
        )


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
            raise ValueError("Can't combine transforms with different global_ settings.")
        if (groupby_1 or groupby_2) and groupby_1 != groupby_2:
            raise ValueError("Can't combine transforms with different groupby settings.")
        self.global_ = global_1
        self.groupby = groupby_1

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

    def _compute_ts_level_from_aggs(self, ts_aggs):
        r1 = self.tfm1._compute_ts_level_from_aggs(ts_aggs)
        r2 = self.tfm2._compute_ts_level_from_aggs(ts_aggs)
        if r1 is not None and r2 is not None:
            return {
                bid: self.operator(r1[bid], r2[bid])
                for bid in r1
                if bid in r2
            }
        return None

    def _compute_latest_from_aggs(self, ts_aggs, target_ords):
        r1 = self.tfm1._compute_latest_from_aggs(ts_aggs, target_ords)
        r2 = self.tfm2._compute_latest_from_aggs(ts_aggs, target_ords)
        if r1 is not None and r2 is not None:
            return {
                bid: self.operator(r1[bid], r2[bid])
                for bid in r1
                if bid in r2
            }
        return None

    def _compute_bucket_feature(
        self,
        bid_arr: np.ndarray,
        ts_arr: np.ndarray,
        y_arr: np.ndarray,
        ord_arr: np.ndarray,
        _ts_aggs=None,
    ) -> Optional[np.ndarray]:
        v1 = self.tfm1._compute_bucket_feature(
            bid_arr, ts_arr, y_arr, ord_arr, _ts_aggs=_ts_aggs,
        )
        v2 = self.tfm2._compute_bucket_feature(
            bid_arr, ts_arr, y_arr, ord_arr, _ts_aggs=_ts_aggs,
        )
        if v1 is not None and v2 is not None:
            return self.operator(v1, v2)
        return None

    def _get_configured_lag(self) -> int:
        lag1 = self.tfm1._get_configured_lag()
        lag2 = self.tfm2._get_configured_lag()
        if lag1 != lag2:
            raise ValueError("Combined transforms must share the same configured lag.")
        return lag1

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
