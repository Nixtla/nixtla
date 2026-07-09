__all__ = ["PooledState", "compute_pooled_features"]

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import narwhals as nw
import numpy as np
import utilsforecast.processing as ufp

from .lag_transforms import _BaseLagTransform, _TIME_AGGS


def _dedupe_preserve_order(items):
    return list(dict.fromkeys(items))


# Sentinel string that every missing (null/NaN) key value is encoded to, so that
# missing matches missing (and only missing) across pandas/polars and all key
# dtypes. NUL-prefixed to make a collision with a real key value negligible.
_NULL_SENTINEL = "\x00__MLF_NULL__"


def _missing_expr(col, dtype):
    """Dtype-aware "is missing" predicate: null, plus NaN for float columns.

    narwhals ``is_null`` flags NaN on pandas but not float NaN on polars, so the
    ``is_nan`` arm is required (and is only valid on float dtypes).
    """
    expr = nw.col(col).is_null()
    if dtype.is_float():
        expr = expr | nw.col(col).is_nan()
    return expr


def _encode_col_expr(col, dtype, *, to_int=False):
    """Sentinel-encode one key column to a String expression.

    Missing values (null/NaN) become ``_NULL_SENTINEL``; non-missing values are
    cast to String. When ``to_int`` the *integral* non-missing values are routed
    through Int64 first so a float ``0.0`` matches an integer ``0`` (``"0"`` on
    both sides); a genuinely fractional float (e.g. ``1.5``) keeps its own float
    string form so it does not collide with an integer bucket. The int cast is
    applied only after missing values are masked out, since casting NaN/null
    through a non-nullable int raises on both engines.
    """
    miss = _missing_expr(col, dtype)
    if not to_int:
        return (
            nw.when(miss)
            .then(nw.lit(_NULL_SENTINEL))
            .otherwise(nw.col(col).cast(nw.String))
        )
    # replace missing with a safe placeholder before the (otherwise-raising) int
    # cast; placeholder rows are overwritten by the sentinel branch below.
    safe = nw.when(miss).then(nw.lit(0.0)).otherwise(nw.col(col))
    as_int = safe.cast(nw.Int64)
    is_integral = safe == as_int  # 1.0 -> True, 1.5 -> False
    # integral -> int string ("1"); fractional -> float string ("1.5", which
    # cannot collide with an integer bucket); missing -> sentinel.
    non_missing = (
        nw.when(is_integral)
        .then(as_int.cast(nw.String))
        .otherwise(safe.cast(nw.String))
    )
    return nw.when(miss).then(nw.lit(_NULL_SENTINEL)).otherwise(non_missing)


def _encode_keys(frame_nw, cols):
    """Add transient ``__enc_<col>`` string columns for bucket *creation*.

    Single-frame: no cross-frame dtype reconcile is needed because creation and
    its matching join operate on the same frame.
    """
    schema = frame_nw.schema
    return frame_nw.with_columns(
        [_encode_col_expr(c, schema[c]).alias(f"__enc_{c}") for c in cols]
    )


def _encode_join_keys(left_nw, right_nw, cols):
    """Sentinel-encode key ``cols`` on both frames for a null-equal *match*.

    Applies a narrow int<->float reconcile: when one side's key column is integer
    and the other float (e.g. a fit column contaminated to float by a NaN, matched
    against clean integer data), the float side's integral non-missing values are
    cast to int so they match. No blanket Float64 widening (that would corrupt
    large int keys above 2**53).
    """
    lschema, rschema = left_nw.schema, right_nw.schema
    left_exprs, right_exprs = [], []
    for c in cols:
        ldt, rdt = lschema[c], rschema[c]
        left_exprs.append(
            _encode_col_expr(c, ldt, to_int=ldt.is_float() and rdt.is_integer()).alias(
                f"__enc_{c}"
            )
        )
        right_exprs.append(
            _encode_col_expr(c, rdt, to_int=rdt.is_float() and ldt.is_integer()).alias(
                f"__enc_{c}"
            )
        )
    return left_nw.with_columns(left_exprs), right_nw.with_columns(right_exprs)


_ROW_ORDER_COL = "_mlf_row_order"


def _order_preserving_left_join(left_nw, right_nw, on):
    """Left join that preserves the left frame's row order.

    Results are consumed positionally against arrays aligned with the left
    frame, but narwhals exposes no ``maintain_order`` for joins and polars does
    not guarantee row order, so attach a row index to the left frame, sort on
    it after the join, and drop it. All call sites join against key-unique
    right frames, so the join never fans out.
    """
    left_idx = left_nw.with_row_index(name=_ROW_ORDER_COL)
    joined = left_idx.join(right_nw, on=list(on), how="left")
    return joined.sort(_ROW_ORDER_COL).drop(_ROW_ORDER_COL)


def _null_equal_left_join(left_nw, right_nw, cols, payload_cols):
    """Left-join ``left_nw`` to ``right_nw`` on ``cols`` treating missing as equal.

    ``right_nw`` contributes only ``payload_cols`` (e.g. ``["_bucket_id"]``); its
    original key columns are excluded so the join emits no ``*_right`` duplicates.
    Encoded columns are transient and dropped before returning.
    """
    enc_cols = [f"__enc_{c}" for c in cols]
    left_enc, right_enc = _encode_join_keys(left_nw, right_nw, cols)
    right_enc = right_enc.select(enc_cols + list(payload_cols))
    joined = _order_preserving_left_join(left_enc, right_enc, on=enc_cols)
    return joined.drop(enc_cols)


def add_bucket_id(data, cols):
    cols = list(cols)
    enc_cols = [f"__enc_{c}" for c in cols]
    data_enc = _encode_keys(nw.from_native(data), cols)
    # One representative original-key row per encoded key, so null and NaN (and
    # pandas-None) collapse into a single bucket and the join below cannot fan out.
    groups_enc = data_enc.select(cols + enc_cols).unique(
        subset=enc_cols, keep="first", maintain_order=True
    )
    groups_enc = groups_enc.with_row_index(name="_bucket_id").with_columns(
        nw.col("_bucket_id").cast(nw.Int64)
    )
    merged_nw = _order_preserving_left_join(
        data_enc, groups_enc.select(enc_cols + ["_bucket_id"]), on=enc_cols
    ).drop(enc_cols)
    groups_nw = groups_enc.drop(enc_cols)
    return (
        ufp.drop_index_if_pandas(nw.to_native(merged_nw)),
        ufp.drop_index_if_pandas(nw.to_native(groups_nw)),
    )


def lookup_bucket_ids(data, groups, cols):
    cols = list(cols)
    joined = _null_equal_left_join(
        nw.from_native(data).select(cols),
        nw.from_native(groups),
        cols,
        ["_bucket_id"],
    )
    return joined.get_column("_bucket_id").to_numpy()


@dataclass
class _TimestampAggregates:
    """Per-timestamp aggregates for a single bucket."""

    unique_times: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    sum_sq: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray


def _build_ts_aggs(
    bid_arr: np.ndarray,
    ord_arr: np.ndarray,
    y_arr: np.ndarray,
) -> Dict[int, _TimestampAggregates]:
    aggs: Dict[int, _TimestampAggregates] = {}
    for bid in np.unique(bid_arr):
        mask = bid_arr == bid
        ord_b = ord_arr[mask]
        y_b = y_arr[mask]
        unique_ord, inv = np.unique(ord_b, return_inverse=True)
        m = len(unique_ord)
        valid = ~np.isnan(y_b)
        y_valid = np.where(valid, y_b, 0.0)
        sums = np.bincount(inv, weights=y_valid, minlength=m)
        counts = np.bincount(inv, weights=valid.astype(float), minlength=m)
        sum_sq = np.bincount(inv, weights=np.where(valid, y_b**2, 0.0), minlength=m)
        mins = np.full(m, np.inf)
        maxs = np.full(m, -np.inf)
        valid_inv = inv[valid]
        valid_y = y_b[valid]
        if len(valid_y) > 0:
            np.minimum.at(mins, valid_inv, valid_y)
            np.maximum.at(maxs, valid_inv, valid_y)
        no_valid = mins == np.inf
        mins[no_valid] = np.nan
        maxs[no_valid] = np.nan
        aggs[int(bid)] = _TimestampAggregates(
            unique_times=unique_ord,
            sums=sums,
            counts=counts,
            sum_sq=sum_sq,
            mins=mins,
            maxs=maxs,
        )
    return aggs


def _time_agg_values(agg: _TimestampAggregates, time_agg: str) -> np.ndarray:
    """Per-timestamp collapsed value ``v_t`` for one bucket's aggregates.

    ``time_agg`` is one of :data:`_TIME_AGGS`.  A timestamp that has rows but
    no non-NaN target (``counts == 0``) is *unobserved* for
    ``sum``/``mean``/``min``/``max`` and gets NaN, matching SQL aggregates over
    all-NULL groups returning NULL.  For ``count`` the value is the non-NaN row
    count — 0 there is a genuine observation (SQL ``COUNT`` over all-NULL
    returns 0).  This is the single source of the per-aggregate value
    derivation, shared by the fast (:func:`_reaggregate_ts_aggs`) and slow
    (:func:`_collapse_rows_by_time`) paths so they cannot diverge.
    """
    if time_agg not in _TIME_AGGS:
        raise ValueError(f"time_agg must be one of {_TIME_AGGS}; got {time_agg!r}.")
    if time_agg == "count":
        return agg.counts.astype(float)
    if time_agg == "min":
        return agg.mins.astype(float)
    if time_agg == "max":
        return agg.maxs.astype(float)
    obs = agg.counts > 0
    if time_agg == "sum":
        return np.where(obs, agg.sums, np.nan)
    # "mean"
    v = np.full(len(agg.unique_times), np.nan)
    np.divide(agg.sums, agg.counts, out=v, where=obs)
    return v


class _ReaggregatedAggregates:
    """Lazy ``_TimestampAggregates`` stand-in over a ``time_agg``-collapsed bucket.

    The collapsed value ``v_t`` (:func:`_time_agg_values`) at each observed
    timestamp becomes a unit-count observation — ``sums=v``, ``counts=1``,
    ``sum_sq=v**2``, ``mins=maxs=v`` — while unobserved timestamps (NaN in the
    value array) keep ``counts=0``, zeroed ``sums``/``sum_sq`` (so they never
    poison the cumulative sums the helpers take) and NaN ``mins``/``maxs`` (the
    range helpers use NaN-skipping fmin/fmax and the count guard excludes
    them).

    Everything is derived on demand: the value array itself is only computed on
    first field access (so re-aggregating before an unsupported hook that
    returns ``None`` does no numpy work at all), and each of the five aggregate
    fields — of which a given helper reads at most three — is derived on access
    rather than materialized up front.  Accessors return freshly allocated
    arrays except ``mins``/``maxs``, which return the shared value array and
    must not be mutated.
    """

    def __init__(self, source: "_TimestampAggregates", time_agg: str):
        self.unique_times = source.unique_times
        self._source = source
        self._time_agg = time_agg
        self._values_cache: Optional[np.ndarray] = None
        self._obs_cache: Optional[np.ndarray] = None

    @property
    def _values(self) -> np.ndarray:
        if self._values_cache is None:
            self._values_cache = _time_agg_values(self._source, self._time_agg)
        return self._values_cache

    @property
    def _obs(self) -> np.ndarray:
        if self._obs_cache is None:
            self._obs_cache = ~np.isnan(self._values)
        return self._obs_cache

    @property
    def sums(self) -> np.ndarray:
        return np.where(self._obs, self._values, 0.0)

    @property
    def counts(self) -> np.ndarray:
        return self._obs.astype(float)

    @property
    def sum_sq(self) -> np.ndarray:
        return np.where(self._obs, self._values**2, 0.0)

    @property
    def mins(self) -> np.ndarray:
        return self._values

    @property
    def maxs(self) -> np.ndarray:
        return self._values


def _reaggregate_ts_aggs(
    ts_aggs: Dict[int, _TimestampAggregates], time_agg: str
) -> Dict[int, _ReaggregatedAggregates]:
    """Collapse each bucket's per-timestamp aggregates to a single value per
    timestamp, returning lazy ``_TimestampAggregates``-compatible views shaped
    so the existing rolling/expanding/ewm helpers compute the transform *over
    the per-timestamp aggregated series* (e.g. a rolling mean of daily sums).
    See :func:`_time_agg_values` for the value/NaN conventions and
    :class:`_ReaggregatedAggregates` for the field layout.

    The input is the shared ``PooledState._ts_aggs`` cache and is never
    mutated.  ``unique_times`` is intentionally shared by reference
    (``append_predictions`` reassigns it via ``np.append`` rather than mutating
    in place) and preserved exactly, because the fit-path fast mapping indexes
    results by that grid.  Callers must recompute this per call — never cache
    it, or it goes stale after ``append_predictions`` extends the underlying
    aggregates.
    """
    if time_agg not in _TIME_AGGS:
        raise ValueError(f"time_agg must be one of {_TIME_AGGS}; got {time_agg!r}.")
    return {bid: _ReaggregatedAggregates(agg, time_agg) for bid, agg in ts_aggs.items()}


def _collapse_rows_by_time(
    bid_arr: np.ndarray,
    ord_arr: np.ndarray,
    y_arr: np.ndarray,
    time_agg: str,
    ts_aggs: Optional[Dict[int, _TimestampAggregates]] = None,
):
    """Collapse raw ``(bucket, ordinal, y)`` rows to one row per
    ``(bucket, timestamp)`` holding the ``time_agg`` aggregate, for the
    row-level (slow-path) transforms that have no aggregate-cache fast path.

    ``ts_aggs`` is an optional pre-built per-timestamp aggregate cache for
    exactly these rows (``PooledState._ts_aggs`` at fit, the query-array
    aggregates at predict); when supplied the collapse is derived from it in
    O(unique timestamps) instead of re-aggregating every raw row.

    Returns ``(bid2, ord2, y2, inv)`` with one entry per unique
    ``(bucket, ord)`` and ``inv`` mapping every input row to its collapsed row,
    so ``vals[inv]`` broadcasts per-timestamp results back to the raw rows.
    All-NaN timestamps get ``y2 = NaN`` for ``sum``/``mean``/``min``/``max`` (so
    the row loops' NaN filtering treats them as unobserved) and ``y2 = 0`` for
    ``count`` (a genuine zero observation).  This mirrors
    :func:`_reaggregate_ts_aggs` — both derive values via
    :func:`_time_agg_values` — i.e.
    ``_build_ts_aggs(*_collapse_rows_by_time(rows, a)[:3])`` equals
    ``_reaggregate_ts_aggs(_build_ts_aggs(rows), a)``.
    """
    if not ts_aggs:  # absent or wiped cache: aggregate the raw rows
        ts_aggs = _build_ts_aggs(bid_arr, ord_arr, y_arr)
    # group the input rows by bucket once (argsort + slices) instead of a
    # full-array mask per bucket
    order = np.argsort(bid_arr, kind="stable")
    sorted_bids = bid_arr[order]
    group_bids, group_starts = np.unique(sorted_bids, return_index=True)
    group_ends = np.append(group_starts[1:], len(order))
    row_groups = {
        int(b): order[s:e] for b, s, e in zip(group_bids, group_starts, group_ends)
    }
    # -1 sentinel: every input row must map to a collapsed row below. Callers
    # always build ``ts_aggs`` from exactly these rows, so no row can reference a
    # bucket absent from ``ts_aggs``; the assert turns a silent out-of-bounds
    # gather (garbage ``vals[inv]``) into a loud failure if that ever breaks.
    inv = np.full(len(bid_arr), -1, dtype=np.int64)
    bids, ords, ys = [], [], []
    offset = 0
    for bid, agg in ts_aggs.items():
        m = len(agg.unique_times)
        if m == 0:
            continue
        bids.append(np.full(m, bid))
        ords.append(agg.unique_times)
        ys.append(_time_agg_values(agg, time_agg))
        rows = row_groups.get(int(bid))
        if rows is not None:
            inv[rows] = offset + np.searchsorted(agg.unique_times, ord_arr[rows])
        offset += m
    assert (inv >= 0).all(), (
        "every input row must map to a collapsed (bucket, timestamp)"
    )
    if not bids:
        return (
            np.empty(0, dtype=bid_arr.dtype),
            np.empty(0, dtype=ord_arr.dtype),
            np.empty(0, dtype=float),
            inv,
        )
    return (
        np.concatenate(bids).astype(bid_arr.dtype),
        np.concatenate(ords).astype(ord_arr.dtype),
        np.concatenate(ys),
        inv,
    )


def _compute_time_index(bid_arr, ts_arr):
    """Assign integer period coordinates over the validated regular time grid.

    For each bucket, maps raw timestamps to consecutive integers based on
    the bucket's sorted unique timestamps. Because the input grid is validated
    as regular (no gaps within a series), this is equivalent to SQL RANGE
    interval semantics. Series that start later simply have no rows at earlier
    periods — no synthetic zeros are injected.
    """
    idx_arr = np.empty(len(ts_arr), dtype=np.int64)
    next_by_bucket = {}
    for bid in np.unique(bid_arr):
        mask = bid_arr == bid
        ts_b = ts_arr[mask]
        unique_ts = np.unique(ts_b)
        idx_arr[mask] = np.searchsorted(unique_ts, ts_b)
        next_by_bucket[int(bid)] = len(unique_ts)
    return idx_arr, next_by_bucket


def _compute_time_index_from_parent(bid_arr, ts_arr, parent_grids):
    """Assign ordinals using the parent calendar's time grid.

    For partition_by states, ordinals must reflect the parent calendar so
    that missing partition values leave holes.  E.g. if the parent has
    timestamps [1,2,3,4,5] but a partition bucket only has [1,3,5], the
    ordinals are [0,2,4] (not [0,1,2]).

    Parameters
    ----------
    bid_arr : np.ndarray
        Bucket ID for each observation.
    ts_arr : np.ndarray
        Timestamp for each observation.
    parent_grids : Dict[int, np.ndarray]
        Maps each bucket_id to the sorted unique timestamps of its parent
        calendar.

    Returns
    -------
    idx_arr : np.ndarray
        Ordinal coordinates derived from parent calendar positions.
    next_by_bucket : Dict[int, int]
        Next ordinal for each bucket (= len(parent_grid)).
    """
    idx_arr = np.empty(len(ts_arr), dtype=np.int64)
    next_by_bucket = {}
    for bid in np.unique(bid_arr):
        mask = bid_arr == bid
        ts_b = ts_arr[mask]
        parent_ts = parent_grids[int(bid)]
        idx_arr[mask] = np.searchsorted(parent_ts, ts_b)
        next_by_bucket[int(bid)] = len(parent_ts)
    return idx_arr, next_by_bucket


def _compute_idsorted_to_bucket_pos(bucket_df, id_col, time_col):
    _pos_col = "_idsorted_tmp_pos"
    df_nw = nw.from_native(bucket_df)
    df_nw = (
        df_nw.select([id_col, time_col])
        .with_row_index(name=_pos_col)
        .with_columns(nw.col(_pos_col).cast(nw.Int64))
    )
    idsorted = ufp.sort(nw.to_native(df_nw), by=[id_col, time_col])
    return idsorted[_pos_col].to_numpy()


@dataclass
class PooledState:
    """Holds all arrays and metadata for a pooled bucket.

    Covers global, groupby, and partition_by modes (and combinations thereof).

    ``time_index`` is an integer period coordinate over a validated regular
    time grid.  For global/group states (no partition), the input is validated
    to have no missing timestamps within any series.  Different series in the
    same bucket may have different start dates; only actual observations are
    included (no synthetic zeros).  Under this invariant, ``time_index``
    is equivalent to SQL ``RANGE BETWEEN … PRECEDING`` interval semantics.

    For partition_by buckets, ``time_index`` is derived from the parent
    series/global calendar — not the bucket's observed timestamps — so that
    missing partition values leave holes and window bounds remain
    interval-correct.

    **Ordering contract**: ``bucket_id``, ``time``, ``time_index``, and ``y``
    arrays are positionally aligned — row *i* in each describes the same
    observation.  All feature computation uses these flat arrays (and the
    derived ``_ts_aggs``) via ``compute_pooled_features``.
    """

    bucket_df: Any
    groups: Any
    group_cols: Optional[List[str]]
    series_bucket_id: np.ndarray
    bucket_id: np.ndarray
    time: np.ndarray
    time_index: np.ndarray
    y: np.ndarray
    next_time_index_by_bucket: Dict[int, int]
    join_cols: List[str]
    mode: str = "nonlocal"
    partition_cols: Optional[List[str]] = None
    key_cols: Optional[List[str]] = None
    parent_scope_cols: Optional[List[str]] = None
    _parent_time_grids: Optional[Dict[int, np.ndarray]] = None
    _bucket_to_parent_id: Optional[Dict[int, int]] = None
    _parent_to_buckets: Optional[Dict[int, List[int]]] = None
    _scope_key_to_parent_id: Optional[Dict[tuple, int]] = None
    _ts_aggs: Dict[int, _TimestampAggregates] = field(default_factory=dict)
    _idsorted_to_bucket_pos: Optional[np.ndarray] = None

    @property
    def group_uids(self):
        if self.groups is None:
            ns = nw.get_native_namespace(self.bucket_df)
            return nw.to_native(
                nw.new_series("_bucket_id", [0], dtype=nw.Int64, backend=ns)
            )
        return nw.to_native(nw.from_native(self.groups).get_column("_bucket_id").sort())

    # Mutable fields touched during recursive prediction. ``snapshot``/``restore``
    # let ``TimeSeries._backup`` save and roll back state cheaply (see below);
    # keep this list in sync with the fields mutated by ``append_predictions`` /
    # ``append_observations`` / ``update_series_bucket_id``.
    _MUTABLE_REF_FIELDS = (
        "bucket_df",
        "groups",
        "series_bucket_id",
        "bucket_id",
        "time",
        "time_index",
        "y",
        "_idsorted_to_bucket_pos",
    )
    _MUTABLE_DICT_FIELDS = (
        "next_time_index_by_bucket",
        "_parent_time_grids",
        "_bucket_to_parent_id",
        "_scope_key_to_parent_id",
    )

    def snapshot(self):
        """Cheap structural backup for recursive prediction.

        Prediction only ever **replaces** the array fields wholesale
        (``np.append`` / ``np.concatenate`` produce new arrays) and rebinds
        attributes on the ``_TimestampAggregates`` instances — it never mutates
        a shared array in place. So a faithful backup needs only reference
        copies of the arrays plus shallow copies of the mutable containers and
        of each aggregate instance; no array data is duplicated (unlike a full
        ``deepcopy`` of every pooled state per model per ``predict``).
        """
        snap = {f: getattr(self, f) for f in self._MUTABLE_REF_FIELDS}
        for f in self._MUTABLE_DICT_FIELDS:
            v = getattr(self, f)
            snap[f] = None if v is None else dict(v)
        # lists are reassigned wholesale today, but copy them defensively so a
        # future in-place ``append`` can't corrupt the backup.
        snap["_parent_to_buckets"] = (
            None
            if self._parent_to_buckets is None
            else {k: list(v) for k, v in self._parent_to_buckets.items()}
        )
        # each agg instance is mutated in place (``agg.sums = np.append(...)``),
        # so copy the instances; their arrays are replaced wholesale, share them.
        snap["_ts_aggs"] = {bid: copy.copy(agg) for bid, agg in self._ts_aggs.items()}
        return snap

    def restore(self, snap):
        for field_name, value in snap.items():
            setattr(self, field_name, value)

    @classmethod
    def from_global(
        cls,
        sorted_df,
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        n_series: int,
    ):
        keep_cols = _dedupe_preserve_order([id_col, time_col, target_col])
        global_df = sorted_df[keep_cols]
        global_df = ufp.sort(global_df, by=[time_col, id_col])
        global_df = ufp.drop_index_if_pandas(global_df)
        ts_raw = global_df[time_col].to_numpy()
        # cast through ga_data_dtype before float to keep numerics bit-identical
        # with the model's working dtype (e.g. float32 rounding).
        y_raw = global_df[target_col].to_numpy().astype(ga_data_dtype)
        unique_ts = np.unique(ts_raw)
        ord_raw = np.searchsorted(unique_ts, ts_raw).astype(np.int64)
        bid_arr = np.zeros(len(global_df), dtype=np.int64)
        y_float = y_raw.astype(float)
        return cls(
            bucket_df=global_df,
            groups=None,
            group_cols=None,
            series_bucket_id=np.zeros(n_series, dtype=np.int64),
            bucket_id=bid_arr,
            time=ts_raw,
            time_index=ord_raw,
            y=y_float,
            next_time_index_by_bucket={0: len(unique_ts)},
            join_cols=[id_col, time_col],
            _ts_aggs=_build_ts_aggs(bid_arr, ord_raw, y_float),
        )

    @classmethod
    def from_groupby(
        cls,
        df_for_group,
        group_cols_list: List[str],
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        static_features,
    ):
        keep_cols = _dedupe_preserve_order(
            [id_col] + group_cols_list + [time_col, target_col]
        )
        bucket_df = df_for_group[keep_cols]
        bucket_df = ufp.sort(bucket_df, by=group_cols_list + [time_col, id_col])
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        bucket_df, groups = add_bucket_id(bucket_df, group_cols_list)
        ts_raw = bucket_df[time_col].to_numpy()
        # cast through ga_data_dtype before float to keep numerics bit-identical
        # with the model's working dtype (e.g. float32 rounding).
        y_raw = bucket_df[target_col].to_numpy().astype(ga_data_dtype)
        bid_raw = bucket_df["_bucket_id"].to_numpy()
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        bid_arr = bid_raw.astype(np.int64)
        ord_arr, next_by_bucket = _compute_time_index(bid_arr, ts_raw)
        series_bucket_id = lookup_bucket_ids(
            static_features, groups, group_cols_list
        ).astype(np.int64, copy=False)
        y_float = y_raw.astype(float)
        return cls(
            bucket_df=bucket_df,
            groups=groups,
            group_cols=group_cols_list,
            series_bucket_id=series_bucket_id,
            bucket_id=bid_arr,
            time=ts_raw,
            time_index=ord_arr,
            y=y_float,
            next_time_index_by_bucket=next_by_bucket,
            join_cols=[id_col, time_col],
            _ts_aggs=_build_ts_aggs(bid_arr, ord_arr, y_float),
        )

    @classmethod
    def from_partition(
        cls,
        sorted_df,
        mode: str,
        group_cols_list: Optional[List[str]],
        partition_cols_list: List[str],
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        static_features,
        n_series: int,
    ):
        """Build a PooledState for partition_by transforms.

        Parameters
        ----------
        mode : str
            "local" — each (id, partition_vals) is its own bucket.
            "nonlocal" — bucket key is (*group_cols, *partition_cols) or
            (*partition_cols) for global+partition_by.
        group_cols_list : list of str or None
            Group columns (from ``groupby``). None for global+partition or
            local+partition.
        partition_cols_list : list of str
            Partition columns (from ``partition_by``).
        """
        if mode == "local":
            key_cols = _dedupe_preserve_order([id_col] + partition_cols_list)
        elif group_cols_list:
            key_cols = _dedupe_preserve_order(
                list(group_cols_list) + partition_cols_list
            )
        else:
            key_cols = _dedupe_preserve_order(partition_cols_list)

        # (id_col, time_col) uniquely identifies each row in bucket_df in every
        # mode, so the slow-path join never needs to key on the (nullable)
        # partition columns — a plain merge on a null partition value would
        # mismatch (NaN != NaN) and drop the feature.
        join_cols = [id_col, time_col]

        keep_cols = _dedupe_preserve_order([id_col] + key_cols + [time_col, target_col])
        bucket_df = sorted_df[keep_cols]
        bucket_df = ufp.sort(bucket_df, by=key_cols + [time_col, id_col])
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        bucket_df, groups = add_bucket_id(bucket_df, key_cols)
        ts_raw = bucket_df[time_col].to_numpy()
        # cast through ga_data_dtype before float to keep numerics bit-identical
        # with the model's working dtype (e.g. float32 rounding).
        y_raw = bucket_df[target_col].to_numpy().astype(ga_data_dtype)
        bid_raw = bucket_df["_bucket_id"].to_numpy()
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        bid_arr = bid_raw.astype(np.int64)

        if mode == "local":
            parent_scope_cols = [id_col]
        elif group_cols_list:
            parent_scope_cols = list(group_cols_list)
        else:
            parent_scope_cols = None

        parent_id_map: Dict[tuple, int] = {}
        bucket_to_parent: Dict[int, int] = {}
        parent_grids: Dict[int, np.ndarray] = {}
        next_pid = 0

        if parent_scope_cols is not None:
            # Sentinel-encode the scope columns so a null/NaN scope value (e.g. a
            # null groupby key under groupby+partition_by) matches itself and only
            # itself: scope keys are used both as dict keys (NaN != NaN would split
            # one scope across parents) and as a grid filter (an `== null` predicate
            # matches nothing). Mirrors the null-equal bucket-key handling above.
            enc_scope_cols = [f"__enc_{c}" for c in parent_scope_cols]
            groups_nw = _encode_keys(nw.from_native(groups), parent_scope_cols)
            sorted_nw = _encode_keys(nw.from_native(sorted_df), parent_scope_cols)
            # Single pass over the unique (scope, time) pairs to build every
            # scope's calendar at once instead of filtering the full frame per
            # bucket. Scope keys are read via .rows() so they compare equal to
            # the .row(0) tuples `_resolve_parent_for_bucket` builds later;
            # times come from .to_numpy() so grids keep their native dtype.
            scope_time = sorted_nw.select(enc_scope_cols + [time_col]).unique()
            grid_scope_keys = scope_time.select(enc_scope_cols).rows()
            grid_ts_vals = scope_time.get_column(time_col).to_numpy()
            times_by_scope: Dict[tuple, list] = {}
            for key, ts in zip(grid_scope_keys, grid_ts_vals):
                times_by_scope.setdefault(key, []).append(ts)

            # `groups` has one row per bucket in ascending _bucket_id order, so
            # iterating it assigns parent ids in the same order as before.
            bucket_scope_keys = groups_nw.select(enc_scope_cols).rows()
            for bid, scope_key in zip(
                groups_nw.get_column("_bucket_id").to_numpy(), bucket_scope_keys
            ):
                if scope_key not in parent_id_map:
                    pid = next_pid
                    next_pid += 1
                    parent_id_map[scope_key] = pid
                    parent_grids[pid] = np.sort(np.asarray(times_by_scope[scope_key]))
                bucket_to_parent[int(bid)] = parent_id_map[scope_key]
        else:
            all_ts = sorted_df[time_col].to_numpy()
            global_ts = np.sort(np.unique(all_ts))
            parent_grids[0] = global_ts
            for bid in np.unique(bid_arr):
                bucket_to_parent[int(bid)] = 0

        parent_to_buckets: Dict[int, List[int]] = {}
        for bid, pid in bucket_to_parent.items():
            parent_to_buckets.setdefault(pid, []).append(bid)

        if parent_scope_cols is not None:
            scope_key_to_parent = dict(parent_id_map)
        else:
            scope_key_to_parent = {(): 0}

        per_bucket_grids = {
            bid: parent_grids[bucket_to_parent[bid]] for bid in bucket_to_parent
        }
        ord_arr, next_by_bucket = _compute_time_index_from_parent(
            bid_arr, ts_raw, per_bucket_grids
        )

        sf_cols = set(static_features.columns)
        can_lookup = all(c in sf_cols for c in key_cols)
        if can_lookup:
            series_bucket_id = lookup_bucket_ids(
                static_features, groups, key_cols
            ).astype(np.int64, copy=False)
        else:
            series_bucket_id = np.zeros(n_series, dtype=np.int64)

        y_float = y_raw.astype(float)
        return cls(
            bucket_df=bucket_df,
            groups=groups,
            group_cols=group_cols_list,
            series_bucket_id=series_bucket_id,
            bucket_id=bid_arr,
            time=ts_raw,
            time_index=ord_arr,
            y=y_float,
            next_time_index_by_bucket=next_by_bucket,
            join_cols=join_cols,
            mode=mode,
            partition_cols=partition_cols_list,
            key_cols=key_cols,
            parent_scope_cols=parent_scope_cols,
            _parent_time_grids=parent_grids,
            _bucket_to_parent_id=bucket_to_parent,
            _parent_to_buckets=parent_to_buckets,
            _scope_key_to_parent_id=scope_key_to_parent,
            _ts_aggs=_build_ts_aggs(bid_arr, ord_arr, y_float),
        )

    def update_series_bucket_id(self, context_df, _id_col: str):
        """Recompute series_bucket_id from a context dataframe.

        This is used at prediction time when partition_by values are dynamic
        (e.g. the partition key comes from exogenous features that change
        each step). If new partition combinations appear, new buckets are
        created.

        Parameters
        ----------
        context_df : DataFrame
            Must contain ``id_col`` and all ``key_cols`` columns. One row per
            series, representing the current partition assignment.
        id_col : str
            The series identifier column.
        """
        if self.key_cols is None:
            return
        key_cols = self.key_cols
        context_with_bid = _attach_bucket_id(context_df, self.groups, key_cols)
        context_with_bid, self.groups = _extend_groups(
            context_with_bid, self.groups, key_cols
        )
        new_bid = context_with_bid["_bucket_id"].to_numpy().astype(np.int64)
        self.series_bucket_id = new_bid
        groups_nw = nw.from_native(self.groups)
        if self.parent_scope_cols:
            # Encode once so the per-bucket `_resolve_parent_for_bucket` loop below
            # reuses it instead of re-encoding each new bucket.
            groups_nw = _encode_keys(groups_nw, self.parent_scope_cols)
        for bid in np.unique(new_bid):
            bid_int = int(bid)
            if bid_int not in self.next_time_index_by_bucket:
                pid = self._resolve_parent_for_bucket(bid_int, groups_nw=groups_nw)
                if pid is not None and self._parent_time_grids is not None:
                    self.next_time_index_by_bucket[bid_int] = len(
                        self._parent_time_grids[pid]
                    )
                else:
                    self.next_time_index_by_bucket[bid_int] = 0
            if bid_int not in self._ts_aggs:
                self._ts_aggs[bid_int] = _TimestampAggregates(
                    unique_times=np.array([], dtype=np.intp),
                    sums=np.array([], dtype=np.float64),
                    counts=np.array([], dtype=np.intp),
                    sum_sq=np.array([], dtype=np.float64),
                    mins=np.array([], dtype=np.float64),
                    maxs=np.array([], dtype=np.float64),
                )

    def _resolve_parent_for_bucket(self, bid: int, groups_nw=None) -> Optional[int]:
        """Find or create the parent_id for a bucket from its scope columns.

        ``groups_nw`` is an optional pre-wrapped Narwhals view of ``self.groups``;
        callers that resolve many buckets in a loop pass it once to avoid
        re-wrapping per bucket. When omitted it is wrapped internally.
        """
        if (
            self._bucket_to_parent_id is None
            or self._parent_to_buckets is None
            or self._parent_time_grids is None
            or self._scope_key_to_parent_id is None
        ):
            return None

        if bid in self._bucket_to_parent_id:
            return self._bucket_to_parent_id[bid]

        if self.parent_scope_cols is None:
            scope_key = ()
        else:
            if groups_nw is None:
                groups_nw = nw.from_native(self.groups)
            # `_scope_key_to_parent_id` is keyed by sentinel-encoded scope tuples
            # (see `from_partition`), so encode here too — null/NaN scope values
            # must resolve to the same parent, not a fresh one per NaN.
            enc_scope_cols = [f"__enc_{c}" for c in self.parent_scope_cols]
            if enc_scope_cols[0] not in groups_nw.columns:
                groups_nw = _encode_keys(groups_nw, self.parent_scope_cols)
            row_nw = groups_nw.filter(nw.col("_bucket_id") == bid)
            if len(row_nw) == 0:
                return None
            scope_key = row_nw.select(enc_scope_cols).row(0)

        if scope_key in self._scope_key_to_parent_id:
            pid = self._scope_key_to_parent_id[scope_key]
            self._bucket_to_parent_id[bid] = pid
            self._parent_to_buckets[pid].append(bid)
            return pid

        pid = max(self._parent_time_grids.keys()) + 1 if self._parent_time_grids else 0
        dtype = (
            next(iter(self._parent_time_grids.values())).dtype
            if self._parent_time_grids
            else self.time.dtype
        )
        self._parent_time_grids[pid] = np.array([], dtype=dtype)
        self._bucket_to_parent_id[bid] = pid
        self._parent_to_buckets[pid] = [bid]
        self._scope_key_to_parent_id[scope_key] = pid
        return pid

    def _advance_parent_calendars(self, new_ts_val):
        """Advance all parent calendars and sync sibling bucket ordinals.

        When a prediction is appended at ``new_ts_val``, every parent
        calendar grows by that timestamp and ALL sibling buckets under
        each parent update their ``next_time_index_by_bucket`` to the
        new parent grid length.
        """
        if self._parent_time_grids is None or self._parent_to_buckets is None:
            return
        for pid, grid in self._parent_time_grids.items():
            if len(grid) == 0 or new_ts_val > grid[-1]:
                self._parent_time_grids[pid] = np.append(grid, new_ts_val)
            else:
                pos = np.searchsorted(grid, new_ts_val)
                if pos >= len(grid) or grid[pos] != new_ts_val:
                    self._parent_time_grids[pid] = np.insert(grid, pos, new_ts_val)
            new_len = len(self._parent_time_grids[pid])
            for bid in self._parent_to_buckets.get(pid, []):
                self.next_time_index_by_bucket[bid] = new_len

    def append_predictions(self, curr_dates, predictions, n_series):
        new_arr = np.asarray(predictions)
        # normalize the scalar to self.time's dtype: a raw pd.Timestamp would
        # produce object arrays in np.full/np.append, silently degrading
        # self.time and the parent calendars from datetime64 to object
        new_ts_val = np.asarray(curr_dates)[:1].astype(self.time.dtype, copy=False)[0]
        if self.groups is None:
            new_ts = np.full(n_series, new_ts_val, dtype=self.time.dtype)
            new_bid = np.zeros(n_series, dtype=np.int64)
            next_ord = self.next_time_index_by_bucket[0]
            new_ord = np.full(n_series, next_ord, dtype=np.int64)
            new_y = new_arr.astype(float)
            self.time = np.concatenate([self.time, new_ts])
            self.y = np.concatenate([self.y, new_y])
            self.bucket_id = np.concatenate([self.bucket_id, new_bid])
            self.time_index = np.concatenate([self.time_index, new_ord])
            self.next_time_index_by_bucket[0] = next_ord + 1
            if 0 in self._ts_aggs:
                agg = self._ts_aggs[0]
                valid = ~np.isnan(new_y)
                agg.unique_times = np.append(agg.unique_times, next_ord)
                agg.sums = np.append(agg.sums, np.sum(np.where(valid, new_y, 0.0)))
                agg.counts = np.append(agg.counts, np.sum(valid))
                agg.sum_sq = np.append(
                    agg.sum_sq, np.sum(np.where(valid, new_y**2, 0.0))
                )
                valid_vals = new_y[valid]
                agg.mins = np.append(
                    agg.mins, np.min(valid_vals) if len(valid_vals) > 0 else np.nan
                )
                agg.maxs = np.append(
                    agg.maxs, np.max(valid_vals) if len(valid_vals) > 0 else np.nan
                )
        else:
            sort_order = np.argsort(self.series_bucket_id, kind="stable")
            sorted_bids = self.series_bucket_id[sort_order]
            new_ts = np.full(n_series, new_ts_val, dtype=self.time.dtype)
            self.time = np.concatenate([self.time, new_ts[sort_order]])
            self.y = np.concatenate([self.y, new_arr[sort_order].astype(float)])
            self.bucket_id = np.concatenate([self.bucket_id, sorted_bids])
            new_ords = np.array(
                [self.next_time_index_by_bucket[int(gid)] for gid in sorted_bids],
                dtype=np.int64,
            )
            self.time_index = np.concatenate([self.time_index, new_ords])
            for bid in np.unique(sorted_bids):
                bid_int = int(bid)
                new_ord = self.next_time_index_by_bucket[bid_int]
                if bid_int in self._ts_aggs:
                    agg = self._ts_aggs[bid_int]
                    bid_mask = sorted_bids == bid
                    y_bid = new_arr[sort_order][bid_mask].astype(float)
                    valid = ~np.isnan(y_bid)
                    agg.unique_times = np.append(agg.unique_times, new_ord)
                    agg.sums = np.append(agg.sums, np.sum(np.where(valid, y_bid, 0.0)))
                    agg.counts = np.append(agg.counts, np.sum(valid))
                    agg.sum_sq = np.append(
                        agg.sum_sq, np.sum(np.where(valid, y_bid**2, 0.0))
                    )
                    valid_vals = y_bid[valid]
                    agg.mins = np.append(
                        agg.mins, np.min(valid_vals) if len(valid_vals) > 0 else np.nan
                    )
                    agg.maxs = np.append(
                        agg.maxs, np.max(valid_vals) if len(valid_vals) > 0 else np.nan
                    )
            if self._parent_time_grids is not None:
                self._advance_parent_calendars(new_ts_val)
            else:
                for bid in np.unique(sorted_bids):
                    self.next_time_index_by_bucket[int(bid)] += 1

    def append_observations(
        self,
        df,
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        static_features=None,
    ):
        if self.groups is None:
            new_df = df[_dedupe_preserve_order([id_col, time_col, target_col])]
            new_df = ufp.sort(new_df, by=[time_col, id_col])
            new_df = ufp.drop_index_if_pandas(new_df)
            new_ts = new_df[time_col].to_numpy()
            new_y = new_df[target_col].to_numpy().astype(float)
            new_bid = np.zeros(len(new_df), dtype=np.int64)
            old_ts = self.time
            all_ts = np.concatenate([old_ts, new_ts])
            unique_all = np.unique(all_ts)
            old_idx = np.searchsorted(unique_all, old_ts).astype(np.int64)
            new_idx = np.searchsorted(unique_all, new_ts).astype(np.int64)
            self.time = all_ts
            self.y = np.concatenate([self.y, new_y])
            self.bucket_id = np.concatenate([self.bucket_id, new_bid])
            self.time_index = np.concatenate([old_idx, new_idx])
            self.next_time_index_by_bucket[0] = len(unique_all)
            self._ts_aggs = _build_ts_aggs(self.bucket_id, self.time_index, self.y)
            old_len = len(self.bucket_df)
            new_df_nw = nw.from_native(new_df)
            new_rows_nw = new_df_nw.with_row_index(name="_bucket_pos").with_columns(
                (nw.col("_bucket_pos") + old_len).cast(nw.Int64).alias("_bucket_pos")
            )
            new_rows = nw.to_native(new_rows_nw)
            cols = list(self.bucket_df.columns)
            self.bucket_df = ufp.vertical_concat([self.bucket_df, new_rows[cols]])
        else:
            group_cols_list = self.key_cols or self.group_cols
            assert group_cols_list is not None
            keep_cols = _dedupe_preserve_order(
                [id_col] + group_cols_list + [time_col, target_col]
            )
            bucket_df = df[keep_cols]
            bucket_df = ufp.sort(bucket_df, by=group_cols_list + [time_col, id_col])
            bucket_df = ufp.drop_index_if_pandas(bucket_df)
            old_uids = self.group_uids
            groups = self.groups
            bucket_df = _attach_bucket_id(bucket_df, groups, group_cols_list)
            bucket_df, groups = _extend_groups(bucket_df, groups, group_cols_list)
            self.groups = groups
            # match_if_categorical normalizes (possibly categorical) bucket ids
            # against the existing registry; new_ids feeds new_bid below.
            _, new_ids = ufp.match_if_categorical(old_uids, bucket_df["_bucket_id"])
            bucket_df = ufp.assign_columns(bucket_df, "_bucket_id", new_ids)
            bucket_df = ufp.sort(bucket_df, by=["_bucket_id", time_col, id_col])
            values = bucket_df[target_col].to_numpy().astype(ga_data_dtype, copy=False)
            new_ts = bucket_df[time_col].to_numpy()
            new_y = values.astype(float)
            new_bid = bucket_df["_bucket_id"].to_numpy().astype(np.int64)
            self.time = np.concatenate([self.time, new_ts])
            self.y = np.concatenate([self.y, new_y])
            self.bucket_id = np.concatenate([self.bucket_id, new_bid])
            all_ts = self.time
            all_bid = self.bucket_id
            if (
                self._parent_time_grids is not None
                and self._bucket_to_parent_id is not None
                and self._parent_to_buckets is not None
            ):
                groups_nw = nw.from_native(self.groups)
                for bid in np.unique(new_bid):
                    bid_int = int(bid)
                    if bid_int not in self._bucket_to_parent_id:
                        self._resolve_parent_for_bucket(bid_int, groups_nw=groups_nw)
                affected_parents: set[int] = set()
                for bid in np.unique(new_bid):
                    pid = self._bucket_to_parent_id.get(int(bid))
                    if pid is not None:
                        affected_parents.add(pid)
                for pid in affected_parents:
                    sibling_bids = self._parent_to_buckets.get(pid, [])
                    all_new_for_parent = []
                    for sbid in sibling_bids:
                        mask = new_bid == sbid
                        if mask.any():
                            all_new_for_parent.append(new_ts[mask])
                    if all_new_for_parent:
                        combined_new = np.concatenate(all_new_for_parent)
                        grid = self._parent_time_grids[pid]
                        self._parent_time_grids[pid] = np.unique(
                            np.concatenate([grid, combined_new])
                        )
                per_bucket_grids = {
                    bid: self._parent_time_grids[self._bucket_to_parent_id[bid]]
                    for bid in self._bucket_to_parent_id
                }
                new_ord_arr, new_next = _compute_time_index_from_parent(
                    all_bid, all_ts, per_bucket_grids
                )
            else:
                new_ord_arr, new_next = _compute_time_index(all_bid, all_ts)
            self.time_index = new_ord_arr
            self.next_time_index_by_bucket = new_next
            self._ts_aggs = _build_ts_aggs(self.bucket_id, self.time_index, self.y)
            old_len = len(self.bucket_df)
            bucket_df_nw = nw.from_native(bucket_df)
            new_rows_nw = bucket_df_nw.with_row_index(name="_bucket_pos").with_columns(
                (nw.col("_bucket_pos") + old_len).cast(nw.Int64).alias("_bucket_pos")
            )
            new_rows = nw.to_native(new_rows_nw)
            cols = list(self.bucket_df.columns)
            self.bucket_df = ufp.vertical_concat([self.bucket_df, new_rows[cols]])
            if self._idsorted_to_bucket_pos is not None:
                self._idsorted_to_bucket_pos = _compute_idsorted_to_bucket_pos(
                    self.bucket_df,
                    id_col,
                    time_col,
                )
            if static_features is not None:
                lookup_cols = self.key_cols or group_cols_list
                sf_cols = set(static_features.columns)
                if all(c in sf_cols for c in lookup_cols):
                    self.series_bucket_id = lookup_bucket_ids(
                        static_features, groups, lookup_cols
                    ).astype(np.int64, copy=False)

    def build_query_arrays(self, _curr_dates, n_series):
        if self.groups is None:
            next_ord = self.next_time_index_by_bucket[0]
            tmp_y = np.concatenate(
                [
                    self.y,
                    np.full(n_series, np.nan),
                ]
            )
            tmp_bid = np.concatenate(
                [
                    self.bucket_id,
                    np.zeros(n_series, dtype=np.int64),
                ]
            )
            tmp_ord = np.concatenate(
                [
                    self.time_index,
                    np.full(n_series, next_ord, dtype=np.int64),
                ]
            )
            return tmp_bid, tmp_ord, tmp_y
        else:
            sort_order = np.argsort(self.series_bucket_id, kind="stable")
            new_bids = self.series_bucket_id[sort_order]
            new_ords = np.array(
                [self.next_time_index_by_bucket[int(gid)] for gid in new_bids],
                dtype=np.int64,
            )
            tmp_y = np.concatenate(
                [
                    self.y,
                    np.full(n_series, np.nan),
                ]
            )
            tmp_bid = np.concatenate([self.bucket_id, new_bids])
            tmp_ord = np.concatenate([self.time_index, new_ords])
            return tmp_bid, tmp_ord, tmp_y

    def trim_to_last(self, n_ordinals: int) -> None:
        """Drop history so each calendar keeps only its last ``n_ordinals`` ordinals.

        Equivalent, by construction, to fitting on only the observations whose
        parent-calendar ordinal falls in the last ``n_ordinals`` positions: the
        flat arrays, ``bucket_df`` and parent grids are trimmed and renumbered
        in lockstep, then every derived structure (``_ts_aggs``,
        ``_idsorted_to_bucket_pos``) is regenerated through the same primitives
        the constructors/append paths use, so all representations stay mutually
        consistent. The caller must pass an ``n_ordinals`` covering every
        transform's window (see the retention rule in ``TimeSeries._transform``),
        so the dropped prefix can never enter a window and the trim is
        prediction-neutral.

        Only valid at fit time, before any prediction/observation append, while
        ``bucket_df`` is still positionally aligned with the flat arrays.

        In every mode ``next_time_index_by_bucket[bid]`` is the length of that
        bucket's calendar (global: distinct global timestamps; groupby: the
        bucket's own distinct timestamps; partition: the shared parent grid), so
        the per-bucket cutoff ``len - n_ordinals`` is uniform across the buckets
        that share a calendar. Renumbering by subtracting the cutoff matches a
        fresh ``searchsorted`` into the retained-suffix calendar, which is also
        what the next ``append_observations`` recomputes.
        """
        if n_ordinals <= 0:
            return
        bid_arr = self.bucket_id
        # Per-row cutoff in a single grouping pass instead of one boolean scan
        # per bucket (the old ``for bid: bid_arr == bid`` loop was O(buckets x
        # rows)). ``inv`` maps each row to its bucket's slot in the sorted
        # ``uniq``; the cutoff is uniform within a bucket, so a gather assigns
        # it per row.
        uniq, inv = np.unique(bid_arr, return_inverse=True)
        next_vals = np.array(
            [self.next_time_index_by_bucket[int(b)] for b in uniq], dtype=np.int64
        )
        cutoff_by_uniq = np.maximum(next_vals - n_ordinals, 0)
        cutoff = cutoff_by_uniq[inv]
        keep = self.time_index >= cutoff
        if keep.all():
            # every calendar already fits in n_ordinals -> nothing to drop.
            return
        new_time_index = self.time_index - cutoff
        self.bucket_id = bid_arr[keep]
        self.time = self.time[keep]
        self.y = self.y[keep]
        self.time_index = new_time_index[keep]
        # bucket_df is row-aligned with the flat arrays at fit time, so the same
        # boolean mask trims it consistently.
        self.bucket_df = ufp.filter_with_mask(self.bucket_df, keep)
        if self._parent_time_grids is not None:
            for pid, grid in self._parent_time_grids.items():
                self._parent_time_grids[pid] = grid[max(len(grid) - n_ordinals, 0) :]
        # A trim drops a whole prefix of ordinals per bucket (``keep`` is
        # ``ord >= cutoff``, uniform within a bucket), so every surviving
        # timestamp group is untouched: its sums/counts/sum_sq/min/max are
        # byte-identical to before and only the ordinal shifts down by
        # ``cutoff``. So slice each aggregate to its surviving suffix rather
        # than re-aggregating from raw ``y`` -- the dominant cost of a trim on
        # large panels. Buckets whose suffix is empty lost all their rows and
        # are dropped, matching ``_build_ts_aggs`` on the filtered rows.
        # The cache is complete whenever it is non-empty (every constructor
        # fills it for all buckets); a cleared cache (slow-path ``_transform``)
        # has no suffixes to slice, so rebuild it from the trimmed arrays.
        if self._ts_aggs:
            new_aggs: Dict[int, _TimestampAggregates] = {}
            for pos, b in enumerate(uniq):
                cut = int(cutoff_by_uniq[pos])
                bid_int = int(b)
                agg = self._ts_aggs[bid_int]
                if cut == 0:
                    new_aggs[bid_int] = agg
                    continue
                m = agg.unique_times >= cut
                if not m.any():
                    continue
                new_aggs[bid_int] = _TimestampAggregates(
                    unique_times=agg.unique_times[m] - cut,
                    sums=agg.sums[m],
                    counts=agg.counts[m],
                    sum_sq=agg.sum_sq[m],
                    mins=agg.mins[m],
                    maxs=agg.maxs[m],
                )
            self._ts_aggs = new_aggs
        else:
            self._ts_aggs = _build_ts_aggs(self.bucket_id, self.time_index, self.y)
        for bid_int, cal_len in self.next_time_index_by_bucket.items():
            self.next_time_index_by_bucket[bid_int] = min(cal_len, n_ordinals)
        if self._idsorted_to_bucket_pos is not None:
            id_col, time_col = self.join_cols
            self._idsorted_to_bucket_pos = _compute_idsorted_to_bucket_pos(
                self.bucket_df, id_col, time_col
            )


def _attach_bucket_id(bucket_df, groups, group_cols_list):
    joined = _null_equal_left_join(
        nw.from_native(bucket_df),
        nw.from_native(groups),
        group_cols_list,
        ["_bucket_id"],
    )
    return nw.to_native(joined)


def _extend_groups(bucket_df, groups, group_cols_list):
    bucket_df_nw = nw.from_native(bucket_df)
    groups_nw = nw.from_native(groups)
    missing = bucket_df_nw.get_column("_bucket_id").is_null()
    if missing.any():
        enc_cols = [f"__enc_{c}" for c in group_cols_list]
        # Create new buckets on the *encoded* key so null/NaN/None collapse into a
        # single new bucket (polars .unique() keeps them as distinct raw rows).
        new_groups_nw = (
            _encode_keys(bucket_df_nw.filter(missing), group_cols_list)
            .select(group_cols_list + enc_cols)
            .unique(subset=enc_cols, keep="first", maintain_order=True)
            .drop(enc_cols)
        )
        start = len(groups_nw)
        new_groups_nw = new_groups_nw.with_row_index(name="_bucket_id").with_columns(
            (nw.col("_bucket_id") + start).cast(nw.Int64).alias("_bucket_id")
        )
        groups = ufp.vertical_concat(
            [
                nw.to_native(groups_nw),
                nw.to_native(new_groups_nw),
            ]
        )
        # Re-lookup every row (including pre-existing missing buckets) so an
        # existing null bucket is matched, not duplicated.
        joined = _null_equal_left_join(
            bucket_df_nw.drop("_bucket_id"),
            nw.from_native(groups),
            group_cols_list,
            ["_bucket_id"],
        )
        bucket_df = nw.to_native(joined)
    return bucket_df, groups


def compute_pooled_features(
    state: PooledState,
    transforms: Dict[str, _BaseLagTransform],
    query_arrays=None,
) -> Dict[str, np.ndarray]:
    if query_arrays is not None:
        bid_arr, idx_arr, y_arr = query_arrays
        ts_aggs = _build_ts_aggs(bid_arr, idx_arr, y_arr)
    else:
        bid_arr = state.bucket_id
        idx_arr = state.time_index
        y_arr = state.y
        ts_aggs = state._ts_aggs
    bucket_vals: Dict[str, np.ndarray] = {}
    for name, tfm in transforms.items():
        computed = tfm._compute_bucket_feature(
            bid_arr,
            idx_arr,
            y_arr,
            _ts_aggs=ts_aggs,
        )
        if computed is None:
            raise NotImplementedError(
                f"Transform {type(tfm).__name__!r} does not support pooled "
                f"(global/groupby/partition_by) computation. Implement "
                f"_compute_bucket_feature to use it with global_, groupby, "
                f"or partition_by."
            )
        bucket_vals[name] = computed
    return bucket_vals
