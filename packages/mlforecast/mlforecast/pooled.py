__all__ = ["PooledState", "compute_pooled_features"]

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import narwhals as nw
import numpy as np
import utilsforecast.processing as ufp

from .grouped_array import GroupedArray
from .lag_transforms import _BaseLagTransform


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


def _null_equal_left_join(left_nw, right_nw, cols, payload_cols):
    """Left-join ``left_nw`` to ``right_nw`` on ``cols`` treating missing as equal.

    ``right_nw`` contributes only ``payload_cols`` (e.g. ``["_bucket_id"]``); its
    original key columns are excluded so the join emits no ``*_right`` duplicates.
    Encoded columns are transient and dropped before returning.
    """
    enc_cols = [f"__enc_{c}" for c in cols]
    left_enc, right_enc = _encode_join_keys(left_nw, right_nw, cols)
    right_enc = right_enc.select(enc_cols + list(payload_cols))
    joined = left_enc.join(right_enc, on=enc_cols, how="left")
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
    merged_nw = data_enc.join(
        groups_enc.select(enc_cols + ["_bucket_id"]), on=enc_cols, how="left"
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
    n_rows: np.ndarray
    is_balanced: bool
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
        n_rows = np.bincount(inv, minlength=m).astype(float)
        is_balanced = bool(n_rows.size > 0 and np.all(n_rows == n_rows[0]))
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
            n_rows=n_rows,
            is_balanced=is_balanced,
            sum_sq=sum_sq,
            mins=mins,
            maxs=maxs,
        )
    return aggs


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


@dataclass
class PooledState:
    """Holds all arrays and metadata for a pooled (global/groupby) bucket.

    ``time_index`` is an integer period coordinate over a validated regular
    time grid.  For global/group states (no partition), the input is validated
    to have no missing timestamps within any series.  Different series in the
    same bucket may have different start dates; only actual observations are
    included (no synthetic zeros).  Under this invariant, ``time_index``
    is equivalent to SQL ``RANGE BETWEEN … PRECEDING`` interval semantics.

    For PR 2 / partition_by buckets, ``time_index`` must be derived from
    the parent series/global calendar — not the bucket's observed timestamps —
    so that missing partition values leave holes and window bounds remain
    interval-correct.

    **Ordering contract**: ``bucket_id``, ``time``, ``time_index``, and ``y``
    arrays are positionally aligned — row *i* in each describes the same
    observation.  ``ga`` is **not** positionally aligned with these arrays
    after mutations (``append_predictions`` / ``append_observations``):
    ``ga.append_several`` interleaves new values within each group, while the
    flat arrays append at the tail.  ``ga`` is only used for
    ``_initialize_lag_transform_states`` (called once before any mutations);
    all feature computation uses the flat arrays via ``compute_pooled_features``.
    """

    ga: GroupedArray
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
        y_raw = global_df[target_col].to_numpy().astype(ga_data_dtype)
        global_df_nw = nw.from_native(global_df)
        global_df_nw = global_df_nw.with_row_index(name="_bucket_pos").with_columns(
            nw.col("_bucket_pos").cast(nw.Int64)
        )
        process_df_nw = global_df_nw.select(
            [
                nw.lit(0).cast(nw.Int64).alias("_bucket_id"),
                "_bucket_pos",
                target_col,
            ]
        )
        global_df = nw.to_native(global_df_nw)
        process_df = nw.to_native(process_df_nw)
        processed = ufp.process_df(
            process_df,
            id_col="_bucket_id",
            time_col="_bucket_pos",
            target_col=target_col,
        )
        ga = GroupedArray(processed.data[:, 0], processed.indptr)
        unique_ts = np.unique(ts_raw)
        ord_raw = np.searchsorted(unique_ts, ts_raw).astype(np.int64)
        bid_arr = np.zeros(len(global_df), dtype=np.int64)
        y_float = y_raw.astype(float)
        return cls(
            ga=ga,
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
        y_raw = bucket_df[target_col].to_numpy().astype(ga_data_dtype)
        bid_raw = bucket_df["_bucket_id"].to_numpy()
        bucket_df_nw = nw.from_native(bucket_df)
        bucket_df_nw = bucket_df_nw.with_columns(
            (nw.col("_bucket_id").cum_count() - 1).cast(nw.Int64).alias("_global_idx")
        )
        group_starts = bucket_df_nw.group_by("_bucket_id").agg(
            nw.col("_global_idx").min().alias("_group_start")
        )
        bucket_df_nw = (
            bucket_df_nw.join(group_starts, on="_bucket_id", how="left")
            .with_columns(
                (nw.col("_global_idx") - nw.col("_group_start"))
                .cast(nw.Int64)
                .alias("_bucket_pos")
            )
            .drop(["_global_idx", "_group_start"])
        )
        process_df_nw = bucket_df_nw.select(["_bucket_id", "_bucket_pos", target_col])
        bucket_df = nw.to_native(bucket_df_nw)
        process_df = nw.to_native(process_df_nw)
        processed = ufp.process_df(
            process_df,
            id_col="_bucket_id",
            time_col="_bucket_pos",
            target_col=target_col,
        )
        if processed.sort_idxs is not None:
            bucket_df = ufp.take_rows(bucket_df, processed.sort_idxs)
            ts_raw = ts_raw[processed.sort_idxs]
            y_raw = y_raw[processed.sort_idxs]
            bid_raw = bid_raw[processed.sort_idxs]
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        ga = GroupedArray(processed.data[:, 0], processed.indptr)
        bid_arr = bid_raw.astype(np.int64)
        ord_arr, next_by_bucket = _compute_time_index(bid_arr, ts_raw)
        series_bucket_id = lookup_bucket_ids(
            static_features, groups, group_cols_list
        ).astype(np.int64, copy=False)
        y_float = y_raw.astype(float)
        return cls(
            ga=ga,
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

    def append_predictions(self, curr_dates, predictions, n_series):
        new_arr = np.asarray(predictions)
        new_ts_val = curr_dates[0]
        if self.groups is None:
            new_ts = np.full(n_series, new_ts_val)
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
                nr = float(n_series)
                agg.n_rows = np.append(agg.n_rows, nr)
                agg.is_balanced = agg.is_balanced and (nr == agg.n_rows[0])
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
            new_sizes = np.array([n_series], dtype=np.int32)
            new_values = new_arr.astype(self.ga.data.dtype)
            self.ga = self.ga.append_several(
                new_sizes=new_sizes,
                new_values=new_values,
                new_groups=np.array([False]),
            )
        else:
            sort_order = np.argsort(self.series_bucket_id, kind="stable")
            sorted_bids = self.series_bucket_id[sort_order]
            new_values = new_arr[sort_order].astype(self.ga.data.dtype, copy=False)
            n_groups = len(self.groups)
            new_sizes = np.zeros(n_groups, dtype=np.int32)
            np.add.at(new_sizes, self.series_bucket_id[sort_order], 1)
            self.ga = self.ga.append_several(
                new_sizes=new_sizes,
                new_values=new_values,
                new_groups=np.zeros(n_groups, dtype=bool),
            )
            new_ts = np.full(n_series, new_ts_val)
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
                self.next_time_index_by_bucket[bid_int] += 1
                if bid_int in self._ts_aggs:
                    agg = self._ts_aggs[bid_int]
                    bid_mask = sorted_bids == bid
                    y_bid = new_arr[sort_order][bid_mask].astype(float)
                    valid = ~np.isnan(y_bid)
                    new_ord = self.next_time_index_by_bucket[bid_int] - 1
                    agg.unique_times = np.append(agg.unique_times, new_ord)
                    agg.sums = np.append(agg.sums, np.sum(np.where(valid, y_bid, 0.0)))
                    agg.counts = np.append(agg.counts, np.sum(valid))
                    nr = float(len(y_bid))
                    agg.n_rows = np.append(agg.n_rows, nr)
                    agg.is_balanced = agg.is_balanced and (nr == agg.n_rows[0])
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
            new_values = new_df[target_col].to_numpy().astype(ga_data_dtype)
            new_sizes = np.array([len(new_values)], dtype=np.int32)
            self.ga = self.ga.append_several(
                new_sizes=new_sizes,
                new_values=new_values,
                new_groups=np.array([False]),
            )
            old_len = len(self.bucket_df)
            new_df_nw = nw.from_native(new_df)
            new_rows_nw = new_df_nw.with_row_index(name="_bucket_pos").with_columns(
                (nw.col("_bucket_pos") + old_len).cast(nw.Int64).alias("_bucket_pos")
            )
            new_rows = nw.to_native(new_rows_nw)
            cols = list(self.bucket_df.columns)
            self.bucket_df = ufp.vertical_concat([self.bucket_df, new_rows[cols]])
        else:
            assert self.group_cols is not None
            group_cols_list = self.group_cols
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
            id_counts = ufp.counts_by_id(bucket_df, "_bucket_id")
            uids = old_uids
            uids, new_ids = ufp.match_if_categorical(uids, bucket_df["_bucket_id"])
            bucket_df = ufp.assign_columns(bucket_df, "_bucket_id", new_ids)
            bucket_df = ufp.sort(bucket_df, by=["_bucket_id", time_col, id_col])
            values = bucket_df[target_col].to_numpy().astype(ga_data_dtype, copy=False)
            new_ts = bucket_df[time_col].to_numpy()
            new_y = values.astype(float)
            new_bid = bucket_df["_bucket_id"].to_numpy().astype(np.int64)
            try:
                sizes = ufp.join(uids, id_counts, on="_bucket_id", how="outer_coalesce")
            except (KeyError, ValueError):
                sizes = ufp.join(uids, id_counts, on="_bucket_id", how="outer")
            sizes = ufp.fill_null(sizes, {"counts": 0})
            sizes = ufp.sort(sizes, by="_bucket_id")
            new_groups_mask = ~ufp.is_in(sizes["_bucket_id"], uids)
            self.ga = self.ga.append_several(
                new_sizes=sizes["counts"].to_numpy().astype(np.int32),
                new_values=values,
                new_groups=new_groups_mask.to_numpy(),
            )
            self.time = np.concatenate([self.time, new_ts])
            self.y = np.concatenate([self.y, new_y])
            self.bucket_id = np.concatenate([self.bucket_id, new_bid])
            all_ts = self.time
            all_bid = self.bucket_id
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
            if static_features is not None:
                self.series_bucket_id = lookup_bucket_ids(
                    static_features, groups, group_cols_list
                ).astype(np.int64, copy=False)

    def build_query_arrays(self, curr_dates, n_series):
        if self.groups is None:
            next_ord = self.next_time_index_by_bucket[0]
            tmp_ts = np.concatenate(
                [
                    self.time,
                    np.full(n_series, curr_dates[0]),
                ]
            )
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
            return tmp_bid, tmp_ts, tmp_y, tmp_ord
        else:
            sort_order = np.argsort(self.series_bucket_id, kind="stable")
            new_bids = self.series_bucket_id[sort_order]
            new_ords = np.array(
                [self.next_time_index_by_bucket[int(gid)] for gid in new_bids],
                dtype=np.int64,
            )
            tmp_ts = np.concatenate(
                [
                    self.time,
                    np.full(n_series, curr_dates[0])[sort_order],
                ]
            )
            tmp_y = np.concatenate(
                [
                    self.y,
                    np.full(n_series, np.nan),
                ]
            )
            tmp_bid = np.concatenate([self.bucket_id, new_bids])
            tmp_ord = np.concatenate([self.time_index, new_ords])
            return tmp_bid, tmp_ts, tmp_y, tmp_ord


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
        bid_arr, ts_arr, y_arr, idx_arr = query_arrays
        ts_aggs = _build_ts_aggs(bid_arr, idx_arr, y_arr)
    else:
        bid_arr = state.bucket_id
        ts_arr = state.time
        y_arr = state.y
        idx_arr = state.time_index
        ts_aggs = state._ts_aggs
    bucket_vals: Dict[str, np.ndarray] = {}
    for name, tfm in transforms.items():
        computed = tfm._compute_bucket_feature(
            bid_arr,
            ts_arr,
            y_arr,
            idx_arr,
            _ts_aggs=ts_aggs,
        )
        if computed is None:
            raise NotImplementedError(
                f"Transform {type(tfm).__name__!r} does not support pooled "
                f"(global/groupby) computation. Implement "
                f"_compute_bucket_feature to use it with global_ or groupby."
            )
        bucket_vals[name] = computed
    return bucket_vals
