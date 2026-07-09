import sqlite3
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from mlforecast.core import TimeSeries
from mlforecast.lag_transforms import (
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingStd,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingStd,
)

# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_SQL_TEMPLATES = {
    "RollingMean": {
        "agg_cols": "SUM(y) OVER w AS w_sum, COUNT(y) OVER w AS w_cnt",
        "result_expr": (
            "CASE WHEN w_cnt >= {min_samples} AND w_cnt > 0"
            " THEN w_sum * 1.0 / w_cnt ELSE NULL END"
        ),
        "window_type": "rolling",
    },
    "RollingStd": {
        "agg_cols": (
            "SUM(y) OVER w AS w_sum,"
            " SUM(y * y) OVER w AS w_sum_sq,"
            " COUNT(y) OVER w AS w_cnt"
        ),
        "result_expr": (
            "CASE WHEN w_cnt >= {min_samples} AND w_cnt > 1"
            " THEN SQRT(MAX((w_sum_sq - w_sum * w_sum * 1.0 / w_cnt)"
            " / (w_cnt - 1), 0.0))"
            " ELSE NULL END"
        ),
        "window_type": "rolling",
    },
    "RollingMin": {
        "agg_cols": "MIN(y) OVER w AS w_min, COUNT(y) OVER w AS w_cnt",
        "result_expr": (
            "CASE WHEN w_cnt >= {min_samples} AND w_cnt > 0 THEN w_min ELSE NULL END"
        ),
        "window_type": "rolling",
    },
    "RollingMax": {
        "agg_cols": "MAX(y) OVER w AS w_max, COUNT(y) OVER w AS w_cnt",
        "result_expr": (
            "CASE WHEN w_cnt >= {min_samples} AND w_cnt > 0 THEN w_max ELSE NULL END"
        ),
        "window_type": "rolling",
    },
    "ExpandingMean": {
        "agg_cols": "SUM(y) OVER w AS w_sum, COUNT(y) OVER w AS w_cnt",
        "result_expr": ("CASE WHEN w_cnt > 0 THEN w_sum * 1.0 / w_cnt ELSE NULL END"),
        "window_type": "expanding",
    },
    "ExpandingStd": {
        "agg_cols": (
            "SUM(y) OVER w AS w_sum,"
            " SUM(y * y) OVER w AS w_sum_sq,"
            " COUNT(y) OVER w AS w_cnt"
        ),
        "result_expr": (
            "CASE WHEN w_cnt > 1"
            " THEN SQRT(MAX((w_sum_sq - w_sum * w_sum * 1.0 / w_cnt)"
            " / (w_cnt - 1), 0.0))"
            " ELSE NULL END"
        ),
        "window_type": "expanding",
    },
    "ExpandingMin": {
        "agg_cols": "MIN(y) OVER w AS w_min, COUNT(y) OVER w AS w_cnt",
        "result_expr": "CASE WHEN w_cnt > 0 THEN w_min ELSE NULL END",
        "window_type": "expanding",
    },
    "ExpandingMax": {
        "agg_cols": "MAX(y) OVER w AS w_max, COUNT(y) OVER w AS w_cnt",
        "result_expr": "CASE WHEN w_cnt > 0 THEN w_max ELSE NULL END",
        "window_type": "expanding",
    },
}

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def _load_to_sqlite(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    group_cols: Optional[List[str]] = None,
    partition_cols: Optional[List[str]] = None,
):
    cols = ["unique_id", "ds", "y"]
    for gc in group_cols or []:
        if gc not in cols:
            cols.append(gc)
    for pc in partition_cols or []:
        if pc not in cols:
            cols.append(pc)
    sub = df[cols].copy()
    sub["y"] = sub["y"].astype(float)
    sub.to_sql("obs", conn, index=False, if_exists="replace")


def _build_window_clause(
    window_type: str,
    lag: int,
    window_size: Optional[int],
    partition_expr: Optional[str],
) -> str:
    parts = []
    if partition_expr:
        parts.append(f"PARTITION BY {partition_expr}")
    parts.append("ORDER BY ord")
    if window_type == "rolling":
        assert window_size is not None
        lower = lag + window_size - 1
        upper = lag
        parts.append(f"RANGE BETWEEN {lower} PRECEDING AND {upper} PRECEDING")
    else:
        parts.append(f"RANGE BETWEEN UNBOUNDED PRECEDING AND {lag} PRECEDING")
    return "WINDOW w AS (" + " ".join(parts) + ")"


_TIME_AGG_SQL_FN = {
    "sum": "SUM",
    "count": "COUNT",
    "mean": "AVG",
    "min": "MIN",
    "max": "MAX",
}


def _build_sql(
    template: dict,
    lag: int,
    window_size: Optional[int],
    min_samples: Optional[int],
    partition_expr: Optional[str],
    ordinal_partition_expr: Optional[str] = None,
    time_agg: Optional[str] = None,
    scope_cols: Optional[List[str]] = None,
) -> str:
    window_clause = _build_window_clause(
        template["window_type"],
        lag,
        window_size,
        partition_expr,
    )
    dense_rank_partition = (
        f"PARTITION BY {ordinal_partition_expr} " if ordinal_partition_expr else ""
    )
    result_expr = template["result_expr"]
    if min_samples is not None:
        result_expr = result_expr.format(min_samples=min_samples)
    # single CTE list with an optional collapsed stage, so the ordinal and
    # window definitions are shared verbatim between the two variants
    ctes = [
        f"base AS ("
        f"  SELECT *,"
        f"    DENSE_RANK() OVER ({dense_rank_partition}ORDER BY ds) - 1 AS ord"
        f"  FROM obs"
        f")"
    ]
    source = "base"
    scope_cols = scope_cols or []
    if time_agg is not None:
        # collapse all rows sharing a (bucket, ordinal) to one per-timestamp
        # aggregate and run the window over that collapsed series
        agg_fn = _TIME_AGG_SQL_FN[time_agg]
        scope_select = "".join(f"{c}, " for c in scope_cols)
        group_by_cols = ", ".join(scope_cols + ["ord"])
        ctes.append(
            f" collapsed AS ("
            f"  SELECT {scope_select}ord, {agg_fn}(y) AS y"
            f"  FROM base"
            f"  GROUP BY {group_by_cols}"
            f")"
        )
        source = "collapsed"
    ctes.append(
        f" aggs AS (  SELECT *, {template['agg_cols']}  FROM {source}  {window_clause})"
    )
    with_block = "WITH " + ",".join(ctes)
    if time_agg is None:
        return (
            f"{with_block}"
            f" SELECT unique_id, ds, {result_expr} AS result"
            f" FROM aggs"
            f" ORDER BY unique_id, ds"
        )
    # LEFT JOIN the windowed per-timestamp result back onto every original
    # (unique_id, ds) row via the scope columns + ord so the value broadcasts
    # to each row (otherwise the oracle would return one row per timestamp,
    # not per obs).
    join_cond = " AND ".join(f"b.{c} IS a.{c}" for c in scope_cols + ["ord"])
    return (
        f"{with_block}"
        f" SELECT b.unique_id, b.ds, {result_expr} AS result"
        f" FROM base b"
        f" LEFT JOIN aggs a ON {join_cond}"
        f" ORDER BY b.unique_id, b.ds"
    )


def sqlite_oracle(
    df: pd.DataFrame,
    transform_name: str,
    lag: int,
    window_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    group_cols: Optional[List[str]] = None,
    partition_cols: Optional[List[str]] = None,
    time_agg: Optional[str] = None,
) -> np.ndarray:
    template = _SQL_TEMPLATES[transform_name]
    if template["window_type"] == "rolling":
        effective_min = min_samples if min_samples is not None else window_size
    else:
        effective_min = None
    ordinal_scope = ", ".join(group_cols) if group_cols else None
    all_window_cols = list(group_cols or []) + list(partition_cols or [])
    window_partition = ", ".join(all_window_cols) if all_window_cols else None
    sql = _build_sql(
        template,
        lag,
        window_size,
        effective_min,
        partition_expr=window_partition,
        ordinal_partition_expr=ordinal_scope,
        time_agg=time_agg,
        scope_cols=all_window_cols,
    )
    conn = sqlite3.connect(":memory:")
    try:
        _load_to_sqlite(df, conn, group_cols, partition_cols)
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    sql_df = pd.DataFrame(rows, columns=["unique_id", "ds", "result"])
    merged = df[["unique_id", "ds"]].merge(sql_df, on=["unique_id", "ds"], how="left")
    result = merged["result"].to_numpy(dtype=float)
    result[pd.isna(merged["result"])] = np.nan
    return result


# ---------------------------------------------------------------------------
# NumPy side via public API
# ---------------------------------------------------------------------------


def _numpy_result(
    df: pd.DataFrame,
    transform,
    lag: int,
    group_cols: Optional[List[str]] = None,
    partition_cols: Optional[List[str]] = None,
) -> np.ndarray:
    ts = TimeSeries(freq=1, lag_transforms={lag: [transform]})
    static_features: Optional[List[str]]
    if partition_cols is not None:
        static_features = [c for c in (group_cols or []) if c != "unique_id"]
        if not static_features:
            static_features = []
    else:
        static_features = list(group_cols) if group_cols else None
    result_df = ts.fit_transform(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        dropna=False,
        static_features=static_features,
    )
    assert isinstance(result_df, pd.DataFrame)
    col_name = transform._get_name(lag)
    merged = df[["unique_id", "ds"]].merge(
        result_df[["unique_id", "ds", col_name]],
        on=["unique_id", "ds"],
        how="left",
    )
    return merged[col_name].to_numpy(dtype=float)


def assert_oracle_matches(
    df: pd.DataFrame,
    transform,
    transform_name: str,
    lag: int,
    group_cols: Optional[List[str]] = None,
    partition_cols: Optional[List[str]] = None,
    window_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    time_agg: Optional[str] = None,
    atol: float = 1e-10,
):
    # mlforecast defaults min_samples to 1 (instead of window_size) in local
    # partition mode, where the RANGE window spans calendar steps but only
    # same-partition observations count. Derived here independently so the
    # oracle pins that default.
    if min_samples is None and window_size is not None:
        is_local_partition = bool(
            getattr(transform, "partition_by", None)
            and not getattr(transform, "global_", False)
            and not getattr(transform, "groupby", None)
        )
        if is_local_partition:
            min_samples = 1
    sql_result = sqlite_oracle(
        df,
        transform_name,
        lag,
        window_size=window_size,
        min_samples=min_samples,
        group_cols=group_cols,
        partition_cols=partition_cols,
        time_agg=time_agg,
    )
    np_result = _numpy_result(df, transform, lag, group_cols, partition_cols)
    np.testing.assert_allclose(np_result, sql_result, atol=atol, equal_nan=True)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _global_df():
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8,
            "ds": list(range(8)) * 2,
            "y": [
                1.0,
                3.0,
                5.0,
                7.0,
                9.0,
                11.0,
                13.0,
                15.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
            ],
        }
    )


def _groupby_df():
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8 + ["c"] * 8 + ["d"] * 8,
            "ds": list(range(8)) * 4,
            "y": [
                1.0,
                3.0,
                5.0,
                7.0,
                9.0,
                11.0,
                13.0,
                15.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                5.0,
                15.0,
                25.0,
                35.0,
                45.0,
                55.0,
                65.0,
                75.0,
            ],
            "brand": ["X"] * 16 + ["Y"] * 16,
        }
    )


# ---------------------------------------------------------------------------
# Parametric tests
# ---------------------------------------------------------------------------

_WINDOW_SIZE = 3

_TRANSFORMS = [
    (
        lambda gc: (
            RollingMean(_WINDOW_SIZE, groupby=gc)
            if gc
            else RollingMean(_WINDOW_SIZE, global_=True)
        ),
        "RollingMean",
    ),
    (
        lambda gc: (
            RollingStd(_WINDOW_SIZE, groupby=gc)
            if gc
            else RollingStd(_WINDOW_SIZE, global_=True)
        ),
        "RollingStd",
    ),
    (
        lambda gc: (
            RollingMin(_WINDOW_SIZE, groupby=gc)
            if gc
            else RollingMin(_WINDOW_SIZE, global_=True)
        ),
        "RollingMin",
    ),
    (
        lambda gc: (
            RollingMax(_WINDOW_SIZE, groupby=gc)
            if gc
            else RollingMax(_WINDOW_SIZE, global_=True)
        ),
        "RollingMax",
    ),
    (
        lambda gc: ExpandingMean(groupby=gc) if gc else ExpandingMean(global_=True),
        "ExpandingMean",
    ),
    (
        lambda gc: ExpandingStd(groupby=gc) if gc else ExpandingStd(global_=True),
        "ExpandingStd",
    ),
    (
        lambda gc: ExpandingMin(groupby=gc) if gc else ExpandingMin(global_=True),
        "ExpandingMin",
    ),
    (
        lambda gc: ExpandingMax(groupby=gc) if gc else ExpandingMax(global_=True),
        "ExpandingMax",
    ),
]


@pytest.mark.parametrize(
    "transform_factory,transform_name", _TRANSFORMS, ids=[t[1] for t in _TRANSFORMS]
)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("mode", ["global", "groupby"])
def test_sqlite_oracle_matches_numpy(transform_factory, transform_name, lag, mode):
    if mode == "global":
        df = _global_df()
        group_cols = None
        gc_arg = None
    else:
        df = _groupby_df()
        group_cols = ["brand"]
        gc_arg = ["brand"]
    transform = transform_factory(gc_arg)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    assert_oracle_matches(
        df,
        transform,
        transform_name,
        lag,
        group_cols=group_cols,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_staggered_start():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "a", "b", "b", "b"],
            "ds": [0, 1, 2, 3, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
        }
    )
    transform = RollingMean(2, global_=True)
    assert_oracle_matches(
        df,
        transform,
        "RollingMean",
        lag=1,
        window_size=2,
    )


def test_single_series():
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 6,
            "ds": list(range(6)),
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    transform = RollingMean(3, global_=True)
    assert_oracle_matches(
        df,
        transform,
        "RollingMean",
        lag=1,
        window_size=3,
    )


def test_identical_values_std_zero():
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8,
            "ds": list(range(8)) * 2,
            "y": [5.0] * 16,
        }
    )
    transform = RollingStd(3, global_=True)
    assert_oracle_matches(
        df,
        transform,
        "RollingStd",
        lag=1,
        window_size=3,
    )


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_random_data(seed):
    rng = np.random.default_rng(seed)
    n_series = 10
    n_times = 15
    ids = []
    ds_vals = []
    y_vals = []
    for i in range(n_series):
        uid = f"s{i}"
        ids.extend([uid] * n_times)
        ds_vals.extend(range(n_times))
        y_vals.extend(rng.standard_normal(n_times).tolist())
    df = pd.DataFrame({"unique_id": ids, "ds": ds_vals, "y": y_vals})
    for transform_factory, transform_name in _TRANSFORMS:
        transform = transform_factory(None)
        window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
        assert_oracle_matches(
            df,
            transform,
            transform_name,
            lag=2,
            window_size=window_size,
        )


@pytest.mark.parametrize(
    "transform_name,tfm_factory",
    [
        ("RollingMean", lambda gc: RollingMean(_WINDOW_SIZE, groupby=gc)),
        ("RollingStd", lambda gc: RollingStd(_WINDOW_SIZE, groupby=gc)),
    ],
)
def test_multi_column_groupby(transform_name, tfm_factory):
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6 + ["d"] * 6,
            "ds": list(range(6)) * 4,
            "y": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
            ],
            "brand": ["X"] * 12 + ["Y"] * 12,
            "region": (["north"] * 6 + ["south"] * 6) * 2,
        }
    )
    group_cols = ["brand", "region"]
    transform = tfm_factory(group_cols)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    assert_oracle_matches(
        df,
        transform,
        transform_name,
        lag=1,
        group_cols=group_cols,
        window_size=window_size,
    )


@pytest.mark.parametrize("min_samples", [1, 2, 5])
def test_custom_min_samples(min_samples):
    df = _global_df()
    transform = RollingMean(_WINDOW_SIZE, min_samples=min_samples, global_=True)
    assert_oracle_matches(
        df,
        transform,
        "RollingMean",
        lag=1,
        window_size=_WINDOW_SIZE,
        min_samples=min_samples,
    )


def test_custom_min_samples_std():
    df = _global_df()
    transform = RollingStd(_WINDOW_SIZE, min_samples=1, global_=True)
    assert_oracle_matches(
        df,
        transform,
        "RollingStd",
        lag=1,
        window_size=_WINDOW_SIZE,
        min_samples=1,
    )


def test_sparse_windows_nan_vs_value():
    """High lag + small window → many rows produce NaN, a few produce values.
    Validates that SQLite NULL ↔ NumPy NaN alignment is correct across
    the boundary."""
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 10 + ["b"] * 10,
            "ds": list(range(10)) * 2,
            "y": [float(i) for i in range(10)] + [float(i * 10) for i in range(10)],
        }
    )
    transform = RollingMean(2, min_samples=2, global_=True)
    assert_oracle_matches(
        df,
        transform,
        "RollingMean",
        lag=3,
        window_size=2,
        min_samples=2,
    )
    transform_std = RollingStd(2, min_samples=2, global_=True)
    assert_oracle_matches(
        df,
        transform_std,
        "RollingStd",
        lag=3,
        window_size=2,
        min_samples=2,
    )


# ---------------------------------------------------------------------------
# partition_by tests
# ---------------------------------------------------------------------------


def _global_partition_df():
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8,
            "ds": list(range(8)) * 2,
            "y": [
                1.0,
                3.0,
                5.0,
                7.0,
                9.0,
                11.0,
                13.0,
                15.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
            ],
            "promo": [0, 0, 1, 1, 0, 1, 0, 1] * 2,
        }
    )


def _groupby_partition_df():
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8 + ["c"] * 8 + ["d"] * 8,
            "ds": list(range(8)) * 4,
            "y": [
                1.0,
                3.0,
                5.0,
                7.0,
                9.0,
                11.0,
                13.0,
                15.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                5.0,
                15.0,
                25.0,
                35.0,
                45.0,
                55.0,
                65.0,
                75.0,
            ],
            "brand": ["X"] * 16 + ["Y"] * 16,
            "promo": [0, 0, 1, 1, 0, 1, 0, 1] * 4,
        }
    )


def _local_partition_df():
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8,
            "ds": list(range(8)) * 2,
            "y": [
                1.0,
                3.0,
                5.0,
                7.0,
                9.0,
                11.0,
                13.0,
                15.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
            ],
            "promo": [0, 0, 1, 1, 0, 1, 0, 1] * 2,
        }
    )


import warnings as _warnings

_PARTITION_TRANSFORMS = [
    (
        lambda mode, gc, pc: (
            RollingMean(_WINDOW_SIZE, **{mode: True, "partition_by": pc})
            if mode == "global_"
            else RollingMean(_WINDOW_SIZE, groupby=gc, partition_by=pc)
            if gc
            else RollingMean(_WINDOW_SIZE, partition_by=pc)
        ),
        "RollingMean",
    ),
    (
        lambda mode, gc, pc: (
            RollingStd(_WINDOW_SIZE, **{mode: True, "partition_by": pc})
            if mode == "global_"
            else RollingStd(_WINDOW_SIZE, groupby=gc, partition_by=pc)
            if gc
            else RollingStd(_WINDOW_SIZE, partition_by=pc)
        ),
        "RollingStd",
    ),
    (
        lambda mode, gc, pc: (
            RollingMin(_WINDOW_SIZE, **{mode: True, "partition_by": pc})
            if mode == "global_"
            else RollingMin(_WINDOW_SIZE, groupby=gc, partition_by=pc)
            if gc
            else RollingMin(_WINDOW_SIZE, partition_by=pc)
        ),
        "RollingMin",
    ),
    (
        lambda mode, gc, pc: (
            RollingMax(_WINDOW_SIZE, **{mode: True, "partition_by": pc})
            if mode == "global_"
            else RollingMax(_WINDOW_SIZE, groupby=gc, partition_by=pc)
            if gc
            else RollingMax(_WINDOW_SIZE, partition_by=pc)
        ),
        "RollingMax",
    ),
    (
        lambda mode, gc, pc: (
            ExpandingMean(**{mode: True, "partition_by": pc})
            if mode == "global_"
            else ExpandingMean(groupby=gc, partition_by=pc)
            if gc
            else ExpandingMean(partition_by=pc)
        ),
        "ExpandingMean",
    ),
    (
        lambda mode, gc, pc: (
            ExpandingStd(**{mode: True, "partition_by": pc})
            if mode == "global_"
            else ExpandingStd(groupby=gc, partition_by=pc)
            if gc
            else ExpandingStd(partition_by=pc)
        ),
        "ExpandingStd",
    ),
    (
        lambda mode, gc, pc: (
            ExpandingMin(**{mode: True, "partition_by": pc})
            if mode == "global_"
            else ExpandingMin(groupby=gc, partition_by=pc)
            if gc
            else ExpandingMin(partition_by=pc)
        ),
        "ExpandingMin",
    ),
    (
        lambda mode, gc, pc: (
            ExpandingMax(**{mode: True, "partition_by": pc})
            if mode == "global_"
            else ExpandingMax(groupby=gc, partition_by=pc)
            if gc
            else ExpandingMax(partition_by=pc)
        ),
        "ExpandingMax",
    ),
]


@pytest.mark.parametrize(
    "transform_factory,transform_name",
    _PARTITION_TRANSFORMS,
    ids=[t[1] for t in _PARTITION_TRANSFORMS],
)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize(
    "mode", ["global_partition", "groupby_partition", "local_partition"]
)
def test_sqlite_oracle_partition_by(transform_factory, transform_name, lag, mode):
    pc = ["promo"]
    if mode == "global_partition":
        df = _global_partition_df()
        group_cols = None
        transform = transform_factory("global_", None, pc)
    elif mode == "groupby_partition":
        df = _groupby_partition_df()
        group_cols = ["brand"]
        transform = transform_factory("groupby", group_cols, pc)
    else:
        df = _local_partition_df()
        group_cols = ["unique_id"]
        transform = transform_factory("local", None, pc)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        assert_oracle_matches(
            df,
            transform,
            transform_name,
            lag,
            group_cols=group_cols,
            partition_cols=pc,
            window_size=window_size,
        )


@pytest.mark.parametrize("seed", [42, 123])
def test_random_partition_data(seed):
    rng = np.random.default_rng(seed)
    n_series = 8
    n_times = 12
    ids = np.repeat([f"s{i}" for i in range(n_series)], n_times)
    ds_vals = np.tile(range(n_times), n_series)
    y_vals = rng.standard_normal(n_series * n_times)
    promo = np.tile(rng.choice([0, 1], size=n_times), n_series)
    df = pd.DataFrame(
        {
            "unique_id": ids,
            "ds": ds_vals,
            "y": y_vals,
            "promo": promo,
        }
    )
    for transform_factory, transform_name in _PARTITION_TRANSFORMS:
        transform = transform_factory("global_", None, ["promo"])
        window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            assert_oracle_matches(
                df,
                transform,
                transform_name,
                lag=2,
                group_cols=None,
                partition_cols=["promo"],
                window_size=window_size,
            )


# NULL partition keys: SQL PARTITION BY collapses all NULLs into one partition,
# and our pooled state must do the same. to_sql maps NaN/None -> SQL NULL, so the
# oracle locks the SQL NULL-partition semantics for partition_by transforms.


def _global_partition_null_df():
    """global + partition_by where the partition key has NaN (-> one bucket)."""
    df = _global_partition_df()
    # float promo with NaN scattered across both series at the same timestamps.
    df["promo"] = pd.Series([0.0, np.nan, 1.0, np.nan, 0.0, 1.0, 0.0, np.nan] * 2)
    return df


def _groupby_partition_null_df():
    """groupby + partition_by where BOTH the group key and the partition key are
    NULL on some rows.

    Exercises (a) null-scope parent-calendar resolution: series b and c share a
    NULL brand, so they must resolve to the *same* parent calendar, and (b) a NULL
    partition (promo) value collapsing into one bucket within each scope — i.e.
    SQL PARTITION BY (brand, promo) with NULLs treated as equal.
    """
    df = _groupby_partition_df()
    df["brand"] = pd.Series(["X"] * 8 + [None] * 8 + [None] * 8 + ["Y"] * 8)
    df["promo"] = pd.Series([0.0, np.nan, 1.0, 1.0, np.nan, 1.0, 0.0, np.nan] * 4)
    return df


def _local_partition_null_df():
    """local + partition_by where the partition key has NaN (-> own bucket)."""
    df = _local_partition_df()
    df["promo"] = pd.Series([0.0, np.nan, 1.0, np.nan, 0.0, 1.0, 0.0, np.nan] * 2)
    return df


@pytest.mark.parametrize(
    "transform_factory,transform_name",
    _PARTITION_TRANSFORMS,
    ids=[t[1] for t in _PARTITION_TRANSFORMS],
)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize(
    "mode", ["global_partition", "groupby_partition", "local_partition"]
)
def test_sqlite_oracle_partition_null_key(transform_factory, transform_name, lag, mode):
    pc = ["promo"]
    if mode == "global_partition":
        df = _global_partition_null_df()
        group_cols = None
        transform = transform_factory("global_", None, pc)
    elif mode == "groupby_partition":
        df = _groupby_partition_null_df()
        group_cols = ["brand"]
        transform = transform_factory("groupby", group_cols, pc)
    else:
        df = _local_partition_null_df()
        group_cols = ["unique_id"]
        transform = transform_factory("local", None, pc)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        assert_oracle_matches(
            df,
            transform,
            transform_name,
            lag,
            group_cols=group_cols,
            partition_cols=pc,
            window_size=window_size,
        )


# Fractional-float partition keys: a dynamic partition value of 0.1/0.25/0.5 must
# bucket the same way SQL `PARTITION BY <REAL>` groups it. Exercises the float->string
# encoding path end to end against the oracle (string-keyed buckets vs SQL float
# equality). One representative transform (RollingMean) is enough — the windowing
# numerics are covered elsewhere; this only checks float key bucketing.
_DISCOUNT_PATTERN = [0.1, 0.1, 0.25, 0.5, 0.1, 0.25, 0.1, 0.5]  # 8 per series


@pytest.mark.parametrize(
    "mode", ["global_partition", "groupby_partition", "local_partition"]
)
def test_sqlite_oracle_partition_fractional_float(mode):
    pc = ["discount"]
    if mode == "global_partition":
        df = _global_partition_df()
        df["discount"] = _DISCOUNT_PATTERN * 2
        group_cols = None
        transform = RollingMean(_WINDOW_SIZE, global_=True, partition_by=pc)
    elif mode == "groupby_partition":
        df = _groupby_partition_df()  # static `brand` group key
        df["discount"] = _DISCOUNT_PATTERN * 4
        group_cols = ["brand"]
        transform = RollingMean(_WINDOW_SIZE, groupby=group_cols, partition_by=pc)
    else:
        df = _local_partition_df()
        df["discount"] = _DISCOUNT_PATTERN * 2
        group_cols = ["unique_id"]
        transform = RollingMean(_WINDOW_SIZE, partition_by=pc)
    assert_oracle_matches(
        df,
        transform,
        "RollingMean",
        lag=1,
        group_cols=group_cols,
        partition_cols=pc,
        window_size=_WINDOW_SIZE,
    )


# ---------------------------------------------------------------------------
# NULL groupby keys (SQL collapses all NULLs into one partition; our pooled
# state must do the same). The oracle harness is pandas->SQLite, where to_sql
# maps NaN/None -> SQL NULL, so it locks the SQL NULL-partition semantics.
# (Polars-engine-origin NaN behavior is exercised in tests/test_pooled.py.)
# ---------------------------------------------------------------------------


def _groupby_string_null_df():
    """String groupby key where two series share a NULL brand (-> one bucket)."""
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8 + ["c"] * 8 + ["d"] * 8,
            "ds": list(range(8)) * 4,
            "y": [
                1.0,
                3.0,
                5.0,
                7.0,
                9.0,
                11.0,
                13.0,
                15.0,
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                5.0,
                15.0,
                25.0,
                35.0,
                45.0,
                55.0,
                65.0,
                75.0,
            ],
            "brand": ["X"] * 8 + [None] * 8 + [None] * 8 + ["Y"] * 8,
        }
    )


def _groupby_numeric_null_df():
    """Numeric groupby key with NaN (float) shared by two series (-> one bucket)."""
    df = _groupby_string_null_df()
    df["brand"] = pd.Series([1.0] * 8 + [np.nan] * 8 + [np.nan] * 8 + [2.0] * 8)
    return df


@pytest.mark.parametrize(
    "transform_factory,transform_name", _TRANSFORMS, ids=[t[1] for t in _TRANSFORMS]
)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize(
    "df_factory",
    [_groupby_string_null_df, _groupby_numeric_null_df],
    ids=["string-null", "numeric-null"],
)
def test_sqlite_oracle_groupby_null_key(
    transform_factory, transform_name, lag, df_factory
):
    df = df_factory()
    group_cols = ["brand"]
    transform = transform_factory(group_cols)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    assert_oracle_matches(
        df,
        transform,
        transform_name,
        lag,
        group_cols=group_cols,
        window_size=window_size,
    )


@pytest.mark.parametrize(
    "transform_name,tfm_factory",
    [
        ("RollingMean", lambda gc: RollingMean(_WINDOW_SIZE, groupby=gc)),
        ("RollingStd", lambda gc: RollingStd(_WINDOW_SIZE, groupby=gc)),
    ],
)
def test_multi_column_groupby_one_null_col(transform_name, tfm_factory):
    """Multi-column key where exactly one column is NULL: (X, NULL) and
    (X, 'north') stay distinct, while two (X, NULL) rows share a bucket."""
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6 + ["d"] * 6,
            "ds": list(range(6)) * 4,
            "y": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
            ],
            "brand": ["X"] * 12 + ["Y"] * 12,
            "region": (["north"] * 6 + [None] * 6) * 2,
        }
    )
    group_cols = ["brand", "region"]
    transform = tfm_factory(group_cols)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    assert_oracle_matches(
        df,
        transform,
        transform_name,
        lag=1,
        group_cols=group_cols,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# time_agg: pre-aggregate rows sharing a timestamp within each bucket, then
# apply the transform over that per-timestamp series (e.g. rolling mean of the
# group's daily sums). The oracle inserts a GROUP BY <scope>, ord CTE before the
# window, so SQLite independently pins the sum/count/mean/min/max semantics and
# the all-NULL-timestamp -> NULL (COUNT -> 0) conventions.
# ---------------------------------------------------------------------------

_TIME_AGGS = ["sum", "count", "mean", "min", "max"]

_TIME_AGG_TRANSFORMS = [
    (
        lambda gc, ta: (
            RollingMean(_WINDOW_SIZE, groupby=gc, time_agg=ta)
            if gc
            else RollingMean(_WINDOW_SIZE, global_=True, time_agg=ta)
        ),
        "RollingMean",
    ),
    (
        lambda gc, ta: (
            RollingStd(_WINDOW_SIZE, groupby=gc, time_agg=ta)
            if gc
            else RollingStd(_WINDOW_SIZE, global_=True, time_agg=ta)
        ),
        "RollingStd",
    ),
    (
        lambda gc, ta: (
            RollingMin(_WINDOW_SIZE, groupby=gc, time_agg=ta)
            if gc
            else RollingMin(_WINDOW_SIZE, global_=True, time_agg=ta)
        ),
        "RollingMin",
    ),
    (
        lambda gc, ta: (
            RollingMax(_WINDOW_SIZE, groupby=gc, time_agg=ta)
            if gc
            else RollingMax(_WINDOW_SIZE, global_=True, time_agg=ta)
        ),
        "RollingMax",
    ),
    (
        lambda gc, ta: (
            ExpandingMean(groupby=gc, time_agg=ta)
            if gc
            else ExpandingMean(global_=True, time_agg=ta)
        ),
        "ExpandingMean",
    ),
    (
        lambda gc, ta: (
            ExpandingStd(groupby=gc, time_agg=ta)
            if gc
            else ExpandingStd(global_=True, time_agg=ta)
        ),
        "ExpandingStd",
    ),
    (
        lambda gc, ta: (
            ExpandingMin(groupby=gc, time_agg=ta)
            if gc
            else ExpandingMin(global_=True, time_agg=ta)
        ),
        "ExpandingMin",
    ),
    (
        lambda gc, ta: (
            ExpandingMax(groupby=gc, time_agg=ta)
            if gc
            else ExpandingMax(global_=True, time_agg=ta)
        ),
        "ExpandingMax",
    ),
]


@pytest.mark.parametrize(
    "transform_factory,transform_name",
    _TIME_AGG_TRANSFORMS,
    ids=[t[1] for t in _TIME_AGG_TRANSFORMS],
)
@pytest.mark.parametrize("time_agg", _TIME_AGGS)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("mode", ["global", "groupby"])
def test_sqlite_oracle_time_agg(transform_factory, transform_name, time_agg, lag, mode):
    # NOTE: the public fit path rejects NaN targets, so the all-NaN-timestamp /
    # SQL-NULL conventions can't be driven through this oracle; they are pinned
    # directly against hand-computed values in tests/test_pooled.py.
    if mode == "global":
        df = _global_df()
        group_cols = None
        gc_arg = None
    else:
        df = _groupby_df()
        group_cols = ["brand"]
        gc_arg = ["brand"]
    transform = transform_factory(gc_arg, time_agg)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    assert_oracle_matches(
        df,
        transform,
        transform_name,
        lag,
        group_cols=group_cols,
        window_size=window_size,
        time_agg=time_agg,
    )


@pytest.mark.parametrize("time_agg", _TIME_AGGS)
@pytest.mark.parametrize("mode", ["global_partition", "groupby_partition"])
def test_sqlite_oracle_time_agg_partition(time_agg, mode):
    pc = ["promo"]
    if mode == "global_partition":
        df = _global_partition_df()
        group_cols = None
        transform = RollingMean(
            _WINDOW_SIZE, global_=True, partition_by=pc, time_agg=time_agg
        )
    else:
        df = _groupby_partition_df()
        group_cols = ["brand"]
        transform = RollingMean(
            _WINDOW_SIZE, groupby=group_cols, partition_by=pc, time_agg=time_agg
        )
    assert_oracle_matches(
        df,
        transform,
        "RollingMean",
        lag=1,
        group_cols=group_cols,
        partition_cols=pc,
        window_size=_WINDOW_SIZE,
        time_agg=time_agg,
    )
