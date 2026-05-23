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
            "CASE WHEN w_cnt >= {min_samples} AND w_cnt > 0"
            " THEN w_min ELSE NULL END"
        ),
        "window_type": "rolling",
    },
    "RollingMax": {
        "agg_cols": "MAX(y) OVER w AS w_max, COUNT(y) OVER w AS w_cnt",
        "result_expr": (
            "CASE WHEN w_cnt >= {min_samples} AND w_cnt > 0"
            " THEN w_max ELSE NULL END"
        ),
        "window_type": "rolling",
    },
    "ExpandingMean": {
        "agg_cols": "SUM(y) OVER w AS w_sum, COUNT(y) OVER w AS w_cnt",
        "result_expr": (
            "CASE WHEN w_cnt > 0"
            " THEN w_sum * 1.0 / w_cnt ELSE NULL END"
        ),
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


def _load_to_sqlite(df: pd.DataFrame, conn: sqlite3.Connection, group_cols: Optional[List[str]] = None):
    cols = ["unique_id", "ds", "y"]
    for gc in group_cols or []:
        if gc not in cols:
            cols.append(gc)
    sub = df[cols].copy()
    sub["y"] = sub["y"].astype(float)
    sub.to_sql("obs", conn, index=False, if_exists="replace")


def _build_window_clause(window_type: str, lag: int, window_size: Optional[int], partition_expr: Optional[str]) -> str:
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


def _build_sql(
    template: dict,
    lag: int,
    window_size: Optional[int],
    min_samples: Optional[int],
    partition_expr: Optional[str],
) -> str:
    window_clause = _build_window_clause(
        template["window_type"], lag, window_size, partition_expr,
    )
    dense_rank_partition = f"PARTITION BY {partition_expr} " if partition_expr else ""
    result_expr = template["result_expr"]
    if min_samples is not None:
        result_expr = result_expr.format(min_samples=min_samples)
    return (
        f"WITH base AS ("
        f"  SELECT *,"
        f"    DENSE_RANK() OVER ({dense_rank_partition}ORDER BY ds) - 1 AS ord"
        f"  FROM obs"
        f"),"
        f" aggs AS ("
        f"  SELECT *, {template['agg_cols']}"
        f"  FROM base"
        f"  {window_clause}"
        f")"
        f" SELECT unique_id, ds, {result_expr} AS result"
        f" FROM aggs"
        f" ORDER BY unique_id, ds"
    )


def sqlite_oracle(
    df: pd.DataFrame,
    transform_name: str,
    lag: int,
    window_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    group_cols: Optional[List[str]] = None,
) -> np.ndarray:
    template = _SQL_TEMPLATES[transform_name]
    if template["window_type"] == "rolling":
        effective_min = min_samples if min_samples is not None else window_size
    else:
        effective_min = None
    partition_expr = ", ".join(group_cols) if group_cols else None
    sql = _build_sql(template, lag, window_size, effective_min, partition_expr)
    conn = sqlite3.connect(":memory:")
    try:
        _load_to_sqlite(df, conn, group_cols)
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
) -> np.ndarray:
    ts = TimeSeries(freq=1, lag_transforms={lag: [transform]})
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
    window_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    atol: float = 1e-10,
):
    sql_result = sqlite_oracle(
        df, transform_name, lag,
        window_size=window_size,
        min_samples=min_samples,
        group_cols=group_cols,
    )
    np_result = _numpy_result(df, transform, lag, group_cols)
    np.testing.assert_allclose(np_result, sql_result, atol=atol, equal_nan=True)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _global_df():
    return pd.DataFrame({
        "unique_id": ["a"] * 8 + ["b"] * 8,
        "ds": list(range(8)) * 2,
        "y": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0,
              2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
    })


def _groupby_df():
    return pd.DataFrame({
        "unique_id": ["a"] * 8 + ["b"] * 8 + ["c"] * 8 + ["d"] * 8,
        "ds": list(range(8)) * 4,
        "y": [
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0,
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
            5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0,
        ],
        "brand": ["X"] * 16 + ["Y"] * 16,
    })


# ---------------------------------------------------------------------------
# Parametric tests
# ---------------------------------------------------------------------------

_WINDOW_SIZE = 3

_TRANSFORMS = [
    (lambda gc: RollingMean(_WINDOW_SIZE, groupby=gc) if gc else RollingMean(_WINDOW_SIZE, global_=True), "RollingMean"),
    (lambda gc: RollingStd(_WINDOW_SIZE, groupby=gc) if gc else RollingStd(_WINDOW_SIZE, global_=True), "RollingStd"),
    (lambda gc: RollingMin(_WINDOW_SIZE, groupby=gc) if gc else RollingMin(_WINDOW_SIZE, global_=True), "RollingMin"),
    (lambda gc: RollingMax(_WINDOW_SIZE, groupby=gc) if gc else RollingMax(_WINDOW_SIZE, global_=True), "RollingMax"),
    (lambda gc: ExpandingMean(groupby=gc) if gc else ExpandingMean(global_=True), "ExpandingMean"),
    (lambda gc: ExpandingStd(groupby=gc) if gc else ExpandingStd(global_=True), "ExpandingStd"),
    (lambda gc: ExpandingMin(groupby=gc) if gc else ExpandingMin(global_=True), "ExpandingMin"),
    (lambda gc: ExpandingMax(groupby=gc) if gc else ExpandingMax(global_=True), "ExpandingMax"),
]


@pytest.mark.parametrize("transform_factory,transform_name", _TRANSFORMS, ids=[t[1] for t in _TRANSFORMS])
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
        df, transform, transform_name, lag,
        group_cols=group_cols,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_staggered_start():
    df = pd.DataFrame({
        "unique_id": ["a", "a", "a", "a", "b", "b", "b"],
        "ds": [0, 1, 2, 3, 1, 2, 3],
        "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
    })
    transform = RollingMean(2, global_=True)
    assert_oracle_matches(
        df, transform, "RollingMean", lag=1, window_size=2,
    )


def test_single_series():
    df = pd.DataFrame({
        "unique_id": ["a"] * 6,
        "ds": list(range(6)),
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    transform = RollingMean(3, global_=True)
    assert_oracle_matches(
        df, transform, "RollingMean", lag=1, window_size=3,
    )


def test_identical_values_std_zero():
    df = pd.DataFrame({
        "unique_id": ["a"] * 8 + ["b"] * 8,
        "ds": list(range(8)) * 2,
        "y": [5.0] * 16,
    })
    transform = RollingStd(3, global_=True)
    assert_oracle_matches(
        df, transform, "RollingStd", lag=1, window_size=3,
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
            df, transform, transform_name, lag=2,
            window_size=window_size,
        )


@pytest.mark.parametrize("transform_name,tfm_factory", [
    ("RollingMean", lambda gc: RollingMean(_WINDOW_SIZE, groupby=gc)),
    ("RollingStd", lambda gc: RollingStd(_WINDOW_SIZE, groupby=gc)),
])
def test_multi_column_groupby(transform_name, tfm_factory):
    df = pd.DataFrame({
        "unique_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6 + ["d"] * 6,
        "ds": list(range(6)) * 4,
        "y": [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ],
        "brand": ["X"] * 12 + ["Y"] * 12,
        "region": (["north"] * 6 + ["south"] * 6) * 2,
    })
    group_cols = ["brand", "region"]
    transform = tfm_factory(group_cols)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    assert_oracle_matches(
        df, transform, transform_name, lag=1,
        group_cols=group_cols,
        window_size=window_size,
    )


@pytest.mark.parametrize("min_samples", [1, 2, 5])
def test_custom_min_samples(min_samples):
    df = _global_df()
    transform = RollingMean(_WINDOW_SIZE, min_samples=min_samples, global_=True)
    assert_oracle_matches(
        df, transform, "RollingMean", lag=1,
        window_size=_WINDOW_SIZE,
        min_samples=min_samples,
    )


def test_custom_min_samples_std():
    df = _global_df()
    transform = RollingStd(_WINDOW_SIZE, min_samples=1, global_=True)
    assert_oracle_matches(
        df, transform, "RollingStd", lag=1,
        window_size=_WINDOW_SIZE,
        min_samples=1,
    )


def test_sparse_windows_nan_vs_value():
    """High lag + small window → many rows produce NaN, a few produce values.
    Validates that SQLite NULL ↔ NumPy NaN alignment is correct across
    the boundary."""
    df = pd.DataFrame({
        "unique_id": ["a"] * 10 + ["b"] * 10,
        "ds": list(range(10)) * 2,
        "y": [float(i) for i in range(10)] + [float(i * 10) for i in range(10)],
    })
    transform = RollingMean(2, min_samples=2, global_=True)
    assert_oracle_matches(
        df, transform, "RollingMean", lag=3,
        window_size=2, min_samples=2,
    )
    transform_std = RollingStd(2, min_samples=2, global_=True)
    assert_oracle_matches(
        df, transform_std, "RollingStd", lag=3,
        window_size=2, min_samples=2,
    )
