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
    key_cols: List[str],
):
    cols = ["unique_id", "ds", "y"]
    cols.extend(c for c in key_cols if c not in cols)
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
    time_agg: Optional[str] = None,
    scope_cols: Optional[List[str]] = None,
) -> str:
    window_clause = _build_window_clause(
        template["window_type"],
        lag,
        window_size,
        partition_expr,
    )
    result_expr = template["result_expr"]
    if min_samples is not None:
        result_expr = result_expr.format(min_samples=min_samples)
    # single CTE list with an optional collapsed stage, so the ordinal and
    # window definitions are shared verbatim between the two variants
    ctes = [
        "base AS ("
        "  SELECT *,"
        "    DENSE_RANK() OVER (ORDER BY ds) - 1 AS ord"
        "  FROM obs"
        ")"
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
    key_cols: List[str],
    window_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    time_agg: Optional[str] = None,
) -> np.ndarray:
    template = _SQL_TEMPLATES[transform_name]
    if template["window_type"] == "rolling":
        effective_min = min_samples if min_samples is not None else window_size
    else:
        effective_min = None
    sql = _build_sql(
        template,
        lag,
        window_size,
        effective_min,
        partition_expr=", ".join(key_cols) or None,
        time_agg=time_agg,
        scope_cols=key_cols,
    )
    conn = sqlite3.connect(":memory:")
    try:
        _load_to_sqlite(df, conn, key_cols)
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
    static_features: List[str],
) -> np.ndarray:
    ts = TimeSeries(freq=1, lag_transforms={lag: [transform]})
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


def _scope(key: Optional[List[str]]) -> dict:
    return {"partition_by": key} if key else {"global_": True}


def assert_oracle_matches(
    df: pd.DataFrame,
    transform,
    transform_name: str,
    lag: int,
    window_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    time_agg: Optional[str] = None,
    atol: float = 1e-10,
):
    key_cols = list(transform.partition_by or [])
    # mlforecast defaults min_samples to 1 (instead of window_size) when the id
    # is in the key, where the RANGE window spans calendar steps but only
    # same-partition observations count. Derived here independently so the
    # oracle pins that default.
    if min_samples is None and window_size is not None and "unique_id" in key_cols:
        min_samples = 1
    sql_result = sqlite_oracle(
        df,
        transform_name,
        lag,
        key_cols,
        window_size=window_size,
        min_samples=min_samples,
        time_agg=time_agg,
    )
    # key columns constant within every series are declared static
    constant = df.groupby("unique_id")[
        [c for c in key_cols if c != "unique_id"]
    ].nunique(dropna=False)
    static_features = [c for c in constant.columns if (constant[c] == 1).all()]
    np_result = _numpy_result(df, transform, lag, static_features)
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


def _static_key_df():
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
    (lambda key: RollingMean(_WINDOW_SIZE, **_scope(key)), "RollingMean"),
    (lambda key: RollingStd(_WINDOW_SIZE, **_scope(key)), "RollingStd"),
    (lambda key: RollingMin(_WINDOW_SIZE, **_scope(key)), "RollingMin"),
    (lambda key: RollingMax(_WINDOW_SIZE, **_scope(key)), "RollingMax"),
    (lambda key: ExpandingMean(**_scope(key)), "ExpandingMean"),
    (lambda key: ExpandingStd(**_scope(key)), "ExpandingStd"),
    (lambda key: ExpandingMin(**_scope(key)), "ExpandingMin"),
    (lambda key: ExpandingMax(**_scope(key)), "ExpandingMax"),
]


@pytest.mark.parametrize(
    "transform_factory,transform_name", _TRANSFORMS, ids=[t[1] for t in _TRANSFORMS]
)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("mode", ["global", "static_key"])
def test_sqlite_oracle_matches_numpy(transform_factory, transform_name, lag, mode):
    if mode == "global":
        df = _global_df()
        gc_arg = None
    else:
        df = _static_key_df()
        gc_arg = ["brand"]
    transform = transform_factory(gc_arg)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    assert_oracle_matches(
        df,
        transform,
        transform_name,
        lag,
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
        ("RollingMean", lambda gc: RollingMean(_WINDOW_SIZE, partition_by=gc)),
        ("RollingStd", lambda gc: RollingStd(_WINDOW_SIZE, partition_by=gc)),
    ],
)
def test_multi_column_static_key(transform_name, tfm_factory):
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


def _dynamic_key_df():
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


def _static_dynamic_key_df():
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


def _id_key_df():
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

_PARTITION_TRANSFORMS = _TRANSFORMS


@pytest.mark.parametrize(
    "transform_factory,transform_name",
    _PARTITION_TRANSFORMS,
    ids=[t[1] for t in _PARTITION_TRANSFORMS],
)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize(
    "mode", ["dynamic_key", "static_dynamic_key", "id_key"]
)
def test_sqlite_oracle_partition_by(transform_factory, transform_name, lag, mode):
    pc = ["promo"]
    if mode == "dynamic_key":
        df = _dynamic_key_df()
        transform = transform_factory(pc)
    elif mode == "static_dynamic_key":
        df = _static_dynamic_key_df()
        group_cols = ["brand"]
        transform = transform_factory([*group_cols, *pc])
    else:
        df = _id_key_df()
        transform = transform_factory(["unique_id", *pc])
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        assert_oracle_matches(
            df,
            transform,
            transform_name,
            lag,
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
        transform = transform_factory(["promo"])
        window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            assert_oracle_matches(
                df,
                transform,
                transform_name,
                lag=2,
                window_size=window_size,
            )


# Long sparse partitions: a bucket observed a handful of times over a long
# calendar. The window then spans many more cells than it holds observations,
# which is the regime where a count recovered as ``k * mean(count)`` can land
# just below an integer and blank out a real value.

#: spans the 49-ordinal gap below, so the window holds 49 cells and 1-2 rows
_SPARSE_WINDOW = 49

_SPARSE_TRANSFORMS = [
    (RollingMean, "RollingMean"),
    (RollingStd, "RollingStd"),
    (RollingMin, "RollingMin"),
    (RollingMax, "RollingMax"),
    (ExpandingMean, "ExpandingMean"),
    (ExpandingStd, "ExpandingStd"),
    (ExpandingMin, "ExpandingMin"),
    (ExpandingMax, "ExpandingMax"),
]


def _sparse_long_partition_df():
    """Two series over 60 steps; ``promo=1`` only at 4 widely spaced ordinals."""
    n_times = 60
    rare = {0, 1, 49, 55}
    promo = [1 if t in rare else 0 for t in range(n_times)]
    return pd.DataFrame(
        {
            "unique_id": ["a"] * n_times + ["b"] * n_times,
            "ds": list(range(n_times)) * 2,
            "y": [float(t) for t in range(n_times)]
            + [float(10 * t + 1) for t in range(n_times)],
            "promo": promo * 2,
        }
    )


@pytest.mark.parametrize(
    "transform_cls,transform_name",
    _SPARSE_TRANSFORMS,
    ids=[t[1] for t in _SPARSE_TRANSFORMS],
)
@pytest.mark.parametrize("min_samples", [1, 2])
@pytest.mark.parametrize("mode", ["dynamic_key", "id_key"])
def test_sqlite_oracle_sparse_long_partition(
    transform_cls, transform_name, min_samples, mode
):
    is_rolling = transform_name.startswith("Rolling")
    if not is_rolling and min_samples != 1:
        pytest.skip("expanding transforms take no min_samples")
    if mode == "dynamic_key":
        kwargs: dict = {"partition_by": ["promo"]}
    else:
        kwargs = {"partition_by": ["unique_id", "promo"]}
    if is_rolling:
        transform = transform_cls(_SPARSE_WINDOW, min_samples=min_samples, **kwargs)
        window_size: Optional[int] = _SPARSE_WINDOW
    else:
        transform = transform_cls(**kwargs)
        window_size = None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        assert_oracle_matches(
            _sparse_long_partition_df(),
            transform,
            transform_name,
            lag=1,
            window_size=window_size,
            min_samples=min_samples if is_rolling else None,
        )


# NULL partition keys: SQL PARTITION BY collapses all NULLs into one partition,
# and our pooled state must do the same. to_sql maps NaN/None -> SQL NULL, so the
# oracle locks the SQL NULL-partition semantics for partition_by transforms.


def _dynamic_key_null_df():
    """Dynamic key where the partition column has NaN (-> one bucket)."""
    df = _dynamic_key_df()
    # float promo with NaN scattered across both series at the same timestamps.
    df["promo"] = pd.Series([0.0, np.nan, 1.0, np.nan, 0.0, 1.0, 0.0, np.nan] * 2)
    return df


def _static_dynamic_key_null_df():
    """Static + dynamic key where BOTH the static and the dynamic column are
    NULL on some rows.

    Exercises (a) null-scope parent-calendar resolution: series b and c share a
    NULL brand, so they must resolve to the *same* parent calendar, and (b) a NULL
    partition (promo) value collapsing into one bucket within each scope — i.e.
    SQL PARTITION BY (brand, promo) with NULLs treated as equal.
    """
    df = _static_dynamic_key_df()
    df["brand"] = pd.Series(["X"] * 8 + [None] * 8 + [None] * 8 + ["Y"] * 8)
    df["promo"] = pd.Series([0.0, np.nan, 1.0, 1.0, np.nan, 1.0, 0.0, np.nan] * 4)
    return df


def _id_key_null_df():
    """Id key where the partition column has NaN (-> own bucket)."""
    df = _id_key_df()
    df["promo"] = pd.Series([0.0, np.nan, 1.0, np.nan, 0.0, 1.0, 0.0, np.nan] * 2)
    return df


@pytest.mark.parametrize(
    "transform_factory,transform_name",
    _PARTITION_TRANSFORMS,
    ids=[t[1] for t in _PARTITION_TRANSFORMS],
)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize(
    "mode", ["dynamic_key", "static_dynamic_key", "id_key"]
)
def test_sqlite_oracle_partition_null_key(transform_factory, transform_name, lag, mode):
    pc = ["promo"]
    if mode == "dynamic_key":
        df = _dynamic_key_null_df()
        transform = transform_factory(pc)
    elif mode == "static_dynamic_key":
        df = _static_dynamic_key_null_df()
        group_cols = ["brand"]
        transform = transform_factory([*group_cols, *pc])
    else:
        df = _id_key_null_df()
        transform = transform_factory(["unique_id", *pc])
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        assert_oracle_matches(
            df,
            transform,
            transform_name,
            lag,
            window_size=window_size,
        )


# Fractional-float partition keys: a dynamic partition value of 0.1/0.25/0.5 must
# bucket the same way SQL `PARTITION BY <REAL>` groups it. Exercises the float->string
# encoding path end to end against the oracle (string-keyed buckets vs SQL float
# equality). One representative transform (RollingMean) is enough — the windowing
# numerics are covered elsewhere; this only checks float key bucketing.
_DISCOUNT_PATTERN = [0.1, 0.1, 0.25, 0.5, 0.1, 0.25, 0.1, 0.5]  # 8 per series


@pytest.mark.parametrize(
    "mode", ["dynamic_key", "static_dynamic_key", "id_key"]
)
def test_sqlite_oracle_partition_fractional_float(mode):
    pc = ["discount"]
    if mode == "dynamic_key":
        df = _dynamic_key_df()
        df["discount"] = _DISCOUNT_PATTERN * 2
        transform = RollingMean(_WINDOW_SIZE, partition_by=pc)
    elif mode == "static_dynamic_key":
        df = _static_dynamic_key_df()  # static `brand` key column
        df["discount"] = _DISCOUNT_PATTERN * 4
        group_cols = ["brand"]
        transform = RollingMean(_WINDOW_SIZE, partition_by=[*group_cols, *pc])
    else:
        df = _id_key_df()
        df["discount"] = _DISCOUNT_PATTERN * 2
        transform = RollingMean(_WINDOW_SIZE, partition_by=["unique_id", *pc])
    assert_oracle_matches(
        df,
        transform,
        "RollingMean",
        lag=1,
        window_size=_WINDOW_SIZE,
    )


# ---------------------------------------------------------------------------
# NULL static keys (SQL collapses all NULLs into one partition; our pooled
# state must do the same). The oracle harness is pandas->SQLite, where to_sql
# maps NaN/None -> SQL NULL, so it locks the SQL NULL-partition semantics.
# (Polars-engine-origin NaN behavior is exercised in tests/test_pooled.py.)
# ---------------------------------------------------------------------------


def _static_key_string_null_df():
    """String static key where two series share a NULL brand (-> one bucket)."""
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


def _static_key_numeric_null_df():
    """Numeric static key with NaN (float) shared by two series (-> one bucket)."""
    df = _static_key_string_null_df()
    df["brand"] = pd.Series([1.0] * 8 + [np.nan] * 8 + [np.nan] * 8 + [2.0] * 8)
    return df


@pytest.mark.parametrize(
    "transform_factory,transform_name", _TRANSFORMS, ids=[t[1] for t in _TRANSFORMS]
)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize(
    "df_factory",
    [_static_key_string_null_df, _static_key_numeric_null_df],
    ids=["string-null", "numeric-null"],
)
def test_sqlite_oracle_static_null_key(
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
        window_size=window_size,
    )


@pytest.mark.parametrize(
    "transform_name,tfm_factory",
    [
        ("RollingMean", lambda gc: RollingMean(_WINDOW_SIZE, partition_by=gc)),
        ("RollingStd", lambda gc: RollingStd(_WINDOW_SIZE, partition_by=gc)),
    ],
)
def test_multi_column_static_key_one_null_col(transform_name, tfm_factory):
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
        lambda key, ta: RollingMean(_WINDOW_SIZE, time_agg=ta, **_scope(key)),
        "RollingMean",
    ),
    (
        lambda key, ta: RollingStd(_WINDOW_SIZE, time_agg=ta, **_scope(key)),
        "RollingStd",
    ),
    (
        lambda key, ta: RollingMin(_WINDOW_SIZE, time_agg=ta, **_scope(key)),
        "RollingMin",
    ),
    (
        lambda key, ta: RollingMax(_WINDOW_SIZE, time_agg=ta, **_scope(key)),
        "RollingMax",
    ),
    (lambda key, ta: ExpandingMean(time_agg=ta, **_scope(key)), "ExpandingMean"),
    (lambda key, ta: ExpandingStd(time_agg=ta, **_scope(key)), "ExpandingStd"),
    (lambda key, ta: ExpandingMin(time_agg=ta, **_scope(key)), "ExpandingMin"),
    (lambda key, ta: ExpandingMax(time_agg=ta, **_scope(key)), "ExpandingMax"),
]


@pytest.mark.parametrize(
    "transform_factory,transform_name",
    _TIME_AGG_TRANSFORMS,
    ids=[t[1] for t in _TIME_AGG_TRANSFORMS],
)
@pytest.mark.parametrize("time_agg", _TIME_AGGS)
@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("mode", ["global", "static_key"])
def test_sqlite_oracle_time_agg(transform_factory, transform_name, time_agg, lag, mode):
    # NOTE: the public fit path rejects NaN targets, so the all-NaN-timestamp /
    # SQL-NULL conventions can't be driven through this oracle; they are pinned
    # directly against hand-computed values in tests/test_pooled.py.
    if mode == "global":
        df = _global_df()
        gc_arg = None
    else:
        df = _static_key_df()
        gc_arg = ["brand"]
    transform = transform_factory(gc_arg, time_agg)
    window_size = _WINDOW_SIZE if transform_name.startswith("Rolling") else None
    assert_oracle_matches(
        df,
        transform,
        transform_name,
        lag,
        window_size=window_size,
        time_agg=time_agg,
    )


@pytest.mark.parametrize("time_agg", _TIME_AGGS)
@pytest.mark.parametrize("mode", ["dynamic_key", "static_dynamic_key"])
def test_sqlite_oracle_time_agg_partition(time_agg, mode):
    pc = ["promo"]
    if mode == "dynamic_key":
        df = _dynamic_key_df()
        transform = RollingMean(
            _WINDOW_SIZE, partition_by=pc, time_agg=time_agg
        )
    else:
        df = _static_dynamic_key_df()
        group_cols = ["brand"]
        transform = RollingMean(
            _WINDOW_SIZE, partition_by=[*group_cols, *pc], time_agg=time_agg
        )
    assert_oracle_matches(
        df,
        transform,
        "RollingMean",
        lag=1,
        window_size=_WINDOW_SIZE,
        time_agg=time_agg,
    )
