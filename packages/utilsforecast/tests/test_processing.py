import datetime
import warnings
from datetime import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from conftest import assert_raises_with_message
from packaging.version import Version
from polars import Series as pl_Series

import utilsforecast.processing as ufp
from utilsforecast.compat import POLARS_INSTALLED
from utilsforecast.data import generate_series
from utilsforecast.processing import (
    DataFrameProcessor,
    _find_boundaries,
    _ranges_to_indexer,
    _multiply_pl_freq,
    add_insample_levels,
    anti_join,
    assign_columns,
    backtest_splits,
    between,
    cast,
    cv_times,
    fill_null,
    group_by,
    horizontal_concat,
    is_in,
    is_nan,
    is_nan_or_none,
    is_none,
    make_future_dataframe,
    match_if_categorical,
    offset_times,
    process_df,
    repeat,
    sort,
    take_rows,
    time_ranges,
    to_numpy,
    vertical_concat,
)

if POLARS_INSTALLED:
    import polars as pl
    import polars.testing


@pytest.fixture(params=["pandas"] + (["polars"] if POLARS_INSTALLED else []))
def engine(request):
    return request.param


def test_assign_columns(engine):
    series = generate_series(2, engine=engine)
    x = np.random.rand(series.shape[0])
    series = assign_columns(series, "x", x)
    series = assign_columns(series, ["y", "z"], np.vstack([x, x]).T)
    series = assign_columns(series, "ones", 1)
    series = assign_columns(series, "zeros", np.zeros(series.shape[0]))
    series = assign_columns(series, "as", "a")
    series = assign_columns(series, "bs", series.shape[0] * ["b"])

    # Select and compare data
    if engine == "pandas":
        np.testing.assert_allclose(
            series[["x", "y", "z"]].values, np.vstack([x, x, x]).T
        )
        np.testing.assert_equal(series["ones"], np.ones(series.shape[0]))
        np.testing.assert_equal(series["as"], np.full(series.shape[0], "a"))
        np.testing.assert_equal(series["bs"], np.full(series.shape[0], "b"))
    else:  # polars
        np.testing.assert_allclose(
            series.select(["x", "y", "z"]).to_numpy(), np.vstack([x, x, x]).T
        )
        np.testing.assert_equal(series["ones"].to_numpy(), np.ones(series.shape[0]))
        np.testing.assert_equal(series["as"].to_numpy(), np.full(series.shape[0], "a"))
        np.testing.assert_equal(series["bs"].to_numpy(), np.full(series.shape[0], "b"))


def test_take_rows(engine):
    series = generate_series(2, engine=engine)
    subset = take_rows(series, np.array([0, 2]))
    assert subset.shape[0] == 2


def test_is_nan():
    np.testing.assert_equal(
        is_nan(pd.Series([np.nan, 1.0, None])).to_numpy(),
        np.array([True, False, True]),
    )
    np.testing.assert_equal(
        is_nan(pl.Series([np.nan, 1.0, None])).to_numpy(),
        np.array([True, False, None]),
    )
    for series in (
        pl.Series(["a", None, "b"]),
        pl.Series([True, None, False]),
        pl.Series([1, None, 2]),
        pl.Series(["a", None, "b"], dtype=pl.Categorical),
    ):
        np.testing.assert_equal(
            is_nan(series).to_numpy(),
            np.array([False, False, False]),
        )
        np.testing.assert_equal(
            is_nan_or_none(series).to_numpy(),
            np.array([False, True, False]),
        )


def test_is_none():
    np.testing.assert_equal(
        is_none(pd.Series([np.nan, 1.0, None])).to_numpy(),
        np.array([True, False, True]),
    )
    if POLARS_INSTALLED:
        np.testing.assert_equal(
            is_none(pl.Series([np.nan, 1.0, None])).to_numpy(),
            np.array([False, False, True]),
        )


def test_is_nan_or_none():
    np.testing.assert_equal(
        is_nan_or_none(pd.Series([np.nan, 1.0, None])).to_numpy(),
        np.array([True, False, True]),
    )
    if POLARS_INSTALLED:
        np.testing.assert_equal(
            is_nan_or_none(pl.Series([np.nan, 1.0, None])).to_numpy(),
            np.array([True, False, True]),
        )


def test_vertical_concat_pd():
    df1 = pd.DataFrame({"x": ["a", "b", "c"]}, dtype="category")
    df2 = pd.DataFrame({"x": ["f", "b", "a"]}, dtype="category")
    pd.testing.assert_series_equal(
        vertical_concat([df1, df2])["x"],
        pd.Series(
            ["a", "b", "c", "f", "b", "a"],
            name="x",
            dtype=pd.CategoricalDtype(categories=["a", "b", "c", "f"]),
        ),
    )


def test_vertical_concat_pl():
    df1 = pl.DataFrame({"x": ["a", "b", "c"]}, schema={"x": pl.Categorical})
    df2 = pl.DataFrame({"x": ["f", "b", "a"]}, schema={"x": pl.Categorical})
    out = vertical_concat([df1, df2])["x"]
    assert out.equals(pl.Series("x", ["a", "b", "c", "f", "b", "a"]))
    assert out.to_physical().equals(pl.Series("x", [0, 1, 2, 3, 1, 0]))
    assert out.cat.get_categories().equals(pl.Series("x", ["a", "b", "c", "f"]))


@pytest.mark.parametrize("as_index", [False, True])
@pytest.mark.parametrize(
    "s2, expected_cats",
    [
        (
            pd.Series(
                pd.Categorical(["a", "b", "c", "d"], categories=["d", "c", "b", "a"])
            ),
            ["d", "c", "a", "b"],
        ),
        (
            pd.Series(pd.Categorical(["a", "c", "d"], categories=["a", "c", "d"])),
            ["d", "c", "a"],
        ),
        (pd.Series(["a", "c", "d"]), ["d", "c", "a"]),
        (pd.Series(["a", np.nan, "d"]), ["d", "c", "a"]),
    ],
)
def test_match_if_categorical_pd(s2, expected_cats, as_index):
    s1 = pd.Categorical(["d", "c", "a"], categories=["d", "c", "a"])
    s1 = pd.CategoricalIndex(s1) if as_index else pd.Series(s1)
    m1, m2 = match_if_categorical(s1, s2)
    cats1 = m1.categories if as_index else m1.cat.categories
    assert cats1.tolist() == expected_cats
    assert m2.cat.categories.tolist() == expected_cats
    pd.testing.assert_series_equal(m2.astype(object), s2.astype(object))


def test_match_if_categorical_pd_keeps_index_and_name():
    s1 = pd.Series(pd.Categorical(["d", "c"], categories=["d", "c"]))
    s2 = pd.Series(["c", "d"], index=[7, 3], name="uid")
    _, m2 = match_if_categorical(s1, s2)
    assert m2.index.tolist() == [7, 3]
    assert m2.name == "uid"


def test_match_if_categorical_pd_keeps_ordered():
    s1 = pd.Series(pd.Categorical(["d", "c"], categories=["d", "c"], ordered=True))
    m1, m2 = match_if_categorical(s1, pd.Series(["c", "x"]))
    assert m1.cat.ordered and m2.cat.ordered
    assert m2.cat.categories.tolist() == ["d", "c", "x"]


@pytest.mark.parametrize(
    "s2, bad",
    [(pd.Series([1.5, 2.0]), r"\[1\.5\]"), (pd.Series(["1", "2"]), r"\['1', '2'\]")],
)
def test_match_if_categorical_pd_mismatched_values(s2, bad):
    s1 = pd.Series(pd.Categorical([1, 2]))
    with pytest.raises(ValueError, match=bad):
        match_if_categorical(s1, s2)


def test_vertical_concat_pd_mismatched_values():
    df1 = pd.DataFrame({"uid": pd.Categorical([1, 2])})
    df2 = pd.DataFrame({"uid": ["1", "2"]})
    with pytest.raises(ValueError, match=r"object.*int64, e\.g\. \['1', '2'\]"):
        vertical_concat([df1, df2])


def test_concat(engine):
    series = generate_series(2, engine=engine)
    doubled = vertical_concat([series, series])
    assert doubled.shape[0] == 2 * series.shape[0]

    series = generate_series(2, engine=engine)
    renamer = {c: f"{c}_2" for c in series.columns}

    if engine == "pandas":
        series2 = series.rename(columns=renamer)
    else:
        series2 = series.rename(renamer)

    doubled = horizontal_concat([series, series2])
    assert doubled.shape[1] == 2 * series.shape[1]


def test_sort_dataframe_pd():
    pd.testing.assert_frame_equal(
        sort(pd.DataFrame({"x": [3, 1, 2]}), "x"), pd.DataFrame({"x": [1, 2, 3]})
    )
    pd.testing.assert_frame_equal(
        sort(pd.DataFrame({"x": [3, 1, 2]}), ["x"]), pd.DataFrame({"x": [1, 2, 3]})
    )


def test_sort_series_and_index_pd():
    pd.testing.assert_series_equal(sort(pd.Series([3, 1, 2])), pd.Series([1, 2, 3]))
    pd.testing.assert_index_equal(sort(pd.Index([3, 1, 2])), pd.Index([1, 2, 3]))
    pl.testing.assert_frame_equal(
        sort(pl.DataFrame({"x": [3, 1, 2]}), "x"),
        pl.DataFrame({"x": [1, 2, 3]}),
    )


def test_sort_dataframe_pl():
    pl.testing.assert_frame_equal(
        sort(pl.DataFrame({"x": [3, 1, 2]}), ["x"]),
        pl.DataFrame({"x": [1, 2, 3]}),
    )
    pl.testing.assert_series_equal(
        sort(pl.Series("x", [3, 1, 2])), pl.Series("x", [1, 2, 3])
    )


def test_sort_series_and_index_pl():
    pl.testing.assert_series_equal(sort(pl.Series([3, 1, 2])), pl.Series([1, 2, 3]))
    # pl.testing.assert_index_equal(sort(pl.Index([3, 1, 2])), pl.Index([1, 2, 3]))
    pl.testing.assert_frame_equal(
        sort(pl.DataFrame({"x": [3, 1, 2]}), "x"),
        pl.DataFrame({"x": [1, 2, 3]}),
    )


def test_multiply_pl_freq_1d():
    assert _multiply_pl_freq("1d", 4) == "4d"


def test_multiply_pl_freq_2d():
    assert _multiply_pl_freq("2d", 4) == "8d"


def test_multiply_freq_pl():
    pl.testing.assert_series_equal(
        _multiply_pl_freq("1d", pl_Series([1, 2])),
        pl_Series(["1d", "2d"]),
    )
    pl.testing.assert_series_equal(
        _multiply_pl_freq("4m", pl_Series([2, 4])),
        pl_Series(["8m", "16m"]),
    )


def test_offset_times_with_month_end():
    pd.testing.assert_index_equal(
        offset_times(
            pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            pd.offsets.MonthEnd(),
            1,
        ),
        pd.Index(pd.to_datetime(["2020-02-29", "2020-03-31", "2020-04-30"])),
    )


def test_offset_times_with_month_begin():
    pd.testing.assert_index_equal(
        offset_times(
            pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            pd.offsets.MonthBegin(),
            1,
        ),
        pd.Index(pd.to_datetime(["2020-02-01", "2020-03-01", "2020-04-01"])),
    )


_ARRAY_N_TIMES = pd.to_datetime(
    ["2020-01-01", "2020-01-31", "2020-02-29", "2021-03-15", "2021-12-31"]
)


def _offset_times_elementwise(times, freq, n):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        return [t + int(k) * freq for t, k in zip(times, n)]


@pytest.mark.parametrize(
    "freq", ["D", "2D", "h", "15min", "MS", "ME", "W-MON", "B", "QS", "YS", "2MS"]
)
def test_offset_times_with_array_n(freq):
    freq = pd.tseries.frequencies.to_offset(freq)
    n = np.random.default_rng(0).integers(-3, 40, _ARRAY_N_TIMES.size)
    times = pd.Series(_ARRAY_N_TIMES)
    assert list(offset_times(times, freq, n)) == _offset_times_elementwise(
        times, freq, n
    )


@pytest.mark.parametrize("freq", [pd.offsets.Day(), pd.offsets.MonthBegin()])
def test_offset_times_with_array_n_keeps_index_and_name(freq):
    n = np.array([1, 0, 2, 1, 0])
    serie = pd.Series(_ARRAY_N_TIMES, index=[5, 6, 7, 8, 9], name="ds")
    res = offset_times(serie, freq, n)
    assert list(res) == _offset_times_elementwise(serie, freq, n)
    assert res.name == "ds"
    pd.testing.assert_index_equal(res.index, serie.index)

    idx = pd.Index(_ARRAY_N_TIMES, name="ds")
    res = offset_times(idx, freq, n)
    assert list(res) == _offset_times_elementwise(idx, freq, n)
    assert res.name == "ds"


@pytest.mark.parametrize("freq", [pd.offsets.Day(), pd.offsets.MonthBegin()])
@pytest.mark.parametrize("tz", ["America/New_York", "Europe/Madrid"])
def test_offset_times_with_array_n_keeps_tz(freq, tz):
    n = np.array([1, 0, 2, 1, 0])
    times = pd.Series(_ARRAY_N_TIMES.tz_localize(tz))
    res = offset_times(times, freq, n)
    assert list(res) == _offset_times_elementwise(times, freq, n)
    assert res.dtype == times.dtype


@pytest.mark.parametrize(
    "freq, expected_unit",
    [
        (pd.offsets.Day(), "s"),
        (pd.offsets.MonthBegin(), "s"),
        (pd.offsets.Milli(), "ms"),
        (pd.offsets.Nano(), "ns"),
    ],
)
def test_offset_times_with_array_n_keeps_resolution(freq, expected_unit):
    n = np.array([1, 0, 2, 1, 0])
    times = pd.Series(_ARRAY_N_TIMES.astype("datetime64[s]"))
    res = offset_times(times, freq, n)
    assert list(res) == _offset_times_elementwise(times, freq, n)
    assert res.dtype == np.dtype(f"datetime64[{expected_unit}]")


@pytest.mark.parametrize("freq", [pd.offsets.Day(), pd.offsets.MonthBegin()])
def test_offset_times_with_array_n_pyarrow(freq):
    n = np.array([1, 0, 2, 1, 0])
    times = pd.Series(_ARRAY_N_TIMES).astype("timestamp[ns][pyarrow]")
    res = offset_times(times, freq, n)
    assert list(res) == _offset_times_elementwise(times, freq, n)
    assert res.dtype == times.dtype


def test_offset_times_with_array_n_pyarrow_date():
    freq = pd.offsets.MonthBegin()
    n = np.array([1, 0, 2, 1, 0])
    times = pd.Series(_ARRAY_N_TIMES).astype("date32[pyarrow]")
    res = offset_times(times, freq, n)
    expected = _offset_times_elementwise(pd.Series(_ARRAY_N_TIMES), freq, n)
    assert list(res) == [t.date() for t in expected]
    assert res.dtype == times.dtype


@pytest.mark.parametrize(
    "n", [np.zeros(5, dtype=np.int64), np.full(5, 3), np.array([2])]
)
def test_offset_times_with_constant_array_n(n):
    freq = pd.offsets.MonthBegin()
    times = pd.Series(_ARRAY_N_TIMES[: n.size])
    assert list(offset_times(times, freq, n)) == _offset_times_elementwise(
        times, freq, n
    )


@pytest.mark.parametrize("freq", [pd.offsets.Day(), pd.offsets.MonthBegin()])
@pytest.mark.parametrize(
    "n",
    [
        np.array([1, 0, 2, 1, 0], dtype="int8"),
        np.array([1, 0, 2, 1, 0], dtype="uint32"),
        pd.Series([1, 0, 2, 1, 0], dtype="uint32[pyarrow]"),
        pd.Series([1, 0, 2, 1, 0], dtype="Int64"),
    ],
)
def test_offset_times_with_narrow_array_n(freq, n):
    times = pd.Series(_ARRAY_N_TIMES)
    assert list(offset_times(times, freq, n)) == _offset_times_elementwise(
        times, freq, n
    )


@pytest.mark.parametrize("freq", [pd.offsets.Day(), pd.offsets.MonthBegin()])
@pytest.mark.parametrize("times", [_ARRAY_N_TIMES, pd.Series(_ARRAY_N_TIMES)])
@pytest.mark.parametrize(
    "n", [np.array([1, 2]), np.array([2]), np.ones((1, 5), dtype=np.int64)]
)
def test_offset_times_with_array_n_wrong_shape(freq, times, n):
    assert_raises_with_message(
        offset_times, "Expected n to have shape (5,)", times, freq, n
    )


@pytest.mark.parametrize("freq", [pd.offsets.Day(), pd.offsets.MonthBegin()])
def test_offset_times_with_float_array_n(freq):
    n = np.array([1.5, 2.0, 0.0, 1.0, 3.0])
    assert_raises_with_message(
        offset_times, "Expected n to have an integer dtype", _ARRAY_N_TIMES, freq, n
    )


@pytest.mark.parametrize(
    "freq",
    [
        pd.DateOffset(milliseconds=500),
        pd.DateOffset(microseconds=1),
        pd.offsets.Milli(1000),
        pd.offsets.Micro(10**6),
        pd.offsets.Nano(10**9),
    ],
)
@pytest.mark.parametrize("tz", [None, "Europe/Madrid"])
def test_offset_times_with_array_n_finer_than_times(freq, tz):
    n = np.array([1, 0, -2, 3, -1])
    times = pd.Series(_ARRAY_N_TIMES.astype("datetime64[s]").tz_localize(tz))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = offset_times(times, freq, n)
    expected = pd.concat(
        [offset_times(times.iloc[[i]], freq, int(k)) for i, k in enumerate(n)]
    )
    pd.testing.assert_series_equal(res, expected)


@pytest.mark.parametrize("freq", [pd.offsets.Second(), pd.offsets.Second(-1)])
def test_offset_times_with_array_n_overflow(freq):
    times = pd.Series(_ARRAY_N_TIMES)
    n = np.full(times.size, 2**62 // 10**9 * 4)
    with pytest.raises(pd.errors.OutOfBoundsTimedelta):
        offset_times(times, freq, int(n[0]))
    with pytest.raises(pd.errors.OutOfBoundsTimedelta):
        offset_times(times, freq, n)


@pytest.mark.parametrize("freq", [pd.offsets.Second(0), pd.offsets.Second(-1)])
def test_offset_times_with_array_n_non_positive_tick(freq):
    times = pd.Series(_ARRAY_N_TIMES)
    res = offset_times(times, freq, np.full(times.size, 2))
    pd.testing.assert_series_equal(res, offset_times(times, freq, 2))


@pytest.mark.parametrize("freq", [pd.offsets.Second(), pd.offsets.MonthBegin()])
def test_offset_times_with_uint64_array_n_overflow(freq):
    times = pd.Series(_ARRAY_N_TIMES)
    n = np.zeros(times.size, dtype=np.uint64)
    n[0] = 2**63 + 5
    with pytest.raises(OverflowError):
        offset_times(times, freq, int(n[0]))
    with pytest.raises(OverflowError):
        offset_times(times, freq, n)


@pytest.mark.parametrize(
    "freq",
    [
        pd.offsets.Day(),
        pd.offsets.MonthBegin(),
        pd.offsets.BusinessHour(),
        pd.DateOffset(milliseconds=500),
    ],
)
@pytest.mark.parametrize("times", [_ARRAY_N_TIMES[:0], pd.Series(_ARRAY_N_TIMES[:0])])
def test_offset_times_with_empty_array_n(freq, times):
    times = times.astype("datetime64[s]")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = offset_times(times, freq, np.array([], dtype=np.int64))
    assert type(res) is type(times)
    assert res is not times
    assert len(res) == 0
    assert res.dtype == times.dtype


_PA_TIMES = pd.DatetimeIndex(
    ["2020-01-01", "2020-01-31", "2020-02-29", "2021-03-13", "2021-10-31"]
)
_PYARROW_DATES = [
    (None, "ns", "timestamp[ns][pyarrow]"),
    (None, "s", "timestamp[s][pyarrow]"),
    ("America/New_York", "ns", "timestamp[ns, tz=America/New_York][pyarrow]"),
    (None, "s", "date32[pyarrow]"),
    (None, "s", "date64[pyarrow]"),
]
_skip_pandas_lt_2 = pytest.mark.skipif(
    Version(pd.__version__) < Version("2.0"),
    reason="pyarrow dates are only supported on pandas>=2",
)


@_skip_pandas_lt_2
@pytest.mark.parametrize("tz, unit, dtype", _PYARROW_DATES)
@pytest.mark.parametrize("freq", ["D", "h", "W", "ME", "MS", "QS"])
@pytest.mark.parametrize("n", [2, np.array([1, 0, 3, -2, 1])])
@pytest.mark.parametrize("container", [pd.Series, pd.Index])
def test_offset_times_pyarrow(tz, unit, dtype, freq, n, container):
    times = _PA_TIMES.tz_localize(tz).as_unit(unit)
    if dtype.startswith("date") and freq == "h":
        pytest.skip("dates can't be offset by a sub-daily frequency")
    times = container(times, name="ds")
    arrow_times = times.astype(dtype)
    res = offset_times(arrow_times, freq, n)
    expected = offset_times(times, freq, n).astype(dtype)
    assert res.dtype == arrow_times.dtype
    if container is pd.Series:
        pd.testing.assert_series_equal(res, expected)
    else:
        pd.testing.assert_index_equal(res, expected)


@_skip_pandas_lt_2
@pytest.mark.parametrize("tz, unit, dtype", _PYARROW_DATES)
@pytest.mark.parametrize("freq", ["D", "h", "W", "ME", "MS", "QS"])
def test_time_ranges_pyarrow(tz, unit, dtype, freq):
    times = _PA_TIMES.tz_localize(tz).as_unit(unit)
    if dtype.startswith("date") and freq == "h":
        pytest.skip("dates can't be offset by a sub-daily frequency")
    starts = pd.Series(times)
    arrow_starts = starts.astype(dtype)
    res = time_ranges(arrow_starts, freq, 3)
    expected = time_ranges(starts, freq, 3).astype(dtype)
    assert res.dtype == arrow_starts.dtype
    pd.testing.assert_series_equal(res, expected)


@_skip_pandas_lt_2
@pytest.mark.parametrize(
    "freq",
    [
        "h",
        "15min",
        "500ms",
        pd.offsets.Hour(25),
        pd.DateOffset(hours=12),
        pd.offsets.BusinessHour(),
        pd.DateOffset(microseconds=1),
    ],
)
@pytest.mark.parametrize("n", [1, np.array([1, 0, 3, -2, 1])])
@pytest.mark.parametrize("dtype", ["date32[pyarrow]", "date64[pyarrow]"])
def test_offset_times_pyarrow_dates_sub_daily(freq, n, dtype):
    dates = pd.Series(_PA_TIMES).astype(dtype)
    with pytest.raises(ValueError, match="aren't at midnight"):
        offset_times(dates, freq, n)
    with pytest.raises(ValueError, match="aren't at midnight"):
        time_ranges(dates, freq, 2)


@_skip_pandas_lt_2
@pytest.mark.parametrize("tz, unit, dtype", _PYARROW_DATES)
def test_offset_times_pyarrow_with_nulls(tz, unit, dtype):
    times = _PA_TIMES.tz_localize(tz).as_unit(unit)
    times = pd.Series(times).astype(dtype)
    times.iloc[1] = None
    for n in [1, np.array([1, 0, 3, -2, 1])]:
        res = offset_times(times, "D", n)
        assert res.isna().tolist() == times.isna().tolist()
    res = time_ranges(times, "D", 2)
    assert res.isna().tolist() == [False, False, True, True] + [False] * 6


@_skip_pandas_lt_2
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("container", [pd.Series, pd.Index])
def test_offset_times_pyarrow_tz_aware_around_dst(unit, container):
    tz = "Europe/Madrid"
    times = pd.DatetimeIndex(
        ["2021-03-28 00:30", "2021-03-28 03:30", None, "2021-10-31 01:30"]
    )
    times = container(times.tz_localize("UTC").tz_convert(tz).as_unit(unit))
    arrow_times = times.astype(pd.ArrowDtype(pa.timestamp(unit, tz=tz)))
    for freq in ["h", "D", "MS"]:
        res = offset_times(arrow_times, freq, 1)
        expected = offset_times(times, freq, 1).astype(arrow_times.dtype)
        if container is pd.Series:
            pd.testing.assert_series_equal(res, expected)
        else:
            pd.testing.assert_index_equal(res, expected)


@_skip_pandas_lt_2
def test_offset_times_pyarrow_date64_not_at_midnight():
    dates = pd.Series(
        pa.array([86_400_000 + 500], pa.date64()), dtype="date64[pyarrow]"
    )
    for n in [1, np.array([1])]:
        with pytest.raises(ValueError, match="some of them have a time part"):
            offset_times(dates, "D", n)
    with pytest.raises(ValueError, match="some of them have a time part"):
        time_ranges(dates, "D", 2)


@_skip_pandas_lt_2
@pytest.mark.parametrize("tz, unit, dtype", _PYARROW_DATES)
def test_time_ranges_pyarrow_zero_periods(tz, unit, dtype):
    starts = pd.Series(_PA_TIMES.tz_localize(tz).as_unit(unit)).astype(dtype)
    res = time_ranges(starts, "D", 0)
    pd.testing.assert_series_equal(res, pd.Series([], dtype=starts.dtype))


@_skip_pandas_lt_2
def test_time_ranges_pyarrow_int_freq():
    starts = pd.Series(_PA_TIMES).astype("date32[pyarrow]")
    with pytest.raises(ValueError, match=r"data type: 'date32\[day\]\[pyarrow\]'"):
        time_ranges(starts, 1, 2)


@_skip_pandas_lt_2
def test_offset_times_pyarrow_dates_whole_days():
    dates = pd.Series(_PA_TIMES).astype("date32[pyarrow]")
    res = offset_times(dates, "h", 48)
    expected = offset_times(dates, "2D", 1)
    pd.testing.assert_series_equal(res, expected)


@_skip_pandas_lt_2
@pytest.mark.parametrize("tz", [None, "America/New_York"])
def test_offset_times_pyarrow_finer_freq(tz):
    times = pd.Series(_PA_TIMES.tz_localize(tz).as_unit("ms"))
    arrow_times = times.astype(pd.ArrowDtype(pa.timestamp("ms", tz=tz)))
    for n in [1, np.array([1, 0, 3, -2, 1])]:
        res = offset_times(arrow_times, "us", n)
        expected = offset_times(times, "us", n)
        assert res.dtype == pd.ArrowDtype(pa.timestamp("us", tz=tz))
        pd.testing.assert_series_equal(res, expected.astype(res.dtype))
    res = time_ranges(arrow_times, "us", 2)
    expected = time_ranges(times, "us", 2)
    assert res.dtype == pd.ArrowDtype(pa.timestamp("us", tz=tz))
    pd.testing.assert_series_equal(res, expected.astype(res.dtype))


@_skip_pandas_lt_2
@pytest.mark.parametrize("tz, unit, dtype", _PYARROW_DATES)
def test_make_future_dataframe_pyarrow(tz, unit, dtype):
    times = _PA_TIMES.tz_localize(tz).as_unit(unit)
    ids = pd.Series(np.arange(times.size))
    res = make_future_dataframe(ids, pd.Series(times).astype(dtype), freq="D", h=3)
    expected = make_future_dataframe(ids, pd.Series(times), freq="D", h=3)
    expected["ds"] = expected["ds"].astype(dtype)
    pd.testing.assert_frame_equal(res, expected)


@_skip_pandas_lt_2
@pytest.mark.parametrize("tz, unit, dtype", _PYARROW_DATES)
def test_backtest_splits_pyarrow(tz, unit, dtype):
    times = _PA_TIMES.tz_localize(tz).as_unit(unit)
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(times.size), 10),
            "ds": time_ranges(pd.Series(times), "D", 10),
            "y": np.arange(10 * times.size, dtype=np.float64),
        }
    )
    arrow_df = df.assign(ds=df["ds"].astype(dtype))
    kwargs = dict(n_windows=2, h=3, id_col="unique_id", time_col="ds", freq="D")
    for res, expected in zip(
        backtest_splits(arrow_df, **kwargs), backtest_splits(df, **kwargs)
    ):
        for res_df, expected_df in zip(res, expected):
            expected_df = expected_df.copy()
            for col in ("ds", "cutoff"):
                if col in expected_df:
                    expected_df[col] = expected_df[col].astype(dtype)
            pd.testing.assert_frame_equal(res_df, expected_df)


def test_offset_times_pl():
    pl.testing.assert_series_equal(
        offset_times(
            pl_Series([dt(2020, 1, 31), dt(2020, 2, 28), dt(2020, 3, 31)]), "1mo", 1
        ),
        pl_Series([dt(2020, 2, 29), dt(2020, 3, 28), dt(2020, 4, 30)]),
    )
    pl.testing.assert_series_equal(
        offset_times(
            pl_Series([dt(2020, 1, 31), dt(2020, 2, 29), dt(2020, 3, 31)]), "1mo", 1
        ),
        pl_Series([dt(2020, 2, 29), dt(2020, 3, 31), dt(2020, 4, 30)]),
    )


# datetimes
def test_time_ranges_daily_pd():
    dates = pd.to_datetime(["2000-01-01", "2010-10-10"])
    pd.testing.assert_series_equal(
        time_ranges(dates, freq="D", periods=3),
        pd.Series(
            pd.to_datetime(
                [
                    "2000-01-01",
                    "2000-01-02",
                    "2000-01-03",
                    "2010-10-10",
                    "2010-10-11",
                    "2010-10-12",
                ]
            )
        ),
    )


def test_time_ranges_every_2_days_pd():
    dates = pd.to_datetime(["2000-01-01", "2010-10-10"])
    pd.testing.assert_series_equal(
        time_ranges(dates, freq="2D", periods=3),
        pd.Series(
            pd.to_datetime(
                [
                    "2000-01-01",
                    "2000-01-03",
                    "2000-01-05",
                    "2010-10-10",
                    "2010-10-12",
                    "2010-10-14",
                ]
            )
        ),
    )


def test_time_ranges_every_4_days_pd():
    dates = pd.to_datetime(["2000-01-01", "2010-10-10"])
    pd.testing.assert_series_equal(
        time_ranges(dates, freq="4D", periods=3),
        pd.Series(
            pd.to_datetime(
                [
                    "2000-01-01",
                    "2000-01-05",
                    "2000-01-09",
                    "2010-10-10",
                    "2010-10-14",
                    "2010-10-18",
                ]
            )
        ),
    )


def test_time_ranges_with_month_begin_offset_pd():
    pd.testing.assert_series_equal(
        time_ranges(
            pd.to_datetime(["2000-01-01", "2010-10-01"]),
            freq=2 * pd.offsets.MonthBegin(),
            periods=2,
        ),
        pd.Series(
            pd.to_datetime(["2000-01-01", "2000-03-01", "2010-10-01", "2010-12-01"])
        ),
    )


def test_time_ranges_with_year_begin_offset_and_timezone_pd():
    tz = "America/New_York"
    pd.testing.assert_series_equal(
        time_ranges(
            pd.to_datetime(["2000-01-01", "2010-01-01"]).tz_localize(tz),
            freq=2 * pd.offsets.YearBegin(),
            periods=2,
        ),
        pd.Series(
            pd.to_datetime(
                ["2000-01-01", "2002-01-01", "2010-01-01", "2012-01-01"]
            ).tz_localize(tz)
        ),
    )


def test_time_ranges_with_year_end_offset_pd():
    pd.testing.assert_series_equal(
        time_ranges(
            pd.to_datetime(["2000-12-31", "2010-12-31"]),
            freq=2 * pd.offsets.YearEnd(),
            periods=2,
        ),
        pd.Series(
            pd.to_datetime(["2000-12-31", "2002-12-31", "2010-12-31", "2012-12-31"])
        ),
    )


@pytest.mark.parametrize(
    "freq", ["D", "h", "MS", "W-MON", "500ms", pd.DateOffset(milliseconds=500)]
)
@pytest.mark.parametrize("tz", [None, "America/New_York"])
def test_time_ranges_matches_offset_times_pd(freq, tz):
    starts = pd.Series(
        _ARRAY_N_TIMES.astype("datetime64[s]").tz_localize(tz), name="ds"
    )
    periods = 4
    res = time_ranges(starts, freq, periods)
    per_period = [offset_times(starts, freq, i) for i in range(periods)]
    expected = pd.Series(
        [per_period[i][j] for j in range(len(starts)) for i in range(periods)],
        dtype=per_period[0].dtype,
    )
    pd.testing.assert_series_equal(res, expected)


@pytest.mark.parametrize("tz", [None, "America/New_York"])
def test_time_ranges_zero_periods_pd(tz):
    starts = _ARRAY_N_TIMES.astype("datetime64[s]").tz_localize(tz)
    res = time_ranges(starts, "D", 0)
    pd.testing.assert_series_equal(res, pd.Series([], dtype=starts.dtype))


# ints
def test_time_ranges_with_integers_pd():
    dates = pd.Series([1, 10])
    pd.testing.assert_series_equal(
        time_ranges(dates, freq=1, periods=3), pd.Series([1, 2, 3, 10, 11, 12])
    )
    pd.testing.assert_series_equal(
        time_ranges(dates, freq=2, periods=3), pd.Series([1, 3, 5, 10, 12, 14])
    )
    pd.testing.assert_series_equal(
        time_ranges(dates, freq=4, periods=3), pd.Series([1, 5, 9, 10, 14, 18])
    )
    arrow_dates = dates.astype("int64[pyarrow]")
    pd.testing.assert_series_equal(
        time_ranges(arrow_dates, freq=2, periods=3),
        pd.Series([1, 3, 5, 10, 12, 14], dtype="int64[pyarrow]"),
    )


# datetimes
def test_time_ranges_daily_pl():
    dates = pl.Series([dt(2000, 1, 1), dt(2010, 10, 10)])
    pl.testing.assert_series_equal(
        time_ranges(dates, freq="1d", periods=3),
        pl.Series(
            [
                dt(2000, 1, 1),
                dt(2000, 1, 2),
                dt(2000, 1, 3),
                dt(2010, 10, 10),
                dt(2010, 10, 11),
                dt(2010, 10, 12),
            ]
        ),
    )


def test_time_ranges_every_2_days_pl():
    dates = pl.Series([dt(2000, 1, 1), dt(2010, 10, 10)])
    pl.testing.assert_series_equal(
        time_ranges(dates, freq="2d", periods=3),
        pl.Series(
            [
                dt(2000, 1, 1),
                dt(2000, 1, 3),
                dt(2000, 1, 5),
                dt(2010, 10, 10),
                dt(2010, 10, 12),
                dt(2010, 10, 14),
            ]
        ),
    )


def test_time_ranges_every_4_days_pl():
    dates = pl.Series([dt(2000, 1, 1), dt(2010, 10, 10)])
    pl.testing.assert_series_equal(
        time_ranges(dates, freq="4d", periods=3),
        pl.Series(
            [
                dt(2000, 1, 1),
                dt(2000, 1, 5),
                dt(2000, 1, 9),
                dt(2010, 10, 10),
                dt(2010, 10, 14),
                dt(2010, 10, 18),
            ]
        ),
    )


def test_time_ranges_month_offset_pl():
    pl.testing.assert_series_equal(
        time_ranges(pl.Series([dt(2010, 2, 28), dt(2000, 1, 31)]), "1mo", 3),
        pl.Series(
            [
                dt(2010, 2, 28),
                dt(2010, 3, 31),
                dt(2010, 4, 30),
                dt(2000, 1, 31),
                dt(2000, 2, 29),
                dt(2000, 3, 31),
            ]
        ),
    )


# dates
def test_dates_with_different_months_pl():
    dates = pl.Series([datetime.date(2000, 1, 1), datetime.date(2010, 10, 10)])
    pl.testing.assert_series_equal(
        time_ranges(dates, freq="1d", periods=2),
        pl.Series(
            [
                datetime.date(2000, 1, 1),
                datetime.date(2000, 1, 2),
                datetime.date(2010, 10, 10),
                datetime.date(2010, 10, 11),
            ]
        ),
    )


# ints


def test_time_ranges_with_integers_pl():
    dates = pl.Series([1, 10])
    pl.testing.assert_series_equal(
        time_ranges(dates, freq=1, periods=3),
        pl.Series([1, 2, 3, 10, 11, 12]),
    )
    pl.testing.assert_series_equal(
        time_ranges(dates, freq=2, periods=3),
        pl.Series([1, 3, 5, 10, 12, 14]),
    )
    pl.testing.assert_series_equal(
        time_ranges(dates, freq=4, periods=3),
        pl.Series([1, 5, 9, 10, 14, 18]),
    )


def test_repeat_objects_pd():
    pd.testing.assert_index_equal(
        repeat(pd.CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c"]), 2),
        pd.CategoricalIndex(["a", "a", "b", "b", "c", "c"], categories=["a", "b", "c"]),
    )
    pd.testing.assert_series_equal(
        repeat(pd.Series([1, 2]), 2), pd.Series([1, 1, 2, 2])
    )
    pd.testing.assert_series_equal(
        repeat(pd.Series([1, 2]), pd.Series([2, 3])),
        pd.Series([1, 1, 2, 2, 2]),
    )


def test_repeat_objects_np():
    np.testing.assert_array_equal(
        repeat(np.array([np.datetime64("2000-01-01"), np.datetime64("2010-10-10")]), 2),
        np.array(
            [
                np.datetime64("2000-01-01"),
                np.datetime64("2000-01-01"),
                np.datetime64("2010-10-10"),
                np.datetime64("2010-10-10"),
            ]
        ),
    )
    np.testing.assert_array_equal(
        repeat(np.array([1, 2]), np.array([2, 3])),
        np.array([1, 1, 2, 2, 2]),
    )


def test_repeat_objects_pl():
    s = pl.Series(["a", "b", "c"], dtype=pl.Categorical)
    pl.testing.assert_series_equal(repeat(s, 2), pl.concat([s, s]).sort())
    pl.testing.assert_series_equal(
        repeat(pl.Series([2, 4]), 2), pl.Series([2, 2, 4, 4])
    )
    pl.testing.assert_series_equal(
        repeat(pl.Series([1, 2]), np.array([2, 3])),
        pl.Series([1, 1, 2, 2, 2]),
    )


@pytest.mark.parametrize(
    "step_size, expected_ds, expected_cutoff",
    [
        (
            1,
            np.hstack([[46, 47, 48], [47, 48, 49], [48, 49, 50]]),
            np.repeat([45, 46, 47], 3),
        ),
        (
            2,
            np.hstack([[46, 47, 48], [48, 49, 50]]),
            np.repeat([45, 47], 3),
        ),
    ],
)
def test_cv_times(step_size, expected_ds, expected_cutoff):
    times = np.arange(51, dtype=np.int64)
    uids = pd.Series(["id_0"])
    indptr = np.array([0, 51])
    h = 3
    test_size = 5

    actual = cv_times(
        times=times,
        uids=uids,
        indptr=indptr,
        h=h,
        test_size=test_size,
        step_size=step_size,
    )
    expected = pd.DataFrame(
        {
            "unique_id": ["id_0"] * len(expected_ds),
            "ds": expected_ds,
            "cutoff": expected_cutoff,
        }
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_backtest_vectorized_helpers_match_expected_indices():
    times = np.array(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-04",
            "2020-01-01",
            "2020-01-03",
            "2020-01-05",
            "2020-01-07",
            "2020-01-02",
            "2020-01-06",
        ],
        dtype="datetime64[ns]",
    )
    indptr = np.array([0, 3, 7, 9])
    bounds = np.array(
        ["2020-01-02", "2020-01-05", "2019-12-31"], dtype="datetime64[ns]"
    )

    np.testing.assert_array_equal(
        _find_boundaries(times, indptr, bounds), np.array([2, 6, 7])
    )
    np.testing.assert_array_equal(
        _ranges_to_indexer(
            starts=np.array([0, 3, 6, 8]),
            stops=np.array([2, 3, 5, 10]),
        ),
        np.array([0, 1, 8, 9]),
    )


@pytest.mark.parametrize(
    "values, lookup, expected", [([1, 2, 3], [1], [True, False, False])]
)
def test_is_in(engine, values, lookup, expected):
    series = pd.Series(values) if engine == "pandas" else pl.Series(values)
    result = is_in(series, lookup)
    np.testing.assert_equal(result, np.array(expected))


@pytest.mark.parametrize(
    "vals, lower, upper, expected",
    [([1, 2, 3], [0, 1, 4], [4, 1, 2], [True, False, False])],
)
def test_between(engine, vals, lower, upper, expected):
    if engine == "pandas":
        series = pd.Series(vals)
        low = pd.Series(lower)
        up = pd.Series(upper)
    else:
        series = pl.Series(vals)
        low = pl.Series(lower)
        up = pl.Series(upper)
    result = between(series, low, up)
    np.testing.assert_equal(result, np.array(expected))


def test_fill_null(engine):
    data = {"x": [1, np.nan], "y": [np.nan, 2]}
    fill_values = {"x": 2, "y": 1}
    if engine == "pandas":
        df = pd.DataFrame(data)
        result = fill_null(df, fill_values)
        expected = pd.DataFrame({"x": [1, 2], "y": [1, 2]}, dtype="float64")
        pd.testing.assert_frame_equal(result, expected)
    else:
        df = pl.DataFrame({"x": [1, None], "y": [None, 2]})
        result = fill_null(df, fill_values)
        expected = pl.DataFrame({"x": [1, 2], "y": [1, 2]})
        pl.testing.assert_frame_equal(result, expected)


def test_cast(engine):
    if engine == "pandas":
        series = pd.Series([1, 2, 3])
        result = cast(series, "int16")
        expected = pd.Series([1, 2, 3], dtype="int16")
        pd.testing.assert_series_equal(result, expected)
    else:
        series = pl.Series("x", [1, 2, 3])
        result = cast(series, pl.Int16).to_pandas()
        expected = pd.Series([1, 2, 3], name="x", dtype="int16")
        pd.testing.assert_series_equal(result, expected)


def test_make_future_dataframe(engine):
    if engine == "pandas":
        ids = pd.Series([1, 2])
        dates = pd.to_datetime(["2000-01-01", "2010-10-10"])
        result = make_future_dataframe(ids, dates, freq="D", h=2)
        expected = pd.DataFrame(
            {
                "unique_id": [1, 1, 2, 2],
                "ds": pd.to_datetime(
                    ["2000-01-02", "2000-01-03", "2010-10-11", "2010-10-12"]
                ),
            }
        )
        pd.testing.assert_frame_equal(result, expected)
    else:
        ids = pl.Series([1, 2])
        dates = pl.Series([dt(2000, 1, 1), dt(2010, 10, 10)])
        result = make_future_dataframe(
            ids, dates, freq="1d", h=2, id_col="uid", time_col="dates"
        )
        expected = pl.DataFrame(
            {
                "uid": [1, 1, 2, 2],
                "dates": [
                    dt(2000, 1, 2),
                    dt(2000, 1, 3),
                    dt(2010, 10, 11),
                    dt(2010, 10, 12),
                ],
            }
        )
        pl.testing.assert_frame_equal(result, expected)


def test_anti_join(engine):
    df1 = {"x": [1, 2]}
    df2 = {"x": [1]}
    if engine == "pandas":
        result = anti_join(pd.DataFrame(df1), pd.DataFrame(df2), on="x")
        expected = pd.DataFrame({"x": [2]})
        pd.testing.assert_frame_equal(result, expected)
        assert (anti_join(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}), on="x").shape[0]== 0)
    else:
        result = anti_join(pl.DataFrame(df1), pl.DataFrame(df2), on="x")
        expected = pl.DataFrame({"x": [2]})
        pl.testing.assert_frame_equal(result, expected)
        assert (anti_join(pl.DataFrame({"x": [1]}), pl.DataFrame({"x": [1]}), on="x").shape[0]== 0)

def test_cv_times_with_horizon_and_test_sizes():
    horizon = 3
    test_size = 5
    for equal_ends in [True, False]:
        n_series = 2
        series = generate_series(n_series, equal_ends=equal_ends)
        uids, last_times, data, indptr, _ = process_df(series, "unique_id", "ds", "y")
        times = series["ds"].to_numpy()
        df_dates = cv_times(
            times=times,
            uids=uids,
            indptr=indptr,
            h=horizon,
            test_size=test_size,
            step_size=1,
        )
        assert len(df_dates) == n_series * horizon * (test_size - horizon + 1)

@pytest.fixture
def static_features():
    return ["static_0", "static_1"]


@pytest.mark.parametrize("n_static_features", [0, 2])
def test_static_features(static_features, n_static_features):
    series_pd = generate_series(
        1_000, n_static_features=n_static_features, equal_ends=False, engine="pandas"
    )
    for i in range(n_static_features):
        series_pd[f"static_{i}"] = (
            series_pd[f"static_{i}"].map(lambda x: f"x_{x}").astype("category")
        )
    scrambled_series_pd = series_pd.sample(frac=1.0)
    dfp = DataFrameProcessor("unique_id", "ds", "y")
    uids, times, data, indptr, _ = dfp.process(scrambled_series_pd)
    np.testing.assert_array_equal(times, series_pd.groupby("unique_id", observed=True)["ds"].max().values)
    np.testing.assert_array_equal(uids, np.sort(series_pd["unique_id"].unique()))
    for i in range(n_static_features):
        series_pd[f"static_{i}"] = series_pd[f"static_{i}"].cat.codes
    np.testing.assert_array_equal(data, series_pd[["y"] + static_features[:n_static_features]].to_numpy())
    np.testing.assert_array_equal(
        np.diff(indptr), series_pd.groupby("unique_id", observed=True).size().values
    )


# test process_df with target_col=None
def test_none_target():
    series_pd = generate_series(10, n_static_features=2, equal_ends=False, engine="pandas")
    series_pd = series_pd.rename(columns={"y": "exog_0"})
    _, _, data, indptr, _ = process_df(series_pd, "unique_id", "ds", None)
    np.testing.assert_allclose(
        data,
        to_numpy(series_pd.drop(columns=["unique_id", "ds"])),
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_process_df_null_ids(engine):
    constructor = pd.DataFrame if engine == "pandas" else pl.DataFrame
    df = constructor(
        {
            "unique_id": ["a", "a", None, "b"],
            "ds": [1, 2, 1, 1],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="id column .* contains null values"):
        process_df(df, "unique_id", "ds", "y")


@pytest.mark.parametrize("n_static_features", [0, 2])
def test_n_static_features(static_features, n_static_features):
    series_pl = generate_series(
        1_000, n_static_features=n_static_features, equal_ends=False, engine="polars"
    )
    scrambled_series_pl = series_pl.sample(fraction=1.0, shuffle=True)
    dfp = DataFrameProcessor("unique_id", "ds", "y")
    uids, times, data, indptr, _ = dfp.process(scrambled_series_pl)
    grouped = group_by(series_pl, "unique_id")
    np.testing.assert_array_equal(times, grouped.agg(pl.col("ds").max()).sort("unique_id")["ds"].to_numpy())
    np.testing.assert_array_equal(uids, series_pl["unique_id"].unique().sort())

    expected_data = (
            scrambled_series_pl
            .sort(["unique_id", "ds"])
            .select([
                pl.col(c).map_batches(lambda s: s.to_physical())
                for c in ["y"] + static_features[:n_static_features]
            ])
            .to_numpy()
        )
    np.testing.assert_array_equal(data, expected_data)
    np.testing.assert_array_equal(np.diff(indptr), grouped.len().sort("unique_id")["len"].to_numpy())

def test_short_stories():
    short_series = generate_series(100, max_length=50)
    list(
        backtest_splits(
            short_series,
            n_windows=1,
            h=49,
            id_col="unique_id",
            time_col="ds",
            freq=pd.offsets.Day(),
        )
    )[0]
    assert_raises_with_message(
        lambda: list(
            backtest_splits(
                short_series,
                n_windows=1,
                h=50,
                id_col="unique_id",
                time_col="ds",
                freq=pd.offsets.Day(),
            )
        ),
        "at least 51 samples are required",
    )
    some_short_series = generate_series(100, min_length=20, max_length=100)
    with warnings.catch_warnings(record=True) as issued_warnings:
        warnings.simplefilter("always", UserWarning)
        _ = list(
            backtest_splits(
                some_short_series,
                n_windows=1,
                h=50,
                id_col="unique_id",
                time_col="ds",
                freq=pd.offsets.Day(),
            )
        )
        assert any("will be dropped" in str(w.message) for w in issued_warnings)


def test_backtest_splits_empty_dataframe(engine):
    if engine == "pandas":
        series = pd.DataFrame(
            {
                "unique_id": pd.Series([], dtype=object),
                "ds": pd.Series([], dtype="datetime64[ns]"),
                "y": pd.Series([], dtype=np.float64),
            }
        )
        freq = pd.offsets.Day()
    else:
        series = pl.DataFrame(
            schema={
                "unique_id": pl.Utf8,
                "ds": pl.Datetime,
                "y": pl.Float64,
            }
        )
        freq = "1d"

    assert_raises_with_message(
        lambda: list(
            backtest_splits(
                series,
                n_windows=1,
                h=1,
                id_col="unique_id",
                time_col="ds",
                freq=freq,
            )
        ),
        "at least 2 samples are required",
    )


def test_backtest_splits_incomparable_ids_fall_back_to_mask_path(monkeypatch):
    series = pd.DataFrame(
        {
            "unique_id": ["a", 1, "a", 1, "a", 1],
            "ds": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "y": np.arange(6),
        }
    )
    mask_calls = 0
    orig_mask = ufp._single_split

    def counting_mask(*args, **kwargs):
        nonlocal mask_calls
        mask_calls += 1
        return orig_mask(*args, **kwargs)

    def fail_fast_path(*args, **kwargs):
        raise AssertionError("incomparable ids should use the mask-based splitter")

    monkeypatch.setattr(ufp, "_single_split", counting_mask)
    monkeypatch.setattr(ufp, "_single_split_sorted", fail_fast_path)

    splits = list(
        backtest_splits(
            series,
            n_windows=1,
            h=1,
            id_col="unique_id",
            time_col="ds",
            freq=pd.offsets.Day(),
        )
    )

    assert len(splits) == 1
    assert mask_calls == 1


def test_backtest_splits_missing_ids_fall_back_to_mask_path(monkeypatch, engine):
    if engine == "pandas":
        series = pd.DataFrame(
            {
                "unique_id": pd.Categorical([None, None, "a", "a", "b", "b"]),
                "ds": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-01",
                        "2020-01-02",
                    ]
                ),
                "y": np.arange(6),
            }
        )
        freq = pd.offsets.Day()
    else:
        series = pl.DataFrame(
            {
                "unique_id": [None, None, "a", "a", "b", "b"],
                "ds": [
                    dt(2020, 1, 1),
                    dt(2020, 1, 2),
                    dt(2020, 1, 1),
                    dt(2020, 1, 2),
                    dt(2020, 1, 1),
                    dt(2020, 1, 2),
                ],
                "y": np.arange(6),
            },
            schema_overrides={"unique_id": pl.Utf8},
        )
        freq = "1d"
    mask_calls = 0
    orig_mask = ufp._single_split

    def counting_mask(*args, **kwargs):
        nonlocal mask_calls
        mask_calls += 1
        return orig_mask(*args, **kwargs)

    def fail_fast_path(*args, **kwargs):
        raise AssertionError("missing ids should use the mask-based splitter")

    monkeypatch.setattr(ufp, "_single_split", counting_mask)
    monkeypatch.setattr(ufp, "_single_split_sorted", fail_fast_path)

    splits = list(
        backtest_splits(
            series,
            n_windows=1,
            h=1,
            id_col="unique_id",
            time_col="ds",
            freq=freq,
        )
    )

    assert len(splits) == 1
    assert mask_calls == 1


def _test_backtest_splits(df, n_windows, h, step_size, input_size):
    max_dates = df.groupby("unique_id", observed=True)["ds"].max()
    day_offset = pd.offsets.Day()
    common_kwargs = dict(
        n_windows=n_windows,
        h=h,
        id_col="unique_id",
        time_col="ds",
        freq=pd.offsets.Day(),
        step_size=step_size,
        input_size=input_size,
    )
    permuted_df = df.sample(frac=1.0)
    splits = backtest_splits(df, **common_kwargs)
    splits_on_permuted = list(backtest_splits(permuted_df, **common_kwargs))
    if step_size is None:
        step_size = h
    test_size = h + step_size * (n_windows - 1)
    for window, (cutoffs, train, valid) in enumerate(splits):
        offset = test_size - window * step_size
        expected_max_train_dates = max_dates - day_offset * offset
        max_train_dates = train.groupby("unique_id", observed=True)["ds"].max()
        pd.testing.assert_series_equal(max_train_dates, expected_max_train_dates)
        pd.testing.assert_frame_equal(
            cutoffs, max_train_dates.rename("cutoff").reset_index()
        )

        if input_size is not None:
            expected_min_train_dates = expected_max_train_dates - day_offset * (
                input_size - 1
            )
            min_train_dates = train.groupby("unique_id", observed=True)["ds"].min()
            pd.testing.assert_series_equal(min_train_dates, expected_min_train_dates)

        expected_min_valid_dates = expected_max_train_dates + day_offset
        min_valid_dates = valid.groupby("unique_id", observed=True)["ds"].min()
        pd.testing.assert_series_equal(min_valid_dates, expected_min_valid_dates)

        expected_max_valid_dates = expected_max_train_dates + day_offset * h
        max_valid_dates = valid.groupby("unique_id", observed=True)["ds"].max()
        pd.testing.assert_series_equal(max_valid_dates, expected_max_valid_dates)

        if window == n_windows - 1:
            pd.testing.assert_series_equal(max_valid_dates, max_dates)

        _, permuted_train, permuted_valid = splits_on_permuted[window]
        pd.testing.assert_frame_equal(
            train, permuted_train.sort_values(["unique_id", "ds"])
        )
    pd.testing.assert_frame_equal(
        valid, permuted_valid.sort_values(["unique_id", "ds"])
    )


@pytest.mark.parametrize("step_size,input_size", [(None, None), (1, None), (2, None), (1, 4), (2, 4)])
def test_backtest_splits_pd(step_size, input_size):
    n_series = 20
    min_length = 100
    max_length = 1000
    series = generate_series(n_series, freq="D", min_length=min_length, max_length=max_length)
    _test_backtest_splits(series, n_windows=3, h=14, step_size=step_size, input_size=input_size)


def test_backtest_splits_polars_alignment():
    h = 10
    n_series = 20
    series_pl = generate_series(n_series, freq="D", min_length=100, max_length=1000, engine="polars")
    splits = backtest_splits(series_pl, n_windows=3, h=h, id_col="unique_id", time_col="ds", freq="1d")

    for cutoffs, train, valid in splits:
        train_ends = train.group_by("unique_id", maintain_order=True).agg(pl.col("ds").max())
        valid_starts = valid.group_by("unique_id", maintain_order=True).agg(pl.col("ds").min())
        valid_ends = valid.group_by("unique_id", maintain_order=True).agg(pl.col("ds").max())

        expected_valid_starts = offset_times(train_ends["ds"], "1d", 1)
        expected_valid_ends = offset_times(train_ends["ds"], "1d", h)

        pl.testing.assert_series_equal(valid_starts["ds"], expected_valid_starts)
        pl.testing.assert_series_equal(valid_ends["ds"], expected_valid_ends)


@pytest.mark.parametrize("engine", ["pandas"] + (["polars"] if POLARS_INSTALLED else []))
def test_backtest_splits_sorted_uses_fast_path(monkeypatch, engine):
    """Sorted inputs should use the lower-memory boundary-based splitter."""
    series = generate_series(20, freq="D", min_length=100, max_length=300, engine=engine)
    n_windows = 3
    freq = pd.offsets.Day() if engine == "pandas" else "1d"
    fast_calls = 0
    orig_fast = ufp._single_split_sorted

    def counting_fast(*args, **kwargs):
        nonlocal fast_calls
        fast_calls += 1
        return orig_fast(*args, **kwargs)

    def fail_mask_path(*args, **kwargs):
        raise AssertionError("sorted inputs should not use the mask-based splitter")

    def fail_sort_indices(*args, **kwargs):
        raise AssertionError("backtest_splits should only check sortedness")

    monkeypatch.setattr(ufp, "maybe_compute_sort_indices", fail_sort_indices)
    monkeypatch.setattr(ufp, "_single_split_sorted", counting_fast)
    monkeypatch.setattr(ufp, "_single_split", fail_mask_path)

    splits = list(
        backtest_splits(
            series,
            n_windows=n_windows,
            h=14,
            id_col="unique_id",
            time_col="ds",
            freq=freq,
        )
    )

    assert len(splits) == n_windows
    assert fast_calls == n_windows


@pytest.mark.parametrize("engine", ["pandas"] + (["polars"] if POLARS_INSTALLED else []))
def test_backtest_splits_unsorted_falls_back_to_mask_path(monkeypatch, engine):
    """Unsorted inputs should preserve the original mask-based behavior."""
    series = generate_series(20, freq="D", min_length=100, max_length=300, engine=engine)
    if engine == "pandas":
        permuted = series.sample(frac=1.0, random_state=0)
        freq = pd.offsets.Day()
    else:
        permuted = series.sample(fraction=1.0, shuffle=True, seed=0)
        freq = "1d"
    n_windows = 3
    mask_calls = 0
    orig_mask = ufp._single_split

    def counting_mask(*args, **kwargs):
        nonlocal mask_calls
        mask_calls += 1
        return orig_mask(*args, **kwargs)

    def fail_fast_path(*args, **kwargs):
        raise AssertionError("unsorted inputs should not use the fast splitter")

    def fail_sort_indices(*args, **kwargs):
        raise AssertionError("backtest_splits should not materialize sort indices")

    monkeypatch.setattr(ufp, "maybe_compute_sort_indices", fail_sort_indices)
    monkeypatch.setattr(ufp, "_single_split", counting_mask)
    monkeypatch.setattr(ufp, "_single_split_sorted", fail_fast_path)

    splits = list(
        backtest_splits(
            permuted,
            n_windows=n_windows,
            h=14,
            id_col="unique_id",
            time_col="ds",
            freq=freq,
        )
    )

    assert len(splits) == n_windows
    assert mask_calls == n_windows


@pytest.mark.parametrize("engine", ["pandas"] + (["polars"] if POLARS_INSTALLED else []))
def test_backtest_splits_sorted_matches_original(engine):
    series = generate_series(
        20, freq="D", min_length=100, max_length=300, engine=engine
    )
    freq = pd.offsets.Day() if engine == "pandas" else "1d"
    kwargs = dict(
        n_windows=3,
        h=14,
        id_col="unique_id",
        time_col="ds",
        freq=freq,
    )
    if engine == "pandas":
        fast = [
            (train.reset_index(drop=True), valid.reset_index(drop=True))
            for _, train, valid in backtest_splits(series, **kwargs)
        ]
        shuffled = series.sample(frac=1.0, random_state=0)
        slow = [
            (
                train.sort_values(["unique_id", "ds"]).reset_index(drop=True),
                valid.sort_values(["unique_id", "ds"]).reset_index(drop=True),
            )
            for _, train, valid in backtest_splits(shuffled, **kwargs)
        ]

        for (fast_train, fast_valid), (slow_train, slow_valid) in zip(fast, slow):
            pd.testing.assert_frame_equal(fast_train, slow_train)
            pd.testing.assert_frame_equal(fast_valid, slow_valid)
    else:
        fast = [
            (
                train.sort(["unique_id", "ds"]),
                valid.sort(["unique_id", "ds"]),
            )
            for _, train, valid in backtest_splits(series, **kwargs)
        ]
        shuffled = series.sample(fraction=1.0, shuffle=True, seed=0)
        slow = [
            (
                train.sort(["unique_id", "ds"]),
                valid.sort(["unique_id", "ds"]),
            )
            for _, train, valid in backtest_splits(shuffled, **kwargs)
        ]

        for (fast_train, fast_valid), (slow_train, slow_valid) in zip(fast, slow):
            pl.testing.assert_frame_equal(fast_train, slow_train)
            pl.testing.assert_frame_equal(fast_valid, slow_valid)


def test_add_insample_levels_agreement():
    series = generate_series(100, n_models=2)
    models = ["model0", "model1"]
    levels = [80, 95]
    with_levels = add_insample_levels(series, models, levels)

    for model in models:
        for lvl in levels:
            assert (with_levels[f"{model}-lo-{lvl}"].lt(with_levels[f"{model}-hi-{lvl}"])).all()

    series_pl = generate_series(100, n_models=2, engine="polars")
    with_levels_pl = add_insample_levels(series_pl, models, levels)
    pd.testing.assert_frame_equal(
        with_levels.drop(columns="unique_id"),
        with_levels_pl.to_pandas().drop(columns="unique_id"),
    )


def _grouped_rows(block_order):
    return [(i, t) for i in block_order for t in range(3)]


_SORT_LAYOUTS = {
    "sorted": (_grouped_rows(sorted(range(12), key=str)), True),
    "numbered": (_grouped_rows(range(12)), True),
    "random_blocks": (_grouped_rows(np.random.default_rng(0).permutation(12)), True),
    "shuffled": (
        [_grouped_rows(range(12))[i] for i in np.random.default_rng(0).permutation(36)],
        False,
    ),
    "interleaved": ([(i, t) for t in range(3) for i in range(12)], False),
    "split_run": ([(1, 0), (1, 1), (0, 0), (1, 2), (2, 0)], False),
    "duplicated": ([(1, 0), (1, 1), (1, 1), (0, 0), (0, 1)], False),
    "descending": ([(1, 1), (1, 0), (0, 0), (0, 1)], False),
    "single_row": ([(3, 0)], False),
    "empty": ([], False),
}


@pytest.mark.skipif(not POLARS_INSTALLED, reason="polars not installed")
@pytest.mark.parametrize("layout", _SORT_LAYOUTS)
@pytest.mark.parametrize("id_dtype", ["str", "cat", "enum", "int"])
@pytest.mark.parametrize("time_dtype", ["int", "datetime"])
def test_maybe_compute_sort_indices_polars(layout, id_dtype, time_dtype):
    rows, grouped = _SORT_LAYOUTS[layout]
    if id_dtype == "int":
        ids = pl.Series([i for i, _ in rows], dtype=pl.Int64)
    else:
        str_ids = [f"id_{i}" for i, _ in rows]
        dtype = {
            "str": pl.String,
            "cat": pl.Categorical,
            "enum": pl.Enum([f"id_{i}" for i in range(12)][::-1]),
        }[id_dtype]
        ids = pl.Series(str_ids, dtype=dtype)
    df = pl.DataFrame(
        {"unique_id": ids, "ds": pl.Series([t for _, t in rows], dtype=pl.Int64)}
    ).with_columns(y=pl.col("ds").cast(pl.Float64))
    if time_dtype == "datetime":
        df = df.with_columns(ds=pl.datetime(2000, 1, 1) + pl.duration(days="ds"))

    res = ufp.maybe_compute_sort_indices(df, "unique_id", "ds")
    if ufp._is_sorted(df, "unique_id", "ds"):
        assert res is None
        return
    fast = ufp._polars_run_sort_indices(df["unique_id"], df["ds"])
    assert (fast is not None) == grouped
    expected = df.select(pl.arg_sort_by(["unique_id", "ds"])).to_series().to_numpy()
    assert res.dtype == expected.dtype
    np.testing.assert_array_equal(res, expected)
    if grouped:
        processed = process_df(df, "unique_id", "ds", "y")
        expected_processed = process_df(df[res], "unique_id", "ds", "y")
        assert expected_processed.sort_idxs is None
        pl.testing.assert_series_equal(processed.uids, expected_processed.uids)
        for got, exp in zip(processed[1:-1], expected_processed[1:-1]):
            np.testing.assert_array_equal(got, exp)


@pytest.mark.parametrize("layout", _SORT_LAYOUTS)
@pytest.mark.parametrize("id_dtype", ["str", "pyarrow_str", "cat", "int"])
@pytest.mark.parametrize("time_dtype", ["int", "datetime", "pyarrow_date"])
def test_maybe_compute_sort_indices_pandas(monkeypatch, layout, id_dtype, time_dtype):
    rows, grouped = _SORT_LAYOUTS[layout]
    if id_dtype == "int":
        ids = pd.Series([i for i, _ in rows], dtype="int64")
    else:
        str_ids = [f"id_{i}" for i, _ in rows]
        dtype = {
            "str": object,
            "pyarrow_str": "string[pyarrow]",
            "cat": pd.CategoricalDtype([f"id_{i}" for i in range(12)][::-1]),
        }[id_dtype]
        ids = pd.Series(str_ids, dtype=dtype)
    times = pd.Series([t for _, t in rows], dtype="int64")
    if time_dtype != "int":
        times = pd.Timestamp("2000-01-01") + pd.to_timedelta(times, unit="D")
    if time_dtype == "pyarrow_date":
        times = times.astype("date32[pyarrow]")
    df = pd.DataFrame(
        {"unique_id": ids, "ds": times, "y": np.arange(len(rows), dtype=float)}
    )

    res = ufp.maybe_compute_sort_indices(df, "unique_id", "ds")
    if ufp._is_sorted(df, "unique_id", "ds"):
        assert res is None
        return
    np_ids = df["unique_id"].cat.codes if id_dtype == "cat" else df["unique_id"]
    fast = ufp._pandas_run_sort_indices(np_ids.to_numpy(), df["ds"].to_numpy())
    assert (fast is not None) == grouped
    monkeypatch.setattr(ufp, "_pandas_run_sort_indices", lambda *_: None)
    expected = ufp.maybe_compute_sort_indices(df, "unique_id", "ds")
    assert res.dtype == expected.dtype
    np.testing.assert_array_equal(res, expected)
