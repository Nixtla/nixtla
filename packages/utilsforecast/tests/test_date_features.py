import copy
import dataclasses
import pickle
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import utilsforecast.date_features as dtf


def _pandas_reference(dates: pd.Series, name: str) -> np.ndarray:
    if name == "week":
        return dates.dt.isocalendar().week.to_numpy()
    attr = {"day_of_week": "dayofweek", "day_of_year": "dayofyear"}.get(name, name)
    return getattr(dates.dt, attr).to_numpy()


def test_available_lists_every_exported_feature():
    features = dtf.available()
    assert [f.name for f in features] == [
        name for name in dtf.__all__ if name != "available"
    ]
    for feature in features:
        assert getattr(dtf, feature.name) is feature


def test_docs_table_matches_available():
    doc = Path(__file__).parents[1] / "docs" / "date_features.html.md"
    rows = [
        l for l in doc.read_text(encoding="utf-8").splitlines() if l.startswith("| `")
    ]
    assert rows == [
        f"| `{f.name}` | {f.description} | `{np.dtype(f.dtype).name}` |"
        for f in dtf.available()
    ]


@pytest.mark.parametrize(
    "dates",
    [
        # spans 1900 and 2100 (not leap years), 2000 (leap) and 53-week ISO years
        pd.Series(pd.date_range("1899-12-01", "2101-02-01", freq="D")),
        pd.Series(
            pd.date_range("1999-12-01", "2031-01-01", freq="7h", tz="America/New_York")
        ),
        pd.Series(
            pd.date_range("1999-12-01", "2031-01-01", freq="7h", tz="Asia/Tokyo")
        ),
        # pre-1970 times and the limits of nanosecond timestamps
        pd.Series(
            [
                pd.Timestamp.min,
                pd.Timestamp("1677-09-30 23:59:59"),
                pd.Timestamp("1969-12-31 23:59:59.5"),
                pd.Timestamp("2262-04-01"),
                pd.Timestamp.max,
            ]
        ),
    ],
    ids=["naive", "tz-behind-utc", "tz-ahead-of-utc", "ns-limits"],
)
@pytest.mark.parametrize(
    "container",
    [
        "pandas_series",
        "pandas_index",
        "numpy",
        "numpy_seconds",
        "pandas_array",
        "pandas_arrow",
        "pandas_arrow_date",
        "polars",
        "polars_date",
    ],
)
def test_compute_matches_pandas_attributes(dates, container):
    is_date = dates.dt.tz is None and (dates.dt.time == time()).all()
    if container.endswith("date") and not is_date:
        pytest.skip("dates have a time or time zone")
    if container.startswith("pandas_arrow") and not dtf._PANDAS_GE_2:
        pytest.skip("pandas<2 doesn't support pyarrow dates")
    if container == "pandas_index":
        inp = pd.DatetimeIndex(dates)
    elif container == "numpy":
        inp = dates.dt.tz_localize(None).to_numpy()
    elif container == "numpy_seconds":
        times = dates.dt.tz_localize(None).to_numpy()
        unit, _ = np.datetime_data(times.dtype)
        per_second = np.timedelta64(1, "s") // np.timedelta64(1, unit)
        inp = (times.view(np.int64) // per_second).astype("datetime64[s]")
    elif container == "pandas_array":
        inp = dates.array
    elif container == "pandas_arrow":
        tz = None if dates.dt.tz is None else str(dates.dt.tz)
        inp = dates.astype(pd.ArrowDtype(pa.timestamp(dates.dt.unit, tz=tz)))
    elif container == "pandas_arrow_date":
        inp = dates.astype("date32[pyarrow]")
    elif container == "polars":
        inp = pl.from_pandas(dates)
    elif container == "polars_date":
        inp = pl.from_pandas(dates).cast(pl.Date)
    else:
        inp = dates
    for feature in dtf.available():
        vals = feature.compute(inp)
        assert vals.dtype == feature.dtype
        np.testing.assert_array_equal(
            vals, _pandas_reference(dates, feature.name), err_msg=feature.name
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_null_dates_raise(engine):
    dates = pd.Series(pd.to_datetime(["2020-01-01", None, "2020-03-01"]))
    if engine == "polars":
        dates = pl.from_pandas(dates)
    with pytest.raises(ValueError, match="found null dates"):
        dtf.month.compute(dates)


@pytest.mark.parametrize(
    "dates",
    [
        pd.Series([1, 2, 3]),
        pl.Series([1, 2, 3]),
        pd.Series(pd.date_range("2020-01-01", periods=3).date),
        pd.Series([1, 2, 3], dtype="int64[pyarrow]"),
        ["2020-01-01", "2020-01-02"],
    ],
    ids=[
        "pandas_int",
        "polars_int",
        "pandas_date_objects",
        "pandas_arrow_int",
        "list_of_str",
    ],
)
def test_non_datetime_dates_raise(dates):
    with pytest.raises(ValueError, match="dates must be datetimes"):
        dtf.month.compute(dates)


def test_numpy_dates_outside_nanosecond_range():
    dates = np.array(["1000-03-01", "3000-12-31"], dtype="datetime64[s]")
    np.testing.assert_array_equal(dtf.year.compute(dates), [1000, 3000])
    np.testing.assert_array_equal(dtf.day_of_year.compute(dates), [60, 365])


@pytest.mark.parametrize("unit", ["10ms", "ps", "as"])
def test_numpy_dates_with_any_unit(unit):
    # attoseconds can only hold dates within a few seconds of 1970
    dates = np.array(
        ["1969-12-31T23:59:55", "1970-01-01T00:00:05"], dtype=f"datetime64[{unit}]"
    )
    np.testing.assert_array_equal(dtf.day.compute(dates), [31, 1])
    np.testing.assert_array_equal(dtf.second.compute(dates), [55, 5])


def test_pickling_keeps_identity():
    assert pickle.loads(pickle.dumps(dtf.month)) is dtf.month
    assert copy.deepcopy(dtf.day_of_week) is dtf.day_of_week


def test_pickling_unlisted_feature_raises():
    fiscal_month = dataclasses.replace(dtf.month, name="fiscal_month")
    assert copy.deepcopy(fiscal_month) is fiscal_month
    with pytest.raises(pickle.PicklingError, match="'fiscal_month'"):
        pickle.dumps(fiscal_month)
    with pytest.raises(pickle.PicklingError, match="'month'"):
        pickle.dumps(dataclasses.replace(dtf.month, dtype=np.int64))


@pytest.mark.parametrize(
    "dates",
    [pd.Series([], dtype="datetime64[ns]"), pl.Series([], dtype=pl.Datetime)],
    ids=["pandas", "polars"],
)
def test_empty_dates(dates):
    for feature in dtf.available():
        vals = feature.compute(dates)
        assert vals.shape == (0,)
        assert vals.dtype == feature.dtype


@pytest.mark.skipif(dtf._PANDAS_GE_2, reason="pandas>=2 supports pyarrow dates")
@pytest.mark.parametrize("tz", [None, "UTC"])
def test_pyarrow_dates_require_pandas_2(tz):
    dtype = pd.ArrowDtype(pa.timestamp("ns", tz=tz))
    dates = pd.Series(
        pa.array([datetime(2020, 1, 1)], dtype.pyarrow_dtype), dtype=dtype
    )
    with pytest.raises(ValueError, match="require pandas>=2"):
        dtf.month.compute(dates)
