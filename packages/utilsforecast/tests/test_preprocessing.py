import warnings
from datetime import date, datetime
from itertools import product

import numpy as np
import pandas as pd
import polars as pl

from utilsforecast.data import generate_series
from utilsforecast.preprocessing import fill_gaps

df = pd.DataFrame(
    {
        "unique_id": [0, 0, 0, 1, 1],
        "ds": pd.to_datetime(["2020", "2021", "2023", "2021", "2022"]),
        "y": np.arange(5),
    }
)
fill_gaps(
    df,
    freq="YS",
)
fill_gaps(
    df,
    freq="YS",
    end="per_serie",
)
fill_gaps(
    df,
    freq="YS",
    end="2024",
)
fill_gaps(df, freq="YS", start="global")
fill_gaps(
    df,
    freq="YS",
    start="2019",
)
df = pd.DataFrame(
    {
        "unique_id": [0, 0, 0, 1, 1],
        "ds": [2020, 2021, 2023, 2021, 2022],
        "y": np.arange(5),
    }
)

fill_gaps(
    df,
    freq=1,
    start=2019,
    end=2024,
)
df = pl.DataFrame(
    {
        "unique_id": [0, 0, 0, 1, 1],
        "ds": [
            datetime(2020, 1, 1),
            datetime(2022, 1, 1),
            datetime(2023, 1, 1),
            datetime(2021, 1, 1),
            datetime(2022, 1, 1),
        ],
        "y": np.arange(5),
    }
)
df
polars_ms = fill_gaps(
    df.with_columns(pl.col("ds").cast(pl.Datetime(time_unit="ms"))),
    freq="1y",
    start=datetime(2019, 1, 1),
    end=datetime(2024, 1, 1),
)


def test_fill_gaps_polars():
    assert polars_ms.schema["ds"].time_unit == "ms"


df = pl.DataFrame(
    {
        "unique_id": [0, 0, 0, 1, 1],
        "ds": [
            date(2020, 1, 1),
            date(2022, 1, 1),
            date(2023, 1, 1),
            date(2021, 1, 1),
            date(2022, 1, 1),
        ],
        "y": np.arange(5),
    }
)

fill_gaps(
    df,
    freq="1y",
    start=date(2020, 1, 1),
    end=date(2024, 1, 1),
)
df = pl.DataFrame(
    {
        "unique_id": [0, 0, 0, 1, 1],
        "ds": [2020, 2021, 2023, 2021, 2022],
        "y": np.arange(5),
    }
)

fill_gaps(
    df,
    freq=1,
    start=2019,
    end=2024,
)


def check_fill(dates, freq, start, end, include_start, include_end):
    base_idxs = []
    if include_start:
        base_idxs.append(0)
    if include_end:
        base_idxs.append(dates.size - 1)
    base_idxs = np.array(base_idxs, dtype=np.int64)
    date_idxs = np.hstack(
        [
            np.append(
                base_idxs,
                np.random.choice(
                    np.arange(1, dates.size - 1),
                    size=n_periods // 2 - len(base_idxs),
                    replace=False,
                ),
            )
            for _ in range(2)
        ],
    )
    data = pd.DataFrame(
        {
            "unique_id": np.repeat([1, 2], n_periods // 2),
            "ds": dates[date_idxs],
            "y": np.arange(n_periods, dtype=np.float64),
        }
    )
    filled = fill_gaps(data, freq, start=start, end=end)
    data_starts_ends = data.groupby("unique_id", observed=True)["ds"].agg(
        ["min", "max"]
    )
    global_start = data_starts_ends["min"].min()
    global_end = data_starts_ends["max"].max()
    filled_starts_ends = filled.groupby("unique_id", observed=True)["ds"].agg(
        ["min", "max"]
    )

    # inferred frequency is the expected
    first_serie = filled[filled["unique_id"] == 1]
    if isinstance(freq, str):
        if first_serie["ds"].dt.tz is not None:
            first_serie = first_serie.copy()
            first_serie["ds"] = first_serie["ds"].dt.tz_convert("UTC")
        inferred_freq = pd.infer_freq(first_serie["ds"].dt.tz_localize(None))
        assert inferred_freq == pd.tseries.frequencies.to_offset(freq)
    else:
        assert all(first_serie["ds"].diff().value_counts().index == [freq])

    # fill keeps original data
    assert filled["y"].count() == n_periods
    # check starts
    if start == "per_serie":
        pd.testing.assert_series_equal(
            data_starts_ends["min"],
            filled_starts_ends["min"],
        )
    else:  # global or specific
        min_dates = filled_starts_ends["min"].unique()
        assert min_dates.size == 1
        expected_start = global_start if start == "global" else start
        assert min_dates[0] == expected_start

    # check ends
    if end == "per_serie":
        pd.testing.assert_series_equal(
            data_starts_ends["max"],
            filled_starts_ends["max"],
        )
    else:  # global or specific
        max_dates = filled_starts_ends["max"].unique()
        assert max_dates.size == 1
        expected_end = global_end if end == "global" else end
        assert max_dates[0] == expected_end


n_periods = 100
freqs = [
    "YE",
    "YS",
    "ME",
    "MS",
    "W",
    "W-TUE",
    "D",
    "s",
    "ms",
    1,
    2,
    "20D",
    "30s",
    "2YE",
    "3YS",
    "30min",
    "B",
    "1h",
    "QS-NOV",
    "QE",
]
try:
    pd.tseries.frequencies.to_offset("YE")
except ValueError:
    freqs = [
        f.replace("YE", "Y").replace("ME", "M").replace("h", "H").replace("QE", "Q")
        for f in freqs
        if isinstance(f, str)
    ]
for freq in freqs:
    if isinstance(freq, (pd.offsets.BaseOffset, str)):
        offset = pd.tseries.frequencies.to_offset(freq)
        if isinstance(freq, str):
            try:
                delta = pd.Timedelta(freq)
                if delta.days > 0:
                    tz = None
                else:
                    tz = "Europe/Berlin"
            except ValueError:
                tz = None
        dates = pd.date_range("1950-01-01", periods=n_periods, freq=freq, tz=tz)
    else:
        dates = np.arange(0, freq * n_periods, freq, dtype=np.int64)
        offset = freq
    global_start = dates[0]
    global_end = dates[-1]
    starts = ["global", "per_serie", global_start - offset]
    ends = ["global", "per_serie", global_end + offset]
    include_starts = [True, False]
    include_ends = [True, False]
    iterable = product(starts, ends, include_starts, include_ends)
    for start, end, include_start, include_end in iterable:
        check_fill(dates, freq, start, end, include_start, include_end)
# last value doesn't meet frequency (year start)
dfx = pd.DataFrame(
    {
        "unique_id": [0, 0, 0, 1, 1],
        "ds": pd.to_datetime(["2020-01", "2021-01", "2023-01", "2021-01", "2022-02"]),
        "y": np.arange(5),
    }
)
with warnings.catch_warnings(record=True) as w:
    fill_gaps(dfx, "YS")
assert "values were lost" in str(w[0].message)


# frequency and time column are not compatible
def error_freq(dates, freq, start, end, include_start, include_end, lib):
    base_idxs = []
    if include_start:
        base_idxs.append(0)
    if include_end:
        base_idxs.append(np.size(dates) - 1)
    base_idxs = np.array(base_idxs, dtype=np.int64)
    date_idxs = np.hstack(
        [
            np.append(
                base_idxs,
                np.random.choice(
                    np.arange(1, np.size(dates) - 1),
                    size=n_periods // 2 - len(base_idxs),
                    replace=False,
                ),
            )
            for _ in range(2)
        ],
    )
    if lib == "pandas":
        data = pd.DataFrame(
            {
                "unique_id": np.repeat([1, 2], n_periods // 2),
                "ds": dates[date_idxs],
                "y": np.arange(n_periods, dtype=np.float64),
            }
        )

    if lib == "polars":
        data = pl.DataFrame(
            {
                "unique_id": np.repeat([1, 2], n_periods // 2),
                "ds": dates[date_idxs],
                "y": np.arange(n_periods, dtype=np.float64),
            }
        )

    try:
        filled = fill_gaps(data, freq, start=start, end=end)
    except Exception as e:
        assert isinstance(e, ValueError)


# https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
freqs_pd = [
    "YE",
    "YS",
    "ME",
    "MS",
    "W",
    "W-TUE",
    "D",
    "s",
    "ms",
    "20D",
    "30s",
    "2YE",
    "3YS",
    "30min",
    "B",
    "1h",
    "QS-NOV",
    "QE",
]

# https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.offset_by.html
freqs_pl = ["1d", "1w", "1mo", "1q", "1y"]

# integer freqs
freqs_int = list(range(1, 10 + 1))

n_periods = 100
for lib in ["pandas", "polars"]:
    freqs_list = freqs_pd if lib == "pandas" else freqs_pl
    for freq_int, freq_str in product(freqs_int, freqs_list):
        dates_int = np.arange(1, (n_periods * freq_int) + 1, freq_int)

        if lib == "pandas":
            dates_str = pd.date_range("1950-01-01", periods=n_periods, freq=freq_str)
            offset = pd.tseries.frequencies.to_offset(freq)
            first_date = dates_str[0] - offset
            last_date = dates_str[-1] + offset

        if lib == "polars":
            pl_dt = pl.date(1950, 1, 1)
            dates_str = pl.date_range(
                pl_dt,
                pl_dt.dt.offset_by(f"{n_periods}{freq_str[1:]}"),
                interval=freq_str,
                eager=True,
            )
            first_date = dates_str.dt.offset_by(f"-{freq_str}")[0]
            last_date = dates_str.dt.offset_by(freq_str)[-1]

        starts = ["global", "per_serie", first_date]
        ends = ["global", "per_serie", last_date]
        include_starts = [True, False]
        include_ends = [True, False]
        iterable = product(starts, ends, include_starts, include_ends)
        for start, end, include_start, include_end in iterable:
            error_freq(dates_str, freq_int, start, end, include_start, include_end, lib)
            error_freq(dates_int, freq_str, start, end, include_start, include_end, lib)
