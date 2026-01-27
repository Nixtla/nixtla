import warnings
from datetime import date, datetime
from itertools import product

import numpy as np
import pandas as pd
import polars as pl
import pytest

from utilsforecast.preprocessing import fill_gaps


@pytest.fixture
def pandas_datetime_df():
    """DataFrame with datetime ds column for pandas."""
    return pd.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": pd.to_datetime(["2020", "2021", "2023", "2021", "2022"]),
            "y": np.arange(5),
        }
    )


@pytest.fixture
def pandas_int_df():
    """DataFrame with integer ds column for pandas."""
    return pd.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": [2020, 2021, 2023, 2021, 2022],
            "y": np.arange(5),
        }
    )


@pytest.fixture
def polars_datetime_df():
    """DataFrame with datetime ds column for polars."""
    return pl.DataFrame(
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


@pytest.fixture
def polars_date_df():
    """DataFrame with date ds column for polars."""
    return pl.DataFrame(
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


@pytest.fixture
def polars_int_df():
    """DataFrame with integer ds column for polars."""
    return pl.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": [2020, 2021, 2023, 2021, 2022],
            "y": np.arange(5),
        }
    )


@pytest.fixture
def polars_ms_df(polars_datetime_df):
    """DataFrame with millisecond precision datetime for polars."""
    return polars_datetime_df.with_columns(pl.col("ds").cast(pl.Datetime(time_unit="ms")))


@pytest.fixture
def warning_df():
    """DataFrame where last value doesn't meet frequency (year start)."""
    return pd.DataFrame(
        {
            "unique_id": [0, 0, 0, 1, 1],
            "ds": pd.to_datetime(["2020-01", "2021-01", "2023-01", "2021-01", "2022-02"]),
            "y": np.arange(5),
        }
    )


# --- Helper functions ---


def get_pandas_freqs():
    """Get list of pandas frequency aliases with version compatibility."""
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
        # Older pandas version - use old aliases
        freqs = [
            f.replace("YE", "Y").replace("ME", "M").replace("h", "H").replace("QE", "Q")
            for f in freqs
            if isinstance(f, str)
        ]
    return freqs


def get_polars_freqs():
    """Get list of polars frequency aliases."""
    return ["1d", "1w", "1mo", "1q", "1y"]


def get_integer_freqs():
    """Get list of integer frequencies."""
    return list(range(1, 11))


# --- Basic fill_gaps tests ---


class TestFillGapsBasic:
    """Basic tests for fill_gaps function."""

    def test_fill_gaps_pandas_datetime_per_serie(self, pandas_datetime_df):
        """Test fill_gaps with pandas datetime and per_serie end."""
        result = fill_gaps(pandas_datetime_df, freq="YS", end="per_serie")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > len(pandas_datetime_df)

    def test_fill_gaps_pandas_datetime_specific_end(self, pandas_datetime_df):
        """Test fill_gaps with pandas datetime and specific end date."""
        result = fill_gaps(pandas_datetime_df, freq="YS", end="2024")
        assert isinstance(result, pd.DataFrame)

    def test_fill_gaps_pandas_datetime_global_start(self, pandas_datetime_df):
        """Test fill_gaps with pandas datetime and global start."""
        result = fill_gaps(pandas_datetime_df, freq="YS", start="global")
        assert isinstance(result, pd.DataFrame)

    def test_fill_gaps_pandas_datetime_specific_start(self, pandas_datetime_df):
        """Test fill_gaps with pandas datetime and specific start date."""
        result = fill_gaps(pandas_datetime_df, freq="YS", start="2019")
        assert isinstance(result, pd.DataFrame)

    def test_fill_gaps_pandas_int(self, pandas_int_df):
        """Test fill_gaps with pandas integer ds column."""
        result = fill_gaps(pandas_int_df, freq=1, start=2019, end=2024)
        assert isinstance(result, pd.DataFrame)

    def test_fill_gaps_polars_datetime(self, polars_datetime_df):
        """Test fill_gaps with polars datetime."""
        result = fill_gaps(
            polars_datetime_df,
            freq="1y",
            start=datetime(2019, 1, 1),
            end=datetime(2024, 1, 1),
        )
        assert isinstance(result, pl.DataFrame)

    def test_fill_gaps_polars_datetime_ms(self, polars_ms_df):
        """Test fill_gaps preserves millisecond precision for polars."""
        result = fill_gaps(
            polars_ms_df,
            freq="1y",
            start=datetime(2019, 1, 1),
            end=datetime(2024, 1, 1),
        )
        assert result.schema["ds"].time_unit == "ms"

    def test_fill_gaps_polars_date(self, polars_date_df):
        """Test fill_gaps with polars date type."""
        result = fill_gaps(
            polars_date_df,
            freq="1y",
            start=date(2020, 1, 1),
            end=date(2024, 1, 1),
        )
        assert isinstance(result, pl.DataFrame)

    def test_fill_gaps_polars_int(self, polars_int_df):
        """Test fill_gaps with polars integer ds column."""
        result = fill_gaps(polars_int_df, freq=1, start=2019, end=2024)
        assert isinstance(result, pl.DataFrame)


class TestFillGapsWarning:
    """Test fill_gaps warning behavior."""

    def test_fill_gaps_warns_on_lost_values(self, warning_df):
        """Test that fill_gaps warns when values are lost due to frequency mismatch."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fill_gaps(warning_df, "YS")
            assert len(w) > 0
            assert "values were lost" in str(w[0].message)


# --- Comprehensive fill_gaps tests ---


N_PERIODS = 100


def generate_test_dates(freq, n_periods):
    """Generate test dates for a given frequency."""
    if isinstance(freq, (pd.offsets.BaseOffset, str)):
        if isinstance(freq, str):
            try:
                delta = pd.Timedelta(freq)
                if delta.days > 0:
                    tz = None
                else:
                    tz = "Europe/Berlin"
            except ValueError:
                tz = None
        else:
            tz = None
        dates = pd.date_range("1950-01-01", periods=n_periods, freq=freq, tz=tz)
    else:
        dates = np.arange(0, freq * n_periods, freq, dtype=np.int64)
    return dates


def create_test_data(dates, n_periods, include_start, include_end):
    """Create test DataFrame with random date selections."""
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
    return pd.DataFrame(
        {
            "unique_id": np.repeat([1, 2], n_periods // 2),
            "ds": dates[date_idxs],
            "y": np.arange(n_periods, dtype=np.float64),
        }
    )


def verify_fill_results(data, filled, freq, start, end):
    """Verify fill_gaps results are correct."""
    data_starts_ends = data.groupby("unique_id", observed=True)["ds"].agg(["min", "max"])
    global_start = data_starts_ends["min"].min()
    global_end = data_starts_ends["max"].max()
    filled_starts_ends = filled.groupby("unique_id", observed=True)["ds"].agg(["min", "max"])

    # Inferred frequency is the expected
    first_serie = filled[filled["unique_id"] == 1]
    if isinstance(freq, str):
        if first_serie["ds"].dt.tz is not None:
            first_serie = first_serie.copy()
            first_serie["ds"] = first_serie["ds"].dt.tz_convert("UTC")
        inferred_freq = pd.infer_freq(first_serie["ds"].dt.tz_localize(None))
        assert inferred_freq == pd.tseries.frequencies.to_offset(freq)
    else:
        assert all(first_serie["ds"].diff().value_counts().index == [freq])

    # Fill keeps original data
    assert filled["y"].count() == N_PERIODS

    # Check starts
    if start == "per_serie":
        pd.testing.assert_series_equal(
            data_starts_ends["min"],
            filled_starts_ends["min"],
            check_dtype=False,
        )
    else:  # global or specific
        min_dates = filled_starts_ends["min"].unique()
        assert min_dates.size == 1
        expected_start = global_start if start == "global" else start
        assert min_dates[0] == expected_start

    # Check ends
    if end == "per_serie":
        pd.testing.assert_series_equal(
            data_starts_ends["max"],
            filled_starts_ends["max"],
            check_dtype=False,
        )
    else:  # global or specific
        max_dates = filled_starts_ends["max"].unique()
        assert max_dates.size == 1
        expected_end = global_end if end == "global" else end
        assert max_dates[0] == expected_end


class TestFillGapsComprehensive:
    """Comprehensive tests for fill_gaps with various frequencies and options."""

    @pytest.mark.parametrize("freq", get_pandas_freqs())
    @pytest.mark.parametrize("start_type", ["global", "per_serie", "specific"])
    @pytest.mark.parametrize("end_type", ["global", "per_serie", "specific"])
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_fill_gaps_comprehensive(self, freq, start_type, end_type, include_start, include_end):
        """Test fill_gaps with various frequency, start, and end combinations."""
        dates = generate_test_dates(freq, N_PERIODS)

        if isinstance(freq, (pd.offsets.BaseOffset, str)):
            offset = pd.tseries.frequencies.to_offset(freq)
        else:
            offset = freq

        global_start = dates[0]
        global_end = dates[-1]

        # Map start_type to actual start value
        if start_type == "global":
            start = "global"
        elif start_type == "per_serie":
            start = "per_serie"
        else:
            start = global_start - offset

        # Map end_type to actual end value
        if end_type == "global":
            end = "global"
        elif end_type == "per_serie":
            end = "per_serie"
        else:
            end = global_end + offset

        data = create_test_data(dates, N_PERIODS, include_start, include_end)
        filled = fill_gaps(data, freq, start=start, end=end)
        verify_fill_results(data, filled, freq, start, end)


# --- Error tests for incompatible frequency and time column ---


def create_error_test_data(dates, n_periods, include_start, include_end, lib):
    """Create test data for error tests."""
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
        return pd.DataFrame(
            {
                "unique_id": np.repeat([1, 2], n_periods // 2),
                "ds": dates[date_idxs],
                "y": np.arange(n_periods, dtype=np.float64),
            }
        )
    else:  # polars
        return pl.DataFrame(
            {
                "unique_id": np.repeat([1, 2], n_periods // 2),
                "ds": dates[date_idxs],
                "y": np.arange(n_periods, dtype=np.float64),
            }
        )


class TestFillGapsIncompatibleFrequency:
    """Test fill_gaps raises errors for incompatible frequency and time column."""

    @pytest.fixture
    def pandas_freq_data(self):
        """Generate date data for pandas frequency tests."""
        freqs_pd = [
            "YE", "YS", "ME", "MS", "W", "W-TUE", "D", "s", "ms",
            "20D", "30s", "2YE", "3YS", "30min", "B", "1h", "QS-NOV", "QE",
        ]
        # Handle version compatibility
        try:
            pd.tseries.frequencies.to_offset("YE")
        except ValueError:
            freqs_pd = [
                f.replace("YE", "Y").replace("ME", "M").replace("h", "H").replace("QE", "Q")
                for f in freqs_pd
            ]
        return freqs_pd

    @pytest.fixture
    def polars_freq_data(self):
        """Generate date data for polars frequency tests."""
        return ["1d", "1w", "1mo", "1q", "1y"]

    @pytest.mark.parametrize("freq_int", list(range(1, 11)))
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_pandas_int_freq_with_datetime_data(self, pandas_freq_data, freq_int, include_start, include_end):
        """Test that integer freq with datetime data raises ValueError for pandas."""
        for freq_str in pandas_freq_data:
            dates_str = pd.date_range("1950-01-01", periods=N_PERIODS, freq=freq_str)
            offset = pd.tseries.frequencies.to_offset(freq_str)
            first_date = dates_str[0] - offset
            last_date = dates_str[-1] + offset

            starts = ["global", "per_serie", first_date]
            ends = ["global", "per_serie", last_date]

            for start, end in product(starts, ends):
                data = create_error_test_data(dates_str, N_PERIODS, include_start, include_end, "pandas")
                with pytest.raises(ValueError):
                    fill_gaps(data, freq_int, start=start, end=end)

    @pytest.mark.parametrize("freq_int", list(range(1, 11)))
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_pandas_str_freq_with_int_data(self, pandas_freq_data, freq_int, include_start, include_end):
        """Test that string freq with int data raises ValueError for pandas."""
        dates_int = np.arange(1, (N_PERIODS * freq_int) + 1, freq_int)

        for freq_str in pandas_freq_data:
            # Use integer boundaries for int data
            first_int = dates_int[0] - freq_int
            last_int = dates_int[-1] + freq_int

            starts = ["global", "per_serie", first_int]
            ends = ["global", "per_serie", last_int]

            for start, end in product(starts, ends):
                data = create_error_test_data(dates_int, N_PERIODS, include_start, include_end, "pandas")
                with pytest.raises(ValueError):
                    fill_gaps(data, freq_str, start=start, end=end)

    @pytest.mark.parametrize("freq_int", list(range(1, 11)))
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_polars_int_freq_with_datetime_data(self, polars_freq_data, freq_int, include_start, include_end):
        """Test that integer freq with datetime data raises ValueError for polars."""
        for freq_str in polars_freq_data:
            pl_dt = pl.date(1950, 1, 1)
            dates_str = pl.date_range(
                pl_dt,
                pl_dt.dt.offset_by(f"{N_PERIODS}{freq_str[1:]}"),
                interval=freq_str,
                eager=True,
            )
            first_date = dates_str.dt.offset_by(f"-{freq_str}")[0]
            last_date = dates_str.dt.offset_by(freq_str)[-1]

            starts = ["global", "per_serie", first_date]
            ends = ["global", "per_serie", last_date]

            for start, end in product(starts, ends):
                data = create_error_test_data(dates_str, N_PERIODS, include_start, include_end, "polars")
                with pytest.raises(ValueError):
                    fill_gaps(data, freq_int, start=start, end=end)

    @pytest.mark.parametrize("freq_int", list(range(1, 11)))
    @pytest.mark.parametrize("include_start", [True, False])
    @pytest.mark.parametrize("include_end", [True, False])
    def test_polars_str_freq_with_int_data(self, polars_freq_data, freq_int, include_start, include_end):
        """Test that string freq with int data raises ValueError for polars."""
        dates_int = np.arange(1, (N_PERIODS * freq_int) + 1, freq_int)

        for freq_str in polars_freq_data:
            # Use integer boundaries for int data
            first_int = dates_int[0] - freq_int
            last_int = dates_int[-1] + freq_int

            starts = ["global", "per_serie", first_int]
            ends = ["global", "per_serie", last_int]

            for start, end in product(starts, ends):
                data = create_error_test_data(dates_int, N_PERIODS, include_start, include_end, "polars")
                with pytest.raises(ValueError):
                    fill_gaps(data, freq_str, start=start, end=end)
