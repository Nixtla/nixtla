import pandas as pd
import pytest

from nixtla.date_features import CountryHolidays, SpecialDates


@pytest.fixture
def periods():
    return 365 * 5


@pytest.fixture
def dates(periods):
    return pd.date_range(end="2023-09-01", periods=periods)


@pytest.fixture
def country_holidays():
    return CountryHolidays(countries=["US", "MX"])


@pytest.fixture
def special_dates():
    return SpecialDates(
        special_dates={
            "Important Dates": ["2021-02-26", "2020-02-26"],
            "Very Important Dates": ["2021-01-26", "2020-01-26", "2019-01-26"],
        }
    )


def test_country_holidays_shape(country_holidays, dates, periods):
    holidays_df = country_holidays(dates)
    assert len(holidays_df) == periods


def test_special_dates_shape_and_sum(special_dates, dates, periods):
    holidays_df = special_dates(dates)
    assert len(holidays_df) == periods
    assert holidays_df.sum().sum() == 5
