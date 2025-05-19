import pandas as pd

from nixtla.date_features import CountryHolidays, SpecialDates

def test_country_holidays_shape():
    c_holidays = CountryHolidays(countries=['US', 'MX'])
    periods = 365 * 5
    dates = pd.date_range(end='2023-09-01', periods=periods)
    holidays_df = c_holidays(dates)
    assert len(holidays_df) == periods

def test_special_dates_shape_and_sum():
    special_dates = SpecialDates(
        special_dates={
            'Important Dates': ['2021-02-26', '2020-02-26'],
            'Very Important Dates': ['2021-01-26', '2020-01-26', '2019-01-26']
        }
    )
    periods = 365 * 5
    dates = pd.date_range(end='2023-09-01', periods=periods)
    holidays_df = special_dates(dates)
    assert len(holidays_df) == periods
    assert holidays_df.sum().sum() == 5