import pytest

import pandas as pd

@pytest.mark.parametrize(
    "df_converter, freq",
    [
        pytest.param(
            lambda series, with_gaps: with_gaps,
            '5min',
            id="gaps"
        ),
        pytest.param(
            lambda series, with_gaps: pd.concat([series, series]),
            '5min',
            id="duplicates"
        ),
        pytest.param(
            lambda series, with_gaps: series,
            '1min',
            id="wrong_freq"
        ),
    ]
)
def test_forecast_with_error(series_with_gaps, nixtla_test_client, df_converter, freq):
    series, with_gaps = series_with_gaps

    with pytest.raises(ValueError, match='missing or duplicate timestamps, or the timestamps do not match'):
        nixtla_test_client.forecast(df=df_converter(series, with_gaps), h=1, freq=freq)