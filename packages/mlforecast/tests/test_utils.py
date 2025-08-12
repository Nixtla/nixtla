import pytest

from mlforecast.utils import generate_daily_series, generate_prices_for_series

from .conftest import assert_raises_with_message


@pytest.fixture
def setup():
    n_series = 20
    min_length = 100
    max_length = 1000
    n_static_features = 2

    return n_series, min_length, max_length, n_static_features


def test_generate_daily_series(setup):
    n_series, min_length, max_length, n_static_features = setup

    series_with_statics = generate_daily_series(
        n_series, min_length, max_length, n_static_features
    )

    for i in range(n_static_features):
        assert all(
            series_with_statics.groupby("unique_id")[f"static_{i}"].nunique() == 1
        )
    assert series_with_statics.groupby("unique_id")["ds"].max().nunique() > 1


def test_generate_daily_series_equal_ends(setup):
    n_series, min_length, max_length, n_static_features = setup

    series = generate_daily_series(n_series, min_length, max_length)

    series_equal_ends = generate_daily_series(
        n_series, min_length, max_length, n_static_features, equal_ends=True
    )
    assert series_equal_ends.groupby("unique_id")["ds"].max().nunique() == 1
    series_for_prices = generate_daily_series(20, n_static_features=2, equal_ends=True)
    series_for_prices.rename(columns={"static_1": "product_id"}, inplace=True)
    prices_catalog = generate_prices_for_series(series_for_prices, horizon=7)

    assert set(prices_catalog["unique_id"]) == set(series_for_prices["unique_id"])
    assert_raises_with_message(lambda: generate_prices_for_series(series), "equal ends")
