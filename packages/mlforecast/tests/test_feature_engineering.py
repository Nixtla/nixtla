import numpy as np
import pandas as pd
import polars as pl

from mlforecast.feature_engineering import transform_exog
from mlforecast.lag_transforms import ExpandingMean
from mlforecast.utils import generate_daily_series


def test_transform_exog():
    rng = np.random.RandomState(0)
    series = generate_daily_series(100, equal_ends=True)
    starts_ends = series.groupby("unique_id", observed=True, as_index=False)["ds"].agg(
        ["min", "max"]
    )
    prices = []
    for r in starts_ends.itertuples():
        dates = pd.date_range(r.min, r.max + 14 * pd.offsets.Day())
        df = pd.DataFrame({"ds": dates, "price": rng.rand(dates.size)})
        df["unique_id"] = r.Index
        prices.append(df)
    prices = pd.concat(prices)
    prices["price2"] = prices["price"] * rng.rand(prices.shape[0])

    transformed = transform_exog(prices, lags=[1, 2], lag_transforms={1: [ExpandingMean()]})

    prices_pl = pl.from_pandas(prices)
    transformed_pl = transform_exog(
        prices_pl,
        lags=[1, 2],
        lag_transforms={1: [ExpandingMean()]},
        num_threads=2,
    )
    pd.testing.assert_frame_equal(transformed, transformed_pl.to_pandas())
