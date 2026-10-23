import warnings

import numpy as np
import pandas as pd
import pytest

from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.spark.xgb import SparkXGBForecast
from mlforecast.utils import generate_daily_series

warnings.simplefilter("ignore", FutureWarning)


def test_spark_distributed_forecast(spark_session):
    """Basic fit + predict smoke test for the Spark engine."""
    series = generate_daily_series(5, equal_ends=True, min_length=50, max_length=100)
    spark_series = spark_session.createDataFrame(series)

    fcst = DistributedMLForecast(
        models=[SparkXGBForecast(random_state=0)],
        freq="D",
        lags=[1, 2, 3],
        date_features=["dayofweek"],
        engine=spark_session,
    )
    fcst.fit(spark_series, static_features=[])
    preds = fcst.predict(10).toPandas()

    assert preds.shape[0] == 5 * 10
    assert {"unique_id", "ds", "SparkXGBForecast"}.issubset(preds.columns)


def test_spark_distributed_forecast_with_x_df(spark_session):
    """predict() with X_df as a Spark DataFrame must give the same result as pandas X_df."""
    h = 7
    series = generate_daily_series(5, equal_ends=True, min_length=50, max_length=100)
    rng = np.random.default_rng(42)
    series["price"] = rng.random(len(series))
    spark_series = spark_session.createDataFrame(series)

    fcst = DistributedMLForecast(
        models=[SparkXGBForecast(random_state=0)],
        freq="D",
        lags=[1, 2, 7],
        date_features=["dayofweek"],
        engine=spark_session,
    )
    fcst.fit(spark_series, static_features=[])

    last_dates = (
        series.groupby("unique_id")["ds"]
        .max()
        .reset_index()
        .rename(columns={"ds": "last_ds"})
    )
    future_rows = []
    for _, row in last_dates.iterrows():
        for step in range(1, h + 1):
            future_rows.append(
                {
                    "unique_id": row["unique_id"],
                    "ds": row["last_ds"] + pd.Timedelta(days=step),
                    "price": rng.random(),
                }
            )
    future_pd = pd.DataFrame(future_rows)
    spark_x_df = spark_session.createDataFrame(future_pd)

    preds_pandas = fcst.predict(h, X_df=future_pd).toPandas()
    preds_spark = fcst.predict(h, X_df=spark_x_df).toPandas()

    pd.testing.assert_frame_equal(
        preds_pandas.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        preds_spark.sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )


def test_spark_distributed_forecast_weight_col(spark_session):
    """fit() with weight_col must produce different predictions from unweighted fit."""
    base_series = generate_daily_series(5, equal_ends=True, min_length=50, max_length=50)
    weights = np.tile([1.0, 1.0, 1.0, 1.0, 100.0], len(base_series) // 5 + 1)[: len(base_series)]

    def _fit_predict(weighted: bool) -> pd.DataFrame:
        data = base_series.copy()
        if weighted:
            data["weight"] = weights
        spark_df = spark_session.createDataFrame(data)
        fcst = DistributedMLForecast(
            models=[SparkXGBForecast(random_state=0)],
            freq="D",
            lags=[1],
            engine=spark_session,
        )
        weight_col = "weight" if weighted else None
        fcst.fit(spark_df, static_features=[], weight_col=weight_col)
        return fcst.predict(5).toPandas()

    preds_unweighted = _fit_predict(weighted=False)
    preds_weighted = _fit_predict(weighted=True)

    assert not preds_unweighted["SparkXGBForecast"].equals(preds_weighted["SparkXGBForecast"])
