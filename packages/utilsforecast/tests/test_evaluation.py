import sys
from functools import partial
from itertools import product

import dask.dataframe as dd
import datasetsforecast.losses as ds_losses
import fugue.api as fa
import pandas as pd
import polars as pl
import polars.testing
import pytest
from datasetsforecast.evaluation import accuracy as ds_evaluate
from pyspark.sql import SparkSession

import utilsforecast.processing as ufp
from utilsforecast.data import generate_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import (
    calibration,
    coverage,
    mae,
    mape,
    mase,
    mqloss,
    mse,
    quantile_loss,
    rmse,
    scaled_crps,
    smape,
)


@pytest.fixture
def setup_series():
    series = generate_series(10, n_models=2, level=[80, 95])
    series["unique_id"] = series["unique_id"].astype("int")
    return series


@pytest.fixture
def setup_models():
    return ["model0", "model1"]


@pytest.fixture
def setup_metrics():
    return [
        mae,
        mse,
        rmse,
        mape,
        smape,
        partial(mase, seasonality=7),
        quantile_loss,
        mqloss,
        coverage,
        calibration,
        scaled_crps,
    ]


def test_evaluate(setup_series, setup_models, setup_metrics):
    evaluation = evaluate(
        setup_series,
        metrics=setup_metrics,
        models=setup_models,
        train_df=setup_series,
        level=[80, 95],
    )
    summary = (
        evaluation.drop(columns="unique_id").groupby("metric").mean().reset_index()
    )

    series_pl = generate_series(10, n_models=2, level=[80, 95], engine="polars")
    pl_evaluation = evaluate(
        series_pl,
        metrics=setup_metrics,
        train_df=series_pl,
        level=[80, 95],
    ).drop("unique_id")
    pl_summary = ufp.group_by(pl_evaluation, "metric").mean()
    pd.testing.assert_frame_equal(
        summary.sort_values("metric"),
        pl_summary.sort("metric").to_pandas(),
    )
    pl.testing.assert_frame_equal(
        evaluate(
            series_pl,
            metrics=setup_metrics,
            train_df=series_pl,
            level=[80, 95],
            agg_fn="mean",
        ).sort("metric"),
        pl_summary.sort("metric"),
    )


def daily_mase(y, y_hat, y_train):
    return ds_losses.mase(y, y_hat, y_train, seasonality=7)


def test_datasets_evaluate(setup_series, setup_models, setup_metrics):
    level = [80, 95]
    for agg_fn in [None, "mean"]:
        uf_res = evaluate(
            setup_series,
            metrics=setup_metrics,
            models=setup_models,
            train_df=setup_series,
            level=level,
            agg_fn=agg_fn,
        )
        agg_by = None if agg_fn == "mean" else ["unique_id"]
        ds_res = ds_evaluate(
            setup_series,
            metrics=[
                ds_losses.mae,
                ds_losses.mse,
                ds_losses.rmse,
                ds_losses.mape,
                daily_mase,
                ds_losses.smape,
                ds_losses.quantile_loss,
                ds_losses.mqloss,
                ds_losses.coverage,
                ds_losses.calibration,
                ds_losses.scaled_crps,
            ],
            level=level,
            Y_df=setup_series,
            agg_by=agg_by,
        )
        ds_res["metric"] = ds_res["metric"].str.replace("-", "_")
        ds_res["metric"] = ds_res["metric"].str.replace("q_", "q")
        ds_res["metric"] = ds_res["metric"].str.replace("lv_", "level")
        ds_res["metric"] = ds_res["metric"].str.replace("daily_mase", "mase")
        # utils doesn't multiply pct metrics by 100
        ds_res.loc[
            ds_res["metric"].str.startswith("coverage"), ["model0", "model1"]
        ] /= 100
        ds_res.loc[ds_res["metric"].eq("mape"), ["model0", "model1"]] /= 100
        # we report smape between 0 and 1 instead of 0-200
        ds_res.loc[ds_res["metric"].eq("smape"), ["model0", "model1"]] /= 200

        ds_res = ds_res[uf_res.columns]
        if agg_fn is None:
            ds_res = ds_res.sort_values(["unique_id", "metric"])
            uf_res = uf_res.sort_values(["unique_id", "metric"])
        else:
            ds_res = ds_res.sort_values("metric")
            uf_res = uf_res.sort_values("metric")

        pd.testing.assert_frame_equal(
            uf_res.reset_index(drop=True),
            ds_res.reset_index(drop=True),
        )

@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_distributed_evaluate(setup_series):
    level = [80, 95]
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("FATAL")
    dask_df = dd.from_pandas(setup_series, npartitions=2)
    spark_df = spark.createDataFrame(setup_series).repartition(2)
    for distributed_df, use_train in product([dask_df, spark_df], [True, False]):
        distr_metrics = [rmse, mae]
        if use_train:
            distr_metrics.append(partial(mase, seasonality=7))
            local_train = setup_series
            distr_train = distributed_df
        else:
            local_train = None
            distr_train = None
        local_res = evaluate(
            setup_series, metrics=distr_metrics, level=level, train_df=local_train
        )
        distr_res = fa.as_fugue_df(
            evaluate(
                distributed_df,
                metrics=distr_metrics,
                level=level,
                train_df=distr_train,
            )
        ).as_pandas()
        pd.testing.assert_frame_equal(
            local_res.sort_values(["unique_id", "metric"]).reset_index(drop=True),
            distr_res.sort_values(["unique_id", "metric"]).reset_index(drop=True),
            check_dtype=False,
        )
