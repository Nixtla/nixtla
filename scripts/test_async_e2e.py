#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end smoke test for the async job API (forecast_async / cross_validation_async /
finetune_async) against a live Nixtla deployment.

Requires NIXTLA_API_KEY / NIXTLA_BASE_URL (or a .env file) to point at a deployment
that has the async endpoints (`/v2/{forecast,cross_validation,finetune}/async`,
`/v2/{forecast,cross_validation,finetune}/jobs/{job_id}`) live.

Usage:
    python scripts/test_async_e2e.py [--spark] [--ray] [--skip-distributed]
                                      [--poll-interval SECONDS] [--poll-timeout SECONDS]
"""

import sys
from argparse import ArgumentParser

import pandas as pd

from nixtla.nixtla_client import NixtlaClient


def make_df(n_series: int = 3, n: int = 40) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"id_{i}",
                    "ds": pd.date_range("2023-01-01", periods=n, freq="D"),
                    "y": [10 + i + (j % 7) for j in range(n)],
                }
            )
            for i in range(n_series)
        ],
        ignore_index=True,
    )


def check_forecast_async(client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs):
    sync_fcst = client.forecast(df=df, h=h)
    async_fcst = client.forecast_async(df=df, h=h, **poll_kwargs)
    assert list(async_fcst.columns) == list(sync_fcst.columns), (
        f"column mismatch: {async_fcst.columns.tolist()} != {sync_fcst.columns.tolist()}"
    )
    assert len(async_fcst) == len(sync_fcst) == df["unique_id"].nunique() * h
    pd.testing.assert_frame_equal(async_fcst, sync_fcst)


def check_forecast_async_add_history(
    client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs
):
    fcst = client.forecast_async(df=df, h=h, add_history=True, **poll_kwargs)
    assert len(fcst) > df["unique_id"].nunique() * h, (
        "expected historical rows in addition to the h forecast rows per series"
    )


def check_cross_validation_async(
    client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs
):
    cv = client.cross_validation_async(df=df, h=h, n_windows=2, **poll_kwargs)
    assert {"unique_id", "ds", "cutoff", "y", "TimeGPT"}.issubset(cv.columns)
    assert len(cv) == df["unique_id"].nunique() * h * 2


def check_finetune_async(client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs):
    model_id = client.finetune_async(df=df, finetune_steps=2, **poll_kwargs)
    try:
        assert isinstance(model_id, str) and model_id
        fcst = client.forecast(df=df, h=h, finetuned_model_id=model_id)
        assert len(fcst) == df["unique_id"].nunique() * h
    finally:
        client.delete_finetuned_model(model_id)


def check_forecast_async_num_partitions(
    client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs
):
    fcst = client.forecast_async(df=df, h=h, num_partitions=2, **poll_kwargs)
    assert len(fcst) == df["unique_id"].nunique() * h


def check_cross_validation_async_num_partitions(
    client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs
):
    cv = client.cross_validation_async(df=df, h=h, num_partitions=2, **poll_kwargs)
    assert len(cv) == df["unique_id"].nunique() * h


def check_forecast_async_distributed_dask(
    client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs
):
    import dask.dataframe as dd
    import fugue.api as fa

    dask_df = dd.from_pandas(df, npartitions=2)
    fcst = client.forecast_async(df=dask_df, h=h, num_partitions=2, **poll_kwargs)
    fcst = fa.as_pandas(fcst)
    assert len(fcst) == df["unique_id"].nunique() * h


def check_forecast_async_distributed_spark(
    client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs
):
    import fugue.api as fa
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[2]").appName("nixtla-async-e2e").getOrCreate()
    try:
        spark_df = spark.createDataFrame(df).repartition(2)
        fcst = client.forecast_async(df=spark_df, h=h, num_partitions=2, **poll_kwargs)
        fcst = fa.as_pandas(fcst)
        assert len(fcst) == df["unique_id"].nunique() * h
    finally:
        spark.stop()


def check_forecast_async_distributed_ray(
    client: NixtlaClient, df: pd.DataFrame, h: int, **poll_kwargs
):
    import fugue.api as fa
    import ray

    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        ray_df = ray.data.from_pandas(df).repartition(2)
        fcst = client.forecast_async(df=ray_df, h=h, num_partitions=2, **poll_kwargs)
        fcst = fa.as_pandas(fcst)
        assert len(fcst) == df["unique_id"].nunique() * h
    finally:
        ray.shutdown()


def run_checks(checks, client, df, h, poll_kwargs) -> bool:
    passed = 0
    failed = 0
    for name, fn in checks:
        try:
            fn(client, df, h, **poll_kwargs)
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            print(f"[FAIL] {name}: {e}")
            failed += 1
        else:
            print(f"[OK] {name}")
            passed += 1
    print(f"\n{passed}/{passed + failed} checks passed")
    return failed == 0


if __name__ == "__main__":
    parser = ArgumentParser(description="End-to-end smoke test for the async job API")
    parser.add_argument(
        "--spark", action="store_true", help="Also run the Spark distributed check"
    )
    parser.add_argument(
        "--ray", action="store_true", help="Also run the Ray distributed check"
    )
    parser.add_argument(
        "--skip-distributed",
        action="store_true",
        help="Skip all distributed checks, including Dask (which runs by default)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5, help="Seconds between job-status polls"
    )
    parser.add_argument(
        "--poll-timeout", type=float, default=600, help="Max seconds to wait per job"
    )
    args = parser.parse_args()

    client = NixtlaClient()
    df = make_df()
    h = 7
    poll_kwargs = {"poll_interval": args.poll_interval, "poll_timeout": args.poll_timeout}

    checks = [
        ("forecast_async", check_forecast_async),
        ("forecast_async(add_history=True)", check_forecast_async_add_history),
        ("cross_validation_async", check_cross_validation_async),
        ("finetune_async", check_finetune_async),
        ("forecast_async(num_partitions=2)", check_forecast_async_num_partitions),
        (
            "cross_validation_async(num_partitions=2)",
            check_cross_validation_async_num_partitions,
        ),
    ]
    if not args.skip_distributed:
        checks.append(("forecast_async[dask]", check_forecast_async_distributed_dask))
        if args.spark:
            checks.append(("forecast_async[spark]", check_forecast_async_distributed_spark))
        if args.ray:
            checks.append(("forecast_async[ray]", check_forecast_async_distributed_ray))

    ok = run_checks(checks, client, df, h, poll_kwargs)
    sys.exit(0 if ok else 1)
