def s3_file_exists(s3, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise e
    return False


def read_s3_file(s3, bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8")


def forecast_partition(
    i: int,
    version: str,
    bucket: str,
    prefix: str,
    n_series: int,
    freq: str = "D",
    h: int = 7,
) -> None:
    import logging
    import math
    import time

    import boto3
    import pandas as pd
    from nixtla import NixtlaClient as V2Client
    from nixtlats import NixtlaClient as V1Client
    from tqdm.auto import tqdm
    from utilsforecast.data import generate_series

    s3 = boto3.client("s3")
    # only process if we haven't saved the time
    time_key = f"{prefix}/times/{version}/{i}.txt"
    if s3_file_exists(s3, bucket, time_key):
        print(f"{i}-th partition already processed, skipping.")
        return
    logging.getLogger("nixtla").setLevel(logging.ERROR)
    logging.getLogger("nixtlats").setLevel(logging.ERROR)

    series = generate_series(
        n_series=n_series,
        freq=freq,
        min_length=100,
        max_length=200,
        seed=i,
    )
    series["unique_id"] = series["unique_id"].astype("uint32") + i * n_series

    start = time.perf_counter()
    if version == "v1":
        client = V1Client()
        # v1 is slower when partitioning, so we do this sequentially
        num_partitions = math.ceil(n_series / 50_000)
        uids = series["unique_id"].unique()
        n_ids = uids.size
        ids_per_part = math.ceil(n_ids / num_partitions)
        results = []
        for j in tqdm(range(0, n_ids, ids_per_part)):
            part_uids = uids[j : j + ids_per_part]
            part = series[series["unique_id"].isin(part_uids)]
            results.append(client.forecast(df=part, h=h, freq=freq))
        forecast = pd.concat(results)
    else:
        client = V2Client()
        num_partitions = math.ceil(n_series / 100_000)
        forecast = client.forecast(
            df=series, h=h, freq=freq, num_partitions=num_partitions
        )
    time_taken = "{:.2f}".format(time.perf_counter() - start)
    forecast.to_parquet(
        f"s3://{bucket}/{prefix}/output/{version}/{i}.parquet", index=False
    )
    s3.put_object(Bucket=bucket, Key=time_key, Body=time_taken)
    print(f"{i}: {time_taken}")


def generate_forecasts(
    version: str,
    bucket: str,
    prefix: str,
    n_partitions: int,
    series_per_partition: int,
    n_jobs: int,
) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    fn = partial(
        forecast_partition,
        version=version,
        bucket=bucket,
        prefix=prefix,
        n_series=series_per_partition,
    )
    with ProcessPoolExecutor(n_jobs) as executor:
        _ = executor.map(fn, range(n_partitions))


def read_times(
    s3, version: str, bucket: str, prefix: str, n_partitions: int
) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm.auto import tqdm

    key = f"{prefix}/{version}_times.txt"
    if s3_file_exists(s3, bucket, key):
        return read_s3_file(s3, bucket, key).splitlines()
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                read_s3_file, s3, bucket, f"{prefix}/times/{version}/{i}.txt"
            )
            for i in range(n_partitions)
        ]
        times = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            times.append(future.result())
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body="\n".join(times),
    )
    return times


def main(
    bucket: str = "datasets-nixtla",
    prefix: str = "one-billion",
    n_partitions: int = 1_000,
    series_per_partition: int = 1_000_000,
    n_jobs: int = 5,
):
    import boto3
    import pandas as pd

    times = {}
    s3 = boto3.client("s3")
    for version in ("v1", "v2"):
        generate_forecasts(
            version=version,
            bucket=bucket,
            prefix=prefix,
            n_partitions=n_partitions,
            series_per_partition=series_per_partition,
            n_jobs=n_jobs,
        )
        times[version] = read_times(
            s3, version=version, bucket=bucket, prefix=prefix, n_partitions=n_partitions
        )
    pd.DataFrame(times).to_csv(f"s3://{bucket}/{prefix}/times.csv", index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
