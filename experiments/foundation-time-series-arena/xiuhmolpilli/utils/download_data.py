import logging
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)


def read_parquet_and_assign(uid, url):
    df = pd.read_parquet(url)
    df["unique_id"] = uid
    df["ds"] = df["ds"].astype(str)
    return df[["unique_id", "ds", "y"]]


def download_data():
    catalogue_splits = pd.read_csv("./data/series_catalogue_hourly.csv")
    catalogue_df = catalogue_splits.query("dataset == 'moirai'")
    catalogue_df["pandas_frequency"] = "H"
    catalogue_df["seasonality"] = 24
    catalogue_df["horizon"] = 24
    catalogue_df = catalogue_df.query("split == 'test'")[
        [
            "unique_id",
            "frequency",
            "url",
            "pandas_frequency",
            "seasonality",
            "horizon",
        ]
    ]
    grouped_df = catalogue_df.groupby(["frequency", "pandas_frequency"])
    for (frequency, pandas_frequency), df in grouped_df:
        uids, urls = df["unique_id"].values, df["url"].values
        main_logger.info(
            f"frequency: {frequency}, pandas_frequency: {pandas_frequency}"
        )
        n_uids = len(uids)
        main_logger.info(f"number of uids: {n_uids}")
        max_workers = min(10, n_uids)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(read_parquet_and_assign, uid, url)
                for uid, url in zip(uids, urls)
            ]
            results = [future.result() for future in futures]
        main_logger.info("dataset read")
        Y_df = pd.concat(results)
        Y_df = Y_df.merge(
            df.drop(columns="url"),
            on="unique_id",
            how="left",
        )
        # Y_df.to_parquet(f"./data/{frequency}_{pandas_frequency}.parquet")
        Y_df.to_parquet(f"./data/filtered_datasets/moirai-data.parquet")
        del Y_df
        main_logger.info("dataset saved")


if __name__ == "__main__":
    download_data()
