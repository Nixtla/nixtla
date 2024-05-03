from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def read_parquet_and_assign(uid, url):
    df = pd.read_parquet(url)
    df["unique_id"] = uid
    df["ds"] = df["ds"].astype(str)
    return df[["unique_id", "ds", "y"]]


def download_data():
    catalogue_splits = pd.read_parquet("./data/catalogue_splits.parquet")
    catalogue_datasets = pd.read_parquet("./data/catalogue_datasets.parquet")
    catalogue_df = catalogue_splits.merge(
        catalogue_datasets,
        on=["dataset", "subdataset", "frequency"],
    )
    del catalogue_splits
    del catalogue_datasets
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
        print(f"frequency: {frequency}, pandas_frequency: {pandas_frequency}")
        print(f"number of uids: {len(uids)}")
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(read_parquet_and_assign, uid, url)
                for uid, url in zip(uids, urls)
            ]
            results = [future.result() for future in futures]
        print("dataset read")
        Y_df = pd.concat(results)
        Y_df = Y_df.merge(
            df.drop(columns="url"),
            on="unique_id",
            how="left",
        )
        print(Y_df)
        Y_df.to_parquet(f"./data/{frequency}_{pandas_frequency}.parquet")
        del Y_df


if __name__ == "__main__":
    download_data()
