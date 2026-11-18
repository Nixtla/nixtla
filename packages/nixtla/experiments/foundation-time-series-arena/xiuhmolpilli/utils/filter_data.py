"""
this module takes Nixtla's benchmarking data 
and filters it to prevent azureml from crashing
in the following cases:
- too short series, see https://learn.microsoft.com/en-us/azure/machine-learning/concept-automl-forecasting-methods?view=azureml-api-2#data-length-requirements
"""
import logging
from pathlib import Path

import fire
import numpy as np
import pandas as pd

main_logger = logging.getLogger(__name__)
main_logger.setLevel(logging.INFO)


def get_min_size_per_series(dataset_path: str) -> int:
    if "Daily" in dataset_path or "Hourly" in dataset_path:
        return 1_000
    elif "Monthly" in dataset_path:
        return 10 * 12
    else:
        return 1_000 // 7


def filter_and_clean_dataset(
    dataset_path: str,
    max_series: int = 1_000,
    random_seed: int = 420,
):
    main_logger.info(f"Processing dataset {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df = df.drop_duplicates(["unique_id", "ds"])  # type: ignore
    df = df.sort_values(["unique_id", "ds"])
    min_size_per_series = get_min_size_per_series(dataset_path)
    df = (
        df.groupby("unique_id")
        .filter(lambda x: len(x) >= min_size_per_series)
        .reset_index(drop=True)
    )
    uids = df["unique_id"].unique()  # type: ignore
    if len(uids) > max_series:
        np.random.seed(random_seed)
        uids = np.random.choice(uids, max_series, replace=False)  # type: ignore
        df = df.query("unique_id in @uids")  # type: ignore
        main_logger.info(f"Filtering out {len(uids) - max_series} series")
    n_series = len(df["unique_id"].unique())  # type: ignore
    main_logger.info(f"Number of series: {n_series}")
    if n_series == 0:
        raise ValueError("No series left after filtering")
    # finally we clean some strange dates
    mask = df["ds"].str.endswith(":01")  # type: ignore
    df.loc[mask, "ds"] = df.loc[mask, "ds"].str[:-3] + ":00"
    # save the dataset
    dataset_path = Path(dataset_path)  # type: ignore
    filtered_dataset_path = dataset_path.parent / "filtered_datasets" / dataset_path.name  # type: ignore
    filtered_dataset_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(filtered_dataset_path)
    main_logger.info(f"Filtered dataset saved to {filtered_dataset_path}")


if __name__ == "__main__":
    fire.Fire(filter_and_clean_dataset)
