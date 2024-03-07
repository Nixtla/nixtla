"""
this module takes Nixtla's benchmarking data 
and filters it to prevent azureml from crashing
in the following cases:
- too short series, see https://learn.microsoft.com/en-us/azure/machine-learning/concept-automl-forecasting-methods?view=azureml-api-2#data-length-requirements
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import fire
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)


@dataclass
class DatasetParams:
    frequency: str
    pandas_frequency: str
    horizon: int
    seasonality: int

    @staticmethod
    def _get_value_from_df_col(
        df: pd.DataFrame,
        col: str,
        dtype: Callable | None = None,
    ) -> Any:
        col_values = df[col].unique()
        if len(col_values) > 1:
            raise ValueError(f"{col} is not unique: {col_values}")
        value = col_values[0]
        if dtype is not None:
            value = dtype(value)
        return value

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DatasetParams":
        dataset_params = {}
        dataset_params_cols = [
            "frequency",
            "pandas_frequency",
            "horizon",
            "seasonality",
        ]
        dataset_params_cols_dtypes = [str, str, int, int]
        for col, dtype in zip(dataset_params_cols, dataset_params_cols_dtypes):
            dataset_params[col] = cls._get_value_from_df_col(df, col, dtype=dtype)
        return cls(**dataset_params)


def filter_and_clean_dataset(
    dataset_path: str,
    max_series: int = 1_000,
    n_train_cv: int = 2,
    n_seasonalities: int = 5,
    max_insample_length: int = 3_000,
    random_seed: int = 420,
):
    main_logger.info(f"Processing dataset {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df = df.drop_duplicates(["unique_id", "ds"])  # type: ignore
    df = df.sort_values(["unique_id", "ds"])
    ds_params = DatasetParams.from_df(df)
    min_train_size_per_series = (
        ds_params.horizon
        + 2 * ds_params.horizon
        + (n_train_cv - 1) * ds_params.horizon
        + 1
    )
    if ds_params.seasonality < 100:
        # if series has low seasonality
        # we add n_seasonalities to min_train_size_per_series
        # to keep the series long enough
        min_train_size_per_series += n_seasonalities * ds_params.seasonality
    uids = df["unique_id"].unique()  # type: ignore
    df = (
        df.groupby("unique_id")
        .filter(lambda x: len(x) >= min_train_size_per_series)
        .groupby("unique_id")  # type: ignore
        .tail(max_insample_length + ds_params.horizon)
        .reset_index(drop=True)
    )
    main_logger.info(
        f"Filtering out {len(uids) - len(df['unique_id'].unique())} series"
    )
    uids = df["unique_id"].unique()  # type: ignore
    if len(uids) > max_series:
        np.random.seed(random_seed)
        uids = np.random.choice(uids, max_series, replace=False)  # type: ignore
        df = df.query("unique_id in @uids")  # type: ignore
        main_logger.info(f"Filtering out {len(uids) - max_series} series")
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
