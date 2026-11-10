from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from tqdm import tqdm
from utilsforecast.processing import make_future_dataframe

from ..utils.forecaster import Forecaster


class TimeSeriesDataset:
    def __init__(
        self,
        data: torch.Tensor,
        uids: Iterable,
        last_times: Iterable,
        batch_size: int,
    ):
        self.data = data
        self.uids = uids
        self.last_times = last_times
        self.batch_size = batch_size
        self.n_batches = len(data) // self.batch_size + (
            0 if len(data) % self.batch_size == 0 else 1
        )
        self.current_batch = 0

    @classmethod
    def from_df(cls, df: pd.DataFrame, batch_size: int):
        num_unique_ids = df["unique_id"].nunique()
        max_series_length = df["unique_id"].value_counts().max()
        padded_tensor = torch.full(
            size=(num_unique_ids, max_series_length),
            fill_value=torch.nan,
            dtype=torch.bfloat16,
        )  # type: ignore
        df_sorted = df.sort_values(by=["unique_id", "ds"])
        for idx, (_, group) in enumerate(df_sorted.groupby("unique_id")):
            series_length = len(group)
            padded_tensor[idx, -series_length:] = torch.tensor(
                group["y"].values,
                dtype=torch.bfloat16,
            )
        uids = df_sorted["unique_id"].unique()
        last_times = df_sorted.groupby("unique_id")["ds"].tail(1)
        return cls(padded_tensor, uids, last_times, batch_size)

    def __len__(self):
        return self.n_batches

    def make_future_dataframe(self, h: int, freq: str) -> pd.DataFrame:
        return make_future_dataframe(
            uids=self.uids,
            last_times=pd.to_datetime(self.last_times),
            h=h,
            freq=freq,
        )  # type: ignore

    def __iter__(self):
        self.current_batch = 0  # Reset for new iteration
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            self.current_batch += 1
            return self.data[start_idx:end_idx]
        else:
            raise StopIteration


class Chronos(Forecaster):
    def __init__(
        self,
        repo_id: str = "amazon/chronos-t5-large",
        batch_size: int = 16,
        alias: str = "Chronos",
    ):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias
        self.model = ChronosPipeline.from_pretrained(
            repo_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        dataset = TimeSeriesDataset.from_df(df, batch_size=self.batch_size)
        fcsts = [
            self.model.predict(batch, prediction_length=h) for batch in tqdm(dataset)
        ]
        fcst = torch.cat(fcsts)
        fcst = fcst.numpy()
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        fcst_df[self.alias] = np.mean(fcst, axis=1).reshape(-1, 1)
        return fcst_df
