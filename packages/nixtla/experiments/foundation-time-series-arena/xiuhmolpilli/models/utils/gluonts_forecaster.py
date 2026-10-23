from typing import Iterable, List, Any

import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.model.forecast import Forecast
from gluonts.torch.model.predictor import PyTorchPredictor
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .forecaster import Forecaster


def fix_freq(freq: str) -> str:
    # see https://github.com/awslabs/gluonts/pull/2462/files
    if len(freq) > 1 and freq.endswith("S"):
        return freq[:-1]
    return freq


def maybe_convert_col_to_float32(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if df[col_name].dtype != "float32":
        df = df.copy()
        df[col_name] = df[col_name].astype("float32")
    return df


class GluonTSForecaster(Forecaster):
    def __init__(self, repo_id: str, filename: str, alias: str):
        self.repo_id = repo_id
        self.filename = filename
        self.alias = alias

    @property
    def checkpoint_path(self) -> str:
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
        )

    @property
    def map_location(self) -> str:
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        return map_location

    def load(self) -> Any:
        return torch.load(
            self.checkpoint_path,
            map_location=self.map_location,
        )

    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        raise NotImplementedError

    def gluonts_instance_fcst_to_df(
        self,
        fcst: Forecast,
        freq: str,
        model_name: str,
    ) -> pd.DataFrame:
        point_forecast = fcst.mean
        h = len(point_forecast)
        dates = pd.date_range(
            fcst.start_date.to_timestamp(),
            freq=freq,
            periods=h,
        )
        fcst_df = pd.DataFrame(
            {
                "ds": dates,
                "unique_id": fcst.item_id,
                model_name: point_forecast,
            }
        )
        return fcst_df

    def gluonts_fcsts_to_df(
        self,
        fcsts: Iterable[Forecast],
        freq: str,
        model_name: str,
    ) -> pd.DataFrame:
        df = []
        for fcst in tqdm(fcsts):
            fcst_df = self.gluonts_instance_fcst_to_df(
                fcst=fcst,
                freq=freq,
                model_name=model_name,
            )
            df.append(fcst_df)
        return pd.concat(df).reset_index(drop=True)

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        df = maybe_convert_col_to_float32(df, "y")
        gluonts_dataset = PandasDataset.from_long_dataframe(
            df,
            target="y",
            item_id="unique_id",
            timestamp="ds",
            freq=fix_freq(freq),
        )
        predictor = self.get_predictor(prediction_length=h)
        fcsts = predictor.predict(gluonts_dataset, num_samples=100)
        fcst_df = self.gluonts_fcsts_to_df(
            fcsts,
            freq=freq,
            model_name=self.alias,
        )
        return fcst_df
