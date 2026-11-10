import pandas as pd
import timesfm
import torch
from paxml import checkpoints

from ..utils.forecaster import Forecaster


class TimesFM(Forecaster):
    def __init__(
        self,
        repo_id: str = "google/timesfm-1.0-200m",
        context_length: int = 512,
        batch_size: int = 64,
        alias: str = "TimesFM",
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias

    def get_predictor(
        self,
        prediction_length: int,
    ) -> timesfm.TimesFm:
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        tfm = timesfm.TimesFm(
            context_len=self.context_length,
            horizon_len=prediction_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=backend,
            per_core_batch_size=self.batch_size,
        )
        tfm.load_from_checkpoint(repo_id=self.repo_id)
        return tfm

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        predictor = self.get_predictor(prediction_length=h)
        fcst_df = predictor.forecast_on_df(
            inputs=df,
            freq=freq,
            value_name="y",
            model_name=self.alias,
            num_jobs=1,
        )
        fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        return fcst_df
