import os

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import (
    AutoNHITS as _AutoNHITS,
    AutoTFT as _AutoTFT,
)
from neuralforecast.common._base_model import BaseModel as NeuralForecastModel
from ray import tune

from ..utils.forecaster import Forecaster

os.environ["NIXTLA_ID_AS_COL"] = "true"


def run_neuralforecast_model(
    model: NeuralForecastModel,
    df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    nf = NeuralForecast(
        models=[model],
        freq=freq,
    )
    nf.fit(df=df)
    fcst_df = nf.predict()
    return fcst_df


class AutoNHITS(Forecaster):
    def __init__(
        self,
        alias: str = "AutoNHITS",
        num_samples: int = 10,
        backend: str = "optuna",
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.backend = backend

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        config = _AutoNHITS.get_default_config(h=h, backend="ray")
        config["scaler_type"] = tune.choice(["robust"])

        if self.backend == "optuna":
            config = _AutoNHITS._ray_config_to_optuna(config)
        fcst_df = run_neuralforecast_model(
            model=_AutoNHITS(
                h=h,
                alias=self.alias,
                num_samples=self.num_samples,
                backend=self.backend,
                config=config,
            ),
            df=df,
            freq=freq,
        )
        return fcst_df


class AutoTFT(Forecaster):
    def __init__(
        self,
        alias: str = "AutoTFT",
        num_samples: int = 10,
        backend: str = "optuna",
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.backend = backend

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        config = _AutoTFT.get_default_config(h=h, backend="ray")
        config["scaler_type"] = tune.choice(["robust"])
        if self.backend == "optuna":
            config = _AutoTFT._ray_config_to_optuna(config)
        fcst_df = run_neuralforecast_model(
            model=_AutoTFT(
                h=h,
                alias=self.alias,
                num_samples=self.num_samples,
                backend=self.backend,
                config=config,
            ),
            df=df,
            freq=freq,
        )
        return fcst_df
