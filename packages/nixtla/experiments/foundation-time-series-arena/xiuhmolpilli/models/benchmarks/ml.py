import os

import pandas as pd
from mlforecast.auto import AutoMLForecast, AutoLightGBM

from ..utils.forecaster import Forecaster, get_seasonality

os.environ["NIXTLA_ID_AS_COL"] = "true"


class AutoLGBM(Forecaster):
    def __init__(
        self,
        alias: str = "AutoLGBM",
        num_samples: int = 10,
        cv_n_windows: int = 5,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.cv_n_windows = cv_n_windows

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        mf = AutoMLForecast(
            models=[AutoLightGBM()],
            freq=freq,
            season_length=get_seasonality(freq),
            num_threads=-1,
        )
        mf.fit(
            df=df,
            n_windows=self.cv_n_windows,
            h=h,
            num_samples=self.num_samples,
        )
        fcst_df = mf.predict(h=h)
        fcst_df = fcst_df.rename(columns={"AutoLightGBM": self.alias})
        return fcst_df
