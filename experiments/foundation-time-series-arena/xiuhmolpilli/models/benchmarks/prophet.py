from copy import deepcopy
from typing import List
from threadpoolctl import threadpool_limits

import pandas as pd
from prophet import Prophet

from ..utils.parallel_forecaster import ParallelForecaster
from ..utils.forecaster import Forecaster


class NixtlaProphet(Prophet, ParallelForecaster, Forecaster):
    def __init__(
        self,
        alias: str = "Prophet",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alias = alias

    def __local_forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        quantiles: List[float] | None = None,
    ) -> pd.DataFrame:
        if quantiles is not None:
            raise NotImplementedError
        model = deepcopy(self)
        model.fit(df=df)
        future_df = model.make_future_dataframe(
            periods=h,
            include_history=False,
            freq=freq,
        )
        fcst_df = model.predict(future_df)
        fcst_df = fcst_df.rename({"yhat": self.alias}, axis=1)
        fcst_df = fcst_df[["ds", self.alias]]
        return fcst_df

    def _local_forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        quantiles: List[float] | None = None,
    ) -> pd.DataFrame:
        with threadpool_limits(limits=1):
            return self.__local_forecast(
                df=df,
                h=h,
                freq=freq,
                quantiles=quantiles,
            )
