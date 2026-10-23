import os

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    _TS as StatsForecastModel,
    ADIDA as _ADIDA,
    AutoARIMA as _AutoARIMA,
    AutoCES as _AutoCES,
    AutoETS as _AutoETS,
    CrostonClassic as _CrostonClassic,
    DynamicOptimizedTheta as _DOTheta,
    HistoricAverage as _HistoricAverage,
    IMAPA as _IMAPA,
    SeasonalNaive as _SeasonalNaive,
    Theta as _Theta,
    ZeroModel as _ZeroModel,
)

from ..utils.forecaster import Forecaster, get_seasonality

os.environ["NIXTLA_ID_AS_COL"] = "true"


def run_statsforecast_model(
    model: StatsForecastModel,
    df: pd.DataFrame,
    h: int,
    freq: str,
) -> pd.DataFrame:
    sf = StatsForecast(
        models=[model],
        freq=freq,
        n_jobs=-1,
        fallback_model=_SeasonalNaive(
            season_length=get_seasonality(freq),
        ),
    )
    fcst_df = sf.forecast(df=df, h=h)
    return fcst_df


class ADIDA(Forecaster):
    def __init__(
        self,
        alias: str = "ADIDA",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_ADIDA(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class AutoARIMA(Forecaster):
    def __init__(
        self,
        alias: str = "AutoARIMA",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_AutoARIMA(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class AutoCES(Forecaster):
    def __init__(
        self,
        alias: str = "AutoCES",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_AutoCES(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class AutoETS(Forecaster):
    def __init__(
        self,
        alias: str = "AutoETS",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_AutoETS(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class CrostonClassic(Forecaster):
    def __init__(
        self,
        alias: str = "CrostonClassic",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_CrostonClassic(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class DOTheta(Forecaster):
    def __init__(
        self,
        alias: str = "DOTheta",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_DOTheta(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class HistoricAverage(Forecaster):
    def __init__(
        self,
        alias: str = "HistoricAverage",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_HistoricAverage(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class IMAPA(Forecaster):
    def __init__(
        self,
        alias: str = "IMAPA",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_IMAPA(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class SeasonalNaive(Forecaster):
    def __init__(
        self,
        alias: str = "SeasonalNaive",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_SeasonalNaive(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class Theta(Forecaster):
    def __init__(
        self,
        alias: str = "Theta",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        seasonality = get_seasonality(freq)
        fcst_df = run_statsforecast_model(
            model=_Theta(season_length=seasonality, alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df


class ZeroModel(Forecaster):
    def __init__(
        self,
        alias: str = "ZeroModel",
    ):
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        fcst_df = run_statsforecast_model(
            model=_ZeroModel(alias=self.alias),
            df=df,
            h=h,
            freq=freq,
        )
        return fcst_df
