import logging
import warnings
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path

import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mae, mase

from src.utils.filter_data import DatasetParams

warnings.simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)


@dataclass
class ExperimentDataset:
    Y_df_train: pd.DataFrame
    Y_df_test: pd.DataFrame
    horizon: int
    seasonality: int
    frequency: str
    pandas_frequency: str

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "ExperimentDataset":
        """
        Parameters
        ----------
        df : pd.DataFrame
            df should have columns: unique_id, ds, y, frequency, pandas_frequency, horizon, seasonality
        """
        ds_params = DatasetParams.from_df(df)
        df = df[["unique_id", "ds", "y"]]  # type: ignore
        Y_df_test = df.groupby("unique_id").tail(ds_params.horizon)
        Y_df_train = df.drop(Y_df_test.index)  # type: ignore
        return cls(
            Y_df_train=Y_df_train,
            Y_df_test=Y_df_test,
            **asdict(ds_params),
        )

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
    ) -> "ExperimentDataset":
        df = pd.read_parquet(parquet_path)
        return cls.from_df(df=df)

    def evaluate_forecast_df(
        self,
        forecast_df: pd.DataFrame,
        model: str,
        total_time: float,
    ) -> pd.DataFrame:
        df_ = self.Y_df_test.copy(deep=True)
        if forecast_df.dtypes["ds"] != df_.dtypes["ds"]:
            df_["ds"] = df_["ds"].astype(forecast_df.dtypes["ds"])
        df = df_.merge(
            forecast_df[["unique_id", "ds", model]],
            on=["unique_id", "ds"],
            how="left",
        )
        if df[model].isna().sum() > 0:
            na_uids = df.loc[df[model].isna()]["unique_id"].unique()
            main_logger.warning(
                f"{model} contains NaN for {len(na_uids)} series: {na_uids}"
                "filling with last values"
            )
            from statsforecast import StatsForecast
            from statsforecast.models import SeasonalNaive

            sf = StatsForecast(
                models=[SeasonalNaive(season_length=self.seasonality)],
                freq=self.pandas_frequency,
            )
            sn_df = sf.forecast(
                df=self.Y_df_train,
                h=self.horizon,
            )
            df = df.merge(sn_df, on=["unique_id", "ds"], how="left")  # type: ignore
            df.loc[df["unique_id"].isin(na_uids), model] = df.loc[
                df["unique_id"].isin(na_uids), "SeasonalNaive"
            ]
            df = df.drop(columns=["SeasonalNaive"])
        partial_mase = partial(mase, seasonality=self.seasonality)
        eval_df = evaluate(
            df=df,
            metrics=[rmse, mae, partial_mase],
            train_df=self.Y_df_train,
            models=[model],
        )
        eval_df = eval_df.groupby("metric").mean(numeric_only=True).reset_index()  # type: ignore
        eval_time_df = pd.DataFrame(
            {
                "metric": ["total_time"],
                model: [total_time],
            }
        )
        eval_df = pd.concat(
            [eval_df, eval_time_df],
            ignore_index=True,
        )  # type: ignore
        return eval_df.set_index("metric")


@dataclass
class ForecastDataset:
    forecast_df: pd.DataFrame
    total_time: float

    @classmethod
    def from_dir(cls, dir: str | Path):
        dir_ = Path(dir)
        forecast_df = pd.read_parquet(dir_ / "forecast_df.parquet")
        with open(dir_ / "total_time.txt", "r") as file:
            total_time = float(file.read())
        return cls(forecast_df=forecast_df, total_time=total_time)

    @staticmethod
    def is_forecast_ready(dir: str | Path):
        dir_ = Path(dir)
        forecast_path = dir_ / "forecast_df.parquet"
        time_path = dir_ / "total_time.txt"
        return forecast_path.exists() and time_path.exists()

    def save_to_dir(self, dir: str | Path):
        dir_ = Path(dir)
        dir_.mkdir(parents=True, exist_ok=True)
        self.forecast_df.to_parquet(dir_ / "forecast_df.parquet")
        with open(dir_ / "total_time.txt", "w") as file:
            file.write(str(self.total_time))
