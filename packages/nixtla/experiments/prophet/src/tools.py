import os
from typing import Optional, Tuple

import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse


class ExperimentHandler:
    def __init__(self, file: str, method: str):
        self.file = file
        self.method = method

    @staticmethod
    def get_parameter(parameter: str, df: pd.DataFrame):
        parameter = df[parameter].unique()
        if len(parameter) > 1:
            raise ValueError(f"{parameter} is not unique: {parameter}")
        return parameter[0]

    def read_data(
        self,
        max_insample_length: int = 3_000,
    ) -> Tuple[pd.DataFrame, str, str, int, int]:
        df = pd.read_parquet(self.file)
        Y_df = df[["unique_id", "ds", "y"]].drop_duplicates(["unique_id", "ds"])
        Y_df = Y_df.sort_values(["unique_id", "ds"])
        Y_df = Y_df.groupby("unique_id").tail(
            max_insample_length
        )  # take only last 3_000 rows
        Y_df["ds"] = Y_df["ds"].str.replace(":01$", ":00", regex=True)
        freq = self.get_parameter("frequency", df)
        pandas_freq = self.get_parameter("pandas_frequency", df)
        h = self.get_parameter("horizon", df)
        seasonality = self.get_parameter("seasonality", df)
        return Y_df, freq, pandas_freq, int(h), int(seasonality)

    def evaluate_model(
        self,
        Y_hat_df: pd.DataFrame,
        model_name: str,
        total_time: float,
    ):
        if "cutoff" in Y_hat_df.columns:
            Y_hat_df = Y_hat_df.drop(columns="cutoff")
        eval_df = evaluate(
            df=Y_hat_df,
            metrics=[rmse, mae],
        )
        total_time_df = pd.DataFrame({"model": [model_name], "time": [total_time]})
        return eval_df, total_time_df

    @staticmethod
    def summarize_df(df: pd.DataFrame):
        n_unique_ids = df["unique_id"].nunique()
        mean_y = df["y"].mean()
        std_y = df["y"].std()
        lengths = df.groupby("unique_id").size()
        min_length = lengths.min()
        max_length = lengths.max()
        n_obs = len(df)
        summary = {
            "n_series": n_unique_ids,
            "mean": mean_y,
            "std": std_y,
            "min_length": min_length,
            "max_length": max_length,
            "n_obs": n_obs,
        }
        summary_df = pd.DataFrame.from_dict(summary, orient="index")
        summary_df = summary_df.transpose()
        return summary_df

    def save_results(
        self,
        freq: str,
        eval_df: pd.DataFrame,
        total_time_df: pd.DataFrame,
        df: Optional[pd.DataFrame] = None,
    ):
        eval_df["frequency"] = freq
        eval_df = eval_df.melt(
            id_vars=["frequency", "metric", "unique_id"],
            var_name="model",
            value_name="value",
        )
        total_time_df["frequency"] = freq
        dir = self.file.split("/")[-1].replace(".parquet", "")
        dir = f"./data/results/{dir}"
        os.makedirs(dir, exist_ok=True)
        eval_df.to_parquet(
            f"{dir}/{self.method}_metrics.parquet",
            index=False,
        )
        total_time_df.to_parquet(
            f"{dir}/{self.method}_time.parquet",
            index=False,
        )
        if df is not None:
            summary_df = self.summarize_df(df)
            summary_df["frequency"] = freq
            print(summary_df)
            summary_df.to_parquet(
                f"{dir}/series_summary.parquet",
                index=False,
            )
