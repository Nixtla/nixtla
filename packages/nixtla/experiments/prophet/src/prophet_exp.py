from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import time
from typing import Optional

import fire
import numpy as np
import pandas as pd
from prophet import Prophet as _Prophet
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)

from src.tools import ExperimentHandler


class ParallelForecaster:
    def _process_group(self, func, df, **kwargs):
        uid = df["unique_id"].iloc[0]
        _df = df.drop("unique_id", axis=1)
        res_df = func(_df, **kwargs)
        res_df.insert(0, "unique_id", uid)
        return res_df

    def _apply_parallel(self, df_grouped, func, **kwargs):
        results = []
        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = [
                executor.submit(self._process_group, func, df, **kwargs)
                for _, df in df_grouped
            ]
            for future in futures:
                results.append(future.result())
        return pd.concat(results)

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
    ):
        df_grouped = df.groupby("unique_id")
        return self._apply_parallel(
            df_grouped,
            self._local_forecast,
            h=h,
        )

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int = 1,
        step_size: Optional[int] = None,
        **kwargs,
    ):
        df_grouped = df.groupby("unique_id")
        kwargs = {"h": h, "n_windows": n_windows, "step_size": step_size, **kwargs}
        return self._apply_parallel(
            df_grouped,
            self._local_cross_validation,
            **kwargs,
        )


class Prophet(_Prophet, ParallelForecaster):
    def __init__(
        self,
        freq: str,
        alias: str = "Prophet",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.alias = alias

    def _local_forecast(
        self,
        df: pd.DataFrame,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        model = deepcopy(self)
        model.fit(df=df)
        future_df = model.make_future_dataframe(
            periods=h, include_history=False, freq=self.freq
        )
        if X_df is not None:
            future_df = future_df.merge(X_df, how="left")
        np.random.seed(1000)
        fcst_df = model.predict(future_df)
        fcst_df = fcst_df.rename({"yhat": self.alias}, axis=1)
        fcst_df = fcst_df[["ds", self.alias]]
        return fcst_df

    def _local_cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int = 1,
        step_size: Optional[int] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df.insert(0, "unique_id", "ts_0")
        # mlforecast cv code
        results = []
        sort_idxs = maybe_compute_sort_indices(df, "unique_id", "ds")
        if sort_idxs is not None:
            df = take_rows(df, sort_idxs)
        splits = backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col="unique_id",
            time_col="ds",
            freq=pd.tseries.frequencies.to_offset(self.freq),
            step_size=h if step_size is None else step_size,
        )
        for i_window, (cutoffs, train, valid) in enumerate(splits):
            if len(valid.columns) > 3:
                # if we have uid, ds, y + exogenous vars
                train_future = valid.drop(columns="y")
            else:
                train_future = None
            y_pred = self._local_forecast(
                df=train[["ds", "y"]],
                h=h,
                X_df=train_future,
            )
            y_pred.insert(0, "unique_id", "ts_0")
            y_pred = join(y_pred, cutoffs, on="unique_id", how="left")
            result = join(
                valid[["unique_id", "ds", "y"]],
                y_pred,
                on=["unique_id", "ds"],
            )
            if result.shape[0] < valid.shape[0]:
                raise ValueError(
                    "Cross validation result produced less results than expected. "
                    "Please verify that the frequency parameter (freq) matches your series' "
                    "and that there aren't any missing periods."
                )
            results.append(result)
        out = vertical_concat(results)
        out = drop_index_if_pandas(out)
        first_out_cols = ["unique_id", "ds", "cutoff", "y"]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]
        fcst_cv_df = out[first_out_cols + remaining_cols]
        return fcst_cv_df.drop(columns="unique_id")


def evaluate_experiment(file: str):
    exp_handler = ExperimentHandler(file=file, method="prophet")
    Y_df, freq, pandas_freq, h, seasonality = exp_handler.read_data()
    model_name = "Prophet"
    print(model_name)
    prophet = Prophet(freq=pandas_freq)
    start = time()
    Y_hat_df = prophet.cross_validation(
        df=Y_df,
        h=h,
        n_windows=1,
    )
    total_time = time() - start
    print(total_time)
    # evaluation
    eval_df, total_time_df = exp_handler.evaluate_model(
        Y_hat_df=Y_hat_df,
        model_name=model_name,
        total_time=total_time,
    )
    exp_handler.save_results(
        freq=freq,
        eval_df=eval_df,
        total_time_df=total_time_df,
    )


if __name__ == "__main__":
    fire.Fire(evaluate_experiment)
