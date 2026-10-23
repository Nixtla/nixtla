import os
from multiprocessing import Pool
from typing import Callable, List

import pandas as pd


class ParallelForecaster:
    def _process_group(
        self,
        df: pd.DataFrame,
        func: Callable,
        **kwargs,
    ) -> pd.DataFrame:
        uid = df["unique_id"].iloc[0]
        _df = df.drop("unique_id", axis=1)
        res_df = func(_df, **kwargs)
        res_df.insert(0, "unique_id", uid)
        return res_df

    def _apply_parallel(
        self,
        df_grouped: pd.DataFrame,
        func: Callable,
        **kwargs,
    ) -> pd.DataFrame:
        with Pool(os.cpu_count() - 1) as executor:
            futures = [
                executor.apply_async(
                    self._process_group,
                    args=(df, func),
                    kwds=kwargs,
                )
                for _, df in df_grouped
            ]
            results = [future.get() for future in futures]
        return pd.concat(results)

    def _local_forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        quantiles: List[float] | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        quantiles: List[float] | None = None,
    ) -> pd.DataFrame:
        fcst_df = self._apply_parallel(
            df.groupby("unique_id"),
            self._local_forecast,
            h=h,
            freq=freq,
            quantiles=quantiles,
        )
        return fcst_df
