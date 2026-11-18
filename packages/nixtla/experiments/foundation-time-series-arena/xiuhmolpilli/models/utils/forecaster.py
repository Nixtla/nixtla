from typing import List

import pandas as pd
from gluonts.time_feature.seasonality import get_seasonality as _get_seasonality
from tqdm import tqdm
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)


def get_seasonality(freq: str) -> int:
    return _get_seasonality(freq, seasonalities={"D": 7})


def maybe_convert_col_to_datetime(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
        df = df.copy()
        df[col_name] = pd.to_datetime(df[col_name])
    return df


class Forecaster:
    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        n_windows: int = 1,
        step_size: int | None = None,
    ) -> pd.DataFrame:
        df = maybe_convert_col_to_datetime(df, "ds")
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
            freq=pd.tseries.frequencies.to_offset(freq),
            step_size=h if step_size is None else step_size,
        )
        for _, (cutoffs, train, valid) in tqdm(enumerate(splits)):
            if len(valid.columns) > 3:
                raise NotImplementedError(
                    "Cross validation with exogenous variables is not yet supported."
                )
            y_pred = self.forecast(
                df=train,
                h=h,
                freq=freq,
            )
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
        return fcst_cv_df
