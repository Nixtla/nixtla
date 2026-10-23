import pandas as pd
import pytest
from utilsforecast.data import generate_series

from .utils import models


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("freq", ["H", "D", "W-MON", "MS"])
@pytest.mark.parametrize("h", [1, 12])
def test_correct_forecast_dates(model, freq, h):
    n_series = 5
    df = generate_series(
        n_series,
        freq=freq,
    )
    df["unique_id"] = df["unique_id"].astype(str)
    df_test = df.groupby("unique_id").tail(h)
    df_train = df.drop(df_test.index)
    fcst_df = model.forecast(
        df_train,
        h=h,
        freq=freq,
    )
    exp_n_cols = 3
    assert fcst_df.shape == (n_series * h, exp_n_cols)
    exp_cols = ["unique_id", "ds"]
    pd.testing.assert_frame_equal(
        fcst_df[exp_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
        df_test[exp_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("freq", ["H", "D", "W-MON", "MS"])
@pytest.mark.parametrize("n_windows", [1, 4])
def test_cross_validation(model, freq, n_windows):
    h = 12
    n_series = 5
    df = generate_series(n_series, freq=freq, equal_ends=True)
    df["unique_id"] = df["unique_id"].astype(str)
    cv_df = model.cross_validation(
        df,
        h=h,
        freq=freq,
        n_windows=n_windows,
    )
    exp_n_cols = 5  # unique_id, cutoff, ds, y, model
    assert cv_df.shape == (n_series * h * n_windows, exp_n_cols)
    cutoffs = cv_df["cutoff"].unique()
    assert len(cutoffs) == n_windows
    df_test = df.groupby("unique_id").tail(h * n_windows)
    exp_cols = ["unique_id", "ds", "y"]
    pd.testing.assert_frame_equal(
        cv_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
        df_test.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
    )
    if n_windows == 1:
        # test same results using predict with less data
        df_test = df.groupby("unique_id").tail(h)
        df_train = df.drop(df_test.index)
        fcst_df = model.forecast(
            df_train,
            h=h,
            freq=freq,
        )
        exp_cols = ["unique_id", "ds"]
        pd.testing.assert_frame_equal(
            cv_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
            fcst_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
        )
