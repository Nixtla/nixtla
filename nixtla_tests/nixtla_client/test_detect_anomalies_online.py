from copy import deepcopy

import numpy as np
import pytest


def test_detect_anomalies_online_univariate(nixtla_test_client, anomaly_online_df):
    df, n_series, detection_size = anomaly_online_df
    anomaly_df = nixtla_test_client.detect_anomalies_online(
        df,
        h=20,
        detection_size=detection_size,
        threshold_method="univariate",
        freq="W-SUN",
        level=99,
    )
    assert len(anomaly_df) == n_series * detection_size
    assert (
        len(anomaly_df.columns) == 8
    )  # [unique_id, ds, TimeGPT, y, anomaly, anomaly_score, hi, lo]
    assert anomaly_df["anomaly"].sum() == 2
    assert anomaly_df["anomaly"].iloc[0] and anomaly_df["anomaly"].iloc[-1]


def test_detect_anomalies_online_multivariate(nixtla_test_client, anomaly_online_df):
    df, n_series, detection_size = anomaly_online_df
    multi_anomaly_df = nixtla_test_client.detect_anomalies_online(
        df,
        h=20,
        detection_size=detection_size,
        threshold_method="multivariate",
        freq="W-SUN",
        level=99,
    )
    assert len(multi_anomaly_df) == n_series * detection_size
    assert (
        len(multi_anomaly_df.columns) == 7
    )  # [unique_id, ds, TimeGPT, y, anomaly, anomaly_score, accumulated_anomaly_score]
    assert multi_anomaly_df["anomaly"].sum() == 4
    assert (
        multi_anomaly_df["anomaly"].iloc[0]
        and multi_anomaly_df["anomaly"].iloc[4]
        and multi_anomaly_df["anomaly"].iloc[5]
        and multi_anomaly_df["anomaly"].iloc[9]
    )


@pytest.mark.parametrize("threshold_method", ["univariate", "multivariate"])
def test_detect_anomalies_online_with_missing_values(
    nixtla_test_client, anomaly_online_df, threshold_method
):
    df, n_series, detection_size = anomaly_online_df
    df = deepcopy(df)
    size = len(df) // n_series
    # inject NaN at index 2 of each series' detection window
    df.loc[df.index[size - 3], "y"] = np.nan
    df.loc[df.index[2 * size - 3], "y"] = np.nan

    anomaly_df = nixtla_test_client.detect_anomalies_online(
        df,
        h=20,
        detection_size=detection_size,
        threshold_method=threshold_method,
        freq="W-SUN",
        level=99,
    )
    assert len(anomaly_df) == n_series * detection_size
    assert not anomaly_df["anomaly"].iloc[2]
    assert not anomaly_df["anomaly"].iloc[detection_size + 2]


def test_detect_anomalies_online_with_consecutive_nans(
    nixtla_test_client, anomaly_online_df
):
    df, n_series, detection_size = anomaly_online_df
    df = deepcopy(df)
    size = len(df) // n_series
    consecutive_nan_count = 3
    # consecutive NaNs at indices 1, 2, 3 of first series' detection window
    for i in range(consecutive_nan_count):
        df.loc[df.index[size - detection_size + 1 + i], "y"] = np.nan

    anomaly_df = nixtla_test_client.detect_anomalies_online(
        df,
        h=20,
        detection_size=detection_size,
        threshold_method="univariate",
        freq="W-SUN",
        level=99,
    )
    assert len(anomaly_df) == n_series * detection_size
    for i in range(1, 1 + consecutive_nan_count):
        assert not anomaly_df["anomaly"].iloc[i]
