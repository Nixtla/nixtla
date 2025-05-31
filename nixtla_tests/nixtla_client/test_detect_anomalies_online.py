

def test_detect_anomalies_online_univariate(nixtla_test_client, anomaly_online_df):
    df, n_series, detection_size = anomaly_online_df
    anomaly_df = nixtla_test_client.detect_anomalies_online(
        df, 
        h=20, 
        detection_size=detection_size, 
        threshold_method="univariate", 
        freq='W-SUN', 
        level=99,
    )
    assert len(anomaly_df) == n_series * detection_size
    assert len(anomaly_df.columns) == 8  # [unique_id, ds, TimeGPT, y, anomaly, anomaly_score, hi, lo]
    assert anomaly_df['anomaly'].sum() == 2
    assert anomaly_df['anomaly'].iloc[0] and anomaly_df['anomaly'].iloc[-1]

def test_detect_anomalies_online_multivariate(nixtla_test_client, anomaly_online_df):
    df, n_series, detection_size = anomaly_online_df
    multi_anomaly_df = nixtla_test_client.detect_anomalies_online(
        df, 
        h=20, 
        detection_size=detection_size, 
        threshold_method="multivariate", 
        freq='W-SUN', 
        level=99,
    )
    assert len(multi_anomaly_df) == n_series * detection_size
    assert len(multi_anomaly_df.columns) == 7  # [unique_id, ds, TimeGPT, y, anomaly, anomaly_score, accumulated_anomaly_score]
    assert multi_anomaly_df['anomaly'].sum() == 4
    assert (
        multi_anomaly_df['anomaly'].iloc[0]
        and multi_anomaly_df['anomaly'].iloc[4]
        and multi_anomaly_df['anomaly'].iloc[5]
        and multi_anomaly_df['anomaly'].iloc[9]
    )