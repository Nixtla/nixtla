def test_forecast_with_missing_values(nixtla_test_client, air_passengers_with_nans):
    fcst = nixtla_test_client.forecast(air_passengers_with_nans, h=12)
    assert len(fcst) == 12
    assert fcst["TimeGPT"].notna().all()


def test_forecast_with_missing_values_multiple_series(
    nixtla_test_client, multi_series_with_nans
):
    data = multi_series_with_nans
    fcst = nixtla_test_client.forecast(data.df, h=data.h, freq=data.freq)
    assert len(fcst) == data.n_ids * data.h
    assert fcst["TimeGPT"].notna().all()


def test_cross_validation_with_missing_values(
    nixtla_test_client, air_passengers_with_nans
):
    cv = nixtla_test_client.cross_validation(air_passengers_with_nans, h=12, n_windows=2)
    assert len(cv) == 12 * 2
    assert cv["TimeGPT"].notna().all()


def test_detect_anomalies_with_missing_values(
    nixtla_test_client, air_passengers_with_nans
):
    anomalies = nixtla_test_client.detect_anomalies(air_passengers_with_nans)
    assert len(anomalies) > 0
    assert "anomaly" in anomalies.columns

def test_finetune_with_missing_values(nixtla_test_client, air_passengers_with_nans):                                        
    model_id = nixtla_test_client.finetune(air_passengers_with_nans, finetune_steps=1)                                      
    assert isinstance(model_id, str)    