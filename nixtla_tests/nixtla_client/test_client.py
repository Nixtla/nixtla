import os
import pytest
import pandas as pd

from nixtla_tests.helpers.client_helper import delete_env_var
from nixtla.nixtla_client import NixtlaClient

def test_api_key_fail():
    with delete_env_var('NIXTLA_API_KEY'), delete_env_var('TIMEGPT_TOKEN'):
        with pytest.raises(KeyError) as excinfo:
            NixtlaClient()
        assert 'NIXTLA_API_KEY' in str(excinfo.value)

def test_api_key_success():
    nixtla_client = NixtlaClient()
    assert nixtla_client.validate_api_key()

def test_custom_client_success():
    custom_client = NixtlaClient(
        base_url=os.environ['NIXTLA_BASE_URL_CUSTOM'],
        api_key=os.environ['NIXTLA_API_KEY_CUSTOM'],
    )
    assert custom_client.validate_api_key()    

    # assert the usage endpoint
    usage = custom_client.usage()
    assert sorted(usage.keys()) == ['minute', 'month']

def test_forecast_with_wrong_api_key():
    with pytest.raises(Exception) as excinfo:
        NixtlaClient(api_key='transphobic').forecast(df=pd.DataFrame(), h=None, validate_api_key=True)

    assert 'nixtla' in str(excinfo.value)

def test_get_model_params(nixtla_test_client):
    assert nixtla_test_client._get_model_params(model='timegpt-1', freq='D') == (28, 7)

def test_client_plot(nixtla_test_client, air_passengers_df):
    nixtla_test_client.plot(air_passengers_df, time_col='timestamp', target_col='value', engine='plotly')


def test_finetune_cv(nixtla_test_client, air_passengers_df):
    finetune_cv = nixtla_test_client.cross_validation(
        air_passengers_df, h=12, time_col="timestamp", target_col="value", n_windows=1, finetune_steps=1
    )
    assert finetune_cv is not None

def test_forecast_warning(nixtla_test_client, air_passengers_df, caplog):
    nixtla_test_client.forecast(
        df=air_passengers_df.tail(3),
        h=100,
        time_col='timestamp',
        target_col='value',
    )
    assert "The specified horizon \"h\" exceeds the model horizon" in caplog.text

@pytest.mark.parametrize(
    "kwargs",
    [
        {"add_history": True},
        {"finetune_steps": 10, "finetune_loss":"mae"},
    ],
    ids=["short horizon with add_history", "short horizon with finetuning"]
)
def test_forecast_error(nixtla_test_client, air_passengers_df, kwargs):
    with pytest.raises(ValueError, match="Some series are too short. Please make sure that each series"):
        nixtla_test_client.forecast(
            df=air_passengers_df.tail(3),
            h=12,
            time_col='timestamp',
            target_col='value',
            **kwargs
        )