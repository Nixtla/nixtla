import os
import pytest

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