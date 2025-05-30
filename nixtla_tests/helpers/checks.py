import pandas as pd
import pytest
import time

from nixtla.nixtla_client import NixtlaClient
from typing import Callable

# test num partitions
# we need to be sure that we can recover the same results
# using a for loop
# A: be aware that num partitons can produce different results
# when used finetune_steps
def check_num_partitions_same_results(method: Callable, num_partitions: int, **kwargs):
    res_partitioned = method(**kwargs, num_partitions=num_partitions)
    res_no_partitioned = method(**kwargs, num_partitions=1)
    sort_by = ['unique_id', 'ds']
    if 'cutoff' in res_partitioned:
        sort_by.extend(['cutoff'])
    pd.testing.assert_frame_equal(
        res_partitioned.sort_values(sort_by).reset_index(drop=True), 
        res_no_partitioned.sort_values(sort_by).reset_index(drop=True),
        rtol=1e-2,
        atol=1e-2,
    )

def check_retry_behavior(df, side_effect, side_effect_exception, max_retries=5, retry_interval=5, max_wait_time=40, should_retry=True, sleep_seconds=5):
    mock_nixtla_client = NixtlaClient(
        max_retries=max_retries,
        retry_interval=retry_interval,
        max_wait_time=max_wait_time,
    )
    mock_nixtla_client._make_request = side_effect
    init_time = time.time()
    with pytest.raises(side_effect_exception):
        mock_nixtla_client.forecast(df=df, h=12, time_col='timestamp', target_col='value')
    total_mock_time = time.time() - init_time
    if should_retry:
        approx_expected_time = min((max_retries - 1) * retry_interval, max_wait_time)
        upper_expected_time = min(max_retries * retry_interval, max_wait_time)
        assert total_mock_time >= approx_expected_time, "It is not retrying as expected"
        # preprocessing time before the first api call should be less than 60 seconds
        assert total_mock_time - upper_expected_time - (max_retries - 1) * sleep_seconds <= sleep_seconds
    else:
        assert total_mock_time <= max_wait_time


# test we recover the same <mean> forecasts
# with and without restricting input
# (add_history)
def check_equal_fcsts_add_history(nixtla_client, **kwargs):
    fcst_no_rest_df = nixtla_client.forecast(**kwargs, add_history=True)
    fcst_no_rest_df = fcst_no_rest_df.groupby('unique_id', observed=True).tail(kwargs['h']).reset_index(drop=True)
    fcst_rest_df = nixtla_client.forecast(**kwargs)
    pd.testing.assert_frame_equal(
        fcst_no_rest_df,
        fcst_rest_df,
        atol=1e-4,
        rtol=1e-3,
    )
    return fcst_rest_df
