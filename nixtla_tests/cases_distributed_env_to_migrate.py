
import os
import pandas as pd
import uuid

#| distributed
import dask.dataframe as dd
import ray
from dask.distributed import Client
from nixtla.nixtla_client import NixtlaClient
from pyspark.sql import SparkSession
from ray.cluster_utils import Cluster

from utilsforecast.data import generate_series
from cases_non_distributed_env_to_migrate import test_quantiles
from cases_non_distributed_env_to_migrate import test_forecast_dataframe 
from cases_non_distributed_env_to_migrate import test_forecast_dataframe_diff_cols
from cases_non_distributed_env_to_migrate import test_anomalies_dataframe
from cases_non_distributed_env_to_migrate import test_anomalies_dataframe_diff_cols
from cases_non_distributed_env_to_migrate import test_anomalies_online_dataframe
from cases_non_distributed_env_to_migrate import test_forecast_x_dataframe    
from cases_non_distributed_env_to_migrate import test_forecast_x_dataframe_diff_cols
from cases_non_distributed_env_to_migrate import test_finetuned_model

#| hide
# custom client
custom_client = NixtlaClient(
    base_url=os.environ['NIXTLA_BASE_URL_CUSTOM'],
    api_key=os.environ['NIXTLA_API_KEY_CUSTOM'],
)

# copy directly to obtain model_id2
#| hide
# finetuning
h = 5
series = generate_series(10, equal_ends=True)
train_end = series['ds'].max() - h * pd.offsets.Day()
train_mask = series['ds'] <= train_end
train = series[train_mask]
valid = series[~train_mask]
model_id1 = str(uuid.uuid4())
finetune_resp = custom_client.finetune(train, output_model_id=model_id1)
assert finetune_resp == model_id1
model_id2 = custom_client.finetune(train, finetuned_model_id=model_id1)


#| hide
#| distributed
n_series = 4
horizon = 7

series = generate_series(n_series, min_length=100)
series['unique_id'] = series['unique_id'].astype(str)

series_diff_cols = series.copy()
renamer = {'unique_id': 'id_col', 'ds': 'time_col', 'y': 'target_col'}
series_diff_cols = series_diff_cols.rename(columns=renamer)

# data for exogenous tests
df_x = pd.read_csv(
    'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-with-ex-vars.csv',
    parse_dates=['ds'],
)
df_x = df_x.rename(columns=str.lower)
future_ex_vars_df = pd.read_csv(
    'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-future-ex-vars.csv',
    parse_dates=['ds'],
)
future_ex_vars_df = future_ex_vars_df.rename(columns=str.lower)


