
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
from test_cases_revised import test_quantiles
from test_cases_revised import test_forecast_dataframe 
from test_cases_revised import test_forecast_dataframe_diff_cols
from test_cases_revised import test_anomalies_dataframe
from test_cases_revised import test_anomalies_dataframe_diff_cols
from test_cases_revised import test_anomalies_online_dataframe
from test_cases_revised import test_forecast_x_dataframe    
from test_cases_revised import test_forecast_x_dataframe_diff_cols
from test_cases_revised import test_finetuned_model

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

#### Spark
#| hide
#| distributed
spark = SparkSession.builder.getOrCreate()
spark_df = spark.createDataFrame(series).repartition(2)
spark_diff_cols_df = spark.createDataFrame(series_diff_cols).repartition(2)

test_quantiles(spark_df, id_col="unique_id", time_col="ds")

test_forecast_dataframe(spark_df)
test_forecast_dataframe_diff_cols(spark_diff_cols_df)
test_anomalies_dataframe(spark_df)
test_anomalies_online_dataframe(spark_df)
test_anomalies_dataframe_diff_cols(spark_diff_cols_df)
# test exogenous variables
spark_df_x = spark.createDataFrame(df_x).repartition(2)
spark_future_ex_vars_df = spark.createDataFrame(future_ex_vars_df).repartition(2)
test_forecast_x_dataframe(spark_df_x, spark_future_ex_vars_df)
# test x different cols
spark_df_x_diff_cols = spark.createDataFrame(df_x.rename(columns=renamer)).repartition(2)
spark_future_ex_vars_df_diff_cols = spark.createDataFrame(
    future_ex_vars_df.rename(columns=renamer)
).repartition(2)
test_forecast_x_dataframe_diff_cols(spark_df_x_diff_cols, spark_future_ex_vars_df_diff_cols)

# test finetuning
test_finetuned_model(spark_df)

spark.stop()

#### Dask

#| hide
#| distributed
client = Client()
dask_df = dd.from_pandas(series, npartitions=2)
dask_diff_cols_df = dd.from_pandas(series_diff_cols, npartitions=2)


test_quantiles(dask_df, id_col="unique_id", time_col="ds")


test_forecast_dataframe(dask_df)
test_forecast_dataframe_diff_cols(dask_diff_cols_df)
test_anomalies_dataframe(dask_df)
test_anomalies_online_dataframe(dask_df)
test_anomalies_dataframe_diff_cols(dask_diff_cols_df)

# test exogenous variables
dask_df_x = dd.from_pandas(df_x, npartitions=2)
dask_future_ex_vars_df = dd.from_pandas(future_ex_vars_df, npartitions=2)
test_forecast_x_dataframe(dask_df_x, dask_future_ex_vars_df)

# test x different cols
dask_df_x_diff_cols = dd.from_pandas(df_x.rename(columns=renamer), npartitions=2)
dask_future_ex_vars_df_diff_cols = dd.from_pandas(future_ex_vars_df.rename(columns=renamer), npartitions=2)
test_forecast_x_dataframe_diff_cols(dask_df_x_diff_cols, dask_future_ex_vars_df_diff_cols)

# test finetuning
test_finetuned_model(dask_df)

client.close()

#### Ray

#| hide
#| distributed
ray_cluster = Cluster(
    initialize_head=True,
    head_node_args={"num_cpus": 2}
)
ray.init(address=ray_cluster.address, ignore_reinit_error=True)
# add mock node to simulate a cluster
mock_node = ray_cluster.add_node(num_cpus=2)

ray_df = ray.data.from_pandas(series)
ray_diff_cols_df = ray.data.from_pandas(series_diff_cols)

test_quantiles(ray_df, id_col="unique_id", time_col="ds")

test_forecast_dataframe(ray_df)
test_forecast_dataframe_diff_cols(ray_diff_cols_df)
test_anomalies_dataframe(ray_df)
test_anomalies_online_dataframe(ray_df)
test_anomalies_dataframe_diff_cols(ray_diff_cols_df)

# test exogenous variables
ray_df_x = ray.data.from_pandas(df_x)
ray_future_ex_vars_df = ray.data.from_pandas(future_ex_vars_df)
test_forecast_x_dataframe(ray_df_x, ray_future_ex_vars_df)

# test x different cols
ray_df_x_diff_cols = ray.data.from_pandas(df_x.rename(columns=renamer))
ray_future_ex_vars_df_diff_cols = ray.data.from_pandas(future_ex_vars_df.rename(columns=renamer))
test_forecast_x_dataframe_diff_cols(ray_df_x_diff_cols, ray_future_ex_vars_df_diff_cols)

# test finetuning
test_finetuned_model(ray_df)

ray.shutdown()
