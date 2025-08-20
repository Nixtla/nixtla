import pytest

try:
    from pyspark.sql import SparkSession

    @pytest.fixture(scope="module")
    def spark_client():
        with SparkSession.builder.getOrCreate() as spark:
            yield spark

    @pytest.fixture(scope="module")
    def spark_df(spark_client, distributed_series):
        spark_df = spark_client.createDataFrame(distributed_series).repartition(2)
        return spark_df

    @pytest.fixture(scope="module")
    def spark_diff_cols_df(spark_client, distributed_series, renamer):
        spark_df = spark_client.createDataFrame(
            distributed_series.rename(columns=renamer)
        ).repartition(2)
        return spark_df

    @pytest.fixture(scope="module")
    def spark_df_x(spark_client, distributed_df_x):
        spark_df = spark_client.createDataFrame(distributed_df_x).repartition(2)
        return spark_df

    @pytest.fixture(scope="module")
    def spark_df_x_diff_cols(spark_client, distributed_df_x, renamer):
        spark_df = spark_client.createDataFrame(
            distributed_df_x.rename(columns=renamer)
        ).repartition(2)
        return spark_df

    @pytest.fixture(scope="module")
    def spark_future_ex_vars_df(spark_client, distributed_future_ex_vars_df):
        spark_df = spark_client.createDataFrame(
            distributed_future_ex_vars_df
        ).repartition(2)
        return spark_df

    @pytest.fixture(scope="module")
    def spark_future_ex_vars_df_diff_cols(
        spark_client, distributed_future_ex_vars_df, renamer
    ):
        spark_df = spark_client.createDataFrame(
            distributed_future_ex_vars_df.rename(columns=renamer)
        ).repartition(2)
        return spark_df
except ImportError:
    # If PySpark is not installed, we skip the fixtures
    pytest.skip(
        "PySpark is not installed, skipping Spark fixtures", allow_module_level=True
    )
