import sys
import pytest

# Skip entire directory if on Windows or required packages are not installed
if sys.platform == "win32":
    pytest.skip("Distributed tests are not supported on Windows", allow_module_level=True)

pytest.importorskip("pyspark", reason="PySpark is required for Spark distributed tests")
pytest.importorskip("xgboost.spark", reason="xgboost[spark] is required for Spark XGBoost tests")


@pytest.fixture(scope="session")
def spark_session():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("mlforecast-spark-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()
