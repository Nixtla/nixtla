import dask.dataframe as dd
import pytest
from dask.distributed import Client


@pytest.fixture(scope="module")
def dask_client():
    with Client() as client:
        yield client


@pytest.fixture(scope="module")
def dask_df(distributed_series):
    return dd.from_pandas(distributed_series, npartitions=2)


@pytest.fixture(scope="module")
def dask_diff_cols_df(distributed_series, renamer):
    return dd.from_pandas(
        distributed_series.rename(columns=renamer),
        npartitions=2,
    )


@pytest.fixture(scope="module")
def dask_df_x(distributed_df_x):
    return dd.from_pandas(distributed_df_x, npartitions=2)


@pytest.fixture(scope="module")
def dask_future_ex_vars_df(distributed_future_ex_vars_df):
    return dd.from_pandas(distributed_future_ex_vars_df, npartitions=2)


@pytest.fixture(scope="module")
def dask_df_x_diff_cols(distributed_df_x, renamer):
    return dd.from_pandas(distributed_df_x.rename(columns=renamer), npartitions=2)


@pytest.fixture(scope="module")
def dask_future_ex_vars_df_diff_cols(distributed_future_ex_vars_df, renamer):
    return dd.from_pandas(
        distributed_future_ex_vars_df.rename(columns=renamer), npartitions=2
    )
