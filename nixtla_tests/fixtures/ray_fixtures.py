import pytest

try:
    import ray
    from ray.cluster_utils import Cluster

    @pytest.fixture(scope="module")
    def ray_cluster_setup():
        ray_cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 2})
        with ray.init(address=ray_cluster.address, ignore_reinit_error=True):
            # add mock node to simulate a cluster
            ray_cluster.add_node(num_cpus=2)
            yield

    @pytest.fixture(scope="module")
    def ray_df(distributed_series):
        return ray.data.from_pandas(distributed_series)

    @pytest.fixture(scope="module")
    def ray_diff_cols_df(distributed_series, renamer):
        return ray.data.from_pandas(distributed_series.rename(columns=renamer))

    @pytest.fixture(scope="module")
    def ray_df_x(distributed_df_x):
        return ray.data.from_pandas(distributed_df_x)

    @pytest.fixture(scope="module")
    def ray_future_ex_vars_df(distributed_future_ex_vars_df):
        return ray.data.from_pandas(distributed_future_ex_vars_df)

    @pytest.fixture(scope="module")
    def ray_df_x_diff_cols(distributed_df_x, renamer):
        return ray.data.from_pandas(distributed_df_x.rename(columns=renamer))

    @pytest.fixture(scope="module")
    def ray_future_ex_vars_df_diff_cols(distributed_future_ex_vars_df, renamer):
        return ray.data.from_pandas(
            distributed_future_ex_vars_df.rename(columns=renamer)
        )
except ImportError:
    # If Ray is not installed, we skip the fixtures
    pytest.skip("Ray is not installed, skipping Ray fixtures", allow_module_level=True)
