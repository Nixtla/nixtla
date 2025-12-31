import sys
import warnings

import dask.dataframe as dd
import pandas as pd
import pytest

from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.dask.lgb import DaskLGBMForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.utils import generate_daily_series

warnings.simplefilter("ignore", FutureWarning)

@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_dask_distributed_forecast():
    series = generate_daily_series(
        100, equal_ends=True, min_length=500, max_length=1_000
    )
    npartitions = 4
    partitioned_series = dd.from_pandas(
        series.set_index("unique_id"), npartitions=npartitions
    )  # make sure we split by the id_col
    partitioned_series = partitioned_series.map_partitions(lambda df: df.reset_index())
    partitioned_series["unique_id"] = partitioned_series["unique_id"].astype(
        str
    )  # can't handle categoricals atm

    # test existing features provide the same result
    fcst = DistributedMLForecast(
        models=[DaskLGBMForecast(verbosity=-1, random_state=0)],
        freq="D",
        lags=[1, 2, 3, 4, 5, 6, 7],
        lag_transforms={
            1: [RollingMean(7), RollingMean(30), ExpandingMean()],
            335: [RollingMean(30)],
        },
        date_features=["dayofweek"],
    )
    # df with features
    training_df_featured = fcst.preprocess(
        partitioned_series, static_features=[], dropna=False
    )
    fcst.fit(training_df_featured, static_features=[], dropna=False)
    preds1 = fcst.predict(10).compute()

    # df without features
    fcst.preprocess(partitioned_series, static_features=[], dropna=False)
    preds2 = fcst.predict(10).compute()
    pd.testing.assert_frame_equal(preds1, preds2)
