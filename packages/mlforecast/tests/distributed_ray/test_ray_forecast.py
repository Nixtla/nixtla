import sys
import warnings

import pandas as pd
import pytest
import ray

from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.ray.lgb import RayLGBMForecast
from mlforecast.distributed.models.ray.xgb import RayXGBForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.utils import generate_daily_series

warnings.simplefilter("ignore", FutureWarning)


@pytest.mark.ray
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Distributed tests are not supported on Python < 3.10")
@pytest.mark.parametrize(
    "model_class,model_kwargs",
    [
        (RayLGBMForecast, {"verbosity": -1, "random_state": 0}),
        (RayXGBForecast, {"random_state": 0}),
    ],
    ids=["lightgbm", "xgboost"]
)
def test_ray_distributed_forecast(model_class, model_kwargs):
    series = generate_daily_series(
        100, equal_ends=True, min_length=500, max_length=1_000
    )

    # Create Ray dataset from pandas
    ray_dataset = ray.data.from_pandas(series)
    ray_dataset = ray_dataset.map_batches(
        lambda batch: batch.assign(unique_id=batch['unique_id'].astype(str)),
        batch_format="pandas"
    )

    # test existing features provide the same result
    fcst = DistributedMLForecast(
        models=[model_class(**model_kwargs)],
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
        ray_dataset, static_features=[], dropna=False
    )
    fcst.fit(training_df_featured, static_features=[], dropna=False)
    preds1 = fcst.predict(10).to_pandas().sort_values(["unique_id", "ds"], ignore_index = True)
    fcst.save(f'/tmp/test_ray_forecast_model_{model_class.__name__}.pkl')

    # df without features
    fcst.preprocess(ray_dataset, static_features=[], dropna=False)
    preds2 = fcst.predict(10).to_pandas().sort_values(["unique_id", "ds"], ignore_index = True)
    pd.testing.assert_frame_equal(preds1, preds2)
