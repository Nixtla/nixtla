import operator
import sys

import pandas as pd
import pytest
import ray

from mlforecast import MLForecast
from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.ray.lgb import RayLGBMForecast
from mlforecast.lag_transforms import Combine, ExpandingMean, RollingMean
from mlforecast.utils import generate_daily_series


@pytest.mark.ray
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_ray_combine_transforms():
    """Test Combine transform with Ray distributed forecast"""
    series = generate_daily_series(
        n_series=20,
        max_length=100,
        static_as_categorical=False,
        with_trend=True
    )

    ray_dataset = ray.data.from_pandas(series)
    ray_dataset = ray_dataset.map_batches(
        lambda batch: batch.assign(unique_id=batch['unique_id'].astype(str)),
        batch_format="pandas"
    )

    # Define lag transforms configuration
    lag_transforms_config = {
        1: [
            Combine(
                RollingMean(window_size=7, min_samples=1),
                ExpandingMean(),
                operator.sub,
            ),
            Combine(
                RollingMean(window_size=7, min_samples=1),
                ExpandingMean(),
                operator.add
            ),
            RollingMean(window_size=7, min_samples=1)
        ]
    }

    # Create distributed MLForecast with Ray
    dmlf = DistributedMLForecast(
        models=[RayLGBMForecast()],
        freq="D",
        lag_transforms=lag_transforms_config,
    )

    # Test preprocessing with Ray creates expected features
    ray_transformed_df = dmlf.preprocess(
        ray_dataset,
        id_col='unique_id',
        time_col='ds',
        target_col='y'
    )
    ray_result = ray_transformed_df.to_pandas().sort_values(["unique_id", "ds"], ignore_index = True)

    # Create local MLForecast for comparison (without model to avoid fitting)
    local_mlf = MLForecast(
        models=[],
        freq="D",
        lag_transforms=lag_transforms_config,
    )

    # Preprocess with local version
    local_result = local_mlf.preprocess(series).sort_values(["unique_id", "ds"], ignore_index = True)

    # Compare feature columns (both should have the same features)
    ray_feature_cols = [col for col in ray_result.columns if col.startswith(('lag', 'rolling', 'expanding'))]
    local_feature_cols = [col for col in local_result.columns if col.startswith(('lag', 'rolling', 'expanding'))]
    assert sorted(ray_feature_cols) == sorted(local_feature_cols), "Feature columns don't match"

    assert not ray_result.empty

    # Compare the generated features (allowing for floating point differences)
    for col in ray_feature_cols:
        pd.testing.assert_series_equal(
            ray_result[col],
            local_result[col],
            check_dtype=False,
            atol=1e-6
        )

    # Test fitting works with Ray
    dmlf.fit(ray_dataset)

    # Test can convert to local
    local_model = dmlf.to_local()
    assert local_model is not None
