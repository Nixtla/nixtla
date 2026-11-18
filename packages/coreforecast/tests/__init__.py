import coreforecast.lag_transforms as lag_tf
import numpy as np
from coreforecast.scalers import (
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
)


def std_scaler_stats(x):
    return np.nanmean(x), np.nanstd(x)


def minmax_scaler_stats(x):
    min, max = np.nanmin(x), np.nanmax(x)
    return min, max - min


def robust_scaler_iqr_stats(x):
    q25, median, q75 = np.nanquantile(x, [0.25, 0.5, 0.75])
    return median, q75 - q25


def robust_scaler_mad_stats(x):
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    return median, mad


def scaler_transform(x, stats):
    offset, scale = stats
    return (x - offset) / scale


def scaler_inverse_transform(x, stats):
    offset, scale = stats
    return x * scale + offset


scaler2fns = {
    "standard": std_scaler_stats,
    "minmax": minmax_scaler_stats,
    "robust-iqr": robust_scaler_iqr_stats,
    "robust-mad": robust_scaler_mad_stats,
}
scaler2core = {
    "standard": LocalStandardScaler(),
    "minmax": LocalMinMaxScaler(),
    "robust-iqr": LocalRobustScaler("iqr"),
    "robust-mad": LocalRobustScaler("mad"),
}
scalers = list(scaler2fns.keys())
dtypes = [np.float32, np.float64]

season_length = 7
window_size = 4
min_samples = 2
lag_tfms_map = {
    "rolling_mean": (
        lag_tf.RollingMean,
        [window_size, min_samples],
    ),
    "rolling_std": (lag_tf.RollingStd, [window_size, min_samples]),
    "rolling_min": (lag_tf.RollingMin, [window_size, min_samples]),
    "rolling_max": (lag_tf.RollingMax, [window_size, min_samples]),
    "seasonal_rolling_mean": (
        lag_tf.SeasonalRollingMean,
        [season_length, window_size, min_samples],
    ),
    "seasonal_rolling_std": (
        lag_tf.SeasonalRollingStd,
        [season_length, window_size, min_samples],
    ),
    "seasonal_rolling_min": (
        lag_tf.SeasonalRollingMin,
        [season_length, window_size, min_samples],
    ),
    "seasonal_rolling_max": (
        lag_tf.SeasonalRollingMax,
        [season_length, window_size, min_samples],
    ),
    "expanding_mean": (lag_tf.ExpandingMean, []),
    "expanding_std": (lag_tf.ExpandingStd, []),
    "expanding_min": (lag_tf.ExpandingMin, []),
    "expanding_max": (lag_tf.ExpandingMax, []),
    "ewm_mean": (lag_tf.ExponentiallyWeightedMean, [0.8]),
}
