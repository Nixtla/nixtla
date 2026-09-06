"""Panel, model stub and config table shared by the pooled benchmark and memory suites.

A plain module rather than a conftest: `mem_pooled.py` spawns child
processes that import it directly, and anything imported here lands in the RSS
those children report.
"""

from typing import Callable, List, NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExponentiallyWeightedMean,
    LookupLag,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingQuantile,
    RollingStd,
    SeasonalRollingMean,
)

HORIZON = 12
N_BRANDS = 20
N_PROMO = 3
#: high-cardinality dynamic key: with one bucket per (series, value) this is the
#: shape whose fit block is widest, which the other configs do not reach
N_SKU = 50


class Constant(BaseEstimator):
    """Zero-cost model, so the measurement stays on the feature engine."""

    def fit(self, X, y=None):  # noqa: ARG002
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Config(NamedTuple):
    name: str
    #: a factory: fitting mutates the transforms, so each use needs a fresh list
    transforms: Callable[[], List]
    #: whether the case needs the dynamic `promo` partition column
    needs_promo: bool
    #: whether it needs the high-cardinality `sku` one; defaulted so the
    #: existing cases keep exactly the frame they had
    needs_sku: bool = False

    @property
    def needs_future(self) -> bool:
        """Whether predict needs an `X_df` carrying the dynamic key."""
        return self.needs_promo or self.needs_sku


#: the cases both suites measure; an entry added here shows up in both
CONFIGS = [
    Config("global_rolling_mean", lambda: [RollingMean(7, global_=True)], False),
    Config(
        "global_rolling_all_stats",
        lambda: [
            RollingMean(7, global_=True),
            RollingStd(7, global_=True),
            RollingMin(7, global_=True),
            RollingMax(7, global_=True),
        ],
        False,
    ),
    Config("groupby_rolling_mean", lambda: [RollingMean(7, groupby=["brand"])], False),
    Config("global_expanding_mean", lambda: [ExpandingMean(global_=True)], False),
    Config(
        "global_ewm",
        lambda: [ExponentiallyWeightedMean(alpha=0.3, global_=True)],
        False,
    ),
    Config(
        "global_seasonal_rolling",
        lambda: [SeasonalRollingMean(season_length=7, window_size=4, global_=True)],
        False,
    ),
    Config(
        "time_aggs_same_bucket",
        lambda: [
            RollingMean(7, global_=True),
            RollingMean(7, global_=True, time_agg="sum"),
            RollingMean(7, global_=True, time_agg="mean"),
        ],
        False,
    ),
    Config(
        "local_partition_rolling",
        lambda: [RollingMean(7, partition_by=["promo"])],
        True,
    ),
    Config(
        "global_partition_rolling",
        lambda: [RollingMean(7, global_=True, partition_by=["promo"])],
        True,
    ),
    Config(
        "partition_expanding",
        lambda: [ExpandingMean(partition_by=["promo"])],
        True,
    ),
    Config(
        "mixed_partition_shared",
        lambda: [
            RollingMean(7, partition_by=["promo"]),
            ExpandingMean(partition_by=["promo"]),
        ],
        True,
    ),
    Config(
        "global_rolling_quantile",
        lambda: [RollingQuantile(p=0.5, window_size=7, global_=True)],
        False,
    ),
    Config(
        "partition_rolling_quantile",
        lambda: [RollingQuantile(p=0.5, window_size=7, partition_by=["promo"])],
        True,
    ),
    Config("lookup_lag_partition", lambda: [LookupLag(partition_by=["promo"])], True),
    Config(
        "partition_high_cardinality",
        lambda: [RollingMean(7, partition_by=["sku"])],
        False,
        needs_sku=True,
    ),
    Config(
        "mixed_realistic",
        lambda: [
            RollingMean(7, global_=True),
            RollingStd(28, groupby=["brand"]),
            ExpandingMean(groupby=["brand"]),
            ExponentiallyWeightedMean(alpha=0.3, global_=True),
        ],
        False,
    ),
]
CONFIG_IDS = [c.name for c in CONFIGS]
CONFIG_BY_NAME = {c.name: c for c in CONFIGS}


def build_series(n_series, n_times):
    """A dense daily panel with a static (`brand`) and a dynamic (`promo`) key."""
    rng = np.random.default_rng(0)
    n = n_series * n_times
    end = pd.Timestamp("2024-01-01") + pd.Timedelta(days=n_times - 1)
    return pd.DataFrame(
        {
            "unique_id": np.repeat([f"s{i:05d}" for i in range(n_series)], n_times),
            "ds": np.tile(pd.date_range(end=end, periods=n_times, freq="D"), n_series),
            "y": rng.normal(10, 3, n),
            "brand": np.repeat([f"b{i % N_BRANDS}" for i in range(n_series)], n_times),
            "promo": rng.integers(0, N_PROMO, n),
            "sku": rng.integers(0, N_SKU, n),
        }
    )


def build_future(series, horizon=HORIZON):
    """One row per (series, horizon step) carrying the dynamic partition key."""
    uids = np.sort(series["unique_id"].unique())
    end = series["ds"].max()
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "unique_id": np.repeat(uids, horizon),
            "ds": np.tile(
                pd.date_range(end + pd.Timedelta(days=1), periods=horizon, freq="D"),
                len(uids),
            ),
            "promo": rng.integers(0, N_PROMO, len(uids) * horizon),
            "sku": rng.integers(0, N_SKU, len(uids) * horizon),
        }
    )


def frame(series, config):
    """The panel as `config` wants it: a dynamic key dropped unless it is used."""
    drop = [
        col
        for col, needed in (("promo", config.needs_promo), ("sku", config.needs_sku))
        if not needed
    ]
    return series.drop(columns=drop) if drop else series


def make_forecast(config, with_model=False):
    """An `MLForecast` for `config`; `with_model` only when fit/predict is needed."""
    return MLForecast(
        models=[Constant()] if with_model else [],
        freq="D",
        lags=[1] if with_model else None,
        lag_transforms={1: config.transforms()},
    )
