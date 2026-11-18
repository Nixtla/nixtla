"""Runtime benchmarks for pooled lag transforms, run by the CodSpeed job.

Not named `test_*.py` so the ordinary pytest run skips it; CodSpeed passes the
path explicitly. Peak memory lives in `mem_pooled.py`.

    MLF_BENCH_SERIES=1000 MLF_BENCH_TIMES=1000 pytest tests/bench_pooled.py --codspeed
"""

import copy
import os

import pandas as pd
import pytest

from tests._pooled_common import (
    CONFIG_IDS,
    CONFIGS,
    HORIZON,
    build_future,
    build_series,
    frame,
    make_forecast,
)

# Modest by default: CodSpeed simulation is slow and differences still show.
N_SERIES = int(os.environ.get("MLF_BENCH_SERIES", 150))
N_TIMES = int(os.environ.get("MLF_BENCH_TIMES", 150))

# the slowest case per round, so only the cheapest few configs run it
UPDATE_CONFIGS = [c for c in CONFIGS if not c.needs_promo][:4]


@pytest.fixture(scope="module")
def series():
    return build_series(N_SERIES, N_TIMES)


@pytest.fixture(scope="module")
def future_promo(series):
    return build_future(series)


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_pooled_preprocess(benchmark, series, config):
    df = frame(series, config)
    benchmark(
        lambda: make_forecast(config).preprocess(
            df, static_features=["brand"], dropna=False
        )
    )


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_pooled_predict(benchmark, series, future_promo, config):
    fcst = make_forecast(config, with_model=True)
    fcst.fit(frame(series, config), static_features=["brand"], dropna=False)
    x_df = future_promo if config.needs_future else None
    benchmark(lambda: fcst.predict(h=HORIZON, X_df=x_df))


@pytest.mark.parametrize("config", UPDATE_CONFIGS, ids=[c.name for c in UPDATE_CONFIGS])
def test_pooled_update(benchmark, series, config):
    """Incremental `update` of the pooled aggregates, without a refit."""
    df = frame(series, config)
    cut = df["ds"].max() - pd.Timedelta(days=5)
    head = df[df["ds"] <= cut]
    tail = df[df["ds"] > cut]
    fcst = make_forecast(config, with_model=True)
    fcst.fit(head, static_features=["brand"], dropna=False)

    def setup():
        # `update` mutates the fitted state, so each round needs its own copy of it;
        # made here, untimed, so the delta is the update and not the copy
        return (copy.deepcopy(fcst), tail), {}

    benchmark.pedantic(lambda f, new: f.update(new), setup=setup)
