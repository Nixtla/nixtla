import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from conftest import assert_raises_with_message

from utilsforecast.compat import POLARS_INSTALLED
from utilsforecast.data import generate_series
from utilsforecast.plotting import plot_series

if POLARS_INSTALLED:
    import polars as pl

try:
    import plotly
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False

try:
    import plotly_resampler
    PLOTLY_RESAMPLER_INSTALLED = True
except ImportError:
    PLOTLY_RESAMPLER_INSTALLED = False


@pytest.fixture
def set_paths():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    IMG_PATH = ROOT_DIR / "nbs" / "imgs"
    IMG_PATH.mkdir(parents=True, exist_ok=True)
    return IMG_PATH


@pytest.fixture
def set_series():
    level = [80, 95]
    series = generate_series(
        4, freq="D", equal_ends=True, with_trend=True, n_models=2, level=level
    )
    test_pd = series.groupby("unique_id", observed=True).tail(10).copy()
    train_pd = series.drop(test_pd.index)
    return series, test_pd, train_pd, level


def test_plot_series(set_series, set_paths):
    _, test_pd, train_pd, level = set_series
    plt.style.use("ggplot")
    fig = plot_series(
        train_pd,
        forecasts_df=test_pd,
        ids=[0, 3],
        plot_random=False,
        level=level,
        max_insample_length=50,
        engine="matplotlib",
        plot_anomalies=True,
    )
    fig.savefig(set_paths / "plotting.png", bbox_inches="tight")


# Prepare combinations
bools = [True, False]
polars_opts = bools if POLARS_INSTALLED else [False]
anomalies = bools
randoms = bools
forecasts = bools
ids_list = [[0], [3, 1], None]
levels = [[80], None]
max_insample_lengths = [None, 50]

engines = ["matplotlib"]
if PLOTLY_INSTALLED:
    engines.append("plotly")
if PLOTLY_RESAMPLER_INSTALLED:
    engines.append("plotly-resampler")

@pytest.mark.parametrize("as_polars,ids,plot_anomalies,level,max_insample_length,engine,plot_random,with_forecasts", product(
    polars_opts, ids_list, anomalies, levels, max_insample_lengths, engines, randoms, forecasts
))
def test_plotting_combinations(
    as_polars,
    ids,
    plot_anomalies,
    level,
    max_insample_length,
    engine,
    plot_random,
    with_forecasts,
    set_series,
):
    _, test_pd, train_pd, _ = set_series

    if POLARS_INSTALLED and as_polars:
        train = pl.DataFrame(train_pd.to_records(index=False))
        test = pl.DataFrame(test_pd.to_records(index=False)) if with_forecasts else None
    else:
        train = train_pd
        test = test_pd if with_forecasts else None

    fn = lambda: plot_series(
        train,
        forecasts_df=test,
        ids=ids,
        plot_random=plot_random,
        plot_anomalies=plot_anomalies,
        level=level,
        max_insample_length=max_insample_length,
        engine=engine,
    )

    if level is None and plot_anomalies:
        assert_raises_with_message(fn, "specify the `level` argument")
    elif level is not None and plot_anomalies and not with_forecasts:
        assert_raises_with_message(fn, "provide a `forecasts_df` with prediction")
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DatetimeProperties.to_pydatetime is deprecated",
                category=FutureWarning,
            )
            fn()
