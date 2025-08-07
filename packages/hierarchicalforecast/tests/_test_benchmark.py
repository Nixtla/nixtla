import numpy as np
import pandas as pd
import pytest
import pytest_benchmark  # noqa: F401
from scipy import sparse
from statsforecast.core import StatsForecast
from statsforecast.models import AutoETS

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import ERM, BottomUp, MinTrace
from hierarchicalforecast.utils import _ma_cov, aggregate


#%% MinT benchmarks
# run using: pytest tests\test_benchmark.py -v -s --benchmark-min-rounds=20
def _create_reconciler_inputs(n_bottom_timeseries):
    # Create random hierarchy
    h = 100
    insample = 1000
    max_levels_random = 5
    max_categories_per_random_level = 10
    rng = np.random.default_rng(0)
    ones = np.ones(n_bottom_timeseries, dtype=np.float64)
    idx_range = np.arange(n_bottom_timeseries)
    n_levels_random = rng.integers(1, max_levels_random + 1)
    S_aggs_list = []
    for _ in range(n_levels_random):
        n_categories_per_level = rng.integers(2, max_categories_per_random_level + 1)
        codes = rng.integers(0, n_categories_per_level, size=(n_bottom_timeseries, ))
        S_agg = sparse.csr_matrix((ones, (codes, idx_range)))
        S_aggs_list.append(S_agg)
    S_aggs = sparse.vstack(S_aggs_list)
    # Create top and bottom level
    S_top = sparse.csr_matrix(ones, dtype=np.float64)
    S_bottom = sparse.eye(n_bottom_timeseries, dtype=np.float64, format="csr")
    # Construct S: stack top, aggregations and bottom
    S_sp = sparse.vstack([S_top, S_aggs, S_bottom])

    y_hat_bottom = np.vstack([i * np.ones(h, dtype=np.float64) for i in range(n_bottom_timeseries)])
    y_hat_bottom_insample = np.vstack([i * np.ones(insample, dtype=np.float64) for i in range(n_bottom_timeseries)])
    y_bottom = y_hat_bottom_insample + rng.normal(size=(n_bottom_timeseries, insample))

    S = S_sp.toarray()
    y_insample = S @ y_bottom
    y_hat_insample = S @ y_hat_bottom_insample
    idx_bottom = np.arange(start=S.shape[0] - n_bottom_timeseries, stop=S.shape[0])
    y_hat=S @ y_hat_bottom

    return S, y_hat, y_insample, y_hat_insample, idx_bottom

@pytest.mark.parametrize("n_bottom_timeseries", [20])
@pytest.mark.parametrize("with_nans", (False, True))
def test_mint(benchmark, n_bottom_timeseries, with_nans):
    S, y_hat, y_insample, y_hat_insample, idx_bottom = _create_reconciler_inputs(n_bottom_timeseries)
    if with_nans:
        y_insample[-1, :-1] = np.nan
        y_hat_insample[-1, :-1] = np.nan

    cls_min_trace = MinTrace(method='mint_shrink')
    result_min_trace = benchmark(cls_min_trace, S=S, y_hat=y_hat, y_insample=y_insample, y_hat_insample=y_hat_insample, idx_bottom=idx_bottom) # noqa: F841

@pytest.mark.parametrize("n_bottom_timeseries", [20])
@pytest.mark.parametrize("with_nans", (False, True))
def test_cov(benchmark, n_bottom_timeseries, with_nans):
    S, y_hat, y_insample, y_hat_insample, idx_bottom = _create_reconciler_inputs(n_bottom_timeseries)
    if with_nans:
        y_insample[-1, :-1] = np.nan
        y_hat_insample[-1, :-1] = np.nan

    residuals = (y_insample - y_hat_insample)
    nan_mask = np.isnan(residuals)

    result = benchmark(_ma_cov, residuals, ~nan_mask) # noqa: F841

@pytest.mark.parametrize("n_bottom_timeseries", [10])
@pytest.mark.parametrize("erm_method", ['reg', 'reg_bu'])
def test_erm_reg(benchmark, n_bottom_timeseries, erm_method):
    S, y_hat, y_insample, y_hat_insample, idx_bottom = _create_reconciler_inputs(n_bottom_timeseries)

    cls_erm = ERM(method=erm_method)
    result_erm = benchmark(cls_erm, S=S, y_hat=y_hat, y_insample=y_insample, y_hat_insample=y_hat_insample, idx_bottom=idx_bottom) # noqa: F841

@pytest.fixture
def load_tourism():
    df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
    df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
    df.insert(0, 'Country', 'Australia')
    return df

# run with: pytest tests\test_benchmark.py::test_reconciler -v -s --benchmark-min-rounds=20 --disable-warnings
@pytest.mark.parametrize("reconciler", [MinTrace(method='mint_shrink'), BottomUp()])
def test_reconciler(benchmark, reconciler, load_tourism):

    # Load TourismSmall dataset
    df = load_tourism

    # Create hierarchical seires based on geographic levels and purpose
    # And Convert quarterly ds string to pd.datetime format
    hierarchy_levels = [['Country'],
                        ['Country', 'State'],
                        ['Country', 'State', 'Region'],
                        ['Country', 'State', 'Region', 'Purpose']]

    Y_df, S_df, tags = aggregate(df=df, spec=hierarchy_levels)
    qs = Y_df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
    Y_df['ds'] = pd.PeriodIndex(qs, freq='Q').to_timestamp()

    # Split train/test sets
    Y_test_df  = Y_df.groupby('unique_id').tail(8)
    Y_train_df = Y_df.drop(Y_test_df.index)

    # Compute base auto-ETS predictions
    # Careful identifying correct data freq, this data quarterly 'Q'
    fcst = StatsForecast(models=[AutoETS(season_length=4, model='ZZA')], freq='QS', n_jobs=-1)
    Y_hat_df = fcst.forecast(df=Y_train_df, h=8, fitted=True).reset_index()
    Y_fitted_df = fcst.forecast_fitted_values().reset_index()

    reconcilers = [reconciler]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)

    result = benchmark(hrec.reconcile, Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)    # noqa: F841