
import numpy as np
import pytest
import pytest_benchmark
from scipy import sparse
from hierarchicalforecast.methods import MinTrace, _ma_cov
#%% MinT benchmarks
# run using: pytest tests\test_benchmark.py -v -s --benchmark-min-rounds=20
def _create_mint_inputs(n_bottom_timeseries):
    # Create random hierarchy
    h = 100
    insample = 1000
    max_levels_random = 5
    max_categories_per_random_level = 10
    rng = np.random.default_rng(0)
    ones = np.ones(n_bottom_timeseries, dtype=np.float32)
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
    S_top = sparse.csr_matrix(ones, dtype=np.float32)
    S_bottom = sparse.eye(n_bottom_timeseries, dtype=np.float32, format="csr")
    # Construct S: stack top, aggregations and bottom 
    S_sp = sparse.vstack([S_top, S_aggs, S_bottom])

    y_hat_bottom = np.vstack([i * np.ones(h, dtype=np.float32) for i in range(n_bottom_timeseries)])
    y_hat_bottom_insample = np.vstack([i * np.ones(insample, dtype=np.float32) for i in range(n_bottom_timeseries)])
    y_bottom = y_hat_bottom_insample + rng.normal(size=(n_bottom_timeseries, insample))

    S = S_sp.toarray()
    y_insample = S @ y_bottom
    y_hat_insample = S @ y_hat_bottom_insample
    idx_bottom = np.arange(start=S.shape[0] - n_bottom_timeseries, stop=S.shape[0])
    y_hat=S @ y_hat_bottom

    return S, y_hat, y_insample, y_hat_insample, idx_bottom

@pytest.mark.parametrize("n_bottom_timeseries", [20, 200, 2000])
@pytest.mark.parametrize("with_nans", (False, True))
def test_mint(benchmark, n_bottom_timeseries, with_nans):
    S, y_hat, y_insample, y_hat_insample, idx_bottom = _create_mint_inputs(n_bottom_timeseries)
    if with_nans:
        y_insample[-1, :-1] = np.nan
        y_hat_insample[-1, :-1] = np.nan

    cls_min_trace = MinTrace(method='mint_shrink')
    result_min_trace = benchmark(cls_min_trace, S=S, y_hat=y_hat, y_insample=y_insample, y_hat_insample=y_hat_insample, idx_bottom=idx_bottom)

@pytest.mark.parametrize("n_bottom_timeseries", [20, 200, 2000])
@pytest.mark.parametrize("with_nans", (False, True))
def test_cov(benchmark, n_bottom_timeseries, with_nans):
    S, y_hat, y_insample, y_hat_insample, idx_bottom = _create_mint_inputs(n_bottom_timeseries)
    if with_nans:
        y_insample[-1, :-1] = np.nan
        y_hat_insample[-1, :-1] = np.nan

    residuals = (y_insample - y_hat_insample).astype(np.float32)
    nan_mask = np.isnan(residuals)

    result = benchmark(_ma_cov, residuals, ~nan_mask)
