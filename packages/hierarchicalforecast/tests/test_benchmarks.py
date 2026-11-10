import numpy as np
import pytest

from hierarchicalforecast.utils import (
    _lasso,
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
)


@pytest.fixture(params=[50, 200, 500, 1000])
def cov_data(request):
    n_ts = request.param
    n_samples = 100
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal((n_ts, n_samples))
    mask = np.ones_like(residuals, dtype=bool)
    return residuals, mask


def test_bench_ma_cov(benchmark, cov_data):
    residuals, mask = cov_data
    benchmark(_ma_cov, residuals, mask)


def test_bench_shrunk_cov_no_nans(benchmark, cov_data):
    residuals, _ = cov_data
    benchmark(_shrunk_covariance_schaferstrimmer_no_nans, residuals, 2e-8)


def test_bench_shrunk_cov_with_nans(benchmark, cov_data):
    residuals, mask = cov_data
    benchmark(_shrunk_covariance_schaferstrimmer_with_nans, residuals, mask, 2e-8)


@pytest.fixture
def lasso_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 50))
    y = rng.standard_normal(200)
    return X, y


def test_bench_lasso(benchmark, lasso_data):
    X, y = lasso_data
    benchmark(_lasso, X, y, 0.1, 1000, 1e-4)
