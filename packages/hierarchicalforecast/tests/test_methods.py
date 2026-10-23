from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse

from hierarchicalforecast.methods import (
    ERM,
    BottomUp,
    MiddleOut,
    MiddleOutSparse,
    MinTrace,
    MinTraceSparse,
    OptimalCombination,
    TopDown,
    TopDownSparse,
    _is_strictly_hierarchical,
    is_strictly_hierarchical,
)
from hierarchicalforecast.utils import _construct_adjacency_matrix


@dataclass
class HierarchicalTestData:
    """Test data for hierarchical forecasting methods."""
    S: np.ndarray  # Summing matrix
    h: int  # Forecast horizon
    y_bottom: np.ndarray  # Bottom level historical data
    y_hat_bottom: np.ndarray  # Bottom level forecasts
    y_hat_bottom_insample: np.ndarray  # Bottom level insample forecasts
    tags: dict[str, np.ndarray]  # Hierarchy level tags

@pytest.fixture
def hierarchical_data():
    """Fixture providing test data for hierarchical forecasting methods."""
    # Summing matrix for 3-level hierarchy (7x4)
    # Level 1: Total (1 series)
    # Level 2: Two groups (2 series)
    # Level 3: Four bottom series (4 series)
    S = np.array([
        [1.0, 1.0, 1.0, 1.0],  # Total
        [1.0, 1.0, 0.0, 0.0],  # Group 1
        [0.0, 0.0, 1.0, 1.0],  # Group 2
        [1.0, 0.0, 0.0, 0.0],  # Bottom 1
        [0.0, 1.0, 0.0, 0.0],  # Bottom 2
        [0.0, 0.0, 1.0, 0.0],  # Bottom 3
        [0.0, 0.0, 0.0, 1.0],  # Bottom 4
    ])

    h = 2  # Forecast horizon
    base_series = np.array([10.0, 5.0, 4.0, 2.0, 1.0])

    # Create bottom level historical data (4 series x 5 time periods)
    y_bottom = np.vstack([i * base_series for i in range(1, 5)])

    # Create insample forecasts with NaN in first period
    y_hat_bottom_insample = np.roll(y_bottom, 1, axis=1)
    y_hat_bottom_insample[:, 0] = np.nan

    # Create simple forecasts for bottom series
    y_hat_bottom = np.vstack([i * np.ones(h) for i in range(1, 5)])

    tags = {
        "level1": np.array([0]),
        "level2": np.array([1, 2]),
        "level3": np.array([3, 4, 5, 6])
    }

    return HierarchicalTestData(
        S=S,
        h=h,
        y_bottom=y_bottom,
        y_hat_bottom=y_hat_bottom,
        y_hat_bottom_insample=y_hat_bottom_insample,
        tags=tags
    )

def test_sigmah_hierarchy(hierarchical_data):
    """Test bottom up forecast recovery matches expected output."""
    data = hierarchical_data
    cls_bottom_up = BottomUp()
    result = cls_bottom_up(S=data.S, y_hat=data.S @ data.y_hat_bottom)["mean"]
    expected = data.S @ data.y_hat_bottom
    assert result.all() == expected.all()


def test_btm_up_forecast_recovery(hierarchical_data):
    """Test bottom up forecast recovery."""
    data = hierarchical_data
    cls_bottom_up = BottomUp()
    bu_bootstrap_intervals = cls_bottom_up(
        S=data.S,
        y_hat=data.S @ data.y_hat_bottom,
    )
    expected = data.S @ data.y_hat_bottom
    assert bu_bootstrap_intervals["mean"].all() == expected.all()


def test_forecast_recovery_fit_predict(hierarchical_data):
    """Test forecast recovery with fit -> predict pattern."""
    data = hierarchical_data
    cls_bottom_up = BottomUp()
    cls_bottom_up.fit(S=data.S, y_hat=data.S @ data.y_hat_bottom)
    y_tilde = cls_bottom_up.predict(S=data.S, y_hat=data.S @ data.y_hat_bottom)["mean"]
    expected = data.S @ data.y_hat_bottom
    assert y_tilde.all() == expected.all()


def test_not_fitted(hierarchical_data):
    """Test not fitted message for unfitted predict."""
    data = hierarchical_data
    cls_bottom_up = BottomUp()
    with pytest.raises(Exception) as exec_info:
        cls_bottom_up.predict(data.S, data.S @ data.y_hat_bottom)
    assert "not fitted yet" in str(exec_info.value)


@pytest.fixture
def tags_non_hier():
    return {
        "Country": np.array([0]),
        "Country/City": np.array([2, 1]),
        "Country/Transgender": np.array([3, 4]),
        "Country-City-Transgender": np.array([5, 6, 7, 8]),
    }

@pytest.fixture
def S_non_hier():
    return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],  # total
                [1.0, 1.0, 0.0, 0.0],  # city 1
                [0.0, 0.0, 1.0, 1.0],  # city 2
                [1.0, 0.0, 1.0, 0.0],  # transgender
                [0.0, 1.0, 0.0, 1.0],  # no transgender
                [1.0, 0.0, 0.0, 0.0],  # city 1 - transgender
                [0.0, 1.0, 0.0, 0.0],  # city 1 - no transgender
                [0.0, 0.0, 1.0, 0.0],  # city 2 - transgender
                [0.0, 0.0, 0.0, 1.0],  # city 2 - no transgender
            ]
        )


def test_is_strictly_hierarchical(hierarchical_data, S_non_hier, tags_non_hier):
    data = hierarchical_data
    assert is_strictly_hierarchical(data.S, data.tags)
    assert not is_strictly_hierarchical(S_non_hier, tags_non_hier)


def test_hidden_is_strictly_hierarchical(hierarchical_data, S_non_hier, tags_non_hier):
    data = hierarchical_data
    A = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
        ]
    )
    assert _construct_adjacency_matrix(sparse.csr_matrix(data.S), data.tags).toarray().all() == A.all()
    assert _is_strictly_hierarchical(sparse.csr_matrix(A, dtype=bool), data.tags)
    A_non_hier = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1],
        ]
    )
    assert _construct_adjacency_matrix(sparse.csr_matrix(S_non_hier), tags_non_hier).toarray().all() == A_non_hier.all()
    assert not _is_strictly_hierarchical(sparse.csr_matrix(A_non_hier, dtype=bool), tags_non_hier)


@pytest.mark.parametrize("method", ["forecast_proportions", "average_proportions", "proportion_averages"])
def test_top_down_forecast_recovery(hierarchical_data, method):
    """Test TopDown methods can recover forecasts when time series share same proportions."""
    data = hierarchical_data

    cls_top_down = TopDown(method=method)
    if cls_top_down.insample:
        assert method in ["average_proportions", "proportion_averages"]
        result = cls_top_down(
            S=data.S, y_hat=data.S @ data.y_hat_bottom, y_insample=data.S @ data.y_bottom, tags=data.tags
        )["mean"]
    else:
        result = cls_top_down(S=data.S, y_hat=data.S @ data.y_hat_bottom, tags=data.tags)["mean"]

    np.testing.assert_allclose(result, data.S @ data.y_hat_bottom)


def test_top_down_sparse_hierarchical_error(S_non_hier, tags_non_hier):
    """Test TopDownSparse raises error for non-hierarchical structures."""
    cls_top_down = TopDownSparse(method="average_proportions")
    # Create dummy data
    y_hat = np.ones((S_non_hier.shape[0], 2))
    y_insample = np.ones((S_non_hier.shape[0], 5))
    with pytest.raises(Exception, match="Top-down reconciliation requires strictly hierarchical structures"):
        cls_top_down(S=sparse.csr_matrix(S_non_hier), y_hat=y_hat, y_insample=y_insample, tags=tags_non_hier)


@pytest.mark.parametrize("method", ["forecast_proportions", "average_proportions", "proportion_averages"])
def test_top_down_sparse_forecast_recovery(hierarchical_data, method):
    """Test TopDownSparse methods can recover forecasts."""
    data = hierarchical_data

    cls_top_down = TopDownSparse(method=method)
    if cls_top_down.insample:
        assert method in ["average_proportions", "proportion_averages"]
        result = cls_top_down(
            S=sparse.csr_matrix(data.S),
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]
    else:
        result = cls_top_down(S=sparse.csr_matrix(data.S), y_hat=data.S @ data.y_hat_bottom, tags=data.tags)["mean"]

    np.testing.assert_allclose(result, data.S @ data.y_hat_bottom)


@pytest.mark.parametrize("method", ["forecast_proportions", "average_proportions", "proportion_averages"])
def test_top_down_vs_sparse_equivalence(hierarchical_data, method):
    """Test TopDown and TopDownSparse produce equivalent results."""
    data = hierarchical_data

    cls_top_down = TopDown(method=method)
    cls_top_down_sparse = TopDownSparse(method=method)

    if cls_top_down.insample:
        assert method in ["average_proportions", "proportion_averages"]
        result_dense = cls_top_down(
            S=data.S, y_hat=data.S @ data.y_hat_bottom, y_insample=data.S @ data.y_bottom, tags=data.tags
        )["mean"]
        result_sparse = cls_top_down_sparse(
            S=sparse.csr_matrix(data.S),
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]
        tolerance = np.finfo(np.float64).eps
    else:
        result_dense = cls_top_down(S=data.S, y_hat=data.S @ data.y_hat_bottom, tags=data.tags)["mean"]
        result_sparse = cls_top_down_sparse(
            S=sparse.csr_matrix(data.S),
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]
        tolerance = 1e-9

    np.testing.assert_allclose(result_dense, result_sparse, tolerance)


@pytest.mark.parametrize("method", ["average_proportions", "proportion_averages"])
def test_top_down_nan_handling(hierarchical_data, method):
    """Test TopDown handles NaN values correctly in insample data."""
    data = hierarchical_data

    cls_top_down = TopDown(method=method)

    # Original result with zeros
    y_insample_orig = data.S @ data.y_bottom
    y_insample_orig[-1, :] = 0
    result_orig = cls_top_down(
        S=data.S, y_hat=data.S @ data.y_hat_bottom, y_insample=y_insample_orig, tags=data.tags
    )["mean"]

    # Result with NaN values
    y_insample_nan = y_insample_orig.copy()
    y_insample_nan[-1, 0] = np.nan
    result_nan = cls_top_down(
        S=data.S, y_hat=data.S @ data.y_hat_bottom, y_insample=y_insample_nan, tags=data.tags
    )["mean"]

    np.testing.assert_allclose(result_orig, result_nan)






@pytest.mark.parametrize("method", ["forecast_proportions", "average_proportions", "proportion_averages"])
def test_middle_out_forecast_recovery(hierarchical_data, method):
    """Test MiddleOut methods can recover forecasts when time series share same proportions."""
    data = hierarchical_data

    cls_middle_out = MiddleOut(middle_level="level2", top_down_method=method)
    if cls_middle_out.insample:
        assert method in ["average_proportions", "proportion_averages"]
        result = cls_middle_out(
            S=data.S, y_hat=data.S @ data.y_hat_bottom, y_insample=data.S @ data.y_bottom, tags=data.tags
        )["mean"]
    else:
        result = cls_middle_out(S=data.S, y_hat=data.S @ data.y_hat_bottom, tags=data.tags)["mean"]

    np.testing.assert_allclose(result, data.S @ data.y_hat_bottom)


@pytest.mark.parametrize("method", ["forecast_proportions", "average_proportions", "proportion_averages"])
def test_middle_out_sparse_forecast_recovery(hierarchical_data, method):
    """Test MiddleOutSparse methods can recover forecasts."""
    data = hierarchical_data

    cls_middle_out = MiddleOutSparse(middle_level="level2", top_down_method=method)
    if cls_middle_out.insample:
        assert method in ["average_proportions", "proportion_averages"]
        result = cls_middle_out(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]
    else:
        result = cls_middle_out(S=data.S, y_hat=data.S @ data.y_hat_bottom, tags=data.tags)["mean"]

    np.testing.assert_allclose(result, data.S @ data.y_hat_bottom)


@pytest.mark.parametrize("method", ["forecast_proportions", "average_proportions", "proportion_averages"])
def test_middle_out_vs_sparse_equivalence(hierarchical_data, method):
    """Test MiddleOut and MiddleOutSparse produce equivalent results."""
    data = hierarchical_data

    cls_middle_out = MiddleOut(middle_level="level2", top_down_method=method)
    cls_middle_out_sparse = MiddleOutSparse(middle_level="level2", top_down_method=method)

    if cls_middle_out.insample:
        assert method in ["average_proportions", "proportion_averages"]
        result_dense = cls_middle_out(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]
        result_sparse = cls_middle_out_sparse(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]
        tolerance = np.finfo(np.float64).eps
    else:
        result_dense = cls_middle_out(S=data.S, y_hat=data.S @ data.y_hat_bottom, tags=data.tags)["mean"]
        result_sparse = cls_middle_out_sparse(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]
        tolerance = np.finfo(np.float64).eps

    np.testing.assert_allclose(result_dense, result_sparse, tolerance)


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink", "emint"])
@pytest.mark.parametrize("nonnegative", [False, True])
def test_min_trace_forecast_recovery(hierarchical_data, method, nonnegative):
    """Test MinTrace methods can recover forecasts and nonnegative behavior."""
    data = hierarchical_data

    cls_min_trace = MinTrace(method=method, nonnegative=nonnegative)
    assert cls_min_trace.nonnegative is nonnegative

    if cls_min_trace.insample:
        assert method in ["wls_var", "mint_cov", "mint_shrink", "emint"]
        result = cls_min_trace(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            y_hat_insample=data.S @ data.y_hat_bottom_insample,
        )["mean"]
    else:
        result = cls_min_trace(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
        )["mean"]

    # EMinT without nonnegative only guarantees coherence, not exact recovery
    # (when nonnegative=True, it uses BottomUp which does have exact recovery)
    if method == "emint" and not nonnegative:
        # Check coherence instead of exact recovery
        bottom_forecasts = result[list(range(data.S.shape[0] - data.S.shape[1], data.S.shape[0])), :]
        aggregated = data.S @ bottom_forecasts
        np.testing.assert_allclose(result, aggregated, rtol=1e-10)
    else:
        np.testing.assert_allclose(result, data.S @ data.y_hat_bottom)


def test_min_trace_threading(hierarchical_data):
    """Test MinTrace produces same results with different thread counts."""
    data = hierarchical_data

    mintrace_1thr = MinTrace(method="ols", nonnegative=False, num_threads=1).fit(
        S=data.S, y_hat=data.S @ data.y_hat_bottom
    )
    mintrace_2thr = MinTrace(method="ols", nonnegative=False, num_threads=2).fit(
        S=data.S, y_hat=data.S @ data.y_hat_bottom
    )

    np.testing.assert_allclose(mintrace_1thr.y_hat, mintrace_2thr.y_hat)


@pytest.mark.parametrize("nonnegative", [False, True])
def test_min_trace_mint_cov_error(hierarchical_data, nonnegative):
    """Test MinTrace raises error for mint_cov method."""
    data = hierarchical_data

    cls_min_trace = MinTrace(method="mint_cov", nonnegative=nonnegative)
    with pytest.raises(Exception, match="min_trace.*mint_cov"):
        cls_min_trace(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            y_hat_insample=data.S @ data.y_hat_bottom_insample,
        )


def test_min_trace_shrink_covariance_stress(hierarchical_data):
    """Test MinTrace-shr covariance with different length data."""
    data = hierarchical_data

    diff_len_y_insample = data.S @ data.y_bottom
    diff_len_y_hat_insample = data.S @ data.y_hat_bottom_insample
    diff_len_y_insample[-1, :-1] = np.nan
    diff_len_y_hat_insample[-1, :-1] = np.nan

    cls_min_trace = MinTrace(method="mint_shrink")
    result_min_trace = cls_min_trace(
        S=data.S,
        y_hat=data.S @ data.y_hat_bottom,
        y_insample=diff_len_y_insample,
        y_hat_insample=diff_len_y_hat_insample,
    )

    # Test passes if no exception is raised
    assert result_min_trace is not None


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink", "emint"])
@pytest.mark.parametrize("nonnegative", [False, True])
def test_min_trace_levels(hierarchical_data, method, nonnegative):
    """Test MinTrace with all levels."""
    data = hierarchical_data

    cls_min_trace = MinTrace(method=method, nonnegative=nonnegative)
    result = cls_min_trace(
        S=data.S,
        y_hat=data.S @ data.y_hat_bottom,
        y_insample=data.S @ data.y_bottom,
        y_hat_insample=data.S @ data.y_hat_bottom_insample,
    )["mean"]

    # EMinT without nonnegative only guarantees coherence, not exact recovery
    # (when nonnegative=True, it uses BottomUp which does have exact recovery)
    if method == "emint" and not nonnegative:
        # Check coherence instead of exact recovery
        bottom_forecasts = result[list(range(data.S.shape[0] - data.S.shape[1], data.S.shape[0])), :]
        aggregated = data.S @ bottom_forecasts
        np.testing.assert_allclose(result, aggregated, rtol=1e-10)
    else:
        np.testing.assert_allclose(result, data.S @ data.y_hat_bottom)


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var"])
@pytest.mark.parametrize("nonnegative,qp", [(False, False), (True, False), (True, True)])
def test_min_trace_sparse_functionality(hierarchical_data, method, nonnegative, qp):
    """Test MinTraceSparse with non-negative heuristic and QP solutions."""
    data = hierarchical_data

    cls_min_trace = MinTraceSparse(method=method, nonnegative=nonnegative, qp=qp)
    result = cls_min_trace(
        S=sparse.csr_matrix(data.S),
        y_hat=data.S @ data.y_hat_bottom,
        y_insample=data.S @ data.y_bottom,
        y_hat_insample=data.S @ data.y_hat_bottom_insample,
    )["mean"]

    np.testing.assert_allclose(result, data.S @ data.y_hat_bottom)


@pytest.mark.parametrize("method", ["ols", "wls_struct"])
@pytest.mark.parametrize("nonnegative", [False, True])
def test_optimal_combination_methods(hierarchical_data, method, nonnegative):
    # test nonnegative behavior
    # we should be able to recover the same forecasts
    # in this example
    cls_optimal_combination = OptimalCombination(
        method=method, nonnegative=nonnegative
    )
    np.testing.assert_allclose(
        cls_optimal_combination(
            S=hierarchical_data.S,
            y_hat=hierarchical_data.S @ hierarchical_data.y_hat_bottom,
        )["mean"],
        hierarchical_data.S @ hierarchical_data.y_hat_bottom,
    )



def test_erm_forecast_recovery(hierarchical_data):
    """Test ERM method can recover forecasts."""
    data = hierarchical_data

    cls_erm = ERM(method="reg_bu", lambda_reg=None)
    result = cls_erm(
        S=data.S,
        y_hat=data.S @ data.y_hat_bottom,
        y_insample=data.S @ data.y_bottom,
        y_hat_insample=data.S @ data.y_hat_bottom_insample,
    )["mean"]

    np.testing.assert_allclose(result, data.S @ data.y_hat_bottom)


@pytest.fixture
def interval_reconciler_args(hierarchical_data):
    """Fixture providing reconciler arguments for interval tests."""
    data = hierarchical_data
    y_base = data.S @ data.y_bottom
    y_hat_base = data.S @ data.y_hat_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample
    sigma = np.nansum((y_base - y_hat_base_insample) ** 2, axis=1) / (y_base.shape[1] - 1)
    sigma = np.sqrt(sigma)
    sigmah = sigma[:, None] * np.sqrt(
        np.vstack([np.arange(1, data.h + 1) for _ in range(y_base.shape[0])])
    )

    return dict(
        S=data.S,
        y_hat=y_hat_base,
        y_insample=y_base,
        y_hat_insample=y_hat_base_insample,
        sigmah=sigmah,
        level=[80, 90],
        intervals_method="normality",
        num_samples=200,
        seed=0,
        tags=data.tags,
    )


def test_bottom_up_normality_intervals(hierarchical_data, interval_reconciler_args):
    """Test bottom up normality prediction intervals recover original sigmah."""
    data = hierarchical_data

    cls_bottom_up = BottomUp()
    result = cls_bottom_up(**interval_reconciler_args)

    idx_bottom = list(range(data.S.shape[0] - data.S.shape[1], data.S.shape[0]))
    assert np.array_equal(result["sigmah"][idx_bottom], interval_reconciler_args["sigmah"][idx_bottom])


def test_bottom_up_interval_names(interval_reconciler_args):
    """Test bottom up normality interval names."""
    cls_bottom_up = BottomUp()
    bu_intervals = cls_bottom_up(**interval_reconciler_args)

    assert list(bu_intervals.keys()) == ["mean", "sigmah", "quantiles"]


def test_bottom_up_permbu_interval_names(interval_reconciler_args):
    """Test bottom up PERMBU interval names."""
    interval_reconciler_args["intervals_method"] = "permbu"
    cls_bottom_up = BottomUp()
    bu_permbu_intervals = cls_bottom_up(**interval_reconciler_args)

    assert list(bu_permbu_intervals.keys()) == ["mean", "quantiles"]


@pytest.mark.parametrize("method", ["average_proportions", "proportion_averages"])
@pytest.mark.parametrize("intervals_method", ["normality", "bootstrap", "permbu"])
def test_top_down_intervals(interval_reconciler_args, method, intervals_method):
    """Test TopDown with different interval methods."""
    interval_reconciler_args["intervals_method"] = intervals_method
    cls_top_down = TopDown(method=method)

    # Test passes if no exception is raised
    result = cls_top_down(**interval_reconciler_args)
    assert result is not None


def test_top_down_forecast_proportions_intervals(hierarchical_data):
    """Test TopDown forecast_proportions with prediction intervals."""
    data = hierarchical_data
    y_hat = data.S @ data.y_hat_bottom
    y_insample = data.S @ data.y_bottom
    y_hat_insample = data.S @ data.y_hat_bottom_insample

    cls_top_down = TopDown(method="forecast_proportions")
    result = cls_top_down(
        S=data.S,
        y_hat=y_hat,
        tags=data.tags,
        y_insample=y_insample,
        y_hat_insample=y_hat_insample,
        level=[80, 90],
        intervals_method="bootstrap",
        num_samples=100,
        seed=42,
    )

    # Check mean shape
    assert result["mean"].shape == y_hat.shape
    # Check quantiles shape: (n_series, horizon, 4) for levels [80, 90]
    assert result["quantiles"].shape == (y_hat.shape[0], y_hat.shape[1], 4)
    # Check quantiles are ordered (lower bounds < upper bounds)
    assert np.all(result["quantiles"][:, :, 0] <= result["quantiles"][:, :, 1])
    assert np.all(result["quantiles"][:, :, 2] <= result["quantiles"][:, :, 3])


def test_top_down_forecast_proportions_intervals_missing_insample(hierarchical_data):
    """Test TopDown forecast_proportions raises error when insample data missing."""
    data = hierarchical_data
    y_hat = data.S @ data.y_hat_bottom

    cls_top_down = TopDown(method="forecast_proportions")
    with pytest.raises(ValueError, match="require.*y_insample.*y_hat_insample"):
        cls_top_down(
            S=data.S,
            y_hat=y_hat,
            tags=data.tags,
            level=[80, 90],
        )


def test_top_down_sparse_forecast_proportions_intervals(hierarchical_data):
    """Test TopDownSparse forecast_proportions with prediction intervals."""
    data = hierarchical_data
    y_hat = data.S @ data.y_hat_bottom
    y_insample = data.S @ data.y_bottom
    y_hat_insample = data.S @ data.y_hat_bottom_insample

    cls_top_down = TopDownSparse(method="forecast_proportions")
    result = cls_top_down(
        S=sparse.csr_matrix(data.S),
        y_hat=y_hat,
        tags=data.tags,
        y_insample=y_insample,
        y_hat_insample=y_hat_insample,
        level=[80, 90],
        intervals_method="bootstrap",
        num_samples=100,
        seed=42,
    )

    # Check mean shape
    assert result["mean"].shape == y_hat.shape
    # Check quantiles shape: (n_series, horizon, 4) for levels [80, 90]
    assert result["quantiles"].shape == (y_hat.shape[0], y_hat.shape[1], 4)
    # Check quantiles are ordered
    assert np.all(result["quantiles"][:, :, 0] <= result["quantiles"][:, :, 1])
    assert np.all(result["quantiles"][:, :, 2] <= result["quantiles"][:, :, 3])


def test_top_down_sparse_forecast_proportions_intervals_missing_insample(hierarchical_data):
    """Test TopDownSparse forecast_proportions raises error when insample data missing."""
    data = hierarchical_data
    y_hat = data.S @ data.y_hat_bottom

    cls_top_down = TopDownSparse(method="forecast_proportions")
    with pytest.raises(ValueError, match="require.*y_insample.*y_hat_insample"):
        cls_top_down(
            S=sparse.csr_matrix(data.S),
            y_hat=y_hat,
            tags=data.tags,
            level=[80, 90],
        )


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink", "emint"])
@pytest.mark.parametrize("intervals_method", ["normality", "bootstrap", "permbu"])
def test_min_trace_intervals(interval_reconciler_args, method, intervals_method):
    """Test MinTrace with different interval methods."""
    interval_reconciler_args["intervals_method"] = intervals_method
    cls_min_trace = MinTrace(method=method, nonnegative=False)

    # Test passes if no exception is raised
    result = cls_min_trace(**interval_reconciler_args)
    assert result is not None


@pytest.mark.parametrize("method", ["ols", "wls_struct"])
@pytest.mark.parametrize("intervals_method", ["normality", "bootstrap", "permbu"])
def test_optimal_combination_intervals(interval_reconciler_args, method, intervals_method):
    """Test OptimalCombination with different interval methods."""
    interval_reconciler_args["intervals_method"] = intervals_method
    cls_optimal_combination = OptimalCombination(method=method, nonnegative=False)

    # Test passes if no exception is raised (only for non-negative=False)
    result = cls_optimal_combination(**interval_reconciler_args)
    assert result is not None


@pytest.mark.parametrize("intervals_method", ["normality", "bootstrap", "permbu"])
def test_erm_intervals(interval_reconciler_args, intervals_method):
    """Test ERM with different interval methods."""
    interval_reconciler_args["intervals_method"] = intervals_method
    cls_erm = ERM(method="reg_bu", lambda_reg=None)

    # Test passes if no exception is raised
    result = cls_erm(**interval_reconciler_args)
    assert result is not None


@pytest.mark.parametrize("intervals_method", ["normality", "bootstrap", "permbu"])
def test_coherent_sample_shapes(hierarchical_data, intervals_method):
    """Test coherent sample shapes are consistent across interval methods."""
    data = hierarchical_data
    y_base = data.S @ data.y_bottom
    y_hat_base = data.S @ data.y_hat_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample
    sigma = np.nansum((y_base - y_hat_base_insample) ** 2, axis=1) / (y_base.shape[1] - 1)
    sigma = np.sqrt(sigma)
    sigmah = sigma[:, None] * np.sqrt(
        np.vstack([np.arange(1, data.h + 1) for _ in range(y_base.shape[0])])
    )

    reconciler_args = dict(
        S=data.S,
        y_hat=y_hat_base,
        y_insample=y_base,
        y_hat_insample=y_hat_base_insample,
        sigmah=sigmah,
        intervals_method=intervals_method,
        tags=data.tags,
    )

    cls_bottom_up = BottomUp()
    cls_bottom_up.fit(**reconciler_args)
    coherent_samples = cls_bottom_up.sample(num_samples=100)

    # Store shape for comparison in a class variable
    if not hasattr(test_coherent_sample_shapes, 'shapes'):
        test_coherent_sample_shapes.shapes = []
    test_coherent_sample_shapes.shapes.append(coherent_samples.shape)

    # If we have multiple shapes, compare them
    if len(test_coherent_sample_shapes.shapes) > 1:
        assert test_coherent_sample_shapes.shapes[0] == test_coherent_sample_shapes.shapes[-1]


def test_mintrace_nonnegative_samples_use_constrained_forecasts(hierarchical_data):
    """Test that MinTrace nonnegative samples are based on constrained forecasts.

    This is a regression test for the bug where _get_sampler() was called with
    the original y_hat instead of the nonnegative-constrained self.y_hat,
    causing samples to potentially contain negative values even when
    nonnegative=True.
    """
    data = hierarchical_data

    # Create base forecasts with NEGATIVE values to trigger nonnegative constraint
    y_hat_base = data.S @ data.y_hat_bottom
    # Make some forecasts negative - use large negative values to make the bug obvious
    y_hat_with_negatives = y_hat_base.copy()
    y_hat_with_negatives[0, :] = -10.0  # Make total negative
    y_hat_with_negatives[1, :] = -5.0   # Make group 1 negative

    # Create in-sample data
    y_base = data.S @ data.y_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample

    # Compute sigmah - use small values to reduce sampling variance
    sigmah = np.ones((data.S.shape[0], data.h)) * 0.5

    # Use MinTrace with nonnegative=True and normality intervals
    # (bootstrap/permbu are not compatible with nonnegative)
    cls_min_trace = MinTrace(method="ols", nonnegative=True)

    result = cls_min_trace.fit_predict(
        S=data.S,
        y_hat=y_hat_with_negatives,
        y_insample=y_base,
        y_hat_insample=y_hat_base_insample,
        sigmah=sigmah,
        level=[80, 90],
        intervals_method="normality",
        num_samples=200,
        seed=42,
        tags=data.tags,
    )

    # Verify the mean reconciled forecasts are non-negative
    assert np.all(result["mean"] >= 0), "Mean forecasts should be non-negative"

    # KEY ASSERTION: Verify the sampler was initialized with the nonnegative-constrained y_hat
    # This is the core fix - previously the sampler was initialized with original y_hat
    assert cls_min_trace.sampler is not None, "Sampler should be initialized"
    np.testing.assert_array_equal(
        cls_min_trace.sampler.y_hat,
        cls_min_trace.y_hat,
        err_msg="Sampler y_hat should match the nonnegative-constrained y_hat"
    )

    # Verify sampler y_hat is non-negative (since we used nonnegative=True)
    assert np.all(cls_min_trace.sampler.y_hat >= -1e-6), (
        "Sampler y_hat should be non-negative when nonnegative=True"
    )

    # Generate samples - if the bug existed, samples would be centered around
    # the original negative values (-10, -5) instead of the constrained values
    samples = cls_min_trace.sample(num_samples=500)
    sample_means = np.mean(samples, axis=2)

    # With the bug fixed, sample means should be close to the non-negative reconciled mean.
    # Tolerance explanation: With sigmah=0.5 and 500 samples, the standard error of the mean
    # is approximately sigmah/sqrt(500) ≈ 0.022. We use atol=0.3 to account for the
    # coherent covariance structure which can increase variance.
    np.testing.assert_allclose(
        sample_means,
        result["mean"],
        atol=0.3,
        err_msg="Sample means should be close to non-negative reconciled forecasts"
    )


def test_mintrace_nonnegative_without_intervals(hierarchical_data):
    """Test MinTrace with nonnegative=True but without probabilistic intervals."""
    data = hierarchical_data

    y_hat_base = data.S @ data.y_hat_bottom
    y_hat_with_negatives = y_hat_base.copy()
    y_hat_with_negatives[0, :] = -10.0

    y_base = data.S @ data.y_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample

    cls_min_trace = MinTrace(method="ols", nonnegative=True)

    # Should work without intervals (level=None, intervals_method=None)
    result = cls_min_trace.fit_predict(
        S=data.S,
        y_hat=y_hat_with_negatives,
        y_insample=y_base,
        y_hat_insample=y_hat_base_insample,
        sigmah=None,
        level=None,
        intervals_method=None,
        num_samples=None,
        seed=None,
        tags=data.tags,
    )

    # Verify the mean reconciled forecasts are non-negative
    assert np.all(result["mean"] >= 0), "Mean forecasts should be non-negative"
    # Should not have quantiles since level=None
    assert "quantiles" not in result or result["quantiles"] is None


@pytest.mark.parametrize("method", ["ols", "wls_struct"])
def test_mintrace_sparse_nonnegative_sampler_initialization(hierarchical_data, method):
    """Test that MinTraceSparse with nonnegative=True correctly initializes sampler."""
    data = hierarchical_data

    # Convert S to sparse matrix as required by MinTraceSparse
    S_sparse = sparse.csr_matrix(data.S)

    y_hat_base = data.S @ data.y_hat_bottom
    y_hat_with_negatives = y_hat_base.copy()
    y_hat_with_negatives[0, :] = -10.0

    y_base = data.S @ data.y_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample
    sigmah = np.ones((data.S.shape[0], data.h)) * 0.5

    cls_min_trace = MinTraceSparse(method=method, nonnegative=True)

    result = cls_min_trace.fit_predict(
        S=S_sparse,
        y_hat=y_hat_with_negatives,
        y_insample=y_base,
        y_hat_insample=y_hat_base_insample,
        sigmah=sigmah,
        level=[80, 90],
        intervals_method="normality",
        num_samples=200,
        seed=42,
        tags=data.tags,
    )

    # Verify the mean reconciled forecasts are non-negative
    assert np.all(result["mean"] >= 0), "Mean forecasts should be non-negative"

    # Verify sampler uses the constrained y_hat
    assert cls_min_trace.sampler is not None
    np.testing.assert_array_equal(
        cls_min_trace.sampler.y_hat,
        cls_min_trace.y_hat,
        err_msg="MinTraceSparse sampler y_hat should match nonnegative-constrained y_hat"
    )


@pytest.mark.parametrize("intervals_method", ["bootstrap", "permbu"])
def test_mintrace_nonnegative_raises_on_bootstrap_permbu(hierarchical_data, intervals_method):
    """Test that MinTrace with nonnegative=True raises error for bootstrap/permbu.

    Nonnegative reconciliation is not compatible with bootstrap or permbu probabilistic
    forecasts because these methods generate samples from historical residuals which
    do not respect the nonnegative constraint applied during reconciliation.
    """
    data = hierarchical_data

    y_hat_base = data.S @ data.y_hat_bottom
    y_base = data.S @ data.y_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample
    sigmah = np.ones((data.S.shape[0], data.h)) * 0.5

    cls_min_trace = MinTrace(method="ols", nonnegative=True)

    with pytest.raises(ValueError, match="nonnegative reconciliation is not compatible"):
        cls_min_trace.fit_predict(
            S=data.S,
            y_hat=y_hat_base,
            y_insample=y_base,
            y_hat_insample=y_hat_base_insample,
            sigmah=sigmah,
            level=[90],
            intervals_method=intervals_method,
            num_samples=200,
            seed=42,
            tags=data.tags,
            )


@pytest.mark.parametrize("nonnegative", [False, True])
def test_emint_forecast_reconciliation(hierarchical_data, nonnegative):
    """Test EMinT method produces coherent forecasts."""
    data = hierarchical_data

    cls_emint = MinTrace(method='emint', nonnegative=nonnegative)
    result = cls_emint(
        S=data.S,
        y_hat=data.S @ data.y_hat_bottom,
        y_insample=data.S @ data.y_bottom,
        y_hat_insample=data.S @ data.y_hat_bottom_insample,
    )["mean"]

    # Check that result is coherent (satisfies aggregation constraints)
    # Bottom level forecasts should aggregate to upper levels
    bottom_forecasts = result[list(range(data.S.shape[0] - data.S.shape[1], data.S.shape[0])), :]
    aggregated = data.S @ bottom_forecasts

    np.testing.assert_allclose(result, aggregated, rtol=1e-10)

    # If nonnegative=True, verify all forecasts are non-negative
    if nonnegative:
        assert np.all(result >= -1e-6), "Forecasts should be non-negative when nonnegative=True"


def test_emint_fit_predict_pattern(hierarchical_data):
    """Test EMinT with fit -> predict pattern."""
    data = hierarchical_data

    cls_emint = MinTrace(method='emint')
    cls_emint.fit(
        S=data.S,
        y_hat=data.S @ data.y_hat_bottom,
        y_insample=data.S @ data.y_bottom,
        y_hat_insample=data.S @ data.y_hat_bottom_insample,
    )
    result = cls_emint.predict(S=data.S, y_hat=data.S @ data.y_hat_bottom)["mean"]

    # Check coherence
    bottom_forecasts = result[list(range(data.S.shape[0] - data.S.shape[1], data.S.shape[0])), :]
    aggregated = data.S @ bottom_forecasts

    np.testing.assert_allclose(result, aggregated, rtol=1e-10)


def test_emint_requires_insample_data(hierarchical_data):
    """Test EMinT raises error when insample data is missing."""
    data = hierarchical_data

    cls_emint = MinTrace(method='emint')

    with pytest.raises(ValueError, match="insample predictions and insample values"):
        cls_emint(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            )


@pytest.mark.parametrize("method_class,kwargs", [
    (MinTrace, {"method": "emint"}),
])
def test_insample_methods_coherence(hierarchical_data, method_class, kwargs):
    """Test that EMinT produces coherent forecasts."""
    data = hierarchical_data

    reconciler = method_class(**kwargs)
    result = reconciler(
        S=data.S,
        y_hat=data.S @ data.y_hat_bottom,
        y_insample=data.S @ data.y_bottom,
        y_hat_insample=data.S @ data.y_hat_bottom_insample,
    )["mean"]

    # Verify coherence: S @ bottom = all
    bottom_forecasts = result[list(range(data.S.shape[0] - data.S.shape[1], data.S.shape[0])), :]
    aggregated = data.S @ bottom_forecasts

    np.testing.assert_allclose(result, aggregated, rtol=1e-10)

    # Verify shape
    assert result.shape == (data.S.shape[0], data.h)


def test_emint_nonnegative_with_negative_forecasts(hierarchical_data):
    """Test EMinT with nonnegative=True handles negative base forecasts."""
    data = hierarchical_data

    # Create base forecasts with NEGATIVE values
    y_hat_base = data.S @ data.y_hat_bottom
    y_hat_with_negatives = y_hat_base.copy()
    y_hat_with_negatives[0, :] = -10.0  # Make total negative
    y_hat_with_negatives[1, :] = -5.0   # Make group 1 negative

    y_base = data.S @ data.y_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample

    cls_emint = MinTrace(method='emint', nonnegative=True)

    result = cls_emint.fit_predict(
        S=data.S,
        y_hat=y_hat_with_negatives,
        y_insample=y_base,
        y_hat_insample=y_hat_base_insample,
    )

    # Verify the mean reconciled forecasts are non-negative
    assert np.all(result["mean"] >= -1e-6), "Mean forecasts should be non-negative"

    # Verify coherence
    bottom_forecasts = result["mean"][list(range(data.S.shape[0] - data.S.shape[1], data.S.shape[0])), :]
    aggregated = data.S @ bottom_forecasts
    np.testing.assert_allclose(result["mean"], aggregated, rtol=1e-10)


def test_emint_nonnegative_sampler_initialization(hierarchical_data):
    """Test that EMinT with nonnegative=True correctly initializes sampler with constrained forecasts."""
    data = hierarchical_data

    y_hat_base = data.S @ data.y_hat_bottom
    y_hat_with_negatives = y_hat_base.copy()
    y_hat_with_negatives[0, :] = -10.0

    y_base = data.S @ data.y_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample
    sigmah = np.ones((data.S.shape[0], data.h)) * 0.5

    cls_emint = MinTrace(method='emint', nonnegative=True)

    result = cls_emint.fit_predict(
        S=data.S,
        y_hat=y_hat_with_negatives,
        y_insample=y_base,
        y_hat_insample=y_hat_base_insample,
        sigmah=sigmah,
        level=[80, 90],
        intervals_method="normality",
        num_samples=200,
        seed=42,
        tags=data.tags,
    )

    # Verify the mean reconciled forecasts are non-negative
    assert np.all(result["mean"] >= -1e-6), "Mean forecasts should be non-negative"

    # Verify the sampler was initialized with the nonnegative-constrained y_hat
    np.testing.assert_array_equal(
        cls_emint.sampler.y_hat,
        cls_emint.y_hat,
        err_msg="Sampler y_hat should match the nonnegative-constrained y_hat"
    )

    # Verify sampler y_hat is non-negative
    assert np.all(cls_emint.sampler.y_hat >= -1e-6), (
        "Sampler y_hat should be non-negative when nonnegative=True"
    )


@pytest.mark.parametrize("intervals_method", ["bootstrap", "permbu"])
def test_emint_nonnegative_raises_on_bootstrap_permbu(hierarchical_data, intervals_method):
    """Test that EMinT with nonnegative=True raises error for bootstrap/permbu."""
    data = hierarchical_data

    y_hat_base = data.S @ data.y_hat_bottom
    y_base = data.S @ data.y_bottom
    y_hat_base_insample = data.S @ data.y_hat_bottom_insample
    sigmah = np.ones((data.S.shape[0], data.h)) * 0.5

    cls_emint = MinTrace(method='emint', nonnegative=True)

    with pytest.raises(ValueError, match="nonnegative reconciliation is not compatible"):
        cls_emint.fit_predict(
            S=data.S,
            y_hat=y_hat_base,
            y_insample=y_base,
            y_hat_insample=y_hat_base_insample,
            sigmah=sigmah,
            level=[90],
            intervals_method=intervals_method,
            num_samples=200,
            seed=42,
            tags=data.tags,
            )
