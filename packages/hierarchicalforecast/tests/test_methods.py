from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pytest

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
    sparse,
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
    idx_bottom: List[int]  # Indices of bottom level series
    tags: Dict[str, np.ndarray]  # Hierarchy level tags

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

    idx_bottom = [3, 4, 5, 6]  # Indices of bottom level in S matrix

    tags = {
        "level1": np.array([0]),
        "level2": np.array([1, 2]),
        "level3": np.array(idx_bottom)
    }

    return HierarchicalTestData(
        S=S,
        h=h,
        y_bottom=y_bottom,
        y_hat_bottom=y_hat_bottom,
        y_hat_bottom_insample=y_hat_bottom_insample,
        idx_bottom=idx_bottom,
        tags=tags
    )

def test_sigmah_hierarchy(hierarchical_data):
    """Test bottom up forecast recovery matches expected output."""
    data = hierarchical_data
    cls_bottom_up = BottomUp()
    result = cls_bottom_up(S=data.S, y_hat=data.S @ data.y_hat_bottom, idx_bottom=data.idx_bottom)["mean"]
    expected = data.S @ data.y_hat_bottom
    assert result.all() == expected.all()


def test_btm_up_forecast_recovery(hierarchical_data):
    """Test bottom up forecast recovery."""
    data = hierarchical_data
    cls_bottom_up = BottomUp()
    bu_bootstrap_intervals = cls_bottom_up(
        S=data.S,
        y_hat=data.S @ data.y_hat_bottom,
        idx_bottom=data.idx_bottom,
    )
    expected = data.S @ data.y_hat_bottom
    assert bu_bootstrap_intervals["mean"].all() == expected.all()


def test_forecast_recovery_fit_predict(hierarchical_data):
    """Test forecast recovery with fit -> predict pattern."""
    data = hierarchical_data
    cls_bottom_up = BottomUp()
    cls_bottom_up.fit(S=data.S, y_hat=data.S @ data.y_hat_bottom, idx_bottom=data.idx_bottom)
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
    with pytest.raises(Exception, match="Top-down reconciliation requires strictly hierarchical structures"):
        cls_top_down(sparse.csr_matrix(S_non_hier), None, tags_non_hier)


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


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink"])
@pytest.mark.parametrize("nonnegative", [False, True])
def test_min_trace_forecast_recovery(hierarchical_data, method, nonnegative):
    """Test MinTrace methods can recover forecasts and nonnegative behavior."""
    data = hierarchical_data

    cls_min_trace = MinTrace(method=method, nonnegative=nonnegative)
    assert cls_min_trace.nonnegative is nonnegative

    if cls_min_trace.insample:
        assert method in ["wls_var", "mint_cov", "mint_shrink"]
        result = cls_min_trace(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            y_hat_insample=data.S @ data.y_hat_bottom_insample,
            idx_bottom=data.idx_bottom if nonnegative else None,
        )["mean"]
    else:
        result = cls_min_trace(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            idx_bottom=data.idx_bottom if nonnegative else None,
        )["mean"]

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
            idx_bottom=data.idx_bottom if nonnegative else None,
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
        idx_bottom=data.idx_bottom,
    )

    # Test passes if no exception is raised
    assert result_min_trace is not None


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink"])
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
        idx_bottom=data.idx_bottom if nonnegative else None,
    )["mean"]

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
        idx_bottom=data.idx_bottom,
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
            idx_bottom=hierarchical_data.idx_bottom if nonnegative else None,
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
        idx_bottom=data.idx_bottom,
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
        idx_bottom=data.idx_bottom,
    )


def test_bottom_up_normality_intervals(hierarchical_data, interval_reconciler_args):
    """Test bottom up normality prediction intervals recover original sigmah."""
    data = hierarchical_data

    cls_bottom_up = BottomUp()
    result = cls_bottom_up(**interval_reconciler_args)

    assert np.array_equal(result["sigmah"][data.idx_bottom], interval_reconciler_args["sigmah"][data.idx_bottom])


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


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink"])
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
        idx_bottom=data.idx_bottom,
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
