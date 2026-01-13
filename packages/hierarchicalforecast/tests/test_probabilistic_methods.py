import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from hierarchicalforecast.evaluation import (
    energy_score,
    log_score,
    msse,
    rel_mse,
    scaled_crps,
)
from hierarchicalforecast.methods import BottomUp
from hierarchicalforecast.probabilistic_methods import (
    PERMBU,
    Bootstrap,
    CovarianceType,
    Normality,
)


@pytest.fixture
def test_data():
    """Fixture to provide test data for hierarchical forecasting tests."""
    S = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    h = 2
    _y = np.array([10.0, 5.0, 4.0, 2.0, 1.0])
    y_bottom = np.vstack([i * _y for i in range(1, 5)])
    y_hat_bottom_insample = np.roll(y_bottom, 1)
    y_hat_bottom_insample[:, 0] = np.nan
    y_hat_bottom = np.vstack([i * np.ones(h) for i in range(1, 5)])
    tags = {"level1": np.array([0]), "level2": np.array([1, 2]), "level3": np.array([3, 4, 5, 6])}

    y_base = S @ y_bottom
    y_hat_base = S @ y_hat_bottom
    y_hat_base_insample = S @ y_hat_bottom_insample
    sigma = np.nansum((y_base - y_hat_base_insample) ** 2, axis=1) / (
        y_base.shape[1] - 1
    )
    sigma = np.sqrt(sigma)
    sigmah = sigma[:, None] * np.sqrt(
        np.vstack([np.arange(1, h + 1) for _ in range(y_base.shape[0])])
    )
    noise = np.random.normal(scale=sigmah)
    y_test = y_hat_base + noise

    return {
        "S": S,
        "h": h,
        "y_bottom": y_bottom,
        "y_hat_bottom": y_hat_bottom,
        "y_hat_bottom_insample": y_hat_bottom_insample,
        "tags": tags,
        "y_base": y_base,
        "y_hat_base": y_hat_base,
        "y_hat_base_insample": y_hat_base_insample,
        "sigmah": sigmah,
        "y_test": y_test,
    }


@pytest.fixture
def samplers(test_data):
    """Fixture to provide samplers for testing."""
    cls_bottom_up = BottomUp()
    P, W = cls_bottom_up._get_PW_matrices(
        S=test_data["S"]    )

    normality_sampler = Normality(
        S=test_data["S"],
        P=P,
        W=W,
        y_hat=test_data["y_hat_base"],
        sigmah=test_data["sigmah"],
    )
    bootstrap_sampler = Bootstrap(
        S=test_data["S"],
        P=P,
        W=W,
        y_hat=test_data["y_hat_base"],
        y_insample=test_data["y_base"],
        y_hat_insample=test_data["y_hat_base_insample"],
        num_samples=1_000,
    )
    empty_bootstrap_sampler = Bootstrap(
        S=test_data["S"],
        P=P,
        W=W,
        y_hat=test_data["y_hat_base"],
        y_insample=test_data["y_base"],
        y_hat_insample=test_data["y_base"],
        num_samples=1_000,
    )
    permbu_sampler = PERMBU(
        S=test_data["S"],
        P=P,
        tags=test_data["tags"],
        y_hat=test_data["y_hat_base"],
        y_insample=test_data["y_base"],
        y_hat_insample=test_data["y_hat_base_insample"],
        sigmah=test_data["sigmah"],
    )
    empty_permbu_sampler = PERMBU(
        S=test_data["S"],
        P=P,
        tags=test_data["tags"],
        y_hat=test_data["y_hat_base"],
        y_insample=test_data["y_base"],
        y_hat_insample=test_data["y_base"],
        sigmah=test_data["sigmah"],
    )

    return {
        "normality_sampler": normality_sampler,
        "bootstrap_sampler": bootstrap_sampler,
        "empty_bootstrap_sampler": empty_bootstrap_sampler,
        "permbu_sampler": permbu_sampler,
        "empty_permbu_sampler": empty_permbu_sampler,
    }


# Tests for Normality covariance_type parameter (Issue #99)
class TestNormalityCovarianceType:
    """Tests for Normality covariance_type parameter."""

    # Basic functionality tests

    def test_normality_diagonal_covariance(self, test_data):
        """Test Normality with diagonal covariance (default, backward compat)."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        # Default (diagonal) should work without residuals
        normality_diag = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type="diagonal",
        )
        samples = normality_diag.get_samples(num_samples=50)
        assert samples.shape == (test_data["S"].shape[0], test_data["h"], 50)

    def test_normality_full_covariance(self, test_data):
        """Test Normality with full empirical covariance."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        # Compute residuals from insample data
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]

        normality_full = Normality(
            S=test_data["S"],
            P=P,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type="full",
            residuals=residuals,
        )
        samples = normality_full.get_samples(num_samples=50)
        assert samples.shape == (test_data["S"].shape[0], test_data["h"], 50)

    def test_normality_shrink_covariance(self, test_data):
        """Test Normality with shrinkage covariance (Schäfer-Strimmer)."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        # Compute residuals from insample data
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]

        normality_shrink = Normality(
            S=test_data["S"],
            P=P,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type="shrink",
            residuals=residuals,
            shrinkage_ridge=2e-8,
        )
        samples = normality_shrink.get_samples(num_samples=50)
        assert samples.shape == (test_data["S"].shape[0], test_data["h"], 50)

    def test_normality_w_optional_for_full(self, test_data):
        """Test that W is optional when covariance_type is 'full'."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]

        # Should work without W
        normality = Normality(
            S=test_data["S"],
            P=P,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type="full",
            residuals=residuals,
        )
        samples = normality.get_samples(num_samples=50)
        assert samples.shape == (test_data["S"].shape[0], test_data["h"], 50)

    def test_normality_w_optional_for_shrink(self, test_data):
        """Test that W is optional when covariance_type is 'shrink'."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]

        # Should work without W
        normality = Normality(
            S=test_data["S"],
            P=P,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type="shrink",
            residuals=residuals,
        )
        samples = normality.get_samples(num_samples=50)
        assert samples.shape == (test_data["S"].shape[0], test_data["h"], 50)

    # Enum and case sensitivity tests

    def test_normality_covariance_type_enum(self, test_data):
        """Test that CovarianceType enum works."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]

        # Test with enum
        normality = Normality(
            S=test_data["S"],
            P=P,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type=CovarianceType.SHRINK,
            residuals=residuals,
        )
        samples = normality.get_samples(num_samples=50)
        assert samples.shape == (test_data["S"].shape[0], test_data["h"], 50)
        assert normality.covariance_type == CovarianceType.SHRINK

    def test_normality_case_insensitive_covariance_type(self, test_data):
        """Test that covariance_type is case insensitive."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]

        # Test various cases
        for cov_type in ["FULL", "Full", "fUlL"]:
            normality = Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type=cov_type,
                residuals=residuals,
            )
            assert normality.covariance_type == CovarianceType.FULL

    # Validation error tests

    def test_normality_invalid_covariance_type(self, test_data):
        """Test that invalid covariance_type raises ValueError."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                W=W,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="invalid_type",
            )
        assert "Unknown covariance_type" in str(exc_info.value)

    def test_normality_invalid_type_for_covariance_type(self, test_data):
        """Test that non-string/non-enum covariance_type raises ValueError."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                W=W,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type=123,  # Invalid type
            )
        assert "must be a string or CovarianceType enum" in str(exc_info.value)

    def test_normality_full_requires_residuals(self, test_data):
        """Test that full covariance requires residuals."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                W=W,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
            )
        assert "requires `residuals` parameter" in str(exc_info.value)

    def test_normality_shrink_requires_residuals(self, test_data):
        """Test that shrink covariance requires residuals."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                W=W,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="shrink",
            )
        assert "requires `residuals` parameter" in str(exc_info.value)

    def test_normality_diagonal_requires_w(self, test_data):
        """Test that diagonal covariance requires W."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="diagonal",
            )
        assert "covariance_type='diagonal' requires `W` parameter" in str(
            exc_info.value
        )

    # Residuals shape validation tests

    def test_normality_residuals_wrong_shape_1d(self, test_data):
        """Test that 1D residuals raises ValueError."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals_1d = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals_1d,
            )
        assert "must be a 2D array" in str(exc_info.value)

    def test_normality_residuals_wrong_n_series(self, test_data):
        """Test that residuals with wrong number of series raises ValueError."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        residuals_wrong = np.random.randn(5, 10)

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals_wrong,
            )
        assert "residuals shape mismatch" in str(exc_info.value)

    def test_normality_residuals_empty(self, test_data):
        """Test that empty residuals raises ValueError."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals_empty = np.empty((7, 0))

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals_empty,
            )
        assert "residuals is empty" in str(exc_info.value)

    def test_normality_residuals_single_observation(self, test_data):
        """Test that residuals with single observation raises ValueError."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals_single = np.random.randn(7, 1)

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals_single,
            )
        assert "At least 2 observations are required" in str(exc_info.value)

    # NaN handling tests

    def test_normality_residuals_all_nan_series(self, test_data):
        """Test that all-NaN series in residuals raises ValueError."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]
        # Make first series all NaN
        residuals[0, :] = np.nan

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals,
            )
        assert "all NaN values" in str(exc_info.value)

    def test_normality_residuals_insufficient_non_nan(self, test_data):
        """Test that series with <2 non-NaN observations raises ValueError."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]
        # Make first series have only 1 non-NaN value
        residuals[0, :] = np.nan
        residuals[0, 0] = 1.0

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals,
            )
        assert "fewer than 2 non-NaN observations" in str(exc_info.value)

    def test_normality_residuals_with_some_nans(self, test_data):
        """Test that residuals with some NaNs work correctly."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]
        # residuals already has NaN from y_hat_base_insample

        # Should work for both full and shrink
        for cov_type in ["full", "shrink"]:
            normality = Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type=cov_type,
                residuals=residuals,
            )
            samples = normality.get_samples(num_samples=50)
            assert samples.shape == (test_data["S"].shape[0], test_data["h"], 50)

    # Warning tests

    def test_normality_warns_w_ignored(self, test_data):
        """Test that warning is issued when W is provided but ignored."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]

        with pytest.warns(UserWarning, match="W parameter is ignored"):
            Normality(
                S=test_data["S"],
                P=P,
                W=W,  # Providing W when using 'full'
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals,
            )

    def test_normality_warns_shrinkage_ridge_ignored(self, test_data):
        """Test that warning is issued when shrinkage_ridge is provided but not used."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        with pytest.warns(UserWarning, match="shrinkage_ridge parameter is only used"):
            Normality(
                S=test_data["S"],
                P=P,
                W=W,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="diagonal",
                shrinkage_ridge=1e-6,  # Non-default value
            )

    def test_normality_warns_n_series_gt_n_obs(self, test_data):
        """Test warning when n_series > n_observations for full covariance."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        # Create residuals with fewer observations than series
        n_series = test_data["S"].shape[0]  # 7 series
        residuals = np.random.randn(n_series, 3)  # Only 3 observations

        with pytest.warns(UserWarning, match="non-positive-definite"):
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals,
            )

    # Zero/near-zero variance tests

    def test_normality_warns_zero_variance(self, test_data):
        """Test warning when series has zero variance."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        n_series = test_data["S"].shape[0]
        residuals = np.random.randn(n_series, 10)
        # Make first series constant (zero variance)
        residuals[0, :] = 5.0

        with pytest.warns(UserWarning, match="zero or near-zero variance"):
            Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals,
            )

    def test_normality_handles_zero_variance_gracefully(self, test_data):
        """Test that zero variance series doesn't crash and produces valid samples."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        n_series = test_data["S"].shape[0]
        residuals = np.random.randn(n_series, 10)
        residuals[0, :] = 5.0  # Constant series

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            normality = Normality(
                S=test_data["S"],
                P=P,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="full",
                residuals=residuals,
            )
            samples = normality.get_samples(num_samples=50)

        assert samples.shape == (n_series, test_data["h"], 50)
        assert np.all(np.isfinite(samples))

    # W matrix validation tests

    def test_normality_w_nan_diagonal(self, test_data):
        """Test that W with NaN diagonal raises ValueError."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        W_with_nan = W.copy()
        W_with_nan[0, 0] = np.nan

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                W=W_with_nan,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="diagonal",
            )
        assert "NaN values" in str(exc_info.value)

    def test_normality_w_non_positive_diagonal(self, test_data):
        """Test that W with non-positive diagonal raises ValueError."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        W_with_zero = W.copy()
        W_with_zero[0, 0] = 0.0

        with pytest.raises(ValueError) as exc_info:
            Normality(
                S=test_data["S"],
                P=P,
                W=W_with_zero,
                y_hat=test_data["y_hat_base"],
                sigmah=test_data["sigmah"],
                covariance_type="diagonal",
            )
        assert "non-positive diagonal" in str(exc_info.value)

    # n_series > n_observations tests (non-PSD risk)

    def test_normality_shrink_handles_high_dimensional_data(self, test_data):
        """Test that shrink covariance handles n_series > n_observations well."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        n_series = test_data["S"].shape[0]
        # Create high-dimensional case: more series than observations
        residuals = np.random.randn(n_series, 3)

        # Shrink should work without warning about non-PSD
        normality = Normality(
            S=test_data["S"],
            P=P,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type="shrink",
            residuals=residuals,
        )
        samples = normality.get_samples(num_samples=50)
        assert samples.shape == (n_series, test_data["h"], 50)
        assert np.all(np.isfinite(samples))

    # Statistical validation tests

    def test_normality_samples_have_correct_mean(self, test_data):
        """Test that samples have approximately correct mean."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        normality = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type="diagonal",
            seed=42,
        )

        # Generate many samples for statistical test
        num_samples = 50000
        samples = normality.get_samples(num_samples=num_samples)
        sample_means = samples.mean(axis=2)

        # Expected mean is SP @ y_hat
        expected_means = normality.SP @ test_data["y_hat_base"]

        # Check that sample means are close to expected means
        # Using relative tolerance due to different scales
        np.testing.assert_allclose(sample_means, expected_means, rtol=0.05)

    def test_normality_samples_have_correct_variance(self, test_data):
        """Test that samples have approximately correct variance."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        normality = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            covariance_type="diagonal",
            seed=42,
        )

        # Generate many samples for statistical test
        num_samples = 10000
        samples = normality.get_samples(num_samples=num_samples)
        sample_stds = samples.std(axis=2)

        # Expected standard deviation
        expected_stds = normality.sigmah_rec

        # Check that sample stds are close to expected stds
        np.testing.assert_allclose(sample_stds, expected_stds, rtol=0.10)

    def test_normality_correlation_matrix_is_valid(self, test_data):
        """Test that computed correlation matrix has valid properties."""
        cls_bottom_up = BottomUp()
        P, _ = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )
        residuals = test_data["y_base"] - test_data["y_hat_base_insample"]

        for cov_type in ["full", "shrink"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normality = Normality(
                    S=test_data["S"],
                    P=P,
                    y_hat=test_data["y_hat_base"],
                    sigmah=test_data["sigmah"],
                    covariance_type=cov_type,
                    residuals=residuals,
                )

            corr = normality._correlation_matrix

            # Check diagonal is 1
            np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

            # Check symmetry
            np.testing.assert_allclose(corr, corr.T, atol=1e-10)

            # Check values in [-1, 1]
            assert np.all(corr >= -1.0)
            assert np.all(corr <= 1.0)

    def test_normality_reproducibility_with_seed(self, test_data):
        """Test that setting seed produces reproducible samples."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        normality1 = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            seed=42,
        )
        samples1 = normality1.get_samples(num_samples=100)

        normality2 = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            seed=42,
        )
        samples2 = normality2.get_samples(num_samples=100)

        np.testing.assert_array_equal(samples1, samples2)

    def test_normality_different_seeds_different_samples(self, test_data):
        """Test that different seeds produce different samples."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        normality1 = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            seed=42,
        )
        samples1 = normality1.get_samples(num_samples=100)

        normality2 = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
            seed=123,
        )
        samples2 = normality2.get_samples(num_samples=100)

        assert not np.allclose(samples1, samples2)

    # get_prediction_levels and get_prediction_quantiles tests

    def test_normality_get_prediction_levels(self, test_data):
        """Test get_prediction_levels method."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        normality = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
        )

        res = {"mean": normality.SP @ test_data["y_hat_base"]}
        levels = [80, 95]
        res = normality.get_prediction_levels(res, levels)

        # Check that required keys are present
        assert "sigmah" in res
        assert "lo-80" in res
        assert "hi-80" in res
        assert "lo-95" in res
        assert "hi-95" in res

        # Check shapes
        n_series, n_horizon = test_data["y_hat_base"].shape
        assert res["sigmah"].shape == (test_data["S"].shape[0], n_horizon)
        assert res["lo-80"].shape == (test_data["S"].shape[0], n_horizon)
        assert res["hi-80"].shape == (test_data["S"].shape[0], n_horizon)

        # Check that intervals are symmetric around mean
        assert np.allclose(res["hi-80"] - res["mean"], res["mean"] - res["lo-80"])

    def test_normality_get_prediction_quantiles(self, test_data):
        """Test get_prediction_quantiles method."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        normality = Normality(
            S=test_data["S"],
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
        )

        res = {"mean": normality.SP @ test_data["y_hat_base"]}
        quantiles = np.array([0.1, 0.5, 0.9])
        res = normality.get_prediction_quantiles(res, quantiles)

        # Check that required keys are present
        assert "sigmah" in res
        assert "quantiles" in res

        # Check shapes
        n_series = test_data["S"].shape[0]
        n_horizon = test_data["y_hat_base"].shape[1]
        assert res["quantiles"].shape == (n_series, n_horizon, len(quantiles))

        # Check that median (0.5) is roughly equal to mean
        np.testing.assert_allclose(res["quantiles"][:, :, 1], res["mean"], rtol=0.01)

    def test_normality_sparse_matrix_handling(self, test_data):
        """Test that sparse S matrix is handled correctly."""
        cls_bottom_up = BottomUp()
        P, W = cls_bottom_up._get_PW_matrices(
            S=test_data["S"]        )

        # Convert S to sparse matrix
        S_sparse = sp.csr_matrix(test_data["S"])

        normality = Normality(
            S=S_sparse,
            P=P,
            W=W,
            y_hat=test_data["y_hat_base"],
            sigmah=test_data["sigmah"],
        )

        samples = normality.get_samples(num_samples=50)
        assert samples.shape == (test_data["S"].shape[0], test_data["h"], 50)


def test_coherent_samples_shape(samplers):
    """Test that coherent samples have the correct shape."""
    normality_samples = samplers["normality_sampler"].get_samples(num_samples=100)
    bootstrap_samples = samplers["bootstrap_sampler"].get_samples(num_samples=100)
    permbu_samples = samplers["permbu_sampler"].get_samples(num_samples=100)

    assert bootstrap_samples.shape == normality_samples.shape
    assert bootstrap_samples.shape == permbu_samples.shape


def test_rel_mse_execution(test_data):
    """Test RelMSE's execution."""
    result = rel_mse(
        y=test_data["y_test"],
        y_hat=test_data["y_hat_base"],
        y_train=test_data["y_base"],
    )
    assert result is not None


def test_msse_execution(test_data):
    """Test MSSE's execution."""
    result = msse(
        y=test_data["y_test"],
        y_hat=test_data["y_hat_base"],
        y_train=test_data["y_base"],
    )
    assert result is not None


def test_energy_score_execution(test_data, samplers):
    """Test energy score's execution."""
    bootstrap_samples = samplers["bootstrap_sampler"].get_samples(num_samples=100)
    permbu_samples = samplers["permbu_sampler"].get_samples(num_samples=100)

    result = energy_score(
        y=test_data["y_test"], y_sample1=bootstrap_samples, y_sample2=permbu_samples
    )
    assert result is not None


def test_scaled_crps_execution(test_data, samplers):
    """Test scaled CRPS' execution."""
    bootstrap_samples = samplers["bootstrap_sampler"].get_samples(num_samples=100)
    quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    bootstrap_quantiles = np.quantile(bootstrap_samples, q=quantiles, axis=2)
    bootstrap_quantiles = bootstrap_quantiles.transpose((1, 2, 0))  # [Q,N,H] -> [N,H,Q]

    result = scaled_crps(
        y=test_data["y_test"], y_hat=bootstrap_quantiles, quantiles=quantiles
    )
    assert result is not None


def test_log_score_execution(test_data, samplers):
    """Test log score's execution."""
    cov = np.concatenate(
        [cov[:, :, None] for cov in samplers["normality_sampler"].cov_rec], axis=2
    )
    result = log_score(
        y=test_data["y_test"],
        y_hat=test_data["y_hat_base"],
        cov=cov,
        allow_singular=True,
    )
    assert result is not None


def test_quantile_loss_protections(test_data, samplers):
    """Test quantile loss protections."""
    bootstrap_samples = samplers["bootstrap_sampler"].get_samples(num_samples=100)
    quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    bootstrap_quantiles = np.quantile(bootstrap_samples, q=quantiles, axis=2)
    bootstrap_quantiles = bootstrap_quantiles.transpose((1, 2, 0))  # [Q,N,H] -> [N,H,Q]

    # Test with invalid quantiles (>1.0)
    invalid_quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2])

    with pytest.raises(Exception) as exc_info:
        scaled_crps(test_data["y_test"], bootstrap_quantiles, invalid_quantiles)

    assert "between 0 and 1" in str(exc_info.value)
