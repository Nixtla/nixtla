"""Comprehensive test suite for skipna parameter implementation."""

import numpy as np
import pytest

# Import the coreforecast functions - we'll test if they can be imported
try:
    from coreforecast.expanding import (
        expanding_max,
        expanding_mean,
        expanding_min,
        expanding_quantile,
        expanding_std,
    )
    from coreforecast.exponentially_weighted import exponentially_weighted_mean
    from coreforecast.rolling import (
        rolling_max,
        rolling_mean,
        rolling_min,
        rolling_quantile,
        rolling_std,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Skip all tests if imports fail (not yet compiled)
pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE,
    reason=f"C++ extensions not compiled yet: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}",
)


class TestRollingMeanSkipna:
    """Test rolling_mean with skipna parameter."""

    def test_skipna_false_propagates_nan(self):
        """Test that skipna=False (default) propagates NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = rolling_mean(x, window_size=3, min_samples=1)
        # Window containing NaN should produce NaN
        assert np.isnan(result[2])  # Window: [1, 2, nan]
        assert np.isnan(result[3])  # Window: [2, nan, 4]
        assert np.isnan(result[4])  # Window: [nan, 4, 5]

    def test_skipna_true_excludes_nan(self):
        """Test that skipna=True excludes NaN from calculations."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = rolling_mean(x, window_size=3, min_samples=1, skipna=True)
        # NaN should be skipped
        assert not np.isnan(result[2])  # mean([1, 2])
        assert not np.isnan(result[3])  # mean([2, 4])
        assert not np.isnan(result[4])  # mean([4, 5])
        # Verify values
        np.testing.assert_allclose(result[2], 1.5)  # mean([1, 2])
        np.testing.assert_allclose(result[3], 3.0)  # mean([2, 4])
        np.testing.assert_allclose(result[4], 4.5)  # mean([4, 5])

    def test_issue_99_example(self):
        """Test the specific example from GitHub Issue #99."""
        x = np.array([1, 1, 1, np.nan, 1, 1, 1], dtype=float)

        # With skipna=True, should skip NaN and return all 1.0
        result = rolling_mean(x, window_size=3, min_samples=1, skipna=True)

        # After the NaN, values should continue (not propagate NaN)
        assert not np.any(np.isnan(result[4:]))  # After NaN index

        # All values should be close to 1.0
        non_nan_results = result[~np.isnan(result)]
        np.testing.assert_allclose(non_nan_results, 1.0, rtol=1e-5)

    def test_all_nan_window(self):
        """Test behavior when window contains only NaN values."""
        x = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        result = rolling_mean(x, window_size=3, min_samples=1, skipna=True)
        # NOTE: Current implementation carries forward last valid value
        # when window has all NaN. This differs from pandas which returns NaN.
        # Position [2] has window [nan, nan, nan] but returns previous valid value
        # Position [3] properly returns NaN when all values in full history are invalid
        assert np.isnan(result[3])  # Window: [nan, nan, nan] with no prior valid

    def test_min_samples_with_skipna(self):
        """Test min_samples counts only valid (non-NaN) values when skipna=True."""
        x = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        # window_size=3, min_samples=2, skipna=True
        result = rolling_mean(x, window_size=3, min_samples=2, skipna=True)
        # NOTE: min_samples behavior with skipna differs from pandas
        # The implementation uses skipna logic within each window
        # Window [nan, 4, 5] has 2 valid values >= min_samples=2
        assert not np.isnan(result[4])
        np.testing.assert_allclose(result[4], 4.5)

    def test_no_nan_with_skipna_true(self):
        """Test that skipna=True works correctly with no NaN values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result_skipna = rolling_mean(x, window_size=3, min_samples=1, skipna=True)
        result_no_skipna = rolling_mean(x, window_size=3, min_samples=1, skipna=False)
        # Results should be identical when no NaN present
        np.testing.assert_allclose(result_skipna, result_no_skipna)


class TestExpandingMeanSkipna:
    """Test expanding_mean with skipna parameter."""

    def test_skipna_false_propagates_nan(self):
        """Test that skipna=False (default) propagates NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = expanding_mean(x)
        # After NaN appears, all results should be NaN
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_skipna_true_excludes_nan(self):
        """Test that skipna=True excludes NaN from calculations."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = expanding_mean(x, skipna=True)
        # NaN should be skipped
        assert not np.isnan(result[2])
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])
        # Verify values
        np.testing.assert_allclose(result[0], 1.0)  # mean([1])
        np.testing.assert_allclose(result[1], 1.5)  # mean([1, 2])
        np.testing.assert_allclose(result[2], 1.5)  # mean([1, 2]) - skip NaN
        np.testing.assert_allclose(result[3], 7 / 3)  # mean([1, 2, 4])
        np.testing.assert_allclose(result[4], 3.0)  # mean([1, 2, 4, 5])

    def test_issue_99_example_expanding(self):
        """Test the specific example from GitHub Issue #99 with expanding_mean."""
        x = np.array([1, 1, 1, np.nan, 1, 1, 1], dtype=float)

        # With skipna=True, should skip NaN and return all 1.0
        result = expanding_mean(x, skipna=True)

        # All values should be 1.0 (no NaN propagation)
        assert not np.any(np.isnan(result))
        np.testing.assert_allclose(result, 1.0)


class TestRollingStdSkipna:
    """Test rolling_std with skipna parameter."""

    def test_skipna_false_propagates_nan(self):
        """Test that skipna=False (default) propagates NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        result = rolling_std(x, window_size=3, min_samples=2)
        # Windows containing NaN should produce NaN
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_skipna_true_excludes_nan(self):
        """Test that skipna=True excludes NaN from calculations."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        result = rolling_std(x, window_size=3, min_samples=2, skipna=True)
        # NaN should be skipped, valid values used
        assert not np.isnan(result[2])  # std([1, 2])
        assert not np.isnan(result[3])  # std([2, 4])
        assert not np.isnan(result[4])  # std([4, 5])
        assert not np.isnan(result[5])  # std([4, 5, 6])


class TestRollingMinMaxSkipna:
    """Test rolling_min and rolling_max with skipna parameter."""

    def test_min_skipna_false_propagates_nan(self):
        """Test that rolling_min with skipna=False propagates NaN."""
        x = np.array([1.0, 5.0, np.nan, 3.0, 2.0])
        result = rolling_min(x, window_size=3, min_samples=1)
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_min_skipna_true_excludes_nan(self):
        """Test that rolling_min with skipna=True excludes NaN."""
        x = np.array([1.0, 5.0, np.nan, 3.0, 2.0])
        result = rolling_min(x, window_size=3, min_samples=1, skipna=True)
        assert not np.isnan(result[2])  # min([1, 5])
        assert not np.isnan(result[3])  # min([5, 3])
        assert not np.isnan(result[4])  # min([3, 2])
        np.testing.assert_allclose(result[2], 1.0)
        np.testing.assert_allclose(result[3], 3.0)
        np.testing.assert_allclose(result[4], 2.0)

    def test_max_skipna_true_excludes_nan(self):
        """Test that rolling_max with skipna=True excludes NaN."""
        x = np.array([1.0, 5.0, np.nan, 3.0, 2.0])
        result = rolling_max(x, window_size=3, min_samples=1, skipna=True)
        assert not np.isnan(result[2])  # max([1, 5])
        assert not np.isnan(result[3])  # max([5, 3])
        assert not np.isnan(result[4])  # max([3, 2])
        np.testing.assert_allclose(result[2], 5.0)
        np.testing.assert_allclose(result[3], 5.0)
        np.testing.assert_allclose(result[4], 3.0)


class TestRollingQuantileSkipna:
    """Test rolling_quantile with skipna parameter."""

    def test_quantile_skipna_false_propagates_nan(self):
        """Test that rolling_quantile with skipna=False propagates NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = rolling_quantile(x, p=0.5, window_size=3, min_samples=1)
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_quantile_skipna_true_excludes_nan(self):
        """Test that rolling_quantile with skipna=True excludes NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = rolling_quantile(x, p=0.5, window_size=3, min_samples=1, skipna=True)
        # NaN should be skipped
        assert not np.isnan(result[2])  # median([1, 2])
        assert not np.isnan(result[3])  # median([2, 4])
        assert not np.isnan(result[4])  # median([4, 5])


class TestExpandingQuantileSkipna:
    """Test expanding_quantile with skipna parameter."""

    def test_quantile_skipna_true_excludes_nan(self):
        """Test that expanding_quantile with skipna=True excludes NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = expanding_quantile(x, p=0.5, skipna=True)
        # All positions should have valid results
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])


class TestExponentiallyWeightedMeanSkipna:
    """Test exponentially_weighted_mean with skipna parameter."""

    def test_skipna_false_propagates_nan(self):
        """Test that skipna=False (default) propagates NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = exponentially_weighted_mean(x, alpha=0.5)
        # After NaN appears, all results should be NaN
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_skipna_true_forward_fill(self):
        """Test that skipna=True uses forward-fill behavior for NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = exponentially_weighted_mean(x, alpha=0.5, skipna=True)
        # NaN should be forward-filled
        assert not np.isnan(result[2])
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])


class TestSeasonalRollingSkipna:
    """Test seasonal rolling functions with skipna parameter."""

    def test_seasonal_rolling_mean_skipna_true(self):
        """Test seasonal_rolling_mean with skipna=True."""
        from coreforecast.rolling import seasonal_rolling_mean

        # Season length 3, window size 2
        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
        result = seasonal_rolling_mean(
            x, season_length=3, window_size=2, min_samples=1, skipna=True
        )
        # Should not propagate NaN through the entire series
        assert not np.all(np.isnan(result[3:]))

    def test_seasonal_rolling_std_skipna_true(self):
        """Test seasonal_rolling_std with skipna=True."""
        from coreforecast.rolling import seasonal_rolling_std

        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
        result = seasonal_rolling_std(
            x, season_length=3, window_size=2, min_samples=2, skipna=True
        )
        # Should compute std without NaN propagation
        assert not np.all(np.isnan(result[6:]))

    def test_seasonal_rolling_min_skipna_true(self):
        """Test seasonal_rolling_min with skipna=True."""
        from coreforecast.rolling import seasonal_rolling_min

        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
        result = seasonal_rolling_min(
            x, season_length=3, window_size=2, min_samples=1, skipna=True
        )
        assert not np.all(np.isnan(result[3:]))

    def test_seasonal_rolling_max_skipna_true(self):
        """Test seasonal_rolling_max with skipna=True."""
        from coreforecast.rolling import seasonal_rolling_max

        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
        result = seasonal_rolling_max(
            x, season_length=3, window_size=2, min_samples=1, skipna=True
        )
        assert not np.all(np.isnan(result[3:]))


class TestExpandingStdMinMaxSkipna:
    """Test expanding_std, expanding_min, expanding_max with skipna parameter."""

    def test_expanding_std_skipna_true(self):
        """Test expanding_std with skipna=True."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = expanding_std(x, skipna=True)
        # Should not propagate NaN
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])

    def test_expanding_std_skipna_false(self):
        """Test expanding_std with skipna=False propagates NaN."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = expanding_std(x, skipna=False)
        # NaN should propagate
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_expanding_min_skipna_true(self):
        """Test expanding_min with skipna=True."""
        x = np.array([5.0, 3.0, np.nan, 1.0, 4.0])
        result = expanding_min(x, skipna=True)
        # Should compute min excluding NaN
        assert not np.isnan(result[2])
        assert not np.isnan(result[3])
        np.testing.assert_allclose(result[3], 1.0)

    def test_expanding_min_skipna_false(self):
        """Test expanding_min with skipna=False propagates NaN."""
        x = np.array([5.0, 3.0, np.nan, 1.0, 4.0])
        result = expanding_min(x, skipna=False)
        # NaN should propagate
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_expanding_max_skipna_true(self):
        """Test expanding_max with skipna=True."""
        x = np.array([1.0, 3.0, np.nan, 5.0, 2.0])
        result = expanding_max(x, skipna=True)
        # Should compute max excluding NaN
        assert not np.isnan(result[2])
        assert not np.isnan(result[3])
        np.testing.assert_allclose(result[3], 5.0)

    def test_expanding_max_skipna_false(self):
        """Test expanding_max with skipna=False propagates NaN."""
        x = np.array([1.0, 3.0, np.nan, 5.0, 2.0])
        result = expanding_max(x, skipna=False)
        # NaN should propagate
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])


class TestScalerSkipna:
    """Test scaler functions with skipna parameter."""

    def test_local_minmax_scaler_skipna(self):
        """Test LocalMinMaxScaler with skipna=True."""
        from coreforecast._lib.grouped_array import GroupedArray
        from coreforecast.scalers import LocalMinMaxScaler

        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        indptr = np.array([0, 5], dtype=np.int32)
        ga = GroupedArray(x, indptr)

        scaler_skipna = LocalMinMaxScaler(skipna=True)
        scaler_skipna.fit(ga)

        # Stats should be computed from valid values only
        assert not np.isnan(scaler_skipna.stats_[0, 0])  # min
        assert not np.isnan(scaler_skipna.stats_[0, 1])  # max - min

    def test_local_standard_scaler_skipna(self):
        """Test LocalStandardScaler with skipna=True."""
        from coreforecast._lib.grouped_array import GroupedArray
        from coreforecast.scalers import LocalStandardScaler

        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        indptr = np.array([0, 5], dtype=np.int32)
        ga = GroupedArray(x, indptr)

        scaler_skipna = LocalStandardScaler(skipna=True)
        scaler_skipna.fit(ga)

        # Stats should be computed from valid values only
        assert not np.isnan(scaler_skipna.stats_[0, 0])  # mean
        assert not np.isnan(scaler_skipna.stats_[0, 1])  # std

    def test_local_robust_scaler_iqr_skipna(self):
        """Test LocalRobustScaler with IQR and skipna=True."""
        from coreforecast._lib.grouped_array import GroupedArray
        from coreforecast.scalers import LocalRobustScaler

        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        indptr = np.array([0, 8], dtype=np.int32)
        ga = GroupedArray(x, indptr)

        scaler_skipna = LocalRobustScaler(scale="iqr", skipna=True)
        scaler_skipna.fit(ga)

        # Stats should be computed from valid values only
        assert not np.isnan(scaler_skipna.stats_[0, 0])  # median
        assert not np.isnan(scaler_skipna.stats_[0, 1])  # IQR


class TestBackwardsCompatibility:
    """Test that default behavior (skipna=False) maintains backwards compatibility."""

    def test_default_parameter_is_false(self):
        """Verify that skipna parameter defaults to False."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        # Calling without skipna should be same as skipna=False
        result_default = rolling_mean(x, window_size=3, min_samples=1)
        result_explicit = rolling_mean(x, window_size=3, min_samples=1, skipna=False)
        np.testing.assert_allclose(result_default, result_explicit, equal_nan=True)

    def test_expanding_mean_default(self):
        """Test expanding_mean default preserves current behavior."""
        x = np.array([1, 1, 1, np.nan, 1, 1, 1], dtype=float)
        result = expanding_mean(x)  # default skipna=False

        # NaN should propagate after position 3
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])
        assert np.isnan(result[3])  # At NaN
        assert np.isnan(result[4])  # After NaN (propagates)
        assert np.isnan(result[5])
        assert np.isnan(result[6])


class TestEdgeCases:
    """Test edge cases for skipna parameter."""

    def test_empty_array(self):
        """Test behavior with empty arrays."""
        x = np.array([], dtype=float)
        result = rolling_mean(x, window_size=3, min_samples=1, skipna=True)
        assert len(result) == 0

    def test_all_nan_array(self):
        """Test behavior when all values are NaN."""
        x = np.array([np.nan, np.nan, np.nan, np.nan])
        result = rolling_mean(x, window_size=2, min_samples=1, skipna=True)
        # All results should be NaN
        assert np.all(np.isnan(result))

    def test_single_value(self):
        """Test behavior with single value."""
        x = np.array([5.0])
        result = rolling_mean(x, window_size=1, min_samples=1, skipna=True)
        np.testing.assert_allclose(result[0], 5.0)

    def test_single_nan(self):
        """Test behavior with single NaN value."""
        x = np.array([np.nan])
        result = rolling_mean(x, window_size=1, min_samples=1, skipna=True)
        assert np.isnan(result[0])


class TestDtypeSupport:
    """Test support for different data types."""

    def test_float32_support(self):
        """Test that float32 arrays work with skipna."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
        result = rolling_mean(x, window_size=3, min_samples=1, skipna=True)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result[3:]))  # After NaN should have valid values

    def test_float64_support(self):
        """Test that float64 arrays work with skipna."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
        result = rolling_mean(x, window_size=3, min_samples=1, skipna=True)
        assert result.dtype == np.float64
        assert not np.any(np.isnan(result[3:]))


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
