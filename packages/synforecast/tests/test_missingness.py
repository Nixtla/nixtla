"""Tests for missing data functionality in BaseGenerator."""

import numpy as np
import polars as pl
import pytest

from synforecast._core import _add_missingness
from synforecast.generators import CopulaGenerator, RandomWalkGenerator, VARGenerator


class TestMissingness:
    """Tests for missing data patterns."""

    def test_no_missingness_by_default(self) -> None:
        """Test that no missing data is added by default."""
        params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "D",
            "engine": "polars",
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)
        df = generator.generate(n_series=1)
        # Check that there are no NaN values
        values = df["y"].to_numpy()
        assert not np.any(np.isnan(values))

    def test_random_missingness(self) -> None:
        """Test random missing data pattern."""
        params = {
            "min_length": 200,
            "max_length": 200,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "random",
            "missing_rate": 0.2,
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)
        df = generator.generate(n_series=1)
        # Check that approximately 20% of values are missing
        values = df["y"].to_numpy()
        nan_count = np.sum(np.isnan(values))
        nan_rate = nan_count / len(values)
        assert 0.1 < nan_rate < 0.3  # Allow some variation

    def test_block_missingness(self) -> None:
        """Test block missing data pattern."""
        params = {
            "min_length": 200,
            "max_length": 200,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "block",
            "missing_rate": 0.2,
            "missing_block_size": 5,
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)
        df = generator.generate(n_series=1)
        # Check that there are missing values
        values = df["y"].to_numpy()
        nan_count = np.sum(np.isnan(values))
        assert nan_count > 0

        # Check for consecutive NaN values (blocks)
        has_consecutive_nans = False
        consecutive_count = 0

        for i in range(len(values)):
            if np.isnan(values[i]):
                consecutive_count += 1
                if consecutive_count >= 2:
                    has_consecutive_nans = True
                    break
            else:
                consecutive_count = 0

        assert has_consecutive_nans

    def test_seasonal_missingness(self) -> None:
        """Test seasonal missing data pattern."""
        params = {
            "min_length": 200,
            "max_length": 200,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "seasonal",
            "missing_rate": 0.15,
            "missing_seasonal_period": 7,
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)
        df = generator.generate(n_series=1)
        # Check that there are missing values
        values = df["y"].to_numpy()
        nan_count = np.sum(np.isnan(values))
        assert nan_count > 0

    def test_invalid_missing_pattern(self) -> None:
        """Test that invalid missing pattern raises error."""
        params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "invalid",
        }

        with pytest.raises(ValueError, match="missing_pattern"):
            RandomWalkGenerator(**params)

    def test_invalid_missing_rate(self) -> None:
        """Test that invalid missing rate raises error."""
        params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_rate": 1.5,  # > 1
        }

        with pytest.raises(ValueError, match="missing_rate"):
            RandomWalkGenerator(**params)

    def test_negative_missing_rate(self) -> None:
        """Test that a negative missing rate raises error."""
        params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_rate": -0.1,
        }

        with pytest.raises(ValueError, match="missing_rate"):
            RandomWalkGenerator(**params)

    def test_invalid_block_size(self) -> None:
        """Test that invalid block size raises error."""
        params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "block",
            "missing_block_size": 0,
        }

        with pytest.raises(ValueError, match="missing_block_size"):
            RandomWalkGenerator(**params)

    def test_invalid_seasonal_period(self) -> None:
        """Test that invalid seasonal period raises error."""
        params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "seasonal",
            "missing_seasonal_period": 0,
        }

        with pytest.raises(ValueError, match="missing_seasonal_period"):
            RandomWalkGenerator(**params)

    def test_multiple_series_with_missingness(self) -> None:
        """Test that missingness works with multiple series."""
        params = {
            "min_length": 100,
            "max_length": 100,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "random",
            "missing_rate": 0.2,
            "seed": 42,
        }

        generator = RandomWalkGenerator(**params)
        df = generator.generate(n_series=3)
        # Check that all series have some missing values
        for series_id in df["unique_id"].unique():
            series_df = df.filter(pl.col("unique_id") == series_id)
            values = series_df["y"].to_numpy()
            nan_count = np.sum(np.isnan(values))
            assert nan_count > 0

    def test_different_missing_rates(self) -> None:
        """Test different missing rates."""
        for rate in [0.1, 0.3, 0.5]:
            params = {
                "min_length": 500,
                "max_length": 500,
                "freq": "D",
                "engine": "polars",
                "missing_data": True,
                "missing_pattern": "random",
                "missing_rate": rate,
                "seed": 42,
            }

            generator = RandomWalkGenerator(**params)
            df = generator.generate(n_series=1)

            values = df["y"].to_numpy()
            nan_count = np.sum(np.isnan(values))
            nan_rate = nan_count / len(values)
            # Allow 50% variation from target rate
            assert rate * 0.5 < nan_rate < rate * 1.5

    def test_var_with_missingness(self) -> None:
        """Test VAR generator with missing data."""
        params = {
            "min_length": 200,
            "max_length": 200,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "random",
            "missing_rate": 0.2,
            "seed": 42,
        }

        generator = VARGenerator(**params)
        df = generator.generate(n_series=3)
        # Check that all three series have missing values
        for series_id in df["unique_id"].unique().sort():
            series_df = df.filter(pl.col("unique_id") == series_id)
            values = series_df["y"].to_numpy()
            nan_count = np.sum(np.isnan(values))
            assert nan_count > 0
            # Check approximate rate
            nan_rate = nan_count / len(values)
            assert 0.1 < nan_rate < 0.3

    def test_copula_with_missingness(self) -> None:
        """Test Copula generator with missing data."""
        params = {
            "min_length": 200,
            "max_length": 200,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "block",
            "missing_rate": 0.15,
            "missing_block_size": 5,
            "seed": 42,
        }

        generator = CopulaGenerator(**params)
        df = generator.generate(n_series=3)
        # Check that all three series have missing values
        for series_id in df["unique_id"].unique().sort():
            series_df = df.filter(pl.col("unique_id") == series_id)
            values = series_df["y"].to_numpy()
            nan_count = np.sum(np.isnan(values))
            assert nan_count > 0

    @pytest.mark.parametrize("engine", ["polars", "pandas"])
    def test_missing_rate_zero_is_noop(self, engine) -> None:
        """A missing_rate of exactly 0 is accepted and injects no NaNs."""
        generator = RandomWalkGenerator(
            min_length=100,
            max_length=100,
            freq="D",
            engine=engine,
            missing_data=True,
            missing_rate=0.0,
            seed=42,
        )
        df = generator.generate(n_series=2)
        values = np.asarray(df["y"])
        assert not np.any(np.isnan(values))

    @pytest.mark.parametrize("engine", ["polars", "pandas"])
    @pytest.mark.parametrize("pattern", ["random", "block", "seasonal"])
    def test_missing_rate_one_all_missing(self, engine, pattern) -> None:
        """A missing_rate of exactly 1 marks every point missing."""
        generator = RandomWalkGenerator(
            min_length=100,
            max_length=100,
            freq="D",
            engine=engine,
            missing_data=True,
            missing_pattern=pattern,
            missing_rate=1.0,
            seed=42,
        )
        df = generator.generate(n_series=2)
        values = np.asarray(df["y"])
        assert np.all(np.isnan(values))

    def test_multivariate_seasonal_missingness(self) -> None:
        """Test multivariate generator with seasonal missing data."""
        params = {
            "min_length": 300,
            "max_length": 300,
            "freq": "D",
            "engine": "polars",
            "missing_data": True,
            "missing_pattern": "seasonal",
            "missing_rate": 0.2,
            "missing_seasonal_period": 7,
            "seed": 42,
        }

        generator = VARGenerator(**params)
        df = generator.generate(n_series=2)
        # Check that both series have missing values
        assert df["unique_id"].n_unique() == 2
        for series_id in df["unique_id"].unique():
            series_df = df.filter(pl.col("unique_id") == series_id)
            values = series_df["y"].to_numpy()
            nan_count = np.sum(np.isnan(values))
            assert nan_count > 0


class TestMissingRateEndpoints:
    """Endpoint semantics of native missingness injection."""

    @pytest.fixture
    def injection_engine(self):
        """Mark tests that exercise native missingness injection."""

    @pytest.mark.parametrize("pattern", ["random", "block", "seasonal"])
    @pytest.mark.usefixtures("injection_engine")
    def test_rate_zero_no_nans(self, pattern) -> None:
        values = np.arange(100, dtype=np.float64)
        out, meta = _add_missingness(
            values, np.random.default_rng(42), pattern, 0.0, 3, 7
        )
        assert not np.any(np.isnan(out))
        assert len(meta["missing_indices"]) == 0

    @pytest.mark.parametrize("pattern", ["random", "block", "seasonal"])
    @pytest.mark.usefixtures("injection_engine")
    def test_rate_one_all_nan_exact_metadata(self, pattern) -> None:
        values = np.arange(100, dtype=np.float64)
        out, meta = _add_missingness(
            values, np.random.default_rng(42), pattern, 1.0, 3, 7
        )
        assert np.all(np.isnan(out))
        np.testing.assert_array_equal(
            np.sort(np.asarray(meta["missing_indices"])), np.arange(100)
        )
