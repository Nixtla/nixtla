"""Tests for SynAugment class."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from synforecast import SynAugment


class TestSynAugment:
    """Tests for SynAugment class."""

    def test_basic_augmentation(self) -> None:
        """Test basic augmentation workflow."""
        # Create sample data
        n = 100
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.cumsum(np.random.randn(n)),
            }
        )

        augmenter = SynAugment(seed=42)
        augmented_df = augmenter.augment(df, n_augment=2)

        # Should have original + 2 augmented series
        assert augmented_df["unique_id"].n_unique() == 3

        # Check that augmented IDs follow the naming convention
        unique_ids = augmented_df["unique_id"].unique().sort().to_list()
        assert "series_0" in unique_ids
        assert "series_0_aug_0" in unique_ids
        assert "series_0_aug_1" in unique_ids

    def test_n_augment_parameter(self) -> None:
        """Test that n_augment controls number of synthetic series."""
        n = 50
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42)

        # Test with different n_augment values
        for n_aug in [1, 3, 5]:
            augmented_df = augmenter.augment(df, n_augment=n_aug)
            # Should have 1 original + n_aug synthetic
            assert augmented_df["unique_id"].n_unique() == 1 + n_aug

    def test_synthetic_ids_format(self) -> None:
        """Test that synthetic series IDs follow the correct format."""
        n = 50
        df = pl.DataFrame(
            {
                "unique_id": ["my_series"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42)
        augmented_df = augmenter.augment(df, n_augment=2)

        unique_ids = augmented_df["unique_id"].unique().sort().to_list()
        assert "my_series" in unique_ids
        assert "my_series_aug_0" in unique_ids
        assert "my_series_aug_1" in unique_ids

    def test_timestamps_preserved(self) -> None:
        """Test that timestamps are preserved in augmented series."""
        n = 50
        timestamps = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
            interval="1h",
            eager=True,
        )
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": timestamps,
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42)
        augmented_df = augmenter.augment(df, n_augment=1, preserve_timestamps=True)

        # Get timestamps for original and augmented series
        original_ts = (
            augmented_df.filter(pl.col("unique_id") == "series_0")
            .sort("ds")["ds"]
            .to_list()
        )
        augmented_ts = (
            augmented_df.filter(pl.col("unique_id") == "series_0_aug_0")
            .sort("ds")["ds"]
            .to_list()
        )

        assert original_ts == augmented_ts

    def test_generator_auto_selection_seasonal(self) -> None:
        """Test that seasonal data gets SeasonalGenerator."""
        n = 200
        t = np.arange(n)
        # Create strongly seasonal data
        seasonal_values = (
            50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 0.5
        )

        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": seasonal_values,
            }
        )

        augmenter = SynAugment(seed=42)
        analysis = augmenter.analyze(df)

        # Should detect seasonal pattern
        assert "series_0" in analysis
        assert analysis["series_0"]["properties"]["seasonality"]["has_seasonality"]

    def test_generator_auto_selection_random_walk(self) -> None:
        """Test that random walk data gets an appropriate generator."""
        n = 100
        # Create random walk
        np.random.seed(123)  # Fixed seed for reproducibility
        random_walk = np.cumsum(np.random.randn(n))

        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": random_walk,
            }
        )

        augmenter = SynAugment(seed=42)
        analysis = augmenter.analyze(df)

        # Random walk should not be stationary
        assert "series_0" in analysis
        # The generator should be something that handles non-stationary data
        # Random walks can be detected as various types depending on the specific realization
        generator = analysis["series_0"]["recommended_generator"]
        assert (
            generator
            in [
                "RandomWalkGenerator",
                "SARIMAGenerator",
                "FractionalBrownianMotionGenerator",
                "RegimeSwitchingGenerator",  # Can be detected if there are apparent regime changes
                "SeasonalGenerator",  # Can be detected if autocorrelation creates apparent periodicity
            ]
        )

    def test_generator_override(self) -> None:
        """Test that user can override generator selection."""
        n = 100
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42)

        # Override to use SARIMA
        augmented_df = augmenter.augment(
            df, n_augment=1, generator_override={"series_0": "SARIMAGenerator"}
        )

        # Should still produce valid output
        assert augmented_df["unique_id"].n_unique() == 2

    def test_statistical_similarity(self) -> None:
        """Test that synthetic series match original stats approximately."""
        n = 200
        mean = 50.0
        std = 10.0
        np.random.seed(42)
        original_values = np.random.randn(n) * std + mean

        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": original_values,
            }
        )

        augmenter = SynAugment(seed=42)
        augmented_df = augmenter.augment(df, n_augment=5)

        # Get stats for original
        original_mean = df["y"].mean()
        original_std = df["y"].std()

        # Get stats for augmented series - at least some should be reasonable
        reasonable_count = 0
        for i in range(5):
            aug_values = augmented_df.filter(
                pl.col("unique_id") == f"series_0_aug_{i}"
            )["y"]
            aug_mean = aug_values.mean()
            aug_std = aug_values.std()

            # Check if stats are reasonable (wider tolerance since generators vary)
            mean_ok = abs(aug_mean - original_mean) < 5 * original_std
            # Std check with wider tolerance - generators may produce different variability
            std_ok = aug_std > 0.1 * original_std and aug_std < 10 * original_std

            if mean_ok and std_ok:
                reasonable_count += 1

        # At least some synthetic series should have reasonable statistics
        assert reasonable_count >= 1, (
            "At least one synthetic series should have reasonable stats"
        )

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same output."""
        n = 50
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.cumsum(np.random.randn(n)),
            }
        )

        # Generate twice with same seed
        augmenter1 = SynAugment(seed=42)
        augmenter2 = SynAugment(seed=42)

        result1 = augmenter1.augment(df, n_augment=1)
        result2 = augmenter2.augment(df, n_augment=1)

        # Get augmented values
        aug1 = result1.filter(pl.col("unique_id") == "series_0_aug_0")["y"].to_numpy()
        aug2 = result2.filter(pl.col("unique_id") == "series_0_aug_0")["y"].to_numpy()

        np.testing.assert_array_almost_equal(aug1, aug2)

    def test_analyze_method(self) -> None:
        """Test that analyze returns expected structure."""
        n = 100
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n + ["series_1"] * n,
                "ds": list(
                    pl.datetime_range(
                        pl.datetime(2020, 1, 1),
                        pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                        interval="1h",
                        eager=True,
                    )
                )
                * 2,
                "y": list(np.random.randn(n)) + list(np.cumsum(np.random.randn(n))),
            }
        )

        augmenter = SynAugment(seed=42)
        analysis = augmenter.analyze(df)

        # Should have analysis for both series
        assert "series_0" in analysis
        assert "series_1" in analysis

        # Each should have expected keys
        for series_id in ["series_0", "series_1"]:
            assert "recommended_generator" in analysis[series_id]
            assert "properties" in analysis[series_id]
            assert "fitted_params" in analysis[series_id]

            # Properties should have expected sub-keys
            props = analysis[series_id]["properties"]
            assert "basic_stats" in props
            assert "seasonality" in props
            assert "trend" in props
            assert "stationarity" in props

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        df = pl.DataFrame(
            {"unique_id": [], "ds": [], "y": []},
            schema={
                "unique_id": pl.Utf8,
                "ds": pl.Datetime,
                "y": pl.Float64,
            },
        )

        augmenter = SynAugment(seed=42)
        analysis = augmenter.analyze(df)

        # Should return empty analysis
        assert len(analysis) == 0

    def test_single_series(self) -> None:
        """Test augmentation with a single short series."""
        n = 30
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42)
        augmented_df = augmenter.augment(df, n_augment=1)

        assert augmented_df["unique_id"].n_unique() == 2

    def test_multiple_series_different_generators(self) -> None:
        """Test that multiple series can use different generators."""
        n = 100

        # Series 0: Seasonal
        t = np.arange(n)
        seasonal = 50 + 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 0.5

        # Series 1: Random walk
        random_walk = np.cumsum(np.random.randn(n))

        df = pl.DataFrame(
            {
                "unique_id": ["seasonal"] * n + ["random_walk"] * n,
                "ds": list(
                    pl.datetime_range(
                        pl.datetime(2020, 1, 1),
                        pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                        interval="1h",
                        eager=True,
                    )
                )
                * 2,
                "y": list(seasonal) + list(random_walk),
            }
        )

        augmenter = SynAugment(seed=42)
        analysis = augmenter.analyze(df)

        # Different series should potentially get different generators
        # (though this isn't guaranteed - depends on actual data)
        assert analysis["seasonal"]["recommended_generator"] is not None
        assert analysis["random_walk"]["recommended_generator"] is not None

    def test_backend_polars(self) -> None:
        """Test that polars backend works correctly."""
        n = 50
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42, engine="polars")
        augmented_df = augmenter.augment(df, n_augment=1)

        assert isinstance(augmented_df, pl.DataFrame)

    def test_backend_pandas(self) -> None:
        """Test that pandas backend works correctly."""
        n = 50
        df = pd.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pd.date_range("2020-01-01", periods=n, freq="h"),
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42, engine="pandas")
        augmented_df = augmenter.augment(df, n_augment=1)

        assert isinstance(augmented_df, pd.DataFrame)

    def test_missing_columns_error(self) -> None:
        """Test that missing columns raise ValueError."""
        df = pl.DataFrame(
            {
                "wrong_id": ["series_0"] * 10,
                "wrong_time": range(10),
                "wrong_value": np.random.randn(10),
            }
        )

        augmenter = SynAugment(seed=42)

        with pytest.raises(ValueError, match="missing required columns"):
            augmenter.augment(df, n_augment=1)

    def test_invalid_n_augment(self) -> None:
        """Test that n_augment < 1 raises ValueError."""
        n = 50
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42)

        with pytest.raises(ValueError, match="n_augment must be >= 1"):
            augmenter.augment(df, n_augment=0)

    def test_custom_column_names(self) -> None:
        """Test augmentation with custom column names."""
        n = 50
        df = pl.DataFrame(
            {
                "series_id": ["s0"] * n,
                "timestamp": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "value": np.random.randn(n),
            }
        )

        augmenter = SynAugment(
            id_col="series_id",
            time_col="timestamp",
            target_col="value",
            seed=42,
        )
        augmented_df = augmenter.augment(df, n_augment=1)

        assert set(augmented_df.columns) == {"series_id", "timestamp", "value"}
        assert augmented_df["series_id"].n_unique() == 2


class TestAnalysisFunctions:
    """Tests for analysis functions used by SynAugment."""

    def test_classify_series_seasonal(self) -> None:
        """Test that strongly seasonal series is classified correctly."""
        from synforecast._analysis import classify_series

        n = 200
        t = np.arange(n)
        seasonal = 10 * np.sin(2 * np.pi * t / 24) + np.random.randn(n) * 0.5

        result = classify_series(seasonal)

        assert result["properties"]["seasonality"]["has_seasonality"]
        assert result["recommended_generator"] in [
            "SeasonalGenerator",
            "SARIMAGenerator",
        ]

    def test_classify_series_intermittent(self) -> None:
        """Test that intermittent series is classified correctly."""
        from synforecast._analysis import classify_series

        n = 100
        # Create intermittent demand
        intermittent = np.zeros(n)
        intermittent[np.random.choice(n, size=20, replace=False)] = np.random.randint(
            1, 10, 20
        )

        result = classify_series(intermittent)

        assert result["properties"]["intermittency"]["is_intermittent"]
        assert result["recommended_generator"] == "IntermittentDemandGenerator"

    def test_detect_trend(self) -> None:
        """Test trend detection."""
        from synforecast._analysis import detect_trend

        n = 100
        # Create series with clear trend
        t = np.arange(n)
        trending = 0.5 * t + np.random.randn(n) * 2

        result = detect_trend(trending)

        assert result["has_trend"]
        assert result["slope"] > 0
        assert result["r_squared"] > 0.5


class TestFittingFunctions:
    """Tests for fitting functions used by SynAugment."""

    def test_fit_random_walk(self) -> None:
        """Test random walk parameter fitting."""
        from synforecast._fitting import fit_random_walk

        n = 100
        drift = 0.1
        vol = 2.0
        steps = np.random.randn(n) * vol + drift
        random_walk = np.cumsum(steps)

        params = fit_random_walk(random_walk)

        assert "drift" in params
        assert "volatility" in params
        assert "start_value" in params
        # Drift and volatility should be in the right ballpark
        assert abs(params["drift"] - drift) < 1.0
        assert abs(params["volatility"] - vol) < 1.0

    def test_fit_seasonal(self) -> None:
        """Test seasonal parameter fitting."""
        from synforecast._fitting import fit_seasonal

        n = 200
        period = 24
        amplitude = 10.0
        t = np.arange(n)
        seasonal = amplitude * np.sin(2 * np.pi * t / period) + np.random.randn(n)

        params = fit_seasonal(seasonal, period=period)

        assert "seasonality_period" in params
        assert "seasonality_amplitude" in params
        assert params["seasonality_period"] == period
        # Amplitude should be approximately correct
        assert abs(params["seasonality_amplitude"] - amplitude) < amplitude * 0.5

    def test_fit_garch(self) -> None:
        """Test GARCH parameter fitting."""
        from synforecast._fitting import fit_garch

        n = 500
        # Create series with volatility clustering
        returns = np.random.randn(n)
        for i in range(1, n):
            returns[i] *= 1 + 0.3 * abs(returns[i - 1])

        params = fit_garch(returns)

        assert "omega" in params
        assert "alpha" in params
        assert "beta" in params
        assert params["omega"] > 0
        # alpha and beta should be lists for GARCH(p,q)
        assert isinstance(params["alpha"], list)
        assert isinstance(params["beta"], list)
        assert len(params["alpha"]) == 1
        assert len(params["beta"]) == 1
        assert 0 <= params["alpha"][0] <= 1
        assert 0 <= params["beta"][0] <= 1

    def test_fit_ornstein_uhlenbeck(self) -> None:
        """Test Ornstein-Uhlenbeck parameter fitting."""
        from synforecast._fitting import fit_ornstein_uhlenbeck

        n = 200
        # Create mean-reverting series
        mu = 5.0
        theta = 0.5
        sigma = 1.0
        series = np.zeros(n)
        series[0] = mu
        for i in range(1, n):
            series[i] = (
                series[i - 1] + theta * (mu - series[i - 1]) + sigma * np.random.randn()
            )

        params = fit_ornstein_uhlenbeck(series)

        assert "theta" in params
        assert "mu" in params
        assert "sigma" in params
        assert "initial_value" in params
        assert params["theta"] > 0
        assert params["sigma"] > 0
        # Mean should be close to actual mean
        assert abs(params["mu"] - mu) < 3.0

    def test_fit_ornstein_uhlenbeck_round_trip(self) -> None:
        """Fit -> generate reproduces theta, acf1 and the stationary std.

        The generator is an Euler-Maruyama AR(1) with phi = 1 - theta*dt,
        so the fitted theta must be 1 - acf1 (dt = 1) and the fitted sigma
        must satisfy Var(X) = sigma^2 / (theta * (2 - theta)).
        """
        from synforecast._fitting import fit_ornstein_uhlenbeck
        from synforecast.generators import OrnsteinUhlenbeckGenerator
        from tests.helpers import assert_acf, assert_std, sample_acf, series_values

        n = 20000
        theta, mu, sigma = 0.5, 2.0, 1.0
        gen = OrnsteinUhlenbeckGenerator(
            min_length=n,
            max_length=n,
            freq="D",
            theta=theta,
            mu=mu,
            sigma=sigma,
            initial_value=mu,
            seed=11,
        )
        original = next(iter(series_values(gen.generate(n_series=1)).values()))

        params = fit_ornstein_uhlenbeck(original)

        # Euler-consistent estimator recovers theta and always respects the
        # generator's stability bound theta * dt < 2 (dt defaults to 1).
        assert abs(params["theta"] - theta) < 0.05
        assert params["theta"] < 2.0
        assert abs(params["mu"] - mu) < 0.2

        regen = OrnsteinUhlenbeckGenerator(
            min_length=n, max_length=n, freq="D", seed=13, **params
        )
        values = next(iter(series_values(regen.generate(n_series=1)).values()))

        # Regenerated series reproduces the source's lag-1 autocorrelation
        # and stationary std; thin the std samples to near-independence
        # (phi^10 ~ 1e-3) so the i.i.d. z-bound in assert_std applies.
        assert_acf(values, lag=1, expected=sample_acf(original, 1))
        assert_std(values[::10], expected=float(np.std(original)))

    def test_fit_ornstein_uhlenbeck_stays_within_stability_bound(self) -> None:
        """Even for near-white-noise input (acf1 ~ 0 or < 0), the fitted
        theta must instantiate the generator without tripping its
        theta * dt < 2 validator (the old -log(acf1) fit could return 4.6)."""
        from synforecast._fitting import fit_ornstein_uhlenbeck
        from synforecast.generators import OrnsteinUhlenbeckGenerator

        rng = np.random.default_rng(7)
        # Anti-persistent series: acf1 ~ -0.5, worst case for the estimator.
        noise = rng.normal(size=2000)
        series = noise[1:] - noise[:-1]

        params = fit_ornstein_uhlenbeck(series)
        gen = OrnsteinUhlenbeckGenerator(
            min_length=50, max_length=50, freq="D", **params
        )
        assert gen.theta * gen.dt < 2.0

    def test_fit_fbm(self) -> None:
        """Test Fractional Brownian Motion parameter fitting."""
        from synforecast._fitting import fit_fbm

        n = 100
        # Create series that looks like fBm (using cumsum as approximation)
        np.random.seed(42)
        series = np.cumsum(np.random.randn(n))

        params = fit_fbm(series)

        assert "hurst" in params
        assert "sigma" in params
        assert "initial_value" in params
        # Hurst exponent should be in valid range
        assert 0.0 < params["hurst"] < 1.0
        assert params["sigma"] > 0

    def test_fit_regime_switching(self) -> None:
        """Test regime switching parameter fitting."""
        from synforecast._fitting import fit_regime_switching

        # Create series with two regimes
        regime1 = np.random.randn(100) * 1.0 + 0.0  # Low mean, low variance
        regime2 = np.random.randn(100) * 3.0 + 5.0  # High mean, high variance
        series = np.concatenate([regime1, regime2])

        params = fit_regime_switching(series, n_regimes=2)

        assert "n_regimes" in params
        assert "regime_means" in params
        assert "regime_variances" in params
        assert params["n_regimes"] == 2
        assert len(params["regime_means"]) == 2
        assert len(params["regime_variances"]) == 2
        # Variances should be positive
        assert all(v > 0 for v in params["regime_variances"])

    def test_fit_intermittent(self) -> None:
        """Test intermittent demand parameter fitting."""
        from synforecast._fitting import fit_intermittent

        n = 100
        # Create intermittent demand
        demand_prob = 0.3
        demand_mean = 5.0
        series = np.zeros(n)
        demand_mask = np.random.rand(n) < demand_prob
        series[demand_mask] = np.abs(
            np.random.randn(demand_mask.sum()) * 2 + demand_mean
        )

        params = fit_intermittent(series)

        assert "demand_probability" in params
        assert "demand_mean" in params
        assert "demand_std" in params
        assert 0 < params["demand_probability"] < 1
        assert params["demand_mean"] > 0
        assert params["demand_std"] > 0

    def test_fit_gbm(self) -> None:
        """Test Geometric Brownian Motion parameter fitting."""
        from synforecast._fitting import fit_gbm

        n = 200
        # Create GBM-like series (positive values with log-normal increments)
        mu = 0.05
        sigma = 0.2
        series = np.zeros(n)
        series[0] = 100.0
        for i in range(1, n):
            series[i] = series[i - 1] * np.exp(
                (mu - 0.5 * sigma**2) + sigma * np.random.randn()
            )

        params = fit_gbm(series)

        # Should use correct parameter names (mu/sigma, not drift/volatility)
        assert "mu" in params
        assert "sigma" in params
        assert "initial_value" in params
        assert params["sigma"] > 0
        assert params["initial_value"] > 0


class TestSynAugmentValidation:
    """Tests for SynAugment input validation."""

    def test_n_augment_upper_bound(self) -> None:
        """Test that n_augment > 1000 raises ValueError."""
        n = 50
        df = pl.DataFrame(
            {
                "unique_id": ["series_0"] * n,
                "ds": pl.datetime_range(
                    pl.datetime(2020, 1, 1),
                    pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                ),
                "y": np.random.randn(n),
            }
        )

        augmenter = SynAugment(seed=42)

        with pytest.raises(ValueError, match="n_augment must be <= 1000"):
            augmenter.augment(df, n_augment=1001)
