"""Tests for exogenous variable generation."""

import numpy as np
import polars as pl
import pytest

from synforecast.dataset import SynSet
from synforecast.exogenous import (
    CorrelatedExogConfig,
    ExogenousConfig,
    SeriesMetadata,
    build_flag_column,
    extract_datetime_features,
    generate_correlated_exog,
)
from synforecast.generators import (
    CopulaGenerator,
    DailyActiveUsersGenerator,
    IoTSensorGenerator,
    RandomWalkGenerator,
    VARGenerator,
)


# ---------------------------------------------------------------------------
# ExogenousConfig validation
# ---------------------------------------------------------------------------
class TestExogenousConfig:
    """Tests for ExogenousConfig and CorrelatedExogConfig validation."""

    def test_default_config(self) -> None:
        cfg = ExogenousConfig()
        assert cfg.datetime_features is False
        assert cfg.datetime_cyclical is False
        assert cfg.anomaly_flags is False
        assert cfg.changepoint_flags is False
        assert cfg.missing_flags is False
        assert cfg.correlated == []

    def test_all_flags_enabled(self) -> None:
        cfg = ExogenousConfig(
            datetime_features=True,
            datetime_cyclical=True,
            anomaly_flags=True,
            changepoint_flags=True,
            missing_flags=True,
        )
        assert cfg.datetime_features is True
        assert cfg.missing_flags is True

    def test_correlated_config_defaults(self) -> None:
        cfg = CorrelatedExogConfig(name="price")
        assert cfg.method == "correlated_noise"
        assert cfg.correlation == 0.7
        assert cfg.lag == 1
        assert cfg.noise_std == 0.1

    def test_correlated_config_correlation_bounds(self) -> None:
        CorrelatedExogConfig(name="a", correlation=-1.0)
        CorrelatedExogConfig(name="b", correlation=1.0)
        with pytest.raises(ValueError):
            CorrelatedExogConfig(name="c", correlation=1.5)
        with pytest.raises(ValueError):
            CorrelatedExogConfig(name="d", correlation=-1.5)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
class TestBackwardCompatibility:
    """Ensure exogenous=None produces the original 3-column output."""

    def test_no_exogenous_gives_three_columns(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "seed": 42,
                "engine": "polars",
            }
        )
        df = gen.generate(n_series=2)
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"unique_id", "ds", "y"}

    def test_empty_exogenous_gives_three_columns(self) -> None:
        """ExogenousConfig with all defaults off should still be 3 columns."""
        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(),
            }
        )
        df = gen.generate(n_series=2)
        assert set(df.columns) == {"unique_id", "ds", "y"}


# ---------------------------------------------------------------------------
# Datetime features
# ---------------------------------------------------------------------------
class TestDatetimeFeatures:
    """Tests for datetime feature extraction."""

    def test_basic_daily_features(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 30,
                "max_length": 30,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(datetime_features=True),
            }
        )
        df = gen.generate(n_series=1)
        expected_cols = {
            "unique_id",
            "ds",
            "y",
            "year",
            "quarter",
            "month",
            "day_of_year",
            "day_of_week",
            "day_of_month",
            "is_weekend",
        }
        assert expected_cols.issubset(set(df.columns))
        # daily data should NOT have hour
        assert "hour" not in df.columns

    def test_hourly_features_include_hour(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(datetime_features=True),
            }
        )
        df = gen.generate(n_series=1)
        assert "hour" in df.columns

    def test_cyclical_features(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(datetime_cyclical=True),
            }
        )
        df = gen.generate(n_series=1)
        for col in [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "doy_sin",
            "doy_cos",
        ]:
            assert col in df.columns, f"Missing cyclical column: {col}"

        # Cyclical values should be in [-1, 1]
        for col in ["hour_sin", "hour_cos"]:
            vals = df[col].to_numpy()
            assert np.all(vals >= -1.0 - 1e-6)
            assert np.all(vals <= 1.0 + 1e-6)

    def test_extract_datetime_features_function(self) -> None:
        timestamps = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-02-01"),
            np.timedelta64(1, "D"),
        ).astype("datetime64[ns]")

        features = extract_datetime_features(
            timestamps, freq="D", include_basic=True, include_cyclical=False
        )
        assert "year" in features
        assert np.all(features["year"] == 2020)
        assert features["month"][0] == 1

    def test_cyclical_encoding_range(self) -> None:
        timestamps = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-02"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[ns]")

        features = extract_datetime_features(
            timestamps, freq="h", include_basic=False, include_cyclical=True
        )
        for col in ["hour_sin", "hour_cos"]:
            assert col in features
            assert np.all(features[col] >= -1.0 - 1e-6)
            assert np.all(features[col] <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Flag columns
# ---------------------------------------------------------------------------
class TestFlagColumns:
    """Tests for anomaly, changepoint, and missing flags."""

    def test_anomaly_flags_present(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "anomaly_fraction": 0.1,
                "exogenous": ExogenousConfig(anomaly_flags=True),
            }
        )
        df = gen.generate(n_series=2)
        assert "anomaly_flag" in df.columns
        flags = df["anomaly_flag"].to_numpy()
        # Should have some anomalies flagged
        assert np.sum(flags) > 0
        # Should be binary
        assert set(np.unique(flags)).issubset({0, 1})

    def test_changepoint_flags_present(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "changepoints": True,
                "num_changepoints": 3,
                "exogenous": ExogenousConfig(changepoint_flags=True),
            }
        )
        df = gen.generate(n_series=2)
        assert "changepoint_flag" in df.columns
        flags = df["changepoint_flag"].to_numpy()
        assert np.sum(flags) > 0
        assert set(np.unique(flags)).issubset({0, 1})

    def test_missing_flags_present(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "missing_data": True,
                "missing_rate": 0.1,
                "exogenous": ExogenousConfig(missing_flags=True),
            }
        )
        df = gen.generate(n_series=2)
        assert "missing_flag" in df.columns
        flags = df["missing_flag"].to_numpy()
        assert np.sum(flags) > 0
        assert set(np.unique(flags)).issubset({0, 1})

    def test_flags_without_injection_are_zero(self) -> None:
        """If anomalies are not enabled, anomaly_flag should be all zeros."""
        gen = RandomWalkGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": False,
                "exogenous": ExogenousConfig(anomaly_flags=True),
            }
        )
        df = gen.generate(n_series=1)
        assert "anomaly_flag" in df.columns
        assert df["anomaly_flag"].sum() == 0

    def test_build_flag_column_function(self) -> None:
        meta = SeriesMetadata(
            values=np.zeros(10),
            timestamps=np.zeros(10),
            series_id=0,
            length=10,
            anomaly_indices=np.array([2, 5, 7]),
        )
        flags = build_flag_column([meta], "anomaly")
        assert len(flags) == 10
        assert flags[2] == 1
        assert flags[5] == 1
        assert flags[7] == 1
        assert flags[0] == 0

    def test_build_flag_column_out_of_bounds(self) -> None:
        """Out-of-bounds indices should be safely ignored."""
        meta = SeriesMetadata(
            values=np.zeros(5),
            timestamps=np.zeros(5),
            series_id=0,
            length=5,
            anomaly_indices=np.array([1, 10, -1]),
        )
        flags = build_flag_column([meta], "anomaly")
        assert len(flags) == 5
        assert flags[1] == 1
        assert np.sum(flags) == 1  # 10 and -1 are out of bounds


# ---------------------------------------------------------------------------
# Correlated exogenous
# ---------------------------------------------------------------------------
class TestCorrelatedExogenous:
    """Tests for correlated exogenous variable generation."""

    def test_correlated_noise_column_exists(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(
                    correlated=[
                        CorrelatedExogConfig(name="price", correlation=0.8),
                    ]
                ),
            }
        )
        df = gen.generate(n_series=2)
        assert "price" in df.columns
        assert len(df.filter(pl.col("price").is_not_null())) > 0

    def test_correlated_noise_correlation(self) -> None:
        """Correlation of correlated_noise output should be roughly correct."""
        gen = RandomWalkGenerator(
            **{
                "min_length": 5000,
                "max_length": 5000,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(
                    correlated=[
                        CorrelatedExogConfig(
                            name="corr_var",
                            method="correlated_noise",
                            correlation=0.9,
                        ),
                    ]
                ),
            }
        )
        df = gen.generate(n_series=1)
        y = df["y"].to_numpy()
        exog = df["corr_var"].to_numpy()
        corr = np.corrcoef(y, exog)[0, 1]
        # With 5000 points, empirical correlation should be close
        assert abs(corr - 0.9) < 0.15, f"Expected ~0.9, got {corr}"

    def test_lagged_copy(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(
                    correlated=[
                        CorrelatedExogConfig(
                            name="lagged_y",
                            method="lagged_copy",
                            lag=3,
                            noise_std=0.0,
                        ),
                    ]
                ),
            }
        )
        df = gen.generate(n_series=1)
        y = df["y"].to_numpy()
        lagged = df["lagged_y"].to_numpy()
        # With noise_std=0, lagged values after the lag offset should equal
        # the original values shifted by lag
        np.testing.assert_array_almost_equal(lagged[3:], y[:-3])

    def test_trend_following(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(
                    correlated=[
                        CorrelatedExogConfig(
                            name="trend",
                            method="trend_following",
                            smoothing_window=5,
                            trend_noise_std=0.0,
                        ),
                    ]
                ),
            }
        )
        df = gen.generate(n_series=1)
        assert "trend" in df.columns
        # Trend-following with no noise should be a moving average
        y = df["y"].to_numpy()
        trend = df["trend"].to_numpy()
        # The moving average smooths the signal; check it's correlated
        corr = np.corrcoef(y, trend)[0, 1]
        assert corr > 0.5, f"Expected positive correlation, got {corr}"

    def test_multiple_correlated(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(
                    correlated=[
                        CorrelatedExogConfig(name="a", correlation=0.5),
                        CorrelatedExogConfig(name="b", method="lagged_copy", lag=2),
                        CorrelatedExogConfig(name="c", method="trend_following"),
                    ]
                ),
            }
        )
        df = gen.generate(n_series=2)
        assert "a" in df.columns
        assert "b" in df.columns
        assert "c" in df.columns

    def test_generate_correlated_exog_function(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(50)
        meta = SeriesMetadata(
            values=values,
            timestamps=np.zeros(50),
            series_id=0,
            length=50,
        )
        config = CorrelatedExogConfig(name="test", correlation=0.8)
        result = generate_correlated_exog([meta], values, config, rng)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# Integration with overriding generators
# ---------------------------------------------------------------------------
class TestOverridingGenerators:
    """Test exogenous support in generators that override generate()."""

    def test_copula_with_exogenous(self) -> None:
        gen = CopulaGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "exogenous": ExogenousConfig(
                    datetime_features=True,
                    anomaly_flags=True,
                ),
            }
        )
        df = gen.generate(n_series=3)
        assert "anomaly_flag" in df.columns
        assert "year" in df.columns
        assert df["unique_id"].n_unique() == 3

    def test_var_with_exogenous(self) -> None:
        gen = VARGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "changepoints": True,
                "num_changepoints": 2,
                "exogenous": ExogenousConfig(
                    changepoint_flags=True,
                    datetime_cyclical=True,
                ),
            }
        )
        df = gen.generate(n_series=3)
        assert "changepoint_flag" in df.columns
        assert "month_sin" in df.columns
        assert df["unique_id"].n_unique() == 3

    def test_iot_sensor_with_exogenous(self) -> None:
        gen = IoTSensorGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "n_sensors": 1,
                "missing_data": True,
                "exogenous": ExogenousConfig(
                    missing_flags=True,
                    datetime_features=True,
                ),
            }
        )
        df = gen.generate(n_series=2)
        assert "missing_flag" in df.columns
        assert "hour" in df.columns

    def test_dau_with_exogenous_and_event_col(self) -> None:
        gen = DailyActiveUsersGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "exogenous": ExogenousConfig(
                    anomaly_flags=True,
                    datetime_features=True,
                    correlated=[
                        CorrelatedExogConfig(name="price", correlation=0.5),
                    ],
                ),
            }
        )
        df = gen.generate(n_series=2)
        # DAU should still have its event column
        assert "event" in df.columns
        assert "anomaly_flag" in df.columns
        assert "year" in df.columns
        assert "price" in df.columns

    def test_iot_multivariate_with_exogenous(self) -> None:
        gen = IoTSensorGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "n_sensors": 3,
                "exogenous": ExogenousConfig(datetime_features=True),
            }
        )
        df = gen.generate(n_series=2)
        assert "hour" in df.columns
        # 2 networks * 3 sensors = 6 series
        assert df["unique_id"].n_unique() == 6


# ---------------------------------------------------------------------------
# All-in-one integration
# ---------------------------------------------------------------------------
class TestFullIntegration:
    """Test combining all exogenous types at once."""

    def test_all_exogenous_types_together(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "changepoints": True,
                "missing_data": True,
                "exogenous": ExogenousConfig(
                    datetime_features=True,
                    datetime_cyclical=True,
                    anomaly_flags=True,
                    changepoint_flags=True,
                    missing_flags=True,
                    correlated=[
                        CorrelatedExogConfig(name="price", correlation=0.8),
                        CorrelatedExogConfig(
                            name="lagged_y", method="lagged_copy", lag=3
                        ),
                    ],
                ),
            }
        )
        df = gen.generate(n_series=3)

        # Core columns
        assert "unique_id" in df.columns
        assert "ds" in df.columns
        assert "y" in df.columns

        # Flag columns
        assert "anomaly_flag" in df.columns
        assert "changepoint_flag" in df.columns
        assert "missing_flag" in df.columns

        # Datetime columns (hourly)
        assert "year" in df.columns
        assert "hour" in df.columns
        assert "hour_sin" in df.columns

        # Correlated columns
        assert "price" in df.columns
        assert "lagged_y" in df.columns

        # Series count
        assert df["unique_id"].n_unique() == 3

    def test_pandas_backend(self) -> None:
        """Exogenous should work with pandas backend too."""
        import pandas as pd

        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "seed": 42,
                "engine": "pandas",
                "anomalies": True,
                "exogenous": ExogenousConfig(
                    anomaly_flags=True,
                    datetime_features=True,
                ),
            }
        )
        df = gen.generate(n_series=1)
        assert isinstance(df, pd.DataFrame)
        assert "anomaly_flag" in df.columns
        assert "year" in df.columns


# ---------------------------------------------------------------------------
# SynSet integration
# ---------------------------------------------------------------------------
class TestSynSetIntegration:
    """Test exogenous config with SynSet."""

    def test_synset_with_exogenous(self) -> None:
        exog = ExogenousConfig(datetime_features=True, anomaly_flags=True)
        gen1 = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "exogenous": exog,
            }
        )
        gen2 = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 43,
                "anomalies": True,
                "exogenous": exog,
            }
        )
        synset = SynSet(generators=[gen1, gen2])
        df = synset.generate(n_series_per_generator=2)
        assert "anomaly_flag" in df.columns
        assert "year" in df.columns

    def test_synset_mismatched_exogenous_no_crash(self) -> None:
        """SynSet with mismatched exogenous configs should not crash."""
        gen1 = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "exogenous": ExogenousConfig(datetime_features=True),
            }
        )
        gen2 = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 43,
                "exogenous": ExogenousConfig(anomaly_flags=True),
            }
        )
        # Construction should not crash even with mismatched configs
        synset = SynSet(generators=[gen1, gen2])
        assert synset is not None
