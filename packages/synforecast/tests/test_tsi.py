"""Tests for TSIGenerator."""

import numpy as np
import pytest
from scipy import signal

from synforecast.base import _GEN_TYPE_MAP
from synforecast.generators.tsi import TSIGenerator
from tests.helpers import assert_acf, assert_long_format, series_values

BASE = {"min_length": 30, "max_length": 60, "freq": "D", "seed": 42}


def pool_metrics(values: np.ndarray) -> tuple[float, float, float]:
    """(roughness, lag-12 acf, spectral entropy), computed exactly like
    tsfm's diversity audit on the standardized series."""
    z = (values - values.mean()) / (values.std() + 1e-9)
    roughness = np.std(np.diff(z)) / (np.std(z) + 1e-9)
    _, p = signal.periodogram(z)
    p = p[1:]  # drop the zero-frequency bin
    p = p / p.sum()
    entropy = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
    demeaned = z - z.mean()
    acf12 = float(demeaned[:-12] @ demeaned[12:] / (demeaned @ demeaned))
    return roughness, acf12, entropy


def dominant_period(values: np.ndarray) -> float:
    """Period of the largest FFT peak after linear detrending."""
    t = np.arange(len(values))
    detrended = values - np.polyval(np.polyfit(t, values, 1), t)
    power = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(len(values))
    peak = np.argmax(power[1:]) + 1
    return 1.0 / freqs[peak]


class TestTSIApi:
    """API and structural tests."""

    def test_long_format(self, engine: str) -> None:
        gen = TSIGenerator(**BASE, engine=engine)
        df = gen.generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=30, max_length=60)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = TSIGenerator(**BASE, engine=engine).generate(n_series=3)
        df2 = TSIGenerator(**BASE, engine=engine).generate(n_series=3)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_different_seeds_differ(self) -> None:
        v1 = TSIGenerator(**{**BASE, "seed": 1}).generate_single_series(50)
        v2 = TSIGenerator(**{**BASE, "seed": 2}).generate_single_series(50)
        assert not np.array_equal(v1, v2)

    def test_series_within_generate_differ(self) -> None:
        """Each series draws its own random configuration."""
        gen = TSIGenerator(min_length=50, max_length=50, freq="D", seed=0)
        values = series_values(gen.generate(n_series=4))
        arrays = list(values.values())
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                assert not np.array_equal(arrays[i], arrays[j])

    @pytest.mark.parametrize(
        "field,value",
        [
            ("trend_types", []),
            ("trend_types", ["bogus"]),
            ("irregular_types", []),
            ("irregular_types", ["bogus"]),
            ("seasonal_periods", []),
            ("seasonal_periods", [0.5]),
            ("n_seasonal_range", (2, 1)),
            ("n_seasonal_range", (-1, 2)),
            ("n_breakpoints_range", (0, 2)),
            ("noise_scale_range", (0.0, 1.0)),
            ("noise_scale_range", (5.0, 1.0)),
            ("seasonal_amplitude_range", (-1.0, 1.0)),
            ("ar1_phi_range", (0.5, 1.5)),
            ("tail_df_range", (1.5, 5.0)),
            ("scale_range", (0.0, 10.0)),
            ("multiplicative_prob", 1.5),
            ("harmonics_prob", -0.1),
        ],
    )
    def test_invalid_parameters(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            TSIGenerator(**BASE, **{field: value})


@pytest.mark.stats
class TestTSIComponents:
    """Forcing a single configuration via narrow ranges reproduces it."""

    def test_linear_trend_with_daily_seasonality(self) -> None:
        """linear trend + one period-24 harmonic + tiny gaussian noise ->
        positive fitted slope and FFT peak at period 24."""
        for seed in range(5):
            gen = TSIGenerator(
                min_length=480,
                max_length=480,
                freq="h",
                trend_types=["linear"],
                trend_slope_range=(4.0, 8.0),
                level_range=(0.0, 0.0),
                n_seasonal_range=(1, 1),
                seasonal_periods=[24.0],
                seasonal_amplitude_range=(2.0, 2.0),
                amplitude_modulation_prob=0.0,
                harmonics_prob=0.0,
                irregular_types=["gaussian"],
                noise_scale_range=(0.01, 0.01),
                multiplicative_prob=0.0,
                scale_range=(1.0, 1.0),
                seed=seed,
            )
            values = gen.generate_single_series(480)
            slope = np.polyfit(np.arange(480), values, 1)[0]
            # total movement in (4, 8) over 480 steps
            assert 3.0 / 480 < slope < 9.0 / 480, f"seed {seed}: slope {slope}"
            period = dominant_period(values)
            assert 22.0 < period < 26.0, f"seed {seed}: period {period}"

    def test_ar1_only_lag1_acf_matches_phi(self) -> None:
        """ar1-only configuration reproduces the AR(1) lag-1 autocorrelation."""
        for seed in range(3):
            gen = TSIGenerator(
                min_length=3000,
                max_length=3000,
                freq="D",
                trend_types=["none"],
                level_range=(0.0, 0.0),
                n_seasonal_range=(0, 0),
                irregular_types=["ar1"],
                ar1_phi_range=(0.7, 0.7),
                noise_scale_range=(1.0, 1.0),
                multiplicative_prob=0.0,
                scale_range=(1.0, 1.0),
                seed=seed,
            )
            values = gen.generate_single_series(3000)
            assert_acf(values, lag=1, expected=0.7)

    def test_multiplicative_composition_stays_positive_base(self) -> None:
        """Forced multiplicative composition with tiny noise stays finite and
        non-degenerate even when the trend crosses zero."""
        gen = TSIGenerator(
            min_length=200,
            max_length=200,
            freq="D",
            trend_types=["linear"],
            trend_slope_range=(-8.0, 8.0),
            level_range=(-10.0, 10.0),
            n_seasonal_range=(1, 2),
            multiplicative_prob=1.0,
            noise_scale_range=(0.01, 0.01),
            seed=7,
        )
        for _ in range(20):
            values = gen.generate_single_series(200)
            assert np.isfinite(values).all()
            assert values.std() > 1e-8


@pytest.mark.stats
class TestTSIPoolDiversity:
    """Default-parameter pool matches the real-data diversity audit targets
    (real: roughness 1.24, |lag-12 acf| 0.26, spectral entropy 0.88)."""

    def test_pool_diversity_metrics(self) -> None:
        gen = TSIGenerator(min_length=512, max_length=512, freq="D", seed=0)
        rows = np.array(
            [pool_metrics(gen.generate_single_series(512)) for _ in range(300)]
        )
        roughness = rows[:, 0]
        abs_acf12 = np.abs(rows[:, 1])
        entropy = rows[:, 2]

        assert np.median(roughness) >= 1.0, f"roughness {np.median(roughness):.3f}"
        assert np.median(abs_acf12) <= 0.5, f"|acf12| {np.median(abs_acf12):.3f}"
        assert np.median(entropy) >= 0.8, f"entropy {np.median(entropy):.3f}"

        # Spread: the pool must span structured through noise-dominated
        spread = np.percentile(entropy, 90) - np.percentile(entropy, 10)
        assert spread >= 0.2, f"entropy p90-p10 {spread:.3f}"
        structured = np.mean(entropy <= 0.7)
        assert structured >= 0.1, f"structured fraction {structured:.3f}"
        noisy = np.mean(entropy >= 0.9)
        assert noisy >= 0.15, f"noise-dominated fraction {noisy:.3f}"
        assert np.mean(roughness > 1.2) >= 0.3  # rough, noise-dominated series
        assert np.mean(roughness < 0.7) >= 0.05  # smooth, trend/season-dominated


class TestTSIRustBatch:
    """Rust batch path (used by generate() when _lib is installed).

    RNG streams differ from numpy, so parity with the Python reference is
    statistical, not bitwise.
    """

    def test_batch_path_is_wired(self) -> None:
        gen = TSIGenerator(**BASE)
        assert gen._batch_gen_type == _GEN_TYPE_MAP["TSIGenerator"] == 28
        scalars, arrays = gen._get_batch_params()
        assert scalars.shape == (23,)
        # [trend type ids, seasonal periods, irregular type ids]
        assert [len(a) for a in arrays] == [6, 12, 5]

    def test_generate_seed_determinism(self) -> None:
        df1 = TSIGenerator(**BASE).generate(n_series=5)
        df2 = TSIGenerator(**BASE).generate(n_series=5)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_n_jobs_invariance(self) -> None:
        df1 = TSIGenerator(**BASE).generate(n_series=8, n_jobs=1)
        df2 = TSIGenerator(**BASE).generate(n_series=8, n_jobs=4)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_batch_output_guards(self) -> None:
        """Finite, |y| < 1e8 and non-constant over many random configs."""
        gen = TSIGenerator(min_length=5, max_length=700, freq="D", seed=123)
        for values in series_values(gen.generate(n_series=150)).values():
            assert np.isfinite(values).all()
            assert np.abs(values).max() < 1e8
            assert values.std() > 1e-8

    def test_extreme_configuration_is_guarded(self) -> None:
        gen = TSIGenerator(
            min_length=100,
            max_length=100,
            freq="D",
            multiplicative_prob=1.0,
            noise_scale_range=(0.01, 1000.0),
            scale_range=(0.001, 10000.0),
            tail_df_range=(2.1, 3.0),
            seed=99,
        )
        for values in series_values(gen.generate(n_series=100)).values():
            assert np.isfinite(values).all()
            assert np.abs(values).max() < 1e8
            assert values.std() > 1e-8


@pytest.mark.stats
class TestTSIRustBatchStats:
    """Statistical parity of the Rust batch path with the Python reference."""

    def test_pool_diversity_metrics(self) -> None:
        """Rust-generated pool meets the same diversity targets as the
        Python pool test above."""
        gen = TSIGenerator(min_length=512, max_length=512, freq="D", seed=0)
        pool = series_values(gen.generate(n_series=300))
        rows = np.array([pool_metrics(v) for v in pool.values()])
        roughness = rows[:, 0]
        abs_acf12 = np.abs(rows[:, 1])
        entropy = rows[:, 2]

        assert np.median(roughness) >= 1.0, f"roughness {np.median(roughness):.3f}"
        assert np.median(abs_acf12) <= 0.5, f"|acf12| {np.median(abs_acf12):.3f}"
        assert np.median(entropy) >= 0.8, f"entropy {np.median(entropy):.3f}"

        # Spread: the pool must span structured through noise-dominated
        spread = np.percentile(entropy, 90) - np.percentile(entropy, 10)
        assert spread >= 0.2, f"entropy p90-p10 {spread:.3f}"
        structured = np.mean(entropy <= 0.7)
        assert structured >= 0.1, f"structured fraction {structured:.3f}"
        noisy = np.mean(entropy >= 0.9)
        assert noisy >= 0.15, f"noise-dominated fraction {noisy:.3f}"
        assert np.mean(roughness > 1.2) >= 0.3  # rough, noise-dominated series
        assert np.mean(roughness < 0.7) >= 0.05  # smooth, trend/season-dominated

    def test_forced_linear_trend_with_daily_seasonality(self) -> None:
        """Forced single-config draw through the batch path: positive
        fitted slope and FFT peak at period 24 (mirrors the Python
        component test)."""
        gen = TSIGenerator(
            min_length=480,
            max_length=480,
            freq="h",
            trend_types=["linear"],
            trend_slope_range=(4.0, 8.0),
            level_range=(0.0, 0.0),
            n_seasonal_range=(1, 1),
            seasonal_periods=[24.0],
            seasonal_amplitude_range=(2.0, 2.0),
            amplitude_modulation_prob=0.0,
            harmonics_prob=0.0,
            irregular_types=["gaussian"],
            noise_scale_range=(0.01, 0.01),
            multiplicative_prob=0.0,
            scale_range=(1.0, 1.0),
            seed=0,
        )
        for uid, values in series_values(gen.generate(n_series=5)).items():
            slope = np.polyfit(np.arange(480), values, 1)[0]
            # total movement in (4, 8) over 480 steps
            assert 3.0 / 480 < slope < 9.0 / 480, f"series {uid}: slope {slope}"
            period = dominant_period(values)
            assert 22.0 < period < 26.0, f"series {uid}: period {period}"

    def test_forced_ar1_lag1_acf_matches_phi(self) -> None:
        """ar1-only configuration through the batch path reproduces the
        AR(1) lag-1 autocorrelation."""
        gen = TSIGenerator(
            min_length=3000,
            max_length=3000,
            freq="D",
            trend_types=["none"],
            level_range=(0.0, 0.0),
            n_seasonal_range=(0, 0),
            irregular_types=["ar1"],
            ar1_phi_range=(0.7, 0.7),
            noise_scale_range=(1.0, 1.0),
            multiplicative_prob=0.0,
            scale_range=(1.0, 1.0),
            seed=0,
        )
        for values in series_values(gen.generate(n_series=3)).values():
            assert_acf(values, lag=1, expected=0.7)


class TestTSIGuards:
    """Output guards over a large sample of random configurations."""

    def test_default_pool_is_well_behaved(self) -> None:
        gen = TSIGenerator(min_length=5, max_length=700, freq="D", seed=123)
        for _ in range(150):
            length = int(gen.rng.integers(5, 700))
            values = gen.generate_single_series(length)
            assert values.shape == (length,)
            assert np.isfinite(values).all()
            assert np.abs(values).max() < 1e8
            assert values.std() > 1e-8

    def test_extreme_configuration_is_guarded(self) -> None:
        gen = TSIGenerator(
            min_length=100,
            max_length=100,
            freq="D",
            multiplicative_prob=1.0,
            noise_scale_range=(0.01, 1000.0),
            scale_range=(0.001, 10000.0),
            tail_df_range=(2.1, 3.0),
            seed=99,
        )
        for _ in range(100):
            values = gen.generate_single_series(100)
            assert np.isfinite(values).all()
            assert np.abs(values).max() < 1e8
            assert values.std() > 1e-8
