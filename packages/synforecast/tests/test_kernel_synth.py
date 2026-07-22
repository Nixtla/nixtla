"""Tests for KernelSynthGenerator."""

import numpy as np
import pytest

from synforecast.generators.kernel_synth import KernelSynthGenerator
from tests.helpers import assert_long_format, series_values

BASE = {"min_length": 64, "max_length": 128, "freq": "h", "seed": 42}


class TestKernelSynthApi:
    """API and structural tests."""

    def test_long_format(self, engine: str) -> None:
        gen = KernelSynthGenerator(**BASE, engine=engine)
        df = gen.generate(n_series=4)
        assert_long_format(df, n_series=4, min_length=64, max_length=128)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = KernelSynthGenerator(**BASE, engine=engine).generate(n_series=4)
        df2 = KernelSynthGenerator(**BASE, engine=engine).generate(n_series=4)
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_different_seeds_differ(self) -> None:
        v1 = KernelSynthGenerator(**{**BASE, "seed": 1}).generate_single_series(96)
        v2 = KernelSynthGenerator(**{**BASE, "seed": 2}).generate_single_series(96)
        assert not np.array_equal(v1, v2)

    def test_requested_length(self) -> None:
        for length in (1, 2, 10, 200):
            values = KernelSynthGenerator(**BASE).generate_single_series(length)
            assert values.shape == (length,)
            assert np.all(np.isfinite(values))


class TestKernelSynthBehavior:
    """Statistical / behavioral guarantees."""

    def test_finite_and_bounded_across_pool(self) -> None:
        gen = KernelSynthGenerator(**{**BASE, "seed": 0})
        for _ in range(64):
            values = gen.generate_single_series(128)
            assert np.all(np.isfinite(values))
            assert np.abs(values).max() < 1e8

    def test_standardized_by_default(self) -> None:
        gen = KernelSynthGenerator(**{**BASE, "seed": 0})
        for _ in range(20):
            values = gen.generate_single_series(256)
            assert abs(values.mean()) < 1e-6
            assert abs(values.std() - 1.0) < 1e-6

    def test_standardize_disabled_keeps_raw_scale(self) -> None:
        gen = KernelSynthGenerator(**{**BASE, "seed": 0}, standardize=False)
        # Raw GP draws should not all be unit-variance standardized.
        stds = [gen.generate_single_series(256).std() for _ in range(20)]
        assert any(abs(s - 1.0) > 1e-3 for s in stds)

    def test_pool_is_diverse(self) -> None:
        """Composed kernels should yield a spread of temporal roughness."""
        gen = KernelSynthGenerator(**{**BASE, "seed": 0})

        def roughness(v: np.ndarray) -> float:
            z = (v - v.mean()) / (v.std() + 1e-9)
            return float(np.std(np.diff(z)))

        rough = np.array(
            [roughness(gen.generate_single_series(256)) for _ in range(48)]
        )
        # Smooth RBF-dominated draws and white-noise-dominated draws both occur.
        assert rough.min() < 0.5
        assert rough.max() > 1.0

    def test_periodic_bank_can_produce_periodicity(self) -> None:
        """A periodic-only bank should concentrate power at the target period."""
        period = 24.0
        # Restrict the bank to a single periodic kernel.
        gen = KernelSynthGenerator(
            min_length=480,
            max_length=480,
            freq="h",
            seed=3,
            max_kernels=1,
            seasonal_periods=[period],
            rbf_length_scales=[],
            rational_quadratic_alphas=[],
            linear_sigmas=[],
            white_noise_levels=[],
            include_constant=False,
        )
        found = []
        for _ in range(20):
            v = gen.generate_single_series(480)
            v = v - v.mean()
            power = np.abs(np.fft.rfft(v))
            freqs = np.fft.rfftfreq(len(v))
            peak = np.argmax(power[1:]) + 1
            found.append(1.0 / freqs[peak])
        # The dominant period should land near 24 for most draws.
        near = np.abs(np.array(found) - period) < 4.0
        assert near.mean() > 0.5


class TestKernelSynthValidation:
    """Parameter validation."""

    def test_rejects_nonpositive_length_scale(self) -> None:
        with pytest.raises(ValueError):
            KernelSynthGenerator(**BASE, rbf_length_scales=[0.0])

    def test_rejects_short_period(self) -> None:
        with pytest.raises(ValueError):
            KernelSynthGenerator(**BASE, seasonal_periods=[1.0])

    def test_rejects_empty_bank(self) -> None:
        with pytest.raises(ValueError):
            KernelSynthGenerator(
                **BASE,
                rbf_length_scales=[],
                rational_quadratic_alphas=[],
                seasonal_periods=[],
                linear_sigmas=[],
                white_noise_levels=[],
                include_constant=False,
            )
