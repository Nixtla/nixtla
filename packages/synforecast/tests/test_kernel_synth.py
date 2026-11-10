"""Tests for KernelSynthGenerator."""

import numpy as np
import pytest

from synforecast._lib import multivariate as _rs_mv
from synforecast.base import _GEN_TYPE_MAP
from synforecast.generators.kernel_synth import KernelSynthGenerator
from tests.helpers import assert_long_format, series_values

BASE = {"min_length": 64, "max_length": 128, "freq": "h", "seed": 42}


def _lag1_autocorrelation(values: np.ndarray) -> float:
    """Near 1 for a smooth path, near 0 for the white-noise fallback."""
    z = values - values.mean()
    return float(z[:-1] @ z[1:] / (z @ z))


_SMOOTH_BANK = {
    "seasonal_periods": [],
    "rbf_length_scales": [10.0],
    "rational_quadratic_alphas": [],
    "linear_sigmas": [10.0],
    "white_noise_levels": [],
    "include_constant": False,
}


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

    @pytest.mark.parametrize("native", [False, True])
    def test_periodic_bank_can_produce_periodicity(self, native: bool) -> None:
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
        draws = (
            list(series_values(gen.generate(20, n_jobs=1)).values())
            if native
            else [gen.generate_single_series(480) for _ in range(20)]
        )
        found = []
        for v in draws:
            v = v - v.mean()
            power = np.abs(np.fft.rfft(v))
            freqs = np.fft.rfftfreq(len(v))
            peak = np.argmax(power[1:]) + 1
            found.append(1.0 / freqs[peak])
        # The dominant period should land near 24 for most draws.
        near = np.abs(np.array(found) - period) < 4.0
        assert near.mean() > 0.5


class TestKernelSynthSampler:
    """Sampler behavior through the public API (exactness is tested in Rust)."""

    def test_single_series_runs_the_rust_sampler(self) -> None:
        gen = KernelSynthGenerator(**BASE)
        values = gen.generate_single_series(100)
        scalars, (kinds, params) = gen._get_batch_params()
        seed = int(np.random.default_rng(BASE["seed"]).integers(0, 2**63))
        np.testing.assert_array_equal(
            values, _rs_mv.kernel_synth(100, scalars, kinds, params, seed)
        )

    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("jitter", [1e-6, 0.0])
    def test_smooth_bank_does_not_fall_back_to_noise(
        self, native: bool, jitter: float
    ) -> None:
        """Near-singular smooth kernels still sample, with or without jitter."""
        gen = KernelSynthGenerator(
            min_length=1000,
            max_length=1000,
            freq="h",
            seed=7,
            max_kernels=3,
            jitter=jitter,
            standardize=False,
            **_SMOOTH_BANK,
        )
        draws = (
            list(series_values(gen.generate(10, n_jobs=1)).values())
            if native
            else [gen.generate_single_series(1000) for _ in range(10)]
        )
        for values in draws:
            assert _lag1_autocorrelation(values) > 0.5

    @pytest.mark.parametrize("native", [False, True])
    def test_linear_only_bank_is_a_random_line(self, native: bool) -> None:
        """``s^2 + x x'`` has rank two: every draw is affine in the grid."""
        gen = KernelSynthGenerator(
            **{**BASE, "seed": 1, "min_length": 50, "max_length": 50},
            max_kernels=1,
            jitter=0.0,
            seasonal_periods=[],
            rbf_length_scales=[],
            rational_quadratic_alphas=[],
            linear_sigmas=[1.0],
            white_noise_levels=[],
            include_constant=False,
            standardize=False,
        )
        x = np.linspace(0.0, 1.0, 50)
        draws = (
            list(series_values(gen.generate(5, n_jobs=1)).values())
            if native
            else [gen.generate_single_series(50) for _ in range(5)]
        )
        for values in draws:
            slope, intercept = np.polyfit(x, values, 1)
            np.testing.assert_allclose(values, intercept + slope * x, atol=1e-10)


class TestKernelSynthNative:
    """The Rust batch path."""

    def test_native_batch_parameter_contract(self) -> None:
        gen = KernelSynthGenerator(
            **BASE,
            max_kernels=3,
            jitter=1e-5,
            standardize=False,
            seasonal_periods=[12.0],
            rbf_length_scales=[0.5],
            rational_quadratic_alphas=[2.0],
            linear_sigmas=[1.0],
            white_noise_levels=[0.1],
        )
        assert gen._batch_gen_type == _GEN_TYPE_MAP["KernelSynthGenerator"] == 31
        scalars, arrays = gen._get_batch_params()
        np.testing.assert_array_equal(scalars, [3.0, 1e-5, 0.0])
        np.testing.assert_array_equal(arrays[0], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(arrays[1], [0.5, 2.0, 12.0, 1.0, 0.1, 1.0])

    def test_native_batch_is_independent_of_worker_count(self) -> None:
        first = series_values(KernelSynthGenerator(**BASE).generate(8, n_jobs=1))
        second = series_values(KernelSynthGenerator(**BASE).generate(8, n_jobs=4))
        assert first.keys() == second.keys()
        for series_id in first:
            np.testing.assert_array_equal(first[series_id], second[series_id])

    def test_native_finite_and_standardized(self) -> None:
        gen = KernelSynthGenerator(**{**BASE, "min_length": 256, "max_length": 256})
        for values in series_values(gen.generate(64, n_jobs=1)).values():
            assert np.all(np.isfinite(values))
            assert abs(values.mean()) < 1e-6
            assert abs(values.std() - 1.0) < 1e-6

    def test_native_standardize_disabled_keeps_raw_scale(self) -> None:
        gen = KernelSynthGenerator(
            **{**BASE, "min_length": 256, "max_length": 256, "seed": 0},
            standardize=False,
        )
        stds = [v.std() for v in series_values(gen.generate(20, n_jobs=1)).values()]
        assert any(abs(s - 1.0) > 1e-3 for s in stds)

    def test_native_pool_is_diverse(self) -> None:
        gen = KernelSynthGenerator(
            **{**BASE, "min_length": 256, "max_length": 256, "seed": 0}
        )
        rough = np.array(
            [
                np.std(np.diff(v))
                for v in series_values(gen.generate(48, n_jobs=1)).values()
            ]
        )
        assert rough.min() < 0.5
        assert rough.max() > 1.0


class TestKernelSynthValidation:
    """Parameter validation."""

    def test_rejects_nonpositive_length_scale(self) -> None:
        with pytest.raises(ValueError):
            KernelSynthGenerator(**BASE, rbf_length_scales=[0.0])

    @pytest.mark.parametrize("max_kernels", [0, 65])
    def test_rejects_max_kernels_out_of_range(self, max_kernels: int) -> None:
        with pytest.raises(ValueError):
            KernelSynthGenerator(**BASE, max_kernels=max_kernels)

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
