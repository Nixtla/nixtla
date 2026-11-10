"""Tests for FractionalBrownianMotionGenerator."""

import numpy as np
import pytest

from synforecast.generators import FractionalBrownianMotionGenerator
from tests.helpers import (
    assert_acf,
    assert_long_format,
    assert_std,
    sample_acf,
    series_values,
)

BASE = {"min_length": 30, "max_length": 60, "freq": "D", "seed": 42}


def theoretical_acf1(hurst: float) -> float:
    """Lag-1 autocorrelation of fGn: gamma(1)/gamma(0) = (2^{2H} - 2) / 2."""
    return 0.5 * (2 ** (2 * hurst) - 2)


class TestFBMApi:
    """API and structural tests."""

    def test_long_format(self, engine: str) -> None:
        gen = FractionalBrownianMotionGenerator(**BASE, engine=engine)
        df = gen.generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=30, max_length=60)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = FractionalBrownianMotionGenerator(**BASE, engine=engine).generate(
            n_series=3
        )
        df2 = FractionalBrownianMotionGenerator(**BASE, engine=engine).generate(
            n_series=3
        )
        v1, v2 = series_values(df1), series_values(df2)
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_different_seeds_differ(self) -> None:
        v1 = FractionalBrownianMotionGenerator(
            **{**BASE, "seed": 1}
        ).generate_single_series(50)
        v2 = FractionalBrownianMotionGenerator(
            **{**BASE, "seed": 2}
        ).generate_single_series(50)
        assert not np.array_equal(v1, v2)

    @pytest.mark.parametrize("hurst", [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_hurst(self, hurst: float) -> None:
        with pytest.raises(ValueError):
            FractionalBrownianMotionGenerator(**BASE, hurst=hurst)

    def test_invalid_sigma(self) -> None:
        with pytest.raises(ValueError):
            FractionalBrownianMotionGenerator(**BASE, sigma=0.0)

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            FractionalBrownianMotionGenerator(**BASE, method="wavelet")

    def test_initial_value_offsets_path(self) -> None:
        gen = FractionalBrownianMotionGenerator(**BASE, initial_value=100.0)
        values = gen.generate_single_series(50)
        # First value is initial_value plus one N(0, sigma^2) increment
        assert abs(values[0] - 100.0) < 6.0

    def test_increments_ignore_initial_value(self) -> None:
        """fGn increments must not be offset by initial_value."""
        gen = FractionalBrownianMotionGenerator(
            **BASE, initial_value=100.0, return_increments=True
        )
        values = gen.generate_single_series(50)
        assert np.abs(values).max() < 50.0

    @pytest.mark.parametrize("method", ["fft", "cholesky", "hosking"])
    def test_increments_identical_regardless_of_initial_value(
        self, method: str
    ) -> None:
        """Same seed => identical increments whatever initial_value is; the
        kernel only applies initial_value in cumulative (path) mode."""
        common = {**BASE, "method": method, "return_increments": True}
        with_iv = FractionalBrownianMotionGenerator(
            **common, initial_value=100.0
        ).generate_single_series(64)
        without_iv = FractionalBrownianMotionGenerator(
            **common, initial_value=0.0
        ).generate_single_series(64)
        np.testing.assert_array_equal(with_iv, without_iv)

    def test_first_increment_distribution_consistent(self) -> None:
        """The first increment is drawn from the same N(0, sigma^2) as the
        rest (no initial_value leak on the Rust path)."""
        gen = FractionalBrownianMotionGenerator(
            **BASE, hurst=0.5, initial_value=100.0, return_increments=True
        )
        first = np.array([gen.generate_single_series(16)[0] for _ in range(500)])
        # sigma=1: mean ~ 0 within 6 SE, std ~ 1
        assert abs(first.mean()) < 6.0 / np.sqrt(500)
        assert abs(first.std() - 1.0) < 0.15

    def test_get_model_info(self) -> None:
        info = FractionalBrownianMotionGenerator(**BASE, hurst=0.8).get_model_info()
        assert info["hurst_exponent"] == 0.8
        assert "persistent" in info["behavior"]
        info_bm = FractionalBrownianMotionGenerator(**BASE, hurst=0.5).get_model_info()
        assert info_bm["long_range_dependence_exponent"] == 0.0

    def test_estimate_hurst_unknown_method(self) -> None:
        gen = FractionalBrownianMotionGenerator(**BASE)
        with pytest.raises(ValueError, match="Unknown method"):
            gen.estimate_hurst(np.arange(100.0), method="bogus")


@pytest.mark.stats
class TestFBMStats:
    """Statistical property tests (fixed seeds)."""

    @pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
    def test_increment_acf_matches_theory(self, hurst: float) -> None:
        gen = FractionalBrownianMotionGenerator(
            min_length=4096,
            max_length=4096,
            freq="D",
            hurst=hurst,
            return_increments=True,
            seed=7,
        )
        values = gen.generate_single_series(4096)
        assert_acf(values, lag=1, expected=theoretical_acf1(hurst))

    @pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
    def test_estimated_hurst_matches_parameter(self, hurst: float) -> None:
        gen = FractionalBrownianMotionGenerator(
            min_length=2048,
            max_length=2048,
            freq="D",
            hurst=hurst,
            return_increments=True,
            seed=11,
        )
        estimates = [
            gen.estimate_hurst(gen.generate_single_series(2048), method="var")
            for _ in range(4)
        ]
        assert abs(float(np.mean(estimates)) - hurst) < 0.1

    def test_h05_reduces_to_brownian_motion(self) -> None:
        """H=0.5: iid N(0, sigma^2) increments and Var(B_k) = sigma^2 * k."""
        gen = FractionalBrownianMotionGenerator(
            min_length=64,
            max_length=64,
            freq="D",
            hurst=0.5,
            sigma=1.5,
            return_increments=True,
            seed=3,
        )
        increments = np.array([gen.generate_single_series(64) for _ in range(200)])
        assert_std(increments.ravel(), expected=1.5)
        assert_acf(increments.ravel(), lag=1, expected=0.0)

        gen_path = FractionalBrownianMotionGenerator(
            min_length=64, max_length=64, freq="D", hurst=0.5, sigma=1.5, seed=4
        )
        paths = np.array([gen_path.generate_single_series(64) for _ in range(300)])
        assert_std(paths[:, 63], expected=1.5 * np.sqrt(64.0))

    def test_path_variance_scaling(self) -> None:
        """Var(B_H(k)) = sigma^2 * k^{2H}: check the k=1 -> k=16 variance ratio."""
        for hurst in (0.3, 0.7):
            gen = FractionalBrownianMotionGenerator(
                min_length=16, max_length=16, freq="D", hurst=hurst, seed=13
            )
            paths = np.array([gen.generate_single_series(16) for _ in range(500)])
            ratio = paths[:, 15].var() / paths[:, 0].var()
            estimated_h = np.log(ratio) / (2 * np.log(16.0))
            assert abs(estimated_h - hurst) < 0.1

    @pytest.mark.parametrize("method,hurst", [("cholesky", 0.7), ("hosking", 0.3)])
    def test_methods_match_theory(self, method: str, hurst: float) -> None:
        """Native cholesky and Hosking paths reproduce the fGn covariance."""
        gen = FractionalBrownianMotionGenerator(
            min_length=256,
            max_length=256,
            freq="D",
            hurst=hurst,
            method=method,
            return_increments=True,
            seed=17,
        )
        vals = [gen.generate_single_series(256) for _ in range(20)]
        acf1 = float(np.mean([sample_acf(v, 1) for v in vals]))
        assert abs(acf1 - theoretical_acf1(hurst)) < 0.06
        assert_std(np.concatenate(vals), expected=1.0)
