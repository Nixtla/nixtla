"""Tests for native distribution functions against SciPy references."""

import numpy as np
import pytest
from scipy import stats

from synforecast._distributions import (
    expon_ppf,
    gamma_ppf,
    lognorm_ppf,
    norm_cdf,
    norm_ppf,
    t_cdf,
    uniform_ppf,
)

X_GRID = np.linspace(-4.0, 4.0, 81)
U_GRID = np.linspace(0.001, 0.999, 99)


@pytest.fixture
def dist_path():
    """Mark tests that exercise the native distribution implementation."""


class TestNormCdf:
    @pytest.mark.usefixtures("dist_path")
    def test_matches_scipy(self):
        np.testing.assert_allclose(norm_cdf(X_GRID), stats.norm.cdf(X_GRID), atol=1e-6)

    @pytest.mark.usefixtures("dist_path")
    def test_loc_scale(self):
        np.testing.assert_allclose(
            norm_cdf(X_GRID, loc=1.0, scale=2.0),
            stats.norm.cdf(X_GRID, loc=1.0, scale=2.0),
            atol=1e-6,
        )

    @pytest.mark.usefixtures("dist_path")
    def test_monotonic_and_bounded(self):
        result = norm_cdf(np.linspace(-10, 10, 201))
        assert np.all(np.diff(result) >= 0)
        assert result[0] < 1e-10
        assert result[-1] > 1 - 1e-10


class TestNormPpf:
    @pytest.mark.usefixtures("dist_path")
    def test_matches_scipy(self):
        np.testing.assert_allclose(norm_ppf(U_GRID), stats.norm.ppf(U_GRID), atol=1e-6)

    @pytest.mark.usefixtures("dist_path")
    def test_roundtrip(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(norm_ppf(norm_cdf(x)), x, atol=1e-4)


class TestTCdf:
    @pytest.mark.parametrize("df", [3.0, 5.0, 10.0])
    @pytest.mark.usefixtures("dist_path")
    def test_matches_scipy(self, df):
        np.testing.assert_allclose(
            t_cdf(X_GRID, df=df), stats.t.cdf(X_GRID, df), atol=1e-6
        )

    @pytest.mark.usefixtures("dist_path")
    def test_symmetry(self):
        result = t_cdf(np.array([-2.0, 0.0, 2.0]), df=5.0)
        assert abs(result[1] - 0.5) < 1e-6
        np.testing.assert_allclose(result[0], 1.0 - result[2], atol=1e-6)

    @pytest.mark.usefixtures("dist_path")
    def test_heavier_tails_than_normal(self):
        x = np.array([3.0])
        assert t_cdf(x, df=3.0)[0] < norm_cdf(x)[0]


class TestGammaPpf:
    @pytest.mark.parametrize("shape,scale", [(1.0, 1.0), (2.0, 1.5), (5.0, 0.5)])
    @pytest.mark.usefixtures("dist_path")
    def test_matches_scipy(self, shape, scale):
        np.testing.assert_allclose(
            gamma_ppf(U_GRID, shape=shape, scale=scale),
            stats.gamma.ppf(U_GRID, shape, scale=scale),
            rtol=1e-6,
            atol=1e-8,
        )

    @pytest.mark.usefixtures("dist_path")
    def test_positive_and_monotonic(self):
        result = gamma_ppf(U_GRID, shape=2.0, scale=1.0)
        assert np.all(result > 0)
        assert np.all(np.diff(result) > 0)


class TestExponPpf:
    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
    @pytest.mark.usefixtures("dist_path")
    def test_matches_scipy(self, scale):
        np.testing.assert_allclose(
            expon_ppf(U_GRID, scale=scale),
            stats.expon.ppf(U_GRID, scale=scale),
            rtol=1e-8,
        )

    @pytest.mark.usefixtures("dist_path")
    def test_median(self):
        np.testing.assert_allclose(
            expon_ppf(np.array([0.5]), scale=1.0), [np.log(2)], atol=1e-6
        )


class TestLognormPpf:
    @pytest.mark.parametrize("s", [0.25, 0.5, 1.0])
    @pytest.mark.usefixtures("dist_path")
    def test_matches_scipy(self, s):
        np.testing.assert_allclose(
            lognorm_ppf(U_GRID, s=s, scale=1.0),
            stats.lognorm.ppf(U_GRID, s),
            rtol=1e-5,
        )

    @pytest.mark.usefixtures("dist_path")
    def test_positive_and_monotonic(self):
        result = lognorm_ppf(U_GRID, s=0.5, scale=1.0)
        assert np.all(result > 0)
        assert np.all(np.diff(result) > 0)


class TestUniformPpf:
    @pytest.mark.usefixtures("dist_path")
    def test_matches_scipy(self):
        np.testing.assert_allclose(
            uniform_ppf(U_GRID, loc=10.0, scale=5.0),
            stats.uniform.ppf(U_GRID, loc=10.0, scale=5.0),
            atol=1e-12,
        )

    @pytest.mark.usefixtures("dist_path")
    def test_endpoints(self):
        u = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            uniform_ppf(u, loc=0.0, scale=1.0), [0.0, 0.5, 1.0], atol=1e-12
        )
