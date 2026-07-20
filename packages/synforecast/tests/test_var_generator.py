"""Tests for VARGenerator."""

import numpy as np
import pytest
from scipy import linalg

from synforecast.generators import VARGenerator
from tests.helpers import assert_long_format, series_values

# Stable VAR(1) used across statistical tests
A1 = np.array([[0.5, 0.2], [0.1, 0.4]])
INTERCEPT = [1.0, 2.0]
SIGMA = np.array([[1.0, 0.3], [0.3, 1.0]])


def make_gen(engine: str = "pandas", **kwargs) -> VARGenerator:
    params = {
        "min_length": 100,
        "max_length": 100,
        "freq": "D",
        "engine": engine,
        "seed": 42,
    }
    params.update(kwargs)
    return VARGenerator(**params)


class TestVARAPI:
    """Structural / API contract tests."""

    def test_long_format(self, engine: str) -> None:
        gen = make_gen(engine, min_length=40, max_length=60)
        df = gen.generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=40, max_length=60)

    def test_all_series_share_one_length(self, engine: str) -> None:
        """n_series correlated variables share a single sampled length."""
        gen = make_gen(engine, min_length=40, max_length=80)
        values = series_values(gen.generate(n_series=4))
        lengths = {len(v) for v in values.values()}
        assert len(lengths) == 1

    def test_start_id(self, engine: str) -> None:
        gen = make_gen(engine)
        values = series_values(gen.generate(n_series=2, start_id=10))
        assert set(values) == {"10", "11"}

    def test_seed_determinism(self, engine: str) -> None:
        df_a = make_gen(engine, seed=7).generate(n_series=3)
        df_b = make_gen(engine, seed=7).generate(n_series=3)
        values_a, values_b = series_values(df_a), series_values(df_b)
        assert set(values_a) == set(values_b)
        for uid in values_a:
            np.testing.assert_array_equal(values_a[uid], values_b[uid])
        df_c = make_gen(engine, seed=8).generate(n_series=3)
        assert not np.array_equal(series_values(df_c)["0"], values_a["0"]), (
            "different seeds should give different values"
        )

    def test_n_jobs_accepted(self, engine: str) -> None:
        df = make_gen(engine).generate(n_series=2, n_jobs=1)
        assert_long_format(df, n_series=2)

    def test_lag_order_two(self, engine: str) -> None:
        gen = make_gen(
            engine,
            lag_order=2,
            coef_matrices=[[[0.4, 0.1], [0.0, 0.3]], [[0.2, 0.0], [0.1, 0.2]]],
        )
        df = gen.generate(n_series=2)
        assert_long_format(df, n_series=2)

    def test_generate_single_series(self) -> None:
        values = make_gen().generate_single_series(50)
        assert values.shape == (50,)
        assert np.isfinite(values).all()

    def test_large_system_stays_stable(self) -> None:
        """Regression: randomly generated coefficients for many variables
        must be rescaled to stability (unscaled uniform(-0.3, 0.3) matrices
        have companion spectral radius > 1 for k ~ 40)."""
        gen = make_gen(min_length=300, max_length=300, seed=0)
        df = gen.generate(n_series=40)
        values = df["y"].to_numpy()
        assert np.isfinite(values).all()
        assert np.abs(values).max() < 1e3


class TestVARValidation:
    """Constructor validation of VAR parameters."""

    def test_coef_count_must_match_lag_order(self) -> None:
        with pytest.raises(ValueError, match="does not match lag_order"):
            make_gen(lag_order=2, coef_matrices=[[[0.5]]])

    def test_coef_matrices_must_be_square(self) -> None:
        with pytest.raises(ValueError, match="must be square"):
            make_gen(coef_matrices=[[[0.5, 0.2]]])

    def test_coef_matrices_must_have_same_size(self) -> None:
        with pytest.raises(ValueError, match="same size"):
            make_gen(
                lag_order=2,
                coef_matrices=[[[0.5]], [[0.2, 0.0], [0.0, 0.2]]],
            )

    def test_unstable_var_rejected(self) -> None:
        with pytest.raises(ValueError, match="unstable"):
            make_gen(coef_matrices=[[[1.05]]])

    def test_unstable_var2_rejected(self) -> None:
        # Each lag matrix is small, but the companion radius exceeds 1
        with pytest.raises(ValueError, match="unstable"):
            make_gen(lag_order=2, coef_matrices=[[[0.7]], [[0.7]]])

    def test_innovation_covariance_must_be_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            make_gen(innovation_covariance=[[1.0, 0.5]])

    def test_innovation_covariance_must_be_symmetric(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            make_gen(innovation_covariance=[[1.0, 0.9], [0.1, 1.0]])

    def test_innovation_covariance_must_be_positive_definite(self) -> None:
        with pytest.raises(ValueError, match="positive definite"):
            make_gen(innovation_covariance=[[1.0, 2.0], [2.0, 1.0]])


@pytest.mark.stats
class TestVARStats:
    """Statistical property tests (fixed seeds)."""

    def _simulate(self, n: int = 6000, seed: int = 123) -> np.ndarray:
        gen = make_gen(
            min_length=n,
            max_length=n,
            seed=seed,
            coef_matrices=[A1.tolist()],
            intercept=INTERCEPT,
            innovation_covariance=SIGMA.tolist(),
        )
        values = series_values(gen.generate(n_series=2))
        return np.column_stack([values["0"], values["1"]])

    def test_stationary_mean(self) -> None:
        """Sample means match mu = (I - A)^-1 c within z * sqrt(lrv / n)."""
        y = self._simulate()
        n = y.shape[0]
        mu = np.linalg.solve(np.eye(2) - A1, np.array(INTERCEPT))
        # Long-run variance of the sample mean of a VAR:
        # (I - A)^-1 Sigma (I - A)^-T
        lrv = np.linalg.solve(np.eye(2) - A1, SIGMA) @ np.linalg.inv(np.eye(2) - A1).T
        for i in range(2):
            tol = 5.0 * np.sqrt(lrv[i, i] / n)
            err = abs(y[:, i].mean() - mu[i])
            assert err < tol, f"mean[{i}] {y[:, i].mean():.3f} != {mu[i]:.3f}"

    def test_stationary_covariance(self) -> None:
        """Sample covariance matches the discrete Lyapunov solution
        Gamma0 = A Gamma0 A' + Sigma."""
        y = self._simulate()
        gamma0 = linalg.solve_discrete_lyapunov(A1, SIGMA)
        sample_cov = np.cov(y.T)
        np.testing.assert_allclose(sample_cov, gamma0, atol=0.15)

    def test_lag_one_cross_covariance(self) -> None:
        """Lag-1 cross-covariance matches Gamma1 = A Gamma0."""
        y = self._simulate()
        gamma0 = linalg.solve_discrete_lyapunov(A1, SIGMA)
        gamma1 = A1 @ gamma0
        yc = y - y.mean(axis=0)
        sample_gamma1 = yc[1:].T @ yc[:-1] / (len(y) - 1)
        np.testing.assert_allclose(sample_gamma1, gamma1, atol=0.15)

    def test_coefficient_and_innovation_recovery(self) -> None:
        """OLS on the simulated data recovers A, c, and the residual
        covariance recovers Sigma."""
        y = self._simulate()
        n = y.shape[0]
        X = np.column_stack([np.ones(n - 1), y[:-1]])
        beta, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
        c_hat = beta[0]
        A_hat = beta[1:].T
        np.testing.assert_allclose(A_hat, A1, atol=0.06)
        np.testing.assert_allclose(c_hat, np.array(INTERCEPT), atol=0.25)

        residuals = y[1:] - X @ beta
        sigma_hat = residuals.T @ residuals / (n - 1)
        np.testing.assert_allclose(sigma_hat, SIGMA, atol=0.1)
