"""Tests for CopulaGenerator."""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import CopulaGenerator
from tests.helpers import assert_distribution, assert_long_format, series_values


def make_gen(engine: str = "pandas", **kwargs) -> CopulaGenerator:
    params = {
        "min_length": 100,
        "max_length": 100,
        "freq": "D",
        "engine": engine,
        "seed": 42,
    }
    params.update(kwargs)
    return CopulaGenerator(**params)


class TestCopulaAPI:
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
        values = series_values(gen.generate(n_series=2, start_id=5))
        assert set(values) == {"5", "6"}

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

    def test_t_copula(self, engine: str) -> None:
        gen = make_gen(engine, copula_type="t", df=4.0)
        df = gen.generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=100, max_length=100)

    def test_correlation_matrix_smaller_than_n_series(self, engine: str) -> None:
        """A 2x2 matrix is padded with an identity block for extra series."""
        gen = make_gen(engine, correlation_matrix=[[1.0, 0.5], [0.5, 1.0]])
        df = gen.generate(n_series=4)
        assert_long_format(df, n_series=4)

    def test_generate_single_series(self) -> None:
        values = make_gen().generate_single_series(50)
        assert values.shape == (50,)
        assert np.isfinite(values).all()


class TestCopulaValidation:
    """Constructor validation of copula parameters."""

    def test_non_square_correlation_matrix(self) -> None:
        with pytest.raises(ValueError, match="square"):
            make_gen(correlation_matrix=[[1.0, 0.5]])

    def test_asymmetric_correlation_matrix(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            make_gen(correlation_matrix=[[1.0, 0.9], [0.1, 1.0]])

    def test_non_unit_diagonal(self) -> None:
        with pytest.raises(ValueError, match="unit diagonal"):
            make_gen(correlation_matrix=[[2.0, 0.5], [0.5, 2.0]])

    def test_non_positive_definite(self) -> None:
        # Symmetric with unit diagonal, but eigenvalues {2.2, -0.2}
        with pytest.raises(ValueError, match="positive definite"):
            make_gen(correlation_matrix=[[1.0, 1.2], [1.2, 1.0]])

    def test_invalid_df(self) -> None:
        with pytest.raises(ValueError):
            make_gen(copula_type="t", df=0.0)

    def test_unknown_marginal_type(self) -> None:
        gen = make_gen(marginal_distributions=[{"type": "cauchy"}])
        with pytest.raises(ValueError, match="Unknown distribution type"):
            gen.generate(n_series=2)


@pytest.mark.stats
class TestCopulaStats:
    """Statistical property tests (fixed seeds)."""

    def test_gaussian_copula_correlations(self) -> None:
        """Normal marginals recover Pearson rho; Spearman matches
        (6/pi) * arcsin(rho/2)."""
        rho = 0.7
        gen = make_gen(
            min_length=4000,
            max_length=4000,
            seed=123,
            correlation_matrix=[[1.0, rho], [rho, 1.0]],
        )
        values = series_values(gen.generate(n_series=2))
        x, y = values["0"], values["1"]

        pearson = np.corrcoef(x, y)[0, 1]
        assert abs(pearson - rho) < 0.05, f"pearson {pearson:.3f} != {rho}"

        spearman_expected = (6.0 / np.pi) * np.arcsin(rho / 2.0)
        spearman = stats.spearmanr(x, y).statistic
        assert abs(spearman - spearman_expected) < 0.08, (
            f"spearman {spearman:.3f} != {spearman_expected:.3f}"
        )

    def test_t_copula_kendall_tau(self) -> None:
        """For elliptical copulas kendall_tau = (2/pi) * arcsin(rho) exactly."""
        rho = 0.6
        gen = make_gen(
            min_length=3000,
            max_length=3000,
            seed=456,
            copula_type="t",
            df=4.0,
            correlation_matrix=[[1.0, rho], [rho, 1.0]],
        )
        values = series_values(gen.generate(n_series=2))
        tau_expected = (2.0 / np.pi) * np.arcsin(rho)
        tau = stats.kendalltau(values["0"], values["1"]).statistic
        assert abs(tau - tau_expected) < 0.06, f"tau {tau:.3f} != {tau_expected:.3f}"

    def test_t_copula_uniform_marginals(self) -> None:
        """The t copula itself must have uniform(0, 1) marginals."""
        gen = make_gen(
            min_length=3000,
            max_length=3000,
            seed=789,
            copula_type="t",
            df=5.0,
            correlation_matrix=[[1.0, 0.5], [0.5, 1.0]],
            marginal_distributions=[{"type": "uniform", "low": 0.0, "high": 1.0}],
        )
        values = series_values(gen.generate(n_series=2))
        for uid in ("0", "1"):
            assert_distribution(values[uid], stats.uniform(loc=0.0, scale=1.0))

    def test_marginal_distributions_ks(self) -> None:
        """Each marginal spec produces the matching scipy distribution."""
        marginals = [
            {"type": "normal", "loc": 5.0, "scale": 2.0},
            {"type": "lognormal", "mean": 0.5, "sigma": 0.8},
            {"type": "exponential", "scale": 2.0},
            {"type": "uniform", "low": 2.0, "high": 5.0},
            {"type": "gamma", "shape": 2.0, "scale": 3.0},
        ]
        scipy_dists = [
            stats.norm(loc=5.0, scale=2.0),
            stats.lognorm(s=0.8, scale=np.exp(0.5)),
            stats.expon(scale=2.0),
            stats.uniform(loc=2.0, scale=3.0),
            stats.gamma(a=2.0, scale=3.0),
        ]
        gen = make_gen(
            min_length=2000,
            max_length=2000,
            seed=321,
            marginal_distributions=marginals,
        )
        values = series_values(gen.generate(n_series=5))
        for i, dist in enumerate(scipy_dists):
            assert_distribution(values[str(i)], dist)

    def test_t_copula_tail_dependence(self) -> None:
        """The t copula has the tail dependence the Gaussian copula lacks.

        For a bivariate t copula with correlation rho and dof nu, both tail
        dependence coefficients equal
        ``lambda = 2 * T_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))`` (McNeil, Frey
        & Embrechts 2015, Prop. 7.37); for the Gaussian copula the limit is 0.
        The empirical conditional joint-tail exceedance at quantile q
        approaches lambda as q -> 0.
        """
        rho, nu, n = 0.7, 3.0, 20_000
        lam_theory = 2 * stats.t.cdf(
            -np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1
        )

        def lam_hat(x: np.ndarray, y: np.ndarray, q: float) -> float:
            """Both-tails conditional joint exceedance at quantile q."""
            lo = np.mean((x <= np.quantile(x, q)) & (y <= np.quantile(y, q)))
            hi = np.mean((x >= np.quantile(x, 1 - q)) & (y >= np.quantile(y, 1 - q)))
            return float((lo + hi) / (2 * q))

        estimates = {}
        for copula_type, extra in [("t", {"df": nu}), ("gaussian", {})]:
            gen = make_gen(
                min_length=n,
                max_length=n,
                seed=1,
                copula_type=copula_type,
                correlation_matrix=[[1.0, rho], [rho, 1.0]],
                marginal_distributions=[{"type": "normal"}, {"type": "normal"}],
                **extra,
            )
            values = series_values(gen.generate(n_series=2))
            x, y = values["0"], values["1"]
            estimates[copula_type] = (lam_hat(x, y, 0.05), lam_hat(x, y, 0.01))

        t_05, t_01 = estimates["t"]
        g_05, g_01 = estimates["gaussian"]
        # Same correlation, but the t copula's joint tails are heavier.
        assert t_05 - g_05 > 0.05, (t_05, g_05)
        # Near the tail the t estimate sits at its theoretical coefficient...
        assert abs(t_01 - lam_theory) < 0.10, (t_01, lam_theory)
        # ...while the Gaussian estimate decays toward its limit of zero.
        assert g_01 < g_05, (g_01, g_05)
        assert g_01 < lam_theory - 0.10, (g_01, lam_theory)
