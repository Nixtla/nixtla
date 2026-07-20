"""Tests for LevyProcessGenerator (alpha-stable increments).

Increments are scale * S(alpha, beta_skew; 1) + location, in the S1
parameterization used by scipy.stats.levy_stable, so KS tests against
scipy validate the Chambers-Mallows-Stuck sampler directly.
"""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import LevyProcessGenerator
from tests.helpers import assert_distribution, assert_long_format, to_pandas


def _gen(**overrides) -> LevyProcessGenerator:
    params = {
        "min_length": 50,
        "max_length": 80,
        "freq": "D",
        "seed": 42,
        **overrides,
    }
    return LevyProcessGenerator(**params)


def _increments(n: int, seed: int = 7, **overrides) -> np.ndarray:
    gen = _gen(seed=seed, cumulative=False, min_length=n, max_length=n, **overrides)
    return gen.generate_single_series(n)


class TestAPI:
    def test_long_format(self, engine: str) -> None:
        df = _gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=50, max_length=80)

    def test_exact_length(self, engine: str) -> None:
        df = _gen(engine=engine, min_length=64, max_length=64).generate(n_series=2)
        assert_long_format(df, n_series=2, min_length=64, max_length=64)

    def test_seed_determinism(self, engine: str) -> None:
        df1 = _gen(engine=engine).generate(n_series=3)
        df2 = _gen(engine=engine).generate(n_series=3)
        assert to_pandas(df1).equals(to_pandas(df2))

    def test_cumulative_is_cumsum_of_increments(self) -> None:
        # Same seed -> same increment stream; cumulative mode must equal
        # initial_value + cumsum of the non-cumulative output.
        kwargs = {"alpha": 1.5, "seed": 5, "min_length": 500, "max_length": 500}
        cum = _gen(cumulative=True, initial_value=10.0, **kwargs)
        inc = _gen(cumulative=False, **kwargs)
        np.testing.assert_allclose(
            cum.generate_single_series(500),
            10.0 + np.cumsum(inc.generate_single_series(500)),
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        "bad",
        [
            {"alpha": 0.0},
            {"alpha": 2.1},
            {"beta_skew": 1.5},
            {"beta_skew": -1.5},
            {"scale": 0.0},
            {"min_length": 100, "max_length": 50},
        ],
    )
    def test_validation_errors(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            _gen(**bad)

    @pytest.mark.parametrize(
        "alpha,beta,expected",
        [
            (2.0, 0.0, "Gaussian"),
            (1.0, 0.0, "Cauchy"),
            (0.5, 1.0, "Levy"),
            (1.5, 0.0, "alpha-stable(alpha=1.5, beta=0.0)"),
        ],
    )
    def test_get_model_info(self, alpha: float, beta: float, expected: str) -> None:
        info = _gen(alpha=alpha, beta_skew=beta).get_model_info()
        assert info["distribution"] == expected
        assert info["infinite_variance"] is (alpha < 2.0)
        assert info["infinite_mean"] is (alpha <= 1.0)


@pytest.mark.stats
class TestStatistics:
    N = 10_000

    def test_alpha_stable_symmetric(self) -> None:
        x = _increments(self.N, seed=7, alpha=1.5, beta_skew=0.0)
        assert_distribution(x, stats.levy_stable(1.5, 0.0))

    def test_alpha_stable_skewed(self) -> None:
        x = _increments(self.N, seed=11, alpha=1.2, beta_skew=0.5)
        assert_distribution(x, stats.levy_stable(1.2, 0.5))

    def test_gaussian_case_with_scale_and_location(self) -> None:
        # alpha=2 is N(location, scale*sqrt(2)).
        x = _increments(self.N, seed=13, alpha=2.0, scale=0.5, location=3.0)
        assert_distribution(x, stats.norm(3.0, 0.5 * np.sqrt(2.0)))

    def test_cauchy_case(self) -> None:
        x = _increments(self.N, seed=17, alpha=1.0, beta_skew=0.0)
        assert_distribution(x, stats.cauchy())

    def test_self_similarity_scaling(self) -> None:
        # Sums of k stable increments scale as k^(1/alpha); estimate alpha
        # from the IQR ratio of block sums to increments.
        alpha = 1.5
        x = _increments(100_000, seed=19, alpha=alpha)
        k = 16
        sums = x.reshape(-1, k).sum(axis=1)
        iqr1 = np.subtract(*np.percentile(x, [75, 25]))
        iqrk = np.subtract(*np.percentile(sums, [75, 25]))
        est_alpha = np.log(k) / np.log(iqrk / iqr1)
        assert abs(est_alpha - alpha) < 0.2, f"estimated alpha {est_alpha:.3f}"

    def test_beta_skew_sign(self) -> None:
        # For beta_skew=1 the right tail is heavier than the left.
        x = _increments(50_000, seed=23, alpha=1.5, beta_skew=1.0)
        q_hi, q_lo = np.percentile(x, [99.5, 0.5])
        assert q_hi > -q_lo
