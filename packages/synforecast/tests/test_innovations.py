"""Statistical tests for BaseGenerator._sample_innovations.

Every innovation distribution is documented to be standardized to mean 0
and standard deviation equal to ``scale``. Each one is verified against
the matching (shifted/rescaled) scipy distribution with a KS test, plus
moment checks with sampling-error bounds.
"""

import numpy as np
import pytest
from scipy import stats

from synforecast.generators import RandomWalkGenerator
from tests.helpers import assert_distribution, assert_mean, assert_std

N_SAMPLES = 100_000
SCALE = 2.5


def _make_gen(dist: str, params: dict | None = None, seed: int = 42):
    return RandomWalkGenerator(
        min_length=10,
        max_length=10,
        freq="D",
        seed=seed,
        innovation_distribution=dist,
        innovation_params=params,
    )


def _skew_normal_frozen(alpha: float, scale: float):
    """Frozen scipy skewnorm matching the centered/rescaled sampler output."""
    delta = alpha / np.sqrt(1.0 + alpha**2)
    mean = delta * np.sqrt(2.0 / np.pi)
    std = np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    return stats.skewnorm(alpha, loc=-mean * (scale / std), scale=scale / std)


# (name, params, frozen scipy dist at scale=SCALE, kurtosis for std bound)
DISTRIBUTIONS = [
    ("normal", None, stats.norm(0.0, SCALE), 3.0),
    (
        "t",
        {"df": 6.0},
        stats.t(6.0, scale=SCALE * np.sqrt((6.0 - 2.0) / 6.0)),
        3.0 + 6.0 / (6.0 - 4.0),
    ),
    ("laplace", None, stats.laplace(scale=SCALE / np.sqrt(2.0)), 6.0),
    (
        "uniform",
        None,
        stats.uniform(loc=-SCALE * np.sqrt(3.0), scale=2 * SCALE * np.sqrt(3.0)),
        1.8,
    ),
    ("skew_normal", {"alpha": 5.0}, _skew_normal_frozen(5.0, SCALE), 3.9),
]
DIST_IDS = [d[0] for d in DISTRIBUTIONS]


@pytest.mark.stats
@pytest.mark.parametrize("case", DISTRIBUTIONS, ids=DIST_IDS)
class TestInnovationDistributions:
    """Each distribution matches its scipy counterpart and its moments."""

    def test_ks_matches_scipy(self, case):
        dist, params, frozen, _ = case
        gen = _make_gen(dist, params)
        samples = gen._sample_innovations(N_SAMPLES, scale=SCALE)
        assert_distribution(samples, frozen)

    def test_mean_zero_std_scale(self, case):
        dist, params, _, kurtosis = case
        gen = _make_gen(dist, params)
        samples = gen._sample_innovations(N_SAMPLES, scale=SCALE)
        assert_mean(samples, 0.0, std=SCALE)
        assert_std(samples, SCALE, kurtosis=kurtosis)

    def test_default_scale_is_unit(self, case):
        dist, params, _, kurtosis = case
        gen = _make_gen(dist, params)
        samples = gen._sample_innovations(N_SAMPLES)
        assert_mean(samples, 0.0, std=1.0)
        assert_std(samples, 1.0, kurtosis=kurtosis)


class TestSampleInnovationsAPI:
    """Shape handling, parameter validation, and reproducibility."""

    def test_tuple_size(self):
        gen = _make_gen("normal")
        out = gen._sample_innovations((3, 50), scale=2.0)
        assert out.shape == (3, 50)

    @pytest.mark.parametrize(
        "dist,params",
        [(d, p) for d, p, _, _ in DISTRIBUTIONS],
        ids=DIST_IDS,
    )
    def test_seeded_reproducibility(self, dist, params):
        a = _make_gen(dist, params, seed=7)._sample_innovations(100)
        b = _make_gen(dist, params, seed=7)._sample_innovations(100)
        np.testing.assert_array_equal(a, b)

    def test_t_low_df_raises(self):
        with pytest.raises(ValueError, match="df.*must be > 2"):
            _make_gen("t", {"df": 1.5})

    def test_t_df_two_raises(self):
        with pytest.raises(ValueError, match="df.*must be > 2"):
            _make_gen("t", {"df": 2.0})

    def test_unknown_distribution_rejected_by_validation(self):
        with pytest.raises(ValueError):
            _make_gen("cauchy")

    def test_uniform_is_bounded(self):
        gen = _make_gen("uniform")
        a = 2.0 * np.sqrt(3.0)
        samples = gen._sample_innovations(50_000, scale=2.0)
        assert np.all(samples >= -a)
        assert np.all(samples <= a)

    def test_skew_normal_is_skewed(self):
        gen = _make_gen("skew_normal", {"alpha": 5.0})
        samples = gen._sample_innovations(50_000)
        # alpha=5 -> theoretical skewness ~0.85; sampling SE ~sqrt(6/n)~0.011
        assert stats.skew(samples) > 0.5

    def test_t_has_heavier_tails_than_normal(self):
        t_samples = _make_gen("t", {"df": 5.0})._sample_innovations(50_000)
        n_samples = _make_gen("normal")._sample_innovations(50_000)
        assert stats.kurtosis(t_samples) > stats.kurtosis(n_samples) + 1.0


class TestInnovationsThroughGenerators:
    """Innovations propagate correctly through generator pipelines."""

    BASE = {"min_length": 200, "max_length": 200, "freq": "D", "seed": 42}

    @pytest.mark.parametrize(
        "dist,params",
        [(d, p) for d, p, _, _ in DISTRIBUTIONS],
        ids=DIST_IDS,
    )
    def test_random_walk_finite(self, dist, params, engine):
        gen = RandomWalkGenerator(
            **self.BASE,
            engine=engine,
            innovation_distribution=dist,
            innovation_params=params,
        )
        df = gen.generate(n_series=2)
        values = np.asarray(
            df["y"].to_numpy() if hasattr(df["y"], "to_numpy") else df["y"]
        )
        assert values.shape == (400,)
        assert np.all(np.isfinite(values))

    @pytest.mark.stats
    def test_random_walk_step_std_matches_volatility(self):
        """Steps of a t-innovation random walk have std == volatility."""
        gen = RandomWalkGenerator(
            min_length=20_000,
            max_length=20_000,
            freq="D",
            seed=11,
            volatility=1.5,
            innovation_distribution="t",
            innovation_params={"df": 6.0},
        )
        values = gen.generate_single_series(20_000)
        steps = np.diff(values)
        assert_std(steps, 1.5, kurtosis=6.0)

    @pytest.mark.parametrize(
        "cls_name,extra",
        [
            ("GARCHGenerator", {}),
            ("SARIMAGenerator", {"p": 1, "q": 1}),
            ("ETSGenerator", {}),
        ],
        ids=["GARCH", "SARIMA", "ETS"],
    )
    def test_t_innovation_other_generators(self, cls_name, extra):
        from synforecast import generators

        cls = getattr(generators, cls_name)
        gen = cls(
            **{
                **self.BASE,
                **extra,
                "innovation_distribution": "t",
                "innovation_params": {"df": 5},
            }
        )
        df = gen.generate(n_series=1)
        values = df["y"].to_numpy()
        assert len(values) == 200
        assert np.all(np.isfinite(values))
