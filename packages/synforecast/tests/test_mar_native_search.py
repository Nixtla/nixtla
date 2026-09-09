"""Independent checks for native search scoring and NumPy-only stationarity."""

import os
import subprocess
import sys

import numpy as np
import pytest

from synforecast._lib import augmentation
from synforecast.generators import MARGenerator


@pytest.mark.parametrize("order", [2, 5, 12])
def test_second_moment_operator_matches_full_companion_products(order):
    rng = np.random.default_rng(order)
    coefficients = [rng.uniform(-0.5, 0.5, order), rng.uniform(-0.5, 0.5, order - 1)]
    weights = [0.3, 0.7]
    matrix = rng.normal(size=(order, order))
    matrix += matrix.T
    expected = np.zeros_like(matrix)
    for weight, coefficient in zip(weights, coefficients, strict=True):
        companion = np.zeros_like(matrix)
        companion[0, : len(coefficient)] = coefficient
        companion[1:, :-1] = np.eye(order - 1)
        expected += weight * companion @ matrix @ companion.T
    indices = np.triu_indices(order)
    actual = MARGenerator._second_moment_operator(weights, coefficients)(
        matrix[indices]
    )
    np.testing.assert_allclose(actual, expected[indices], atol=1e-12)


@pytest.mark.parametrize("order", [5, 9, 17])
def test_stationarity_matches_full_kronecker_operator(order):
    rng = np.random.default_rng(order)
    for _ in range(4):
        coefficients = [
            MARGenerator._pacf_to_ar(rng.uniform(-0.8, 0.8, order)) for _ in range(2)
        ]
        weights = [0.4, 0.6]
        # Explicit companion Kronecker products are independent of matvec.
        operator = np.zeros((order**2, order**2))
        for weight, coefficient in zip(weights, coefficients, strict=True):
            companion = np.zeros((order, order))
            companion[0] = coefficient
            companion[1:, :-1] = np.eye(order - 1)
            operator += weight * np.kron(companion, companion)
        expected = max(abs(np.linalg.eigvals(operator))) < 1 - 1e-8
        assert (
            MARGenerator._is_mixture_stationary(np.array(weights), coefficients)
            == expected
        )


@pytest.mark.parametrize("failure", ["nonconvergence", "nonfinite"])
def test_eigensolver_failure_is_not_accepted(monkeypatch, failure):
    def fail(*_args, **_kwargs):
        if failure == "nonfinite":
            return np.array([np.nan])
        raise np.linalg.LinAlgError("did not converge")

    MARGenerator._mixture_stationary_cached.cache_clear()
    monkeypatch.setattr(np.linalg, "eigvals", fail)
    coefficients = [
        MARGenerator._apply_seasonal_factor(np.array([0.3, 0.1]), 4, s)
        for s in [0.999, 0.998]
    ]
    with pytest.raises(ValueError, match="could not verify"):
        MARGenerator._is_mixture_stationary(np.array([0.5, 0.5]), coefficients)


def test_periodic_stability_certificate_avoids_eigensolver(monkeypatch):
    def unexpected_solver(*_args, **_kwargs):
        pytest.fail("periodic model should be certified by covariance powers")

    monkeypatch.setattr(np.linalg, "eigvals", unexpected_solver)
    MARGenerator._mixture_stationary_cached.cache_clear()
    # Removing the common stable AR factor leaves x_t = s_k*x[t-24]+eps.
    # E[s_k**2] < 1 proves stationarity independently of our covariance map.
    coefficients = [
        MARGenerator._apply_seasonal_factor(np.array([0.3, 0.1]), 24, s)
        for s in [0.6, 0.5]
    ]
    assert MARGenerator._is_mixture_stationary(np.array([0.4, 0.6]), coefficients)


def test_unstable_covariance_certificate_avoids_dense_eigensolver(monkeypatch):
    def unexpected_solver(*_args, **_kwargs):
        pytest.fail("expanding covariance should certify instability")

    coefficients = [
        np.array([1.9, -0.95, 0.0, 0.0, 0.001]),
        np.array([-1.9, -0.95, 0.0, 0.0, 0.001]),
    ]
    # Each component is stable, but switching creates an unstable mixture.
    assert all(MARGenerator._is_stationary(c) for c in coefficients)
    full = np.zeros((25, 25))
    for c in coefficients:
        companion = np.zeros((5, 5))
        companion[0] = c
        companion[1:, :-1] = np.eye(4)
        full += 0.5 * np.kron(companion, companion)
    assert max(abs(np.linalg.eigvals(full))) > 1.0
    monkeypatch.setattr(np.linalg, "eigvals", unexpected_solver)
    MARGenerator._mixture_stationary_cached.cache_clear()
    assert not MARGenerator._is_mixture_stationary(np.array([0.5, 0.5]), coefficients)


@pytest.mark.parametrize("seasonal", [[0.999, 0.998], [1 - 1e-12, 1 - 2e-12]])
def test_near_periodic_fallback_checks_full_spectrum(monkeypatch, seasonal):
    coefficients = [
        MARGenerator._apply_seasonal_factor(np.array([0.3, 0.1]), 4, s)
        for s in seasonal
    ]
    eigvals = np.linalg.eigvals
    calls = []

    def record(matrix):
        calls.append(matrix.shape)
        return eigvals(matrix)

    monkeypatch.setattr(np.linalg, "eigvals", record)
    MARGenerator._mixture_stationary_cached.cache_clear()
    actual = MARGenerator._is_mixture_stationary(np.array([0.4, 0.6]), coefficients)
    # Removing the common stable AR factor leaves x_t = s_k*x[t-4]+eps,
    # so its second-moment spectral radius is E[s_k**2]**(1/4).
    radius = (0.4 * seasonal[0] ** 2 + 0.6 * seasonal[1] ** 2) ** 0.25
    assert actual == (radius < 1 - 1e-8)
    assert calls == [(21, 21)]


def test_import_generation_and_targeting_without_scipy():
    # A fresh interpreter prevents development dependencies from masking a
    # runtime import, including future lazy imports in generation or tuning.
    script = """
import importlib.abc
import sys

class NoSciPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scipy' or fullname.startswith('scipy.'):
            raise ModuleNotFoundError('SciPy is unavailable', name=fullname)

assert 'scipy' not in sys.modules
sys.meta_path.insert(0, NoSciPy())
import numpy as np
import synforecast
from synforecast.generators import MARGenerator

coefficients = [
    MARGenerator._apply_seasonal_factor(np.array([.3, .1]), 4, s).tolist()
    for s in [.999, .998]
]
generator = MARGenerator(
    min_length=64, max_length=64, freq=1, seed=42,
    weights=[.4, .6], ar_coefficients=coefficients,
    intercepts=[0., 0.], noise_scales=[1., 1.],
)
assert len(generator.generate(1)) == 64
assert np.isfinite(generator.generate_single_series(64)).all()
tuned = MARGenerator.tune_to_features(
    {'acf1': .5}, min_length=64, max_length=96, freq=1,
    n_generations=1, population_size=4, seed=42, n_jobs=1,
)
assert tuned.tuning_diagnostics is not None
assert 'scipy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "RAYON_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_native_feature_batch_isolates_failed_candidates_and_is_deterministic():
    generator = MARGenerator(
        min_length=64,
        max_length=96,
        freq=1,
        weights=[1.0],
        ar_coefficients=[[0.0]],
        intercepts=[0.0],
        noise_scales=[1.0],
    )
    sp, ap = generator._get_batch_params()
    arrays = [a.tolist() for a in ap]
    bad = [a.copy() for a in arrays]
    bad[3] = [1e9]  # Valid layout, but simulation violates the output guard.
    args = ([sp.tolist()] * 3, [arrays, bad, arrays], [64, 96], [[1, 2]] * 3, None)
    one = augmentation.mar_features_batch(*args, 1)
    four = augmentation.mar_features_batch(*args, 4)
    assert one == four
    assert one[0] == one[2]
    assert one[1] is None
    for row in one[0]:
        assert len(row) == 4 and np.isfinite(row).all()


def test_native_feature_batch_matches_batch_generation():
    from synforecast._lib import batch

    generator = MARGenerator(
        min_length=64,
        max_length=96,
        freq=1,
        weights=[1.0],
        ar_coefficients=[[0.5]],
        intercepts=[2.0],
        noise_scales=[0.5],
        standardize=False,
        innovation_distribution="t",
        innovation_params={"df": 6},
    )
    sp, ap = generator._get_batch_params()
    lengths, seeds = [64, 96], [11, 12]
    actual = augmentation.mar_features_batch(
        [sp.tolist()], [[a.tolist() for a in ap]], lengths, [seeds], None, 2
    )[0]
    # Match the public native generation entry point with identical explicit seeds.
    draws = batch.generate_batch(
        30,
        sp,
        ap,
        np.array(lengths, dtype=np.int32),
        np.array(seeds, dtype=np.uint64),
        np.zeros(6, dtype=np.uint64),
        generator._build_pi_config_dict(),
        n_workers=1,
    )
    for got, draw in zip(actual, draws, strict=True):
        np.testing.assert_allclose(
            got, augmentation.compute_features(draw["values"], None)
        )


def test_targeting_options_and_worker_count(monkeypatch):
    def python_path(*_args):
        pytest.fail("search used Python per-timestep simulation")

    monkeypatch.setattr(MARGenerator, "generate_single_series", python_path)
    options = {
        "target_features": {"acf1": 0.5},
        "min_length": 64,
        "max_length": 96,
        "freq": 1,
        "n_generations": 2,
        "population_size": 8,
        "seed": 42,
        "engine": "polars",
        "id_col": "series",
        "time_col": "time",
        "target_col": "value",
        "alias": "tuned",
        "standardize": False,
        "burn_in": 15,
        "innovation_distribution": "t",
        "innovation_params": {"df": 6},
    }
    first = MARGenerator.tune_to_features(**options, n_jobs=1)
    second = MARGenerator.tune_to_features(**options, n_jobs=4)
    assert first.ar_coefficients == second.ar_coefficients
    assert first.tuning_diagnostics == second.tuning_diagnostics
    assert not first.standardize and first.burn_in == 15
    assert first.innovation_distribution == "t" and first.innovation_params == {"df": 6}
    frame = first.generate(1)
    assert frame.columns == ["series", "time", "value"]
    assert first.tuning_diagnostics is not None


@pytest.mark.parametrize(
    "options",
    [
        {"n_jobs": 0},
        {"engine": "invalid"},
        {"weights": [1.0]},
        {"max_length": 1_000_001},
    ],
)
def test_bad_options_fail_before_search(monkeypatch, options):
    def no_search(*_args):
        pytest.fail("started search with invalid options")

    monkeypatch.setattr(MARGenerator, "_random_candidate", no_search)
    config = {"min_length": 64, "max_length": 96, "freq": 1, **options}
    with pytest.raises(ValueError):
        MARGenerator.tune_to_features({"acf1": 0.5}, **config)
