"""Tests for Rust/Python parity.
"""

import importlib

import numpy as np
import pytest

from synforecast.generators import (
    BoundedProcessGenerator,
    ChaoticSystemGenerator,
    ClickstreamGenerator,
    CopulaGenerator,
    CyclicGenerator,
    DailyActiveUsersGenerator,
    EnergyLoadGenerator,
    ETSGenerator,
    FractionalBrownianMotionGenerator,
    GARCHGenerator,
    GaussianProcessGenerator,
    GeometricBrownianMotionGenerator,
    HawkesProcessGenerator,
    INARGenerator,
    IntermittentDemandGenerator,
    IoTSensorGenerator,
    JumpDiffusionGenerator,
    LevyProcessGenerator,
    OrnsteinUhlenbeckGenerator,
    PoissonProcessGenerator,
    RandomWalkGenerator,
    RegimeSwitchingGenerator,
    SARIMAGenerator,
    SeasonalGenerator,
    StateSpaceGenerator,
    StochasticVolatilityGenerator,
    VARGenerator,
    VitalSignsGenerator,
)

BASE = {
    "min_length": 200,
    "max_length": 200,
    "freq": "D",
    "seed": 42,
    "engine": "polars",
}


def _generate_values(gen):
    """Generate a single series and return the numpy array."""
    return gen.generate_single_series(200)


def _set_backend(cls, monkeypatch, has_rust: bool) -> None:
    """Force the Rust or the pure-Python path for a generator's module."""
    mod = importlib.import_module(cls.__module__)
    monkeypatch.setattr(mod, "_HAS_RUST", has_rust)


GENERATORS = [
    # Statistical
    ("RandomWalk", RandomWalkGenerator, {}),
    ("Seasonal", SeasonalGenerator, {}),
    ("SARIMA", SARIMAGenerator, {"p": 1, "q": 1}),
    ("ETS", ETSGenerator, {}),
    ("INAR", INARGenerator, {}),
    # Stochastic
    ("GARCH", GARCHGenerator, {}),
    ("OrnsteinUhlenbeck", OrnsteinUhlenbeckGenerator, {}),
    ("GeometricBrownianMotion", GeometricBrownianMotionGenerator, {}),
    ("JumpDiffusion", JumpDiffusionGenerator, {}),
    ("PoissonProcess", PoissonProcessGenerator, {}),
    ("Cyclic", CyclicGenerator, {}),
    (
        "FractionalBrownianMotion",
        FractionalBrownianMotionGenerator,
        {"method": "cholesky"},
    ),
    ("HawkesProcess", HawkesProcessGenerator, {}),
    ("StochasticVolatility", StochasticVolatilityGenerator, {}),
    (
        "RegimeSwitching",
        RegimeSwitchingGenerator,
        {"n_regimes": 2, "regime_means": [0.0, 1.0]},
    ),
    ("ChaoticSystem", ChaoticSystemGenerator, {}),
    ("BoundedProcess", BoundedProcessGenerator, {}),
    ("LevyProcess", LevyProcessGenerator, {"alpha": 1.8}),
    # Multivariate (single-series output via generate_single_series)
    ("Copula", CopulaGenerator, {}),
    ("VAR", VARGenerator, {"lag_order": 1}),
    ("GaussianProcess", GaussianProcessGenerator, {}),
    # Domain-Specific
    ("IntermittentDemand", IntermittentDemandGenerator, {}),
    ("IoTSensor", IoTSensorGenerator, {}),
    ("EnergyLoad", EnergyLoadGenerator, {}),
    ("StateSpace", StateSpaceGenerator, {}),
    ("DailyActiveUsers", DailyActiveUsersGenerator, {}),
    ("VitalSigns", VitalSignsGenerator, {}),
    ("Clickstream", ClickstreamGenerator, {}),
]


@pytest.mark.parametrize("name,cls,extra", GENERATORS, ids=[g[0] for g in GENERATORS])
class TestRustPythonParity:
    """Both Rust and Python paths should produce valid, equivalent output."""

    def test_rust_path_valid(self, name, cls, extra):  # noqa: ARG002
        """Rust path produces valid output."""
        gen = cls(**{**BASE, **extra})
        values = _generate_values(gen)
        assert values.shape == (200,)
        assert np.all(np.isfinite(values))

    def test_python_path_valid(self, name, cls, extra, monkeypatch):  # noqa: ARG002
        """Python fallback path produces valid output."""
        _set_backend(cls, monkeypatch, False)
        gen = cls(**{**BASE, **extra})
        values = _generate_values(gen)
        assert values.shape == (200,)
        assert np.all(np.isfinite(values))

    def test_both_paths_same_shape(self, name, cls, extra, monkeypatch):  # noqa: ARG002
        """Both paths should produce output of the same shape."""
        rust_values = _generate_values(cls(**{**BASE, **extra}))
        _set_backend(cls, monkeypatch, False)
        py_values = _generate_values(cls(**{**BASE, **extra}))
        assert rust_values.shape == py_values.shape

    def test_ensemble_statistics_match(self, name, cls, extra, monkeypatch):
        """Ensemble spread should match across backends (same model).

        The two backends draw from independent RNGs, so a single series is not
        comparable — but if both implement the same data-generating process,
        the *distribution* of per-series dispersion agrees. 
        """
        n = 40

        def ensemble_median_std(has_rust: bool) -> float:
            _set_backend(cls, monkeypatch, has_rust)
            gen = cls(**{**BASE, **extra, "seed": 123})
            stds = [np.nanstd(gen.generate_single_series(200)) for _ in range(n)]
            return float(np.median(stds))

        rust_std = ensemble_median_std(True)
        py_std = ensemble_median_std(False)

        if py_std <= 1e-9:
            assert rust_std <= 1e-6, f"{name}: python≈0 std but rust={rust_std:.4g}"
            return
        ratio = rust_std / py_std
        assert 0.4 < ratio < 2.5, (
            f"{name}: ensemble std ratio {ratio:.3f} out of range "
            f"(rust={rust_std:.4g}, py={py_std:.4g}) — backends may diverge"
        )


# Documented dynamics parameters that were, or could be, silently dropped on
# one backend. Each case toggles a parameter between two values that must
# change the output; a backend that ignores the parameter fails here. This is
# the check that catches Rust/Python model drift (e.g. an unmarshalled arg).
#   (label, cls, extra_kwargs, param, value_a, value_b)
WIRING_CASES = [
    (
        "iot_drift_noise",
        IoTSensorGenerator,
        {"drift_rate": 0.0},
        "drift_noise",
        0.0,
        5.0,
    ),
    (
        "iot_battery_degradation_rate",
        IoTSensorGenerator,
        {"battery_life": 20},
        "battery_degradation_rate",
        0.0,
        0.5,
    ),
    (
        "iot_stuck_value",
        IoTSensorGenerator,
        {"failure_type": "stuck", "failure_probability": 0.15},
        "stuck_value",
        None,
        999.0,
    ),
    (
        "cyclic_innovation_distribution",
        CyclicGenerator,
        {"noise_std": 2.0},
        "innovation_distribution",
        "normal",
        "t",
    ),
]


@pytest.mark.parametrize(
    "label,cls,extra,param,val_a,val_b",
    WIRING_CASES,
    ids=[c[0] for c in WIRING_CASES],
)
class TestParameterWiring:
    """Every documented parameter must reach both backends and change output."""

    def _values(self, cls, kwargs) -> np.ndarray:
        return cls(**kwargs).generate_single_series(200)

    def test_param_wired_default_backend(self, label, cls, extra, param, val_a, val_b):
        """The installed (Rust when built) backend must honor the parameter."""
        a = self._values(cls, {**BASE, **extra, param: val_a})
        b = self._values(cls, {**BASE, **extra, param: val_b})
        assert not np.allclose(a, b, equal_nan=True), (
            f"[{label}] {cls.__name__}.{param} has no effect on the default backend"
        )

    def test_param_wired_python_backend(
        self, label, cls, extra, param, val_a, val_b, monkeypatch
    ):
        """The pure-Python fallback must honor the parameter."""
        _set_backend(cls, monkeypatch, False)
        a = self._values(cls, {**BASE, **extra, param: val_a})
        b = self._values(cls, {**BASE, **extra, param: val_b})
        assert not np.allclose(a, b, equal_nan=True), (
            f"[{label}] {cls.__name__}.{param} has no effect on the Python path"
        )
