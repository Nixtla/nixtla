"""Tests for Rust/Python parity — both paths should produce valid output."""

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
    """Both Rust and Python paths should produce valid (finite, correct shape) output."""

    def test_rust_path_valid(self, name, cls, extra):  # noqa: ARG002
        """Rust path produces valid output."""
        gen = cls(**{**BASE, **extra})
        values = _generate_values(gen)
        assert values.shape == (200,)
        assert np.all(np.isfinite(values))

    def test_python_path_valid(self, name, cls, extra, monkeypatch):  # noqa: ARG002
        """Python fallback path produces valid output."""
        module = cls.__module__
        import importlib

        mod = importlib.import_module(module)
        monkeypatch.setattr(mod, "_HAS_RUST", False)
        gen = cls(**{**BASE, **extra})
        values = _generate_values(gen)
        assert values.shape == (200,)
        assert np.all(np.isfinite(values))

    def test_both_paths_same_shape(self, name, cls, extra, monkeypatch):  # noqa: ARG002
        """Both paths should produce output of the same shape."""
        gen_rust = cls(**{**BASE, **extra})
        rust_values = _generate_values(gen_rust)

        module = cls.__module__
        import importlib

        mod = importlib.import_module(module)
        monkeypatch.setattr(mod, "_HAS_RUST", False)
        gen_py = cls(**{**BASE, **extra})
        py_values = _generate_values(gen_py)

        assert rust_values.shape == py_values.shape

    def test_statistical_properties_comparable(self, name, cls, extra, monkeypatch):
        """Both paths should produce output with similar statistical properties."""
        gen_rust = cls(**{**BASE, **extra})
        rust_values = _generate_values(gen_rust)

        module = cls.__module__
        import importlib

        mod = importlib.import_module(module)
        monkeypatch.setattr(mod, "_HAS_RUST", False)
        gen_py = cls(**{**BASE, **extra})
        py_values = _generate_values(gen_py)

        # Standard deviations should be in the same order of magnitude
        rust_std = np.std(rust_values)
        py_std = np.std(py_values)
        if py_std > 0:
            std_ratio = max(rust_std, 1e-9) / max(py_std, 1e-9)
            assert 0.01 < std_ratio < 100.0, (
                f"{name}: std ratio {std_ratio:.4f} "
                f"(rust_std={rust_std:.4f}, py_std={py_std:.4f})"
            )
