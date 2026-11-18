"""Pinned draws of the released overlays (changepoints, anomalies, missingness) on every generation path.

The hashes were recorded before the overlay pipeline moved into one native
executor; every path must keep drawing exactly these series, flags and
post-call generator states.
"""

import hashlib
import platform
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from synforecast import SynSet
from synforecast.base import BaseGenerator
from synforecast.exogenous import ExogenousConfig
from synforecast.generators import (
    CopulaGenerator,
    DailyActiveUsersGenerator,
    ETSGenerator,
    IoTSensorGenerator,
    RandomWalkGenerator,
    SeasonalGenerator,
    StateSpaceGenerator,
    TCMGenerator,
    VARGenerator,
)

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64",
    reason="hashes pin floating-point output on Linux x86_64",
)

_FLAGS = ExogenousConfig(anomaly_flags=True, changepoint_flags=True, missing_flags=True)

CONFIGS: dict[str, dict[str, Any]] = {
    "off": {},
    "all": {
        "changepoints": True,
        "num_changepoints": 3,
        "changepoint_type": "mixed",
        "anomalies": True,
        "anomaly_types": ["spike", "dip", "level_shift"],
        "anomaly_fraction": 0.03,
        "missing_data": True,
        "missing_pattern": "block",
        "missing_rate": 0.1,
        "missing_block_size": 4,
        "exogenous": _FLAGS,
    },
    "relative": {
        "changepoints": True,
        "num_changepoints": 2,
        "changepoint_type": "level",
        "changepoint_locations": [0.3, 0.7],
        "changepoint_level_changes": [1.5, -2.0],
        "anomalies": True,
        "anomaly_types": ["spike", "level_shift"],
        "anomaly_fraction": 0.02,
        "spike_magnitude": 4.0,
        "level_shift_magnitude": -2.0,
        "missing_data": True,
        "missing_rate": 0.05,
        "overlay_scale": "relative",
        "exogenous": _FLAGS,
    },
}


class _ThreadedWalk(BaseGenerator):
    """A generator without a native batch kernel, so it takes the threaded path."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.cumsum(self.rng.normal(size=length))


_COMMON = {"min_length": 60, "max_length": 90, "freq": "h", "seed": 7}


def _hash_frame(
    df: pd.DataFrame, gen: BaseGenerator | None = None, *, values_too: bool = True
) -> str:
    """Hash every column; without ``values_too``, only the target's missing mask."""
    digest = hashlib.sha256()
    for col in sorted(df.columns):
        values = df[col].to_numpy()
        if col == "y" and not values_too:
            values = np.isnan(values)
        if values.dtype.kind == "M":
            # Nanoseconds whatever the frame's datetime resolution (pandas 3 defaults to microseconds).
            values = values.astype("datetime64[ns]").astype("int64")
        digest.update(col.encode())
        digest.update(np.ascontiguousarray(values.astype(np.float64)).tobytes())
    if gen is not None:
        digest.update(gen.rng.integers(0, 2**63, size=2, dtype=np.uint64).tobytes())
    return digest.hexdigest()[:16]


def _hash_values(values: list[np.ndarray], gen: BaseGenerator) -> str:
    digest = hashlib.sha256()
    for v in values:
        digest.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    digest.update(gen.rng.integers(0, 2**63, size=2, dtype=np.uint64).tobytes())
    return digest.hexdigest()[:16]


def _draw(path: str, config: dict[str, Any]) -> str:
    """Draw on one path with one overlay configuration and hash the result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if path == "batch":
            gen = RandomWalkGenerator(**_COMMON, **config)
            return _hash_frame(gen.generate(6, n_jobs=1), gen)
        if path == "batch_jobs":
            gen = RandomWalkGenerator(**_COMMON, **config)
            return _hash_frame(gen.generate(6, n_jobs=4), gen)
        if path == "batch_values":
            gen = SeasonalGenerator(**_COMMON, **config)
            return _hash_values(gen.generate_values(5, n_jobs=1), gen)
        if path == "threaded":
            gen = _ThreadedWalk(**_COMMON, **config)
            return _hash_frame(gen.generate(6, n_jobs=3), gen)
        if path == "synset":
            gens = [
                RandomWalkGenerator(**_COMMON, **config),
                SeasonalGenerator(**{**_COMMON, "seed": 8}, **config),
                _ThreadedWalk(**{**_COMMON, "seed": 9}, **config),
            ]
            df = SynSet(gens).generate(n_series_per_generator=3, n_jobs=2)
            digest = _hash_frame(df.reset_index(drop=True))
            return digest + "".join(_hash_values([], g)[:4] for g in gens)
        if path == "var":
            gen = VARGenerator(**_COMMON, **config)
            return _hash_frame(gen.generate(3), gen)
        if path == "copula":
            gen = CopulaGenerator(**_COMMON, **config)
            return _hash_frame(gen.generate(3), gen, values_too=False)
        if path == "dau":
            gen = DailyActiveUsersGenerator(**{**_COMMON, "freq": "D"}, **config)
            return _hash_frame(gen.generate(3), gen)
        if path == "iot":
            gen = IoTSensorGenerator(**_COMMON, **config)
            return _hash_frame(gen.generate(2), gen)
        if path == "tcm":
            gen = TCMGenerator(**_COMMON, multivariate=True, **config)
            return _hash_frame(gen.generate(3), gen, values_too=False)
        if path == "ets_states":
            gen = ETSGenerator(**_COMMON, **config)
            obs, states = gen.generate_with_states(3)
            return _hash_frame(obs, gen) + _hash_frame(states)[:8]
        if path == "state_space_states":
            gen = StateSpaceGenerator(**_COMMON, **config)
            obs, states = gen.generate_with_states(3)
            return _hash_frame(obs, gen) + _hash_frame(states)[:8]
    raise ValueError(path)


PATHS = (
    "batch",
    "batch_jobs",
    "batch_values",
    "threaded",
    "synset",
    "var",
    "copula",
    "dau",
    "iot",
    "tcm",
    "ets_states",
    "state_space_states",
)

# Recorded before the overlay pipeline refactor; see the module docstring.
EXPECTED: dict[tuple[str, str], str] = {
    ("batch", "all"): "70460d9f720e8270",
    ("batch", "off"): "0c12174b57270354",
    ("batch", "relative"): "791c7768fa8aa36b",
    ("batch_jobs", "all"): "70460d9f720e8270",
    ("batch_jobs", "off"): "0c12174b57270354",
    ("batch_jobs", "relative"): "791c7768fa8aa36b",
    ("batch_values", "all"): "c78de70aec815d72",
    ("batch_values", "off"): "7773bdc6d51ddd0a",
    ("batch_values", "relative"): "413d85e12d2d7eab",
    ("threaded", "all"): "7b45b0c673a3e394",
    ("threaded", "off"): "89231e5ce0729f69",
    ("threaded", "relative"): "eeba0b2da30aa968",
    ("synset", "all"): "038a86278ae48cc13e2cb289f418",
    ("synset", "off"): "15e6cc620f2c1bb53e2cb289f418",
    ("synset", "relative"): "9c2873348f50b8e13e2cb289f418",
    ("var", "all"): "5cc641405982821a",
    ("var", "off"): "a20f6f75e6df29e6",
    ("var", "relative"): "f87c6061b7777fae",
    # Copula and TCM go through BLAS, whose last bits vary across CPUs; only the
    # overlay placement (missing mask, flags) and the generator state are pinned.
    ("copula", "all"): "5933b419a00f3d98",
    ("copula", "off"): "17dff25786f71b41",
    ("copula", "relative"): "42d2f1be0dcd1c59",
    ("dau", "all"): "d7260fee882f037e",
    ("dau", "off"): "18e72408e5052b55",
    ("dau", "relative"): "f73a7dfd7caf7b5d",
    ("iot", "all"): "378ee97ab869797c",
    ("iot", "off"): "0f0ab72f30b10f48",
    ("iot", "relative"): "a0b520ed2427608e",
    ("tcm", "all"): "74242a3460ba36ab",
    ("tcm", "off"): "a3bb3be5a39b857a",
    ("tcm", "relative"): "3e1f34004390f8b6",
    ("ets_states", "all"): "bd8b260c64080258fce33096",
    ("ets_states", "off"): "630f0c91c922f109fce33096",
    ("ets_states", "relative"): "01b135a1d3115bb0fce33096",
    # generate_with_states used to apply only missingness; it now applies every
    # overlay to the observations (the states and their hash suffix are unchanged).
    ("state_space_states", "all"): "2054b2117a8c4319d21bd99e",
    ("state_space_states", "off"): "d8671ad50a581899d21bd99e",
    ("state_space_states", "relative"): "5aadf20c67b74456d21bd99e",
}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_released_overlays_draw_as_before(path: str, config: str) -> None:
    expected = EXPECTED.get((path, config))
    if expected is None:
        pytest.skip("no pinned draw for this combination")
    assert _draw(path, CONFIGS[config]) == expected
