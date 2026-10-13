"""Overlays are orthogonal to the processes they decorate, and to each other.

Every generator accepts changepoints, anomalies and missingness as fields on
the base class. For a controlled perturbation experiment, toggling an overlay
must change the sample in exactly one respect:

* the base trajectory is untouched: every observation the overlay did not
  touch is bit-identical to the no-overlay run, and series lengths are
  unchanged (so the overlay cannot shift a later series' draws either);
* the other overlays are untouched: where an overlay lands does not depend
  on which other overlays are enabled.

Touched positions come from the exogenous flag columns rather than from the
values, so the test does not assume its conclusion. Changepoints are
structural (a break alters everything after it), so for them only the prefix
before the first changepoint must survive. A level shift moves
``level_shift_duration`` observations but is flagged at its start only, so its
span is widened here by the duration.
"""

from __future__ import annotations

import numpy as np
import pytest

import synforecast.generators as generators
from synforecast.exogenous import ExogenousConfig

from .helpers import to_pandas

N_SERIES = 4
SEED = 20260807
LEVEL_SHIFT_DURATION = 5
BASE = {"min_length": 120, "max_length": 160, "freq": "D", "seed": SEED}

OVERLAYS = {
    "missing": {"missing_data": True, "missing_rate": 0.1, "missing_pattern": "random"},
    "anomaly": {
        "anomalies": True,
        "anomaly_fraction": 0.05,
        "anomaly_types": ["spike", "dip", "level_shift"],
        "level_shift_duration": LEVEL_SHIFT_DURATION,
    },
    "changepoint": {
        "changepoints": True,
        "num_changepoints": 2,
        "changepoint_type": "level",
    },
}
FLAGS = ExogenousConfig(missing_flags=True, anomaly_flags=True, changepoint_flags=True)

GENERATORS = pytest.mark.parametrize(
    "cls",
    [getattr(generators, name) for name in generators.__all__],
    ids=list(generators.__all__),
)


def _generate(cls, overlays: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """{column: {series_id: values in time order}} for y and the three flags."""
    kwargs = {**BASE, "exogenous": FLAGS}
    for name in overlays:
        kwargs.update(OVERLAYS[name])
    pdf = to_pandas(cls(**kwargs).generate(n_series=N_SERIES, n_jobs=2))
    pdf = pdf.sort_values(["unique_id", "ds"])
    cols = ["y", "missing_flag", "anomaly_flag", "changepoint_flag"]
    return {
        col: {
            str(uid): group[col].to_numpy()
            for uid, group in pdf.groupby("unique_id", observed=True)
        }
        for col in cols
    }


def _touched(flags: dict[str, np.ndarray], overlay: str) -> dict[str, np.ndarray]:
    """Mask of positions the overlay may have changed, per series."""
    out = {}
    for uid, flag in flags[f"{overlay}_flag"].items():
        hit = flag.astype(bool)
        if overlay == "changepoint":
            idx = np.flatnonzero(hit)
            mask = np.zeros_like(hit)
            if len(idx):
                mask[idx[0] :] = True
        elif overlay == "anomaly":
            mask = hit.copy()
            for start in np.flatnonzero(hit):
                mask[start : start + LEVEL_SHIFT_DURATION] = True
        else:
            mask = hit
        out[uid] = mask
    return out


@GENERATORS
@pytest.mark.parametrize("overlay", list(OVERLAYS))
def test_overlay_preserves_base_trajectory(cls, overlay):
    base = _generate(cls, ())["y"]
    out = _generate(cls, (overlay,))
    touched = _touched(out, overlay)

    assert base.keys() == out["y"].keys()
    for uid, b in base.items():
        g = out["y"][uid]
        assert len(g) == len(b), f"{uid}: length changed {len(b)} -> {len(g)}"
        keep = ~touched[uid]
        # Positions that are NaN without the overlay (e.g. simulated sensor
        # failures) cannot be compared with ==, but must stay NaN.
        np.testing.assert_array_equal(np.isnan(g[keep]), np.isnan(b[keep]))
        finite = keep & ~np.isnan(b)
        assert np.array_equal(g[finite], b[finite]), (
            f"{uid}: {int((g[finite] != b[finite]).sum())} untouched "
            f"observations changed under {overlay}"
        )
    assert any(m.any() for m in touched.values()), f"{overlay} touched nothing"


@GENERATORS
def test_overlay_placement_independent_of_other_overlays(cls):
    combined = _generate(cls, tuple(OVERLAYS))
    for overlay in OVERLAYS:
        alone = _generate(cls, (overlay,))
        col = f"{overlay}_flag"
        for uid, flags in alone[col].items():
            np.testing.assert_array_equal(
                combined[col][uid],
                flags,
                err_msg=f"{uid}: {overlay} placement moved when other overlays "
                "were enabled",
            )
