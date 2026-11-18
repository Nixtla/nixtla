"""Post-hoc diagnosis of the calendar re-check failure (written after seeing it).

Hypothesis: #98 sized the additive calendar on the standard deviation of the
series after changepoints and anomalies; the new pipeline sizes it on the base
series. If so, the drop concentrates in calendar draws that also have
changepoints or anomalies, and additive draws without them barely move.

Replicates tsfm's v4 generate() for Hourly to keep the generator, then splits
the median daily seasonal strength by group.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.environ["TSFM_ROOT"], "tmp/subdaily_diag"))
import subdaily_seasonality as SS  # noqa: E402
from tsfm_train import cpm_synforecast as C  # noqa: E402

import synforecast  # noqa: E402
from synforecast import pretraining_pool  # noqa: E402

NEW = hasattr(synforecast, "OverlayConfig")
FREQ, LAG = "Hourly", 24


def config(gen):
    if NEW:
        o = gen.overlays
        cal = o.calendar
        return {
            "calendar": cal is not None,
            "multiplicative": cal is not None and cal.mode == "multiplicative",
            "structural": o.changepoints is not None or o.anomalies is not None,
        }
    return {
        "calendar": gen.calendar_cycles,
        "multiplicative": gen.calendar_cycles and gen.calendar_mode == "multiplicative",
        "structural": gen.changepoints or gen.anomalies,
    }


def draw(member, index):
    rng = np.random.default_rng(np.random.SeedSequence([7, 0, index, 7, member]))
    pool_seed = int(rng.integers(0, 2**62))
    overlay_seed = int(rng.integers(0, 2**62))
    step = pd.Timedelta("1h")
    start = str(
        pd.Timestamp("2000-01-01")
        + step * int(rng.integers(0, pd.Timedelta(days=7) // step))
    )
    gen = pretraining_pool(
        min_length=SS.LENGTH,
        max_length=SS.LENGTH,
        freq="h",
        seed=pool_seed,
        members=[member],
        include_extended=True,
        overlay_seed=overlay_seed,
        calendar_rate=C.CALENDAR_RATES[FREQ],
        start_datetime=start,
    )[0]
    return gen.generate_values(1, n_jobs=1)[0], config(gen)


rows = []
for member in range(len(C.members_of(C.VERSION_V4))):
    for i in range(SS.PER_MEMBER):
        values, cfg = draw(member, i)
        rows.append(dict(cfg, strength=SS.seasonal_strength(values, LAG)))
Path(sys.argv[1]).write_text(json.dumps(rows))
