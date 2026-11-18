"""Select a calendar-prior adjustment that recovers #98's daily seasonal strength.

Declared before any candidate was evaluated; never retuned afterwards.

Why: the restructured pipeline adds the calendar before changepoints and
anomalies and sizes it on the base series, so in draws that also have breaks
the daily cycle is diluted (hourly median strength 0.69 -> 0.63). Olivier asked
to recover the strength through the prior.

Candidates (15), changing only ranges, never the number or order of draws:
- additive amplitude log-uniform in [k, 8k] for k in (1, 1.25, 1.5, 2, 2.5);
- multiplicative swing uniform in (0.2, 0.9), (0.3, 0.95) or (0.4, 0.95).

Design set: tsfm's v4 recipe, seed 7 (the draws of the failed re-check), for
Hourly, 30Minutely, 15Minutely and 10Minutely; synthetic only.
Targets: #98's v4 values on the same draws (recheck_pinned.json).
Eligible: |median daily strength - target| <= 0.03 and |share with a daily
cycle - target| <= 0.05 at all four frequencies.
Choice: the eligible candidate with the smallest change (k ascending, then the
swing ranges in the order listed). If none is eligible, report the closest and
change nothing.
Validation (calendar_validate.py, after the library change): seed 8 draws,
library code, original criteria (+/-0.05 strength and share vs #98 on the same
draws) at all four frequencies.

    PYTHONPATH=<new build> $TSFM_ROOT/.venv/bin/python calendar_select.py <recheck_pinned.json> <out.json>
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.environ["TSFM_ROOT"], "tmp/subdaily_diag"))
import subdaily_seasonality as SS  # noqa: E402
from tsfm_train import cpm_synforecast as C  # noqa: E402

import synforecast.presets as P  # noqa: E402
from synforecast.overlays import Calendar, CalendarCycle  # noqa: E402

FREQS = ("Hourly", "30Minutely", "15Minutely", "10Minutely")
KS = (1.0, 1.25, 1.5, 2.0, 2.5)
SWINGS = ((0.2, 0.9), (0.3, 0.95), (0.4, 0.95))


def prior(k: float, swing: tuple[float, float]):
    """calendar_overlays with scaled ranges; identical draws otherwise."""

    def calendar_overlays(rng: np.random.Generator) -> Calendar:
        multiplicative = rng.random() < 0.5
        amplitude = P._log_uniform(rng, 1.0 * k, 8.0 * k)
        drawn_swing = float(rng.uniform(*swing))
        weekly = 0.0 if rng.random() < 0.3 else P._log_uniform(rng, 0.2, 2.0)
        cycles = [CalendarCycle(phase="day", harmonics=int(rng.integers(1, 6)))]
        if weekly > 0:
            weekend = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.5))
            cycles.append(
                CalendarCycle(
                    phase="week",
                    profile="levels",
                    values=[0.0] * 5 + [weekend] * 2,
                    jitter=0.25,
                    weight=weekly,
                )
            )
        return Calendar(
            cycles=cycles,
            mode="multiplicative" if multiplicative else "additive",
            amplitude=amplitude,
            swing=drawn_swing,
            on_unsupported="skip",
        )

    return calendar_overlays


def daily_summary(freq: str, seed: int) -> dict:
    period, daily, weekly, _ = SS.FREQS[freq]
    members = SS.members_of(C.VERSION_V4)
    draws = [
        SS.generate(idx, freq, i, seed=seed, length=SS.LENGTH, version=C.VERSION_V4)[0]
        for idx in range(len(members))
        for i in range(SS.PER_MEMBER)
    ]
    d = SS.summarize(draws, (period, daily, weekly))["daily"]
    return {"median": d["strength_quartiles"][1], "share": d["share_present"]}


def main(pinned_path: str, out: str) -> None:
    pinned = json.loads(Path(pinned_path).read_text())["v4"]
    targets = {
        f: {
            "median": pinned[f]["synthetic"]["daily"]["strength_quartiles"][1],
            "share": pinned[f]["synthetic"]["daily"]["share_present"],
        }
        for f in FREQS
    }
    original = P.calendar_overlays
    rows = []
    for k, swing in itertools.product(KS, SWINGS):
        P.calendar_overlays = prior(k, swing)
        got = {f: daily_summary(f, seed=7) for f in FREQS}
        dev_median = max(abs(got[f]["median"] - targets[f]["median"]) for f in FREQS)
        dev_share = max(abs(got[f]["share"] - targets[f]["share"]) for f in FREQS)
        row = {
            "k": k,
            "swing": swing,
            "got": got,
            "dev_median": dev_median,
            "dev_share": dev_share,
            "eligible": dev_median <= 0.03 and dev_share <= 0.05,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in ("k", "swing", "dev_median", "dev_share", "eligible")
                }
            ),
            flush=True,
        )
    P.calendar_overlays = original
    eligible = [r for r in rows if r["eligible"]]
    chosen = eligible[0] if eligible else min(rows, key=lambda r: r["dev_median"])
    Path(out).write_text(
        json.dumps(
            {
                "targets": targets,
                "rows": rows,
                "chosen": chosen,
                "any_eligible": bool(eligible),
            },
            indent=1,
        )
    )
    print(
        "CHOSEN:",
        chosen["k"],
        chosen["swing"],
        "eligible" if eligible else "NONE ELIGIBLE",
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
