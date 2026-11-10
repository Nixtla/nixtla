"""Overlay throughput: PR #98 head (0edf7bc) against the restructured pipeline (9a21d38).

Run once per build, each in its own Python 3.13 venv, one after the other:
    <venv>/bin/python bench_overlays.py <label> <out.json>

Declared before any run (expectations, not pass/fail):
- Overlays off: both builds run the same native kernels, so times should match
  within noise (no regression from the plan object).
- Waveform/observation on: #98 runs them per series in a serial Python loop
  after the parallel kernel; the new build runs them inside rayon. Expect the
  gap to grow with the share of affected series and to vanish at n_jobs=1.
- Calendar on: #98 adds it in a serial per-series post-pass; the new build in
  the kernel. Same expectation.
Timing: one warm-up, then the median of 3 repeats; wall clock.
"""

import json
import statistics
import sys
import time
import warnings

import numpy as np

import synforecast
from synforecast import pretraining_pool
from synforecast.generators import RandomWalkGenerator

warnings.simplefilter("ignore")
NEW_API = hasattr(synforecast, "OverlayConfig")
CORES = 20


def timed(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def walk(overlay: str, length: int):
    base = {"min_length": length, "max_length": length, "freq": "10min", "seed": 1}
    if overlay == "off":
        return RandomWalkGenerator(**base)
    if NEW_API:
        from synforecast import (
            Calendar,
            CalendarCycle,
            Observation,
            OverlayConfig,
            Waveform,
        )

        config = {
            "calendar": OverlayConfig(
                calendar=Calendar(
                    cycles=[
                        CalendarCycle(phase="day", harmonics=3),
                        CalendarCycle(phase="week", profile="levels", weight=0.5),
                    ]
                )
            ),
            "waveform+observation": OverlayConfig(
                waveform=Waveform(period=96, timing="jitter"),
                observation=Observation(quantum=0.1, hold=3, stuck_fraction=0.05),
            ),
        }[overlay]
        return RandomWalkGenerator(**base, overlays=config)
    kwargs = {
        "calendar": {
            "calendar_cycles": True,
            "calendar_harmonics": 3,
            "calendar_weekly": 0.5,
        },
        "waveform+observation": {
            "waveform": True,
            "waveform_period": 96,
            "waveform_timing": "jitter",
            "observation": True,
            "observation_quantum": 0.1,
            "observation_hold": 3,
            "observation_stuck_fraction": 0.05,
        },
    }[overlay]
    return RandomWalkGenerator(**base, **kwargs)


def main(label: str, out: str) -> None:
    results = {
        "label": label,
        "version": synforecast.__version__,
        "new_api": NEW_API,
        "rows": [],
    }

    def record(
        case: str, n_jobs: int, seconds: float, n_series: int, length: int
    ) -> None:
        row = {
            "case": case,
            "n_jobs": n_jobs,
            "seconds": round(seconds, 4),
            "n_series": n_series,
            "length": length,
        }
        results["rows"].append(row)
        print(json.dumps(row), flush=True)

    # Single generator, values only: isolates the overlay pipeline.
    n_series, length = 4000, 2048
    for overlay in ("off", "calendar", "waveform+observation"):
        for n_jobs in (CORES, 1):
            gen = walk(overlay, length)
            seconds = timed(lambda: gen.generate_values(n_series, n_jobs=n_jobs))
            record(f"random_walk/{overlay}", n_jobs, seconds, n_series, length)

    # The extended pretraining pool, each member drawn with generate_values as tsfm draws it
    # (#98's SynSet cannot concatenate this pool's generator-specific columns).
    per_gen, length = 60, 1024
    for calendar_rate, observability_rate in (
        (0.0, 0.0),
        (0.8, 0.0),
        (0.0, 0.3),
        (0.0, 1.0),
        (0.8, 0.3),
    ):
        pool = pretraining_pool(
            min_length=length,
            max_length=length,
            freq="10min",
            seed=1,
            include_extended=True,
            overlay_seed=7,
            calendar_rate=calendar_rate,
            observability_rate=observability_rate,
        )
        for n_jobs in (CORES, 1):
            seconds = timed(
                lambda: [g.generate_values(per_gen, n_jobs=n_jobs) for g in pool]
            )
            record(
                f"pool/calendar={calendar_rate}/observability={observability_rate}",
                n_jobs,
                seconds,
                per_gen * len(pool),
                length,
            )
    with open(out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(sys.argv[1], sys.argv[2])
