"""Held-out validation of the chosen calendar-prior adjustment (declared in calendar_select.py).

Seed 8 draws (never used for selection), tsfm's v4 recipe, library code as
installed (no patching). Run once with the pinned #98 build and once with the
adjusted build, then compare:

    $TSFM_ROOT/.venv/bin/python calendar_validate.py run <out.json>
    PYTHONPATH=<adjusted build> $TSFM_ROOT/.venv/bin/python calendar_validate.py run <out.json>
    $TSFM_ROOT/.venv/bin/python calendar_validate.py verdict <pinned.json> <adjusted.json>

Pass: |median daily strength| and |share with a daily cycle| within 0.05 of
#98's at Hourly, 30Minutely, 15Minutely and 10Minutely. Also reported: v3
(no calendar) control.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ["TSFM_ROOT"], "tmp/subdaily_diag"))
import subdaily_seasonality as SS  # noqa: E402
from tsfm_train import cpm_synforecast as C  # noqa: E402

FREQS = ("Hourly", "30Minutely", "15Minutely", "10Minutely")
SEED = 8


def summary(freq: str, version: str) -> dict:
    period, daily, weekly, _ = SS.FREQS[freq]
    draws = [
        SS.generate(idx, freq, i, seed=SEED, length=SS.LENGTH, version=version)[0]
        for idx in range(len(SS.members_of(version)))
        for i in range(SS.PER_MEMBER)
    ]
    d = SS.summarize(draws, (period, daily, weekly))["daily"]
    return {"median": d["strength_quartiles"][1], "share": d["share_present"]}


def run(out: str) -> None:
    result = {
        tag: {f: summary(f, v) for f in FREQS}
        for tag, v in (("v3", C.VERSION_V3), ("v4", C.VERSION_V4))
    }
    Path(out).write_text(json.dumps(result, indent=1))


def cell(r: dict) -> str:
    return f"{r['median']:.2f} / {r['share']:.2f}"


def verdict(pinned_path: str, adjusted_path: str) -> None:
    a, b = (
        json.loads(Path(pinned_path).read_text()),
        json.loads(Path(adjusted_path).read_text()),
    )
    ok = True
    print(
        f"{'freq':11s} {'v3 #98':>12s} {'v3 adj':>12s} {'v4 #98':>12s} {'v4 adj':>12s}  strength share"
    )
    for f in FREQS:
        c1 = abs(b["v4"][f]["median"] - a["v4"][f]["median"]) <= 0.05
        c2 = abs(b["v4"][f]["share"] - a["v4"][f]["share"]) <= 0.05
        ok &= c1 and c2
        print(
            f"{f:11s} {cell(a['v3'][f]):>12s} {cell(b['v3'][f]):>12s} {cell(a['v4'][f]):>12s} {cell(b['v4'][f]):>12s}"
            f"  {'pass' if c1 else 'FAIL':8s} {'pass' if c2 else 'FAIL'}"
        )
    print("VALIDATION:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else verdict(sys.argv[2], sys.argv[3])
