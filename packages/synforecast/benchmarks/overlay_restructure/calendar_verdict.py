"""Apply the criteria declared in calendar_recheck.py to its two outputs."""

import json
import os
import sys
from pathlib import Path

S = sys.argv[1]
STORED = os.path.join(
    os.environ["TSFM_ROOT"], "tmp/subdaily_diag/subdaily_seasonality_{}.json"
)
JUDGED = ("Hourly", "30Minutely", "15Minutely", "10Minutely")
TOL = 0.05

pinned = json.loads(Path(f"{S}/recheck_pinned.json").read_text())
new = json.loads(Path(f"{S}/recheck_new.json").read_text())

replay = {
    tag: pinned[tag] == json.loads(Path(STORED.format(tag)).read_text())
    for tag in ("v3", "v4")
}
print("replay gate (pinned run == stored tsfm JSON):", replay)
if not all(replay.values()):
    for tag in ("v3", "v4"):
        stored = json.loads(Path(STORED.format(tag)).read_text())
        for freq in stored:
            for part in ("synthetic", "real"):
                if stored[freq][part] != pinned[tag][freq][part]:
                    print("  differs:", tag, freq, part)


def daily(report, freq, part="synthetic"):
    d = report[freq][part] and report[freq][part].get("daily")
    return (None, None) if not d else (d["strength_quartiles"][1], d["share_present"])


def fmt(c: tuple) -> str:
    return "      -     " if c[0] is None else f"{c[0]:.2f} / {c[1]:.2f}"


rows, passed = [], True
print(
    f"\n{'freq':11s} {'real':>12s} {'v3 #98':>12s} {'v3 new':>12s} {'v4 #98':>12s} {'v4 new':>12s}  C1   C2"
)
for freq in pinned["v4"]:
    real = daily(pinned["v4"], freq, "real")
    cells = [
        daily(pinned["v3"], freq),
        daily(new["v3"], freq),
        daily(pinned["v4"], freq),
        daily(new["v4"], freq),
    ]
    verdict = ""
    if freq in JUDGED:
        c1 = abs(cells[3][0] - cells[2][0]) <= TOL
        c2 = abs(cells[3][1] - cells[2][1]) <= TOL
        passed &= c1 and c2
        verdict = f"{'pass' if c1 else 'FAIL':4s} {'pass' if c2 else 'FAIL'}"
    print(
        f"{freq:11s} {fmt(real):>12s} {fmt(cells[0]):>12s} {fmt(cells[1]):>12s} {fmt(cells[2]):>12s} {fmt(cells[3]):>12s}  {verdict}"
    )
print("\ncells: median daily seasonal strength / share with a daily cycle")
print("OVERALL:", "PASS" if passed and all(replay.values()) else "FAIL")
