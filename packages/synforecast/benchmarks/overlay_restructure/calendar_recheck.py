"""Does the restructured calendar prior still give sub-daily synthetic series the daily cycle #98 measured?

Re-runs tsfm's tmp/subdaily_diag/subdaily_seasonality.py recipe unchanged (same real
buckets and sampling, same v3/v4 draw recipe through tsfm_train.cpm_synforecast,
same seasonal-strength and difference-ACF definitions) against whichever
synforecast is first on the path:

    $TSFM_ROOT/.venv/bin/python calendar_recheck.py <label> <out.json>              # pinned 5f26ef0 (#98)
    PYTHONPATH=<new build> $TSFM_ROOT/.venv/bin/python calendar_recheck.py ...      # 9a21d38

Declared before any result was computed; never retuned afterwards.
- Replay gate: the pinned run must reproduce tsfm's stored
  subdaily_seasonality_v3.json and _v4.json exactly (synthetic and real), else
  the comparison is void.
- C1: for each of Hourly, 30Minutely, 15Minutely, 10Minutely, the new v4 median
  daily seasonal strength is within +/-0.05 of the #98 v4 median.
- C2: same frequencies, the share of synthetic series with a daily cycle
  (difference ACF at the daily lag > 0.2) is within +/-0.05 of #98's.
- Pass = C1 and C2 at all four. Minutely and Daily are reported, not judged.
- Control (reported): v3 new vs v3 pinned. The new build also carries main's
  KernelSynth / periodic GaussianProcess changes, so v3 can move; v4 shifts
  are read against it.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.environ["TSFM_ROOT"], "tmp/subdaily_diag"))
import subdaily_seasonality as SS  # noqa: E402
from tsfm_train import cpm_synforecast as C  # noqa: E402

import synforecast  # noqa: E402


def stream_report(tag: str) -> dict:
    """SS.main() for one stream, returned instead of written into the tsfm tree."""
    rng = np.random.default_rng(0)
    stream = {"v3": C.VERSION_V3, "v4": C.VERSION_V4}[tag]
    members = SS.members_of(stream)
    report = {}
    for freq, (period, daily, weekly, buckets) in SS.FREQS.items():
        synthetic, by_member = [], {}
        for idx, m in enumerate(members):
            draws = [
                SS.generate(idx, freq, i, seed=7, length=SS.LENGTH, version=stream)[0]
                for i in range(SS.PER_MEMBER)
            ]
            synthetic += draws
            by_member[m] = SS.summarize(draws, (period, daily, weekly))
        per_bucket = {b: SS.real_series(b, rng) for b in buckets}
        real = [s for b in buckets for s in per_bucket[b]]
        report[freq] = {
            "synthetic": dict(
                n=len(synthetic), **SS.summarize(synthetic, (period, daily, weekly))
            ),
            "real": dict(n=len(real), **SS.summarize(real, (period, daily, weekly)))
            if real
            else None,
            "real_by_bucket": {
                b: dict(n=len(v), **SS.summarize(v, (period, daily, weekly)))
                for b, v in per_bucket.items()
            },
            "synthetic_members_with_daily": sorted(
                m
                for m, s in by_member.items()
                if s["daily"] and s["daily"]["share_present"] > 0.5
            ),
        }
        print(tag, freq, json.dumps(report[freq]["synthetic"].get("daily")), flush=True)
    return report


def main(label: str, out: str) -> None:
    result = {"label": label, "synforecast": synforecast.__file__}
    for tag in ("v3", "v4"):
        result[tag] = stream_report(tag)
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
