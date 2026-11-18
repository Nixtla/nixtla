"""Compare two `mem_pooled.py` result files and render a markdown report.

    python tests/mem_compare.py <base.json> <pr.json>

Writes the table to stdout, and appends it to $GITHUB_STEP_SUMMARY when set.
Always exits 0: this reports, it does not gate.

Both sides are measured on the same runner in one job, so the interpreter and
dependency cost cancels in the delta. `baseline_rss_mb` -- the RSS of a process that
builds the panel and stops -- is the control: it measures identical code on both
sides, so if it moves, the runner was noisy and the table is not worth reading.
"""

import json
import os
import pathlib
import sys

# a row is flagged only if it clears both bars: percent alone is noise on small
# configs, absolute alone hides a big config's drift
PCT_THRESHOLD = 5.0
MB_THRESHOLD = 20.0

#: control drift above this means the two runs aren't comparable
CONTROL_DRIFT_MB = 10.0

MARKER = "<!-- pooled-memory-report -->"


def _load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _control(data):
    """RSS of the panel-only process, identical across configs in a run."""
    return next((v["baseline_rss_mb"] for v in data.values()), None)


def _rows(base, pr):
    for name in sorted(pr):
        new = pr[name]["peak_rss_mb"]
        if name not in base:
            yield f"| `{name}` | new | {new:.0f} | | |"
            continue
        old = base[name]["peak_rss_mb"]
        delta = new - old
        pct = 100 * delta / old if old else 0.0
        flag = " ⚠️" if delta > MB_THRESHOLD and pct > PCT_THRESHOLD else ""
        yield f"| `{name}` | {old:.0f} | {new:.0f} | {delta:+.0f}{flag} | {pct:+.1f}% |"


def report(base, pr):
    out = [MARKER, "## Pooled peak memory"]
    if not pr:
        out.append("No results: the PR run produced nothing.")
        return "\n".join(out)
    if not base:
        out.append("Baseline unavailable (the base run failed). PR numbers only:\n")
        out.append("| config | peak RSS (MB) |")
        out.append("|---|--:|")
        out += [f"| `{n}` | {v['peak_rss_mb']:.0f} |" for n, v in sorted(pr.items())]
        return "\n".join(out)

    base_ctl, pr_ctl = _control(base), _control(pr)
    drift = pr_ctl - base_ctl
    n_rows = next(iter(pr.values()))["n_rows"]
    out.append(
        f"Peak RSS per config, {n_rows:,} rows, both sides measured on this runner."
    )
    if abs(drift) > CONTROL_DRIFT_MB:
        out.append(
            f"\n⚠️ **Noisy run**: the panel-only control moved {drift:+.0f} MB "
            f"({base_ctl:.0f} → {pr_ctl:.0f} MB). It measures identical code on both "
            "sides, so the deltas below are not trustworthy."
        )
    out.append("")
    out.append("| config | base (MB) | PR (MB) | Δ MB | Δ % |")
    out.append("|---|--:|--:|--:|--:|")
    out += list(_rows(base, pr))
    out.append("")
    out.append(
        f"Panel-only control: {base_ctl:.0f} → {pr_ctl:.0f} MB ({drift:+.0f}). "
        f"⚠️ marks a config over +{MB_THRESHOLD:.0f} MB and +{PCT_THRESHOLD:.0f}%."
    )
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <base.json> <pr.json>")
    text = report(_load(sys.argv[1]), _load(sys.argv[2]))
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
