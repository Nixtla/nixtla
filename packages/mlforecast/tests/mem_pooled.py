"""Peak process memory for pooled lag transforms, run by the memory-tests job.

Not named `test_*.py` so the ordinary pytest run skips it; the job passes the
path explicitly. Each config runs in a spawned process and reports its own
high-water RSS, so transient allocations count and configs can't inherit each
other's peak. Reported, not asserted, unless MLF_MEM_BUDGET_MB sets a ceiling.

    MLF_MEM_SERIES=1000 MLF_MEM_TIMES=1000 pytest tests/mem_pooled.py -s
    MLF_MEM_JSON=mem.json pytest tests/mem_pooled.py   # results to a known path
"""

import json
import multiprocessing
import os
import pathlib
import sys

import pytest

from tests._pooled_common import (
    CONFIG_BY_NAME,
    CONFIG_IDS,
    CONFIGS,
    HORIZON,
    build_future,
    build_series,
    frame,
    make_forecast,
)

# Larger than the CodSpeed benchmarks: the pooled work has to be a visible
# fraction of the interpreter baseline for a regression to stand out.
N_SERIES = int(os.environ.get("MLF_MEM_SERIES", 600))
N_TIMES = int(os.environ.get("MLF_MEM_TIMES", 600))

MEM_BUDGET_MB = float(os.environ.get("MLF_MEM_BUDGET_MB", 0)) or None
MEM_JSON = os.environ.get("MLF_MEM_JSON")


def _child(conn, config_name):
    """Peak RSS of one config; ``config_name is None`` builds the panel only."""
    import resource

    df = build_series(N_SERIES, N_TIMES)
    if config_name is not None:
        config = CONFIG_BY_NAME[config_name]
        x = build_future(df) if config.needs_future else None
        fcst = make_forecast(config, with_model=True)
        fcst.fit(frame(df, config), static_features=["brand"], dropna=False)
        fcst.predict(h=HORIZON, X_df=x)
    conn.send(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    conn.close()


def _peak_rss_mb(config_name):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, config_name))
    proc.start()
    child.close()
    try:
        value = parent.recv()
    finally:
        proc.join(900)
    assert proc.exitcode == 0, f"{config_name}: child exited {proc.exitcode}"
    return value


@pytest.fixture(scope="module")
def baseline_rss_mb():
    """Peak RSS of a process that only builds the panel."""
    return _peak_rss_mb(None)


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_pooled_peak_memory(record_property, config, baseline_rss_mb, tmp_path_factory):
    """Peak process memory for fit + predict, measured in isolation."""
    peak = _peak_rss_mb(config.name)
    over_baseline = peak - baseline_rss_mb

    record_property("peak_rss_mb", round(peak, 1))
    record_property("baseline_rss_mb", round(baseline_rss_mb, 1))
    record_property("over_baseline_mb", round(over_baseline, 1))
    record_property("n_rows", N_SERIES * N_TIMES)

    out = (
        pathlib.Path(MEM_JSON)
        if MEM_JSON
        else pathlib.Path(tmp_path_factory.getbasetemp()) / "pooled_memory.json"
    )
    data = json.loads(out.read_text()) if out.exists() else {}
    data[config.name] = {
        "peak_rss_mb": round(peak, 1),
        "baseline_rss_mb": round(baseline_rss_mb, 1),
        "over_baseline_mb": round(over_baseline, 1),
        "n_rows": N_SERIES * N_TIMES,
    }
    out.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(
        f"\n{config.name}: peak RSS {peak:.0f} MB "
        f"({over_baseline:+.0f} MB over the {baseline_rss_mb:.0f} MB data baseline)",
        file=sys.stderr,
    )
    if MEM_BUDGET_MB is not None:
        assert peak <= MEM_BUDGET_MB, (
            f"{config.name}: peak RSS {peak:.0f} MB exceeds budget {MEM_BUDGET_MB:.0f} MB"
        )
