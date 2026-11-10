# Overlay restructure: throughput and calendar prior (2026-11-02)

Two checks of PR #102 against the closed PR #98 it replaces.

## What held and what did not

- **Throughput held.**
  - Waveform, observation and calendar stages run 33–38× faster on a 20-core machine.
  - They also run 5–7× faster on a single core.
  - With overlays off, the two builds take the same time.
- **The calendar prior first failed, then recovered after an adjustment.**
  - As first restructured, the prior gave sub-daily draws a weaker daily cycle than #98. Median daily seasonal strength fell by 0.06–0.08, beyond the tolerance of ±0.05 declared in advance.
  - The drop is a side effect of the new stage order. After widening the prior's additive peak from [1, 8] to [1.25, 10] standard deviations, fresh draws match #98 within 0.02.

## Throughput

Script: [`throughput.py`](../../overlay_restructure/throughput.py). Results: [`throughput_summary.json`](throughput_summary.json).

**Setup:**
- Builds: #98 head (`0edf7bc`) and #102 (`9a21d38`), each installed in its own Python 3.13 environment.
- Runs: one warm-up, then the median of 3; the two builds ran one after the other.
- The pool is drawn member by member with `generate_values`, as tsfm draws it. #98's `SynSet` cannot combine the extended pool's generator-specific columns.

| Case | Cores | #98 (s) | #102 (s) | Speed-up |
|---|---|---|---|---|
| RandomWalk 4,000 × 2,048, overlays off | 20 / 1 | 0.020 / 0.071 | 0.020 / 0.070 | 1.0× / 1.0× |
| RandomWalk, calendar (day + week) | 20 / 1 | 1.42 / 1.46 | 0.043 / 0.28 | 33× / 5.2× |
| RandomWalk, waveform + observation | 20 / 1 | 1.09 / 1.14 | 0.029 / 0.17 | 38× / 6.9× |
| Extended pool 65 × 60 × 1,024, rates 0 | 20 / 1 | 0.14 / 0.54 | 0.14 / 0.53 | 1.0× / 1.0× |
| Pool, calendar rate 0.8 | 20 / 1 | 0.88 / 1.27 | 0.21 / 0.71 | 4.1× / 1.8× |
| Pool, observability rate 0.3 | 20 / 1 | 0.28 / 0.67 | 0.15 / 0.57 | 1.9× / 1.2× |
| Pool, observability rate 1.0 | 20 / 1 | 0.79 / 1.16 | 0.17 / 0.58 | 4.6× / 2.0× |
| Pool, calendar 0.8 + observability 0.3 | 20 / 1 | 1.04 / 1.38 | 0.24 / 0.71 | 4.4× / 2.0× |

**Reading the table:**
- The expectation declared beforehand was that the gap would vanish on one core. It does not: the Rust stages are also faster per series than #98's NumPy code.
- Pool gains are smaller because the per-member call overhead dominates at this size.

## Calendar prior

Scripts: [`calendar_*.py`](../../overlay_restructure/). Results: [`calendar_prior_summary.json`](calendar_prior_summary.json).

The scripts reuse tsfm's sub-daily seasonality analysis (`tmp/subdaily_diag/subdaily_seasonality.py`) unchanged:
- **Real series:** real `/timenet` training buckets at Hourly, 30-, 15- and 10-minute frequencies.
- **Synthetic series:** tsfm's v3 stream (no calendar) and v4 stream (calendar rate 0.8), 65 pool members × 25 draws of length 2,273 each.
- **Measures:**
  - median daily seasonal strength, 1 − Var(remainder)/Var(detrended);
  - the share of series whose differenced autocorrelation at the daily lag exceeds 0.2.

Each step's criteria were written into its script before it ran and were not changed afterwards.

**1. Re-check of the restructured prior** (`calendar_recheck.py`, `calendar_verdict.py`; draws with seed 7). Criterion: the new v4 is within ±0.05 of #98 on both measures at all four frequencies.
- **Replay gate: passed.** The pinned #98 build (`5f26ef0`) reproduced tsfm's stored results exactly.
- **Verdict: failed.**

| Median strength / share | Real | v3 #98 / new | v4 #98 | v4 new |
|---|---|---|---|---|
| Hourly | 0.77 / 0.84 | 0.01 / 0.01 | 0.69 / 0.54 | 0.63 / 0.49 |
| 30-minute | 0.30 / 0.10 | 0.02 / 0.02 | 0.67 / 0.37 | 0.60 / 0.33 |
| 15-minute | 0.49 / 0.25 | 0.04 / 0.04 | 0.65 / 0.25 | 0.58 / 0.24 |
| 10-minute | 0.69 / 0.68 | 0.07 / 0.07 | 0.64 / 0.19 | 0.57 / 0.18 |

v3 is identical in both builds, so the drop comes from the calendar change, not from other generator changes.

**2. Diagnosis** (`calendar_diagnose.py`; post hoc, written after seeing the failure). Hourly median daily strength by group, with the same 1,625 draws in each group for both builds:

| Group | n | #98 | new |
|---|---|---|---|
| No calendar | 314 | 0.01 | 0.01 |
| Calendar, additive, no changepoints/anomalies | 331 | 0.81 | 0.80 |
| Calendar, multiplicative, no changepoints/anomalies | 338 | 0.77 | 0.76 |
| Calendar, additive, with changepoints/anomalies | 327 | 0.79 | 0.70 |
| Calendar, multiplicative, with changepoints/anomalies | 315 | 0.78 | 0.66 |

The drop is confined to draws that also have breaks or anomalies:
- #98 added the calendar last and sized it on the already-broken series.
- #102 adds it before changepoints and anomalies and sizes it on the base series, so later breaks dilute it.

This order is intended. The prior was adjusted instead of the pipeline.

**3. Selection** (`calendar_select.py`; seed 7).
- **Candidates:** 15 combinations of the additive peak range [k, 8k], with k in {1, 1.25, 1.5, 2, 2.5}, and the multiplicative swing range, in {[0.2, 0.9], [0.3, 0.95], [0.4, 0.95]}.
- **Eligibility:** within ±0.03 of #98's strength and ±0.05 of its share at all four frequencies.
- **Rule:** the smallest eligible change wins.
- **Result:**
  - The three k = 1.25 candidates are eligible; k = 1 falls short, and k ≥ 1.5 overshoots.
  - The chosen prior: additive peak log-uniform in [1.25, 10] standard deviations, with the swing unchanged at [0.2, 0.9].

**4. Validation** (`calendar_validate.py`; fresh seed-8 draws never used for selection, with the shipped library code at `e942b06`). Criterion: within ±0.05 of #98 on both measures at all four frequencies. **Verdict: passed.**

| Median strength / share | v4 #98 | v4 adjusted |
|---|---|---|
| Hourly | 0.66 / 0.52 | 0.64 / 0.52 |
| 30-minute | 0.63 / 0.36 | 0.61 / 0.36 |
| 15-minute | 0.60 / 0.25 | 0.59 / 0.25 |
| 10-minute | 0.60 / 0.21 | 0.58 / 0.20 |

**Limits:**
- **No independent real data:** `/timenet` has no sub-daily training buckets beyond those above, only test splits of the same series. The held-out check therefore uses fresh synthetic draws, and the target is #98's strength, not a new fit to real data.
- **Draw-to-draw noise:** #98's own medians move by about 0.03 between seeds 7 and 8, as much as the selection tolerance.

**Reproducing:**
- **Requirements:** a tsfm checkout and its virtual environment, the `/timenet` mount, and `TSFM_ROOT` set to the tsfm checkout.
- **Builds:**
  - Run the #98 side with tsfm's pinned install.
  - For the #102 side, build a wheel for tsfm's Python (`maturin build --release -i $TSFM_ROOT/.venv/bin/python`), unpack it, and put it first on `PYTHONPATH`.
- **Recorded environment:** tsfm `a022a1f6` with local changes, Python 3.11.10, numpy 2.4.6, pandas 2.3.3. These runs did not record their environment through `benchmarks/_env.py`.
- **How the committed scripts differ from what ran:** only in the tsfm path, which is now read from `TSFM_ROOT`, and in lint fixes (dict literals, `Path` instead of bare `open`, named functions instead of lambdas). Both verdict steps, re-run on the saved results, print the same output.
