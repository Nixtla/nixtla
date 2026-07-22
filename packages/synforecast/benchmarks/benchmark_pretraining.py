"""Does pretraining on synthetic data help a neural forecaster with short histories?

Paired, multi-seed benchmark on M3 Monthly (a dataset the augmentation benchmark
deliberately does *not* touch, so a win reflects genuine transfer rather than a
tuned match). For each seed we pretrain a small NHITS on a synthetic corpus from
``pretraining_pool()`` and compare three ways of forecasting real series:

    A. from_scratch  — NHITS trained on the real training split only.
    B. zero_shot     — the synthetic-pretrained NHITS, applied to real history
                       with NO real training at all (pure transfer).
    C. pretrain_ft   — the synthetic-pretrained NHITS, then fine-tuned on the
                       real training split at a reduced learning rate.

The hypothesis, fixed in advance: pretraining helps most when each real series
has a *short* history — few training windows for a from-scratch model to learn
from — and the benefit fades as histories lengthen. So we fix the number of
series and sweep how much history each one keeps.

(An earlier version swept the *number* of series instead. That was the wrong
axis: NHITS trains on sliding windows, so even 20 full-length series yield
thousands of windows and a from-scratch model is never starved. History length
is what actually controls scarcity for a windowed model.)

Fairness controls (the discipline that turned the augmentation benchmark honest):
  * The synthetic corpus is seeded independently and never sees the real data.
  * ``from_scratch`` and ``pretrain_ft`` get the SAME real-training step budget,
    architecture, and per-series scaling — the differences are the pretrained
    initialization and the fine-tune learning rate (reduced, as fine-tuning at
    the full rate overwrites the pretrained features / negative transfer).
  * ``zero_shot`` does no real training; real history is only model input.
  * We forecast the last ``h`` months of each real series (a true holdout) and
    score seasonal MASE + sMAPE, paired per series within each seed.

We report mean MASE/sMAPE per history length with bootstrap CIs, and the paired
improvement (win rate + Wilcoxon signed-rank p) of B and C over A.

Usage:
    uv run python benchmarks/benchmark_pretraining.py             # full run
    uv run python benchmarks/benchmark_pretraining.py --quick     # fast smoke
    uv run python benchmarks/benchmark_pretraining.py --seeds 10 \
        --save-summary benchmarks/data/pretraining_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import warnings
from functools import partial
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datasetsforecast.m3 import M3  # noqa: E402
from neuralforecast import NeuralForecast  # noqa: E402
from neuralforecast.models import NHITS  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from utilsforecast.evaluation import evaluate  # noqa: E402
from utilsforecast.losses import mase, smape  # noqa: E402

from synforecast import SynSet, pretraining_pool  # noqa: E402

# M3 Monthly conventions.
FREQ = "ME"
HORIZON = 18
SEASON = 12
# A short input window so the model can still form training windows when a
# series keeps only a few dozen observations.
INPUT_SIZE = HORIZON
BASE_LR = 1e-3
# Real series in M3 Monthly span ~66-144 observations; match the synthetic range.
SYN_MIN_LEN = 66
SYN_MAX_LEN = 144
# Sentinel history length meaning "keep everything available".
FULL_HIST = 9999


def _quiet_nhits(max_steps: int, seed: int, learning_rate: float = BASE_LR) -> NHITS:
    """A small NHITS with logging/progress silenced, running on the GPU."""
    return NHITS(
        h=HORIZON,
        input_size=INPUT_SIZE,
        max_steps=max_steps,
        learning_rate=learning_rate,
        n_blocks=[1, 1, 1],
        mlp_units=[[128, 128]] * 3,
        random_seed=seed,
        enable_progress_bar=False,
        logger=False,
        accelerator="gpu",
        devices=1,
    )


def _nf(model: NHITS) -> NeuralForecast:
    return NeuralForecast(models=[model], freq=FREQ, local_scaler_type="standard")


def load_real() -> pd.DataFrame:
    """Load M3 Monthly as a long [unique_id, ds, y] pandas frame."""
    with tempfile.TemporaryDirectory() as d:
        df, *_ = M3.load(directory=d, group="Monthly")
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def make_synthetic(seed: int, n_per_generator: int) -> pd.DataFrame:
    """Generate a synthetic pretraining corpus with ``pretraining_pool``.

    Seeds are offset far from any real-data seed so the corpus is independent of
    the evaluation. Unique ids are namespaced to avoid collision with real ids.
    """
    pool = pretraining_pool(
        min_length=SYN_MIN_LEN,
        max_length=SYN_MAX_LEN,
        freq=FREQ,
        seed=100_000 + seed,
        engine="pandas",
    )
    df = SynSet(pool).generate(n_series_per_generator=n_per_generator)
    df["unique_id"] = "syn_" + df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    # Drop whole series that a generator left with NaN/inf gaps (e.g. IoT sensor
    # failures) or that are too short to form a training window — NHITS needs
    # clean, sufficiently long histories to pretrain on.
    finite = np.isfinite(df["y"].to_numpy())
    bad_ids = df.loc[~finite, "unique_id"].unique()
    long_enough = df.groupby("unique_id")["y"].transform("size") >= (
        INPUT_SIZE + HORIZON
    )
    keep = long_enough & ~df["unique_id"].isin(bad_ids)
    return df.loc[keep].reset_index(drop=True)


def split_real(
    df: pd.DataFrame, ids: list[str], hist_len: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last HORIZON obs; keep only the last ``hist_len`` for training."""
    sub = df[df["unique_id"].isin(ids)].sort_values(["unique_id", "ds"])
    test = sub.groupby("unique_id").tail(HORIZON)
    train_full = sub.drop(test.index)
    train = train_full.groupby("unique_id").tail(hist_len)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def score(
    forecast: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame, model_col: str
) -> pd.DataFrame:
    """Per-series seasonal MASE + sMAPE for one condition."""
    merged = test.merge(
        forecast[["unique_id", "ds", model_col]], on=["unique_id", "ds"], how="left"
    ).rename(columns={model_col: "model"})
    res = evaluate(
        merged,
        metrics=[partial(mase, seasonality=SEASON), smape],
        train_df=train,
        models=["model"],
    )
    # evaluate() returns a long [unique_id, metric, model] frame; reshape to one
    # row per series with mase/smape columns for paired scoring.
    return res.pivot(index="unique_id", columns="metric", values="model").reset_index()


def run_seed(
    real: pd.DataFrame,
    hist_lengths: list[int],
    seed: int,
    n_series: int,
    pretrain_steps: int,
    finetune_steps: int,
    finetune_lr_scale: float,
    n_per_generator: int,
) -> list[dict]:
    """Pretrain once, then evaluate all three conditions across history lengths."""
    rng = np.random.default_rng(seed)
    all_ids = np.array(sorted(real["unique_id"].unique()))
    ids = list(rng.choice(all_ids, size=min(n_series, len(all_ids)), replace=False))

    # Pretrain on synthetic ONCE per seed, then reuse for every history length.
    synth = make_synthetic(seed, n_per_generator)
    nf_pt = _nf(_quiet_nhits(pretrain_steps, seed))
    nf_pt.fit(synth)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as ckpt:
        nf_pt.save(ckpt, overwrite=True, save_dataset=False)

        for hist in hist_lengths:
            train, test = split_real(real, ids, hist)

            # A. from scratch — same real-training step budget as fine-tune.
            nf_fs = _nf(_quiet_nhits(finetune_steps, seed))
            nf_fs.fit(train)
            fs = score(nf_fs.predict(), test, train, "NHITS").assign(
                condition="from_scratch"
            )

            # B. zero-shot transfer — pretrained model, no real training.
            zs = score(nf_pt.predict(df=train), test, train, "NHITS").assign(
                condition="zero_shot"
            )

            # C. pretrain + fine-tune — reload pretrained weights, continue on
            #    real at a REDUCED learning rate to avoid overwriting them.
            nf_ft = NeuralForecast.load(ckpt)
            nf_ft.models[0].max_steps = finetune_steps
            nf_ft.models[0].learning_rate = BASE_LR * finetune_lr_scale
            nf_ft.fit(train, use_init_models=False)
            ft = score(nf_ft.predict(), test, train, "NHITS").assign(
                condition="pretrain_ft"
            )

            for cond_df in (fs, zs, ft):
                for _, r in cond_df.iterrows():
                    rows.append(
                        {
                            "seed": seed,
                            "hist_len": int(hist),
                            "unique_id": r["unique_id"],
                            "condition": r["condition"],
                            "mase": float(r["mase"]),
                            "smape": float(r["smape"]),
                        }
                    )
    return rows


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(
        axis=1
    )
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def build_summary(records: list[dict], meta: dict) -> dict:
    """Aggregate per-series records into a per-history-length summary."""
    df = pd.DataFrame.from_records(records)
    by_hist = []
    for hist, g in df.groupby("hist_len"):
        wide = g.pivot_table(
            index=["seed", "unique_id"], columns="condition", values="mase"
        ).dropna()
        entry: dict = {"hist_len": int(hist), "n_pairs": int(len(wide))}
        for cond in ("from_scratch", "zero_shot", "pretrain_ft"):
            vals = df[(df.hist_len == hist) & (df.condition == cond)]["mase"].to_numpy()
            entry[f"{cond}_mase"] = float(np.mean(vals))
            entry[f"{cond}_ci"] = _bootstrap_ci(vals)
            entry[f"{cond}_smape"] = float(
                df[(df.hist_len == hist) & (df.condition == cond)]["smape"].mean()
            )
        # Paired improvement of each pretrained condition over from_scratch.
        for cond in ("zero_shot", "pretrain_ft"):
            base, other = wide["from_scratch"].to_numpy(), wide[cond].to_numpy()
            delta = base - other  # positive => pretrained is better (lower MASE)
            entry[f"{cond}_win_rate"] = float(np.mean(delta > 0))
            entry[f"{cond}_mean_improve_pct"] = float(
                100.0 * np.mean(delta / np.where(base == 0, np.nan, base))
            )
            try:
                entry[f"{cond}_wilcoxon_p"] = float(wilcoxon(base, other).pvalue)
            except ValueError:
                entry[f"{cond}_wilcoxon_p"] = float("nan")
        by_hist.append(entry)
    return {**meta, "by_hist_len": sorted(by_hist, key=lambda e: e["hist_len"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-series", type=int, default=300)
    p.add_argument(
        "--history-lengths", type=int, nargs="+", default=[48, 72, 108, FULL_HIST]
    )
    p.add_argument("--pretrain-steps", type=int, default=1000)
    p.add_argument("--finetune-steps", type=int, default=300)
    p.add_argument("--finetune-lr-scale", type=float, default=0.1)
    p.add_argument("--n-per-generator", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="fast smoke run")
    p.add_argument("--save-summary", type=str, default=None)
    args = p.parse_args()

    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.n_series = 40
        args.history_lengths = [48, FULL_HIST]
        args.pretrain_steps = 60
        args.finetune_steps = 30
        args.n_per_generator = 3

    def _lbl(h: int) -> str:
        return "full" if h >= FULL_HIST else str(h)

    print(
        f"Loading M3 Monthly | seeds={args.seeds} n_series={args.n_series} "
        f"hist={[_lbl(h) for h in args.history_lengths]} "
        f"ft_lr={BASE_LR * args.finetune_lr_scale:g}"
    )
    real = load_real()

    records: list[dict] = []
    for seed in range(args.seeds):
        print(
            f"  seed {seed}: pretrain({args.pretrain_steps}) -> hist sweep", flush=True
        )
        records.extend(
            run_seed(
                real,
                args.history_lengths,
                seed,
                args.n_series,
                args.pretrain_steps,
                args.finetune_steps,
                args.finetune_lr_scale,
                args.n_per_generator,
            )
        )

    summary = build_summary(
        records,
        {
            "dataset": "M3 Monthly (last 18 months held out)",
            "model": "NHITS via NeuralForecast, per-series standardization",
            "metric": "seasonal MASE (season length 12)",
            "seeds": args.seeds,
            "n_series": args.n_series,
            "pretrain_steps": args.pretrain_steps,
            "finetune_steps": args.finetune_steps,
            "finetune_lr": BASE_LR * args.finetune_lr_scale,
        },
    )

    print("\n history | from_scratch |  zero_shot  | pretrain_ft | ft win% | ft p")
    for e in summary["by_hist_len"]:
        print(
            f" {_lbl(e['hist_len']):>7} | {e['from_scratch_mase']:>12.3f} "
            f"| {e['zero_shot_mase']:>11.3f} | {e['pretrain_ft_mase']:>11.3f} "
            f"| {e['pretrain_ft_win_rate']:>6.0%} | {e['pretrain_ft_wilcoxon_p']:.3f}"
        )

    if args.save_summary:
        Path(args.save_summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_summary).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.save_summary}")


if __name__ == "__main__":
    main()
