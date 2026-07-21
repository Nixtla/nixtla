"""Does SynAugment augmentation help a global forecaster in the cold-start regime?

Paired, multi-seed benchmark on M4 Monthly. For each seed we sample a panel of
``n`` series and compare a global gradient-boosted forecaster (with per-series
standardization, the correct configuration for M4's varied scales) trained on

    A. observed data only
    B. observed data + SynAugment counterparts (leakage-safe: fit on train only)

against the official 18-month M4 holdout, sweeping the panel size ``n``.

The hypothesis, fixed in advance: augmentation helps most when the observed
panel is *narrow* — only a handful of series, so a global model has little
shared structure to learn — and the benefit fades as the panel grows. (History
length is not the right axis here: per-series standardization already removes
the scale advantage that a short-history sweep would otherwise pick up.)

A third, secondary condition trains on an independent synthetic panel and
forecasts the observed series zero-shot.

We report mean seasonal MASE per panel size with bootstrap CIs, and the paired
improvement (win rate + Wilcoxon signed-rank p-value) of B over A.

Usage:
    uv run python benchmarks/benchmark_augmentation.py            # full run
    uv run python benchmarks/benchmark_augmentation.py --quick    # fast smoke
    uv run python benchmarks/benchmark_augmentation.py --save benchmarks/data/augmentation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import tempfile
import warnings
from functools import partial
from pathlib import Path
from urllib.request import urlretrieve

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlforecast import MLForecast  # noqa: E402
from mlforecast.target_transforms import LocalStandardScaler  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from utilsforecast.evaluation import evaluate  # noqa: E402
from utilsforecast.losses import mase, smape  # noqa: E402

from synforecast import SynAugment, generate_series  # noqa: E402
from synforecast.generators import (  # noqa: E402
    ETSGenerator,
    SARIMAGenerator,
    SeasonalGenerator,
)

HORIZON = 18
SEASON_LENGTH = 12
ANCHOR = pd.Timestamp("2015-12-31")  # shared train-end so holdouts align
LAGS = [1, 2, 3, 6, 12, 13]

M4_BASE_URL = (
    "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
)
CACHE_DIR = Path(tempfile.gettempdir()) / "synforecast-m4"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _download(split: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"Monthly-{split.lower()}.csv"
    if not path.exists():
        urlretrieve(f"{M4_BASE_URL}/{split}/Monthly-{split.lower()}.csv", path)
    return path


def _read_series(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            values = [float(v) for v in row[1:] if v != ""]
            out[row[0]] = np.asarray(values, dtype=np.float64)
    return out


def load_m4_monthly(min_len: int) -> tuple[list[str], dict, dict]:
    """Return (candidate ids with >= min_len train history, train, test)."""
    train = _read_series(_download("Train"))
    test = _read_series(_download("Test"))
    candidates = sorted(
        uid
        for uid, values in train.items()
        if len(values) >= min_len and uid in test and len(test[uid]) >= HORIZON
    )
    return candidates, train, test


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------
def build_panels(ids: list[str], train: dict, test: dict) -> tuple:
    """Full-history training panel and the aligned 18-month holdout.

    Every series' training window ends at ANCHOR, so all holdouts share the
    same dates and a global model forecasts a common horizon.
    """
    train_rows, test_rows = [], []
    test_ds = pd.date_range(
        start=ANCHOR + pd.offsets.MonthEnd(1), periods=HORIZON, freq="ME"
    )
    for uid in ids:
        hist = train[uid]
        train_ds = pd.date_range(end=ANCHOR, periods=len(hist), freq="ME")
        train_rows.extend(zip([uid] * len(hist), train_ds, hist, strict=True))
        test_rows.extend(
            zip([uid] * HORIZON, test_ds, test[uid][:HORIZON], strict=True)
        )
    cols = ["unique_id", "ds", "y"]
    return (
        pd.DataFrame(train_rows, columns=cols),
        pd.DataFrame(test_rows, columns=cols),
    )


def monthly_synthetic_pool(seed: int) -> list:
    """SARIMA/ETS/seasonal monthly processes for independent pretraining."""
    common = {"min_length": 96, "max_length": 180, "freq": "ME", "engine": "polars"}
    return [
        SARIMAGenerator(**common, seasonal_period=12, d=1, D=1, noise_std=2.0, seed=seed),
        SARIMAGenerator(
            **common, seasonal_period=12, p=2, P=0, Q=1, noise_std=1.5, seed=seed + 1
        ),
        ETSGenerator(
            **common, seasonal_period=12, level=100.0, trend=0.2, noise_std=2.0,
            seed=seed + 2,
        ),
        SeasonalGenerator(
            **common, seasonality_period=12, seasonality_amplitude=10.0, trend=0.15,
            noise_level=2.0, seed=seed + 3,
        ),
    ]


# ---------------------------------------------------------------------------
# Modeling + evaluation
# ---------------------------------------------------------------------------
def _forecaster(seed: int) -> MLForecast:
    return MLForecast(
        models={
            "model": HistGradientBoostingRegressor(max_iter=150, random_state=seed)
        },
        freq="ME",
        lags=LAGS,
        date_features=["month"],
        # Per-series standardization: M4 series span very different scales, so
        # a global model must be scale-invariant (mirrors NHITS scaler_type).
        target_transforms=[LocalStandardScaler()],
    )


def _score(holdout: pd.DataFrame, forecast: pd.DataFrame, train: pd.DataFrame) -> tuple:
    """Mean seasonal MASE and sMAPE over the target series."""
    merged = holdout.merge(
        forecast[["unique_id", "ds", "model"]], on=["unique_id", "ds"], how="inner"
    )
    result = evaluate(
        merged,
        metrics=[partial(mase, seasonality=SEASON_LENGTH), smape],
        models=["model"],
        train_df=train,
    )
    by_metric = result.groupby("metric", observed=True)["model"].mean()
    return float(by_metric["mase"]), float(by_metric["smape"])


def run_panel(
    seed: int,
    ids: list[str],
    train: dict,
    test: dict,
    n_augment: int,
    with_pretrain: bool,
) -> list[dict]:
    train_df, holdout = build_panels(ids, train, test)
    target_ids = set(ids)
    rows = []

    def record(condition: str, forecast: pd.DataFrame) -> None:
        fc = forecast[forecast["unique_id"].astype(str).isin(target_ids)]
        mase_v, smape_v = _score(holdout, fc, train_df)
        rows.append(
            {
                "seed": seed,
                "n_series": len(ids),
                "condition": condition,
                "mase": mase_v,
                "smape": smape_v,
            }
        )

    # A. observed only
    a = _forecaster(seed)
    a.fit(train_df)
    record("observed", a.predict(h=HORIZON))

    # B. observed + augmented (SynAugment fit on the observed panel only)
    augmented = SynAugment(seed=seed).augment(train_df, n_augment=n_augment)
    b = _forecaster(seed)
    b.fit(augmented)
    record("augmented", b.predict(h=HORIZON))

    # C. independent synthetic pretraining, zero-shot (secondary)
    if with_pretrain:
        synth = generate_series(
            n_series=64, generators=monthly_synthetic_pool(seed)
        ).to_pandas()
        synth = synth[np.isfinite(synth["y"])]
        c = _forecaster(seed)
        c.fit(synth)
        record("synthetic_zeroshot", c.predict(h=HORIZON, new_df=train_df))

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _ci(values: np.ndarray, seed: int = 0, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame, with_pretrain: bool) -> None:
    sizes = sorted(df["n_series"].unique())

    print("\n=== Mean seasonal MASE by panel size (lower is better) ===")
    header = f"{'n_series':>9} {'observed':>22} {'augmented':>22}"
    if with_pretrain:
        header += f" {'synth zero-shot':>18}"
    print(header)
    for n in sizes:
        sub = df[df["n_series"] == n]
        cells = [f"{n:>9}"]
        for cond in ["observed", "augmented"]:
            vals = sub[sub["condition"] == cond]["mase"].to_numpy()
            lo, hi = _ci(vals)
            cells.append(f"{vals.mean():>8.3f} [{lo:.3f},{hi:.3f}]")
        if with_pretrain:
            vals = sub[sub["condition"] == "synthetic_zeroshot"]["mase"].to_numpy()
            cells.append(f"{vals.mean():>17.3f}")
        print(" ".join(cells))

    print("\n=== Paired improvement of augmented over observed (per seed) ===")
    print(
        f"{'n_series':>9} {'mean dMASE':>12} {'mean dMASE %':>13} "
        f"{'win rate':>10} {'wilcoxon p':>12}"
    )
    verdict = []
    for n in sizes:
        sub = df[df["n_series"] == n]
        pivot = sub.pivot_table(index="seed", columns="condition", values="mase")
        delta = pivot["observed"] - pivot["augmented"]  # positive => B better
        pct = 100.0 * delta / pivot["observed"]
        win = float((delta > 0).mean())
        try:
            p = float(wilcoxon(pivot["augmented"], pivot["observed"]).pvalue)
        except ValueError:
            p = float("nan")
        print(
            f"{n:>9} {delta.mean():>12.4f} {pct.mean():>12.1f}% "
            f"{win:>9.0%} {p:>12.4f}"
        )
        verdict.append((n, delta.mean(), win, p))

    smallest, largest = verdict[0], verdict[-1]
    helps_small = smallest[1] > 0 and smallest[2] >= 0.6 and smallest[3] < 0.05
    fades = smallest[1] > largest[1]
    print("\n=== Verdict ===")
    print(
        f"Augmentation helps at the smallest panel (n={smallest[0]}): "
        f"{'YES' if helps_small else 'no clear win'} "
        f"(mean dMASE={smallest[1]:.4f}, win rate={smallest[2]:.0%}, p={smallest[3]:.4f})"
    )
    print(
        f"Benefit fades as the panel grows: {'YES' if fades else 'not monotone'} "
        f"(n={smallest[0]} dMASE={smallest[1]:.4f} vs n={largest[0]} dMASE={largest[1]:.4f})"
    )


def build_summary(df: pd.DataFrame, with_pretrain: bool) -> dict:
    """Aggregate the raw per-seed results into a compact, publishable summary.

    Written as JSON (``*.csv`` is git-ignored) so a docs page can present the
    benchmark without re-running any models.
    """
    sizes = sorted(int(n) for n in df["n_series"].unique())
    by_size = []
    for n in sizes:
        sub = df[df["n_series"] == n]
        pivot = sub.pivot_table(index="seed", columns="condition", values="mase")
        delta = pivot["observed"] - pivot["augmented"]
        try:
            p = float(wilcoxon(pivot["augmented"], pivot["observed"]).pvalue)
        except ValueError:
            p = float("nan")
        entry = {"n_series": n}
        for cond in pivot.columns:
            vals = pivot[cond].to_numpy()
            lo, hi = _ci(vals)
            entry[f"{cond}_mase"] = float(vals.mean())
            entry[f"{cond}_ci"] = [lo, hi]
        entry.update(
            mean_dmase=float(delta.mean()),
            mean_dmase_pct=float((100.0 * delta / pivot["observed"]).mean()),
            win_rate=float((delta > 0).mean()),
            wilcoxon_p=p,
        )
        by_size.append(entry)
    return {
        "dataset": "M4 Monthly (official 18-month holdout)",
        "model": "HistGradientBoostingRegressor via MLForecast, per-series scaling",
        "metric": "seasonal MASE (season length 12)",
        "seeds": int(df["seed"].nunique()),
        "with_pretrain": with_pretrain,
        "by_size": by_size,
    }


def save_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(df["n_series"].unique())
    xs = list(range(len(sizes)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, color in [("observed", "steelblue"), ("augmented", "crimson")]:
        means, los, his = [], [], []
        for n in sizes:
            vals = df[(df["n_series"] == n) & (df["condition"] == cond)][
                "mase"
            ].to_numpy()
            lo, hi = _ci(vals)
            means.append(vals.mean())
            los.append(lo)
            his.append(hi)
        ax.plot(xs, means, marker="o", color=color, label=cond)
        ax.fill_between(xs, los, his, color=color, alpha=0.15)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set(
        xlabel="number of observed series in the panel",
        ylabel="mean seasonal MASE",
        title="SynAugment helps most in the cold-start regime (few series)",
    )
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"\nSaved figure to {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--sizes", default="3,5,10,30", help="panel sizes to sweep")
    parser.add_argument("--n-augment", type=int, default=4)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--quick", action="store_true", help="4 seeds, {3, 30}")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--save-summary", type=Path, default=None)
    parser.add_argument("--save-figure", type=Path, default=None)
    args = parser.parse_args()

    # Silence SynAugment's per-series generator-fallback logging (informational).
    logging.getLogger("synforecast").setLevel(logging.ERROR)

    if args.quick:
        args.seeds, args.sizes = 4, "3,30"

    sizes = [int(x) for x in args.sizes.split(",")]
    with_pretrain = not args.no_pretrain
    print(
        f"Config: seeds={args.seeds} sizes={sizes} "
        f"n_augment={args.n_augment} pretrain={with_pretrain}"
    )
    candidates, train, test = load_m4_monthly(min_len=24 + HORIZON)
    print(f"Eligible M4 Monthly series: {len(candidates)}")

    records: list[dict] = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for n in sizes:
            ids = list(rng.choice(candidates, size=n, replace=False))
            records.extend(
                run_panel(seed, ids, train, test, args.n_augment, with_pretrain)
            )
        print(f"  seed {seed} done ({len(records)} rows so far)")

    df = pd.DataFrame(records)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save, index=False)
        print(f"Saved raw results to {args.save}")
    if args.save_summary:
        import json

        args.save_summary.parent.mkdir(parents=True, exist_ok=True)
        args.save_summary.write_text(
            json.dumps(build_summary(df, with_pretrain), indent=2)
        )
        print(f"Saved summary to {args.save_summary}")
    if args.save_figure:
        save_figure(df, args.save_figure)

    summarize(df, with_pretrain)


if __name__ == "__main__":
    main()
