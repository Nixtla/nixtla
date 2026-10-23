"""Generate the charts for the coupled retail simulation tutorial.

The example data is synthetic and is built here exactly as the tutorial shows
it, so every number in the guide is reproducible from the guide itself. The
product relationships (substitute, complement) are planted in the generator,
which is what lets the tutorial label them.

See ``_docs_client.py`` for how to reach a model. The default is in-process
tsfm:

    PYTHONPATH=/path/to/nixtla uv run --no-sync python \
      /path/to/nixtla/timegpt-docs/scripts/generate_coupled_retail_tutorial.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from _docs_client import make_docs_client

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DOCS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DOCS_DIR / "images" / "forecasting"
PROMOTED = "Promoted product"
ALTERNATIVE = "Alternative product"
RELATED = "Related product"
COLORS = {
    PROMOTED: "#6C63FF",
    ALTERNATIVE: "#F59E5B",
    RELATED: "#2BBFA3",
}
PURPLE = "#6C63FF"
DARK = "#202124"
GRID = "#DADCE0"
N = 1095
H = 28
DISCOUNT = 0.85
PLANTED_PCT = {PROMOTED: 11.9, ALTERNATIVE: -11.5, RELATED: 15.0}


def _style_axis(ax):
    ax.grid(axis="y", color=GRID, alpha=0.55, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#4A4A4A")


def build_example() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The synthetic product group, identical to the tutorial's snippet."""
    rng = np.random.default_rng(11)
    t = np.arange(N + H)

    promoted_price = (
        6.00
        + 0.60 * np.sin(2 * np.pi * t / 97)
        + 0.30 * np.sin(2 * np.pi * t / 29)
        + rng.normal(0, 0.30, N + H)
    )
    alternative_price = (
        9.00
        + 0.85 * np.sin(2 * np.pi * t / 73 + 1.1)
        + 0.40 * np.sin(2 * np.pi * t / 23)
        + rng.normal(0, 0.36, N + H)
    )
    related_price = (
        4.50
        + 0.45 * np.sin(2 * np.pi * t / 113 + 2.3)
        + 0.22 * np.sin(2 * np.pi * t / 31)
        + rng.normal(0, 0.24, N + H)
    )

    shock = rng.normal(0, 1, N + H)
    footfall = np.zeros(N + H)
    for i in range(1, N + H):
        footfall[i] = 0.92 * footfall[i - 1] + shock[i]
    footfall = 1 + 0.059 * footfall
    weekly = np.sin(2 * np.pi * t / 7)

    promoted_demand = (
        40 * footfall
        - 5.3 * (promoted_price - 6.0)
        + 2.0 * (alternative_price - 9.0)
        + 3.0 * weekly
        + rng.normal(0, 1.6, N + H)
    )
    alternative_demand = (
        55 * footfall
        + 7.0 * (promoted_price - 6.0)
        - 4.0 * (alternative_price - 9.0)
        + 3.0 * weekly
        + rng.normal(0, 2.0, N + H)
    )
    related_demand = (
        30 * footfall
        - 5.0 * (promoted_price - 6.0)
        - 3.5 * (related_price - 4.5)
        + 2.0 * weekly
        + rng.normal(0, 1.4, N + H)
    )

    dates = pd.date_range("2023-01-01", periods=N + H, freq="D")
    prices = {
        "promoted_price": promoted_price,
        "alternative_price": alternative_price,
        "related_price": related_price,
    }
    demands = {
        PROMOTED: promoted_demand,
        ALTERNATIVE: alternative_demand,
        RELATED: related_demand,
    }

    history, future = [], []
    for label, demand in demands.items():
        history.append(
            pd.DataFrame(
                {
                    "unique_id": label,
                    "ds": dates[:N],
                    "y": np.maximum(demand[:N], 0).round(),
                    **{name: values[:N] for name, values in prices.items()},
                }
            )
        )
        future.append(
            pd.DataFrame(
                {
                    "unique_id": label,
                    "ds": dates[N:],
                    **{name: values[N:] for name, values in prices.items()},
                }
            )
        )

    df = pd.concat(history, ignore_index=True)
    current_X_df = pd.concat(future, ignore_index=True)
    promotion_X_df = current_X_df.copy()
    promotion_X_df["promoted_price"] *= DISCOUNT
    return df, current_X_df, promotion_X_df


def _path_totals(paths: pd.DataFrame) -> pd.DataFrame:
    nonnegative = paths.assign(TimeGPT=paths["TimeGPT"].clip(lower=0))
    return nonnegative.pivot_table(
        index="sample_id",
        columns="unique_id",
        values="TimeGPT",
        aggfunc="sum",
    )


def _daily_paths(paths: pd.DataFrame) -> pd.DataFrame:
    nonnegative = paths.assign(TimeGPT=paths["TimeGPT"].clip(lower=0))
    return nonnegative.pivot(
        index=["sample_id", "ds"],
        columns="unique_id",
        values="TimeGPT",
    )


def _plot_history(df: pd.DataFrame) -> None:
    history = df.pivot(index="ds", columns="unique_id", values="y").tail(365)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.5), sharex=True)
    for ax, label in zip(axes, COLORS):
        color = COLORS[label]
        ax.plot(
            history.index,
            history[label],
            color=color,
            alpha=0.22,
            linewidth=0.8,
        )
        ax.plot(
            history.index,
            history[label].rolling(7).mean(),
            color=color,
            linewidth=2,
        )
        ax.set_ylabel(label.replace(" product", ""))
        _style_axis(ax)

    axes[0].set_title("Daily unit sales for the three products")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation-retail-history.png", dpi=180)
    plt.close(fig)


def _plot_same_day_risk(
    independent_daily: pd.DataFrame,
    coupled_daily: pd.DataFrame,
) -> dict[str, float]:
    high_thresholds = independent_daily.quantile(0.9)

    def probability(daily: pd.DataFrame) -> float:
        return float(
            daily.gt(high_thresholds)
            .sum(axis=1)
            .ge(2)
            .groupby("sample_id")
            .any()
            .mean()
        )

    independent_probability = probability(independent_daily)
    coupled_probability = probability(coupled_daily)

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    bars = ax.bar(
        ["Products simulated\nseparately", "Products simulated\ntogether"],
        [independent_probability, coupled_probability],
        color=["#A8A8A8", PURPLE],
        width=0.58,
    )
    for bar, value in zip(
        bars,
        [independent_probability, coupled_probability],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            color=DARK,
            fontweight="bold",
            fontsize=13,
        )
    ax.set(
        title="Chance that at least two products run high on the same day",
        ylabel="Probability during the 28-day promotion",
        ylim=(0, max(independent_probability, coupled_probability) * 1.22),
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation-retail-coupled-risk.png", dpi=180)
    plt.close(fig)

    return {
        "independent_probability": independent_probability,
        "coupled_probability": coupled_probability,
    }


def _plot_simulated_paths(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    independent: pd.DataFrame,
    coupled: pd.DataFrame,
) -> None:
    history = df[["unique_id", "ds", "y"]]
    path_ids = list(range(5))
    path_colors = plt.get_cmap("tab10").colors[: len(path_ids)]
    BAND = "#9AA0A6"

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13, 10),
        sharex=True,
        sharey="row",
    )
    columns = [
        ("Products simulated separately", independent),
        ("Products simulated together", coupled),
    ]

    for row, label in enumerate(COLORS):
        item_history = history.query("unique_id == @label").tail(35)
        for col, (title, paths) in enumerate(columns):
            ax = axes[row, col]
            item_paths = paths.query(
                "unique_id == @label and sample_id in @path_ids"
            ).copy()
            item_paths["TimeGPT"] = item_paths["TimeGPT"].clip(lower=0)

            ax.plot(
                item_history["ds"],
                item_history["y"],
                color=DARK,
                linewidth=1.5,
            )

            band = (
                paths.query("unique_id == @label")
                .assign(TimeGPT=lambda frame: frame["TimeGPT"].clip(lower=0))
                .groupby("ds", observed=True)["TimeGPT"]
                .quantile([0.05, 0.25, 0.75, 0.95])
                .unstack()
            )
            ax.fill_between(
                band.index,
                band[0.05],
                band[0.95],
                color=BAND,
                alpha=0.30,
                linewidth=0,
            )
            ax.fill_between(
                band.index,
                band[0.25],
                band[0.75],
                color=BAND,
                alpha=0.45,
                linewidth=0,
            )

            for color, sample_id in zip(path_colors, path_ids):
                path = item_paths.query("sample_id == @sample_id")
                ax.plot(
                    path["ds"],
                    path["TimeGPT"],
                    color=color,
                    alpha=0.9,
                    linewidth=1.7,
                )

            median = (
                paths.query("unique_id == @label")
                .assign(TimeGPT=lambda frame: frame["TimeGPT"].clip(lower=0))
                .groupby("ds", observed=True)["TimeGPT"]
                .median()
            )
            ax.plot(
                median.index,
                median,
                color=DARK,
                linewidth=2.3,
                linestyle="--",
            )
            ax.axvline(cutoff, color="#8A8A8A", linewidth=1, linestyle=":")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{label}\nDaily units")
            _style_axis(ax)

    axes[-1, 0].set_xlabel("Date")
    axes[-1, 1].set_xlabel("Date")
    fig.suptitle(
        "All 500 promotion paths, with five highlighted",
        fontsize=16,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation-retail-paths.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, current_X_df, promotion_X_df = build_example()
    cutoff = current_X_df["ds"].min()

    client = make_docs_client(timeout=300, max_retries=1)
    common = {
        "df": df,
        "h": H,
        "freq": "D",
        "n_paths": 500,
        "seed": 42,
        "model": "timegpt-2.1",
    }
    current = client.simulate(X_df=current_X_df, multivariate=True, **common)
    coupled = client.simulate(X_df=promotion_X_df, multivariate=True, **common)
    independent = client.simulate(
        X_df=promotion_X_df,
        multivariate=False,
        **{**common, "seed": 7},
    )

    if not current["coupled"].all() or not coupled["coupled"].all():
        raise RuntimeError("The expected cross-series coupling was not applied.")
    if independent["coupled"].any():
        raise RuntimeError("The separate-product comparison is unexpectedly coupled.")

    current_totals = _path_totals(current)
    coupled_totals = _path_totals(coupled)
    independent_daily = _daily_paths(independent)
    coupled_daily = _daily_paths(coupled)

    _plot_history(df)
    _plot_simulated_paths(df, cutoff, independent, coupled)
    risk = _plot_same_day_risk(independent_daily, coupled_daily)

    impact = pd.DataFrame(
        {
            "current_prices": current_totals.median(),
            "promotion": coupled_totals.median(),
        }
    )
    impact["change"] = impact["promotion"] - impact["current_prices"]
    impact["change_pct"] = 100 * (impact["promotion"] / impact["current_prices"] - 1)
    impact["planted_pct"] = pd.Series(PLANTED_PCT)

    history_wide = df.pivot(index="ds", columns="unique_id", values="y")
    print("history rows:", len(df))
    print("history dates:", df["ds"].min(), "to", df["ds"].max())
    print("\nMean daily units")
    print(history_wide.mean().round(1).to_string())
    print("\nDemand co-movement (spearman)")
    print(history_wide.corr(method="spearman").round(3).to_string())
    print("\nMedian 28-day units: planted vs recovered")
    print(impact.round(2).to_string())
    signs_match = all(
        np.sign(impact.loc[label, "change_pct"]) == np.sign(planted)
        for label, planted in PLANTED_PCT.items()
    )
    print("\nAll recovered signs match the planted structure:", signs_match)
    if not signs_match:
        raise RuntimeError("The simulation did not recover the planted directions.")
    print("\nSame-day risk")
    print(risk)


if __name__ == "__main__":
    main()
