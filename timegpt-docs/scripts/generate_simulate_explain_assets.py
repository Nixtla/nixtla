"""Regenerate the simulation and explanation case-study charts.

See ``_docs_client.py`` for how to reach a model. The default is in-process
tsfm:

    PYTHONPATH=/path/to/nixtla uv run --no-sync python \
      /path/to/nixtla/timegpt-docs/scripts/generate_simulate_explain_assets.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from _docs_client import make_docs_client
from nixtla import NixtlaClient

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "images" / "forecasting"
PURPLE = "#6C63FF"
MINT = "#2BBFA3"
ORANGE = "#F59E5B"
BLUE = "#4C8BF5"
DARK = "#202124"
GRID = "#DADCE0"


def _style_axis(ax):
    ax.grid(axis="y", color=GRID, alpha=0.55, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#4A4A4A")


def build_energy_case(client: NixtlaClient) -> dict[str, object]:
    data_url = (
        "https://raw.githubusercontent.com/Nixtla/"
        "transfer-learning-time-series/main/datasets/"
        "electricity-short-with-ex-vars.csv"
    )
    energy = pd.read_csv(data_url, parse_dates=["ds"])
    germany = energy.query('unique_id == "DE"').copy()
    train = germany.query('ds < "2017-12-29"').copy()
    future = germany.query('ds >= "2017-12-29"').copy()

    baseline_X = future.drop(columns="y")
    stress_X = baseline_X.copy()
    stress_hours = stress_X["ds"].dt.date == stress_X["ds"].dt.date.max()
    stress_X.loc[stress_hours, "Exogenous1"] *= 1.15
    stress_X.loc[stress_hours, "Exogenous2"] *= 0.85

    common = {
        "df": train,
        "h": 48,
        "freq": "h",
        "n_paths": 500,
        "seed": 42,
        "model": "timegpt-2.1",
    }
    baseline_paths = client.simulate(X_df=baseline_X, **common)
    stress_paths = client.simulate(X_df=stress_X, **common)

    def path_band(paths: pd.DataFrame) -> pd.DataFrame:
        return (
            paths.groupby("ds", observed=True)["TimeGPT"]
            .quantile([0.1, 0.5, 0.9])
            .unstack()
            .rename(columns={0.1: "q10", 0.5: "q50", 0.9: "q90"})
        )

    baseline_band = path_band(baseline_paths)
    stress_band = path_band(stress_paths)
    history = train.tail(72)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(
        history["ds"],
        history["y"],
        color=DARK,
        linewidth=1.8,
        label="Observed history",
    )
    ax.plot(
        future["ds"],
        future["y"],
        color="#7A7A7A",
        linewidth=1.4,
        linestyle=":",
        label="Observed later",
    )
    ax.fill_between(
        baseline_band.index,
        baseline_band["q10"],
        baseline_band["q90"],
        color=PURPLE,
        alpha=0.13,
    )
    ax.plot(
        baseline_band.index,
        baseline_band["q50"],
        color=PURPLE,
        linewidth=2.2,
        label="Published inputs",
    )
    ax.fill_between(
        stress_band.index,
        stress_band["q10"],
        stress_band["q90"],
        color=ORANGE,
        alpha=0.13,
    )
    ax.plot(
        stress_band.index,
        stress_band["q50"],
        color=ORANGE,
        linewidth=2.2,
        label="High-load / low-generation scenario",
    )
    stress_dates = stress_X.loc[stress_hours, "ds"]
    ax.axvspan(
        stress_dates.min(),
        stress_dates.max(),
        color=ORANGE,
        alpha=0.08,
        label="Changed inputs",
    )
    ax.axvline(future["ds"].min(), color="#8A8A8A", linewidth=1, linestyle="--")
    ax.set(
        title="Changing system inputs shifts the simulated electricity-price paths",
        xlabel="Date",
        ylabel="Electricity price (EUR/MWh)",
    )
    _style_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation-energy-scenarios.png", dpi=180)
    plt.close(fig)

    baseline_cost = baseline_paths.groupby("sample_id", observed=True)["TimeGPT"].sum()
    stress_cost = stress_paths.groupby("sample_id", observed=True)["TimeGPT"].sum()
    cost_threshold = 300
    baseline_probability = float(baseline_cost.gt(cost_threshold).mean())
    stress_probability = float(stress_cost.gt(cost_threshold).mean())
    price_threshold = 40
    baseline_paths_above_price = float(
        baseline_paths.assign(exceeds=baseline_paths["TimeGPT"] > price_threshold)
        .groupby("sample_id", observed=True)["exceeds"]
        .any()
        .mean()
    )

    bins = np.linspace(
        min(float(baseline_cost.min()), float(stress_cost.min())),
        max(float(baseline_cost.max()), float(stress_cost.max())),
        34,
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.hist(
        baseline_cost,
        bins=bins,
        color=PURPLE,
        alpha=0.6,
        label="Published inputs",
    )
    ax.hist(
        stress_cost,
        bins=bins,
        color=ORANGE,
        alpha=0.55,
        label="High-load / low-generation",
    )
    ax.axvline(cost_threshold, color=DARK, linewidth=1.8, linestyle="--")
    ax.text(
        cost_threshold + 12,
        ax.get_ylim()[1] * 0.88,
        (
            f"Cost above EUR {cost_threshold}\n"
            f"Published inputs: {baseline_probability:.1%}\n"
            f"Changed inputs: {stress_probability:.1%}"
        ),
        color=DARK,
        va="top",
    )
    ax.set(
        title="Cost distribution for buying 1 MWh in every forecast hour",
        xlabel="Total 48-hour energy cost (EUR)",
        ylabel="Number of simulated paths",
    )
    _style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation-energy-cost-risk.png", dpi=180)
    plt.close(fig)

    return {
        "baseline_head": baseline_paths.head().to_dict(orient="records"),
        "baseline_cost_p50": float(baseline_cost.quantile(0.5)),
        "baseline_cost_p90": float(baseline_cost.quantile(0.9)),
        "stress_cost_p50": float(stress_cost.quantile(0.5)),
        "stress_cost_p90": float(stress_cost.quantile(0.9)),
        "baseline_probability_above_300": baseline_probability,
        "stress_probability_above_300": stress_probability,
        "baseline_paths_above_price_40": baseline_paths_above_price,
        "stress_day_median_shift_min": float(
            (
                stress_band.loc[stress_dates, "q50"]
                - baseline_band.loc[stress_dates, "q50"]
            ).min()
        ),
        "stress_day_median_shift_max": float(
            (
                stress_band.loc[stress_dates, "q50"]
                - baseline_band.loc[stress_dates, "q50"]
            ).max()
        ),
    }


def _retail_example() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 365
    h = 14
    t = np.arange(n + h)

    price = 20 + 1.8 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 0.35, n + h)
    promotion = ((t % 42) >= 35).astype(int)
    temperature = (
        18 + 9 * np.sin(2 * np.pi * (t - 30) / 365) + rng.normal(0, 0.8, n + h)
    )
    weekly = 6 * np.sin(2 * np.pi * t / 7)
    demand = (
        118
        - 2.4 * price[:n]
        + 16 * promotion[:n]
        + 0.55 * temperature[:n]
        + weekly[:n]
        + rng.normal(0, 2.0, n)
    )

    dates = pd.date_range("2024-01-01", periods=n + h, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": "store-a",
            "ds": dates[:n],
            "y": demand,
            "price": price[:n],
            "promotion": promotion[:n],
            "temperature": temperature[:n],
        }
    )
    X_df = pd.DataFrame(
        {
            "unique_id": "store-a",
            "ds": dates[n:],
            "price": price[n:],
            "promotion": promotion[n:],
            "temperature": temperature[n:],
        }
    )
    return df, X_df


def _forecast_contributions(
    client: NixtlaClient,
    df: pd.DataFrame,
    X_df: pd.DataFrame,
    contribution_type: str,
) -> pd.DataFrame:
    client.forecast(
        df=df,
        X_df=X_df,
        h=len(X_df),
        freq="D",
        model="timegpt-2.1",
        feature_contributions=True,
        feature_contributions_type=contribution_type,
    )
    return client.feature_contributions.copy()


def build_explanation_case(client: NixtlaClient) -> dict[str, object]:
    df, X_df = _retail_example()
    features = ["price", "promotion", "temperature"]

    forecast_explanations = {
        explanation: _forecast_contributions(
            client,
            df,
            X_df,
            contribution_type=explanation,
        )
        for explanation in [
            "shapley",
            "intervention",
            "granger",
            "transfer_entropy",
        ]
    }

    shapley = forecast_explanations["shapley"]
    intervention = forecast_explanations["intervention"]
    selected_ds = X_df.loc[X_df["promotion"].eq(1), "ds"].min()
    selected_shapley = shapley.loc[shapley["ds"].eq(selected_ds)].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    recent = df.tail(42)
    axes[0].plot(
        recent["ds"],
        recent["y"],
        color=DARK,
        linewidth=1.8,
        label="Observed demand",
    )
    axes[0].plot(
        shapley["ds"],
        shapley["TimeGPT"],
        color=PURPLE,
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        label="TimeGPT forecast",
    )
    promotion_dates = X_df.loc[X_df["promotion"].eq(1), "ds"]
    axes[0].axvspan(
        promotion_dates.min(),
        promotion_dates.max(),
        color=ORANGE,
        alpha=0.16,
        label="Planned promotion",
    )
    axes[0].set(title="Forecast for the next 14 days", xlabel="Date", ylabel="Units")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].tick_params(axis="x", rotation=35)
    _style_axis(axes[0])

    values = selected_shapley[features].astype(float)
    colors = [MINT if value >= 0 else ORANGE for value in values]
    axes[1].barh([name.title() for name in features], values, color=colors)
    axes[1].axvline(0, color=DARK, linewidth=0.9)
    axes[1].set(
        title=f"Why the forecast is {selected_shapley['TimeGPT']:.1f} units",
        xlabel="Contribution to forecast (units)",
    )
    axes[1].text(
        0.02,
        0.03,
        f"Starting value: {selected_shapley['base_value']:.1f} units",
        transform=axes[1].transAxes,
        color="#4A4A4A",
    )
    _style_axis(axes[1])
    axes[1].grid(axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Explain one retail-demand forecast", y=1.01, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "explain-retail-shap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for feature, color in {
        "price": BLUE,
        "promotion": ORANGE,
        "temperature": MINT,
    }.items():
        ax.plot(
            intervention["ds"],
            intervention[feature],
            marker="o",
            linewidth=2,
            label=feature.title(),
            color=color,
        )
    ax.axhline(0, color=DARK, linewidth=0.9)
    ax.axvspan(
        promotion_dates.min(),
        promotion_dates.max(),
        color=ORANGE,
        alpha=0.12,
        label="Planned promotion",
    )
    ax.set(
        title="How the forecast changes when each input is set to its typical value",
        xlabel="Forecast date",
        ylabel="Change in forecast (units)",
    )
    _style_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "explain-retail-intervention.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    analyses = {}
    for analysis in ["granger", "transfer_entropy"]:
        result = client.explain(df, method=analysis)
        analyses[analysis] = result.set_index("feature")["weight"]

    comparison = pd.DataFrame(analyses)
    labels = ["Price", "Promotion", "Temperature"]
    y = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    default_weights = comparison["granger"]
    bars = ax.barh(labels, default_weights, color=PURPLE, alpha=0.9)
    ax.invert_yaxis()
    ax.set(
        title="Predictive signals in the last year of store data",
        xlabel="Relative historical signal",
    )
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in default_weights], padding=5)
    ax.set_xlim(0, max(1.0, float(default_weights.max()) * 1.15))
    _style_axis(ax)
    ax.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "explain-retail-historical-signals.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(
        y - width / 2,
        comparison["granger"],
        height=width,
        color=PURPLE,
        label="Granger",
    )
    ax.barh(
        y + width / 2,
        comparison["transfer_entropy"],
        height=width,
        color=MINT,
        label="Transfer entropy",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(
        title="Historical feature weights answer different relationship questions",
        xlabel="Normalized weight",
    )
    _style_axis(ax)
    ax.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "explain-retail-analysis-comparison.png", dpi=180)
    plt.close(fig)

    selected_granger = (
        forecast_explanations["granger"]
        .loc[lambda x: x["ds"].eq(selected_ds), features]
        .iloc[0]
    )
    selected_entropy = (
        forecast_explanations["transfer_entropy"]
        .loc[lambda x: x["ds"].eq(selected_ds), features]
        .iloc[0]
    )
    allocation = pd.DataFrame(
        {
            "Granger-weighted": selected_granger,
            "Transfer-entropy-weighted": selected_entropy,
        }
    )
    y = np.arange(len(features))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(
        y - width / 2,
        allocation["Granger-weighted"],
        height=width,
        color=PURPLE,
        label="Granger-weighted",
    )
    ax.barh(
        y + width / 2,
        allocation["Transfer-entropy-weighted"],
        height=width,
        color=MINT,
        label="Transfer-entropy-weighted",
    )
    ax.set_yticks(y, [feature.title() for feature in features])
    ax.invert_yaxis()
    ax.axvline(0, color=DARK, linewidth=0.9)
    ax.set(
        title="Two history-weighted allocations of the same forecast",
        xlabel="Allocated forecast movement (units)",
    )
    _style_axis(ax)
    ax.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "explain-retail-advanced-allocations.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    endpoints = [185, 215, 245, 275, 305, 335, 365]
    rolling_rows = []
    for end in endpoints:
        window = df.iloc[end - 180 : end]
        for analysis in ["granger", "transfer_entropy"]:
            result = client.explain(window, method=analysis)
            result["analysis"] = analysis
            result["window_end"] = df.iloc[end - 1]["ds"]
            rolling_rows.append(result)
    rolling = pd.concat(rolling_rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {"price": BLUE, "promotion": ORANGE, "temperature": MINT}
    titles = {
        "granger": "Granger",
        "transfer_entropy": "Transfer entropy",
    }
    for ax, analysis in zip(axes, ["granger", "transfer_entropy"]):
        subset = rolling.query("analysis == @analysis")
        for feature, color in colors.items():
            feature_rows = subset.query("feature == @feature")
            ax.plot(
                feature_rows["window_end"],
                feature_rows["weight"],
                marker="o",
                linewidth=2,
                color=color,
                label=feature.title(),
            )
        ax.set(title=titles[analysis], xlabel="End of 180-day window")
        _style_axis(ax)
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Normalized weight")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Feature rankings can change across historical windows", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "explain-retail-window-stability.png", dpi=180)
    plt.close(fig)

    return {
        "selected_forecast_date": str(selected_ds.date()),
        "shapley": {
            column: float(selected_shapley[column])
            for column in ["TimeGPT", "base_value", *features]
        },
        "intervention_average": {
            feature: float(intervention[feature].mean()) for feature in features
        },
        "intervention_promotion_days": {
            feature: float(
                intervention.loc[
                    intervention["ds"].isin(promotion_dates),
                    feature,
                ].mean()
            )
            for feature in features
        },
        "historical": {
            analysis: {feature: float(weight) for feature, weight in values.items()}
            for analysis, values in analyses.items()
        },
        "advanced": {
            "granger": {
                feature: float(selected_granger[feature]) for feature in features
            },
            "transfer_entropy": {
                feature: float(selected_entropy[feature]) for feature in features
            },
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = make_docs_client(timeout=300, max_retries=1)
    energy_summary = build_energy_case(client)
    explanation_summary = build_explanation_case(client)
    print("Energy summary:", energy_summary)
    print("Explanation summary:", explanation_summary)
    print("Wrote chart assets to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
