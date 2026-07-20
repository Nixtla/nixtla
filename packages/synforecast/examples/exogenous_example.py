"""Example: Generating time series with exogenous variables.

Demonstrates all exogenous variable types supported by SynForecast:
- Datetime calendar features (year, month, day_of_week, etc.)
- Cyclical sin/cos encodings (hour_sin, dow_cos, etc.)
- Anomaly, changepoint, and missing data flags
- Correlated exogenous variables (correlated noise, lagged copy, trend following)

Each section generates data and produces a plot showing the exogenous
features alongside the base time series.
"""

import matplotlib.pyplot as plt
import numpy as np

from synforecast.exogenous import CorrelatedExogConfig, ExogenousConfig
from synforecast.generators import RandomWalkGenerator


def plot_cyclical_features() -> None:
    """Plot cyclical datetime encodings alongside the time series."""
    gen = RandomWalkGenerator(
        min_length=168,  # 7 days of hourly data
        max_length=168,
        freq="h",
        engine="polars",
        seed=42,
        exogenous=ExogenousConfig(
            datetime_features=True,
            datetime_cyclical=True,
        ),
    )
    df = gen.generate(n_series=1)
    ts = df["ds"].to_numpy()
    y = df["y"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Panel 1: time series with weekend shading
    is_weekend = df["is_weekend"].to_numpy().astype(bool)
    axes[0].plot(ts, y, linewidth=0.8, color="steelblue", label="y")
    axes[0].fill_between(
        ts,
        y.min(),
        y.max(),
        where=is_weekend,
        alpha=0.15,
        color="salmon",
        label="Weekend",
    )
    axes[0].set_ylabel("y")
    axes[0].set_title("Time Series with Weekend Shading")
    axes[0].legend(loc="upper right")

    # Panel 2: hour-of-day cyclical encoding
    axes[1].plot(ts, df["hour_sin"].to_numpy(), label="hour_sin", color="darkorange")
    axes[1].plot(ts, df["hour_cos"].to_numpy(), label="hour_cos", color="purple")
    axes[1].set_ylabel("Encoding value")
    axes[1].set_title("Cyclical Hour-of-Day Encoding")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(-1.15, 1.15)

    # Panel 3: day-of-week cyclical encoding
    axes[2].plot(ts, df["dow_sin"].to_numpy(), label="dow_sin", color="teal")
    axes[2].plot(ts, df["dow_cos"].to_numpy(), label="dow_cos", color="crimson")
    axes[2].set_ylabel("Encoding value")
    axes[2].set_title("Cyclical Day-of-Week Encoding")
    axes[2].legend(loc="upper right")
    axes[2].set_ylim(-1.15, 1.15)

    fig.tight_layout()
    fig.savefig("exogenous_cyclical_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved exogenous_cyclical_features.png")


def plot_pattern_flags() -> None:
    """Plot anomaly, changepoint, and missing flags over the time series."""
    gen = RandomWalkGenerator(
        min_length=300,
        max_length=300,
        freq="D",
        engine="polars",
        seed=42,
        drift=0.1,
        volatility=1.5,
        anomalies=True,
        anomaly_fraction=0.05,
        anomaly_types=["spike", "dip"],
        spike_magnitude=15.0,
        dip_magnitude=-15.0,
        changepoints=True,
        num_changepoints=3,
        changepoint_type="level",
        missing_data=True,
        missing_rate=0.08,
        missing_pattern="block",
        missing_block_size=5,
        exogenous=ExogenousConfig(
            anomaly_flags=True,
            changepoint_flags=True,
            missing_flags=True,
        ),
    )
    df = gen.generate(n_series=1)
    ts = df["ds"].to_numpy()
    y = df["y"].to_numpy()
    anom = df["anomaly_flag"].to_numpy().astype(bool)
    cp = df["changepoint_flag"].to_numpy().astype(bool)
    miss = df["missing_flag"].to_numpy().astype(bool)

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]}
    )

    # Panel 1: time series with flagged points overlaid
    axes[0].plot(ts, y, linewidth=0.7, color="steelblue", label="y", zorder=1)
    if anom.any():
        axes[0].scatter(
            ts[anom],
            y[anom],
            color="red",
            s=30,
            zorder=3,
            label="Anomaly",
        )
    if cp.any():
        for t_cp in ts[cp]:
            axes[0].axvline(t_cp, color="green", linewidth=1.2, alpha=0.7)
        # single legend entry
        axes[0].axvline(
            ts[cp][0],
            color="green",
            linewidth=1.2,
            alpha=0.7,
            label="Changepoint",
        )
    if miss.any():
        axes[0].scatter(
            ts[miss],
            np.full(miss.sum(), np.nanmin(y) - 2),
            marker="|",
            color="orange",
            s=40,
            zorder=2,
            label="Missing",
        )
    axes[0].set_ylabel("y")
    axes[0].set_title("Time Series with Pattern Injection Flags")
    axes[0].legend(loc="upper left")

    # Panel 2-4: binary flag traces
    flag_info = [
        ("anomaly_flag", anom, "red"),
        ("changepoint_flag", cp, "green"),
        ("missing_flag", miss, "orange"),
    ]
    for ax, (name, flag, color) in zip(axes[1:], flag_info, strict=False):
        ax.fill_between(ts, 0, flag.astype(int), color=color, alpha=0.5)
        ax.set_ylabel(name, fontsize=9)
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1])

    fig.tight_layout()
    fig.savefig("exogenous_pattern_flags.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved exogenous_pattern_flags.png")


def plot_correlated_exogenous() -> None:
    """Plot correlated exogenous variables against the target series."""
    gen = RandomWalkGenerator(
        min_length=200,
        max_length=200,
        freq="D",
        engine="polars",
        seed=42,
        drift=0.05,
        volatility=1.0,
        exogenous=ExogenousConfig(
            correlated=[
                CorrelatedExogConfig(
                    name="corr_noise",
                    method="correlated_noise",
                    correlation=0.8,
                ),
                CorrelatedExogConfig(
                    name="lagged_y",
                    method="lagged_copy",
                    lag=7,
                    noise_std=0.3,
                ),
                CorrelatedExogConfig(
                    name="trend",
                    method="trend_following",
                    smoothing_window=14,
                    trend_noise_std=0.1,
                ),
            ]
        ),
    )
    df = gen.generate(n_series=1)
    ts = df["ds"].to_numpy()
    y = df["y"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: correlated noise
    ax = axes[0]
    ax.plot(ts, y, linewidth=0.8, color="steelblue", label="y")
    ax2 = ax.twinx()
    ax2.plot(
        ts,
        df["corr_noise"].to_numpy(),
        linewidth=0.8,
        color="darkorange",
        alpha=0.8,
        label="corr_noise (r=0.8)",
    )
    ax.set_ylabel("y", color="steelblue")
    ax2.set_ylabel("corr_noise", color="darkorange")
    corr = np.corrcoef(y, df["corr_noise"].to_numpy())[0, 1]
    ax.set_title(f"Correlated Noise (target r=0.8, actual r={corr:.3f})")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Panel 2: lagged copy
    ax = axes[1]
    ax.plot(ts, y, linewidth=0.8, color="steelblue", label="y")
    ax.plot(
        ts,
        df["lagged_y"].to_numpy(),
        linewidth=0.8,
        color="green",
        alpha=0.8,
        linestyle="--",
        label="lagged_y (lag=7)",
    )
    ax.set_ylabel("Value")
    ax.set_title("Lagged Copy (lag=7 days, noise_std=0.3)")
    ax.legend(loc="upper left")

    # Panel 3: trend following
    ax = axes[2]
    ax.plot(ts, y, linewidth=0.8, color="steelblue", label="y")
    ax.plot(
        ts,
        df["trend"].to_numpy(),
        linewidth=1.5,
        color="crimson",
        alpha=0.9,
        label="trend (window=14)",
    )
    ax.set_ylabel("Value")
    ax.set_title("Trend Following (smoothing_window=14)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig("exogenous_correlated.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved exogenous_correlated.png")


def plot_combined() -> None:
    """Plot a combined dashboard with all exogenous types at once."""
    gen = RandomWalkGenerator(
        min_length=200,
        max_length=200,
        freq="h",
        engine="polars",
        seed=42,
        drift=0.02,
        volatility=1.0,
        anomalies=True,
        anomaly_fraction=0.04,
        spike_magnitude=12.0,
        dip_magnitude=-12.0,
        changepoints=True,
        num_changepoints=2,
        missing_data=True,
        missing_rate=0.05,
        exogenous=ExogenousConfig(
            datetime_features=True,
            datetime_cyclical=True,
            anomaly_flags=True,
            changepoint_flags=True,
            missing_flags=True,
            correlated=[
                CorrelatedExogConfig(name="price", correlation=0.7),
                CorrelatedExogConfig(
                    name="trend",
                    method="trend_following",
                    smoothing_window=12,
                    trend_noise_std=0.05,
                ),
            ],
        ),
    )
    df = gen.generate(n_series=1)
    ts = df["ds"].to_numpy()
    y = df["y"].to_numpy()
    anom = df["anomaly_flag"].to_numpy().astype(bool)
    cp = df["changepoint_flag"].to_numpy().astype(bool)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1]},
    )

    # Panel 1: main series with annotations
    ax = axes[0]
    ax.plot(ts, y, linewidth=0.7, color="steelblue", label="y")
    ax.plot(
        ts,
        df["trend"].to_numpy(),
        linewidth=1.5,
        color="crimson",
        alpha=0.8,
        label="trend_following",
    )
    if anom.any():
        ax.scatter(ts[anom], y[anom], color="red", s=25, zorder=3, label="Anomaly")
    for t_cp in ts[cp]:
        ax.axvline(t_cp, color="green", linewidth=1, alpha=0.6)
    if cp.any():
        ax.axvline(
            ts[cp][0], color="green", linewidth=1, alpha=0.6, label="Changepoint"
        )
    ax.set_ylabel("y")
    ax.set_title("Combined Exogenous: Series + Trend + Flags")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: correlated price on twin axis
    ax = axes[1]
    ax.plot(ts, y, linewidth=0.6, color="steelblue", alpha=0.5, label="y")
    ax2 = ax.twinx()
    ax2.plot(
        ts,
        df["price"].to_numpy(),
        linewidth=0.7,
        color="darkorange",
        label="price (r=0.7)",
    )
    ax.set_ylabel("y", color="steelblue")
    ax2.set_ylabel("price", color="darkorange")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.set_title("Correlated Exogenous: price")

    # Panel 3: cyclical hour encoding
    ax = axes[2]
    ax.plot(
        ts,
        df["hour_sin"].to_numpy(),
        linewidth=0.8,
        color="darkorange",
        label="hour_sin",
    )
    ax.plot(
        ts, df["hour_cos"].to_numpy(), linewidth=0.8, color="purple", label="hour_cos"
    )
    ax.set_ylabel("Encoding")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Cyclical Hour Encoding")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 4: combined binary flags
    ax = axes[3]
    ax.fill_between(
        ts, 0, anom.astype(int) * 0.9 + 2.0, color="red", alpha=0.5, label="anomaly"
    )
    ax.fill_between(
        ts, 0, cp.astype(int) * 0.9 + 1.0, color="green", alpha=0.5, label="changepoint"
    )
    miss = df["missing_flag"].to_numpy().astype(bool)
    ax.fill_between(
        ts, 0, miss.astype(int) * 0.9, color="orange", alpha=0.5, label="missing"
    )
    ax.set_ylabel("Flags")
    ax.set_yticks([0.45, 1.45, 2.45])
    ax.set_yticklabels(["missing", "changepoint", "anomaly"], fontsize=8)
    ax.set_ylim(-0.1, 3.2)
    ax.set_title("Pattern Injection Flags")

    fig.tight_layout()
    fig.savefig("exogenous_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved exogenous_combined.png")

    print(f"\nGenerated DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {df.columns}")


def main() -> None:
    """Run all exogenous plotting examples."""
    plot_cyclical_features()
    plot_pattern_flags()
    plot_correlated_exogenous()
    plot_combined()


if __name__ == "__main__":
    main()
