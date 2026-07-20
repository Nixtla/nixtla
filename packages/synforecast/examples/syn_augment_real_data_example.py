"""SynAugment on real M4 competition data: compare statistical properties of
synthetic series against the originals. Requires datasetsforecast."""

import tempfile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from datasetsforecast.m4 import M4

from synforecast.dataset import SynAugment


def compute_series_stats(values: np.ndarray) -> dict:
    """Compute statistical properties of a time series."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0,
        "skewness": float(np.mean(((values - np.mean(values)) / np.std(values)) ** 3))
        if np.std(values) > 0
        else 0,
        "autocorr_lag1": float(np.corrcoef(values[:-1], values[1:])[0, 1])
        if len(values) > 1
        else 0,
    }


def print_comparison_table(original_stats: dict, synthetic_stats_list: list) -> None:
    """Print a comparison table of original vs synthetic statistics."""
    # Compute average stats across synthetic series
    avg_synthetic = {}
    for key in original_stats:
        avg_synthetic[key] = np.mean([s[key] for s in synthetic_stats_list])

    print(f"{'Statistic':<15} {'Original':>12} {'Synthetic (avg)':>15} {'Diff %':>10}")
    print("-" * 55)
    for key in original_stats:
        orig = original_stats[key]
        synth = avg_synthetic[key]
        if abs(orig) > 1e-6:
            diff_pct = abs(synth - orig) / abs(orig) * 100
        else:
            diff_pct = 0 if abs(synth) < 1e-6 else 100
        print(f"{key:<15} {orig:>12.4f} {synth:>15.4f} {diff_pct:>9.1f}%")


def plot_original_vs_synthetic(
    augmented_df: pl.DataFrame,
    original_id: str,
    n_synthetic: int = 3,
    title: str = "Original vs Synthetic Series",
) -> plt.Figure:
    """Plot original series alongside its synthetic augmentations.

    Args:
        augmented_df: DataFrame containing both original and synthetic series
        original_id: The unique_id of the original series
        n_synthetic: Number of synthetic series to plot
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(n_synthetic + 1, 1, figsize=(12, 3 * (n_synthetic + 1)))

    # Plot original series
    original = augmented_df.filter(pl.col("unique_id") == original_id).sort("ds")
    original_values = original["y"].to_numpy()
    original_ts = np.arange(len(original_values))

    axes[0].plot(original_ts, original_values, "b-", linewidth=1.5, label="Original")
    axes[0].set_title(f"Original: {original_id}", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Add stats annotation
    stats = compute_series_stats(original_values)
    stats_text = (
        f"μ={stats['mean']:.1f}, σ={stats['std']:.1f}, AC1={stats['autocorr_lag1']:.2f}"
    )
    axes[0].text(
        0.02,
        0.95,
        stats_text,
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # Plot synthetic series
    colors = plt.cm.Set2(np.linspace(0, 1, n_synthetic))
    for i in range(n_synthetic):
        aug_id = f"{original_id}_aug_{i}"
        synthetic = augmented_df.filter(pl.col("unique_id") == aug_id).sort("ds")

        if len(synthetic) == 0:
            continue

        syn_values = synthetic["y"].to_numpy()
        syn_ts = np.arange(len(syn_values))

        axes[i + 1].plot(
            syn_ts, syn_values, color=colors[i], linewidth=1.5, label=f"Synthetic {i}"
        )
        axes[i + 1].set_title(f"Synthetic: {aug_id}", fontsize=11)
        axes[i + 1].set_ylabel("Value")
        axes[i + 1].grid(True, alpha=0.3)
        axes[i + 1].legend(loc="upper right")

        # Add stats annotation
        syn_stats = compute_series_stats(syn_values)
        syn_stats_text = f"μ={syn_stats['mean']:.1f}, σ={syn_stats['std']:.1f}, AC1={syn_stats['autocorr_lag1']:.2f}"
        axes[i + 1].text(
            0.02,
            0.95,
            syn_stats_text,
            transform=axes[i + 1].transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.5},
        )

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


def plot_overlay_comparison(
    augmented_df: pl.DataFrame,
    original_id: str,
    n_synthetic: int = 3,
    title: str = "Overlay Comparison",
) -> plt.Figure:
    """Plot original and synthetic series overlaid on the same axes.

    Args:
        augmented_df: DataFrame containing both original and synthetic series
        original_id: The unique_id of the original series
        n_synthetic: Number of synthetic series to plot
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot original series
    original = augmented_df.filter(pl.col("unique_id") == original_id).sort("ds")
    original_values = original["y"].to_numpy()
    original_ts = np.arange(len(original_values))

    ax.plot(
        original_ts,
        original_values,
        "b-",
        linewidth=2.5,
        label=f"Original ({original_id})",
        zorder=10,
    )

    # Plot synthetic series with transparency
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, n_synthetic))
    for i in range(n_synthetic):
        aug_id = f"{original_id}_aug_{i}"
        synthetic = augmented_df.filter(pl.col("unique_id") == aug_id).sort("ds")

        if len(synthetic) == 0:
            continue

        syn_values = synthetic["y"].to_numpy()
        syn_ts = np.arange(len(syn_values))

        ax.plot(
            syn_ts,
            syn_values,
            color=colors[i],
            linewidth=1.2,
            alpha=0.7,
            label=f"Synthetic {i}",
        )

    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_distribution_comparison(
    augmented_df: pl.DataFrame,
    original_id: str,
    n_synthetic: int = 3,
    title: str = "Distribution Comparison",
) -> plt.Figure:
    """Plot distribution histograms comparing original and synthetic series.

    Args:
        augmented_df: DataFrame containing both original and synthetic series
        original_id: The unique_id of the original series
        n_synthetic: Number of synthetic series to include
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Get original values
    original = augmented_df.filter(pl.col("unique_id") == original_id).sort("ds")
    original_values = original["y"].to_numpy()

    # Collect all synthetic values
    all_synthetic_values = []
    for i in range(n_synthetic):
        aug_id = f"{original_id}_aug_{i}"
        synthetic = augmented_df.filter(pl.col("unique_id") == aug_id)
        if len(synthetic) > 0:
            all_synthetic_values.extend(synthetic["y"].to_list())
    all_synthetic_values = np.array(all_synthetic_values)

    # Plot 1: Value distribution
    bins = 30
    axes[0].hist(
        original_values,
        bins=bins,
        alpha=0.6,
        label="Original",
        color="blue",
        density=True,
    )
    axes[0].hist(
        all_synthetic_values,
        bins=bins,
        alpha=0.6,
        label="Synthetic (all)",
        color="orange",
        density=True,
    )
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Value Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: First differences distribution (captures volatility)
    original_diff = np.diff(original_values)
    synthetic_diff = np.diff(all_synthetic_values)

    axes[1].hist(
        original_diff,
        bins=bins,
        alpha=0.6,
        label="Original",
        color="blue",
        density=True,
    )
    axes[1].hist(
        synthetic_diff,
        bins=bins,
        alpha=0.6,
        label="Synthetic",
        color="orange",
        density=True,
    )
    axes[1].set_xlabel("First Difference")
    axes[1].set_ylabel("Density")
    axes[1].set_title("First Differences Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Autocorrelation comparison
    max_lag = min(50, len(original_values) // 4)
    original_acf = [
        np.corrcoef(original_values[:-lag], original_values[lag:])[0, 1]
        for lag in range(1, max_lag + 1)
    ]

    # Average ACF across synthetic series
    synthetic_acfs = []
    for i in range(n_synthetic):
        aug_id = f"{original_id}_aug_{i}"
        synthetic = augmented_df.filter(pl.col("unique_id") == aug_id)
        if len(synthetic) > 0:
            syn_vals = synthetic["y"].to_numpy()
            acf = [
                np.corrcoef(syn_vals[:-lag], syn_vals[lag:])[0, 1]
                for lag in range(1, min(max_lag + 1, len(syn_vals)))
            ]
            if len(acf) == max_lag:
                synthetic_acfs.append(acf)

    lags = np.arange(1, max_lag + 1)
    axes[2].plot(
        lags,
        original_acf,
        "b-",
        linewidth=2,
        label="Original",
        marker="o",
        markersize=3,
    )

    if synthetic_acfs:
        mean_synthetic_acf = np.mean(synthetic_acfs, axis=0)
        std_synthetic_acf = np.std(synthetic_acfs, axis=0)
        axes[2].plot(
            lags,
            mean_synthetic_acf,
            "orange",
            linewidth=2,
            label="Synthetic (mean)",
            marker="s",
            markersize=3,
        )
        axes[2].fill_between(
            lags,
            mean_synthetic_acf - std_synthetic_acf,
            mean_synthetic_acf + std_synthetic_acf,
            color="orange",
            alpha=0.2,
            label="Synthetic (±1 std)",
        )

    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Autocorrelation")
    axes[2].set_title("Autocorrelation Function")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


def plot_multiple_series_comparison(
    augmented_df: pl.DataFrame,
    original_ids: list,
    title: str = "Multiple Series Comparison",
) -> plt.Figure:
    """Plot multiple original series with one synthetic each for comparison.

    Args:
        augmented_df: DataFrame containing both original and synthetic series
        original_ids: List of original series unique_ids
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    n_series = len(original_ids)
    fig, axes = plt.subplots(n_series, 2, figsize=(14, 3 * n_series))

    if n_series == 1:
        axes = axes.reshape(1, -1)

    for idx, series_id in enumerate(original_ids):
        # Plot original
        original = augmented_df.filter(pl.col("unique_id") == series_id).sort("ds")
        if len(original) == 0:
            continue

        original_values = original["y"].to_numpy()
        original_ts = np.arange(len(original_values))

        axes[idx, 0].plot(original_ts, original_values, "b-", linewidth=1.5)
        axes[idx, 0].set_title(f"Original: {series_id}", fontsize=10)
        axes[idx, 0].set_ylabel("Value")
        axes[idx, 0].grid(True, alpha=0.3)

        # Plot first synthetic
        aug_id = f"{series_id}_aug_0"
        synthetic = augmented_df.filter(pl.col("unique_id") == aug_id).sort("ds")

        if len(synthetic) > 0:
            syn_values = synthetic["y"].to_numpy()
            syn_ts = np.arange(len(syn_values))

            axes[idx, 1].plot(syn_ts, syn_values, "orange", linewidth=1.5)
            axes[idx, 1].set_title(f"Synthetic: {aug_id}", fontsize=10)
            axes[idx, 1].set_ylabel("Value")
            axes[idx, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time Step")
    axes[-1, 1].set_xlabel("Time Step")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


def plot_stats_comparison_bar(
    original_stats: dict,
    synthetic_stats_list: list,
    title: str = "Statistics Comparison",
) -> plt.Figure:
    """Create bar chart comparing original and synthetic statistics.

    Args:
        original_stats: Dictionary of original series statistics
        synthetic_stats_list: List of dictionaries with synthetic series statistics
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    metrics = ["mean", "std", "cv", "autocorr_lag1"]
    metric_labels = ["Mean", "Std Dev", "Coef. of Variation", "Autocorr (lag 1)"]

    for ax, metric, label in zip(axes, metrics, metric_labels, strict=True):
        orig_val = original_stats[metric]
        syn_vals = [s[metric] for s in synthetic_stats_list]

        x = np.arange(len(syn_vals) + 1)
        values = [orig_val] + syn_vals
        colors = ["blue"] + ["orange"] * len(syn_vals)
        labels = ["Original"] + [f"Syn {i}" for i in range(len(syn_vals))]

        bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, values, strict=True):
            height = bar.get_height()
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


# Download M4 data into a temporary directory
with tempfile.TemporaryDirectory() as tmpdir:
    df_hourly, *_ = M4.load(directory=tmpdir, group="Hourly")

    df_hourly = pl.from_pandas(df_hourly)
    sample_ids = df_hourly["unique_id"].unique().head(5).to_list()
    df_sample = df_hourly.filter(pl.col("unique_id").is_in(sample_ids))

    print(f"Loaded {df_hourly['unique_id'].n_unique()} hourly series from M4")
    print(f"Using {len(sample_ids)} series: {sample_ids}")
    print(df_sample.filter(pl.col("unique_id") == sample_ids[0]).head(5))

    # Analyze the series to see which generators will be used
    augmenter = SynAugment(seed=42)
    analysis = augmenter.analyze(df_sample)

    for series_id in sample_ids:
        info = analysis[series_id]
        print(f"\n  {series_id}:")
        print(f"    Recommended generator: {info['recommended_generator']}")
        props = info["properties"]
        print(f"    Has seasonality: {props['seasonality']['has_seasonality']}")
        if props["seasonality"]["has_seasonality"]:
            print(f"    Seasonality period: {props['seasonality']['period']}")
        print(f"    Has trend: {props['trend']['has_trend']}")
        print(f"    Is stationary: {props['stationarity']['is_stationary']}")

    augmented_df = augmenter.augment(df_sample, n_augment=3)

    print(f"\nOriginal series: {df_sample['unique_id'].n_unique()}")
    print(f"Total series after augmentation: {augmented_df['unique_id'].n_unique()}")

    test_id = sample_ids[0]

    fig1 = plot_original_vs_synthetic(
        augmented_df,
        test_id,
        n_synthetic=3,
        title=f"Original vs Synthetic: {test_id} (M4 Hourly)",
    )
    fig1.savefig("synaugment_sidebyside.png", dpi=150, bbox_inches="tight")
    print("Saved synaugment_sidebyside.png")

    fig2 = plot_overlay_comparison(
        augmented_df,
        test_id,
        n_synthetic=3,
        title=f"Overlay Comparison: {test_id} - Original (blue) vs Synthetic (orange)",
    )
    fig2.savefig("synaugment_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved synaugment_overlay.png")

    fig3 = plot_distribution_comparison(
        augmented_df,
        test_id,
        n_synthetic=3,
        title=f"Distribution Comparison: {test_id}",
    )
    fig3.savefig("synaugment_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved synaugment_distributions.png")

    # Statistical comparison for one series
    print(f"\nDetailed comparison for series: {test_id}")

    original_values = (
        augmented_df.filter(pl.col("unique_id") == test_id).sort("ds")["y"].to_numpy()
    )
    original_stats = compute_series_stats(original_values)

    synthetic_stats_list = []
    for i in range(3):
        aug_id = f"{test_id}_aug_{i}"
        aug_values = (
            augmented_df.filter(pl.col("unique_id") == aug_id)
            .sort("ds")["y"]
            .to_numpy()
        )
        synthetic_stats_list.append(compute_series_stats(aug_values))

    print_comparison_table(original_stats, synthetic_stats_list)

    fig4 = plot_stats_comparison_bar(
        original_stats,
        synthetic_stats_list,
        title=f"Statistics Comparison: {test_id}",
    )
    fig4.savefig("synaugment_stats_bars.png", dpi=150, bbox_inches="tight")
    print("Saved synaugment_stats_bars.png")

    # M4 daily data
    df_daily, *_ = M4.load(directory=tmpdir, group="Daily")
    df_daily = pl.from_pandas(df_daily)

    daily_ids = df_daily["unique_id"].unique().head(4).to_list()
    df_daily_sample = df_daily.filter(pl.col("unique_id").is_in(daily_ids))

    print(f"\nLoaded {df_daily['unique_id'].n_unique()} daily series from M4")
    print(f"Using {len(daily_ids)} series: {daily_ids}")

    augmenter_daily = SynAugment(seed=123)
    analysis_daily = augmenter_daily.analyze(df_daily_sample)

    print("\nAnalysis results:")
    for series_id in daily_ids:
        info = analysis_daily[series_id]
        print(f"  {series_id}: {info['recommended_generator']}")

    augmented_daily = augmenter_daily.augment(df_daily_sample, n_augment=2)
    n_daily = augmented_daily["unique_id"].n_unique()
    print(f"\nAugmented from {len(daily_ids)} to {n_daily} series")

    fig5 = plot_multiple_series_comparison(
        augmented_daily,
        daily_ids,
        title="M4 Daily: Original vs Synthetic Comparison",
    )
    fig5.savefig("synaugment_daily_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved synaugment_daily_comparison.png")

    # Cross-series statistics: all original vs all synthetic hourly series
    original_means = []
    original_stds = []
    original_autocorrs = []

    for series_id in sample_ids:
        values = (
            df_sample.filter(pl.col("unique_id") == series_id)
            .sort("ds")["y"]
            .to_numpy()
        )
        original_means.append(np.mean(values))
        original_stds.append(np.std(values))
        if len(values) > 1:
            original_autocorrs.append(np.corrcoef(values[:-1], values[1:])[0, 1])

    synthetic_means = []
    synthetic_stds = []
    synthetic_autocorrs = []

    synthetic_ids = [
        uid
        for uid in augmented_df["unique_id"].unique().to_list()
        if "_aug_" in str(uid)
    ]

    for series_id in synthetic_ids:
        values = (
            augmented_df.filter(pl.col("unique_id") == series_id)
            .sort("ds")["y"]
            .to_numpy()
        )
        synthetic_means.append(np.mean(values))
        synthetic_stds.append(np.std(values))
        if len(values) > 1:
            synthetic_autocorrs.append(np.corrcoef(values[:-1], values[1:])[0, 1])

    print(f"\nOriginal series ({len(sample_ids)} series):")
    print(
        f"  Mean of means: {np.mean(original_means):.4f} (std: {np.std(original_means):.4f})"
    )
    print(
        f"  Mean of stds:  {np.mean(original_stds):.4f} (std: {np.std(original_stds):.4f})"
    )
    print(
        f"  Mean autocorr: {np.mean(original_autocorrs):.4f} (std: {np.std(original_autocorrs):.4f})"
    )

    print(f"\nSynthetic series ({len(synthetic_ids)} series):")
    print(
        f"  Mean of means: {np.mean(synthetic_means):.4f} (std: {np.std(synthetic_means):.4f})"
    )
    print(
        f"  Mean of stds:  {np.mean(synthetic_stds):.4f} (std: {np.std(synthetic_stds):.4f})"
    )
    print(
        f"  Mean autocorr: {np.mean(synthetic_autocorrs):.4f} (std: {np.std(synthetic_autocorrs):.4f})"
    )

    fig6, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Helper function to style boxplot
    def style_boxplot(bp, colors):
        for patch, color in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(color)

    # Mean comparison
    bp0 = axes[0].boxplot(
        [original_means, synthetic_means],
        labels=["Original", "Synthetic"],
        patch_artist=True,
    )
    style_boxplot(bp0, ["lightblue", "lightsalmon"])
    axes[0].set_ylabel("Mean Value")
    axes[0].set_title("Distribution of Series Means")
    axes[0].grid(True, alpha=0.3)

    # Std comparison
    bp1 = axes[1].boxplot(
        [original_stds, synthetic_stds],
        labels=["Original", "Synthetic"],
        patch_artist=True,
    )
    style_boxplot(bp1, ["lightblue", "lightsalmon"])
    axes[1].set_ylabel("Standard Deviation")
    axes[1].set_title("Distribution of Series Std Dev")
    axes[1].grid(True, alpha=0.3)

    # Autocorrelation comparison
    bp2 = axes[2].boxplot(
        [original_autocorrs, synthetic_autocorrs],
        labels=["Original", "Synthetic"],
        patch_artist=True,
    )
    style_boxplot(bp2, ["lightblue", "lightsalmon"])
    axes[2].set_ylabel("Autocorrelation (lag 1)")
    axes[2].set_title("Distribution of Series Autocorrelation")
    axes[2].grid(True, alpha=0.3)

    fig6.suptitle(
        "Cross-Series Statistics: Original vs Synthetic (M4 Hourly)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig6.savefig("synaugment_crossseries_stats.png", dpi=150, bbox_inches="tight")
    print("Saved synaugment_crossseries_stats.png")

    # Common use case: expand a small dataset for ML training
    small_dataset = df_hourly.filter(
        pl.col("unique_id").is_in(df_hourly["unique_id"].unique().head(10).to_list())
    )

    print(f"\nOriginal training set: {small_dataset['unique_id'].n_unique()} series")
    print(f"Total observations: {len(small_dataset)}")

    ml_augmenter = SynAugment(seed=42)
    expanded_dataset = ml_augmenter.augment(small_dataset, n_augment=5)

    print(f"\nExpanded training set: {expanded_dataset['unique_id'].n_unique()} series")
    print(f"Total observations: {len(expanded_dataset)}")
    print(f"Expansion factor: {len(expanded_dataset) / len(small_dataset):.1f}x")

    lengths = expanded_dataset.group_by("unique_id").agg(pl.len().alias("length"))
    print("\nSeries length distribution:")
    print(f"  Min: {lengths['length'].min()}, Max: {lengths['length'].max()}")
    print(f"  Mean: {lengths['length'].mean():.1f}")

    plt.close("all")
