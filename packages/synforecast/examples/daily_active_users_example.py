"""Example usage of Daily Active Users generator.

Simulates DAU for consumer, business, and gaming apps with weekly seasonality,
organic growth, and event-driven jumps; the 'event' column marks event days.
"""

import matplotlib.pyplot as plt
import polars as pl

from synforecast.generators import DailyActiveUsersGenerator


def main() -> None:
    """Generate and display Daily Active Users examples."""
    # Consumer app (social media)
    consumer_gen = DailyActiveUsersGenerator(
        min_length=90,  # 3 months of daily data
        max_length=90,
        freq="D",
        engine="polars",
        app_type="consumer",
        base_users=50000.0,
        growth_rate=0.001,  # 0.1% daily growth
        event_probability=0.03,  # chance of a marketing event per day
        event_impact_min=1.3,
        event_impact_max=1.8,
        event_decay_rate=0.15,
        seed=42,
    )
    consumer_df = consumer_gen.generate(n_series=1)

    print("Consumer app, first 14 days:")
    print(consumer_df.head(14))
    n_events = consumer_df["event"].sum()
    print(
        f"Events: {n_events} ({n_events / len(consumer_df) * 100:.1f}%), "
        f"DAU mean={consumer_df['y'].mean():,.0f}, "
        f"min={consumer_df['y'].min():,.0f}, max={consumer_df['y'].max():,.0f}"
    )

    # Business app (B2B SaaS) with strong weekday/weekend contrast
    business_gen = DailyActiveUsersGenerator(
        min_length=90,
        max_length=90,
        freq="D",
        engine="polars",
        app_type="business",
        base_users=5000.0,
        growth_rate=0.0005,
        weekend_factor=0.3,  # very low weekend usage
        event_probability=0.02,
        event_impact_min=1.2,
        event_impact_max=1.5,
        noise_std=0.03,
        seed=42,
    )
    business_df = business_gen.generate(n_series=1)

    print("\nBusiness app, first 14 days:")
    print(business_df.head(14))
    print(
        f"DAU mean={business_df['y'].mean():,.0f}, "
        f"min={business_df['y'].min():,.0f}, max={business_df['y'].max():,.0f}"
    )

    # Gaming app with higher weekend usage and frequent events
    gaming_gen = DailyActiveUsersGenerator(
        min_length=90,
        max_length=90,
        freq="D",
        engine="polars",
        app_type="gaming",
        base_users=100000.0,
        growth_rate=0.002,
        weekend_factor=1.4,
        event_probability=0.05,  # game updates, tournaments
        event_impact_min=1.5,
        event_impact_max=2.5,
        event_decay_rate=0.2,
        seed=42,
    )
    gaming_df = gaming_gen.generate(n_series=1)

    print(
        f"\nGaming app: {gaming_df['event'].sum()} events, "
        f"DAU mean={gaming_df['y'].mean():,.0f}, "
        f"min={gaming_df['y'].min():,.0f}, max={gaming_df['y'].max():,.0f}"
    )

    # Event impact on the consumer series
    event_days = consumer_df.filter(pl.col("event") == 1)
    if len(event_days) > 0:
        avg_event_day = consumer_df.filter(pl.col("event") == 1)["y"].mean()
        avg_normal_day = consumer_df.filter(pl.col("event") == 0)["y"].mean()
        print(
            f"\nConsumer event days: {len(event_days)}, "
            f"avg DAU on event days {avg_event_day:,.0f} vs "
            f"normal days {avg_normal_day:,.0f} "
            f"(lift {(avg_event_day / avg_normal_day - 1) * 100:.1f}%)"
        )

    # Hourly active users
    hourly_gen = DailyActiveUsersGenerator(
        min_length=168,  # 1 week of hourly data
        max_length=168,
        freq="h",
        engine="polars",
        app_type="consumer",
        base_users=10000.0,
        event_probability=0.01,
        noise_std=0.08,
        seed=42,
    )
    hourly_df = hourly_gen.generate(n_series=1)

    print("\nHourly active users, first 24 hours:")
    print(hourly_df.head(24))

    # Multiple products with per-series growth rate variation
    multi_gen = DailyActiveUsersGenerator(
        min_length=30,
        max_length=30,
        freq="D",
        engine="polars",
        app_type="consumer",
        base_users=20000.0,
        growth_rate_std=0.0003,
        event_probability=0.03,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=5)

    print("\nProduct 0 preview:")
    print(multi_df.filter(pl.col("unique_id") == "0").head(10))
    print("Events per product:")
    print(multi_df.group_by("unique_id").agg(pl.col("event").sum()))

    # Custom column names
    custom_gen = DailyActiveUsersGenerator(
        min_length=30,
        max_length=30,
        freq="D",
        engine="polars",
        base_users=15000.0,
        event_probability=0.05,
        event_col="marketing_campaign",
        id_col="product_id",
        time_col="date",
        target_col="dau",
        seed=42,
    )
    custom_df = custom_gen.generate(n_series=1)

    print(f"\nCustom columns: {custom_df.columns}")
    print(custom_df.head(10))

    # Plot 1: app type comparison with events highlighted
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, df, label in zip(
        axes,
        [consumer_df, business_df, gaming_df],
        ["Consumer App", "Business App", "Gaming App"],
        strict=True,
    ):
        ax.plot(df["ds"], df["y"], linewidth=0.8)
        event_mask = df["event"] == 1
        ax.scatter(
            df.filter(event_mask)["ds"],
            df.filter(event_mask)["y"],
            color="red",
            s=20,
            zorder=5,
            label="Event",
        )
        ax.set_ylabel("DAU")
        ax.set_title(label)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("DAU by App Type (90 days)", fontsize=14)
    fig.tight_layout()
    fig.savefig("dau_app_types.png", dpi=150)
    print("Saved dau_app_types.png")

    # Plot 2: multiple products overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    for sid in multi_df["unique_id"].unique().sort().to_list():
        series = multi_df.filter(pl.col("unique_id") == sid)
        ax.plot(series["ds"], series["y"], linewidth=0.8, label=str(sid))
    ax.set_xlabel("Date")
    ax.set_ylabel("DAU")
    ax.set_title("Multiple Products (30 days)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("dau_multi_products.png", dpi=150)
    print("Saved dau_multi_products.png")

    # Plot 3: long-horizon series with trend changepoints
    long_gen = DailyActiveUsersGenerator(
        min_length=800,
        max_length=800,
        freq="D",
        engine="polars",
        app_type="consumer",
        base_users=1000.0,
        growth_rate=0.003,
        growth_rate_std=0.001,
        event_probability=0.02,
        event_impact_min=1.3,
        event_impact_max=2.0,
        event_decay_rate=0.1,
        noise_std=0.01,
        seed=42,
        weekly_pattern=False,
        changepoints=True,
        changepoint_type="trend",
    )
    long_df = long_gen.generate(n_series=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for sid in long_df["unique_id"].unique().sort().to_list():
        series = long_df.filter(pl.col("unique_id") == sid)
        ax.plot(series["ds"], series["y"], linewidth=1, label=str(sid))
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.set_title("Long-horizon DAU (800 days, trend changepoints)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("dau_long_horizon.png", dpi=150)
    print("Saved dau_long_horizon.png")

    # Plot 4: hourly pattern
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hourly_df["ds"], hourly_df["y"], linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Active Users")
    ax.set_title("Hourly Active Users (1 week)")
    fig.tight_layout()
    fig.savefig("dau_hourly.png", dpi=150)
    print("Saved dau_hourly.png")

    plt.show()


if __name__ == "__main__":
    main()
