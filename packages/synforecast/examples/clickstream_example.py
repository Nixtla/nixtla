"""Example usage of ClickstreamGenerator for web analytics time series."""

import polars as pl

from synforecast.generators import ClickstreamGenerator


def main() -> None:
    """Generate and display clickstream time series."""
    # One week of hourly session counts with bots and seasonality
    generator = ClickstreamGenerator(
        min_length=168,
        max_length=168,
        freq="h",
        engine="polars",
        base_sessions=500,
        traffic_source="mixed",
        conversion_rate=0.03,
        bounce_rate=0.40,
        include_seasonality=True,
        include_bots=True,
        bot_fraction=0.15,
        output_type="sessions",
        seed=42,
    )
    df = generator.generate(n_series=3)

    print(f"Generated {df['unique_id'].n_unique()} time series")
    print(f"Total hourly observations: {len(df)}")

    stats = df.group_by("unique_id").agg(
        [
            pl.col("y").sum().alias("total_sessions"),
            pl.col("y").mean().alias("avg_per_hour"),
            pl.col("y").max().alias("peak_hour"),
        ]
    )
    print(f"\nSession statistics:\n{stats}")

    # Traffic source comparison: conversion rates differ per source
    print("\nTraffic source comparison:")
    for source in ["organic", "paid", "direct", "referral"]:
        gen = ClickstreamGenerator(
            min_length=168,
            max_length=168,
            freq="h",
            engine="polars",
            base_sessions=500,
            traffic_source=source,
            output_type="conversions",
            seed=42,
        )
        source_df = gen.generate(n_series=1)
        conv_total = source_df["y"].sum()

        gen_sessions = ClickstreamGenerator(
            min_length=168,
            max_length=168,
            freq="h",
            engine="polars",
            base_sessions=500,
            traffic_source=source,
            output_type="sessions",
            seed=42,
        )
        session_df = gen_sessions.generate(n_series=1)
        session_total = session_df["y"].sum()

        conv_rate = conv_total / session_total * 100 if session_total > 0 else 0
        print(
            f"{source:10s}: {session_total:,.0f} sessions, "
            f"{conv_total:,.0f} conversions ({conv_rate:.2f}%)"
        )

    # Full metrics beyond the single output_type series
    full_metrics = generator.generate_full_metrics(n_series=1)

    print("\nComplete metrics:")
    print(f"Total sessions: {full_metrics['sessions'].sum():,.0f}")
    print(f"Total pageviews: {full_metrics['pageviews'].sum():,.0f}")
    print(f"Total conversions: {full_metrics['conversions'].sum():,.0f}")
    print(f"Total bounces: {full_metrics['bounces'].sum():,.0f}")
    print(f"Avg bounce rate: {full_metrics['bounce_rate'].mean() * 100:.1f}%")
    print(f"Avg conversion rate: {full_metrics['conversion_rate'].mean() * 100:.2f}%")
    print(f"Avg pages/session: {full_metrics['pages_per_session'].mean():.2f}")

    # Conversion funnel for 10,000 sessions
    funnel = generator.generate_funnel(n_sessions=10000)

    print("\nConversion funnel:")
    print("Stage                | Count    | Rate")
    prev_count = None
    for stage, count in funnel.items():
        rate = 100.0 if prev_count is None else count / prev_count * 100
        print(f"{stage:20s} | {count:8,d} | {rate:5.1f}%")
        prev_count = count

    overall_conv = funnel[list(funnel.keys())[-1]] / funnel["visit"] * 100
    print(f"Overall funnel conversion: {overall_conv:.2f}%")

    # Hour-of-day seasonality averaged over 4 weeks
    hourly_gen = ClickstreamGenerator(
        min_length=168 * 4,
        max_length=168 * 4,
        freq="h",
        engine="polars",
        base_sessions=1000,
        include_seasonality=True,
        include_bots=False,
        seed=42,
    )
    hourly_df = hourly_gen.generate(n_series=1)

    sessions = hourly_df["y"].to_numpy()
    hourly_avg = [sessions[i::24].mean() for i in range(24)]

    print("\nHour-of-day pattern (average sessions):")
    print("Hour | Avg Sessions | Relative")
    mean_hourly = sum(hourly_avg) / 24
    for hour, avg in enumerate(hourly_avg):
        relative = avg / mean_hourly
        bar = "*" * int(relative * 10)
        print(f"  {hour:02d} |    {avg:7.1f}  | {bar}")

    print("\nModel information:")
    info = generator.get_model_info()
    for key, value in info.items():
        if key != "source_params":
            print(f"{key}: {value}")

    output_file = "clickstream_data.csv"
    df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
