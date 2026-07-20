"""Example usage of EnergyLoadGenerator: electricity demand with daily/weekly
seasonality, temperature sensitivity, holiday effects, and extreme weather."""

import polars as pl

from synforecast.generators import EnergyLoadGenerator


def main() -> None:
    """Generate and display Energy Load examples."""
    # Residential load: morning/evening peaks
    residential_gen = EnergyLoadGenerator(
        min_length=168,  # 1 week of hourly data
        max_length=168,
        freq="h",
        engine="polars",
        load_type="residential",
        base_load=2.0,  # kW
        temperature_sensitivity=0.05,
        seed=42,
    )
    residential_df = residential_gen.generate(n_series=1)
    print("Residential load, first day:")
    print(residential_df.head(24))
    print(
        f"Mean={residential_df['y'].mean():.2f} kW, "
        f"Min={residential_df['y'].min():.2f} kW, "
        f"Max={residential_df['y'].max():.2f} kW"
    )

    # Commercial load: office-hours profile
    commercial_gen = EnergyLoadGenerator(
        min_length=168,
        max_length=168,
        freq="h",
        engine="polars",
        load_type="commercial",
        base_load=50.0,
        temperature_sensitivity=0.08,
        seed=42,
    )
    commercial_df = commercial_gen.generate(n_series=1)
    print(
        f"\nCommercial load: Mean={commercial_df['y'].mean():.2f} kW, "
        f"Min={commercial_df['y'].min():.2f} kW, "
        f"Max={commercial_df['y'].max():.2f} kW"
    )

    # Industrial load: flatter profile, less temperature sensitive
    industrial_gen = EnergyLoadGenerator(
        min_length=168,
        max_length=168,
        freq="h",
        engine="polars",
        load_type="industrial",
        base_load=200.0,
        temperature_sensitivity=0.02,
        seed=42,
    )
    industrial_df = industrial_gen.generate(n_series=1)
    print(
        f"\nIndustrial load: Mean={industrial_df['y'].mean():.2f} kW, "
        f"Min={industrial_df['y'].min():.2f} kW, "
        f"Max={industrial_df['y'].max():.2f} kW"
    )

    # Extreme weather events multiply the load
    extreme_gen = EnergyLoadGenerator(
        min_length=336,  # 2 weeks
        max_length=336,
        freq="h",
        engine="polars",
        load_type="residential",
        base_load=2.0,
        temperature_sensitivity=0.05,
        extreme_weather_prob=0.1,
        extreme_weather_multiplier=2.5,
        seed=42,
    )
    extreme_df = extreme_gen.generate(n_series=1)
    print(
        f"\nExtreme weather: Mean={extreme_df['y'].mean():.2f} kW, "
        f"Min={extreme_df['y'].min():.2f} kW, "
        f"Max={extreme_df['y'].max():.2f} kW"
    )

    # Holiday effect reduces commercial load on holidays/weekends
    holiday_gen = EnergyLoadGenerator(
        min_length=336,
        max_length=336,
        freq="h",
        engine="polars",
        load_type="commercial",
        base_load=50.0,
        temperature_sensitivity=0.08,
        holiday_effect=0.3,
        seed=42,
    )
    holiday_df = holiday_gen.generate(n_series=1)
    print(
        f"\nHoliday effect: Mean={holiday_df['y'].mean():.2f} kW, "
        f"Min={holiday_df['y'].min():.2f} kW, "
        f"Max={holiday_df['y'].max():.2f} kW"
    )

    # High temperature sensitivity, e.g. heavy AC/heating usage
    high_temp_gen = EnergyLoadGenerator(
        min_length=168,
        max_length=168,
        freq="h",
        engine="polars",
        load_type="residential",
        base_load=3.0,
        temperature_sensitivity=0.15,
        seed=42,
    )
    high_temp_df = high_temp_gen.generate(n_series=1)
    print(
        f"\nHigh temperature sensitivity: Mean={high_temp_df['y'].mean():.2f} kW, "
        f"Min={high_temp_df['y'].min():.2f} kW, "
        f"Max={high_temp_df['y'].max():.2f} kW"
    )

    # Multiple customers
    multi_gen = EnergyLoadGenerator(
        min_length=168,
        max_length=168,
        freq="h",
        engine="polars",
        load_type="residential",
        base_load=2.5,
        temperature_sensitivity=0.05,
        seed=42,
    )
    multi_df = multi_gen.generate(n_series=5)
    total_load = (
        multi_df.group_by("ds").agg(pl.col("y").sum()).select("y").mean().item()
    )
    print(
        f"\n5 residential customers: Mean={multi_df['y'].mean():.2f} kW, "
        f"Total load={total_load:.2f} kW"
    )
    print("First customer, first day:")
    print(multi_df.filter(pl.col("unique_id") == "0").head(24))

    residential_df.write_csv("energy_load_residential_example.csv")
    commercial_df.write_csv("energy_load_commercial_example.csv")
    industrial_df.write_csv("energy_load_industrial_example.csv")
    extreme_df.write_csv("energy_load_extreme_weather_example.csv")
    holiday_df.write_csv("energy_load_holiday_effect_example.csv")
    high_temp_df.write_csv("energy_load_high_temperature_example.csv")
    multi_df.write_csv("energy_load_multiple_customers_example.csv")


if __name__ == "__main__":
    main()
