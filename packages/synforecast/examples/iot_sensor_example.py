"""Example usage of IoT Sensor generator."""

import numpy as np
import polars as pl

from synforecast.generators import IoTSensorGenerator


def main() -> None:
    """Generate and display IoT sensor data with various patterns."""
    # Temperature sensor with gradual drift and calibration offset
    gen_temp = IoTSensorGenerator(
        min_length=500,
        max_length=500,
        freq="min",
        engine="polars",
        n_sensors=1,
        sensor_type="temperature",
        base_value=22.0,
        drift_rate=0.002,
        measurement_noise=0.1,
        calibration_error=0.5,
        seed=42,
    )
    df_temp = gen_temp.generate(n_series=1)

    print(f"Temperature sensor: {len(df_temp)} readings")
    print(df_temp.head(10))

    values = df_temp["y"].to_numpy()
    first_100 = np.mean(values[:100])
    last_100 = np.mean(values[-100:])
    print(
        f"Mean: {np.mean(values):.2f} C, Std: {np.std(values):.2f} C, "
        f"drift (first 100 vs last 100): {last_100 - first_100:.2f} C"
    )

    # Humidity sensor with a daily cycle at 1-minute resolution
    gen_humidity = IoTSensorGenerator(
        min_length=1440,  # 24 hours of 1-minute readings
        max_length=1440,
        freq="min",
        engine="polars",
        n_sensors=1,
        sensor_type="humidity",
        base_value=60.0,
        seasonal_period=1440,  # daily cycle
        seasonal_amplitude=15.0,
        measurement_noise=1.0,
        seed=123,
    )
    df_humidity = gen_humidity.generate(n_series=1)

    print("\nHumidity sensor, sampled every 2 hours:")
    print(
        df_humidity.filter(pl.col("ds").dt.minute() == 0)
        .filter(pl.col("ds").dt.hour() % 2 == 0)
        .head(12)
    )
    values_humidity = df_humidity["y"].to_numpy()
    print(
        f"Mean: {np.mean(values_humidity):.1f}%, "
        f"Min: {np.min(values_humidity):.1f}%, "
        f"Max: {np.max(values_humidity):.1f}%"
    )

    # Pressure sensor whose noise increases after battery degradation begins
    gen_pressure = IoTSensorGenerator(
        min_length=400,
        max_length=400,
        freq="min",
        engine="polars",
        n_sensors=1,
        sensor_type="pressure",
        base_value=1013.25,  # standard atmospheric pressure (hPa)
        measurement_noise=0.5,
        battery_life=200,  # degradation starts after 200 readings
        battery_degradation_rate=0.005,
        seed=456,
    )
    df_pressure = gen_pressure.generate(n_series=1)

    values_pressure = df_pressure["y"].to_numpy()
    first_half_std = np.std(values_pressure[:200])
    second_half_std = np.std(values_pressure[200:])
    print(
        f"\nPressure sensor noise before/after battery degradation: "
        f"{first_half_std:.2f} / {second_half_std:.2f} hPa "
        f"({(second_half_std / first_half_std - 1) * 100:.1f}% increase)"
    )

    # Light sensor with intermittent failures (NaN outages)
    gen_light = IoTSensorGenerator(
        min_length=300,
        max_length=300,
        freq="1s",
        engine="polars",
        n_sensors=1,
        sensor_type="light",
        base_value=800.0,
        measurement_noise=10.0,
        failure_probability=0.05,
        failure_type="intermittent",
        failure_duration=5,
        seed=789,
    )
    df_light = gen_light.generate(n_series=1)

    values_light = df_light["y"].to_numpy()
    nan_count = np.sum(np.isnan(values_light))
    valid_count = len(values_light) - nan_count
    print(
        f"\nLight sensor: {valid_count} valid readings, {nan_count} failed "
        f"({nan_count / len(values_light) * 100:.1f}%)"
    )

    # Sensor network: 4 spatially correlated temperature sensors
    gen_network = IoTSensorGenerator(
        min_length=200,
        max_length=200,
        freq="min",
        engine="polars",
        n_sensors=4,
        sensor_type="temperature",
        base_value=20.0,
        spatial_correlation=0.7,
        measurement_noise=0.5,
        seed=1011,
    )
    df_network = gen_network.generate(n_series=1)  # 1 network with 4 sensors

    n_sensors = df_network["unique_id"].n_unique()
    print(f"\nSensor network: {n_sensors} sensors, {len(df_network)} readings")
    print(df_network.head(10))

    sensor = {
        i: df_network.filter(pl.col("unique_id") == str(i))["y"].to_numpy()
        for i in range(4)
    }
    print("Spatial correlation between sensors:")
    print(f"  0-1 (adjacent): {np.corrcoef(sensor[0], sensor[1])[0, 1]:.3f}")
    print(f"  1-2 (adjacent): {np.corrcoef(sensor[1], sensor[2])[0, 1]:.3f}")
    print(f"  2-3 (adjacent): {np.corrcoef(sensor[2], sensor[3])[0, 1]:.3f}")
    print(f"  0-3 (distant):  {np.corrcoef(sensor[0], sensor[3])[0, 1]:.3f}")

    # Motion sensor with a chance of complete failure (all NaN after failure)
    gen_motion = IoTSensorGenerator(
        min_length=200,
        max_length=200,
        freq="100ms",
        engine="polars",
        n_sensors=1,
        sensor_type="motion",
        base_value=0.5,
        measurement_noise=0.2,
        failure_probability=0.3,
        failure_type="complete",
        seed=1213,
    )
    df_motion = gen_motion.generate(n_series=1)

    values_motion = df_motion["y"].to_numpy()
    nan_count_motion = np.sum(np.isnan(values_motion))
    if nan_count_motion > 0:
        first_nan = np.where(np.isnan(values_motion))[0][0]
        print(
            f"\nMotion sensor failed at reading {first_nan} "
            f"({nan_count_motion} failed readings)"
        )
    else:
        print("\nMotion sensor operated normally (no failure occurred)")

    # Multiple independent sensors: each generated series is its own sensor
    gen_multi = IoTSensorGenerator(
        min_length=100,
        max_length=100,
        freq="min",
        engine="polars",
        n_sensors=1,
        sensor_type="temperature",
        base_value=20.0,
        measurement_noise=0.3,
        seed=1415,
    )
    df_multi = gen_multi.generate(n_series=3)

    print(f"\nIndependent sensors: {df_multi['unique_id'].n_unique()}")
    for series_id in df_multi["unique_id"].unique().sort():
        values_series = df_multi.filter(pl.col("unique_id") == series_id)[
            "y"
        ].to_numpy()
        print(
            f"  Sensor {series_id}: Mean={np.mean(values_series):.2f} C, "
            f"Std={np.std(values_series):.2f} C"
        )

    outputs = {
        "iot_sensor_temperature_drift.csv": df_temp,
        "iot_sensor_humidity_seasonal.csv": df_humidity,
        "iot_sensor_pressure_battery.csv": df_pressure,
        "iot_sensor_light_failures.csv": df_light,
        "iot_sensor_network_multivariate.csv": df_network,
    }
    for filename, df in outputs.items():
        df.write_csv(filename)
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()
