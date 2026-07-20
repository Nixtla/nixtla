"""Example usage of VitalSignsGenerator for patient monitoring time series."""

from synforecast.generators import VitalSignsGenerator


def main() -> None:
    """Generate and display vital signs time series."""
    # 24 hours of per-minute heart rate for a healthy patient
    generator = VitalSignsGenerator(
        min_length=1440,
        max_length=1440,
        freq="min",
        engine="polars",
        patient_type="healthy",
        vital_sign="heart_rate",
        include_circadian=True,
        include_hrv=True,
        include_events=True,
        seed=42,
    )
    df = generator.generate(n_series=1)

    values = df["y"].to_numpy()
    print("Heart rate, healthy patient (24 hours):")
    print(f"Range: [{values.min():.1f}, {values.max():.1f}] bpm")
    print(f"Mean: {values.mean():.1f} bpm, std: {values.std():.1f} bpm")

    print("\nHeart rate by patient type:")
    for ptype in ["healthy", "cardiac", "sepsis", "respiratory", "hypertensive"]:
        gen = VitalSignsGenerator(
            min_length=1440,
            max_length=1440,
            freq="min",
            engine="polars",
            patient_type=ptype,
            vital_sign="heart_rate",
            seed=42,
        )
        df_patient = gen.generate(n_series=1)
        hr = df_patient["y"].to_numpy()
        print(
            f"{ptype:15s}: mean={hr.mean():.1f}, std={hr.std():.1f}, "
            f"range=[{hr.min():.0f}, {hr.max():.0f}]"
        )

    # All vital signs at once for a sepsis patient (8 hours)
    sepsis_gen = VitalSignsGenerator(
        min_length=480,
        max_length=480,
        freq="min",
        engine="polars",
        patient_type="sepsis",
        seed=42,
    )
    all_vitals_df = sepsis_gen.generate_all_vitals(n_series=1)

    print("\nAll vital signs, sepsis patient:")
    vital_cols = [
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "spo2",
        "temperature",
    ]
    for col in vital_cols:
        vals = all_vitals_df[col].to_numpy()
        print(
            f"  {col:18s}: mean={vals.mean():.1f}, "
            f"range=[{vals.min():.1f}, {vals.max():.1f}]"
        )

    print("\nOxygen saturation (SpO2) by patient type:")
    for ptype in ["healthy", "respiratory", "sepsis"]:
        gen = VitalSignsGenerator(
            min_length=1440,
            max_length=1440,
            freq="min",
            engine="polars",
            patient_type=ptype,
            vital_sign="spo2",
            seed=42,
        )
        df_spo2 = gen.generate(n_series=1)
        spo2 = df_spo2["y"].to_numpy()
        below_95 = (spo2 < 95).sum() / len(spo2) * 100
        print(
            f"{ptype:12s}: mean={spo2.mean():.1f}%, min={spo2.min():.1f}%, "
            f"time <95%: {below_95:.1f}%"
        )

    info = generator.get_model_info()
    print("\nModel information:")
    print(f"Patient type: {info['patient_type']}")
    print(f"Current vital sign: {info['vital_sign']}")
    print(f"Circadian rhythm: {info['include_circadian']}")
    print(f"HRV included: {info['include_hrv']}")

    print("\nBaseline values for healthy patient:")
    for vital, params in info["baselines"].items():
        print(
            f"  {vital}: mean={params['mean']}, "
            f"range=[{params['min']}, {params['max']}]"
        )

    output_file = "vital_signs_data.csv"
    all_vitals_df.write_csv(output_file)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
