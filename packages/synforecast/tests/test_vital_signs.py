"""Tests for VitalSignsGenerator."""

import numpy as np
import pytest
from pydantic import ValidationError

from synforecast.generators import VitalSignsGenerator
from synforecast.generators.vital_signs import _ARCHETYPES, _VITAL_SIGNS
from tests.helpers import assert_long_format, series_values, to_pandas


def make_gen(**kwargs):
    params = {
        "min_length": 200,
        "max_length": 300,
        "freq": "min",
        "seed": 42,
    }
    params.update(kwargs)
    return VitalSignsGenerator(**params)


class TestVitalSignsAPI:
    def test_long_format(self, engine: str) -> None:
        df = make_gen(engine=engine).generate(n_series=3)
        assert_long_format(df, n_series=3, min_length=200, max_length=300)

    def test_seed_determinism(self) -> None:
        v1 = series_values(make_gen(seed=7).generate(n_series=3))
        v2 = series_values(make_gen(seed=7).generate(n_series=3))
        assert v1.keys() == v2.keys()
        for k in v1:
            np.testing.assert_array_equal(v1[k], v2[k])

    def test_all_vital_signs_generate(self) -> None:
        for vital in _VITAL_SIGNS:
            values = make_gen(vital_sign=vital).generate_single_series(200)
            assert values.shape == (200,)
            assert np.isfinite(values).all()

    def test_generate_all_vitals(self, engine: str) -> None:
        """Multi-vital mode outputs all six vitals aligned per patient."""
        df = make_gen(engine=engine).generate_all_vitals(n_series=2)
        pdf = to_pandas(df)
        assert set(pdf.columns) == {"unique_id", "ds"} | set(_VITAL_SIGNS)
        assert pdf["unique_id"].nunique() == 2
        lengths = pdf.groupby("unique_id", observed=True).size()
        assert ((lengths >= 200) & (lengths <= 300)).all()
        for vital in _VITAL_SIGNS:
            assert np.isfinite(pdf[vital].to_numpy()).all()

    def test_generate_all_vitals_respects_start_id(self) -> None:
        pdf = to_pandas(make_gen().generate_all_vitals(n_series=2, start_id=5))
        assert sorted(int(i) for i in pdf["unique_id"].unique()) == [5, 6]

    def test_get_model_info(self) -> None:
        info = make_gen(patient_type="sepsis", vital_sign="spo2").get_model_info()
        assert info["patient_type"] == "sepsis"
        assert info["vital_sign"] == "spo2"
        assert info["baselines"] == _ARCHETYPES["sepsis"]

    def test_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            make_gen(patient_type="zombie")
        with pytest.raises(ValidationError):
            make_gen(vital_sign="blood_sugar")
        with pytest.raises(ValidationError):
            make_gen(event_probability=1.5)


@pytest.mark.stats
class TestVitalSignsStats:
    @pytest.mark.parametrize("patient_type", list(_ARCHETYPES))
    def test_physiological_ranges(self, patient_type: str) -> None:
        """Every vital stays within the archetype's physiological bounds."""
        for vital in _VITAL_SIGNS:
            gen = make_gen(patient_type=patient_type, vital_sign=vital, seed=1)
            values = gen.generate_single_series(400)
            bounds = _ARCHETYPES[patient_type][vital]
            assert values.min() >= bounds["min"], f"{patient_type}/{vital}"
            assert values.max() <= bounds["max"], f"{patient_type}/{vital}"

    def test_archetype_separation(self) -> None:
        """Septic patients are tachycardic and febrile vs healthy ones."""

        def mean_of(patient_type: str, vital: str) -> float:
            gen = make_gen(
                min_length=500, max_length=500, patient_type=patient_type,
                vital_sign=vital, seed=2,
            )  # fmt: skip
            return float(
                np.mean([gen.generate_single_series(500).mean() for _ in range(4)])
            )

        assert mean_of("sepsis", "heart_rate") > mean_of("healthy", "heart_rate") + 10
        assert (
            mean_of("sepsis", "temperature") > mean_of("healthy", "temperature") + 0.5
        )
        assert (
            mean_of("hypertensive", "systolic_bp")
            > mean_of("healthy", "systolic_bp") + 15
        )

    def test_circadian_component_shape(self) -> None:
        """Averaged series correlate with the 1440-minute circadian profile."""
        gen = make_gen(
            min_length=2880,
            max_length=2880,
            vital_sign="heart_rate",
            patient_type="cardiac",  # wide bounds limit clipping
            include_hrv=False,
            include_events=False,
            seed=3,
        )
        avg = np.mean([gen.generate_single_series(2880) for _ in range(16)], axis=0)
        t = np.arange(2880)
        expected = np.sin(2 * np.pi * t / 1440 - np.pi / 2) + 0.3 * np.sin(
            4 * np.pi * t / 1440
        )
        corr = np.corrcoef(avg, expected)[0, 1]
        assert corr > 0.5

    def test_no_circadian_flattens_profile(self) -> None:
        """Disabling circadian removes the day/night correlation."""
        gen = make_gen(
            min_length=2880,
            max_length=2880,
            vital_sign="heart_rate",
            patient_type="cardiac",
            include_circadian=False,
            include_hrv=False,
            include_events=False,
            seed=3,
        )
        avg = np.mean([gen.generate_single_series(2880) for _ in range(16)], axis=0)
        t = np.arange(2880)
        expected = np.sin(2 * np.pi * t / 1440 - np.pi / 2)
        corr = np.corrcoef(avg, expected)[0, 1]
        assert abs(corr) < 0.5

    def test_events_increase_variability(self) -> None:
        common = {
            "min_length": 800,
            "max_length": 800,
            "vital_sign": "heart_rate",
            "include_circadian": False,
            "include_hrv": False,
            "seed": 4,
        }
        std_with = np.mean(
            [
                make_gen(include_events=True, event_probability=0.05, **common)
                .generate_single_series(800)
                .std()
                for _ in range(3)
            ]
        )
        std_without = np.mean(
            [
                make_gen(include_events=False, **common)
                .generate_single_series(800)
                .std()
                for _ in range(3)
            ]
        )
        assert std_with > std_without

    def test_hrv_increases_variability(self) -> None:
        common = {
            "min_length": 800,
            "max_length": 800,
            "vital_sign": "heart_rate",
            "include_circadian": False,
            "include_events": False,
            "seed": 5,
        }
        std_with = (
            make_gen(include_hrv=True, **common).generate_single_series(800).std()
        )
        std_without = (
            make_gen(include_hrv=False, **common).generate_single_series(800).std()
        )
        assert std_with > std_without
