"""Vital Signs (Healthcare) time series generator."""

from typing import Literal

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import Field, model_validator

from synforecast._lib import domain as _rs_dom
from synforecast.base import BaseGenerator, _categorize_ids

_VITAL_SIGNS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature",
]

# Physiological baselines per patient archetype: mean/std of the baseline and
# hard clipping bounds (units: bpm, mmHg, breaths/min, %, Celsius)
_ARCHETYPES: dict[str, dict[str, dict[str, float]]] = {
    "healthy": {
        "heart_rate": {"mean": 70, "std": 8, "min": 50, "max": 100},
        "systolic_bp": {"mean": 120, "std": 10, "min": 90, "max": 140},
        "diastolic_bp": {"mean": 80, "std": 8, "min": 60, "max": 90},
        "respiratory_rate": {"mean": 14, "std": 2, "min": 10, "max": 20},
        "spo2": {"mean": 98, "std": 1, "min": 95, "max": 100},
        "temperature": {"mean": 36.8, "std": 0.3, "min": 36.0, "max": 37.5},
    },
    "cardiac": {
        "heart_rate": {"mean": 85, "std": 15, "min": 50, "max": 140},
        "systolic_bp": {"mean": 135, "std": 20, "min": 90, "max": 180},
        "diastolic_bp": {"mean": 85, "std": 12, "min": 60, "max": 110},
        "respiratory_rate": {"mean": 18, "std": 4, "min": 12, "max": 28},
        "spo2": {"mean": 95, "std": 2, "min": 88, "max": 99},
        "temperature": {"mean": 36.9, "std": 0.4, "min": 36.0, "max": 38.0},
    },
    "sepsis": {
        # Tachycardia, hypotension, tachypnea, fever
        "heart_rate": {"mean": 105, "std": 20, "min": 70, "max": 150},
        "systolic_bp": {"mean": 95, "std": 15, "min": 70, "max": 130},
        "diastolic_bp": {"mean": 60, "std": 10, "min": 40, "max": 85},
        "respiratory_rate": {"mean": 24, "std": 5, "min": 16, "max": 35},
        "spo2": {"mean": 93, "std": 3, "min": 85, "max": 98},
        "temperature": {"mean": 38.5, "std": 0.8, "min": 36.5, "max": 40.5},
    },
    "respiratory": {
        "heart_rate": {"mean": 80, "std": 12, "min": 55, "max": 120},
        "systolic_bp": {"mean": 125, "std": 12, "min": 95, "max": 150},
        "diastolic_bp": {"mean": 82, "std": 10, "min": 60, "max": 95},
        "respiratory_rate": {"mean": 22, "std": 4, "min": 14, "max": 32},
        "spo2": {"mean": 92, "std": 3, "min": 85, "max": 97},
        "temperature": {"mean": 37.2, "std": 0.5, "min": 36.2, "max": 38.5},
    },
    "hypertensive": {
        "heart_rate": {"mean": 78, "std": 10, "min": 55, "max": 110},
        "systolic_bp": {"mean": 155, "std": 15, "min": 130, "max": 200},
        "diastolic_bp": {"mean": 95, "std": 10, "min": 80, "max": 120},
        "respiratory_rate": {"mean": 16, "std": 3, "min": 12, "max": 24},
        "spo2": {"mean": 97, "std": 1.5, "min": 93, "max": 100},
        "temperature": {"mean": 36.8, "std": 0.3, "min": 36.0, "max": 37.5},
    },
}

_VITAL_TYPE_IDS = {vs: i for i, vs in enumerate(_VITAL_SIGNS)}

# Magnitude of the circadian effect per vital sign
_CIRCADIAN_MAGNITUDES = {
    "heart_rate": 8,
    "systolic_bp": 10,
    "diastolic_bp": 6,
    "respiratory_rate": 2,
    "spo2": 0.5,
    "temperature": 0.4,
}


class VitalSignsGenerator(BaseGenerator):
    """Generate realistic vital signs time series for healthcare applications.

    Simulates one of six vital signs (heart rate, systolic/diastolic blood
    pressure, respiratory rate, SpO2, temperature) as a per-series baseline
    plus a slow random-walk drift, a circadian rhythm, heart rate variability
    (for HR and BP), random physiological events (activity bursts, rest
    periods, spikes), measurement noise, and cross-vital correlations with
    heart rate. Values are clipped to physiological bounds that depend on the
    patient archetype.

    Note:
        Circadian and HRV components assume one time step = 1 minute
        (freq='min'); other frequencies distort those cycle periods.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Frequency of the data; use 'min' for correct
            circadian/HRV periods.
        patient_type (str): 'healthy', 'cardiac', 'sepsis', 'respiratory' or
            'hypertensive' (default: 'healthy').
        vital_sign (str): Which vital sign to output (default: 'heart_rate').
        include_circadian (bool): Include circadian rhythm effects
            (default: True).
        include_hrv (bool): Include heart rate variability (default: True).
        include_events (bool): Include random physiological events
            (default: True).
        event_probability (float): Per-step probability of an event
            (default: 0.01).
        seed (int | None): Random seed for reproducibility (default: None).

    Example:
        >>> gen = VitalSignsGenerator(
        ...     min_length=1440,  # 24 hours of per-minute data
        ...     max_length=1440,
        ...     freq="min",
        ...     patient_type="healthy",
        ...     vital_sign="heart_rate",
        ...     seed=42,
        ... )
        >>> df = gen.generate(n_series=10)
    """

    patient_type: Literal[
        "healthy", "cardiac", "sepsis", "respiratory", "hypertensive"
    ] = Field(default="healthy", description="Patient archetype")

    vital_sign: Literal[
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "spo2",
        "temperature",
    ] = Field(default="heart_rate", description="Vital sign to generate")

    include_circadian: bool = Field(
        default=True, description="Include circadian rhythm effects"
    )
    include_hrv: bool = Field(
        default=True, description="Include heart rate variability"
    )
    include_events: bool = Field(
        default=True, description="Include random physiological events"
    )
    event_probability: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Probability of event per timestep"
    )

    _baselines: dict = {}

    @model_validator(mode="after")
    def setup_patient_parameters(self) -> "VitalSignsGenerator":
        """Initialize patient-specific parameters based on archetype."""
        object.__setattr__(self, "_baselines", _ARCHETYPES[self.patient_type])
        return self

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        params = self._baselines[self.vital_sign]
        return (
            np.array(
                [
                    float(params["mean"]),
                    float(params["std"]),
                    float(params["min"]),
                    float(params["max"]),
                    1.0 if self.include_circadian else 0.0,
                    float(_CIRCADIAN_MAGNITUDES[self.vital_sign]),
                    1.0 if self.include_hrv else 0.0,
                    1.0 if self.include_events else 0.0,
                    self.event_probability,
                    float(_VITAL_TYPE_IDS[self.vital_sign]),
                ]
            ),
            [],
        )

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single vital signs time series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            Array of vital sign values.
        """
        params = self._baselines[self.vital_sign]

        seed = int(self.rng.integers(0, 2**63))
        return _rs_dom.vital_signs(
            length,
            params["mean"],
            params["std"],
            params["min"],
            params["max"],
            self.include_circadian,
            _CIRCADIAN_MAGNITUDES[self.vital_sign],
            self.include_hrv,
            self.include_events,
            self.event_probability,
            _VITAL_TYPE_IDS[self.vital_sign],
            seed,
        )

    def generate_all_vitals(
        self, n_series: int = 1, start_id: int = 0
    ) -> IntoDataFrameT:
        """Generate all six vital signs for complete patient monitoring.

        Args:
            n_series (int): Number of patients/series to generate.
            start_id (int): Starting ID for the series numbering.

        Returns:
            DataFrame with columns [id_col, time_col] plus one column per vital sign, aligned on the same timestamps per patient.
        """
        ids: list[np.ndarray] = []
        timestamps: list[np.ndarray] = []
        vitals: dict[str, list[np.ndarray]] = {vs: [] for vs in _VITAL_SIGNS}

        original_vital = self.vital_sign
        try:
            for i in range(n_series):
                length = int(self.rng.integers(self.min_length, self.max_length + 1))
                ids.append(np.full(length, start_id + i, dtype=np.int64))
                timestamps.append(self._timestamps(length))
                for vs in _VITAL_SIGNS:
                    object.__setattr__(self, "vital_sign", vs)
                    vitals[vs].append(self.generate_single_series(length))
        finally:
            object.__setattr__(self, "vital_sign", original_vital)

        result = {
            self.id_col: np.concatenate(ids),
            self.time_col: np.concatenate(timestamps),
        }
        for vs in _VITAL_SIGNS:
            result[vs] = np.concatenate(vitals[vs])

        df = nw.DataFrame.from_dict(result, backend=self.engine)
        return _categorize_ids(df, self.id_col).to_native()

    def get_model_info(self) -> dict:
        """Get information about the vital signs model.

        Returns:
            Model parameters and patient characteristics.
        """
        return {
            "patient_type": self.patient_type,
            "vital_sign": self.vital_sign,
            "baselines": self._baselines,
            "include_circadian": self.include_circadian,
            "include_hrv": self.include_hrv,
            "include_events": self.include_events,
        }
