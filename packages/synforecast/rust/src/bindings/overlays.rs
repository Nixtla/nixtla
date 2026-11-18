use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::overlays::{
    apply_overlays as apply, AnomalyPlan, CalendarPlan, ChangepointPlan, CyclePlan, MissingPlan,
    ObservationPlan, OverlayPlan, Profile, WaveShape, WaveTiming, WaveformPlan,
};

/// Required key of a plan dict; a missing key or a wrong type is an error, never a default.
fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("overlay plan is missing '{key}'")))?
        .extract::<T>()
        .map_err(|e| PyValueError::new_err(format!("overlay plan '{key}': {}", e.into())))
}

/// Optional sub-dict: absent or None means the stage is off.
fn section<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.cast_into::<PyDict>()?)),
        _ => Ok(None),
    }
}

fn get_array(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let item = d
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("overlay plan is missing '{key}'")))?;
    if let Ok(arr) = item.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(arr.as_array().to_vec());
    }
    item.extract::<Vec<f64>>()
        .map_err(|e| PyValueError::new_err(format!("overlay plan '{key}': {e}")))
}

fn get_optional_array(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<f64>>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => get_array(d, key).map(Some),
        _ => Ok(None),
    }
}

fn invalid(key: &str, value: &str) -> PyErr {
    PyValueError::new_err(format!("overlay plan '{key}': unknown value '{value}'"))
}

fn parse_cycle(d: &Bound<'_, PyDict>) -> PyResult<CyclePlan> {
    let kind: String = get(d, "profile")?;
    let profile = match kind.as_str() {
        "fourier" => Profile::Fourier {
            harmonics: get(d, "harmonics")?,
        },
        "levels" => Profile::Levels {
            bins: get(d, "bins")?,
        },
        other => return Err(invalid("profile", other)),
    };
    let values = get_optional_array(d, "values")?;
    let expected = match profile {
        Profile::Fourier { harmonics } => 2 * harmonics,
        Profile::Levels { bins } => bins,
    };
    if expected == 0 {
        return Err(PyValueError::new_err(
            "a calendar cycle needs at least one harmonic or bin",
        ));
    }
    if let Some(v) = &values {
        if v.len() != expected {
            return Err(PyValueError::new_err(format!(
                "calendar cycle values: expected {expected}, got {}",
                v.len()
            )));
        }
    }
    let phase = get_array(d, "phase")?;
    if phase.iter().any(|p| !(0.0..1.0).contains(p)) {
        return Err(PyValueError::new_err("calendar phases must lie in [0, 1)"));
    }
    Ok(CyclePlan {
        profile,
        values,
        jitter: get(d, "jitter")?,
        weight: get(d, "weight")?,
        phase,
    })
}

fn parse_calendar(d: &Bound<'_, PyDict>) -> PyResult<CalendarPlan> {
    let cycles_obj = d
        .get_item("cycles")?
        .ok_or_else(|| PyValueError::new_err("overlay plan is missing 'cycles'"))?;
    let cycles = cycles_obj
        .cast::<PyList>()?
        .iter()
        .map(|c| parse_cycle(c.cast::<PyDict>()?))
        .collect::<PyResult<Vec<_>>>()?;
    if cycles.is_empty() {
        return Err(PyValueError::new_err("a calendar needs at least one cycle"));
    }
    Ok(CalendarPlan {
        cycles,
        multiplicative: get::<String>(d, "mode")? == "multiplicative",
        amplitude: get(d, "amplitude")?,
        swing: get(d, "swing")?,
        relative: get(d, "relative")?,
        fallback_additive: get(d, "fallback_additive")?,
    })
}

fn parse_waveform(d: &Bound<'_, PyDict>) -> PyResult<WaveformPlan> {
    let shape: String = get(d, "shape")?;
    let shape = match shape.as_str() {
        "pulse" => WaveShape::Pulse,
        "spike" => WaveShape::Spike,
        "sawtooth" => WaveShape::Sawtooth,
        "trapezoid" => WaveShape::Trapezoid,
        "levels" => WaveShape::Levels,
        "staircase" => WaveShape::Staircase,
        other => return Err(invalid("shape", other)),
    };
    let timing: String = get(d, "timing")?;
    let timing = match timing.as_str() {
        "periodic" => WaveTiming::Periodic,
        "jitter" => WaveTiming::Jitter,
        "renewal" => WaveTiming::Renewal,
        other => return Err(invalid("timing", other)),
    };
    let period: usize = get(d, "period")?;
    let levels: usize = get(d, "levels")?;
    if period < 2 || levels < 2 {
        return Err(PyValueError::new_err(
            "waveform period and levels must be >= 2",
        ));
    }
    Ok(WaveformPlan {
        shape,
        period,
        duty: get(d, "duty")?,
        width: get(d, "width")?,
        ramp: get(d, "ramp")?,
        levels,
        timing,
        jitter: get(d, "jitter")?,
        lognormal: get::<String>(d, "renewal")? == "lognormal",
        cv: get(d, "cv")?,
        dip: get::<String>(d, "direction")? == "dip",
        multiplicative: get::<String>(d, "mode")? == "multiplicative",
        amplitude: get(d, "amplitude")?,
        swing: get(d, "swing")?,
        relative: get(d, "relative")?,
        fallback_additive: get(d, "fallback_additive")?,
    })
}

fn parse_observation(d: &Bound<'_, PyDict>) -> PyResult<ObservationPlan> {
    let hold: usize = get(d, "hold")?;
    let stuck_length: usize = get(d, "stuck_length")?;
    if hold < 1 || stuck_length < 2 {
        return Err(PyValueError::new_err(
            "observation hold must be >= 1 and stuck_length >= 2",
        ));
    }
    Ok(ObservationPlan {
        floor: get(d, "floor")?,
        cap: get(d, "cap")?,
        quantum: get(d, "quantum")?,
        hold,
        stuck_fraction: get(d, "stuck_fraction")?,
        stuck_length,
    })
}

/// Parse the dict built by `BaseGenerator._overlay_plan_dict` into an [`OverlayPlan`].
pub fn parse_plan(d: &Bound<'_, PyDict>) -> PyResult<OverlayPlan> {
    let changepoints = section(d, "changepoints")?
        .map(|c| -> PyResult<ChangepointPlan> {
            Ok(ChangepointPlan {
                count: get(&c, "count")?,
                kind: get(&c, "kind")?,
                locations: get_array(&c, "locations")?,
                level_changes: get_array(&c, "level_changes")?,
                trend_changes: get_array(&c, "trend_changes")?,
                variance_changes: get_array(&c, "variance_changes")?,
                relative: get(&c, "relative")?,
            })
        })
        .transpose()?;
    let anomalies = section(d, "anomalies")?
        .map(|a| -> PyResult<AnomalyPlan> {
            let types: Vec<String> = get(&a, "types")?;
            if types.is_empty() {
                return Err(PyValueError::new_err("anomalies need at least one type"));
            }
            Ok(AnomalyPlan {
                types,
                fraction: get(&a, "fraction")?,
                spike_magnitude: get(&a, "spike_magnitude")?,
                dip_magnitude: get(&a, "dip_magnitude")?,
                level_shift_magnitude: get(&a, "level_shift_magnitude")?,
                level_shift_duration: get(&a, "level_shift_duration")?,
                relative: get(&a, "relative")?,
            })
        })
        .transpose()?;
    let missing = section(d, "missing")?
        .map(|m| -> PyResult<MissingPlan> {
            let block_size: i32 = get(&m, "block_size")?;
            let seasonal_period: i32 = get(&m, "seasonal_period")?;
            if block_size < 1 || seasonal_period < 1 {
                return Err(PyValueError::new_err(
                    "missing block_size and seasonal_period must be >= 1",
                ));
            }
            Ok(MissingPlan {
                pattern: get(&m, "pattern")?,
                rate: get(&m, "rate")?,
                block_size,
                seasonal_period,
            })
        })
        .transpose()?;
    Ok(OverlayPlan {
        base_weight: get(d, "base_weight")?,
        calendar: section(d, "calendar")?
            .map(|c| parse_calendar(&c))
            .transpose()?,
        waveform: section(d, "waveform")?
            .map(|w| parse_waveform(&w))
            .transpose()?,
        changepoints,
        anomalies,
        observation: section(d, "observation")?
            .map(|o| parse_observation(&o))
            .transpose()?,
        missing,
    })
}

/// A parsed overlay plan, built once per draw and shared by every series.
#[pyclass(frozen, name = "OverlayPlan", module = "synforecast._lib.overlays")]
pub struct PyOverlayPlan {
    pub inner: Arc<OverlayPlan>,
}

#[pymethods]
impl PyOverlayPlan {
    #[new]
    fn new(plan: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(parse_plan(plan)?),
        })
    }
}

type OverlayOutput<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// Run the overlay pipeline on one series; returns (values, changepoint, anomaly, missing indices).
#[pyfunction]
pub fn apply_overlays<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    plan: &Bound<'py, PyOverlayPlan>,
    raw_seeds: [u64; 3],
    legacy_seeds: [u64; 3],
) -> PyResult<OverlayOutput<'py>> {
    let mut vals = values.as_array().to_vec();
    let plan = Arc::clone(&plan.get().inner);
    plan.check_length(vals.len())
        .map_err(PyValueError::new_err)?;
    let indices = py.detach(|| apply(&mut vals, &plan, raw_seeds, legacy_seeds));
    Ok((
        PyArray1::from_vec(py, vals),
        PyArray1::from_vec(py, indices.changepoints),
        PyArray1::from_vec(py, indices.anomalies),
        PyArray1::from_vec(py, indices.missing),
    ))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "overlays")?;
    m.add_class::<PyOverlayPlan>()?;
    m.add_function(wrap_pyfunction!(apply_overlays, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.overlays", &m)?;
    Ok(())
}
