use crate::pattern_injection as pi;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (values, seed, num_changepoints, locations, changepoint_type, level_changes, trend_changes, variance_changes))]
pub fn add_changepoints<'py>(
    py: Python<'py>,
    values: Bound<'py, PyArray1<f64>>,
    seed: u64,
    num_changepoints: i32,
    locations: PyReadonlyArray1<'py, f64>,
    changepoint_type: &str,
    level_changes: PyReadonlyArray1<'py, f64>,
    trend_changes: PyReadonlyArray1<'py, f64>,
    variance_changes: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyAny>> {
    // SAFETY: Caller ensures exclusive access to this array during the call.
    let vals = unsafe { values.as_slice_mut()? };
    let locs = locations.as_slice()?;
    let lc = level_changes.as_slice()?;
    let tc = trend_changes.as_slice()?;
    let vc = variance_changes.as_slice()?;

    let result = py.detach(|| {
        pi::add_changepoints(
            vals,
            seed,
            num_changepoints,
            locs,
            changepoint_type,
            lc,
            tc,
            vc,
        )
    });

    let indices = PyArray1::from_vec(py, result.changepoint_indices);
    let metadata = PyDict::new(py);
    metadata.set_item("changepoint_indices", indices)?;
    let tuple = pyo3::types::PyTuple::new(py, [values.as_any(), metadata.as_any()])?;
    Ok(tuple.into())
}

#[pyfunction]
#[pyo3(signature = (values, seed, pattern, missing_rate, missing_block_size, missing_seasonal_period))]
pub fn add_missingness<'py>(
    py: Python<'py>,
    values: Bound<'py, PyArray1<f64>>,
    seed: u64,
    pattern: &str,
    missing_rate: f64,
    missing_block_size: i32,
    missing_seasonal_period: i32,
) -> PyResult<Py<PyAny>> {
    if pattern == "block" && missing_block_size <= 0 {
        return Err(PyValueError::new_err(
            "missing_block_size must be > 0 for block pattern",
        ));
    }
    if pattern == "seasonal" && missing_seasonal_period <= 0 {
        return Err(PyValueError::new_err(
            "missing_seasonal_period must be > 0 for seasonal pattern",
        ));
    }
    // SAFETY: Caller ensures exclusive access to this array during the call.
    let vals = unsafe { values.as_slice_mut()? };

    let result = py.detach(|| {
        pi::add_missingness(
            vals,
            seed,
            pattern,
            missing_rate,
            missing_block_size,
            missing_seasonal_period,
        )
    });

    let indices = PyArray1::from_vec(py, result.missing_indices);
    let metadata = PyDict::new(py);
    metadata.set_item("missing_indices", indices)?;
    let tuple = pyo3::types::PyTuple::new(py, [values.as_any(), metadata.as_any()])?;
    Ok(tuple.into())
}

#[pyfunction]
#[pyo3(signature = (values, seed, anomaly_types, anomaly_fraction, spike_magnitude, dip_magnitude, level_shift_magnitude, level_shift_duration))]
pub fn add_anomalies<'py>(
    py: Python<'py>,
    values: Bound<'py, PyArray1<f64>>,
    seed: u64,
    anomaly_types: Vec<String>,
    anomaly_fraction: f64,
    spike_magnitude: f64,
    dip_magnitude: f64,
    level_shift_magnitude: f64,
    level_shift_duration: i32,
) -> PyResult<Py<PyAny>> {
    // SAFETY: Caller ensures exclusive access to this array during the call.
    let vals = unsafe { values.as_slice_mut()? };

    let result = py.detach(|| {
        pi::add_anomalies(
            vals,
            seed,
            &anomaly_types,
            anomaly_fraction,
            spike_magnitude,
            dip_magnitude,
            level_shift_magnitude,
            level_shift_duration,
        )
    });

    let indices = PyArray1::from_vec(py, result.anomaly_indices);
    let metadata = PyDict::new(py);
    metadata.set_item("anomaly_indices", indices)?;
    let tuple = pyo3::types::PyTuple::new(py, [values.as_any(), metadata.as_any()])?;
    Ok(tuple.into())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pattern_injection")?;
    m.add_function(wrap_pyfunction!(add_changepoints, &m)?)?;
    m.add_function(wrap_pyfunction!(add_missingness, &m)?)?;
    m.add_function(wrap_pyfunction!(add_anomalies, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.pattern_injection", &m)?;
    Ok(())
}
