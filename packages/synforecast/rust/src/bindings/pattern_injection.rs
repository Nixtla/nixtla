use crate::pattern_injection as pi;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn copy_out(values: &Bound<'_, PyArray1<f64>>) -> PyResult<Vec<f64>> {
    let borrow = values.try_readwrite()?;
    Ok(borrow.as_slice()?.to_vec())
}

fn copy_back(values: &Bound<'_, PyArray1<f64>>, buf: &[f64]) -> PyResult<()> {
    let mut borrow = values.try_readwrite()?;
    let dest = borrow.as_slice_mut()?;
    if dest.len() != buf.len() {
        return Err(PyValueError::new_err(
            "values array changed length during the call",
        ));
    }
    dest.copy_from_slice(buf);
    Ok(())
}

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
    let mut vals = copy_out(&values)?;
    let locs = locations.as_slice()?.to_vec();
    let lc = level_changes.as_slice()?.to_vec();
    let tc = trend_changes.as_slice()?.to_vec();
    let vc = variance_changes.as_slice()?.to_vec();

    let result = py.detach(|| {
        pi::add_changepoints(
            &mut vals,
            seed,
            num_changepoints,
            &locs,
            changepoint_type,
            &lc,
            &tc,
            &vc,
        )
    });
    copy_back(&values, &vals)?;

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
    let mut vals = copy_out(&values)?;

    let result = py.detach(|| {
        pi::add_missingness(
            &mut vals,
            seed,
            pattern,
            missing_rate,
            missing_block_size,
            missing_seasonal_period,
        )
    });
    copy_back(&values, &vals)?;

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
    let mut vals = copy_out(&values)?;

    let result = py.detach(|| {
        pi::add_anomalies(
            &mut vals,
            seed,
            &anomaly_types,
            anomaly_fraction,
            spike_magnitude,
            dip_magnitude,
            level_shift_magnitude,
            level_shift_duration,
        )
    });
    copy_back(&values, &vals)?;

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
