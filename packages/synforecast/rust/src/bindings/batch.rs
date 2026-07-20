use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::batch;

/// Extract a scalar value from a Python dict, returning `default` if the key
/// is absent or the value cannot be extracted as `T`.
#[inline]
fn dict_get<'py, T: FromPyObjectOwned<'py> + Copy>(
    d: &Bound<'py, PyDict>,
    key: &str,
    default: T,
) -> PyResult<T> {
    Ok(d.get_item(key)?
        .map_or(default, |v| v.extract::<T>().unwrap_or(default)))
}

/// Extract a `String` from a Python dict, returning `default` if absent or
/// wrong type.
#[inline]
fn dict_get_string(d: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    Ok(d.get_item(key)?.map_or_else(
        || default.to_string(),
        |v| {
            v.extract::<String>()
                .unwrap_or_else(|_| default.to_string())
        },
    ))
}

/// Extract a `Vec<String>` from a Python dict, returning empty vec if absent.
#[inline]
fn dict_get_string_vec(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    Ok(d.get_item(key)?
        .map_or_else(Vec::new, |v| v.extract::<Vec<String>>().unwrap_or_default()))
}

/// Extract a `Vec<f64>` from a numpy array in a dict, returning empty vec if absent.
#[inline]
fn extract_f64_vec(d: &Bound<'_, PyDict>, key: &str) -> Vec<f64> {
    d.get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<Vec<f64>>().ok())
        .unwrap_or_default()
}

/// Parse a Python dict into a [`PatternInjectionConfig`](batch::PatternInjectionConfig).
///
/// Uses lenient extraction: missing keys or wrong types silently fall back to
/// defaults rather than raising errors, since the Python layer validates inputs.
fn parse_pi_config(d: &Bound<'_, PyDict>) -> PyResult<batch::PatternInjectionConfig> {
    Ok(batch::PatternInjectionConfig {
        changepoints: dict_get(d, "changepoints", false)?,
        num_changepoints: dict_get(d, "num_changepoints", 0i32)?,
        changepoint_type: dict_get_string(d, "changepoint_type", "level")?,
        changepoint_locations: extract_f64_vec(d, "changepoint_locations"),
        changepoint_level_changes: extract_f64_vec(d, "changepoint_level_changes"),
        changepoint_trend_changes: extract_f64_vec(d, "changepoint_trend_changes"),
        changepoint_variance_changes: extract_f64_vec(d, "changepoint_variance_changes"),
        anomalies: dict_get(d, "anomalies", false)?,
        anomaly_types: dict_get_string_vec(d, "anomaly_types")?,
        anomaly_fraction: dict_get(d, "anomaly_fraction", 0.0f64)?,
        spike_magnitude: dict_get(d, "spike_magnitude", 0.0f64)?,
        dip_magnitude: dict_get(d, "dip_magnitude", 0.0f64)?,
        level_shift_magnitude: dict_get(d, "level_shift_magnitude", 0.0f64)?,
        level_shift_duration: dict_get(d, "level_shift_duration", 0i32)?,
        missing_data: dict_get(d, "missing_data", false)?,
        missing_pattern: dict_get_string(d, "missing_pattern", "random")?,
        missing_rate: dict_get(d, "missing_rate", 0.0f64)?,
        missing_block_size: dict_get(d, "missing_block_size", 1i32)?,
        missing_seasonal_period: dict_get(d, "missing_seasonal_period", 1i32)?,
    })
}

/// Generate multiple series of a single generator type in parallel.
///
/// Returns a list of dicts, each with:
///   "values": numpy array of f64
///   "cp_indices": numpy array of i64
///   "anom_indices": numpy array of i64
///   "miss_indices": numpy array of i64
#[pyfunction]
#[pyo3(signature = (gen_type, scalar_params, array_params, lengths, gen_seeds, pi_seeds, pi_config_dict, n_workers=0))]
pub fn generate_batch<'py>(
    py: Python<'py>,
    gen_type: i32,
    scalar_params: PyReadonlyArray1<'py, f64>,
    array_params: &Bound<'py, PyList>,
    lengths: PyReadonlyArray1<'py, i32>,
    gen_seeds: PyReadonlyArray1<'py, u64>,
    pi_seeds: PyReadonlyArray1<'py, u64>,
    pi_config_dict: &Bound<'py, PyDict>,
    n_workers: usize,
) -> PyResult<Py<PyAny>> {
    let sp = scalar_params.as_slice()?.to_vec();
    let ap: Vec<Vec<f64>> = array_params
        .iter()
        .map(|item| {
            let arr: PyReadonlyArray1<'_, f64> = item.extract()?;
            Ok(arr.as_slice()?.to_vec())
        })
        .collect::<PyResult<Vec<Vec<f64>>>>()?;
    let lens = lengths.as_slice()?.to_vec();
    let gs = gen_seeds.as_slice()?.to_vec();
    let ps = pi_seeds.as_slice()?.to_vec();
    let pi_config = parse_pi_config(pi_config_dict)?;

    if lens.is_empty() {
        return Ok(PyList::empty(py).into());
    }

    // Run batch generation with GIL released
    let results = py
        .detach(|| {
            batch::generate_batch(gen_type, &sp, &ap, &lens, &gs, &ps, &pi_config, n_workers)
        })
        .map_err(PyValueError::new_err)?;

    results_to_py(py, results)
}

/// Generate series from multiple generators in parallel.
///
/// Takes a list of spec dicts, each with:
///   "gen_type": int, "scalar_params": numpy array,
///   "array_params": list of numpy arrays,
///   "lengths": numpy array, "gen_seeds": numpy array,
///   "pi_seeds": numpy array, "pi_config": dict
///
/// Returns list[list[dict]] — outer list per generator, inner per series.
#[pyfunction]
#[pyo3(signature = (specs, n_workers=0))]
pub fn generate_multi_batch<'py>(
    py: Python<'py>,
    specs: &Bound<'py, PyList>,
    n_workers: usize,
) -> PyResult<Py<PyAny>> {
    let mut rust_specs: Vec<batch::GeneratorSpec> = Vec::with_capacity(specs.len());

    for spec_item in specs.iter() {
        let d: &Bound<'_, PyDict> = spec_item.cast()?;

        let gen_type: i32 = d
            .get_item("gen_type")?
            .ok_or_else(|| PyValueError::new_err("missing gen_type"))?
            .extract()?;
        let sp_arr: PyReadonlyArray1<'_, f64> = d
            .get_item("scalar_params")?
            .ok_or_else(|| PyValueError::new_err("missing scalar_params"))?
            .extract()?;
        let ap_obj = d
            .get_item("array_params")?
            .ok_or_else(|| PyValueError::new_err("missing array_params"))?;
        let ap_list: &Bound<'_, PyList> = ap_obj.cast()?;
        let lens_arr: PyReadonlyArray1<'_, i32> = d
            .get_item("lengths")?
            .ok_or_else(|| PyValueError::new_err("missing lengths"))?
            .extract()?;
        let gs_arr: PyReadonlyArray1<'_, u64> = d
            .get_item("gen_seeds")?
            .ok_or_else(|| PyValueError::new_err("missing gen_seeds"))?
            .extract()?;
        let ps_arr: PyReadonlyArray1<'_, u64> = d
            .get_item("pi_seeds")?
            .ok_or_else(|| PyValueError::new_err("missing pi_seeds"))?
            .extract()?;
        let pi_obj = d
            .get_item("pi_config")?
            .ok_or_else(|| PyValueError::new_err("missing pi_config"))?;
        let pi_dict: &Bound<'_, PyDict> = pi_obj.cast()?;

        let ap: Vec<Vec<f64>> = ap_list
            .iter()
            .map(|item| {
                let arr: PyReadonlyArray1<'_, f64> = item.extract()?;
                Ok(arr.as_slice()?.to_vec())
            })
            .collect::<PyResult<Vec<Vec<f64>>>>()?;

        rust_specs.push(batch::GeneratorSpec {
            gen_type,
            scalar_params: sp_arr.as_slice()?.to_vec(),
            array_params: ap,
            lengths: lens_arr.as_slice()?.to_vec(),
            gen_seeds: gs_arr.as_slice()?.to_vec(),
            pi_seeds: ps_arr.as_slice()?.to_vec(),
            pi_config: parse_pi_config(pi_dict)?,
        });
    }

    // Run multi-batch generation with GIL released
    let grouped_results = py
        .detach(|| batch::generate_multi_batch(&rust_specs, n_workers))
        .map_err(PyValueError::new_err)?;

    // Convert to list[list[dict]]
    let outer = PyList::empty(py);
    for gen_results in grouped_results {
        let inner = results_to_py_list(py, gen_results)?;
        outer.append(inner)?;
    }
    Ok(outer.into())
}

/// Convert `Vec<SeriesResult>` to a Python list of dicts (as `PyObject`).
fn results_to_py(py: Python<'_>, results: Vec<batch::SeriesResult>) -> PyResult<Py<PyAny>> {
    let list = results_to_py_list(py, results)?;
    Ok(list.into())
}

/// Convert `Vec<SeriesResult>` to a Python list of dicts.
///
/// Each dict contains "values", "cp_indices", "anom_indices", "miss_indices"
/// as numpy arrays.
#[inline]
fn results_to_py_list<'py>(
    py: Python<'py>,
    results: Vec<batch::SeriesResult>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("values", PyArray1::from_vec(py, r.values))?;
        d.set_item("cp_indices", PyArray1::from_vec(py, r.cp_indices))?;
        d.set_item("anom_indices", PyArray1::from_vec(py, r.anom_indices))?;
        d.set_item("miss_indices", PyArray1::from_vec(py, r.miss_indices))?;
        list.append(d)?;
    }
    Ok(list)
}

/// Register the `batch` submodule under `synforecast._lib`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "batch")?;
    m.add_function(wrap_pyfunction!(generate_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(generate_multi_batch, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.batch", &m)?;
    Ok(())
}
