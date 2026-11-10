use crate::generators::statistical as gen;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (length, drift, volatility, start_value, seed, innov_dist, innov_param))]
pub fn random_walk(
    py: Python<'_>,
    length: i32,
    drift: f64,
    volatility: f64,
    start_value: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::random_walk(
            &mut s,
            drift,
            volatility,
            start_value,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, seasonality_period, seasonality_amplitude, trend, noise_level, base_level, seed, innov_dist=0, innov_param=0.0))]
pub fn seasonal(
    py: Python<'_>,
    length: i32,
    seasonality_period: i32,
    seasonality_amplitude: f64,
    trend: f64,
    noise_level: f64,
    base_level: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if seasonality_period <= 0 {
        return Err(PyValueError::new_err("seasonality_period must be > 0"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::seasonal(
            &mut s,
            seasonality_period,
            seasonality_amplitude,
            trend,
            noise_level,
            base_level,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, full_ar_poly, full_ma_poly, d, seasonal_d, seasonal_period, mean, drift_val, noise_std, burn_in, seed, innov_dist, innov_param))]
pub fn sarima(
    py: Python<'_>,
    length: i32,
    full_ar_poly: PyReadonlyArray1<'_, f64>,
    full_ma_poly: PyReadonlyArray1<'_, f64>,
    d: i32,
    seasonal_d: i32,
    seasonal_period: i32,
    mean: f64,
    drift_val: f64,
    noise_std: f64,
    burn_in: i32,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let ar = full_ar_poly.as_slice()?.to_vec();
    let ma = full_ma_poly.as_slice()?.to_vec();
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::sarima(
            &mut s,
            &ar,
            &ma,
            d,
            seasonal_d,
            seasonal_period,
            mean,
            drift_val,
            noise_std,
            burn_in,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, error_type, trend_type, seasonal_type, seasonal_period, level, trend_init, seasonal_init, alpha, beta_param, gamma, phi, damped, noise_std, seed, innov_dist, innov_param))]
#[allow(clippy::too_many_arguments)]
pub fn ets(
    py: Python<'_>,
    length: i32,
    error_type: i32,
    trend_type: i32,
    seasonal_type: i32,
    seasonal_period: i32,
    level: f64,
    trend_init: f64,
    seasonal_init: PyReadonlyArray1<'_, f64>,
    alpha: f64,
    beta_param: f64,
    gamma: f64,
    phi: f64,
    damped: bool,
    noise_std: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if seasonal_period <= 0 {
        return Err(PyValueError::new_err("seasonal_period must be > 0"));
    }
    let s_init = seasonal_init.as_slice()?.to_vec();
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::ets(
            &mut s,
            error_type,
            trend_type,
            seasonal_type,
            seasonal_period,
            level,
            trend_init,
            &s_init,
            alpha,
            beta_param,
            gamma,
            phi,
            damped,
            noise_std,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, p, alpha_arr, innov_type, innov_mean, innov_dispersion, seed))]
pub fn inar(
    py: Python<'_>,
    length: i32,
    p: i32,
    alpha_arr: PyReadonlyArray1<'_, f64>,
    innov_type: i32,
    innov_mean: f64,
    innov_dispersion: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if p <= 0 {
        return Err(PyValueError::new_err("p must be > 0"));
    }
    let a = alpha_arr.as_slice()?.to_vec();
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::inar(
            &mut s,
            p,
            &a,
            innov_type,
            innov_mean,
            innov_dispersion,
            seed,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "statistical")?;
    m.add_function(wrap_pyfunction!(random_walk, &m)?)?;
    m.add_function(wrap_pyfunction!(seasonal, &m)?)?;
    m.add_function(wrap_pyfunction!(sarima, &m)?)?;
    m.add_function(wrap_pyfunction!(ets, &m)?)?;
    m.add_function(wrap_pyfunction!(inar, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.statistical", &m)?;
    Ok(())
}
