use crate::generators::volatility as gen;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (length, model_type, initial_price, initial_vol, drift, mean_vol, vol_mean_reversion, vol_of_vol, correlation, beta_param, dt, output_type, seed, innov_dist, innov_param))]
#[allow(clippy::too_many_arguments)]
pub fn stochastic_volatility(
    py: Python<'_>,
    length: i32,
    model_type: i32,
    initial_price: f64,
    initial_vol: f64,
    drift: f64,
    mean_vol: f64,
    vol_mean_reversion: f64,
    vol_of_vol: f64,
    correlation: f64,
    beta_param: f64,
    dt: f64,
    output_type: i32,
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
        gen::stochastic_volatility(
            &mut s,
            model_type,
            initial_price,
            initial_vol,
            drift,
            mean_vol,
            vol_mean_reversion,
            vol_of_vol,
            correlation,
            beta_param,
            dt,
            output_type,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

/// `initial_regime < 0` draws the starting regime from `stationary_probs`
/// (which must then have at least `n_regimes` entries).
#[pyfunction]
#[pyo3(signature = (length, n_regimes, regime_means, regime_variances, regime_ar_coeffs, transition_matrix, stationary_probs, initial_regime, seed, innov_dist, innov_param))]
#[allow(clippy::too_many_arguments)]
pub fn regime_switching(
    py: Python<'_>,
    length: i32,
    n_regimes: i32,
    regime_means: PyReadonlyArray1<'_, f64>,
    regime_variances: PyReadonlyArray1<'_, f64>,
    regime_ar_coeffs: PyReadonlyArray1<'_, f64>,
    transition_matrix: PyReadonlyArray1<'_, f64>,
    stationary_probs: PyReadonlyArray1<'_, f64>,
    initial_regime: i32,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if n_regimes <= 0 {
        return Err(PyValueError::new_err("n_regimes must be > 0"));
    }
    let rm = regime_means.as_slice()?.to_vec();
    let rv = regime_variances.as_slice()?.to_vec();
    let ra = regime_ar_coeffs.as_slice()?.to_vec();
    let tm = transition_matrix.as_slice()?.to_vec();
    let sp = stationary_probs.as_slice()?.to_vec();
    if initial_regime < 0 && sp.len() < n_regimes as usize {
        return Err(PyValueError::new_err(
            "stationary_probs must have n_regimes entries when initial_regime < 0",
        ));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::regime_switching(
            &mut s,
            n_regimes,
            &rm,
            &rv,
            &ra,
            &tm,
            &sp,
            initial_regime,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "volatility")?;
    m.add_function(wrap_pyfunction!(stochastic_volatility, &m)?)?;
    m.add_function(wrap_pyfunction!(regime_switching, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.volatility", &m)?;
    Ok(())
}
