use crate::generators::multivariate as gen;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (length, n_variables, copula_type, df, correlation_matrix, marginal_distribution, marginal_param1, marginal_param2, seed))]
pub fn copula(
    py: Python<'_>,
    length: i32,
    n_variables: i32,
    copula_type: i32,
    df: f64,
    correlation_matrix: PyReadonlyArray1<'_, f64>,
    marginal_distribution: i32,
    marginal_param1: f64,
    marginal_param2: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if n_variables <= 0 {
        return Err(PyValueError::new_err("n_variables must be > 0"));
    }
    let cm = correlation_matrix.as_slice()?.to_vec();
    let n = length as usize;
    let mut s = vec![0.0; n];
    let nv = n_variables as usize;
    py.detach(|| {
        gen::copula(
            &mut s,
            nv,
            copula_type,
            df,
            &cm,
            marginal_distribution,
            marginal_param1,
            marginal_param2,
            seed,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, n_variables, order, coef_matrices, intercept, innovation_cov, seed, innov_dist, innov_param))]
pub fn var_process(
    py: Python<'_>,
    length: i32,
    n_variables: i32,
    order: i32,
    coef_matrices: PyReadonlyArray1<'_, f64>,
    intercept: PyReadonlyArray1<'_, f64>,
    innovation_cov: PyReadonlyArray1<'_, f64>,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if n_variables <= 0 {
        return Err(PyValueError::new_err("n_variables must be > 0"));
    }
    if order <= 0 {
        return Err(PyValueError::new_err("order must be > 0"));
    }
    let cm = coef_matrices.as_slice()?.to_vec();
    let ic = intercept.as_slice()?.to_vec();
    let iv = innovation_cov.as_slice()?.to_vec();
    let n = length as usize;
    let mut s = vec![0.0; n];
    let nv = n_variables as usize;
    let ord = order as usize;
    py.detach(|| {
        gen::var_process(
            &mut s,
            nv,
            ord,
            &cm,
            &ic,
            &iv,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, state_dim, obs_dim, f_mat, h_mat, q_mat, r_mat, initial_state, initial_state_cov, seed, innov_dist, innov_param))]
#[allow(clippy::too_many_arguments)]
pub fn state_space(
    py: Python<'_>,
    length: i32,
    state_dim: i32,
    obs_dim: i32,
    f_mat: PyReadonlyArray1<'_, f64>,
    h_mat: PyReadonlyArray1<'_, f64>,
    q_mat: PyReadonlyArray1<'_, f64>,
    r_mat: PyReadonlyArray1<'_, f64>,
    initial_state: PyReadonlyArray1<'_, f64>,
    initial_state_cov: PyReadonlyArray1<'_, f64>,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if state_dim <= 0 {
        return Err(PyValueError::new_err("state_dim must be > 0"));
    }
    if obs_dim <= 0 {
        return Err(PyValueError::new_err("obs_dim must be > 0"));
    }
    let f = f_mat.as_slice()?.to_vec();
    let h = h_mat.as_slice()?.to_vec();
    let q = q_mat.as_slice()?.to_vec();
    let r = r_mat.as_slice()?.to_vec();
    let is = initial_state.as_slice()?.to_vec();
    let isc = initial_state_cov.as_slice()?.to_vec();
    let n = length as usize;
    let mut s = vec![0.0; n];
    let sd = state_dim as usize;
    let od = obs_dim as usize;
    py.detach(|| {
        gen::state_space(
            &mut s,
            sd,
            od,
            &f,
            &h,
            &q,
            &r,
            &is,
            &isc,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, hurst, sigma, initial_value, cumulative, method, seed))]
pub fn fbm(
    py: Python<'_>,
    length: i32,
    hurst: f64,
    sigma: f64,
    initial_value: f64,
    cumulative: bool,
    method: i32,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::fbm(
            &mut s,
            hurst,
            sigma,
            initial_value,
            cumulative,
            method,
            seed,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, kernel_id, length_scale, amplitude, period, mean, noise_variance, seed))]
pub fn gaussian_process(
    py: Python<'_>,
    length: i32,
    kernel_id: i32,
    length_scale: f64,
    amplitude: f64,
    period: f64,
    mean: f64,
    noise_variance: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::gaussian_process(
            &mut s,
            kernel_id,
            length_scale,
            amplitude,
            period,
            mean,
            noise_variance,
            seed,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "multivariate")?;
    m.add_function(wrap_pyfunction!(copula, &m)?)?;
    m.add_function(wrap_pyfunction!(var_process, &m)?)?;
    m.add_function(wrap_pyfunction!(state_space, &m)?)?;
    m.add_function(wrap_pyfunction!(fbm, &m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_process, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.multivariate", &m)?;
    Ok(())
}
