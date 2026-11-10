use crate::generators::stochastic as gen;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (length, theta, mu, sigma, initial_value, dt, seed, innov_dist, innov_param))]
pub fn ornstein_uhlenbeck(
    py: Python<'_>,
    length: i32,
    theta: f64,
    mu: f64,
    sigma: f64,
    initial_value: f64,
    dt: f64,
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
        gen::ornstein_uhlenbeck(
            &mut s,
            theta,
            mu,
            sigma,
            initial_value,
            dt,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, mu, sigma, initial_value, dt, seed, innov_dist, innov_param))]
pub fn geometric_brownian_motion(
    py: Python<'_>,
    length: i32,
    mu: f64,
    sigma: f64,
    initial_value: f64,
    dt: f64,
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
        gen::geometric_brownian_motion(
            &mut s,
            mu,
            sigma,
            initial_value,
            dt,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, mu, sigma, lambda_jump, jump_mean, jump_std, initial_value, dt, seed, innov_dist, innov_param))]
pub fn jump_diffusion(
    py: Python<'_>,
    length: i32,
    mu: f64,
    sigma: f64,
    lambda_jump: f64,
    jump_mean: f64,
    jump_std: f64,
    initial_value: f64,
    dt: f64,
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
        gen::jump_diffusion(
            &mut s,
            mu,
            sigma,
            lambda_jump,
            jump_mean,
            jump_std,
            initial_value,
            dt,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, lambda_rate, cumulative, seed))]
pub fn poisson_process(
    py: Python<'_>,
    length: i32,
    lambda_rate: f64,
    cumulative: bool,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| gen::poisson_process(&mut s, lambda_rate, cumulative, seed));
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, base_level, trend, cycle_period_mean, cycle_period_std, cycle_amplitude_mean, cycle_amplitude_std, num_cycles, noise_std, seed, innov_dist, innov_param))]
pub fn cyclic(
    py: Python<'_>,
    length: i32,
    base_level: f64,
    trend: f64,
    cycle_period_mean: f64,
    cycle_period_std: f64,
    cycle_amplitude_mean: f64,
    cycle_amplitude_std: f64,
    num_cycles: i32,
    noise_std: f64,
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
        gen::cyclic(
            &mut s,
            base_level,
            trend,
            cycle_period_mean,
            cycle_period_std,
            cycle_amplitude_mean,
            cycle_amplitude_std,
            num_cycles,
            noise_std,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, p, q, omega, alpha_arr, beta_arr, mu, initial_variance, seed, innov_dist, innov_param))]
pub fn garch(
    py: Python<'_>,
    length: i32,
    p: i32,
    q: i32,
    omega: f64,
    alpha_arr: PyReadonlyArray1<'_, f64>,
    beta_arr: PyReadonlyArray1<'_, f64>,
    mu: f64,
    initial_variance: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let a = alpha_arr.as_slice()?.to_vec();
    let b = beta_arr.as_slice()?.to_vec();
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::garch(
            &mut s,
            p,
            q,
            omega,
            &a,
            &b,
            mu,
            initial_variance,
            seed,
            innov_dist,
            innov_param,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, baseline_intensity, excitation_amplitude, decay_rate, kernel_type, power_law_exponent, output_type, max_events, seed))]
pub fn hawkes_process(
    py: Python<'_>,
    length: i32,
    baseline_intensity: f64,
    excitation_amplitude: f64,
    decay_rate: f64,
    kernel_type: i32,
    power_law_exponent: f64,
    output_type: i32,
    max_events: i32,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::hawkes_process(
            &mut s,
            baseline_intensity,
            excitation_amplitude,
            decay_rate,
            kernel_type,
            power_law_exponent,
            output_type,
            max_events,
            seed,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, system_id, sigma, rho, beta, dt, logistic_r, mg_beta, mg_gamma, mg_n, mg_tau, observation_noise, initial_perturbation, seed))]
#[allow(clippy::too_many_arguments)]
pub fn chaotic_system(
    py: Python<'_>,
    length: i32,
    system_id: i32,
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    logistic_r: f64,
    mg_beta: f64,
    mg_gamma: f64,
    mg_n: f64,
    mg_tau: i32,
    observation_noise: f64,
    initial_perturbation: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::chaotic_system(
            &mut s,
            system_id,
            sigma,
            rho,
            beta,
            dt,
            logistic_r,
            mg_beta,
            mg_gamma,
            mg_n,
            mg_tau,
            observation_noise,
            initial_perturbation,
            seed,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, model_id, phi, omega, kappa, sigma_param, initial_value, lower, upper, seed))]
#[allow(clippy::too_many_arguments)]
pub fn bounded_process(
    py: Python<'_>,
    length: i32,
    model_id: i32,
    phi: f64,
    omega: f64,
    kappa: f64,
    sigma_param: f64,
    initial_value: f64,
    lower: f64,
    upper: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if lower >= upper {
        return Err(PyValueError::new_err("lower must be < upper"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::bounded_process(
            &mut s,
            model_id,
            phi,
            omega,
            kappa,
            sigma_param,
            initial_value,
            lower,
            upper,
            seed,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

#[pyfunction]
#[pyo3(signature = (length, alpha, beta_skew, scale, location, cumulative, initial_value, seed))]
pub fn levy_process(
    py: Python<'_>,
    length: i32,
    alpha: f64,
    beta_skew: f64,
    scale: f64,
    location: f64,
    cumulative: bool,
    initial_value: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let mut s = vec![0.0; n];
    py.detach(|| {
        gen::levy_process(
            &mut s,
            alpha,
            beta_skew,
            scale,
            location,
            cumulative,
            initial_value,
            seed,
        )
    });
    Ok(PyArray1::from_vec(py, s).into())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "stochastic")?;
    m.add_function(wrap_pyfunction!(ornstein_uhlenbeck, &m)?)?;
    m.add_function(wrap_pyfunction!(geometric_brownian_motion, &m)?)?;
    m.add_function(wrap_pyfunction!(jump_diffusion, &m)?)?;
    m.add_function(wrap_pyfunction!(poisson_process, &m)?)?;
    m.add_function(wrap_pyfunction!(cyclic, &m)?)?;
    m.add_function(wrap_pyfunction!(garch, &m)?)?;
    m.add_function(wrap_pyfunction!(hawkes_process, &m)?)?;
    m.add_function(wrap_pyfunction!(chaotic_system, &m)?)?;
    m.add_function(wrap_pyfunction!(bounded_process, &m)?)?;
    m.add_function(wrap_pyfunction!(levy_process, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.stochastic", &m)?;
    Ok(())
}
