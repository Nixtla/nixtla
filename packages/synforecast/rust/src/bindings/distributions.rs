use crate::distributions as dist;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
pub fn norm_cdf(py: Python<'_>, x: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_slice()?;
    let n = input.len();
    let mut out_slice = vec![0.0; n];
    py.detach(|| {
        for i in 0..n {
            out_slice[i] = dist::norm_cdf(input[i]);
        }
    });
    Ok(PyArray1::from_vec(py, out_slice).into())
}

#[pyfunction]
pub fn norm_ppf(py: Python<'_>, p: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = p.as_slice()?;
    let n = input.len();
    let mut out_slice = vec![0.0; n];
    py.detach(|| {
        for i in 0..n {
            out_slice[i] = dist::norm_ppf(input[i]);
        }
    });
    Ok(PyArray1::from_vec(py, out_slice).into())
}

#[pyfunction]
pub fn t_cdf(py: Python<'_>, x: PyReadonlyArray1<'_, f64>, df: f64) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_slice()?;
    let n = input.len();
    let mut out_slice = vec![0.0; n];
    py.detach(|| {
        for i in 0..n {
            out_slice[i] = dist::t_cdf(input[i], df);
        }
    });
    Ok(PyArray1::from_vec(py, out_slice).into())
}

#[pyfunction]
pub fn gamma_ppf(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    a: f64,
    scale: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let input = u.as_slice()?;
    let n = input.len();
    let mut out_slice = vec![0.0; n];
    py.detach(|| {
        for i in 0..n {
            out_slice[i] = dist::gamma_ppf(input[i], a, scale);
        }
    });
    Ok(PyArray1::from_vec(py, out_slice).into())
}

#[pyfunction]
pub fn expon_ppf(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    scale: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let input = u.as_slice()?;
    let n = input.len();
    let mut out_slice = vec![0.0; n];
    py.detach(|| {
        for i in 0..n {
            out_slice[i] = dist::expon_ppf(input[i], scale);
        }
    });
    Ok(PyArray1::from_vec(py, out_slice).into())
}

#[pyfunction]
pub fn lognorm_ppf(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    s: f64,
    scale: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let input = u.as_slice()?;
    let n = input.len();
    let mut out_slice = vec![0.0; n];
    py.detach(|| {
        for i in 0..n {
            out_slice[i] = dist::lognorm_ppf(input[i], s, scale);
        }
    });
    Ok(PyArray1::from_vec(py, out_slice).into())
}

#[pyfunction]
pub fn uniform_ppf(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    loc: f64,
    scale: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let input = u.as_slice()?;
    let n = input.len();
    let mut out_slice = vec![0.0; n];
    py.detach(|| {
        for i in 0..n {
            out_slice[i] = dist::uniform_ppf(input[i], loc, scale);
        }
    });
    Ok(PyArray1::from_vec(py, out_slice).into())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "distributions")?;
    m.add_function(wrap_pyfunction!(norm_cdf, &m)?)?;
    m.add_function(wrap_pyfunction!(norm_ppf, &m)?)?;
    m.add_function(wrap_pyfunction!(t_cdf, &m)?)?;
    m.add_function(wrap_pyfunction!(gamma_ppf, &m)?)?;
    m.add_function(wrap_pyfunction!(expon_ppf, &m)?)?;
    m.add_function(wrap_pyfunction!(lognorm_ppf, &m)?)?;
    m.add_function(wrap_pyfunction!(uniform_ppf, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.distributions", &m)?;
    Ok(())
}
