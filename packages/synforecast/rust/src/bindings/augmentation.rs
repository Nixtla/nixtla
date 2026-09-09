use crate::augmentation as algorithms;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Mutex;

type PyDecomposition = (Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>);

fn parse_band(band: Option<i64>) -> PyResult<Option<usize>> {
    match band {
        Some(value) if value < 0 => Err(PyValueError::new_err(
            "band must be non-negative when provided",
        )),
        Some(value) => Ok(Some(value as usize)),
        None => Ok(None),
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, band=None))]
fn dtw_alignment(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    band: Option<i64>,
) -> PyResult<(f64, Vec<(usize, usize)>)> {
    let a = a.as_slice()?.to_vec();
    let b = b.as_slice()?.to_vec();
    let band = parse_band(band)?;
    py.detach(|| algorithms::dtw_alignment(&a, &b, band))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (a, b, band=None))]
fn dtw_distance(
    py: Python<'_>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    band: Option<i64>,
) -> PyResult<f64> {
    let a = a.as_slice()?.to_vec();
    let b = b.as_slice()?.to_vec();
    let band = parse_band(band)?;
    py.detach(|| algorithms::dtw_distance(&a, &b, band))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
fn nearest_dtw_neighbors(
    py: Python<'_>,
    series: Vec<PyReadonlyArray1<'_, f64>>,
    window_fraction: f64,
    n_neighbors: usize,
) -> PyResult<Vec<Vec<(usize, f64)>>> {
    let series = series
        .iter()
        .map(|values| values.as_slice().map(ToOwned::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    py.detach(|| algorithms::nearest_dtw_neighbors(&series, window_fraction, n_neighbors))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (reference, neighbors, weights, n_iterations, band=None))]
fn dba_barycenter(
    py: Python<'_>,
    reference: PyReadonlyArray1<'_, f64>,
    neighbors: Vec<PyReadonlyArray1<'_, f64>>,
    weights: PyReadonlyArray1<'_, f64>,
    n_iterations: usize,
    band: Option<i64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let reference = reference.as_slice()?.to_vec();
    let neighbors = neighbors
        .iter()
        .map(|values| values.as_slice().map(ToOwned::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    let weights = weights.as_slice()?.to_vec();
    let band = parse_band(band)?;
    let mut result = run_dba(py, &reference, &neighbors, &[weights], n_iterations, band)?;
    let result = result.remove(0);
    Ok(PyArray1::from_vec(py, result).into())
}

fn run_dba(
    py: Python<'_>,
    reference: &[f64],
    neighbors: &[Vec<f64>],
    weights: &[Vec<f64>],
    n_iterations: usize,
    band: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let interruption = Mutex::new(None);
    let result = py.detach(|| {
        algorithms::dba_barycenters(reference, neighbors, weights, n_iterations, band, || {
            Python::attach(|py| py.check_signals()).map_err(|error| {
                *interruption.lock().unwrap() = Some(error);
                "DBA interrupted".to_string()
            })
        })
    });
    if let Some(error) = interruption.into_inner().unwrap() {
        return Err(error);
    }
    result.map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (reference, neighbors, weights, n_iterations, band=None))]
fn dba_barycenters(
    py: Python<'_>,
    reference: PyReadonlyArray1<'_, f64>,
    neighbors: Vec<PyReadonlyArray1<'_, f64>>,
    weights: Vec<Vec<f64>>,
    n_iterations: usize,
    band: Option<i64>,
) -> PyResult<Vec<Py<PyArray1<f64>>>> {
    let reference = reference.as_slice()?.to_vec();
    let neighbors = neighbors
        .iter()
        .map(|v| v.as_slice().map(ToOwned::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    let result = run_dba(
        py,
        &reference,
        &neighbors,
        &weights,
        n_iterations,
        parse_band(band)?,
    )?;
    Ok(result
        .into_iter()
        .map(|v| PyArray1::from_vec(py, v).into())
        .collect())
}

/// Score independent native MAR draws in one pool; failed candidates return None.
#[pyfunction]
#[pyo3(signature = (scalars, arrays, lengths, seeds, period=None, n_workers=0))]
fn mar_features_batch(
    py: Python<'_>,
    scalars: Vec<Vec<f64>>,
    arrays: Vec<Vec<Vec<f64>>>,
    lengths: Vec<usize>,
    seeds: Vec<Vec<u64>>,
    period: Option<usize>,
    n_workers: usize,
) -> PyResult<Vec<Option<Vec<[f64; 4]>>>> {
    if scalars.len() != arrays.len()
        || scalars.len() != seeds.len()
        || lengths.is_empty()
        || lengths.iter().any(|n| !(3..=1_000_000).contains(n))
        || seeds.iter().any(|s| s.len() != lengths.len())
    {
        return Err(PyValueError::new_err(
            "invalid MAR feature batch dimensions or lengths",
        ));
    }
    let work = || {
        (0..scalars.len())
            .into_par_iter()
            .map(|i| {
                lengths
                    .iter()
                    .zip(&seeds[i])
                    .map(|(&length, &seed)| {
                        let mut values = vec![0.0; length];
                        crate::generators::mar::mar(&mut values, &scalars[i], &arrays[i], seed)
                            .ok()?;
                        let (entropy, trend, seasonal, acf) =
                            algorithms::compute_features(&values, period).ok()?;
                        Some([entropy, trend, seasonal, acf])
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .collect()
    };
    py.detach(|| {
        if n_workers == 0 {
            Ok(work())
        } else {
            Ok(crate::batch::get_or_create_pool(n_workers)
                .map_err(PyValueError::new_err)?
                .install(work))
        }
    })
}

#[pyfunction]
#[pyo3(signature = (values, period=None))]
fn classical_decompose(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    period: Option<usize>,
) -> PyResult<PyDecomposition> {
    let values = values.as_slice()?.to_vec();
    let (trend, seasonal, remainder) = py
        .detach(|| algorithms::classical_decompose(&values, period))
        .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, trend).into(),
        PyArray1::from_vec(py, seasonal).into(),
        PyArray1::from_vec(py, remainder).into(),
    ))
}

#[pyfunction]
#[pyo3(signature = (values, period=None))]
fn compute_features(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    period: Option<usize>,
) -> PyResult<(f64, f64, f64, f64)> {
    let values = values.as_slice()?.to_vec();
    py.detach(|| algorithms::compute_features(&values, period))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (values, block_size, seed, period=None))]
fn moving_block_bootstrap(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    block_size: usize,
    seed: u64,
    period: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let values = values.as_slice()?.to_vec();
    let generated = py
        .detach(|| algorithms::moving_block_bootstrap(&values, period, block_size, seed))
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, generated).into())
}

#[pyfunction]
#[pyo3(signature = (values, block_size, seeds, period=None))]
fn moving_block_bootstrap_many(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    block_size: usize,
    seeds: Vec<u64>,
    period: Option<usize>,
) -> PyResult<Vec<Py<PyArray1<f64>>>> {
    let values = values.as_slice()?.to_vec();
    let generated = py
        .detach(|| algorithms::moving_block_bootstrap_many(&values, period, block_size, &seeds))
        .map_err(PyValueError::new_err)?;
    Ok(generated
        .into_iter()
        .map(|values| PyArray1::from_vec(py, values).into())
        .collect())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "augmentation")?;
    module.add_function(wrap_pyfunction!(dtw_alignment, &module)?)?;
    module.add_function(wrap_pyfunction!(dtw_distance, &module)?)?;
    module.add_function(wrap_pyfunction!(nearest_dtw_neighbors, &module)?)?;
    module.add_function(wrap_pyfunction!(dba_barycenter, &module)?)?;
    module.add_function(wrap_pyfunction!(dba_barycenters, &module)?)?;
    module.add_function(wrap_pyfunction!(mar_features_batch, &module)?)?;
    module.add_function(wrap_pyfunction!(classical_decompose, &module)?)?;
    module.add_function(wrap_pyfunction!(compute_features, &module)?)?;
    module.add_function(wrap_pyfunction!(moving_block_bootstrap, &module)?)?;
    module.add_function(wrap_pyfunction!(moving_block_bootstrap_many, &module)?)?;
    parent.add_submodule(&module)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.augmentation", &module)?;
    Ok(())
}
