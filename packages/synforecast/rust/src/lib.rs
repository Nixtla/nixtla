#![deny(unsafe_code)]
#![deny(clippy::correctness)]
#![warn(clippy::suspicious)]
#![warn(clippy::perf)]
#![warn(clippy::style)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

pub mod batch;
pub mod bindings;
pub mod distributions;
pub mod fft;
pub mod generators;
pub mod linalg;
pub mod pattern_injection;
pub mod rng;

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "SynForecast Rust accelerated generators")?;
    bindings::pattern_injection::register(m)?;
    bindings::distributions::register(m)?;
    bindings::statistical::register(m)?;
    bindings::stochastic::register(m)?;
    bindings::volatility::register(m)?;
    bindings::multivariate::register(m)?;
    bindings::domain::register(m)?;
    bindings::batch::register(m)?;
    Ok(())
}
