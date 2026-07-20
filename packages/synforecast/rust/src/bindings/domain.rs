use crate::generators::domain as gen;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (length, demand_probability, demand_distribution, demand_mean, demand_std, intermittent_pattern, cluster_size, seasonal_period, seasonal_peak_prob, min_demand, seed))]
#[allow(clippy::too_many_arguments)]
pub fn intermittent_demand(
    py: Python<'_>,
    length: i32,
    demand_probability: f64,
    demand_distribution: i32,
    demand_mean: f64,
    demand_std: f64,
    intermittent_pattern: i32,
    cluster_size: i32,
    seasonal_period: i32,
    seasonal_peak_prob: f64,
    min_demand: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let out = PyArray1::<f64>::zeros(py, n, false);
    // SAFETY: Array was just allocated with no other references to its data.
    let s = unsafe { out.as_slice_mut()? };
    py.detach(|| {
        gen::intermittent_demand(
            s,
            demand_probability,
            demand_distribution,
            demand_mean,
            demand_std,
            intermittent_pattern,
            cluster_size,
            seasonal_period,
            seasonal_peak_prob,
            min_demand,
            seed,
        )
    });
    Ok(out.into())
}

#[pyfunction]
#[pyo3(signature = (length, base_users, growth_rate, weekend_factor, weekly_pattern, event_probability, event_impact_min, event_impact_max, event_decay_rate, noise_std, step_hours, seed))]
#[allow(clippy::too_many_arguments)]
pub fn daily_active_users(
    py: Python<'_>,
    length: i32,
    base_users: f64,
    growth_rate: f64,
    weekend_factor: f64,
    weekly_pattern: bool,
    event_probability: f64,
    event_impact_min: f64,
    event_impact_max: f64,
    event_decay_rate: f64,
    noise_std: f64,
    step_hours: f64,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if step_hours <= 0.0 {
        return Err(PyValueError::new_err("step_hours must be > 0"));
    }
    let n = length as usize;
    let out = PyArray1::<f64>::zeros(py, n, false);
    // SAFETY: Array was just allocated with no other references to its data.
    let out_s = unsafe { out.as_slice_mut()? };
    let result = py.detach(|| {
        gen::daily_active_users(
            out_s,
            base_users,
            growth_rate,
            weekend_factor,
            weekly_pattern,
            event_probability,
            event_impact_min,
            event_impact_max,
            event_decay_rate,
            noise_std,
            step_hours,
            seed,
        )
    });
    let events = PyArray1::from_vec(py, result.events);
    let tuple = pyo3::types::PyTuple::new(py, [out.as_any(), events.as_any()])?;
    Ok(tuple.into())
}

#[pyfunction]
#[pyo3(signature = (length, base_load, load_type, daily_pattern, daily_amplitude, weekly_pattern, weekly_amplitude, yearly_pattern, yearly_amplitude, temperature_sensitive, temperature_sensitivity, base_temperature, morning_peak_hour, evening_peak_hour, peak_amplitude, holiday_effect, holiday_days, extreme_weather_prob, extreme_weather_impact, noise_std, step_hours, seed))]
#[allow(clippy::too_many_arguments)]
pub fn energy_load(
    py: Python<'_>,
    length: i32,
    base_load: f64,
    load_type: i32,
    daily_pattern: bool,
    daily_amplitude: f64,
    weekly_pattern: bool,
    weekly_amplitude: f64,
    yearly_pattern: bool,
    yearly_amplitude: f64,
    temperature_sensitive: bool,
    temperature_sensitivity: f64,
    base_temperature: f64,
    morning_peak_hour: i32,
    evening_peak_hour: i32,
    peak_amplitude: f64,
    holiday_effect: f64,
    holiday_days: PyReadonlyArray1<'_, i32>,
    extreme_weather_prob: f64,
    extreme_weather_impact: f64,
    noise_std: f64,
    step_hours: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    if step_hours <= 0.0 {
        return Err(PyValueError::new_err("step_hours must be > 0"));
    }
    let hd = holiday_days.as_slice()?;
    let n = length as usize;
    let out = PyArray1::<f64>::zeros(py, n, false);
    // SAFETY: Array was just allocated with no other references to its data.
    let s = unsafe { out.as_slice_mut()? };
    py.detach(|| {
        gen::energy_load(
            s,
            base_load,
            load_type,
            daily_pattern,
            daily_amplitude,
            weekly_pattern,
            weekly_amplitude,
            yearly_pattern,
            yearly_amplitude,
            temperature_sensitive,
            temperature_sensitivity,
            base_temperature,
            morning_peak_hour,
            evening_peak_hour,
            peak_amplitude,
            holiday_effect,
            hd,
            extreme_weather_prob,
            extreme_weather_impact,
            noise_std,
            step_hours,
            seed,
        )
    });
    Ok(out.into())
}

#[pyfunction]
#[pyo3(signature = (length, baseline_mean, baseline_std, min_val, max_val, include_circadian, circadian_magnitude, include_hrv, include_events, event_probability, vital_sign_type, seed))]
#[allow(clippy::too_many_arguments)]
pub fn vital_signs(
    py: Python<'_>,
    length: i32,
    baseline_mean: f64,
    baseline_std: f64,
    min_val: f64,
    max_val: f64,
    include_circadian: bool,
    circadian_magnitude: f64,
    include_hrv: bool,
    include_events: bool,
    event_probability: f64,
    vital_sign_type: i32,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let out = PyArray1::<f64>::zeros(py, n, false);
    // SAFETY: Array was just allocated with no other references to its data.
    let s = unsafe { out.as_slice_mut()? };
    py.detach(|| {
        gen::vital_signs(
            s,
            baseline_mean,
            baseline_std,
            min_val,
            max_val,
            include_circadian,
            circadian_magnitude,
            include_hrv,
            include_events,
            event_probability,
            vital_sign_type,
            seed,
        )
    });
    Ok(out.into())
}

#[pyfunction]
#[pyo3(signature = (length, base_value, trend, amplitude, period, measurement_noise, drift_rate, battery_degradation, battery_life, calibration_offset, failure_mode, failure_probability, failure_duration, spatial_correlation, seed))]
#[allow(clippy::too_many_arguments)]
pub fn iot_sensor(
    py: Python<'_>,
    length: i32,
    base_value: f64,
    trend: f64,
    amplitude: f64,
    period: f64,
    measurement_noise: f64,
    drift_rate: f64,
    battery_degradation: bool,
    battery_life: f64,
    calibration_offset: f64,
    failure_mode: i32,
    failure_probability: f64,
    failure_duration: i32,
    spatial_correlation: f64,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let out = PyArray1::<f64>::zeros(py, n, false);
    // SAFETY: Array was just allocated with no other references to its data.
    let s = unsafe { out.as_slice_mut()? };
    py.detach(|| {
        gen::iot_sensor(
            s,
            base_value,
            trend,
            amplitude,
            period,
            measurement_noise,
            drift_rate,
            battery_degradation,
            battery_life,
            calibration_offset,
            failure_mode,
            failure_probability,
            failure_duration,
            spatial_correlation,
            seed,
        )
    });
    Ok(out.into())
}

#[pyfunction]
#[pyo3(signature = (length, base_sessions, conversion_mult, bounce_mult, depth_mult, seasonality_amp, conversion_rate, bounce_rate, avg_session_depth, include_seasonality, include_bots, bot_fraction, output_type, seed))]
#[allow(clippy::too_many_arguments)]
pub fn clickstream(
    py: Python<'_>,
    length: i32,
    base_sessions: f64,
    conversion_mult: f64,
    bounce_mult: f64,
    depth_mult: f64,
    seasonality_amp: f64,
    conversion_rate: f64,
    bounce_rate: f64,
    avg_session_depth: f64,
    include_seasonality: bool,
    include_bots: bool,
    bot_fraction: f64,
    output_type: i32,
    seed: u64,
) -> PyResult<Py<PyArray1<f64>>> {
    if length <= 0 {
        return Err(PyValueError::new_err("length must be > 0"));
    }
    let n = length as usize;
    let out = PyArray1::<f64>::zeros(py, n, false);
    // SAFETY: Array was just allocated with no other references to its data.
    let s = unsafe { out.as_slice_mut()? };
    py.detach(|| {
        gen::clickstream(
            s,
            base_sessions,
            conversion_mult,
            bounce_mult,
            depth_mult,
            seasonality_amp,
            conversion_rate,
            bounce_rate,
            avg_session_depth,
            include_seasonality,
            include_bots,
            bot_fraction,
            output_type,
            seed,
        )
    });
    Ok(out.into())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "domain")?;
    m.add_function(wrap_pyfunction!(intermittent_demand, &m)?)?;
    m.add_function(wrap_pyfunction!(daily_active_users, &m)?)?;
    m.add_function(wrap_pyfunction!(energy_load, &m)?)?;
    m.add_function(wrap_pyfunction!(vital_signs, &m)?)?;
    m.add_function(wrap_pyfunction!(iot_sensor, &m)?)?;
    m.add_function(wrap_pyfunction!(clickstream, &m)?)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("synforecast._lib.domain", &m)?;
    Ok(())
}
