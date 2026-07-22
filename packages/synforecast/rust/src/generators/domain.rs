use crate::rng::SfRng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Intermittent Demand
// ---------------------------------------------------------------------------

/// Intermittent demand generator.
///
/// demand_distribution: 0=poisson, 1=negative_binomial, 2=lognormal, 3=gamma
/// intermittent_pattern: 0=random, 1=clustered, 2=seasonal
pub fn intermittent_demand(
    out: &mut [f64],
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
) {
    let length = out.len();
    let mut rng = SfRng::new(seed);

    // Step 1: Generate occurrence pattern
    let mut occurrence = vec![false; length];

    if intermittent_pattern == 0 {
        // Random: bernoulli(demand_probability) for each t
        for occ in occurrence.iter_mut() {
            *occ = rng.bernoulli(demand_probability);
        }
    } else if intermittent_pattern == 1 {
        // Clustered: Geometric(p) gaps (>= 1, mean 1/p) before each cluster
        // of size cluster_size, matching the Python implementation.
        let mut t: i32 = 0;
        while (t as usize) < length {
            let gap = rng.geometric(demand_probability);
            t += gap;
            // Generate a cluster of demands
            for _c in 0..cluster_size {
                if (t as usize) >= length {
                    break;
                }
                occurrence[t as usize] = true;
                t += 1;
            }
        }
    } else if intermittent_pattern == 2 {
        // Seasonal: probability varies with cosine pattern
        for (t, occ) in occurrence.iter_mut().enumerate() {
            let phase = 2.0 * PI * ((t as i32 % seasonal_period) as f64) / (seasonal_period as f64);
            let mut prob = demand_probability
                + (seasonal_peak_prob - demand_probability) * (phase.cos() + 1.0) / 2.0;
            prob = prob.clamp(0.0, 1.0);
            *occ = rng.bernoulli(prob);
        }
    }

    // Step 2 & 3: Generate demand sizes and combine
    let variance = demand_std * demand_std;

    for t in 0..length {
        if !occurrence[t] {
            out[t] = 0.0;
            continue;
        }

        let mut demand_size: f64;

        if demand_distribution == 0 {
            // Poisson
            demand_size = rng.poisson(demand_mean) as f64;
        } else if demand_distribution == 1 {
            // Negative binomial: variance = demand_std^2
            // p = demand_mean / variance, n = demand_mean * p / (1 - p).
            // Sampled via the exact gamma-Poisson mixture with real-valued n
            // (rounding n to an integer biases the mean; numpy's
            // negative_binomial accepts real n, so match it).
            if variance > demand_mean {
                let p = demand_mean / variance;
                let n_param = demand_mean * p / (1.0 - p);
                let lambda = rng.gamma(n_param, (1.0 - p) / p);
                demand_size = rng.poisson(lambda.max(0.0)) as f64;
            } else {
                demand_size = rng.poisson(demand_mean) as f64;
            }
        } else if demand_distribution == 2 {
            // Lognormal: mu = ln(mean^2 / sqrt(var + mean^2))
            //            sigma = sqrt(ln(1 + var / mean^2))
            let mean_sq = demand_mean * demand_mean;
            let mu = (mean_sq / (variance + mean_sq).sqrt()).ln();
            let sigma = (1.0 + variance / mean_sq).ln().sqrt();
            demand_size = rng.lognormal(mu, sigma);
        } else if demand_distribution == 3 {
            // Gamma: shape = (mean/std)^2, scale = std^2/mean
            let shape = (demand_mean / demand_std) * (demand_mean / demand_std);
            let scale = variance / demand_mean;
            demand_size = rng.gamma(shape, scale);
        } else {
            demand_size = rng.poisson(demand_mean) as f64;
        }

        // Step 4: Clip minimum
        demand_size = demand_size.max(min_demand);
        out[t] = demand_size;
    }
}

// ---------------------------------------------------------------------------
// Daily Active Users
// ---------------------------------------------------------------------------

/// Result struct for daily_active_users which returns both main values and event indices.
pub struct DailyActiveUsersResult {
    pub events: Vec<i32>,
}

/// Daily active users generator.
///
/// Returns the main values in `out` and event indices via the returned struct.
pub fn daily_active_users(
    out: &mut [f64],
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
) -> DailyActiveUsersResult {
    let length = out.len();
    let mut rng = SfRng::new(seed);
    let mut events = vec![0_i32; length];

    let mut event_boost = 0.0;

    for t in 0..length {
        // Mirrors the Python helper: int(t * step_hours) // 24
        let day_index = ((t as f64 * step_hours) as i64 / 24) as i32;

        // Base with exponential growth
        let mut base = base_users * (1.0 + growth_rate).powi(day_index);

        // Weekly pattern: reduce on weekends
        if weekly_pattern {
            let day_of_week = day_index % 7;
            if day_of_week >= 5 {
                base *= weekend_factor;
            }
        }

        // Random events
        events[t] = 0;
        if rng.uniform01() < event_probability {
            events[t] = 1;
            let impact = rng.uniform(event_impact_min, event_impact_max);
            event_boost += (impact - 1.0) * base;
        }

        // Combine base and event boost
        let mut dau = base + event_boost;

        // Decay event boost
        event_boost *= 1.0 - event_decay_rate;

        // Add noise proportional to dau
        let noise = rng.normal(0.0, noise_std * dau);
        dau += noise;

        // Clip to non-negative
        out[t] = dau.max(0.0);
    }

    DailyActiveUsersResult { events }
}

// ---------------------------------------------------------------------------
// Energy Load
// ---------------------------------------------------------------------------

/// Energy load generator.
///
/// load_type: 0=residential, 1=commercial, 2=industrial
pub fn energy_load(
    out: &mut [f64],
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
    holiday_days: &[i32],
    extreme_weather_prob: f64,
    extreme_weather_impact: f64,
    noise_std: f64,
    step_hours: f64,
    seed: u64,
) {
    let length = out.len();
    let mut rng = SfRng::new(seed);
    let n_holidays = holiday_days.len();

    // Mirrors the Python helpers: hour = int(t*step_hours) % 24,
    // day_index = int(t*step_hours) // 24 (t and step_hours are >= 0, so
    // truncation equals floor).
    let temporal_indices = |t: usize| -> (i32, i64) {
        let total_hours = (t as f64 * step_hours) as i64;
        ((total_hours % 24) as i32, total_hours / 24)
    };

    // Pre-generate temperature if needed
    let mut temperature = Vec::new();
    if temperature_sensitive {
        temperature.resize(length, 0.0);
        for (t, temp) in temperature.iter_mut().enumerate() {
            let (hour_of_day, day_index) = temporal_indices(t);
            let day_of_year = day_index % 365;

            // Daily cycle: -5*cos(2*pi*hour/24)
            let daily_temp = -5.0 * (2.0 * PI * hour_of_day as f64 / 24.0).cos();
            // Yearly cycle: -10*cos(2*pi*day_of_year/365)
            let yearly_temp = -10.0 * (2.0 * PI * day_of_year as f64 / 365.0).cos();
            let temp_noise = rng.normal(0.0, 2.0);
            *temp = base_temperature + daily_temp + yearly_temp + temp_noise;
        }
    }

    for t in 0..length {
        let mut load = base_load;

        // Compute temporal indices
        let (hour_of_day, day_index) = temporal_indices(t);
        let day_of_week = (day_index % 7) as i32;
        let day_of_year = (day_index % 365) as i32;

        // Daily pattern
        if daily_pattern {
            let daily_load;
            if load_type == 0 {
                // Residential: two Gaussian peaks (morning + evening)
                let morning_peak = peak_amplitude
                    * (-((hour_of_day - morning_peak_hour) * (hour_of_day - morning_peak_hour))
                        as f64
                        / 8.0)
                        .exp();
                let evening_peak = peak_amplitude
                    * (-((hour_of_day - evening_peak_hour) * (hour_of_day - evening_peak_hour))
                        as f64
                        / 8.0)
                        .exp();
                daily_load = morning_peak + evening_peak;
            } else if load_type == 1 {
                // Commercial: single broad peak during business hours
                daily_load = daily_amplitude
                    * (-((hour_of_day - 14) * (hour_of_day - 14)) as f64 / 50.0).exp();
            } else if load_type == 2 {
                // Industrial: more constant with slight dip at night
                daily_load = -daily_amplitude
                    * (-((hour_of_day - 3) * (hour_of_day - 3)) as f64 / 20.0).exp();
            } else {
                daily_load = 0.0;
            }
            load += daily_load;
        }

        // Weekly pattern
        if weekly_pattern {
            if load_type == 0 || load_type == 1 {
                // Residential/Commercial: subtract on weekends
                if day_of_week >= 5 {
                    load -= weekly_amplitude;
                }
            } else {
                // Industrial: sinusoidal weekly pattern
                load += weekly_amplitude * (2.0 * PI * day_of_week as f64 / 7.0).sin();
            }
        }

        // Yearly pattern
        if yearly_pattern {
            load += yearly_amplitude * (1.0 + (2.0 * PI * day_of_year as f64 / 365.0).cos());
        }

        // Temperature sensitivity
        if temperature_sensitive {
            let temp_deviation = temperature[t] - base_temperature;
            load += temperature_sensitivity * temp_deviation.abs();
        }

        // Holiday effect
        if n_holidays > 0 {
            for &hday in &holiday_days[..n_holidays] {
                if day_of_year == hday {
                    load *= 1.0 - holiday_effect;
                    break;
                }
            }
        }

        // Extreme weather events
        if rng.uniform01() < extreme_weather_prob {
            load *= extreme_weather_impact;
        }

        // Clip to non-negative
        out[t] = load.max(0.0);
    }

    // Add noise after loop (matches Python which adds noise vectorized)
    for v in out.iter_mut() {
        *v += rng.normal(0.0, noise_std);
        *v = v.max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Vital Signs
// ---------------------------------------------------------------------------

/// Vital signs generator.
///
/// vital_sign_type: 0=HR, 1=SBP, 2=DBP, 3=RR, 4=SpO2, 5=Temperature
pub fn vital_signs(
    out: &mut [f64],
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
) {
    let length = out.len();
    let mut rng = SfRng::new(seed);

    // Step 1: Generate baseline with individual variation
    let baseline = baseline_mean + rng.normal(0.0, baseline_std * 0.3);

    // Step 2: Initialize with baseline and add random walk drift
    let mut drift = vec![0.0_f64; length];
    let mut drift_cumsum = 0.0;
    for d in drift.iter_mut() {
        drift_cumsum += rng.normal(0.0, baseline_std * 0.05);
        *d = drift_cumsum;
    }
    // Normalize drift to start at 0 (match Python: drift = drift - drift[0])
    let drift_start = drift[0];
    for d in drift.iter_mut() {
        *d -= drift_start;
    }

    for (out_v, &d) in out.iter_mut().zip(drift.iter()) {
        *out_v = baseline + d;
    }

    // Step 3: Circadian rhythm
    // Assumes each time step is 1 minute (1440 minutes per day)
    if include_circadian {
        for (t, out_v) in out.iter_mut().enumerate() {
            let phase = 2.0 * PI * t as f64 / 1440.0;
            let circadian =
                circadian_magnitude * ((phase - PI / 2.0).sin() + 0.3 * (2.0 * phase).sin());
            *out_v += circadian;
        }
    }

    // Step 4: Heart rate variability (for HR, SBP, DBP: types 0, 1, 2)
    if include_hrv && vital_sign_type <= 2 {
        // VLF, LF, HF frequency components (assuming 1-min sampling)
        let vlf_freq = 2.0 * PI * 0.02 * 60.0; // ~0.02 Hz
        let lf_freq = 2.0 * PI * 0.1 * 60.0; // ~0.1 Hz
        let hf_freq = 2.0 * PI * 0.25 * 60.0; // ~0.25 Hz

        let lf_phase = rng.uniform(0.0, 2.0 * PI);
        let hf_phase = rng.uniform(0.0, 2.0 * PI);

        let hrv_scale = baseline_std * 0.5;

        for (t, out_v) in out.iter_mut().enumerate() {
            let tf = t as f64;
            let vlf = 0.3 * (vlf_freq * tf).sin();
            let lf = 0.4 * (lf_freq * tf + lf_phase).sin();
            let hf = 0.3 * (hf_freq * tf + hf_phase).sin();
            let hrv_noise = rng.normal(0.0, 0.2);
            *out_v += hrv_scale * (vlf + lf + hf + hrv_noise);
        }
    }

    // Step 5: Random events (activity, rest, spikes)
    if include_events {
        let mut events = vec![0.0_f64; length];
        for t in 0..length {
            if rng.uniform01() < event_probability {
                // Determine event type: 0=activity, 1=rest, 2=spike
                let event_type = rng.integers(0, 3);
                let duration = rng.integers(5, 30);
                let end = (t as i32 + duration).min(length as i32) as usize;

                if event_type == 0 {
                    // Activity: gradual increase then decay with random magnitude
                    let magnitude = rng.uniform(0.5, 2.0);
                    let span = end - t;
                    for d in 0..span {
                        let envelope = (PI * d as f64 / span as f64).sin();
                        events[t + d] += magnitude * envelope;
                    }
                } else if event_type == 1 {
                    // Rest: flat decrease with random magnitude
                    let magnitude = rng.uniform(-1.5, -0.5);
                    for ev in &mut events[t..end] {
                        *ev += magnitude;
                    }
                } else {
                    // Spike: sharp transient with uniform magnitude
                    events[t] += rng.uniform(1.0, 3.0);
                }
            }
        }
        for t in 0..length {
            out[t] += baseline_std * events[t];
        }
    }

    // Step 6: Correlations with heart rate for non-HR vitals
    if vital_sign_type >= 1 {
        // Generate HR deviation from events (matching Python _generate_events)
        let hr_std = 8.0; // heart_rate baseline std for "healthy" archetype
        let mut hr_deviation = vec![0.0_f64; length];
        for t in 0..length {
            if rng.uniform01() < event_probability {
                let event_type = rng.integers(0, 3);
                let duration = rng.integers(5, 30);
                let end = (t as i32 + duration).min(length as i32) as usize;
                if event_type == 0 {
                    let magnitude = rng.uniform(0.5, 2.0);
                    let span = end - t;
                    for d in 0..span {
                        let envelope = (PI * d as f64 / span as f64).sin();
                        hr_deviation[t + d] += magnitude * envelope;
                    }
                } else if event_type == 1 {
                    let magnitude = rng.uniform(-1.5, -0.5);
                    for hr in &mut hr_deviation[t..end] {
                        *hr += magnitude;
                    }
                } else {
                    hr_deviation[t] += rng.uniform(1.0, 3.0);
                }
            }
        }
        // Scale by HR std
        for hr in hr_deviation.iter_mut() {
            *hr *= hr_std;
        }

        // Apply correlations based on vital sign type
        if vital_sign_type == 1 {
            // Systolic BP: positively correlated with HR
            for t in 0..length {
                out[t] += 0.3 * hr_deviation[t];
            }
        } else if vital_sign_type == 2 {
            // Diastolic BP: positively correlated with HR
            for t in 0..length {
                out[t] += 0.2 * hr_deviation[t];
            }
        } else if vital_sign_type == 3 {
            // Respiratory rate: increases with HR
            for t in 0..length {
                out[t] += 0.15 * hr_deviation[t];
            }
        } else if vital_sign_type == 4 {
            // SpO2: slightly decreases during high activity
            for t in 0..length {
                out[t] -= 0.05 * hr_deviation[t].max(0.0);
            }
        }
        // Temperature (type 5): no direct HR correlation applied
    }

    // Step 7: Measurement noise
    for out_v in out.iter_mut() {
        *out_v += rng.normal(0.0, baseline_std * 0.2);
    }

    // Step 8: Clip to [min_val, max_val]
    for out_v in out.iter_mut() {
        *out_v = out_v.clamp(min_val, max_val);
    }
}

// ---------------------------------------------------------------------------
// IoT Sensor
// ---------------------------------------------------------------------------

/// IoT sensor simulation with drift, degradation, and failure modes.
///
/// failure_mode: 0=none, 1=complete, 2=intermittent, 3=stuck
pub fn iot_sensor(
    out: &mut [f64],
    base_value: f64,
    trend: f64,
    amplitude: f64,
    period: f64,
    measurement_noise: f64,
    drift_rate: f64,
    drift_noise: f64,
    battery_degradation: bool,
    battery_life: f64,
    battery_degradation_rate: f64,
    calibration_offset: f64,
    failure_mode: i32,
    failure_probability: f64,
    failure_duration: i32,
    stuck_value: Option<f64>,
    _spatial_correlation: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);

    // Track sensor state
    let mut drift_accum = 0.0_f64;
    let mut complete_failure_active = false;
    let mut intermittent_remaining = 0_i32;
    let mut stuck_remaining = 0_i32;
    let mut stuck_reading = 0.0_f64;

    for (t, out_v) in out.iter_mut().enumerate() {
        let tf = t as f64;

        // 1. Generate base signal: base_value + trend*t + amplitude*sin(2*pi*t/period)
        let mut signal = base_value + trend * tf;
        if period > 0.0 {
            signal += amplitude * (2.0 * PI * tf / period).sin();
        }

        // 2. Add calibration offset
        signal += calibration_offset;

        // 3. Add sensor drift: cumulative random walk with per-step mean
        //    drift_rate and std drift_noise (first increment is zero).
        if t > 0 {
            drift_accum += rng.normal(drift_rate, drift_noise);
        }
        signal += drift_accum;

        // 4. Add base measurement noise (all steps)
        signal += rng.normal(0.0, measurement_noise);

        // 5. Battery degradation: after battery_life steps, add extra noise
        //    scaled by battery_degradation_rate and attenuate the signal.
        if battery_degradation && tf >= battery_life {
            let steps_since = tf - battery_life;
            let degradation_factor = 1.0 + steps_since * battery_degradation_rate;
            signal += rng.normal(0.0, measurement_noise * degradation_factor);
            signal *= (1.0 - steps_since * battery_degradation_rate * 0.1).max(0.0);
        }

        // 6. Failure modes
        match failure_mode {
            1 => {
                // Complete failure: random onset, all subsequent become NaN
                if !complete_failure_active && rng.uniform01() < failure_probability {
                    complete_failure_active = true;
                }
                if complete_failure_active {
                    signal = f64::NAN;
                }
            }
            2 => {
                // Intermittent: probabilistic NaN bursts of failure_duration
                if intermittent_remaining > 0 {
                    signal = f64::NAN;
                    intermittent_remaining -= 1;
                } else if rng.uniform01() < failure_probability {
                    signal = f64::NAN;
                    intermittent_remaining = failure_duration - 1;
                }
            }
            3 => {
                // Stuck: probabilistic stuck-value bursts of failure_duration.
                // Freeze at the configured stuck_value if given, else at the
                // reading when the episode starts.
                if stuck_remaining > 0 {
                    signal = stuck_reading;
                    stuck_remaining -= 1;
                } else if rng.uniform01() < failure_probability {
                    stuck_reading = stuck_value.unwrap_or(signal);
                    signal = stuck_reading;
                    stuck_remaining = failure_duration - 1;
                }
            }
            _ => {
                // failure_mode == 0: no failures
            }
        }

        *out_v = signal;
    }
}

// ---------------------------------------------------------------------------
// Clickstream
// ---------------------------------------------------------------------------

/// Clickstream generator.
///
/// output_type: 0=sessions, 1=pageviews, 2=conversions, 3=bounce_rate
pub fn clickstream(
    out: &mut [f64],
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
) {
    let length = out.len();
    let mut rng = SfRng::new(seed);

    // Hardcoded hourly pattern (matching Python _hourly_pattern)
    let hourly_raw: [f64; 24] = [
        0.3, 0.2, 0.15, 0.1, 0.1, 0.15, // 0-5 AM
        0.3, 0.5, 0.7, 0.9, 1.0, 1.1, // 6-11 AM
        1.0, 0.95, 0.9, 0.85, 0.9, 1.0, // 12-5 PM
        1.1, 1.2, 1.3, 1.2, 0.9, 0.5, // 6-11 PM
    ];
    // Normalize to mean 1.0
    let hourly_sum: f64 = hourly_raw.iter().sum();
    let hourly_mean = hourly_sum / 24.0;
    let hourly_pattern: Vec<f64> = hourly_raw.iter().map(|&h| h / hourly_mean).collect();

    // Hardcoded daily pattern (matching Python _daily_pattern)
    // B2C pattern: higher weekends
    let daily_raw: [f64; 7] = [0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1]; // Mon-Sun
    let daily_sum: f64 = daily_raw.iter().sum();
    let daily_mean = daily_sum / 7.0;
    let daily_pattern: Vec<f64> = daily_raw.iter().map(|&d| d / daily_mean).collect();

    // Generate trend as random walk in log space (matching Python _generate_trend)
    let drift = rng.uniform(-0.0005, 0.001); // Slight upward bias
    let mut trend_log = vec![0.0_f64; length];
    let mut log_trend_cumsum = 0.0;
    for tl in trend_log.iter_mut() {
        log_trend_cumsum += drift + rng.normal(0.0, 0.002);
        *tl = log_trend_cumsum;
    }
    // Normalize to start at 1 (match Python: exp(log_trend - log_trend[0]))
    let log_trend_start = trend_log[0];
    let mut trend = vec![0.0_f64; length];
    for t in 0..length {
        trend[t] = (trend_log[t] - log_trend_start).exp();
    }

    // Generate seasonality for all timesteps
    let mut seasonality = vec![1.0_f64; length];
    if include_seasonality {
        for (t, seas) in seasonality.iter_mut().enumerate() {
            let hour = t % 24;
            let day = (t / 24) % 7;
            let combined = hourly_pattern[hour] * daily_pattern[day];
            *seas = 1.0 + seasonality_amp * (combined - 1.0);
        }
    }

    // Generate human sessions with Poisson noise
    let mut human_sessions = vec![0_i32; length];
    for t in 0..length {
        let expected = base_sessions * seasonality[t] * trend[t];
        human_sessions[t] = rng.poisson(expected.max(0.1));
    }

    // Generate bot traffic (matching Python _generate_bot_traffic)
    let mut bot_sessions_d = vec![0.0_f64; length];
    if include_bots {
        // Bot baseline: scale by bot_fraction / (1 - bot_fraction)
        for t in 0..length {
            let bot_base = human_sessions[t] as f64 * bot_fraction / (1.0 - bot_fraction);
            bot_sessions_d[t] = bot_base * (0.8 + 0.4 * rng.uniform01());
        }
        // Add occasional crawl spikes
        let max_spikes = (length / 100).max(1) + 1;
        let n_spikes = rng.integers(0, max_spikes as i32);
        for _s in 0..n_spikes {
            let spike_time = rng.integers(0, length as i32);
            let spike_duration = rng.integers(1, 6);
            let spike_magnitude = rng.uniform(2.0, 10.0);
            let end = ((spike_time + spike_duration) as usize).min(length);
            for bs in &mut bot_sessions_d[(spike_time as usize)..end] {
                *bs *= spike_magnitude;
            }
        }
    }

    // Convert bot sessions to int
    let mut bot_sessions_i = vec![0_i32; length];
    for t in 0..length {
        bot_sessions_i[t] = bot_sessions_d[t] as i32;
    }

    // Compute derived metrics
    let effective_bounce_rate = (bounce_rate * bounce_mult).clamp(0.0, 1.0);
    let effective_conv_rate = (conversion_rate * conversion_mult).clamp(0.0, 1.0);
    let depth = avg_session_depth * depth_mult;

    for t in 0..length {
        let total_sessions = human_sessions[t] + bot_sessions_i[t];

        // Bounces from human sessions
        let bounces = rng.binomial(human_sessions[t], effective_bounce_rate);

        // Engaged sessions
        let engaged = human_sessions[t] - bounces;

        // Pageviews: bounces contribute 1 page, engaged contribute geometric
        // depth, bots contribute ~5 pages each
        let pageviews_engaged = if engaged > 0 {
            // Sum of `engaged` independent Geometric(p) RVs (min 1, mean 1/p)
            // = NegBin(engaged, p) failures + engaged. SfRng::negative_binomial
            // returns the failure count, so the +engaged shift is required.
            let p = 1.0 / depth;
            rng.negative_binomial(engaged, p) as f64 + engaged as f64
        } else {
            0.0
        };
        let pageviews = bounces as f64 + pageviews_engaged + bot_sessions_d[t] * 5.0;

        // Conversions from engaged sessions
        let conversions = rng.binomial(engaged, effective_conv_rate);

        // Output based on type
        if output_type == 0 {
            out[t] = total_sessions as f64;
        } else if output_type == 1 {
            out[t] = pageviews;
        } else if output_type == 2 {
            out[t] = conversions as f64;
        } else if output_type == 3 {
            // Bounced fraction of total observed traffic (including bots),
            // matching the Python implementation: bounces / total_sessions.
            out[t] = if total_sessions > 0 {
                bounces as f64 / total_sessions as f64
            } else {
                0.0
            };
        } else {
            out[t] = total_sessions as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 200;

    fn assert_finite(out: &[f64], name: &str) {
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "{name}: non-finite at index {i}: {v}");
        }
    }

    fn assert_deterministic(f: impl Fn(&mut [f64]), name: &str) {
        let mut a = vec![0.0; N];
        let mut b = vec![0.0; N];
        f(&mut a);
        f(&mut b);
        assert_eq!(a, b, "{name}: same seed produced different results");
    }

    // --- Intermittent Demand ---

    #[test]
    fn test_intermittent_demand_has_zeros() {
        let mut out = vec![0.0; N];
        intermittent_demand(&mut out, 0.3, 0, 5.0, 2.0, 0, 3, 12, 0.5, 0.0, 42);
        assert_finite(&out, "intermittent_demand");
        let zeros = out.iter().filter(|&&v| v == 0.0).count();
        assert!(zeros > 0, "intermittent demand should have zeros");
        for &v in &out {
            assert!(v >= 0.0, "demand should be non-negative");
        }
    }

    #[test]
    fn test_intermittent_demand_clustered() {
        let mut out = vec![0.0; N];
        intermittent_demand(&mut out, 0.3, 0, 5.0, 2.0, 1, 5, 12, 0.5, 0.0, 42);
        assert_finite(&out, "intermittent_clustered");
    }

    #[test]
    fn test_intermittent_demand_seasonal() {
        let mut out = vec![0.0; N];
        intermittent_demand(&mut out, 0.3, 0, 5.0, 2.0, 2, 3, 20, 0.6, 0.0, 42);
        assert_finite(&out, "intermittent_seasonal");
    }

    #[test]
    fn test_intermittent_demand_distributions() {
        for dist in 0..4 {
            let mut out = vec![0.0; N];
            intermittent_demand(&mut out, 0.5, dist, 5.0, 2.0, 0, 3, 12, 0.5, 0.0, 42);
            assert_finite(&out, &format!("intermittent_dist_{dist}"));
        }
    }

    #[test]
    fn test_intermittent_demand_min_demand() {
        let mut out = vec![0.0; N];
        intermittent_demand(&mut out, 0.5, 0, 5.0, 2.0, 0, 3, 12, 0.5, 2.0, 42);
        for &v in &out {
            // Non-zero values should be >= min_demand
            if v > 0.0 {
                assert!(v >= 2.0, "demand {v} should be >= min_demand 2.0");
            }
        }
    }

    #[test]
    fn test_intermittent_demand_deterministic() {
        assert_deterministic(
            |o| intermittent_demand(o, 0.3, 0, 5.0, 2.0, 0, 3, 12, 0.5, 0.0, 42),
            "intermittent",
        );
    }

    // --- Daily Active Users ---

    #[test]
    fn test_daily_active_users_nonnegative() {
        let mut out = vec![0.0; N];
        let result = daily_active_users(
            &mut out, 1000.0, 0.001, 0.7, true, 0.05, 1.0, 2.0, 0.1, 0.05, 24.0, 42,
        );
        assert_finite(&out, "dau");
        for &v in &out {
            assert!(v >= 0.0, "DAU should be non-negative");
        }
        assert_eq!(result.events.len(), N);
    }

    #[test]
    fn test_daily_active_users_deterministic() {
        let mut a = vec![0.0; N];
        let mut b = vec![0.0; N];
        daily_active_users(
            &mut a, 1000.0, 0.001, 0.7, true, 0.05, 1.0, 2.0, 0.1, 0.05, 24.0, 42,
        );
        daily_active_users(
            &mut b, 1000.0, 0.001, 0.7, true, 0.05, 1.0, 2.0, 0.1, 0.05, 24.0, 42,
        );
        assert_eq!(a, b, "DAU should be deterministic with same seed");
    }

    // --- Energy Load ---

    #[test]
    fn test_energy_load_nonnegative() {
        let holidays = [1, 50, 100, 200, 300];
        let mut out = vec![0.0; N];
        energy_load(
            &mut out, 100.0, 0, true, 20.0, true, 10.0, true, 15.0, true, 0.5, 20.0, 8, 18, 30.0,
            0.2, &holidays, 0.01, 1.5, 5.0, 1.0, 42,
        );
        assert_finite(&out, "energy_load");
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_energy_load_commercial() {
        let mut out = vec![0.0; N];
        energy_load(
            &mut out,
            200.0,
            1,
            true,
            30.0,
            true,
            15.0,
            false,
            0.0,
            false,
            0.0,
            20.0,
            8,
            18,
            30.0,
            0.0,
            &[],
            0.0,
            1.0,
            5.0,
            1.0,
            42,
        );
        assert_finite(&out, "energy_commercial");
    }

    #[test]
    fn test_energy_load_industrial() {
        let mut out = vec![0.0; N];
        energy_load(
            &mut out,
            500.0,
            2,
            true,
            20.0,
            true,
            10.0,
            false,
            0.0,
            false,
            0.0,
            20.0,
            8,
            18,
            30.0,
            0.0,
            &[],
            0.0,
            1.0,
            10.0,
            1.0,
            42,
        );
        assert_finite(&out, "energy_industrial");
    }

    #[test]
    fn test_energy_load_deterministic() {
        assert_deterministic(
            |o| {
                energy_load(
                    o,
                    100.0,
                    0,
                    true,
                    20.0,
                    true,
                    10.0,
                    false,
                    0.0,
                    false,
                    0.0,
                    20.0,
                    8,
                    18,
                    30.0,
                    0.0,
                    &[],
                    0.0,
                    1.0,
                    5.0,
                    1.0,
                    42,
                )
            },
            "energy_load",
        );
    }

    // --- Vital Signs ---

    #[test]
    fn test_vital_signs_clamped() {
        let mut out = vec![0.0; N];
        vital_signs(
            &mut out, 72.0, 8.0, 40.0, 200.0, true, 5.0, true, true, 0.02, 0, 42,
        );
        assert_finite(&out, "vital_signs");
        for &v in &out {
            assert!(
                (40.0..=200.0).contains(&v),
                "vital signs should be clamped to [40, 200], got {v}"
            );
        }
    }

    #[test]
    fn test_vital_signs_spo2() {
        let mut out = vec![0.0; N];
        vital_signs(
            &mut out, 97.0, 1.0, 85.0, 100.0, true, 0.5, false, false, 0.01, 4, 42,
        );
        assert_finite(&out, "vital_spo2");
        for &v in &out {
            assert!((85.0..=100.0).contains(&v));
        }
    }

    #[test]
    fn test_vital_signs_deterministic() {
        assert_deterministic(
            |o| {
                vital_signs(
                    o, 72.0, 8.0, 40.0, 200.0, true, 5.0, true, true, 0.02, 0, 42,
                )
            },
            "vital_signs",
        );
    }

    // --- IoT Sensor ---

    #[test]
    fn test_iot_sensor_no_failure() {
        let mut out = vec![0.0; N];
        iot_sensor(
            &mut out, 25.0, 0.001, 5.0, 100.0, 0.5, 0.0001, 0.01, false, 500.0, 0.001, 0.0, 0, 0.0,
            10, None, 0.0, 42,
        );
        assert_finite(&out, "iot_no_failure");
    }

    #[test]
    fn test_iot_sensor_complete_failure_has_nan() {
        let mut out = vec![0.0; 500];
        iot_sensor(
            &mut out, 25.0, 0.0, 0.0, 100.0, 0.5, 0.0, 0.01, false, 500.0, 0.001, 0.0, 1, 0.1, 10,
            None, 0.0, 42,
        );
        // Complete failure (mode=1, prob=0.1) should eventually produce NaN
        let nan_count = out.iter().filter(|v| v.is_nan()).count();
        assert!(nan_count > 0, "complete failure should produce NaN values");
    }

    #[test]
    fn test_iot_sensor_with_degradation() {
        let mut out = vec![0.0; N];
        iot_sensor(
            &mut out, 25.0, 0.001, 5.0, 100.0, 0.5, 0.0001, 0.01, true, 100.0, 0.001, 1.0, 0, 0.0,
            10, None, 0.0, 42,
        );
        assert_finite(&out, "iot_degradation");
    }

    #[test]
    fn test_iot_sensor_deterministic() {
        assert_deterministic(
            |o| {
                iot_sensor(
                    o, 25.0, 0.001, 5.0, 100.0, 0.5, 0.0001, 0.01, false, 500.0, 0.001, 0.0, 0,
                    0.0, 10, None, 0.0, 42,
                )
            },
            "iot_sensor",
        );
    }

    // --- Clickstream ---

    #[test]
    fn test_clickstream_sessions_nonnegative() {
        let mut out = vec![0.0; N];
        clickstream(
            &mut out, 100.0, 1.0, 1.0, 1.0, 0.5, 0.03, 0.4, 3.0, true, false, 0.0, 0, 42,
        );
        assert_finite(&out, "clickstream_sessions");
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_clickstream_all_output_types() {
        for ot in 0..4 {
            let mut out = vec![0.0; N];
            clickstream(
                &mut out, 100.0, 1.0, 1.0, 1.0, 0.5, 0.03, 0.4, 3.0, true, false, 0.0, ot, 42,
            );
            assert_finite(&out, &format!("clickstream_type_{ot}"));
            for &v in &out {
                assert!(v >= 0.0, "output type {ot} should be non-negative");
            }
        }
    }

    #[test]
    fn test_clickstream_with_bots() {
        let mut out = vec![0.0; N];
        clickstream(
            &mut out, 100.0, 1.0, 1.0, 1.0, 0.5, 0.03, 0.4, 3.0, true, true, 0.2, 0, 42,
        );
        assert_finite(&out, "clickstream_bots");
    }

    #[test]
    fn test_clickstream_deterministic() {
        assert_deterministic(
            |o| {
                clickstream(
                    o, 100.0, 1.0, 1.0, 1.0, 0.5, 0.03, 0.4, 3.0, true, false, 0.0, 0, 42,
                )
            },
            "clickstream",
        );
    }
}
