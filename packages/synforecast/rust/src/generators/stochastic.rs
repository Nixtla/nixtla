use crate::rng::SfRng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Ornstein-Uhlenbeck process
// ---------------------------------------------------------------------------
pub fn ornstein_uhlenbeck(
    out: &mut [f64],
    theta: f64,
    mu: f64,
    sigma: f64,
    initial_value: f64,
    dt: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let mut rng = SfRng::new(seed);
    let sqrt_dt = dt.sqrt();
    let length = out.len();
    if length == 0 {
        return;
    }
    out[0] = initial_value;
    for t in 1..length {
        let drift = theta * (mu - out[t - 1]) * dt;
        let diffusion = sigma * sqrt_dt * rng.sample_innovation(1.0, innov_dist, innov_param);
        out[t] = out[t - 1] + drift + diffusion;
    }
}

// ---------------------------------------------------------------------------
// Geometric Brownian Motion
// ---------------------------------------------------------------------------
pub fn geometric_brownian_motion(
    out: &mut [f64],
    mu: f64,
    sigma: f64,
    initial_value: f64,
    dt: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let mut rng = SfRng::new(seed);
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let sqrt_dt = dt.sqrt();
    let length = out.len();
    if length == 0 {
        return;
    }
    out[0] = initial_value;
    for t in 1..length {
        let diffusion = sigma * sqrt_dt * rng.sample_innovation(1.0, innov_dist, innov_param);
        out[t] = out[t - 1] * (drift + diffusion).exp();
    }
}

// ---------------------------------------------------------------------------
// Jump Diffusion (Merton model)
// ---------------------------------------------------------------------------
pub fn jump_diffusion(
    out: &mut [f64],
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
) {
    let mut rng = SfRng::new(seed);
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let sqrt_dt = dt.sqrt();
    let length = out.len();
    if length == 0 {
        return;
    }
    out[0] = initial_value;
    for t in 1..length {
        // Continuous part (GBM)
        let diffusion = sigma * sqrt_dt * rng.sample_innovation(1.0, innov_dist, innov_param);

        // Jump part (Compound Poisson)
        let num_jumps = rng.poisson(lambda_jump * dt);
        let mut jump_component = 0.0;
        for _ in 0..num_jumps {
            jump_component += jump_mean + rng.sample_innovation(jump_std, innov_dist, innov_param);
        }

        out[t] = out[t - 1] * (drift + diffusion + jump_component).exp();
    }
}

// ---------------------------------------------------------------------------
// Poisson Process
// ---------------------------------------------------------------------------
pub fn poisson_process(out: &mut [f64], lambda_rate: f64, cumulative: bool, seed: u64) {
    let mut rng = SfRng::new(seed);
    if cumulative {
        let mut cumsum = 0.0;
        for v in out.iter_mut() {
            cumsum += rng.poisson(lambda_rate) as f64;
            *v = cumsum;
        }
    } else {
        for v in out.iter_mut() {
            *v = rng.poisson(lambda_rate) as f64;
        }
    }
}

// ---------------------------------------------------------------------------
// Cyclic process
// ---------------------------------------------------------------------------
pub fn cyclic(
    out: &mut [f64],
    base_level: f64,
    trend: f64,
    cycle_period_mean: f64,
    cycle_period_std: f64,
    cycle_amplitude_mean: f64,
    cycle_amplitude_std: f64,
    num_cycles: i32,
    noise_std: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);

    // Initialize with base + trend
    for (t, v) in out.iter_mut().enumerate() {
        *v = base_level + trend * t as f64;
    }

    // Generate cycle parameters
    let mut cycle_periods = vec![0.0_f64; num_cycles as usize];
    let mut cycle_amplitudes = vec![0.0_f64; num_cycles as usize];
    let mut cycle_phases = vec![0.0_f64; num_cycles as usize];

    for i in 0..num_cycles as usize {
        cycle_periods[i] = rng.normal(cycle_period_mean, cycle_period_std).abs();
        cycle_amplitudes[i] = rng.normal(cycle_amplitude_mean, cycle_amplitude_std);
        cycle_phases[i] = rng.uniform(0.0, 2.0 * PI);
    }

    // Add each cycle component
    for i in 0..num_cycles as usize {
        let period = cycle_periods[i];
        let amplitude = cycle_amplitudes[i];
        let phase = cycle_phases[i];

        let mut cumulative_phase = phase;
        for (t, v) in out.iter_mut().enumerate() {
            // Period variation over time for irregularity
            let period_variation = 1.0 + 0.2 * (2.0 * PI * t as f64 / (period * 2.0)).sin();
            let frequency = 2.0 * PI / (period * period_variation);
            cumulative_phase += frequency;
            *v += amplitude * cumulative_phase.sin();
        }
    }

    // Add noise
    for v in out.iter_mut() {
        *v += rng.normal(0.0, noise_std);
    }
}

// ---------------------------------------------------------------------------
// GARCH(p,q)
// ---------------------------------------------------------------------------
pub fn garch(
    out: &mut [f64],
    p: i32,
    q: i32,
    omega: f64,
    alpha: &[f64],
    beta: &[f64],
    mu: f64,
    initial_variance: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let mut rng = SfRng::new(seed);
    let length = out.len();
    let burn_in = 100;
    let total = length + burn_in;
    let mut returns = vec![0.0_f64; total];
    // eps_t = sigma_t * z_t: the ARCH term must use lagged squared
    // innovations, not lagged squared returns (they differ when mu != 0).
    let mut eps = vec![0.0_f64; total];
    let mut variances = vec![initial_variance; total];
    let max_lag = p.max(q) as usize;

    for t in max_lag..total {
        let mut variance = omega;
        for i in 0..q as usize {
            variance += alpha[i] * (eps[t - i - 1] * eps[t - i - 1]);
        }
        for j in 0..p as usize {
            variance += beta[j] * variances[t - j - 1];
        }
        variances[t] = variance;
        eps[t] = variance.sqrt() * rng.sample_innovation(1.0, innov_dist, innov_param);
        returns[t] = mu + eps[t];
    }

    out[..length].copy_from_slice(&returns[burn_in..burn_in + length]);
}

// ---------------------------------------------------------------------------
// Hawkes (self-exciting) point process
// kernel_type: 0 = exponential decay, 1 = power-law decay
// output_type: 0 = counts (binned), 1 = intensity, 2 = events (inter-arrival)
// ---------------------------------------------------------------------------
pub fn hawkes_process(
    out: &mut [f64],
    baseline_intensity: f64,
    excitation_amplitude: f64,
    decay_rate: f64,
    kernel_type: i32,
    power_law_exponent: f64,
    output_type: i32,
    max_events: i32,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);
    let length = out.len();
    let t_max = length as f64;
    let mut event_times: Vec<f64> = Vec::with_capacity(max_events as usize);

    let mut t = 0.0;

    if kernel_type == 0 {
        // === Exponential kernel: O(1) per intensity evaluation ===
        let mut a = 0.0_f64; // Aggregate excitation state
        let mut last_time = 0.0_f64; // Time reference for decay

        // O(1) intensity: lambda(t) = baseline + alpha * A * exp(-beta*(t-last_time))
        let fast_intensity = |time: f64, a: f64, last_time: f64| -> f64 {
            let dt = time - last_time;
            baseline_intensity + excitation_amplitude * a * (-decay_rate * dt).exp()
        };

        // Ogata's thinning algorithm with O(1) intensity
        while t < t_max && (event_times.len() as i32) < max_events {
            let mut lambda_bar = fast_intensity(t, a, last_time);
            lambda_bar = lambda_bar.max(baseline_intensity) * 1.01 + 1e-10;

            let u1 = rng.exponential(lambda_bar);
            t += u1;

            if t >= t_max {
                break;
            }

            let lambda_t = fast_intensity(t, a, last_time);
            let u2 = rng.uniform(0.0, 1.0);
            if u2 <= lambda_t / lambda_bar {
                // Advance time (decay the aggregate)
                let dt = t - last_time;
                a *= (-decay_rate * dt).exp();
                last_time = t;
                a += 1.0;
                event_times.push(t);
            }
        }

        // Produce output
        if output_type == 0 {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            for &et in &event_times {
                let bin = et as i32;
                if bin >= 0 && (bin as usize) < length {
                    out[bin as usize] += 1.0;
                }
            }
        } else if output_type == 1 {
            // Intensity output: use O(1) recursive computation per bin
            let mut a_out = 0.0_f64;
            let mut t_ref = 0.0_f64;
            let mut ev_idx = 0_usize;
            for (i, v) in out.iter_mut().enumerate() {
                let mid = i as f64 + 0.5;
                while ev_idx < event_times.len() && event_times[ev_idx] <= mid {
                    let dt = event_times[ev_idx] - t_ref;
                    a_out = a_out * (-decay_rate * dt).exp() + 1.0;
                    t_ref = event_times[ev_idx];
                    ev_idx += 1;
                }
                let dt = mid - t_ref;
                *v = baseline_intensity + excitation_amplitude * a_out * (-decay_rate * dt).exp();
            }
        } else {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            for &et in &event_times {
                let bin = et as i32;
                if bin >= 0 && (bin as usize) < length {
                    out[bin as usize] = 1.0;
                }
            }
        }
    } else {
        // === Power-law kernel: original O(n_events) per evaluation ===
        // g(t) = alpha / (1 + beta*t)^p, matching the Python implementation
        // (synforecast/generators/hawkes_process.py::_power_law_kernel).
        // Events at exactly dt == 0 are included (g(0) = alpha) so the
        // thinning envelope right after an accepted event stays valid.
        let compute_intensity = |time: f64, events: &[f64]| -> f64 {
            let mut lambda = baseline_intensity;
            for &et in events {
                let dt = time - et;
                if dt < 0.0 {
                    break;
                }
                lambda += excitation_amplitude / (1.0 + decay_rate * dt).powf(power_law_exponent);
            }
            lambda
        };

        while t < t_max && (event_times.len() as i32) < max_events {
            let mut lambda_bar = compute_intensity(t, &event_times);
            lambda_bar = lambda_bar.max(baseline_intensity) * 1.01 + 1e-10;

            let u1 = rng.exponential(lambda_bar);
            t += u1;

            if t >= t_max {
                break;
            }

            let lambda_t = compute_intensity(t, &event_times);
            let u2 = rng.uniform(0.0, 1.0);
            if u2 <= lambda_t / lambda_bar {
                event_times.push(t);
            }
        }

        if output_type == 0 {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            for &et in &event_times {
                let bin = et as i32;
                if bin >= 0 && (bin as usize) < length {
                    out[bin as usize] += 1.0;
                }
            }
        } else if output_type == 1 {
            for (i, v) in out.iter_mut().enumerate() {
                let mid = i as f64 + 0.5;
                *v = compute_intensity(mid, &event_times);
            }
        } else {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            for &et in &event_times {
                let bin = et as i32;
                if bin >= 0 && (bin as usize) < length {
                    out[bin as usize] = 1.0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chaotic systems
// ---------------------------------------------------------------------------

/// Lorenz attractor with RK4 integration
fn chaotic_lorenz(
    out: &mut [f64],
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    observation_noise: f64,
    initial_perturbation: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);
    let length = out.len();
    let burn_in: usize = 1000;
    let total = length + burn_in;
    let steps_per_obs = (1.0 / dt) as i64;
    let steps_per_obs = steps_per_obs.clamp(1, 100000);
    let total_steps = total as i64 * steps_per_obs;

    let perturb_x = rng.standard_normal() * initial_perturbation;
    let perturb_y = rng.standard_normal() * initial_perturbation;
    let perturb_z = rng.standard_normal() * initial_perturbation;
    let mut x = 1.0 + perturb_x;
    let mut y = 1.0 + perturb_y;
    let mut z = 1.0 + perturb_z;

    let mut buf = vec![0.0_f64; total];
    let mut idx: usize = 0;

    for i in 0..total_steps {
        // RK4 integration
        let dx1 = sigma * (y - x);
        let dy1 = x * (rho - z) - y;
        let dz1 = x * y - beta * z;

        let x2 = x + 0.5 * dt * dx1;
        let y2 = y + 0.5 * dt * dy1;
        let z2 = z + 0.5 * dt * dz1;

        let dx2 = sigma * (y2 - x2);
        let dy2 = x2 * (rho - z2) - y2;
        let dz2 = x2 * y2 - beta * z2;

        let x3 = x + 0.5 * dt * dx2;
        let y3 = y + 0.5 * dt * dy2;
        let z3 = z + 0.5 * dt * dz2;

        let dx3 = sigma * (y3 - x3);
        let dy3 = x3 * (rho - z3) - y3;
        let dz3 = x3 * y3 - beta * z3;

        let x4 = x + dt * dx3;
        let y4 = y + dt * dy3;
        let z4 = z + dt * dz3;

        let dx4 = sigma * (y4 - x4);
        let dy4 = x4 * (rho - z4) - y4;
        let dz4 = x4 * y4 - beta * z4;

        x += dt / 6.0 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
        y += dt / 6.0 * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4);
        z += dt / 6.0 * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4);

        if (i + 1) % steps_per_obs == 0 && idx < total {
            buf[idx] = x;
            idx += 1;
        }
    }

    for i in 0..length {
        out[i] = buf[burn_in + i];
        if observation_noise > 0.0 {
            out[i] += rng.normal(0.0, observation_noise);
        }
    }
}

/// Logistic map
fn chaotic_logistic(
    out: &mut [f64],
    r: f64,
    observation_noise: f64,
    initial_perturbation: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);
    let length = out.len();
    let burn_in: usize = 500;
    let total = length + burn_in;

    let mut x = 0.5 + rng.standard_normal() * initial_perturbation;
    x = x.clamp(0.01, 0.99);

    let mut buf = vec![0.0_f64; total];
    for v in buf.iter_mut().take(total) {
        x = r * x * (1.0 - x);
        *v = x;
    }

    for i in 0..length {
        out[i] = buf[burn_in + i];
        if observation_noise > 0.0 {
            out[i] += rng.normal(0.0, observation_noise);
        }
    }
}

/// Mackey-Glass delay differential equation
fn chaotic_mackey_glass(
    out: &mut [f64],
    mg_beta: f64,
    mg_gamma: f64,
    mg_n: f64,
    mg_tau: i32,
    observation_noise: f64,
    initial_perturbation: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);
    let length = out.len();
    let burn_in: usize = 500;
    let total = length + burn_in;
    let history_len = (mg_tau + 1) as usize;

    let init_val = 1.2 + rng.standard_normal() * initial_perturbation;
    let mut history = vec![init_val; history_len];

    let mut buf = vec![0.0_f64; total];
    let mut x = init_val;

    for i in 0..total {
        let x_tau = history[i % history_len];
        let dx = mg_beta * x_tau / (1.0 + x_tau.powf(mg_n)) - mg_gamma * x;
        x += dx;
        buf[i] = x;
        history[(i + mg_tau as usize + 1) % history_len] = x;
    }

    for i in 0..length {
        out[i] = buf[burn_in + i];
        if observation_noise > 0.0 {
            out[i] += rng.normal(0.0, observation_noise);
        }
    }
}

/// Chaotic system wrapper: system_id 0=lorenz, 1=logistic, 2=mackey_glass
pub fn chaotic_system(
    out: &mut [f64],
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
) {
    match system_id {
        1 => chaotic_logistic(
            out,
            logistic_r,
            observation_noise,
            initial_perturbation,
            seed,
        ),
        2 => chaotic_mackey_glass(
            out,
            mg_beta,
            mg_gamma,
            mg_n,
            mg_tau,
            observation_noise,
            initial_perturbation,
            seed,
        ),
        // 0 and default
        _ => chaotic_lorenz(
            out,
            sigma,
            rho,
            beta,
            dt,
            observation_noise,
            initial_perturbation,
            seed,
        ),
    }
}

// ---------------------------------------------------------------------------
// Bounded process
// model_id: 0=beta_ar, 1=logit_normal
// ---------------------------------------------------------------------------

#[inline]
fn logit(x: f64) -> f64 {
    let x = x.clamp(1e-10, 1.0 - 1e-10);
    (x / (1.0 - x)).ln()
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    let x = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-x).exp())
}

#[allow(clippy::too_many_arguments)]
pub fn bounded_process(
    out: &mut [f64],
    model_id: i32,
    phi: f64,
    omega: f64,
    kappa: f64,
    sigma_param: f64,
    initial_value: f64,
    lower: f64,
    upper: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);
    let span = upper - lower;

    if model_id == 0 {
        // Beta-AR(1)
        let mut x = initial_value;
        for v in out.iter_mut() {
            let mu = omega + phi * x;
            let mu = mu.clamp(1e-6, 1.0 - 1e-6);
            let a = mu * kappa;
            let b = (1.0 - mu) * kappa;
            // Beta(a, b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
            let ga = rng.gamma(a, 1.0);
            let gb = rng.gamma(b, 1.0);
            x = ga / (ga + gb);
            x = x.clamp(1e-10, 1.0 - 1e-10);
            // Affine map from the unit interval to [lower, upper]
            *v = lower + x * span;
        }
    } else {
        // Logit-normal random walk
        let mut z = logit(initial_value);
        for v in out.iter_mut() {
            z = phi * z + sigma_param * rng.standard_normal();
            *v = lower + sigmoid(z) * span;
        }
    }
}

// ---------------------------------------------------------------------------
// Levy process (alpha-stable increments)
// ---------------------------------------------------------------------------

/// Chambers-Mallows-Stuck algorithm for alpha-stable random variables
fn sample_stable(rng: &mut SfRng, alpha: f64, beta: f64) -> f64 {
    if (alpha - 2.0).abs() < 1e-10 {
        // Gaussian case
        return rng.standard_normal() * 2.0_f64.sqrt();
    }

    // Uniform on (-pi/2, pi/2)
    let v = rng.uniform(-PI / 2.0, PI / 2.0);
    // Standard exponential
    let w = rng.exponential(1.0);

    if (alpha - 1.0).abs() < 1e-10 {
        // Cauchy-like case
        let b_term = (PI / 2.0 + beta * v) * v.tan();
        let log_arg = (PI / 2.0) * w * v.cos() / (PI / 2.0 + beta * v);
        return (1.0 / (PI / 2.0)) * (b_term - beta * log_arg.max(1e-300).ln());
    }

    // General case
    let b_alpha = (beta * (PI * alpha / 2.0).tan()).atan() / alpha;
    let s_alpha = (1.0 + beta * beta * (PI * alpha / 2.0).tan().powi(2)).powf(1.0 / (2.0 * alpha));

    let big_b = alpha * (v + b_alpha);
    let numerator = big_b.sin();
    let denom = v.cos().powf(1.0 / alpha);
    let factor = ((v - big_b).cos() / w).powf((1.0 - alpha) / alpha);

    s_alpha * numerator / denom * factor
}

pub fn levy_process(
    out: &mut [f64],
    alpha: f64,
    beta_skew: f64,
    scale: f64,
    location: f64,
    cumulative: bool,
    initial_value: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);

    if cumulative {
        let mut cumsum = initial_value;
        for v in out.iter_mut() {
            let increment = scale * sample_stable(&mut rng, alpha, beta_skew) + location;
            cumsum += increment;
            *v = cumsum;
        }
    } else {
        for v in out.iter_mut() {
            *v = scale * sample_stable(&mut rng, alpha, beta_skew) + location;
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

    // --- Ornstein-Uhlenbeck ---

    #[test]
    fn test_ornstein_uhlenbeck_finite() {
        let mut out = vec![0.0; N];
        ornstein_uhlenbeck(&mut out, 1.0, 0.0, 0.5, 0.0, 0.01, 42, 0, 0.0);
        assert_finite(&out, "ou");
    }

    #[test]
    fn test_ornstein_uhlenbeck_mean_reversion() {
        let mut out = vec![0.0; 1000];
        ornstein_uhlenbeck(&mut out, 5.0, 10.0, 0.5, 0.0, 0.01, 42, 0, 0.0);
        // Strong mean reversion (theta=5) towards mu=10; tail should be near 10
        let tail_mean: f64 = out[800..].iter().sum::<f64>() / 200.0;
        assert!(
            (tail_mean - 10.0).abs() < 3.0,
            "OU tail mean should be near mu=10, got {tail_mean}"
        );
    }

    #[test]
    fn test_ornstein_uhlenbeck_deterministic() {
        assert_deterministic(
            |o| ornstein_uhlenbeck(o, 1.0, 0.0, 0.5, 0.0, 0.01, 42, 0, 0.0),
            "ou",
        );
    }

    // --- Geometric Brownian Motion ---

    #[test]
    fn test_gbm_positive() {
        let mut out = vec![0.0; N];
        geometric_brownian_motion(&mut out, 0.05, 0.2, 100.0, 0.01, 42, 0, 0.0);
        assert_finite(&out, "gbm");
        for (i, &v) in out.iter().enumerate() {
            assert!(v > 0.0, "GBM should be positive, got {v} at {i}");
        }
    }

    #[test]
    fn test_gbm_initial_value() {
        let mut out = vec![0.0; 10];
        geometric_brownian_motion(&mut out, 0.0, 0.0001, 50.0, 0.01, 42, 0, 0.0);
        assert!((out[0] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_gbm_deterministic() {
        assert_deterministic(
            |o| geometric_brownian_motion(o, 0.05, 0.2, 100.0, 0.01, 42, 0, 0.0),
            "gbm",
        );
    }

    // --- Jump Diffusion ---

    #[test]
    fn test_jump_diffusion_positive() {
        let mut out = vec![0.0; N];
        jump_diffusion(&mut out, 0.05, 0.2, 0.5, 0.0, 0.1, 100.0, 0.01, 42, 0, 0.0);
        assert_finite(&out, "jump_diffusion");
        for &v in &out {
            assert!(v > 0.0, "jump diffusion should be positive");
        }
    }

    #[test]
    fn test_jump_diffusion_deterministic() {
        assert_deterministic(
            |o| jump_diffusion(o, 0.05, 0.2, 0.5, 0.0, 0.1, 100.0, 0.01, 42, 0, 0.0),
            "jump_diffusion",
        );
    }

    // --- Poisson Process ---

    #[test]
    fn test_poisson_process_nonnegative() {
        let mut out = vec![0.0; N];
        poisson_process(&mut out, 2.0, false, 42);
        assert_finite(&out, "poisson");
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_poisson_process_cumulative_monotone() {
        let mut out = vec![0.0; N];
        poisson_process(&mut out, 2.0, true, 42);
        assert_finite(&out, "poisson_cumulative");
        for i in 1..out.len() {
            assert!(
                out[i] >= out[i - 1],
                "cumulative poisson should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_poisson_process_deterministic() {
        assert_deterministic(|o| poisson_process(o, 2.0, false, 42), "poisson");
    }

    // --- Cyclic ---

    #[test]
    fn test_cyclic_finite() {
        let mut out = vec![0.0; N];
        cyclic(&mut out, 0.0, 0.0, 50.0, 5.0, 10.0, 2.0, 3, 0.5, 42);
        assert_finite(&out, "cyclic");
    }

    #[test]
    fn test_cyclic_deterministic() {
        assert_deterministic(
            |o| cyclic(o, 0.0, 0.0, 50.0, 5.0, 10.0, 2.0, 3, 0.5, 42),
            "cyclic",
        );
    }

    // --- GARCH ---

    #[test]
    fn test_garch_finite() {
        let alpha = [0.1];
        let beta = [0.8];
        let mut out = vec![0.0; N];
        garch(&mut out, 1, 1, 0.01, &alpha, &beta, 0.0, 0.01, 42, 0, 0.0);
        assert_finite(&out, "garch");
    }

    #[test]
    fn test_garch_deterministic() {
        let alpha = [0.1];
        let beta = [0.8];
        assert_deterministic(
            |o| garch(o, 1, 1, 0.01, &alpha, &beta, 0.0, 0.01, 42, 0, 0.0),
            "garch",
        );
    }

    // --- Hawkes Process ---

    #[test]
    fn test_hawkes_exponential_counts() {
        let mut out = vec![0.0; 100];
        hawkes_process(&mut out, 1.0, 0.5, 1.0, 0, 1.0, 0, 1000, 42);
        assert_finite(&out, "hawkes_exp_counts");
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_hawkes_exponential_intensity() {
        let mut out = vec![0.0; 100];
        hawkes_process(&mut out, 1.0, 0.5, 1.0, 0, 1.0, 1, 1000, 42);
        assert_finite(&out, "hawkes_exp_intensity");
        for &v in &out {
            assert!(v >= 0.0, "intensity should be non-negative");
        }
    }

    #[test]
    fn test_hawkes_power_law() {
        let mut out = vec![0.0; 100];
        hawkes_process(&mut out, 1.0, 0.5, 1.0, 1, 1.5, 0, 1000, 42);
        assert_finite(&out, "hawkes_power");
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_hawkes_deterministic() {
        assert_deterministic(
            |o| hawkes_process(o, 1.0, 0.5, 1.0, 0, 1.0, 0, 1000, 42),
            "hawkes",
        );
    }

    // --- Chaotic System ---

    #[test]
    fn test_chaotic_lorenz_finite() {
        let mut out = vec![0.0; N];
        chaotic_system(
            &mut out,
            0,
            10.0,
            28.0,
            8.0 / 3.0,
            0.01,
            3.9,
            0.2,
            0.1,
            10.0,
            17,
            0.0,
            0.01,
            42,
        );
        assert_finite(&out, "lorenz");
    }

    #[test]
    fn test_chaotic_logistic_bounded() {
        let mut out = vec![0.0; N];
        chaotic_system(
            &mut out,
            1,
            10.0,
            28.0,
            8.0 / 3.0,
            0.01,
            3.9,
            0.2,
            0.1,
            10.0,
            17,
            0.0,
            0.001,
            42,
        );
        assert_finite(&out, "logistic");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "logistic map should be in [0,1], got {v} at {i}"
            );
        }
    }

    #[test]
    fn test_chaotic_mackey_glass_finite() {
        let mut out = vec![0.0; N];
        chaotic_system(
            &mut out,
            2,
            10.0,
            28.0,
            8.0 / 3.0,
            0.01,
            3.9,
            0.2,
            0.1,
            10.0,
            17,
            0.0,
            0.01,
            42,
        );
        assert_finite(&out, "mackey_glass");
    }

    #[test]
    fn test_chaotic_deterministic() {
        assert_deterministic(
            |o| {
                chaotic_system(
                    o,
                    0,
                    10.0,
                    28.0,
                    8.0 / 3.0,
                    0.01,
                    3.9,
                    0.2,
                    0.1,
                    10.0,
                    17,
                    0.0,
                    0.01,
                    42,
                )
            },
            "chaotic",
        );
    }

    // --- Bounded Process ---

    #[test]
    fn test_bounded_beta_ar_in_unit_interval() {
        let mut out = vec![0.0; N];
        bounded_process(&mut out, 0, 0.5, 0.2, 10.0, 0.1, 0.5, 0.0, 1.0, 42);
        assert_finite(&out, "beta_ar");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v > 0.0 && v < 1.0,
                "beta_ar should be in (0,1), got {v} at {i}"
            );
        }
    }

    #[test]
    fn test_bounded_logit_normal_in_unit_interval() {
        let mut out = vec![0.0; N];
        bounded_process(&mut out, 1, 0.9, 0.0, 10.0, 0.5, 0.5, 0.0, 1.0, 42);
        assert_finite(&out, "logit_normal");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v > 0.0 && v < 1.0,
                "logit_normal should be in (0,1), got {v} at {i}"
            );
        }
    }

    #[test]
    fn test_bounded_custom_bounds() {
        let mut out = vec![0.0; N];
        bounded_process(&mut out, 0, 0.5, 0.2, 10.0, 0.1, 0.5, 10.0, 20.0, 42);
        assert_finite(&out, "beta_ar_bounds");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v > 10.0 && v < 20.0,
                "beta_ar should be in (10,20), got {v} at {i}"
            );
        }
    }

    #[test]
    fn test_bounded_deterministic() {
        assert_deterministic(
            |o| bounded_process(o, 0, 0.5, 0.2, 10.0, 0.1, 0.5, 0.0, 1.0, 42),
            "bounded",
        );
    }

    // --- Levy Process ---

    #[test]
    fn test_levy_gaussian_case() {
        // alpha=2 should give Gaussian increments
        let mut out = vec![0.0; N];
        levy_process(&mut out, 2.0, 0.0, 1.0, 0.0, false, 0.0, 42);
        assert_finite(&out, "levy_gaussian");
    }

    #[test]
    fn test_levy_cumulative_uses_initial_value() {
        let mut out = vec![0.0; N];
        levy_process(&mut out, 2.0, 0.0, 0.001, 0.0, true, 100.0, 42);
        // With tiny scale, cumulative sum should stay near initial value
        assert!(
            (out[0] - 100.0).abs() < 5.0,
            "first cumulative value should be near 100, got {}",
            out[0]
        );
    }

    #[test]
    fn test_levy_cauchy_case() {
        // alpha=1 (Cauchy-like)
        let mut out = vec![0.0; N];
        levy_process(&mut out, 1.0, 0.0, 1.0, 0.0, false, 0.0, 42);
        // Cauchy has heavy tails; just check it runs without panic
        // Some values may be very large but should be finite
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_levy_deterministic() {
        assert_deterministic(
            |o| levy_process(o, 1.5, 0.0, 1.0, 0.0, false, 0.0, 42),
            "levy",
        );
    }
}
