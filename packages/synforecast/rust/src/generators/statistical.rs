use crate::rng::SfRng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Random Walk
// ---------------------------------------------------------------------------

pub fn random_walk(
    out: &mut [f64],
    drift: f64,
    volatility: f64,
    start_value: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let mut rng = SfRng::new(seed);
    let mut cumsum = start_value;
    for v in out.iter_mut() {
        let step = drift + rng.sample_innovation(volatility, innov_dist, innov_param);
        cumsum += step;
        *v = cumsum;
    }
}

// ---------------------------------------------------------------------------
// Seasonal
// ---------------------------------------------------------------------------

pub fn seasonal(
    out: &mut [f64],
    seasonality_period: i32,
    seasonality_amplitude: f64,
    trend: f64,
    noise_level: f64,
    base_level: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let mut rng = SfRng::new(seed);
    let two_pi_over_period = 2.0 * PI / seasonality_period as f64;
    for (t, v) in out.iter_mut().enumerate() {
        let seasonal_val = seasonality_amplitude * (two_pi_over_period * t as f64).sin();
        let trend_val = trend * t as f64;
        let noise = rng.sample_innovation(noise_level, innov_dist, innov_param);
        *v = base_level + seasonal_val + trend_val + noise;
    }
}

// ---------------------------------------------------------------------------
// SARIMA
// ---------------------------------------------------------------------------

pub fn sarima(
    out: &mut [f64],
    full_ar_poly: &[f64],
    full_ma_poly: &[f64],
    d: i32,
    d_seasonal: i32,
    seasonal_period: i32,
    mean: f64,
    drift_val: f64,
    noise_std: f64,
    burn_in: i32,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let mut rng = SfRng::new(seed);
    let length = out.len() as i32;
    let ar_len = full_ar_poly.len();
    let ma_len = full_ma_poly.len();

    // Compute total length needed including burn-in and differencing expansion
    let diff_expand = d + d_seasonal * seasonal_period;
    let total = (length + burn_in + diff_expand) as usize;

    // Generate innovations
    let mut innov = vec![0.0_f64; total];
    rng.sample_innovations(&mut innov, noise_std, innov_dist, innov_param);

    // MA filter: u[t] = innov[t] + sum(ma_poly[i] * innov[t-i])
    let mut u = vec![0.0_f64; total];
    for t in 0..total {
        u[t] = innov[t];
        for i in 1..=ma_len {
            if t >= i {
                u[t] += full_ma_poly[i - 1] * innov[t - i];
            }
        }
    }

    // AR filter: y[t] = sum(ar_poly[i] * y[t-i]) + u[t]
    let mut y = vec![0.0_f64; total];
    for t in 0..total {
        y[t] = u[t];
        for i in 1..=ar_len {
            if t >= i {
                y[t] += full_ar_poly[i - 1] * y[t - i];
            }
        }
    }

    // Match the Python convention: stationary models (d = D = 0) get the
    // process mean; integrated models get a constant drift added to the
    // differenced series, which integrates to slope `drift_val` per step.
    if d > 0 || d_seasonal > 0 {
        for y_t in y.iter_mut() {
            *y_t += drift_val;
        }
    } else {
        for y_t in y.iter_mut() {
            *y_t += mean;
        }
    }

    // Inverse differencing: cumsum d times (non-seasonal)
    for _dd in 0..d {
        for t in 1..total {
            y[t] += y[t - 1];
        }
    }

    // Inverse seasonal differencing: seasonal cumsum D times
    let sp = seasonal_period as usize;
    for _dd in 0..d_seasonal {
        for t in sp..total {
            y[t] += y[t - sp];
        }
    }

    // Extract final length from total (skip burn_in + diff_expand)
    let offset = (burn_in + diff_expand) as usize;
    out.copy_from_slice(&y[offset..offset + out.len()]);
}

// ---------------------------------------------------------------------------
// ETS (Error-Trend-Seasonal state-space model)
// ---------------------------------------------------------------------------

const ETS_MIN_LEVEL: f64 = 1e-6;
const ETS_MAX_LEVEL: f64 = 1e12;
const ETS_MIN_SEASONAL_MUL: f64 = 0.01;
const ETS_MAX_SEASONAL_MUL: f64 = 100.0;
const ETS_MAX_TREND_ADD: f64 = 1e6;
const ETS_MIN_TREND_MUL: f64 = 0.01;
const ETS_MAX_TREND_MUL: f64 = 100.0;

// error_type: 0=additive, 1=multiplicative
// trend_type: 0=none, 1=additive, 2=multiplicative
// seasonal_type: 0=none, 1=additive, 2=multiplicative

pub fn ets(
    out: &mut [f64],
    error_type: i32,
    trend_type: i32,
    seasonal_type: i32,
    seasonal_period: i32,
    level: f64,
    trend_init: f64,
    seasonal_init: &[f64],
    alpha: f64,
    beta_param: f64,
    gamma: f64,
    phi: f64,
    damped: bool,
    noise_std: f64,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let mut rng = SfRng::new(seed);

    let mut l = level;
    let mut b = trend_init;

    // Initialize seasonal components
    let sp = seasonal_period as usize;
    let mut s = vec![0.0_f64; sp];
    if seasonal_type != 0 && !seasonal_init.is_empty() {
        s[..sp].copy_from_slice(&seasonal_init[..sp]);
    } else if seasonal_type == 2 {
        // Default multiplicative seasonal to 1.0
        for v in s.iter_mut() {
            *v = 1.0;
        }
    }

    let phi_eff = if damped { phi } else { 1.0 };

    for (t, out_v) in out.iter_mut().enumerate() {
        let s_idx = t % sp;
        let s_t = s[s_idx];

        // Compute forecast (y_hat)
        let y_hat = match (trend_type, seasonal_type) {
            (0, 0) => l,                         // N,N
            (1, 0) => l + phi_eff * b,           // A,N
            (2, 0) => l * b.powf(phi_eff),       // M,N
            (0, 1) => l + s_t,                   // N,A
            (1, 1) => l + phi_eff * b + s_t,     // A,A
            (2, 1) => l * b.powf(phi_eff) + s_t, // M,A
            (0, 2) => l * s_t,                   // N,M
            (1, 2) => (l + phi_eff * b) * s_t,   // A,M
            (2, 2) => l * b.powf(phi_eff) * s_t, // M,M
            _ => l,
        };

        // Generate error
        let eps = rng.sample_innovation(noise_std, innov_dist, innov_param);

        // Compute observed value
        let y = if error_type == 0 {
            (y_hat + eps).clamp(-ETS_MAX_LEVEL, ETS_MAX_LEVEL)
        } else {
            (y_hat * (1.0 + eps)).clamp(ETS_MIN_LEVEL, ETS_MAX_LEVEL)
        };
        *out_v = y;

        // Update states per Hyndman et al. (2008), Tables 2.2/2.3: all 30
        // model variants reduce to one scheme in terms of the additive
        // one-step error e_t = y_t - mu_t (= mu_t * eps for multiplicative
        // error). Mirrors ETSGenerator._update_state in
        // synforecast/generators/ets.py exactly.
        let l_old = l;
        let b_old = b;

        let trend_base = match trend_type {
            1 => l_old + phi_eff * b_old,
            2 => l_old * b_old.powf(phi_eff),
            _ => l_old,
        };

        let e = if error_type == 0 { eps } else { y_hat * eps };

        let s_div = if seasonal_type == 2 && s_t != 0.0 {
            s_t
        } else {
            1.0
        };

        // Level update
        l = trend_base + alpha * e / s_div;
        l = if error_type == 0 {
            l.clamp(-ETS_MAX_LEVEL, ETS_MAX_LEVEL)
        } else {
            l.clamp(ETS_MIN_LEVEL, ETS_MAX_LEVEL)
        };

        // Trend update
        if trend_type == 1 {
            b = phi_eff * b_old + beta_param * e / s_div;
            b = b.clamp(-ETS_MAX_TREND_ADD, ETS_MAX_TREND_ADD);
        } else if trend_type == 2 {
            let denom = s_div * l_old;
            b = b_old.powf(phi_eff)
                + if denom != 0.0 {
                    beta_param * e / denom
                } else {
                    0.0
                };
            b = b.clamp(ETS_MIN_TREND_MUL, ETS_MAX_TREND_MUL);
        }

        // Seasonal update
        if seasonal_type == 1 {
            s[s_idx] = s_t + gamma * e;
        } else if seasonal_type == 2 {
            s[s_idx] = s_t
                + if trend_base != 0.0 {
                    gamma * e / trend_base
                } else {
                    0.0
                };
            s[s_idx] = s[s_idx].clamp(ETS_MIN_SEASONAL_MUL, ETS_MAX_SEASONAL_MUL);
        }
    }
}

// ---------------------------------------------------------------------------
// INAR(p) with binomial thinning
// ---------------------------------------------------------------------------

// innov_type: 0=poisson, 1=negative_binomial

pub fn inar(
    out: &mut [f64],
    p: i32,
    alpha: &[f64],
    innov_type: i32,
    innov_mean: f64,
    innov_dispersion: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);
    let length = out.len() as i32;
    let burn_in: i32 = 100;
    let total = (length + burn_in) as usize;
    let p = p as usize;

    // Compute stationary mean for initialization
    let mut alpha_sum = 0.0_f64;
    for &a in alpha.iter().take(p) {
        alpha_sum += a;
    }
    let stationary_mean = if alpha_sum < 1.0 {
        (innov_mean / (1.0 - alpha_sum)).max(0.0) as i32
    } else {
        innov_mean.max(0.0) as i32
    };

    let mut values = vec![stationary_mean; total];

    for t in p..total {
        let mut thinned: i32 = 0;
        for i in 0..p {
            let count = values[t - i - 1];
            if count > 0 && alpha[i] > 0.0 {
                thinned += rng.binomial(count, alpha[i]);
            }
        }

        // Sample innovation
        let innovation = if innov_type == 0 {
            rng.poisson(innov_mean)
        } else {
            let r = innov_dispersion;
            let prob = r / (r + innov_mean);
            let lambda = rng.gamma(r, (1.0 - prob) / prob);
            rng.poisson(lambda.max(0.0))
        };

        values[t] = thinned + innovation;
    }

    let burn_in = burn_in as usize;
    for i in 0..out.len() {
        out[i] = values[burn_in + i] as f64;
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

    #[test]
    fn test_random_walk_finite_and_deterministic() {
        let mut out = vec![0.0; N];
        random_walk(&mut out, 0.0, 1.0, 0.0, 42, 0, 0.0);
        assert_finite(&out, "random_walk");
        assert_deterministic(|o| random_walk(o, 0.0, 1.0, 0.0, 42, 0, 0.0), "random_walk");
    }

    #[test]
    fn test_random_walk_with_drift() {
        let mut out = vec![0.0; 500];
        random_walk(&mut out, 1.0, 0.01, 0.0, 42, 0, 0.0);
        // With drift=1.0 and low volatility, last value should be positive
        assert!(out[499] > 0.0, "drift should push values positive");
    }

    #[test]
    fn test_random_walk_start_value() {
        let mut out = vec![0.0; 10];
        random_walk(&mut out, 0.0, 0.0001, 100.0, 42, 0, 0.0);
        // With near-zero volatility, values should be close to start_value
        for v in &out {
            assert!((v - 100.0).abs() < 5.0);
        }
    }

    #[test]
    fn test_seasonal_has_periodic_structure() {
        let mut out = vec![0.0; 200];
        seasonal(&mut out, 20, 10.0, 0.0, 0.001, 0.0, 42, 0, 0.0);
        assert_finite(&out, "seasonal");
        // Check that values oscillate (not monotone)
        let mut sign_changes = 0;
        for i in 1..out.len() {
            if out[i] * out[i - 1] < 0.0 {
                sign_changes += 1;
            }
        }
        assert!(
            sign_changes > 5,
            "seasonal should oscillate, got {sign_changes} sign changes"
        );
    }

    #[test]
    fn test_seasonal_deterministic() {
        assert_deterministic(
            |o| seasonal(o, 12, 5.0, 0.1, 1.0, 10.0, 42, 0, 0.0),
            "seasonal",
        );
    }

    #[test]
    fn test_seasonal_t_innovations_heavier_tails() {
        // With amplitude/trend disabled the series is pure noise; t(df=3)
        // innovations must have visibly heavier tails than normal ones.
        fn excess_kurtosis(x: &[f64]) -> f64 {
            let n = x.len() as f64;
            let mean = x.iter().sum::<f64>() / n;
            let m2 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let m4 = x.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
            m4 / (m2 * m2) - 3.0
        }
        let n = 20_000;
        let mut normal = vec![0.0; n];
        let mut heavy = vec![0.0; n];
        seasonal(&mut normal, 24, 0.0, 0.0, 1.0, 0.0, 42, 0, 0.0);
        seasonal(&mut heavy, 24, 0.0, 0.0, 1.0, 0.0, 42, 1, 3.0);
        let k_normal = excess_kurtosis(&normal);
        let k_heavy = excess_kurtosis(&heavy);
        assert!(
            k_heavy > k_normal + 1.0,
            "t(3) innovations should be heavier-tailed: normal={k_normal}, t={k_heavy}"
        );
    }

    #[test]
    fn test_sarima_finite_and_deterministic() {
        let ar = [0.5];
        let ma = [0.3];
        let mut out = vec![0.0; N];
        sarima(&mut out, &ar, &ma, 1, 0, 12, 0.0, 0.0, 1.0, 50, 42, 0, 0.0);
        assert_finite(&out, "sarima");
        assert_deterministic(
            |o| sarima(o, &ar, &ma, 1, 0, 12, 0.0, 0.0, 1.0, 50, 42, 0, 0.0),
            "sarima",
        );
    }

    #[test]
    fn test_ets_additive_finite() {
        let seasonal_init = vec![0.0; 12];
        let mut out = vec![0.0; N];
        ets(
            &mut out,
            0,
            1,
            1,
            12,
            100.0,
            1.0,
            &seasonal_init,
            0.3,
            0.1,
            0.1,
            1.0,
            false,
            1.0,
            42,
            0,
            0.0,
        );
        assert_finite(&out, "ets_additive");
    }

    #[test]
    fn test_ets_multiplicative_finite() {
        let seasonal_init = vec![1.0; 12];
        let mut out = vec![0.0; N];
        ets(
            &mut out,
            1,
            2,
            2,
            12,
            100.0,
            1.01,
            &seasonal_init,
            0.2,
            0.05,
            0.05,
            0.98,
            true,
            0.01,
            42,
            0,
            0.0,
        );
        assert_finite(&out, "ets_multiplicative");
    }

    #[test]
    fn test_ets_deterministic() {
        let si = vec![0.0; 12];
        assert_deterministic(
            |o| {
                ets(
                    o, 0, 1, 1, 12, 100.0, 1.0, &si, 0.3, 0.1, 0.1, 1.0, false, 1.0, 42, 0, 0.0,
                )
            },
            "ets",
        );
    }

    #[test]
    fn test_inar_nonnegative_integers() {
        let alpha = [0.3, 0.2];
        let mut out = vec![0.0; N];
        inar(&mut out, 2, &alpha, 0, 5.0, 1.0, 42);
        assert_finite(&out, "inar");
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "inar should be non-negative, got {v} at {i}");
            assert!(
                (v - v.round()).abs() < 1e-10,
                "inar should produce integers"
            );
        }
    }

    #[test]
    fn test_inar_deterministic() {
        let alpha = [0.5];
        assert_deterministic(|o| inar(o, 1, &alpha, 0, 3.0, 1.0, 42), "inar");
    }

    #[test]
    fn test_inar_negative_binomial_innovation() {
        let alpha = [0.3];
        let mut out = vec![0.0; N];
        inar(&mut out, 1, &alpha, 1, 5.0, 2.0, 42);
        assert_finite(&out, "inar_negbin");
        for &v in &out {
            assert!(v >= 0.0);
        }
    }
}
