use crate::rng::SfRng;

/// Stochastic volatility models: Heston and SABR
///
/// model_type: 0 = Heston, 1 = SABR
/// output_type: 0 = price, 1 = returns, 2 = volatility
///
/// Heston model:
///   dS = mu*S*dt + sqrt(V)*S*dW1
///   dV = kappa*(theta-V)*dt + sigma_v*sqrt(V)*dW2
///   Corr(dW1, dW2) = rho
///   Full truncation: use max(V, 0) in drift and diffusion
///
/// SABR model:
///   dF = sigma * F^beta * dW1
///   dsigma = alpha * sigma * dW2
///   Corr(dW1, dW2) = rho
pub fn stochastic_volatility(
    out: &mut [f64],
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
) {
    let length = out.len();
    let mut rng = SfRng::new(seed);
    let sqrt_dt = dt.sqrt();

    let mut prices = vec![0.0_f64; length];
    let mut vols = vec![0.0_f64; length];

    prices[0] = initial_price;
    // For SABR, initial_vol is variance; store sqrt for vol dynamics (matching Python)
    vols[0] = if model_type == 1 {
        initial_vol.sqrt()
    } else {
        initial_vol
    };

    for t in 1..length {
        // Generate correlated Brownian motions
        let z1 = rng.sample_innovation(1.0, innov_dist, innov_param);
        let z2 = rng.sample_innovation(1.0, innov_dist, innov_param);
        let dw1 = z1 * sqrt_dt;
        let dw2 = (correlation * z1 + (1.0 - correlation * correlation).sqrt() * z2) * sqrt_dt;

        if model_type == 0 {
            // Heston model with full truncation scheme
            let v = vols[t - 1];
            let v_pos = v.max(0.0);
            let s = prices[t - 1];

            // Price dynamics: dS = mu*S*dt + sqrt(V+)*S*dW1
            let ds = drift * s * dt + v_pos.sqrt() * s * dw1;
            prices[t] = s + ds;

            // Variance dynamics: dV = kappa*(theta-V+)*dt + sigma_v*sqrt(V+)*dW2
            let dv = vol_mean_reversion * (mean_vol - v_pos) * dt + vol_of_vol * v_pos.sqrt() * dw2;
            vols[t] = v + dv;
        } else {
            // SABR model
            let f = prices[t - 1];
            let sigma = vols[t - 1];
            let sigma_pos = sigma.max(1e-10);
            let f_abs = f.abs().max(1e-10);

            // Forward dynamics: dF = sigma * F^beta * dW1
            let df = sigma_pos * f_abs.powf(beta_param) * dw1;
            prices[t] = f + df;

            // Vol dynamics: dsigma = alpha * sigma * dW2
            let dsigma = vol_of_vol * sigma_pos * dw2;
            vols[t] = sigma + dsigma;
        }
    }

    // Write output based on output_type
    if output_type == 0 {
        // Price
        out[..length].copy_from_slice(&prices[..length]);
    } else if output_type == 1 {
        // Returns (log returns)
        out[0] = 0.0;
        for i in 1..length {
            if prices[i] > 0.0 && prices[i - 1] > 0.0 {
                out[i] = (prices[i] / prices[i - 1]).ln();
            } else {
                out[i] = 0.0;
            }
        }
    } else {
        // Volatility: return sqrt of variance (matching Python)
        // For Heston, vols[] stores variance; for SABR, vols[] stores vol
        for i in 0..length {
            out[i] = if model_type == 0 {
                vols[i].max(0.0).sqrt()
            } else {
                vols[i] // SABR: vols already stores volatility
            };
        }
    }
}

/// Markov regime-switching AR(1) model.
///
/// transition_matrix is row-major: transition_matrix[i * n_regimes + j] =
///   probability of transitioning from regime i to regime j.
///
/// `initial_regime >= 0` fixes the starting regime; a negative value draws
/// s_0 from `stationary_probs` using this series' RNG (matching the Python
/// path, which draws a fresh initial regime per series from the stationary
/// distribution of the transition matrix).
///
/// For each timestep:
///   1. Sample next regime from transition_matrix[current_regime]
///   2. Generate value: values[t] = mu + phi*(values[t-1] - mu) + sigma*N(0,1)
#[allow(clippy::too_many_arguments)]
pub fn regime_switching(
    out: &mut [f64],
    n_regimes: i32,
    regime_means: &[f64],
    regime_variances: &[f64],
    regime_ar_coeffs: &[f64],
    transition_matrix: &[f64],
    stationary_probs: &[f64],
    initial_regime: i32,
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let length = out.len();
    let mut rng = SfRng::new(seed);
    let n = n_regimes as usize;

    let mut current_regime = if initial_regime >= 0 {
        initial_regime as usize
    } else {
        // Inverse-CDF draw of s_0 from the stationary distribution, so each
        // series in a batch gets its own independent initial regime.
        let u = rng.uniform01();
        let mut cumprob = 0.0;
        let mut regime = n - 1; // fallback to last regime
        for (j, &p) in stationary_probs.iter().take(n).enumerate() {
            cumprob += p;
            if u <= cumprob {
                regime = j;
                break;
            }
        }
        regime
    };

    // Initialize first value from initial regime
    let mu = regime_means[current_regime];
    let sigma = regime_variances[current_regime].sqrt();
    out[0] = mu + sigma * rng.sample_innovation(1.0, innov_dist, innov_param);

    for t in 1..length {
        // Sample next regime from transition probabilities
        let u = rng.uniform(0.0, 1.0);
        let mut cumprob = 0.0;
        let mut next_regime = n - 1; // fallback to last regime
        for j in 0..n {
            cumprob += transition_matrix[current_regime * n + j];
            if u <= cumprob {
                next_regime = j;
                break;
            }
        }
        current_regime = next_regime;

        // Generate value using regime parameters
        let mu = regime_means[current_regime];
        let sigma = regime_variances[current_regime].sqrt();
        let phi = regime_ar_coeffs[current_regime];
        out[t] = mu
            + phi * (out[t - 1] - mu)
            + sigma * rng.sample_innovation(1.0, innov_dist, innov_param);
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

    // --- Stochastic Volatility: Heston ---

    #[test]
    fn test_heston_price_finite() {
        let mut out = vec![0.0; N];
        stochastic_volatility(
            &mut out,
            0,
            100.0,
            0.04,
            0.05,
            0.04,
            2.0,
            0.3,
            -0.7,
            1.0,
            1.0 / 252.0,
            0,
            42,
            0,
            0.0,
        );
        assert_finite(&out, "heston_price");
    }

    #[test]
    fn test_heston_returns_finite() {
        let mut out = vec![0.0; N];
        stochastic_volatility(
            &mut out,
            0,
            100.0,
            0.04,
            0.05,
            0.04,
            2.0,
            0.3,
            -0.7,
            1.0,
            1.0 / 252.0,
            1,
            42,
            0,
            0.0,
        );
        assert_finite(&out, "heston_returns");
        // First return should be 0
        assert!((out[0]).abs() < 1e-10);
    }

    #[test]
    fn test_heston_volatility_output() {
        let mut out = vec![0.0; N];
        stochastic_volatility(
            &mut out,
            0,
            100.0,
            0.04,
            0.05,
            0.04,
            2.0,
            0.3,
            -0.7,
            1.0,
            1.0 / 252.0,
            2,
            42,
            0,
            0.0,
        );
        assert_finite(&out, "heston_vol");
        for &v in &out {
            assert!(v >= 0.0, "volatility should be non-negative");
        }
    }

    #[test]
    fn test_heston_deterministic() {
        assert_deterministic(
            |o| {
                stochastic_volatility(
                    o,
                    0,
                    100.0,
                    0.04,
                    0.05,
                    0.04,
                    2.0,
                    0.3,
                    -0.7,
                    1.0,
                    1.0 / 252.0,
                    0,
                    42,
                    0,
                    0.0,
                )
            },
            "heston",
        );
    }

    // --- Stochastic Volatility: SABR ---

    #[test]
    fn test_sabr_finite() {
        let mut out = vec![0.0; N];
        stochastic_volatility(
            &mut out,
            1,
            100.0,
            0.04,
            0.0,
            0.0,
            0.0,
            0.3,
            -0.5,
            0.5,
            1.0 / 252.0,
            0,
            42,
            0,
            0.0,
        );
        assert_finite(&out, "sabr");
    }

    // --- Regime Switching ---

    #[test]
    fn test_regime_switching_finite() {
        let means = [0.0, 5.0];
        let vars = [1.0, 4.0];
        let ar = [0.5, 0.3];
        let tm = [0.95, 0.05, 0.1, 0.9]; // row-major 2x2
        let mut out = vec![0.0; N];
        regime_switching(&mut out, 2, &means, &vars, &ar, &tm, &[], 0, 42, 0, 0.0);
        assert_finite(&out, "regime_switching");
    }

    #[test]
    fn test_regime_switching_deterministic() {
        let means = [0.0, 5.0];
        let vars = [1.0, 4.0];
        let ar = [0.5, 0.3];
        let tm = [0.95, 0.05, 0.1, 0.9];
        assert_deterministic(
            |o| regime_switching(o, 2, &means, &vars, &ar, &tm, &[], 0, 42, 0, 0.0),
            "regime_switching",
        );
    }

    #[test]
    fn test_regime_switching_with_student_t() {
        let means = [0.0, 5.0];
        let vars = [1.0, 4.0];
        let ar = [0.5, 0.3];
        let tm = [0.95, 0.05, 0.1, 0.9];
        let mut out = vec![0.0; N];
        regime_switching(&mut out, 2, &means, &vars, &ar, &tm, &[], 0, 42, 1, 5.0);
        assert_finite(&out, "regime_switching_t");
    }

    #[test]
    fn test_regime_switching_sentinel_draws_initial_regime_per_seed() {
        // initial_regime = -1: s_0 is drawn from stationary_probs using the
        // per-series RNG, so different seeds must not all share one regime
        // and occupancy at t=0 must match the stationary distribution.
        let means = [0.0, 100.0]; // far apart so out[0] identifies s_0
        let vars = [1e-6, 1e-6];
        let ar = [0.0, 0.0];
        let tm = [0.9, 0.1, 0.3, 0.7]; // stationary pi = (0.75, 0.25)
        let pi = [0.75, 0.25];
        let n_series = 2000;
        let mut count_regime1 = 0;
        for seed in 0..n_series {
            let mut out = vec![0.0; 2];
            regime_switching(&mut out, 2, &means, &vars, &ar, &tm, &pi, -1, seed, 0, 0.0);
            if out[0] > 50.0 {
                count_regime1 += 1;
            }
        }
        assert!(
            count_regime1 > 0 && count_regime1 < n_series,
            "first-step regimes should vary across seeds, got {count_regime1}/{n_series}"
        );
        let frac = count_regime1 as f64 / n_series as f64;
        assert!(
            (frac - 0.25).abs() < 0.05,
            "t=0 occupancy of regime 1 should be ~0.25, got {frac}"
        );
    }
}
