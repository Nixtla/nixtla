//! TSI (Trend / Seasonality / Irregularity) composition generator.
//!
//! Port of `synforecast/generators/tsi.py`. Every series draws a fresh
//! random configuration (trend type, 0+ seasonal harmonics, irregular
//! process, additive/multiplicative composition) from field-level ranges
//! that are passed once per batch. RNG streams do not match numpy —
//! parity with the Python reference is statistical, not bitwise.
//!
//! Scalar parameter layout (must match `TSIGenerator._get_batch_params`):
//!   0: trend_slope_lo         1: trend_slope_hi
//!   2: trend_growth_lo        3: trend_growth_hi
//!   4: n_breakpoints_lo       5: n_breakpoints_hi
//!   6: level_lo               7: level_hi
//!   8: n_seasonal_lo          9: n_seasonal_hi
//!  10: seasonal_amp_lo       11: seasonal_amp_hi
//!  12: amplitude_modulation_prob
//!  13: harmonics_prob
//!  14: noise_scale_lo        15: noise_scale_hi
//!  16: ar1_phi_lo            17: ar1_phi_hi
//!  18: tail_df_lo            19: tail_df_hi
//!  20: multiplicative_prob
//!  21: scale_lo              22: scale_hi
//!
//! Array parameter layout:
//!   0: trend type id pool (ids below, duplicates allowed)
//!   1: seasonal period pool (time steps, > 1, may be non-integer)
//!   2: irregular type id pool (ids below, duplicates allowed)

use std::f64::consts::PI;

use crate::rng::SfRng;

// Trend type ids — mirror the order of `_TREND_TYPES` in tsi.py
const TREND_NONE: i32 = 0;
const TREND_LINEAR: i32 = 1;
const TREND_EXPONENTIAL: i32 = 2;
const TREND_LOGISTIC: i32 = 3;
const TREND_PIECEWISE_LINEAR: i32 = 4;
const TREND_DAMPED: i32 = 5;

// Irregular type ids — mirror the order of `_IRREGULAR_TYPES` in tsi.py
const IRR_GAUSSIAN: i32 = 0;
const IRR_AR1: i32 = 1;
const IRR_GARCH_LIKE: i32 = 2;
const IRR_STUDENT_T: i32 = 3;
const IRR_LAPLACE: i32 = 4;

// Output guards — mirror _MAX_ABS / _MIN_STD / _MAX_RETRIES in tsi.py
const MAX_ABS: f64 = 1e8;
const MIN_STD: f64 = 1e-8;
const MAX_RETRIES: usize = 8;

/// Per-batch configuration: field-level ranges and pools shared by all series.
struct Cfg<'a> {
    trend_slope: (f64, f64),
    trend_growth: (f64, f64),
    n_breakpoints: (i32, i32),
    level: (f64, f64),
    n_seasonal: (i32, i32),
    seasonal_amp: (f64, f64),
    am_prob: f64,
    harmonics_prob: f64,
    noise_scale: (f64, f64),
    ar1_phi: (f64, f64),
    tail_df: (f64, f64),
    mult_prob: f64,
    scale: (f64, f64),
    trend_types: &'a [f64],
    periods: &'a [f64],
    irregular_types: &'a [f64],
}

/// Uniform draw that tolerates a degenerate (lo == hi) range.
fn uniform_range(rng: &mut SfRng, lo: f64, hi: f64) -> f64 {
    if lo >= hi {
        lo
    } else {
        rng.uniform(lo, hi)
    }
}

/// Log-uniform draw: exp of a uniform draw in log space.
fn log_uniform(rng: &mut SfRng, lo: f64, hi: f64) -> f64 {
    if lo >= hi {
        lo
    } else {
        rng.uniform(lo.ln(), hi.ln()).exp()
    }
}

/// Integer draw inclusive of both endpoints (numpy `integers(lo, hi + 1)`).
fn integer_incl(rng: &mut SfRng, lo: i32, hi: i32) -> i32 {
    if lo >= hi {
        lo
    } else {
        rng.integers(lo, hi + 1)
    }
}

/// Uniform pick from a non-empty pool.
fn pick(rng: &mut SfRng, pool: &[f64]) -> f64 {
    let idx = if pool.len() > 1 {
        rng.integers(0, pool.len() as i32) as usize
    } else {
        0
    };
    pool[idx]
}

/// Population (ddof = 0) standard deviation, matching `np.ndarray.std`.
fn pop_std(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt()
}

/// Linear interpolation over sorted knots, matching `np.interp` for
/// query points inside `[xs[0], xs[last]]`.
fn interp(u: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let mut i = 1;
    while i < xs.len() - 1 && u > xs[i] {
        i += 1;
    }
    let (x0, x1) = (xs[i - 1], xs[i]);
    let (y0, y1) = (ys[i - 1], ys[i]);
    if x1 <= x0 {
        y0
    } else {
        y0 + (y1 - y0) * (u - x0) / (x1 - x0)
    }
}

/// Sample a trend component: level + normalized trend shape.
///
/// Shapes are normalized so the total movement over the series is the
/// `trend_slope` draw regardless of length (piecewise stays bounded by
/// |movement|; damped approaches it), matching `_sample_trend` in tsi.py.
fn sample_trend(rng: &mut SfRng, cfg: &Cfg, length: usize) -> Vec<f64> {
    let kind = pick(rng, cfg.trend_types) as i32;
    let level = uniform_range(rng, cfg.level.0, cfg.level.1);
    let denom = length.saturating_sub(1).max(1) as f64;
    let movement = uniform_range(rng, cfg.trend_slope.0, cfg.trend_slope.1);

    let mut shape = vec![0.0_f64; length];
    match kind {
        TREND_LINEAR => {
            for (i, s) in shape.iter_mut().enumerate() {
                *s = movement * (i as f64 / denom);
            }
        }
        TREND_EXPONENTIAL => {
            let mut g = uniform_range(rng, cfg.trend_growth.0, cfg.trend_growth.1);
            if rng.uniform01() < 0.5 {
                g = -g;
            }
            let d = g.exp() - 1.0;
            for (i, s) in shape.iter_mut().enumerate() {
                let u = i as f64 / denom;
                *s = movement * ((g * u).exp() - 1.0) / d;
            }
        }
        TREND_LOGISTIC => {
            let steepness = uniform_range(rng, 5.0, 15.0);
            let center = uniform_range(rng, 0.25, 0.75);
            let raw = |u: f64| 1.0 / (1.0 + (-steepness * (u - center)).exp());
            let raw0 = raw(0.0);
            let raw1 = raw((length.saturating_sub(1)) as f64 / denom);
            let span = raw1 - raw0 + 1e-12;
            for (i, s) in shape.iter_mut().enumerate() {
                let u = i as f64 / denom;
                *s = movement * (raw(u) - raw0) / span;
            }
        }
        TREND_PIECEWISE_LINEAR => {
            let n_breaks = integer_incl(rng, cfg.n_breakpoints.0, cfg.n_breakpoints.1).max(1);
            let mut knots_x = Vec::with_capacity(n_breaks as usize + 2);
            knots_x.push(0.0);
            let mut breaks: Vec<f64> = (0..n_breaks)
                .map(|_| uniform_range(rng, 0.1, 0.9))
                .collect();
            breaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
            knots_x.extend(breaks);
            knots_x.push(1.0);
            let m = movement.abs();
            let knots_y: Vec<f64> = (0..n_breaks + 2)
                .map(|_| uniform_range(rng, -m, m))
                .collect();
            for (i, s) in shape.iter_mut().enumerate() {
                let u = i as f64 / denom;
                *s = interp(u, &knots_x, &knots_y);
            }
        }
        TREND_DAMPED => {
            let tau = uniform_range(rng, 0.15, 0.5);
            for (i, s) in shape.iter_mut().enumerate() {
                let u = i as f64 / denom;
                *s = movement * (1.0 - (-u / tau).exp());
            }
        }
        TREND_NONE => {} // flat shape
        _ => {}          // unknown id: treat as flat, like "none"
    }

    for s in shape.iter_mut() {
        *s += level;
    }
    shape
}

/// Sample 0+ harmonics with random period/amplitude/phase, prob-gated
/// phase-locked 2f/3f overtones and slowly varying amplitude envelope.
fn sample_seasonality(rng: &mut SfRng, cfg: &Cfg, length: usize) -> Vec<f64> {
    let n_harmonics = integer_incl(rng, cfg.n_seasonal.0, cfg.n_seasonal.1).max(0);
    let mut season = vec![0.0_f64; length];

    for _ in 0..n_harmonics {
        let period = pick(rng, cfg.periods);
        let amplitude = log_uniform(rng, cfg.seasonal_amp.0, cfg.seasonal_amp.1);
        let phase = uniform_range(rng, 0.0, 2.0 * PI);
        let overtones = rng.uniform01() < cfg.harmonics_prob;
        let modulated = rng.uniform01() < cfg.am_prob;
        let (env_period, env_depth, env_phase) = if modulated {
            (
                length as f64 * uniform_range(rng, 0.3, 1.5),
                uniform_range(rng, 0.3, 0.8),
                uniform_range(rng, 0.0, 2.0 * PI),
            )
        } else {
            (1.0, 0.0, 0.0)
        };

        for (t, s) in season.iter_mut().enumerate() {
            let angle = 2.0 * PI * (t as f64) / period + phase;
            let mut wave = amplitude * angle.sin();
            if overtones {
                // Phase-locked overtones -> non-sinusoidal seasonal shape
                wave += amplitude * (0.4 * (2.0 * angle).sin() + 0.2 * (3.0 * angle).sin());
            }
            if modulated {
                wave *= 1.0 + env_depth * (2.0 * PI * (t as f64) / env_period + env_phase).sin();
            }
            *s += wave;
        }
    }

    season
}

/// Sample an irregular component with marginal std ~= sigma.
fn sample_irregular(rng: &mut SfRng, cfg: &Cfg, length: usize, sigma: f64) -> Vec<f64> {
    let kind = pick(rng, cfg.irregular_types) as i32;
    let mut values = vec![0.0_f64; length];
    if length == 0 || sigma <= 0.0 {
        return values;
    }

    match kind {
        IRR_LAPLACE => {
            // Laplace with scale sigma/sqrt(2) has std sigma
            // (innov_dist 2 in SfRng::sample_innovations)
            rng.sample_innovations(&mut values, sigma, 2, 0.0);
        }
        IRR_STUDENT_T => {
            let df = uniform_range(rng, cfg.tail_df.0, cfg.tail_df.1);
            // Student-t rescaled by sqrt((df-2)/df) has std sigma
            // (innov_dist 1 in SfRng::sample_innovations)
            rng.sample_innovations(&mut values, sigma, 1, df);
        }
        IRR_AR1 => {
            let phi = uniform_range(rng, cfg.ar1_phi.0, cfg.ar1_phi.1);
            // Innovation variance sigma^2 (1 - phi^2) gives marginal std sigma
            let innov_std = sigma * (1.0 - phi * phi).max(0.0).sqrt();
            values[0] = rng.normal(0.0, sigma);
            for i in 1..length {
                values[i] = phi * values[i - 1] + rng.normal(0.0, innov_std);
            }
        }
        IRR_GARCH_LIKE => {
            // GARCH(1,1) recursion, kept stationary (alpha + beta < 1)
            let mut alpha = uniform_range(rng, 0.05, 0.25);
            let mut beta = uniform_range(rng, 0.5, 0.9);
            let persistence = alpha + beta;
            if persistence > 0.98 {
                alpha *= 0.98 / persistence;
                beta *= 0.98 / persistence;
            }
            let omega = sigma * sigma * (1.0 - alpha - beta);
            let mut h = sigma * sigma;
            for v in values.iter_mut() {
                let z = rng.normal(0.0, 1.0);
                *v = h.sqrt() * z;
                h = omega + alpha * *v * *v + beta * h;
            }
        }
        IRR_GAUSSIAN => {
            rng.normal_array(&mut values, 0.0, sigma);
        }
        _ => {
            // Unknown id: treat as gaussian
            rng.normal_array(&mut values, 0.0, sigma);
        }
    }

    values
}

/// Draw one random T/S/I configuration and compose a series.
fn compose(rng: &mut SfRng, cfg: &Cfg, length: usize) -> Vec<f64> {
    let trend = sample_trend(rng, cfg, length);
    let season = sample_seasonality(rng, cfg, length);

    let has_season = season.iter().any(|&x| x != 0.0);
    let multiplicative = has_season && rng.uniform01() < cfg.mult_prob;

    let mut structural = vec![0.0_f64; length];
    if multiplicative {
        let low = trend.iter().copied().fold(f64::INFINITY, f64::min);
        // Shift so the base level stays positive
        let shift = if low <= 0.0 {
            let max_abs = trend.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            -low + 0.1 * (max_abs + 1.0)
        } else {
            0.0
        };
        // Cap the relative seasonal swing so the factor stays positive
        let max_abs_season = season.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let cap = (1.25 * max_abs_season).max(1.0);
        for i in 0..length {
            structural[i] = (trend[i] + shift) * (1.0 + season[i] / cap);
        }
    } else {
        for i in 0..length {
            structural[i] = trend[i] + season[i];
        }
    }

    let mut signal_std = if length > 1 {
        pop_std(&structural)
    } else {
        0.0
    };
    if signal_std < 1e-9 {
        signal_std = 1.0; // structureless draw: noise gets an absolute scale
    }
    let noise_fraction = log_uniform(rng, cfg.noise_scale.0, cfg.noise_scale.1);
    let irregular = sample_irregular(rng, cfg, length, noise_fraction * signal_std);

    let scale = log_uniform(rng, cfg.scale.0, cfg.scale.1);
    structural
        .iter()
        .zip(irregular.iter())
        .map(|(&s, &e)| scale * (s + e))
        .collect()
}

/// Generate one TSI-composed series into `out`.
///
/// `sp` and the three pool arrays follow the layout documented at the top
/// of this file. Degenerate or exploding draws (non-finite, |y| >= 1e8, or
/// constant) are redrawn up to `MAX_RETRIES` times; if the budget is
/// exhausted the output falls back to N(0, 1) noise, matching tsi.py.
pub fn tsi(
    out: &mut [f64],
    sp: &[f64],
    trend_types: &[f64],
    periods: &[f64],
    irregular_types: &[f64],
    seed: u64,
) {
    let length = out.len();
    if length == 0 {
        return;
    }
    assert!(sp.len() >= 23, "tsi: expected >= 23 scalar params");
    assert!(!trend_types.is_empty(), "tsi: trend type pool is empty");
    assert!(!periods.is_empty(), "tsi: seasonal period pool is empty");
    assert!(
        !irregular_types.is_empty(),
        "tsi: irregular type pool is empty"
    );

    let cfg = Cfg {
        trend_slope: (sp[0], sp[1]),
        trend_growth: (sp[2], sp[3]),
        n_breakpoints: (sp[4] as i32, sp[5] as i32),
        level: (sp[6], sp[7]),
        n_seasonal: (sp[8] as i32, sp[9] as i32),
        seasonal_amp: (sp[10], sp[11]),
        am_prob: sp[12],
        harmonics_prob: sp[13],
        noise_scale: (sp[14], sp[15]),
        ar1_phi: (sp[16], sp[17]),
        tail_df: (sp[18], sp[19]),
        mult_prob: sp[20],
        scale: (sp[21], sp[22]),
        trend_types,
        periods,
        irregular_types,
    };

    let mut rng = SfRng::new(seed);
    for _ in 0..MAX_RETRIES {
        let values = compose(&mut rng, &cfg, length);
        if !values.iter().all(|v| v.is_finite()) {
            continue;
        }
        let max_abs = values.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        if max_abs >= MAX_ABS {
            continue;
        }
        if length > 1 && pop_std(&values) <= MIN_STD {
            continue;
        }
        out.copy_from_slice(&values);
        return;
    }
    // Retry budget exhausted (pathological ranges): fall back to noise
    rng.normal_array(out, 0.0, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default scalar params mirroring TSIGenerator's field defaults.
    fn default_sp() -> Vec<f64> {
        vec![
            -8.0, 8.0, // trend_slope
            1.0, 4.0, // trend_growth
            1.0, 3.0, // n_breakpoints
            -10.0, 10.0, // level
            0.0, 3.0, // n_seasonal
            0.2, 3.0, // seasonal_amp
            0.4, // amplitude_modulation_prob
            0.4, // harmonics_prob
            0.5, 12.0, // noise_scale
            0.3, 0.95, // ar1_phi
            2.5, 12.0, // tail_df
            0.3,  // multiplicative_prob
            0.1, 100.0, // scale
        ]
    }

    fn default_pools() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![
                5.5, 7.0, 11.3, 12.0, 19.7, 24.0, 29.53, 48.0, 96.0, 168.0, 336.0, 365.25,
            ],
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
        )
    }

    fn forced_cfg<'a>(
        trend_types: &'a [f64],
        periods: &'a [f64],
        irregular_types: &'a [f64],
    ) -> Cfg<'a> {
        Cfg {
            trend_slope: (5.0, 5.0),
            trend_growth: (2.0, 2.0),
            n_breakpoints: (2, 2),
            level: (0.0, 0.0),
            n_seasonal: (0, 0),
            seasonal_amp: (1.0, 1.0),
            am_prob: 0.0,
            harmonics_prob: 0.0,
            noise_scale: (1.0, 1.0),
            ar1_phi: (0.7, 0.7),
            tail_df: (5.0, 5.0),
            mult_prob: 0.0,
            scale: (1.0, 1.0),
            trend_types,
            periods,
            irregular_types,
        }
    }

    fn lag1_acf(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let denom: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
        let num: f64 = x
            .windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>();
        num / denom
    }

    // --- Trend normalization ---

    #[test]
    fn test_trend_normalization_end_to_end_movement() {
        // With slope range (5, 5), linear / exponential / logistic trends
        // must move (approximately) 5.0 end-to-end regardless of length.
        let periods = [24.0];
        let irr = [0.0];
        let cases: [(f64, f64); 3] = [
            (TREND_LINEAR as f64, 1e-12),
            (TREND_EXPONENTIAL as f64, 1e-12),
            (TREND_LOGISTIC as f64, 1e-9),
        ];
        for &(kind, tol) in &cases {
            for &length in &[64usize, 512, 2048] {
                let pool = [kind];
                let cfg = forced_cfg(&pool, &periods, &irr);
                let mut rng = SfRng::new(1234);
                let trend = sample_trend(&mut rng, &cfg, length);
                let movement = trend[length - 1] - trend[0];
                assert!(
                    (movement.abs() - 5.0).abs() < tol.max(1e-6),
                    "kind {kind}, len {length}: movement {movement}"
                );
            }
        }
    }

    #[test]
    fn test_trend_piecewise_and_damped_bounded_by_movement() {
        let periods = [24.0];
        let irr = [0.0];
        for &kind in &[TREND_PIECEWISE_LINEAR as f64, TREND_DAMPED as f64] {
            let pool = [kind];
            let cfg = forced_cfg(&pool, &periods, &irr);
            for seed in 0..20 {
                let mut rng = SfRng::new(seed);
                let trend = sample_trend(&mut rng, &cfg, 256);
                for &v in &trend {
                    assert!(
                        v.abs() <= 5.0 + 1e-9,
                        "kind {kind}, seed {seed}: |{v}| exceeds movement 5"
                    );
                }
            }
        }
    }

    #[test]
    fn test_trend_none_is_flat_level() {
        let pool = [TREND_NONE as f64];
        let periods = [24.0];
        let irr = [0.0];
        let mut cfg = forced_cfg(&pool, &periods, &irr);
        cfg.level = (3.0, 3.0);
        let mut rng = SfRng::new(7);
        let trend = sample_trend(&mut rng, &cfg, 100);
        for &v in &trend {
            assert!((v - 3.0).abs() < 1e-12);
        }
    }

    // --- Irregular process stationarity ---

    #[test]
    fn test_ar1_stationary_std_and_acf() {
        let pool = [TREND_NONE as f64];
        let periods = [24.0];
        let irr = [IRR_AR1 as f64];
        let cfg = forced_cfg(&pool, &periods, &irr);
        let mut rng = SfRng::new(42);
        let values = sample_irregular(&mut rng, &cfg, 50_000, 1.0);
        let std = pop_std(&values);
        assert!((std - 1.0).abs() < 0.05, "ar1 marginal std {std} != 1.0");
        let acf = lag1_acf(&values);
        assert!((acf - 0.7).abs() < 0.03, "ar1 lag-1 acf {acf} != 0.7");
    }

    #[test]
    fn test_garch_like_stationary_std() {
        let pool = [TREND_NONE as f64];
        let periods = [24.0];
        let irr = [IRR_GARCH_LIKE as f64];
        let cfg = forced_cfg(&pool, &periods, &irr);
        let mut rng = SfRng::new(42);
        let sigma = 2.0;
        let values = sample_irregular(&mut rng, &cfg, 100_000, sigma);
        assert!(values.iter().all(|v| v.is_finite()));
        let std = pop_std(&values);
        // GARCH sample std is noisy (fat tails); generous tolerance
        assert!(
            (std - sigma).abs() < 0.4 * sigma,
            "garch marginal std {std} far from {sigma}"
        );
    }

    #[test]
    fn test_student_t_and_laplace_std() {
        let pool = [TREND_NONE as f64];
        let periods = [24.0];
        for &(irr_kind, name) in &[
            (IRR_STUDENT_T as f64, "student_t"),
            (IRR_LAPLACE as f64, "laplace"),
        ] {
            let irr = [irr_kind];
            let cfg = forced_cfg(&pool, &periods, &irr);
            let mut rng = SfRng::new(11);
            let values = sample_irregular(&mut rng, &cfg, 100_000, 1.0);
            let std = pop_std(&values);
            assert!((std - 1.0).abs() < 0.1, "{name}: marginal std {std} != 1.0");
        }
    }

    // --- Redraw guard ---

    #[test]
    fn test_redraw_guard_falls_back_to_unit_noise() {
        // scale = 1e12 makes every compose() draw exceed MAX_ABS, so all
        // retries fail and the kernel must fall back to N(0, 1).
        let mut sp = default_sp();
        sp[21] = 1e12;
        sp[22] = 1e12;
        let (tt, pp, ii) = default_pools();
        let mut out = vec![0.0_f64; 2000];
        tsi(&mut out, &sp, &tt, &pp, &ii, 42);
        assert!(out.iter().all(|v| v.is_finite()));
        let max_abs = out.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(
            max_abs < 10.0,
            "fallback should be unit noise, max {max_abs}"
        );
        let std = pop_std(&out);
        assert!((std - 1.0).abs() < 0.1, "fallback std {std} should be ~1.0");
    }

    #[test]
    fn test_default_config_guards_hold() {
        let sp = default_sp();
        let (tt, pp, ii) = default_pools();
        for seed in 0..200 {
            let mut out = vec![0.0_f64; 128];
            tsi(&mut out, &sp, &tt, &pp, &ii, seed);
            assert!(out.iter().all(|v| v.is_finite()), "seed {seed}: non-finite");
            let max_abs = out.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            assert!(max_abs < MAX_ABS, "seed {seed}: |y| {max_abs} >= 1e8");
            assert!(pop_std(&out) > MIN_STD, "seed {seed}: constant output");
        }
    }

    #[test]
    fn test_deterministic_per_seed() {
        let sp = default_sp();
        let (tt, pp, ii) = default_pools();
        let mut a = vec![0.0_f64; 256];
        let mut b = vec![0.0_f64; 256];
        tsi(&mut a, &sp, &tt, &pp, &ii, 99);
        tsi(&mut b, &sp, &tt, &pp, &ii, 99);
        assert_eq!(a, b, "same seed must reproduce identical series");
        let mut c = vec![0.0_f64; 256];
        tsi(&mut c, &sp, &tt, &pp, &ii, 100);
        assert_ne!(a, c, "different seeds must differ");
    }

    #[test]
    fn test_multiplicative_composition_finite() {
        // Force multiplicative composition with a zero-crossing trend
        let mut sp = default_sp();
        sp[8] = 1.0; // n_seasonal_lo
        sp[9] = 2.0; // n_seasonal_hi
        sp[14] = 0.01; // noise_scale_lo
        sp[15] = 0.01; // noise_scale_hi
        sp[20] = 1.0; // multiplicative_prob
        let (tt, pp, ii) = default_pools();
        for seed in 0..50 {
            let mut out = vec![0.0_f64; 200];
            tsi(&mut out, &sp, &tt, &pp, &ii, seed);
            assert!(out.iter().all(|v| v.is_finite()), "seed {seed}");
            assert!(pop_std(&out) > MIN_STD, "seed {seed}");
        }
    }

    #[test]
    fn test_interp_matches_expected() {
        let xs = [0.0, 0.5, 1.0];
        let ys = [0.0, 2.0, 1.0];
        assert!((interp(0.0, &xs, &ys) - 0.0).abs() < 1e-12);
        assert!((interp(0.25, &xs, &ys) - 1.0).abs() < 1e-12);
        assert!((interp(0.5, &xs, &ys) - 2.0).abs() < 1e-12);
        assert!((interp(0.75, &xs, &ys) - 1.5).abs() < 1e-12);
        assert!((interp(1.0, &xs, &ys) - 1.0).abs() < 1e-12);
    }
}
