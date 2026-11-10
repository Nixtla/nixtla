//! TCM (Temporal Causal Model) generator: random causal-graph autoregression.
//!
//! Port of the univariate path of `synforecast/generators/tcm.py`. Every
//! series draws a fresh random SCM (sparse dependency graph over the
//! (variable x lag) space, per-edge random edge functions, per-node noise
//! config), rescales the linear-part coefficients toward a per-series
//! spectral-radius target, rolls the system out autoregressively with
//! burn-in and a soft clamp, and returns node 0. RNG streams do not match
//! numpy — parity with the Python reference is statistical, not bitwise.
//!
//! Scalar parameter layout (must match `TCMGenerator._get_batch_params`):
//!   0: n_vars_lo          1: n_vars_hi
//!   2: max_lag_lo         3: max_lag_hi
//!   4: edge_prob_lo       5: edge_prob_hi
//!   6: coef_lo            7: coef_hi
//!   8: stability_margin
//!   9: clamp_threshold
//!  10: noise_scale_lo    11: noise_scale_hi
//!  12: heteroscedastic_prob
//!
//! Array parameter layout:
//!   0: edge kind id pool (ids below, duplicates allowed)
//!   1: noise type id pool (ids below, duplicates allowed)
//!
//! Spectral radius of the companion matrix (up to ~120x120) is estimated by
//! power iteration / growth-rate estimation instead of a dense eigensolver:
//! `||A^m v||^(1/m)` converges to the radius, the estimate only feeds the
//! stability rescale, and the soft clamp + finiteness guard backstop it.

use std::f64::consts::PI;

use crate::rng::SfRng;

// Edge kind ids — mirror the order of `_EDGE_KINDS` in tcm.py
const KIND_LINEAR: i32 = 0;
const KIND_TANH: i32 = 1;
const KIND_RELU: i32 = 2;
const KIND_PRODUCT: i32 = 3;
const KIND_THRESHOLD: i32 = 4;

// Noise type ids — mirror `_NOISE_TYPES` in tcm.py. The ids coincide with
// the `SfRng::sample_innovations` dist encoding (0=gaussian, 1=student_t,
// 2=laplace), so they are passed through directly.
// gaussian = 0, student_t = 1, laplace = 2

// Divergence guard thresholds — mirror _MAX_ABS / _MIN_STD in tcm.py
const MAX_ABS: f64 = 1e8;
const MIN_STD: f64 = 1e-8;
// SCM redraws allowed before falling back to a guaranteed-stable AR(1)
const MAX_REDRAWS: usize = 5;
const MAX_BURN_IN: usize = 200;
// Bimodal per-series spectral-radius target (see tcm.py for rationale)
const PERSISTENT_PROB: f64 = 0.22;
const RADIUS_TARGET_HIGH: (f64, f64) = (0.95, 1.0);
const RADIUS_TARGET_LOW: (f64, f64) = (0.1, 0.6);
// Per-series geometric decay of coefficient magnitude with lag
const LAG_DECAY_RANGE: (f64, f64) = (0.4, 1.0);
// Cap on uniformly boosting a weak linear part toward the radius target
const UPSCALE_CAP: f64 = 4.0;
// Log-uniform softness scale of saturating edges: contribute c*s*tanh(x/s)
const SOFTNESS_RANGE: (f64, f64) = (0.5, 4.0);
// Heteroscedastic envelope: scale(t) = exp(a * sin(2*pi*t/P + phase))
const ENV_PERIOD_RANGE: (f64, f64) = (50.0, 500.0);
const ENV_AMP_RANGE: (f64, f64) = (0.3, 1.0);
const STUDENT_T_DF_RANGE: (f64, f64) = (3.0, 10.0);
// Power-iteration schedule for the companion spectral radius: growth rates
// are averaged over RADIUS_MEASURE steps after RADIUS_WARMUP mixing steps.
// The relative error is O(log(m)/m) — well within the >= 5% stability slack
// left by the radius targets (target <= 0.95 * stability_margin < margin).
const RADIUS_WARMUP: usize = 100;
const RADIUS_MEASURE: usize = 300;

/// Per-batch configuration: field-level ranges and pools shared by all series.
struct Cfg<'a> {
    n_vars: (i32, i32),
    max_lag: (i32, i32),
    edge_prob: (f64, f64),
    coef: (f64, f64),
    stability_margin: f64,
    clamp: f64,
    noise_scale: (f64, f64),
    hetero_prob: f64,
    edge_kinds: &'a [f64],
    noise_types: &'a [f64],
}

/// One sampled edge of the dependency graph.
struct Edge {
    pvar: usize,
    plag: usize,
    kind: i32,
    coef: f64,
    soft: f64,
    tau: f64,
    qvar: usize,
    qlag: usize,
}

impl Edge {
    /// Kinds whose contribution is unbounded-or-linear near 0; they enter
    /// the stability (companion) matrix with gain = coef
    /// (`_LINEAR_PART_CODES` in tcm.py).
    fn is_linear_part(&self) -> bool {
        matches!(self.kind, KIND_LINEAR | KIND_TANH | KIND_RELU)
    }
}

/// Per-node innovation configuration.
struct NoiseCfg {
    kind: i32,
    scale: f64,
    df: f64,
    hetero: bool,
    env_period: f64,
    env_amp: f64,
    env_phase: f64,
}

/// One sampled structural causal model.
struct Scm {
    n_vars: usize,
    max_lag: usize,
    nodes: Vec<Vec<Edge>>,
    noise: Vec<NoiseCfg>,
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

/// Categorical pick from a pool with the given probabilities.
fn categorical(rng: &mut SfRng, pool: &[f64], probs: &[f64]) -> i32 {
    let u = rng.uniform01();
    let mut acc = 0.0;
    for (&v, &p) in pool.iter().zip(probs.iter()) {
        acc += p;
        if u < acc {
            return v as i32;
        }
    }
    pool[pool.len() - 1] as i32
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

/// Assemble per-lag gain matrices from linear-part edges, flattened as
/// `lin[(lag * k + i) * k + j]` = gain of node j at lag+1 into node i.
fn linear_part(scm: &Scm) -> Vec<f64> {
    let k = scm.n_vars;
    let p = scm.max_lag;
    let mut lin = vec![0.0_f64; p * k * k];
    for (i, edges) in scm.nodes.iter().enumerate() {
        for e in edges {
            if e.is_linear_part() {
                lin[((e.plag - 1) * k + i) * k + e.pvar] += e.coef;
            }
        }
    }
    lin
}

/// One companion-matrix application: `y[0..k] = sum_lag A_lag x_lag`,
/// `y[k..] = x[0..dim-k]` (the identity shift of VAR companion form).
fn companion_apply(lin: &[f64], k: usize, p: usize, x: &[f64], y: &mut [f64]) {
    for i in 0..k {
        let mut acc = 0.0;
        for lag in 0..p {
            let row = &lin[((lag * k) + i) * k..((lag * k) + i) * k + k];
            let block = &x[lag * k..lag * k + k];
            for (a, b) in row.iter().zip(block.iter()) {
                acc += a * b;
            }
        }
        y[i] = acc;
    }
    if p > 1 {
        let (_, tail) = y.split_at_mut(k);
        tail.copy_from_slice(&x[..k * (p - 1)]);
    }
}

/// Spectral radius of the companion matrix of the linear part, estimated by
/// power iteration with per-step renormalization: the geometric mean of the
/// growth rates `||A v|| / ||v||` over RADIUS_MEASURE steps (after
/// RADIUS_WARMUP mixing steps) converges to the dominant eigenvalue modulus.
/// A nilpotent/zero linear part collapses to the zero vector and returns 0.
fn companion_spectral_radius(lin: &[f64], k: usize, p: usize) -> f64 {
    let dim = k * p;
    // Deterministic pseudo-random start vector: generic (non-orthogonal to
    // the dominant eigenspace for all practical matrices) without consuming
    // the series RNG stream.
    let mut v: Vec<f64> = (0..dim)
        .map(|i| 1.0 + 0.5 * ((i as f64) * 0.7368 + 0.31).sin())
        .collect();
    let norm0 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    for a in v.iter_mut() {
        *a /= norm0;
    }
    let mut w = vec![0.0_f64; dim];
    let mut log_sum = 0.0;
    for it in 0..(RADIUS_WARMUP + RADIUS_MEASURE) {
        companion_apply(lin, k, p, &v, &mut w);
        let nrm = w.iter().map(|a| a * a).sum::<f64>().sqrt();
        if nrm <= f64::MIN_POSITIVE {
            return 0.0;
        }
        if it >= RADIUS_WARMUP {
            log_sum += nrm.ln();
        }
        for (vi, wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / nrm;
        }
    }
    (log_sum / RADIUS_MEASURE as f64).exp()
}

/// Rescale linear-part coefficients toward a per-series radius target.
///
/// Scaling lag-l gains by s**l maps every companion eigenvalue z to s*z, so
/// the down-scale sets the spectral radius to the target (exactly, up to the
/// power-iteration estimate error). Weak linear parts are boosted uniformly
/// (capped) instead, then re-checked. Mirrors `_rescale_for_stability`.
fn rescale_for_stability(rng: &mut SfRng, cfg: &Cfg, scm: &mut Scm) {
    let target_range = if rng.uniform01() < PERSISTENT_PROB {
        RADIUS_TARGET_HIGH
    } else {
        RADIUS_TARGET_LOW
    };
    let target = cfg.stability_margin * uniform_range(rng, target_range.0, target_range.1);
    let radius = companion_spectral_radius(&linear_part(scm), scm.n_vars, scm.max_lag);

    let scale_linear_edges = |scm: &mut Scm, per_lag: bool, s: f64| {
        for edges in scm.nodes.iter_mut() {
            for e in edges.iter_mut() {
                if e.is_linear_part() {
                    e.coef *= if per_lag { s.powi(e.plag as i32) } else { s };
                }
            }
        }
    };

    if radius > target {
        scale_linear_edges(scm, true, target / radius);
    } else if radius > 1e-9 {
        let u = (target / radius).min(UPSCALE_CAP);
        scale_linear_edges(scm, false, u);
        let new_radius = companion_spectral_radius(&linear_part(scm), scm.n_vars, scm.max_lag);
        if new_radius >= cfg.stability_margin {
            scale_linear_edges(scm, true, target / new_radius);
        }
    }
}

/// Draw a random SCM: graph, edge functions, noise config; then rescale the
/// linear part for stability. Mirrors `_sample_scm` in tcm.py.
fn sample_scm(rng: &mut SfRng, cfg: &Cfg) -> Scm {
    let n = integer_incl(rng, cfg.n_vars.0.max(1), cfg.n_vars.1.max(1)) as usize;
    let max_lag = integer_incl(rng, cfg.max_lag.0.max(1), cfg.max_lag.1.max(1)) as usize;
    let p_edge = uniform_range(rng, cfg.edge_prob.0, cfg.edge_prob.1);
    let lag_decay = uniform_range(rng, LAG_DECAY_RANGE.0, LAG_DECAY_RANGE.1);
    // Per-series Dirichlet(1, ..., 1) mixture over edge kinds: normalized
    // Exp(1) draws are exactly Dirichlet with unit concentration.
    let mut kind_probs: Vec<f64> = (0..cfg.edge_kinds.len())
        .map(|_| rng.exponential(1.0))
        .collect();
    let total: f64 = kind_probs.iter().sum();
    for q in kind_probs.iter_mut() {
        *q /= total;
    }
    let linear_in_pool = cfg.edge_kinds.iter().any(|&kk| kk as i32 == KIND_LINEAR);

    let mut nodes: Vec<Vec<Edge>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut edges: Vec<Edge> = Vec::new();
        for pvar in 0..n {
            for lag0 in 0..max_lag {
                // Every node gets a self lag-1 edge so it is well-defined
                // and carries its own persistence.
                let forced_self = pvar == i && lag0 == 0;
                if !forced_self && rng.uniform01() >= p_edge {
                    continue;
                }
                let plag = lag0 + 1;
                let mut kind = categorical(rng, cfg.edge_kinds, &kind_probs);
                let mut coef = uniform_range(rng, cfg.coef.0, cfg.coef.1);
                if rng.uniform01() < 0.5 {
                    coef = -coef;
                }
                coef *= lag_decay.powi(plag as i32 - 1);
                if forced_self && linear_in_pool {
                    // Unit positive linear self-persistence on every node,
                    // set to the per-series radius target by the rescale.
                    kind = KIND_LINEAR;
                    coef = 1.0;
                }
                edges.push(Edge {
                    pvar,
                    plag,
                    kind,
                    coef,
                    soft: log_uniform(rng, SOFTNESS_RANGE.0, SOFTNESS_RANGE.1),
                    tau: rng.normal(0.0, 1.0),
                    // second parent for product-interaction edges
                    qvar: rng.integers(0, n as i32) as usize,
                    qlag: rng.integers(1, max_lag as i32 + 1) as usize,
                });
            }
        }
        nodes.push(edges);
    }

    let noise: Vec<NoiseCfg> = (0..n)
        .map(|_| NoiseCfg {
            kind: pick(rng, cfg.noise_types) as i32,
            scale: uniform_range(rng, cfg.noise_scale.0, cfg.noise_scale.1),
            df: uniform_range(rng, STUDENT_T_DF_RANGE.0, STUDENT_T_DF_RANGE.1),
            hetero: rng.uniform01() < cfg.hetero_prob,
            env_period: uniform_range(rng, ENV_PERIOD_RANGE.0, ENV_PERIOD_RANGE.1),
            env_amp: uniform_range(rng, ENV_AMP_RANGE.0, ENV_AMP_RANGE.1),
            env_phase: uniform_range(rng, 0.0, 2.0 * PI),
        })
        .collect();

    let mut scm = Scm {
        n_vars: n,
        max_lag,
        nodes,
        noise,
    };
    rescale_for_stability(rng, cfg, &mut scm);
    scm
}

/// Roll the SCM out with burn-in and return the target node (node 0).
fn rollout(rng: &mut SfRng, cfg: &Cfg, scm: &Scm, length: usize) -> Vec<f64> {
    let n = scm.n_vars;
    let max_lag = scm.max_lag;
    let burn_in = MAX_BURN_IN.min(length);
    let total = max_lag + burn_in + length;
    let clamp = cfg.clamp;

    // Per-node innovation sequences, stored contiguously per node:
    // noise[i * total + t]. Unit-variance draw scaled to the node's scale
    // (the sample_innovations dist encoding matches the noise type ids),
    // with an optional slow heteroscedastic envelope.
    let mut noise = vec![0.0_f64; n * total];
    for (i, ncfg) in scm.noise.iter().enumerate() {
        let sl = &mut noise[i * total..(i + 1) * total];
        rng.sample_innovations(sl, ncfg.scale, ncfg.kind, ncfg.df);
        if ncfg.hetero {
            for (t, v) in sl.iter_mut().enumerate() {
                *v *= (ncfg.env_amp
                    * (2.0 * PI * t as f64 / ncfg.env_period + ncfg.env_phase).sin())
                .exp();
            }
        }
    }

    // State x[t * n + i]; the first max_lag rows are seeded with noise.
    let mut x = vec![0.0_f64; total * n];
    for t in 0..max_lag {
        for i in 0..n {
            x[t * n + i] = noise[i * total + t];
        }
    }

    for t in max_lag..total {
        for i in 0..n {
            let mut raw = noise[i * total + t];
            for e in &scm.nodes[i] {
                let pv = x[(t - e.plag) * n + e.pvar];
                let val = match e.kind {
                    KIND_LINEAR => pv,
                    KIND_TANH => e.soft * (pv / e.soft).tanh(),
                    KIND_RELU => pv.max(0.0),
                    KIND_PRODUCT => {
                        let q = x[(t - e.qlag) * n + e.qvar];
                        e.soft * (pv / e.soft).tanh() * (q / e.soft).tanh()
                    }
                    KIND_THRESHOLD if pv > e.tau => 1.0,
                    KIND_THRESHOLD => 0.0,
                    _ => 0.0, // unknown id: contributes nothing
                };
                raw += e.coef * val;
            }
            // soft clamp: identity for |raw| << clamp, saturates at clamp
            x[t * n + i] = clamp * (raw / clamp).tanh();
        }
    }

    (max_lag + burn_in..total).map(|t| x[t * n]).collect()
}

/// Finiteness/scale guard: finite, |x| < 1e8, std > 1e-8 (`_series_ok`).
fn series_ok(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
        && values.iter().fold(0.0_f64, |a, &b| a.max(b.abs())) < MAX_ABS
        && pop_std(values) > MIN_STD
}

/// Guaranteed-stable linear AR(1) draw used when redraws are exhausted,
/// matching `_fallback_series` in tcm.py (phi = 0.7, unit noise, burn-in).
fn fallback_series(rng: &mut SfRng, out: &mut [f64]) {
    // out is non-empty here, so burn_in >= 1 and every output step advances
    // the recursion (x[0] = eps[0]; x[t] = 0.7 x[t-1] + eps[t]; drop burn-in)
    let burn_in = MAX_BURN_IN.min(out.len());
    let mut x = rng.normal(0.0, 1.0);
    for _ in 1..burn_in {
        x = 0.7 * x + rng.normal(0.0, 1.0);
    }
    for v in out.iter_mut() {
        x = 0.7 * x + rng.normal(0.0, 1.0);
        *v = x;
    }
}

/// Generate one TCM series into `out`.
///
/// `sp` and the two pool arrays follow the layout documented at the top of
/// this file. A fresh SCM is sampled per attempt; trajectories failing the
/// finiteness/scale guard are redrawn up to `MAX_REDRAWS` times, then a
/// guaranteed-stable AR(1) is used, matching tcm.py.
pub fn tcm(out: &mut [f64], sp: &[f64], edge_kinds: &[f64], noise_types: &[f64], seed: u64) {
    let length = out.len();
    if length == 0 {
        return;
    }
    assert!(sp.len() >= 13, "tcm: expected >= 13 scalar params");
    assert!(!edge_kinds.is_empty(), "tcm: edge kind pool is empty");
    assert!(!noise_types.is_empty(), "tcm: noise type pool is empty");

    let cfg = Cfg {
        n_vars: (sp[0] as i32, sp[1] as i32),
        max_lag: (sp[2] as i32, sp[3] as i32),
        edge_prob: (sp[4], sp[5]),
        coef: (sp[6], sp[7]),
        stability_margin: sp[8],
        clamp: sp[9],
        noise_scale: (sp[10], sp[11]),
        hetero_prob: sp[12],
        edge_kinds,
        noise_types,
    };

    let mut rng = SfRng::new(seed);
    for _ in 0..=MAX_REDRAWS {
        let scm = sample_scm(&mut rng, &cfg);
        let values = rollout(&mut rng, &cfg, &scm, length);
        if series_ok(&values) {
            out.copy_from_slice(&values);
            return;
        }
    }
    fallback_series(&mut rng, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default scalar params mirroring TCMGenerator's field defaults.
    fn default_sp() -> Vec<f64> {
        vec![
            1.0, 5.0, // n_vars_range
            1.0, 24.0, // max_lag_range
            0.05, 0.3, // edge_probability_range
            0.1, 0.8,  // coef_range
            0.95, // stability_margin
            1e6,  // clamp_threshold
            0.5, 2.0, // noise_scale_range
            0.2, // heteroscedastic_prob
        ]
    }

    fn default_pools() -> (Vec<f64>, Vec<f64>) {
        (vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0, 2.0])
    }

    fn cfg_from<'a>(sp: &[f64], edge_kinds: &'a [f64], noise_types: &'a [f64]) -> Cfg<'a> {
        Cfg {
            n_vars: (sp[0] as i32, sp[1] as i32),
            max_lag: (sp[2] as i32, sp[3] as i32),
            edge_prob: (sp[4], sp[5]),
            coef: (sp[6], sp[7]),
            stability_margin: sp[8],
            clamp: sp[9],
            noise_scale: (sp[10], sp[11]),
            hetero_prob: sp[12],
            edge_kinds,
            noise_types,
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

    // --- Spectral-radius estimation on known matrices ---

    #[test]
    fn test_radius_ar1_scalar() {
        // k=1, p=1, coefficient 0.8: radius exactly 0.8
        let lin = vec![0.8];
        let r = companion_spectral_radius(&lin, 1, 1);
        assert!((r - 0.8).abs() < 1e-9, "radius {r} != 0.8");
    }

    #[test]
    fn test_radius_ar2_real_roots() {
        // x_t = 1.2 x_{t-1} - 0.35 x_{t-2}: roots of z^2 - 1.2 z + 0.35
        // are 0.7 and 0.5, so the radius is 0.7
        let lin = vec![1.2, -0.35];
        let r = companion_spectral_radius(&lin, 1, 2);
        assert!((r - 0.7).abs() < 1e-3, "radius {r} != 0.7");
    }

    #[test]
    fn test_radius_ar2_complex_roots() {
        // z^2 - z + 0.5 has roots (1 ± i)/2 with modulus sqrt(0.5)
        let lin = vec![1.0, -0.5];
        let r = companion_spectral_radius(&lin, 1, 2);
        let expected = 0.5_f64.sqrt();
        assert!((r - expected).abs() < 5e-3, "radius {r} != {expected}");
    }

    #[test]
    fn test_radius_var1_diagonal() {
        // k=2, p=1, diag(0.9, 0.3): radius 0.9
        let lin = vec![0.9, 0.0, 0.0, 0.3];
        let r = companion_spectral_radius(&lin, 2, 1);
        assert!((r - 0.9).abs() < 1e-6, "radius {r} != 0.9");
    }

    #[test]
    fn test_radius_unstable_matrix() {
        // AR(1) with phi = 1.3: radius above 1 must be detected
        let lin = vec![1.3];
        let r = companion_spectral_radius(&lin, 1, 1);
        assert!((r - 1.3).abs() < 1e-9, "radius {r} != 1.3");
    }

    #[test]
    fn test_radius_zero_and_nilpotent() {
        // Zero linear part: radius 0
        let r = companion_spectral_radius(&[0.0; 4], 2, 1);
        assert_eq!(r, 0.0);
        // Nilpotent companion (only a lag-2 cross edge, no feedback loop):
        // k=2, p=2 with A_2[0][1] != 0 only — all eigenvalues are 0
        let mut lin = vec![0.0; 8];
        // lin[((lag * k) + i) * k + j] with lag=1 (lag 2), i=0, j=1, k=2
        lin[5] = 0.5;
        let r = companion_spectral_radius(&lin, 2, 2);
        assert!(r < 1e-9, "nilpotent radius {r} should be ~0");
    }

    // --- Stability of rescaled systems ---

    #[test]
    fn test_sampled_scms_are_rescaled_below_margin() {
        let sp = default_sp();
        let (kinds, ntypes) = default_pools();
        let cfg = cfg_from(&sp, &kinds, &ntypes);
        for seed in 0..100 {
            let mut rng = SfRng::new(seed);
            let scm = sample_scm(&mut rng, &cfg);
            let r = companion_spectral_radius(&linear_part(&scm), scm.n_vars, scm.max_lag);
            assert!(
                r < cfg.stability_margin + 0.02,
                "seed {seed}: rescaled radius {r} above margin"
            );
        }
    }

    #[test]
    fn test_long_rollouts_stay_bounded() {
        let sp = default_sp();
        let (kinds, ntypes) = default_pools();
        for seed in 0..50 {
            let mut out = vec![0.0_f64; 4000];
            tcm(&mut out, &sp, &kinds, &ntypes, seed);
            assert!(out.iter().all(|v| v.is_finite()), "seed {seed}: non-finite");
            let max_abs = out.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            assert!(max_abs < MAX_ABS, "seed {seed}: |x| {max_abs} >= 1e8");
            assert!(pop_std(&out) > MIN_STD, "seed {seed}: degenerate output");
        }
    }

    #[test]
    fn test_bounded_only_edge_kinds_zero_radius_path() {
        // product/threshold edges have no linear part (radius 0): the
        // rescale is skipped and the rollout must still be well-behaved
        let sp = default_sp();
        let kinds = vec![3.0, 4.0];
        let ntypes = vec![0.0];
        for seed in 0..20 {
            let mut out = vec![0.0_f64; 512];
            tcm(&mut out, &sp, &kinds, &ntypes, seed);
            assert!(out.iter().all(|v| v.is_finite()), "seed {seed}");
            assert!(pop_std(&out) > MIN_STD, "seed {seed}");
        }
    }

    // --- Redraw guard and AR(1) fallback ---

    #[test]
    fn test_redraw_guard_falls_back_to_ar1() {
        // Noise scale 1e-12 makes every rollout's std fall below MIN_STD
        // (linear-only edges decay toward the noise floor), so all redraws
        // fail and the kernel must fall back to the stable AR(1) with
        // phi = 0.7 and unit noise: std 1/sqrt(1 - 0.49) ~= 1.4.
        let mut sp = default_sp();
        sp[10] = 1e-12;
        sp[11] = 1e-12;
        let kinds = vec![0.0]; // linear only
        let ntypes = vec![0.0]; // gaussian only
        let mut out = vec![0.0_f64; 20_000];
        tcm(&mut out, &sp, &kinds, &ntypes, 42);
        assert!(out.iter().all(|v| v.is_finite()));
        let std = pop_std(&out);
        let expected_std = (1.0_f64 / (1.0 - 0.49)).sqrt();
        assert!(
            (std - expected_std).abs() < 0.1,
            "fallback std {std} != AR(1) marginal std {expected_std}"
        );
        let acf = lag1_acf(&out);
        assert!(
            (acf - 0.7).abs() < 0.03,
            "fallback lag-1 acf {acf} != AR(1) phi 0.7"
        );
    }

    #[test]
    fn test_forced_linear_self_lag1_behaves_like_ar1() {
        // n_vars = 1, max_lag = 1, linear edges only: the SCM is an AR(1)
        // whose coefficient is the (bimodal) radius target, so the lag-1
        // ACF is significantly positive and below the margin.
        let mut sp = default_sp();
        sp[0] = 1.0;
        sp[1] = 1.0; // n_vars (1, 1)
        sp[2] = 1.0;
        sp[3] = 1.0; // max_lag (1, 1)
        sp[12] = 0.0; // no heteroscedastic envelope
        let kinds = vec![0.0];
        let ntypes = vec![0.0];
        for seed in [1_u64, 7, 42] {
            let mut out = vec![0.0_f64; 20_000];
            tcm(&mut out, &sp, &kinds, &ntypes, seed);
            let acf = lag1_acf(&out);
            assert!(
                acf > 0.05 && acf < 0.95,
                "seed {seed}: lag-1 acf {acf} not AR(1)-like"
            );
        }
    }

    // --- Guards and determinism ---

    #[test]
    fn test_default_config_guards_hold() {
        let sp = default_sp();
        let (kinds, ntypes) = default_pools();
        for seed in 0..200 {
            let mut out = vec![0.0_f64; 128];
            tcm(&mut out, &sp, &kinds, &ntypes, seed);
            assert!(out.iter().all(|v| v.is_finite()), "seed {seed}: non-finite");
            let max_abs = out.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            assert!(max_abs < MAX_ABS, "seed {seed}: |x| {max_abs} >= 1e8");
            assert!(pop_std(&out) > MIN_STD, "seed {seed}: constant output");
        }
    }

    #[test]
    fn test_deterministic_per_seed() {
        let sp = default_sp();
        let (kinds, ntypes) = default_pools();
        let mut a = vec![0.0_f64; 256];
        let mut b = vec![0.0_f64; 256];
        tcm(&mut a, &sp, &kinds, &ntypes, 99);
        tcm(&mut b, &sp, &kinds, &ntypes, 99);
        assert_eq!(a, b, "same seed must reproduce identical series");
        let mut c = vec![0.0_f64; 256];
        tcm(&mut c, &sp, &kinds, &ntypes, 100);
        assert_ne!(a, c, "different seeds must differ");
    }
}
