//! KernelSynth generator: GP samples from randomly composed kernels.
//!
//! The sampler behind `KernelSynthGenerator` (`synforecast/generators/kernel_synth.py`),
//! used by both the batch path and `generate_single_series`. The linear kernel is `s^2 + x x'` and the others
//! are stationary, so every composition is `K = sum_d (x x')^d T_d(x - x')`
//! with PSD `T_d`, and `y = sum_d x^d * f_d` with independent
//! `f_d ~ GP(0, T_d)` (Durbin-Levinson, O(n^2)) has covariance exactly `K`.
//!
//! Scalar parameter layout (must match `KernelSynthGenerator._get_batch_params`):
//!   0: max_kernels   1: jitter   2: standardize (0/1)
//!
//! Array parameter layout:
//!   0: kernel kind ids of the bank (ids below)
//!   1: kernel parameters of the bank, aligned with the kind ids

use std::f64::consts::PI;

use crate::rng::SfRng;

// Kernel kind ids — mirror `_KIND_IDS` in kernel_synth.py
const KIND_RBF: i32 = 0;
const KIND_RATIONAL_QUADRATIC: i32 = 1;
const KIND_PERIODIC: i32 = 2;
const KIND_LINEAR: i32 = 3;
const KIND_WHITE: i32 = 4;
const KIND_CONSTANT: i32 = 5;

// Divergence guards and retry budget, as in the other pretraining generators (TSI/TCM).
const MAX_ABS: f64 = 1e8;
const MIN_STD: f64 = 1e-8;
const MAX_RETRIES: usize = 8;
/// Levinson attempts per term: the user's jitter, then three tenfold escalations.
const MAX_REGULARIZATIONS: usize = 4;
/// Relative jitter every escalation reaches at least, so `jitter = 0` still regularizes.
const MIN_ESCALATION: f64 = 1e-6;
/// Upper bound on `max_kernels`; mirrors the `le` bound of the Python field.
pub(crate) const MAX_KERNELS: f64 = 64.0;

/// Coefficients `T_d` of the composition, indexed by the degree `d`; `None` is a zero term.
type Terms = Vec<Option<Vec<f64>>>;

/// Stationary base kernel at the normalized-grid lags `lag`.
fn lag_kernel(kind: i32, param: f64, lag: &[f64]) -> Vec<f64> {
    let length = lag.len() as f64;
    match kind {
        KIND_RBF => lag
            .iter()
            .map(|&r| (-0.5 * r * r / (param * param)).exp())
            .collect(),
        KIND_RATIONAL_QUADRATIC => lag
            .iter()
            .map(|&r| (1.0 + r * r / (2.0 * param)).powf(-param))
            .collect(),
        KIND_PERIODIC => {
            let p_norm = param / length;
            lag.iter()
                .map(|&r| {
                    let s = (PI * r / p_norm).sin();
                    (-2.0 * s * s).exp()
                })
                .collect()
        }
        KIND_WHITE => {
            let mut values = vec![0.0; lag.len()];
            values[0] = param;
            values
        }
        KIND_CONSTANT => vec![param; lag.len()],
        _ => unreachable!("kernel kinds are validated by kernel_synth"),
    }
}

/// One bank entry as polynomial terms: `s^2 + x x'` for the linear kernel, a degree-0 term otherwise.
fn base_terms(kind: i32, param: f64, lag: &[f64]) -> Terms {
    if kind == KIND_LINEAR {
        let offset = (param > 0.0).then(|| vec![param * param; lag.len()]);
        return vec![offset, Some(vec![1.0; lag.len()])];
    }
    vec![Some(lag_kernel(kind, param, lag))]
}

fn add_terms(a: Terms, b: Terms) -> Terms {
    let n = a.len().max(b.len());
    let mut a = a.into_iter();
    let mut b = b.into_iter();
    (0..n)
        .map(|_| match (a.next().flatten(), b.next().flatten()) {
            (Some(mut left), Some(right)) => {
                left.iter_mut().zip(&right).for_each(|(l, r)| *l += r);
                Some(left)
            }
            (left, right) => left.or(right),
        })
        .collect()
}

fn multiply_terms(a: &Terms, b: &Terms) -> Terms {
    let mut out: Terms = vec![None; a.len() + b.len() - 1];
    for (i, left) in a.iter().enumerate() {
        let Some(left) = left else { continue };
        for (j, right) in b.iter().enumerate() {
            let Some(right) = right else { continue };
            match &mut out[i + j] {
                Some(acc) => acc
                    .iter_mut()
                    .zip(left.iter().zip(right))
                    .for_each(|(o, (l, r))| *o += l * r),
                slot => *slot = Some(left.iter().zip(right).map(|(l, r)| l * r).collect()),
            }
        }
    }
    out
}

/// Terms of the composition `bank[indices[0]] op_1 bank[indices[1]] ...` (true = add).
fn kernel_terms(
    indices: &[usize],
    ops: &[bool],
    kinds: &[f64],
    params: &[f64],
    lag: &[f64],
) -> Terms {
    let term = |i: usize| base_terms(kinds[i] as i32, params[i], lag);
    let mut terms = term(indices[0]);
    for (&index, &add) in indices[1..].iter().zip(ops) {
        let comp = term(index);
        terms = if add {
            add_terms(terms, comp)
        } else {
            multiply_terms(&terms, &comp)
        };
    }
    terms
}

fn dot_reversed(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let (left, left_tail) = a.as_chunks::<4>();
    let (right_head, right) = b.as_rchunks::<4>();
    // The unpaired tails: the last `n % 4` of `a` with the first `n % 4` of `b`.
    let tail: f64 = left_tail
        .iter()
        .zip(right_head.iter().rev())
        .map(|(x, y)| x * y)
        .sum();
    let mut acc = [0.0_f64; 4];
    for (x, y) in left.iter().zip(right.iter().rev()) {
        acc[0] += x[0] * y[3];
        acc[1] += x[1] * y[2];
        acc[2] += x[2] * y[1];
        acc[3] += x[3] * y[0];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// `L z` for the Cholesky factor `L` of `Toeplitz(c) + extra I`, by Durbin-Levinson in O(n^2);
/// `None` if the matrix is numerically not positive definite.
pub(crate) fn levinson_sample(c: &[f64], extra: f64, z: &[f64]) -> Option<Vec<f64>> {
    let n = c.len();
    let mut v = c[0] + extra;
    if v.is_nan() || v <= 0.0 {
        return None;
    }
    let mut x = vec![0.0_f64; n];
    // phi[k] holds the order-t prediction coefficient phi_{t,k+1}.
    let mut phi = vec![0.0_f64; n];
    x[0] = v.sqrt() * z[0];
    for t in 1..n {
        let m = t - 1;
        let acc = dot_reversed(&phi[..m], &c[1..t]);
        let kappa = (c[t] - acc) / v;
        if kappa.is_nan() || kappa.abs() >= 1.0 {
            return None;
        }
        // phi_{t,k} = phi_{t-1,k} - kappa phi_{t-1,t-k}, in place over symmetric pairs.
        let (mut lo, mut hi) = (0, m);
        while lo + 1 < hi {
            hi -= 1;
            let (a, b) = (phi[lo], phi[hi]);
            phi[lo] = a - kappa * b;
            phi[hi] = b - kappa * a;
            lo += 1;
        }
        if lo + 1 == hi {
            phi[lo] *= 1.0 - kappa;
        }
        phi[m] = kappa;
        v *= 1.0 - kappa * kappa;
        if v <= 0.0 {
            return None;
        }
        let mu = dot_reversed(&phi[..t], &x[..t]);
        x[t] = mu + v.sqrt() * z[t];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// One draw from `N(0, Toeplitz(c) + jitter I)`; the jitter escalates on a numerical breakdown.
fn sample_stationary(rng: &mut SfRng, c: &[f64], jitter: f64) -> Option<Vec<f64>> {
    let n = c.len();
    let c0 = c[0];
    let scale = c0.abs().max(f64::MIN_POSITIVE);
    if c.iter().all(|&v| (v - c0).abs() <= 1e-12 * scale) {
        // Constant (rank-one) kernel: one shared normal, which Levinson cannot factor.
        let level = c0.max(0.0).sqrt() * rng.standard_normal();
        let sd = jitter.sqrt();
        return Some((0..n).map(|_| level + sd * rng.standard_normal()).collect());
    }
    if c[1..].iter().all(|&v| v == 0.0) {
        let sd = (c0 + jitter).max(0.0).sqrt();
        return Some((0..n).map(|_| sd * rng.standard_normal()).collect());
    }
    let z: Vec<f64> = (0..n).map(|_| rng.standard_normal()).collect();
    let mut extra = jitter;
    for _ in 0..MAX_REGULARIZATIONS {
        if let Some(values) = levinson_sample(c, extra, &z) {
            return Some(values);
        }
        extra = (10.0 * extra).max(MIN_ESCALATION * scale);
    }
    None
}

/// Draw one composition and sample a path from its GP prior; `None` on breakdown.
fn compose_and_sample(
    rng: &mut SfRng,
    max_kernels: i32,
    jitter: f64,
    kinds: &[f64],
    params: &[f64],
    x: &[f64],
) -> Option<Vec<f64>> {
    let n_bank = kinds.len() as i32;
    let n_kernels = rng.integers(1, max_kernels + 1) as usize;
    let indices: Vec<usize> = (0..n_kernels)
        .map(|_| rng.integers(0, n_bank) as usize)
        .collect();
    // true = add, false = multiply
    let ops: Vec<bool> = (1..n_kernels).map(|_| rng.integers(0, 2) == 0).collect();

    // x[0] == 0, so the grid doubles as the lag vector.
    let terms = kernel_terms(&indices, &ops, kinds, params, x);
    let mut values = vec![0.0_f64; x.len()];
    for (degree, c) in terms.iter().enumerate() {
        let Some(c) = c else { continue };
        if c.iter().all(|&v| v == 0.0) {
            continue;
        }
        let part = sample_stationary(rng, c, jitter)?;
        for ((v, &p), &xv) in values.iter_mut().zip(&part).zip(x) {
            *v += xv.powi(degree as i32) * p;
        }
    }
    Some(values)
}

fn mean_and_pop_std(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Check the encoded parameters of [`kernel_synth`].
fn validate(sp: &[f64], kinds: &[f64], params: &[f64]) -> Result<(), String> {
    if sp.len() < 3 {
        return Err(format!(
            "kernel_synth: expected >= 3 scalar params, got {}",
            sp.len()
        ));
    }
    if !(1.0..=MAX_KERNELS).contains(&sp[0]) {
        return Err(format!(
            "kernel_synth: max_kernels must be in [1, {MAX_KERNELS}], got {}",
            sp[0]
        ));
    }
    if !(sp[1] >= 0.0 && sp[1].is_finite()) {
        return Err(format!(
            "kernel_synth: jitter must be finite and >= 0, got {}",
            sp[1]
        ));
    }
    if kinds.is_empty() {
        return Err("kernel_synth: kernel bank is empty".to_string());
    }
    if kinds.len() != params.len() {
        return Err(format!(
            "kernel_synth: {} kind ids but {} parameters",
            kinds.len(),
            params.len()
        ));
    }
    let valid_kind =
        |k: f64| k.fract() == 0.0 && (KIND_RBF as f64..=KIND_CONSTANT as f64).contains(&k);
    if let Some(bad) = kinds.iter().find(|&&k| !valid_kind(k)) {
        return Err(format!("kernel_synth: unknown kernel kind id {bad}"));
    }
    Ok(())
}

/// Generate one KernelSynth series into `out`; rejected draws are retried, then fall back to noise.
pub fn kernel_synth(
    out: &mut [f64],
    sp: &[f64],
    kinds: &[f64],
    params: &[f64],
    seed: u64,
) -> Result<(), String> {
    validate(sp, kinds, params)?;
    let length = out.len();
    if length == 0 {
        return Ok(());
    }
    let max_kernels = sp[0] as i32;
    let jitter = sp[1];
    let standardize = sp[2] != 0.0;

    let x: Vec<f64> = if length > 1 {
        let step = 1.0 / (length - 1) as f64;
        (0..length).map(|i| i as f64 * step).collect()
    } else {
        vec![0.0]
    };

    let mut rng = SfRng::new(seed);
    for _ in 0..MAX_RETRIES {
        let Some(mut values) = compose_and_sample(&mut rng, max_kernels, jitter, kinds, params, &x)
        else {
            continue;
        };
        if !values.iter().all(|v| v.is_finite()) {
            continue;
        }
        if values.iter().fold(0.0_f64, |m, v| m.max(v.abs())) >= MAX_ABS {
            continue;
        }
        if length > 1 {
            let (mean, std) = mean_and_pop_std(&values);
            if std <= MIN_STD {
                continue;
            }
            if standardize {
                values.iter_mut().for_each(|v| *v = (*v - mean) / std);
            }
        }
        out.copy_from_slice(&values);
        return Ok(());
    }
    out.iter_mut().for_each(|v| *v = rng.standard_normal());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    const DEFAULT_KINDS: [f64; 9] = [0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0];
    const DEFAULT_PARAMS: [f64; 9] = [0.1, 10.0, 1.0, 7.0, 24.0, 0.0, 10.0, 0.1, 1.0];

    fn grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    fn toeplitz(c: &[f64]) -> DMatrix<f64> {
        let n = c.len();
        DMatrix::from_fn(n, n, |i, j| c[i.abs_diff(j)])
    }

    /// Dense Gram matrix of one base kernel, straight from its definition.
    fn dense_kernel(kind: i32, param: f64, x: &[f64]) -> DMatrix<f64> {
        let n = x.len();
        DMatrix::from_fn(n, n, |i, j| {
            let r = (x[i] - x[j]).abs();
            match kind {
                KIND_RBF => (-0.5 * r * r / (param * param)).exp(),
                KIND_RATIONAL_QUADRATIC => (1.0 + r * r / (2.0 * param)).powf(-param),
                KIND_PERIODIC => {
                    let s = (PI * r / (param / n as f64)).sin();
                    (-2.0 * s * s).exp()
                }
                KIND_LINEAR => param * param + x[i] * x[j],
                KIND_WHITE => {
                    if i == j {
                        param
                    } else {
                        0.0
                    }
                }
                _ => param,
            }
        })
    }

    #[test]
    fn test_terms_reproduce_dense_composition() {
        let n = 30;
        let x = grid(n);
        let compositions: [(&[usize], &[bool]); 6] = [
            (&[5], &[]),
            (&[0, 6], &[true]),
            (&[3, 6], &[false]),
            (&[6, 5, 4], &[false, false]),
            (&[2, 5, 7, 6], &[true, false, false]),
            (&[8, 6, 1, 5, 3], &[false, true, false, true]),
        ];
        for (indices, ops) in compositions {
            let mut dense = dense_kernel(
                DEFAULT_KINDS[indices[0]] as i32,
                DEFAULT_PARAMS[indices[0]],
                &x,
            );
            for (&i, &add) in indices[1..].iter().zip(ops) {
                let comp = dense_kernel(DEFAULT_KINDS[i] as i32, DEFAULT_PARAMS[i], &x);
                dense = if add {
                    dense + comp
                } else {
                    dense.component_mul(&comp)
                };
            }
            let terms = kernel_terms(indices, ops, &DEFAULT_KINDS, &DEFAULT_PARAMS, &x);
            let mut rebuilt = DMatrix::<f64>::zeros(n, n);
            for (d, c) in terms.iter().enumerate() {
                let Some(c) = c else { continue };
                let t = toeplitz(c);
                rebuilt += DMatrix::from_fn(n, n, |i, j| (x[i] * x[j]).powi(d as i32) * t[(i, j)]);
            }
            let err = (&rebuilt - &dense).abs().max();
            assert!(
                err < 1e-9 * dense.abs().max().max(1.0),
                "{indices:?}: max error {err}"
            );
        }
    }

    #[test]
    fn test_levinson_equals_cholesky_factor() {
        let n = 80;
        let x = grid(n);
        let mut rng = SfRng::new(5);
        let z: Vec<f64> = (0..n).map(|_| rng.standard_normal()).collect();
        for (kind, param) in [
            (KIND_RBF, 0.1),
            (KIND_RATIONAL_QUADRATIC, 1.0),
            (KIND_PERIODIC, 12.0),
        ] {
            let c = lag_kernel(kind, param, &x);
            let extra = 1e-3;
            let path = levinson_sample(&c, extra, &z).expect("PD kernel should not break down");
            let k = toeplitz(&c) + DMatrix::identity(n, n) * extra;
            let l = k.cholesky().expect("PD kernel").l();
            let expected = l * DVector::from_column_slice(&z);
            for (t, (p, e)) in path.iter().zip(expected.iter()).enumerate() {
                assert!((p - e).abs() < 1e-8, "kind {kind} t {t}: {p} vs {e}");
            }
        }
    }

    #[test]
    fn test_levinson_reports_breakdown() {
        let z = vec![1.0; 4];
        assert!(levinson_sample(&[0.0, 0.0, 0.0, 0.0], 0.0, &z).is_none());
        assert!(levinson_sample(&[1.0, 1.0, 1.0, 1.0], 0.0, &z).is_none());
        assert!(levinson_sample(&[1.0, -1.5, 0.5, 0.2], 0.0, &z).is_none());
        assert!(levinson_sample(&[1.0, f64::NAN, 0.0, 0.0], 0.0, &z).is_none());
    }

    #[test]
    fn test_empirical_covariance_matches_composition() {
        // (rbf(0.1) + linear(s=0)) * periodic(7): a degree-1 composition through two Levinson draws.
        let (n, draws) = (12, 50_000);
        let x = grid(n);
        let kinds = [0.0, 3.0, 2.0];
        let params = [0.1, 0.0, 7.0];
        let (indices, ops) = ([0, 1, 2], [true, false]);
        let mut rng = SfRng::new(17);
        let mut cross = vec![0.0_f64; n];
        for _ in 0..draws {
            let terms = kernel_terms(&indices, &ops, &kinds, &params, &x);
            let mut y = vec![0.0; n];
            for (d, c) in terms.iter().enumerate() {
                let Some(c) = c else { continue };
                let part = sample_stationary(&mut rng, c, 0.0).unwrap();
                for ((yv, &p), &xv) in y.iter_mut().zip(&part).zip(&x) {
                    *yv += xv.powi(d as i32) * p;
                }
            }
            let last = y[n - 1];
            for (acc, &yv) in cross.iter_mut().zip(&y) {
                *acc += last * yv;
            }
        }
        let mut dense = dense_kernel(KIND_RBF, 0.1, &x) + dense_kernel(KIND_LINEAR, 0.0, &x);
        dense = dense.component_mul(&dense_kernel(KIND_PERIODIC, 7.0, &x));
        for (t, acc) in cross.iter().enumerate() {
            let (cov, exact) = (acc / draws as f64, dense[(n - 1, t)]);
            assert!((cov - exact).abs() < 0.06, "t {t}: {cov} vs {exact}");
        }
    }

    fn default_sp() -> [f64; 3] {
        [5.0, 1e-6, 1.0]
    }

    #[test]
    fn test_kernel_synth_standardized_and_deterministic() {
        for seed in 0..40 {
            let mut a = vec![0.0; 300];
            let mut b = vec![0.0; 300];
            kernel_synth(&mut a, &default_sp(), &DEFAULT_KINDS, &DEFAULT_PARAMS, seed).unwrap();
            kernel_synth(&mut b, &default_sp(), &DEFAULT_KINDS, &DEFAULT_PARAMS, seed).unwrap();
            assert_eq!(a, b);
            let (mean, std) = mean_and_pop_std(&a);
            assert!(
                mean.abs() < 1e-9 && (std - 1.0).abs() < 1e-9,
                "seed {seed}: mean {mean} std {std}"
            );
        }
    }

    #[test]
    fn test_kernel_synth_short_lengths() {
        for length in [1, 2, 3] {
            let mut out = vec![f64::NAN; length];
            kernel_synth(&mut out, &default_sp(), &DEFAULT_KINDS, &DEFAULT_PARAMS, 3).unwrap();
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_constant_bank_without_jitter_falls_back_to_noise() {
        // Constant draws are rejected on every retry.
        let mut out = vec![0.0; 200];
        kernel_synth(&mut out, &[1.0, 0.0, 1.0], &[5.0], &[1.0], 9).unwrap();
        let (_, std) = mean_and_pop_std(&out);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!((std - 1.0).abs() < 0.2, "fallback std {std}");
    }

    /// Lag-1 autocorrelation; near 1 for a smooth path, near 0 for the white-noise fallback.
    fn lag1_autocorrelation(x: &[f64]) -> f64 {
        let (mean, std) = mean_and_pop_std(x);
        let cov = x
            .windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>()
            / (x.len() - 1) as f64;
        cov / (std * std)
    }

    #[test]
    fn test_smooth_bank_samples_without_breakdown() {
        // Near-singular smooth kernels (rbf l=10, linear s=10). A breakdown would end in the noise
        // fallback (autocorrelation ~0); escalated jitter only roughens a nearly flat path slightly.
        let mut out = vec![0.0; 1000];
        for jitter in [1e-6, 0.0] {
            for seed in 0..10 {
                kernel_synth(
                    &mut out,
                    &[3.0, jitter, 0.0],
                    &[0.0, 3.0],
                    &[10.0, 10.0],
                    seed,
                )
                .unwrap();
                assert!(out.iter().all(|v| v.is_finite()));
                let rho = lag1_autocorrelation(&out);
                assert!(
                    rho > 0.5,
                    "jitter {jitter} seed {seed}: lag-1 autocorrelation {rho}"
                );
            }
        }
    }

    #[test]
    fn test_escalation_recovers_without_jitter() {
        // rbf l=10 is numerically singular at n=1000: the unregularized recursion breaks down,
        // and the escalation ladder must still produce a draw.
        let n = 1000;
        let c = lag_kernel(KIND_RBF, 10.0, &grid(n));
        let z = vec![1.0; n];
        assert!(levinson_sample(&c, 0.0, &z).is_none());
        let mut rng = SfRng::new(4);
        let path = sample_stationary(&mut rng, &c, 0.0).expect("escalation should recover");
        assert!(lag1_autocorrelation(&path) > 0.99);
    }

    #[test]
    fn test_dot_reversed_matches_naive() {
        for n in 0..14 {
            let a: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
            let b: Vec<f64> = (0..n).map(|i| 0.5 - 0.25 * i as f64).collect();
            let naive: f64 = (0..n).map(|k| a[k] * b[n - 1 - k]).sum();
            assert!((dot_reversed(&a, &b) - naive).abs() < 1e-12, "n {n}");
        }
    }

    #[test]
    fn test_rejects_malformed_params() {
        let mut out = vec![0.0; 10];
        let kinds = [0.0, 3.0];
        let params = [0.1, 1.0];
        let bad: [(&[f64], &[f64], &[f64]); 9] = [
            (&[5.0, 1e-6], &kinds, &params),
            (&[0.0, 1e-6, 1.0], &kinds, &params),
            (&[65.0, 1e-6, 1.0], &kinds, &params),
            (&[f64::NAN, 1e-6, 1.0], &kinds, &params),
            (&[5.0, -1.0, 1.0], &kinds, &params),
            (&[5.0, 1e-6, 1.0], &[], &[]),
            (&[5.0, 1e-6, 1.0], &kinds, &[0.1]),
            (&[5.0, 1e-6, 1.0], &[0.0, 6.0], &params),
            (&[5.0, 1e-6, 1.0], &[0.0, 1.5], &params),
        ];
        for (sp, k, p) in bad {
            assert!(
                kernel_synth(&mut out, sp, k, p, 1).is_err(),
                "{sp:?} {k:?} {p:?}"
            );
        }
        assert!(kernel_synth(&mut out, &[64.0, 0.0, 1.0], &kinds, &params, 1).is_ok());
    }
}
