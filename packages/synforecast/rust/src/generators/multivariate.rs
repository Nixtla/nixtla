use crate::distributions;
use crate::fft;
use crate::linalg;
use crate::rng::SfRng;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Copula
// ---------------------------------------------------------------------------

/// Copula generator.
///
/// copula_type: 0=gaussian, 1=t
/// marginal_distribution: 0=normal, 1=lognormal, 2=exponential, 3=uniform, 4=gamma
pub fn copula(
    out: &mut [f64],
    n_variables: usize,
    copula_type: i32,
    df: f64,
    correlation_matrix: &[f64],
    marginal_distribution: i32,
    marginal_param1: f64,
    marginal_param2: f64,
    seed: u64,
) {
    let mut rng = SfRng::new(seed);

    // Build correlation matrix from flat array
    let corr_mat = linalg::flat_to_matrix(correlation_matrix, n_variables, n_variables);

    // Cholesky decomposition with fallback for non-PD matrices
    let l = linalg::cholesky_with_fallback(&corr_mat);

    let mut z = DVector::zeros(n_variables);
    let mut u = DVector::zeros(n_variables);

    for out_v in out.iter_mut() {
        // Generate z ~ N(0, I)
        for i in 0..n_variables {
            z[i] = rng.standard_normal();
        }

        // Correlate: z_corr = L * z
        let z_corr = &l * &z;

        if copula_type == 0 {
            // Gaussian copula: u[i] = Phi(z_corr[i])
            for i in 0..n_variables {
                u[i] = distributions::norm_cdf(z_corr[i]);
            }
        } else {
            // t copula: chi-squared via gamma distribution (handles non-integer df)
            let chi2 = rng.gamma(df / 2.0, 2.0);
            // u[i] = t_cdf(z_corr[i] * sqrt(df / chi2), df)
            let scale = (df / chi2).sqrt();
            for i in 0..n_variables {
                u[i] = distributions::t_cdf(z_corr[i] * scale, df);
            }
        }

        // Transform marginals for each variable
        // We output only the first variable (column 0)
        let mut u0 = u[0];

        // Clamp u0 to avoid boundary issues with inverse CDF
        u0 = u0.clamp(1e-15, 1.0 - 1e-15);

        let x = match marginal_distribution {
            0 => {
                // Normal: param1=loc, param2=scale
                marginal_param1 + marginal_param2 * distributions::norm_ppf(u0)
            }
            1 => {
                // Lognormal: param1=s, param2=scale
                distributions::lognorm_ppf(u0, marginal_param1, marginal_param2)
            }
            2 => {
                // Exponential: param1=scale
                distributions::expon_ppf(u0, marginal_param1)
            }
            3 => {
                // Uniform: param1=loc, param2=scale
                distributions::uniform_ppf(u0, marginal_param1, marginal_param2)
            }
            4 => {
                // Gamma: param1=a (shape), param2=scale
                distributions::gamma_ppf(u0, marginal_param1, marginal_param2)
            }
            _ => {
                // Default to standard normal
                distributions::norm_ppf(u0)
            }
        };

        *out_v = x;
    }
}

// ---------------------------------------------------------------------------
// VAR (Vector Autoregressive)
// ---------------------------------------------------------------------------

/// Vector Autoregressive (VAR) process.
///
/// coef_matrices: flat array of shape [order, n_variables, n_variables]
/// intercept: [n_variables]
/// innovation_cov: [n_variables, n_variables]
pub fn var_process(
    out: &mut [f64],
    n_variables: usize,
    order: usize,
    coef_matrices: &[f64],
    intercept: &[f64],
    innovation_cov: &[f64],
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let length = out.len();
    let mut rng = SfRng::new(seed);
    let n = n_variables;

    // Map intercept to vector
    let c = DVector::from_row_slice(intercept);

    // Map coefficient matrices to vector of matrices
    let mut a_mats: Vec<DMatrix<f64>> = Vec::with_capacity(order);
    for lag in 0..order {
        let start = lag * n * n;
        let mat = linalg::flat_to_matrix(&coef_matrices[start..start + n * n], n, n);
        a_mats.push(mat);
    }

    // Map innovation covariance and Cholesky decompose
    let cov = linalg::flat_to_matrix(innovation_cov, n, n);
    let l = linalg::cholesky_with_fallback(&cov);

    // Total steps: burn-in + requested length
    const BURNIN: usize = 100;
    let total = BURNIN + length;

    // Store multivariate series: total steps x n_variables
    let mut series: Vec<DVector<f64>> = (0..total).map(|_| DVector::zeros(n)).collect();

    let mut z = DVector::zeros(n);

    for t in 0..total {
        // Start with intercept
        let mut val = c.clone();

        // Add AR terms
        for (lag, a_mat) in a_mats.iter().enumerate() {
            let idx = t as i64 - lag as i64 - 1;
            if idx >= 0 {
                val += a_mat * &series[idx as usize];
            }
        }

        // Add innovation: L * z where z ~ innovation_dist(0, 1)
        for i in 0..n {
            z[i] = rng.sample_innovation(1.0, innov_dist, innov_param);
        }
        val += &l * &z;

        series[t] = val;
    }

    // Output column 0 (first variable) after burn-in
    for t in 0..length {
        out[t] = series[BURNIN + t][0];
    }
}

// ---------------------------------------------------------------------------
// State Space
// ---------------------------------------------------------------------------

/// General linear-Gaussian state-space model.
///
/// F_mat: [state_dim x state_dim] transition matrix
/// H_mat: [obs_dim x state_dim] observation matrix
/// Q_mat: [state_dim x state_dim] process noise covariance
/// R_mat: [obs_dim x obs_dim] observation noise covariance
/// initial_state: [state_dim] initial state mean
/// initial_state_cov: [state_dim x state_dim] initial state covariance
///
/// Follows the canonical simulation convention of the Python implementation
/// (synforecast/generators/state_space.py): x[0] ~ N(initial_state, P0),
/// each step observes the current state (y[t] = H x[t] + v) and then
/// transitions (x[t+1] = F x[t] + w).
#[allow(clippy::too_many_arguments)]
pub fn state_space(
    out: &mut [f64],
    state_dim: usize,
    obs_dim: usize,
    f_mat: &[f64],
    h_mat: &[f64],
    q_mat: &[f64],
    r_mat: &[f64],
    initial_state: &[f64],
    initial_state_cov: &[f64],
    seed: u64,
    innov_dist: i32,
    innov_param: f64,
) {
    let mut rng = SfRng::new(seed);

    // Map flat arrays to matrices
    let f = linalg::flat_to_matrix(f_mat, state_dim, state_dim);
    let h = linalg::flat_to_matrix(h_mat, obs_dim, state_dim);
    let q = linalg::flat_to_matrix(q_mat, state_dim, state_dim);
    let r = linalg::flat_to_matrix(r_mat, obs_dim, obs_dim);
    let p0 = linalg::flat_to_matrix(initial_state_cov, state_dim, state_dim);

    // Cholesky decompositions for noise generation
    let l_q = linalg::cholesky_with_fallback(&q);
    let l_r = linalg::cholesky_with_fallback(&r);
    let l_p0 = linalg::cholesky_with_fallback(&p0);

    // Initialize state: x[0] ~ N(initial_state, P0)
    let mut z_q = DVector::zeros(state_dim);
    let mut z_r = DVector::zeros(obs_dim);
    for i in 0..state_dim {
        z_q[i] = rng.standard_normal();
    }
    let mut x = DVector::from_row_slice(initial_state) + &l_p0 * &z_q;

    for out_v in out.iter_mut() {
        // Observation noise: v = L_R * z_r where z_r ~ N(0, I)
        for i in 0..obs_dim {
            z_r[i] = rng.standard_normal();
        }
        let v = &l_r * &z_r;

        // Observe the current state: y = H * x + v
        let y = &h * &x + &v;

        // Output first observation dimension
        *out_v = y[0];

        // Process noise: w = L_Q * z_q where z_q ~ innovation_dist(0, 1)
        for i in 0..state_dim {
            z_q[i] = rng.sample_innovation(1.0, innov_dist, innov_param);
        }
        let w = &l_q * &z_q;

        // State transition: x = F * x + w
        x = &f * &x + &w;
    }
}

// ---------------------------------------------------------------------------
// Fractional Brownian Motion
// ---------------------------------------------------------------------------

/// Autocovariance function for fractional Gaussian noise
#[inline]
fn fbm_autocovariance(k: i32, hurst: f64, sigma: f64) -> f64 {
    let two_h = 2.0 * hurst;
    let abs_k = (k as f64).abs();
    (sigma * sigma / 2.0)
        * ((abs_k + 1.0).powf(two_h) - 2.0 * abs_k.powf(two_h) + (abs_k - 1.0).abs().powf(two_h))
}

/// Cholesky method for fractional Brownian motion / fractional Gaussian noise
pub fn fbm_cholesky(
    out: &mut [f64],
    hurst: f64,
    sigma: f64,
    initial_value: f64,
    cumulative: bool,
    seed: u64,
) {
    let n = out.len();
    let mut rng = SfRng::new(seed);

    // Pre-compute the n unique autocovariance values (Toeplitz structure)
    let mut acov = vec![0.0_f64; n];
    for (k, acov_v) in acov.iter_mut().enumerate() {
        *acov_v = fbm_autocovariance(k as i32, hurst, sigma);
    }

    // Build covariance matrix directly via DMatrix::from_fn (avoids intermediate vec)
    let c_mat = DMatrix::from_fn(n, n, |i, j| {
        let diff = j.abs_diff(i);
        acov[diff]
    });

    // Cholesky decomposition with fallback
    let l = linalg::cholesky_with_fallback(&c_mat);

    // Generate z ~ N(0, I)
    let mut z = DVector::zeros(n);
    for i in 0..n {
        z[i] = rng.standard_normal();
    }

    // Compute fGn = L * z
    let fgn = &l * &z;

    if cumulative {
        // Cumulative sum + initial value
        let mut cumsum = initial_value;
        for i in 0..n {
            cumsum += fgn[i];
            out[i] = cumsum;
        }
    } else {
        // Raw fGn increments; initial_value only applies in cumulative mode
        for (out_v, &fgn_v) in out.iter_mut().zip(fgn.iter()) {
            *out_v = fgn_v;
        }
    }
}

/// Hosking method (Durbin-Levinson) for fractional Brownian motion
pub fn fbm_hosking(
    out: &mut [f64],
    hurst: f64,
    sigma: f64,
    initial_value: f64,
    cumulative: bool,
    seed: u64,
) {
    let n = out.len();
    let mut rng = SfRng::new(seed);

    // Pre-compute autocovariance lookup table to avoid redundant pow() calls
    let mut acov = vec![0.0_f64; n];
    for (k, acov_v) in acov.iter_mut().enumerate() {
        *acov_v = fbm_autocovariance(k as i32, hurst, sigma);
    }

    // Ping-pong buffers for Durbin-Levinson coefficients (O(n) memory)
    let mut phi_prev = vec![0.0_f64; n];
    let mut phi_curr = vec![0.0_f64; n];
    let mut v = vec![0.0_f64; n];
    let mut fgn = vec![0.0_f64; n];

    // Initialize
    v[0] = acov[0];
    fgn[0] = v[0].sqrt() * rng.standard_normal();

    // Durbin-Levinson recursion (1-indexed coefficients)
    for i in 1..n {
        // Compute reflection coefficient phi_curr[i]
        if i == 1 {
            phi_curr[i] = acov[i] / v[0];
        } else {
            let mut sum_num = 0.0;
            for k in 1..i {
                sum_num += phi_prev[k] * acov[i - k];
            }
            phi_curr[i] = (acov[i] - sum_num) / v[i - 1];

            // Update phi_curr[k] for 1 <= k < i
            for k in 1..i {
                phi_curr[k] = phi_prev[k] - phi_curr[i] * phi_prev[i - k];
            }
        }

        // Update variance
        v[i] = v[i - 1] * (1.0 - phi_curr[i] * phi_curr[i]);

        // Compute conditional mean using iterator pattern for better vectorization
        let mu: f64 = phi_curr[1..=i]
            .iter()
            .zip(fgn[..i].iter().rev())
            .map(|(a, b)| a * b)
            .sum();

        // Generate fGn sample
        fgn[i] = mu + v[i].max(0.0).sqrt() * rng.standard_normal();

        // Swap buffers for next iteration; no need to zero — positions
        // 1..=i are always fully overwritten before being read next iteration
        std::mem::swap(&mut phi_prev, &mut phi_curr);
    }

    if cumulative {
        let mut cumsum = initial_value;
        for i in 0..n {
            cumsum += fgn[i];
            out[i] = cumsum;
        }
    } else {
        // Raw fGn increments; initial_value only applies in cumulative mode
        out[..n].copy_from_slice(&fgn[..n]);
    }
}

/// Davies-Harte (circulant embedding + FFT) method for fractional Gaussian noise.
/// O(n log n) time, O(n) memory.  Exact for H > 0.5; pads for H < 0.5.
/// Negative eigenvalues are clamped to zero, so this always succeeds.
pub fn fbm_fft(
    out: &mut [f64],
    hurst: f64,
    sigma: f64,
    initial_value: f64,
    cumulative: bool,
    seed: u64,
) {
    let n = out.len();
    let mut rng = SfRng::new(seed);

    // Circulant size: 2 * next_pow2(n) to ensure power-of-2 for FFT
    let m = 2 * fft::next_pow2(n);

    // Build first row of the circulant matrix from autocovariance
    let mut row = vec![0.0_f64; m];
    for (k, row_v) in row.iter_mut().enumerate().take(m / 2 + 1) {
        *row_v = fbm_autocovariance(k as i32, hurst, sigma);
    }
    // Mirror: row[m-k] = row[k]
    for k in 1..m / 2 {
        row[m - k] = row[k];
    }

    // FFT of the first row -> eigenvalues of the circulant matrix
    let mut eigenvalues = fft::rfft(&row);

    // Clamp negative eigenvalues to zero
    for i in 0..m {
        if i < eigenvalues.len() && eigenvalues[i].re < 0.0 {
            eigenvalues[i] = Complex64::new(0.0, 0.0);
        }
    }
    // Ensure we have m entries
    eigenvalues.resize(m, Complex64::new(0.0, 0.0));

    // Generate complex Gaussian noise in frequency domain, scale by sqrt(eigenvalues)
    let mut z = vec![Complex64::new(0.0, 0.0); m];
    // z[0] and z[m/2] are real-valued
    z[0] = Complex64::new(eigenvalues[0].re.sqrt() * rng.standard_normal(), 0.0);
    z[m / 2] = Complex64::new(eigenvalues[m / 2].re.sqrt() * rng.standard_normal(), 0.0);
    for k in 1..m / 2 {
        let sq = (eigenvalues[k].re / 2.0).sqrt();
        let re = sq * rng.standard_normal();
        let im = sq * rng.standard_normal();
        z[k] = Complex64::new(re, im);
        z[m - k] = Complex64::new(re, -im); // Hermitian symmetry
    }

    // IFFT to get samples (real part of first n entries is fGn)
    fft::fft_radix2(&mut z, true); // inverse FFT (normalizes by 1/m)

    // Compensate for the 1/m normalization in the IFFT: scale by sqrt(m)
    // The Davies-Harte method assumes unnormalized IFFT; our IFFT divides by m,
    // so each sample is sqrt(m) times too small in amplitude.
    let scale = (m as f64).sqrt();

    if cumulative {
        let mut cumsum = initial_value;
        for i in 0..n {
            cumsum += z[i].re * scale;
            out[i] = cumsum;
        }
    } else {
        // Raw fGn increments; initial_value only applies in cumulative mode
        for i in 0..n {
            out[i] = z[i].re * scale;
        }
    }
}

/// Wrapper: method=0 for cholesky, method=1 for hosking, method=2 for fft
pub fn fbm(
    out: &mut [f64],
    hurst: f64,
    sigma: f64,
    initial_value: f64,
    cumulative: bool,
    method: i32,
    seed: u64,
) {
    if method == 2 {
        // FFT method (Davies-Harte) — clamps negative eigenvalues, always succeeds
        fbm_fft(out, hurst, sigma, initial_value, cumulative, seed);
    } else if method == 1 {
        fbm_hosking(out, hurst, sigma, initial_value, cumulative, seed);
    } else {
        fbm_cholesky(out, hurst, sigma, initial_value, cumulative, seed);
    }
}

// ---------------------------------------------------------------------------
// Gaussian Process
// ---------------------------------------------------------------------------

/// Kernel functions: given distance r, return covariance
#[inline]
fn gp_kernel(r: f64, kernel_id: i32, length_scale: f64, amplitude: f64, period: f64) -> f64 {
    let a2 = amplitude * amplitude;
    let ls = length_scale;

    match kernel_id {
        0 => {
            // RBF
            a2 * (-0.5 * (r / ls) * (r / ls)).exp()
        }
        1 => {
            // Matern 0.5
            a2 * (-r / ls).exp()
        }
        2 => {
            // Matern 1.5
            let s = 3.0_f64.sqrt() * r / ls;
            a2 * (1.0 + s) * (-s).exp()
        }
        3 => {
            // Matern 2.5
            let s = 5.0_f64.sqrt() * r / ls;
            a2 * (1.0 + s + s * s / 3.0) * (-s).exp()
        }
        4 => {
            // Periodic
            let sin_val = (PI * r / period).sin();
            a2 * (-2.0 * (sin_val / ls) * (sin_val / ls)).exp()
        }
        _ => {
            // Default to RBF
            a2 * (-0.5 * (r / ls) * (r / ls)).exp()
        }
    }
}

/// FFT-based sampling for stationary kernels on regular 1D grids.
/// Uses circulant embedding: O(n log n) time, O(n) memory.
/// Negative eigenvalues are clamped to zero (standard approximate fix).
pub fn gaussian_process_fft(
    out: &mut [f64],
    kernel_id: i32,
    length_scale: f64,
    amplitude: f64,
    period: f64,
    mean: f64,
    noise_variance: f64,
    seed: u64,
) {
    let n = out.len();
    let mut rng = SfRng::new(seed);

    // Circulant size: 2 * next_pow2(n)
    let m = 2 * fft::next_pow2(n);

    // Build first row of circulant from kernel function
    let mut row = vec![0.0_f64; m];
    for (k, row_v) in row.iter_mut().enumerate().take(m / 2 + 1) {
        let r = k as f64;
        let mut val = gp_kernel(r, kernel_id, length_scale, amplitude, period);
        if k == 0 {
            val += noise_variance;
        }
        *row_v = val;
    }
    // Mirror
    for k in 1..m / 2 {
        row[m - k] = row[k];
    }

    // FFT of first row -> eigenvalues
    let mut eigenvalues = fft::rfft(&row);

    // Clamp negative eigenvalues to zero
    for i in 0..eigenvalues.len().min(m) {
        if eigenvalues[i].re < 0.0 {
            eigenvalues[i] = Complex64::new(0.0, 0.0);
        }
    }
    // Ensure we have m entries
    eigenvalues.resize(m, Complex64::new(0.0, 0.0));

    // Generate complex Gaussian noise, scale by sqrt(eigenvalues)
    let mut z = vec![Complex64::new(0.0, 0.0); m];
    z[0] = Complex64::new(eigenvalues[0].re.sqrt() * rng.standard_normal(), 0.0);
    z[m / 2] = Complex64::new(eigenvalues[m / 2].re.sqrt() * rng.standard_normal(), 0.0);
    for k in 1..m / 2 {
        let sq = (eigenvalues[k].re / 2.0).sqrt();
        let re = sq * rng.standard_normal();
        let im = sq * rng.standard_normal();
        z[k] = Complex64::new(re, im);
        z[m - k] = Complex64::new(re, -im);
    }

    // IFFT (normalizes by 1/m)
    fft::fft_radix2(&mut z, true);

    // Compensate for the 1/m normalization: scale by sqrt(m)
    let scale = (m as f64).sqrt();

    for i in 0..n {
        out[i] = mean + z[i].re * scale;
    }
}

/// Cholesky-based GP sampling (fallback for non-stationary or when FFT fails)
pub fn gaussian_process_cholesky(
    out: &mut [f64],
    kernel_id: i32,
    length_scale: f64,
    amplitude: f64,
    period: f64,
    mean: f64,
    noise_variance: f64,
    seed: u64,
) {
    let n = out.len();
    let mut rng = SfRng::new(seed);

    let mut k_data = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let r = (j - i) as f64;
            let mut val = gp_kernel(r, kernel_id, length_scale, amplitude, period);
            if i == j {
                val += noise_variance;
            }
            k_data[i * n + j] = val;
            k_data[j * n + i] = val;
        }
    }
    let k_mat = linalg::flat_to_matrix(&k_data, n, n);

    let l = linalg::cholesky_with_fallback(&k_mat);

    let mut z = DVector::zeros(n);
    for i in 0..n {
        z[i] = rng.standard_normal();
    }

    let f_vec = &l * &z;
    for i in 0..n {
        out[i] = mean + f_vec[i];
    }
}

/// Gaussian process wrapper.
/// Uses FFT-based circulant embedding (O(n log n)) for rapidly-decaying
/// stationary kernels. The periodic kernel's circulant embedding is strongly
/// non-PSD when the period does not divide the padded size, and clamping the
/// negative eigenvalues inflates the marginal variance and attenuates the
/// periodicity — so periodic kernels use exact Cholesky sampling instead.
pub fn gaussian_process(
    out: &mut [f64],
    kernel_id: i32,
    length_scale: f64,
    amplitude: f64,
    period: f64,
    mean: f64,
    noise_variance: f64,
    seed: u64,
) {
    if kernel_id == 4 {
        gaussian_process_cholesky(
            out,
            kernel_id,
            length_scale,
            amplitude,
            period,
            mean,
            noise_variance,
            seed,
        );
    } else {
        gaussian_process_fft(
            out,
            kernel_id,
            length_scale,
            amplitude,
            period,
            mean,
            noise_variance,
            seed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 64; // Power of 2 for FFT-based methods

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

    // --- Copula ---

    #[test]
    fn test_copula_gaussian_finite() {
        let corr = [1.0, 0.5, 0.5, 1.0]; // 2x2 correlation
        let mut out = vec![0.0; N];
        copula(&mut out, 2, 0, 5.0, &corr, 0, 0.0, 1.0, 42);
        assert_finite(&out, "copula_gaussian");
    }

    #[test]
    fn test_copula_t_finite() {
        let corr = [1.0, 0.5, 0.5, 1.0];
        let mut out = vec![0.0; N];
        copula(&mut out, 2, 1, 5.0, &corr, 0, 0.0, 1.0, 42);
        assert_finite(&out, "copula_t");
    }

    #[test]
    fn test_copula_lognormal_marginal() {
        let corr = [1.0, 0.0, 0.0, 1.0]; // independent
        let mut out = vec![0.0; N];
        copula(&mut out, 2, 0, 5.0, &corr, 1, 0.5, 1.0, 42);
        assert_finite(&out, "copula_lognormal");
        for &v in &out {
            assert!(v > 0.0, "lognormal marginal should be positive");
        }
    }

    #[test]
    fn test_copula_uniform_marginal() {
        let corr = [1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; N];
        copula(&mut out, 2, 0, 5.0, &corr, 3, 0.0, 1.0, 42);
        assert_finite(&out, "copula_uniform");
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "uniform marginal should be in [0,1]"
            );
        }
    }

    #[test]
    fn test_copula_deterministic() {
        let corr = [1.0, 0.5, 0.5, 1.0];
        assert_deterministic(|o| copula(o, 2, 0, 5.0, &corr, 0, 0.0, 1.0, 42), "copula");
    }

    // --- VAR ---

    #[test]
    fn test_var_process_finite() {
        // 2-variable VAR(1)
        let coef = [0.5, 0.1, 0.2, 0.3]; // 1 lag, 2x2
        let intercept = [0.0, 0.0];
        let innov_cov = [1.0, 0.0, 0.0, 1.0]; // identity
        let mut out = vec![0.0; N];
        var_process(&mut out, 2, 1, &coef, &intercept, &innov_cov, 42, 0, 0.0);
        assert_finite(&out, "var");
    }

    #[test]
    fn test_var_process_deterministic() {
        let coef = [0.5, 0.1, 0.2, 0.3];
        let intercept = [0.0, 0.0];
        let innov_cov = [1.0, 0.0, 0.0, 1.0];
        assert_deterministic(
            |o| var_process(o, 2, 1, &coef, &intercept, &innov_cov, 42, 0, 0.0),
            "var",
        );
    }

    // --- State Space ---

    #[test]
    fn test_state_space_finite() {
        // Simple 1D state space: random walk + noise
        let f = [1.0]; // F = [1] (random walk)
        let h = [1.0]; // H = [1] (direct observation)
        let q = [0.1]; // process noise variance
        let r = [0.5]; // observation noise variance
        let x0 = [0.0]; // initial state
        let p0 = [1.0]; // initial state covariance
        let mut out = vec![0.0; N];
        state_space(&mut out, 1, 1, &f, &h, &q, &r, &x0, &p0, 42, 0, 0.0);
        assert_finite(&out, "state_space");
    }

    #[test]
    fn test_state_space_2d() {
        // 2D state, 1D observation
        let f = [1.0, 0.1, 0.0, 0.9]; // state transition
        let h = [1.0, 0.0]; // observe first state only
        let q = [0.1, 0.0, 0.0, 0.1]; // process noise
        let r = [0.5]; // observation noise
        let x0 = [0.0, 0.0];
        let p0 = [1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; N];
        state_space(&mut out, 2, 1, &f, &h, &q, &r, &x0, &p0, 42, 0, 0.0);
        assert_finite(&out, "state_space_2d");
    }

    #[test]
    fn test_state_space_deterministic() {
        let f = [1.0];
        let h = [1.0];
        let q = [0.1];
        let r = [0.5];
        let x0 = [0.0];
        let p0 = [1.0];
        assert_deterministic(
            |o| state_space(o, 1, 1, &f, &h, &q, &r, &x0, &p0, 42, 0, 0.0),
            "state_space",
        );
    }

    // --- Fractional Brownian Motion ---

    #[test]
    fn test_fbm_cholesky_finite() {
        let mut out = vec![0.0; N];
        fbm(&mut out, 0.7, 1.0, 0.0, true, 0, 42);
        assert_finite(&out, "fbm_cholesky");
    }

    #[test]
    fn test_fbm_hosking_finite() {
        let mut out = vec![0.0; N];
        fbm(&mut out, 0.7, 1.0, 0.0, true, 1, 42);
        assert_finite(&out, "fbm_hosking");
    }

    #[test]
    fn test_fbm_fft_finite() {
        let mut out = vec![0.0; N];
        fbm(&mut out, 0.7, 1.0, 0.0, true, 2, 42);
        assert_finite(&out, "fbm_fft");
    }

    #[test]
    fn test_fbm_initial_value() {
        let mut out = vec![0.0; N];
        fbm(&mut out, 0.7, 0.001, 100.0, true, 0, 42);
        // With tiny sigma, cumulative should stay near initial_value
        assert!(
            (out[0] - 100.0).abs() < 1.0,
            "first value should be near 100"
        );
    }

    #[test]
    fn test_fbm_deterministic() {
        assert_deterministic(|o| fbm(o, 0.7, 1.0, 0.0, true, 0, 42), "fbm");
    }

    #[test]
    fn test_fbm_increments_ignore_initial_value() {
        // In increments mode (cumulative=false), initial_value must not leak
        // into the first increment for any method.
        for method in 0..3 {
            let mut with_iv = vec![0.0; N];
            let mut without_iv = vec![0.0; N];
            fbm(&mut with_iv, 0.7, 1.0, 100.0, false, method, 42);
            fbm(&mut without_iv, 0.7, 1.0, 0.0, false, method, 42);
            assert_eq!(
                with_iv, without_iv,
                "method {method}: increments must be independent of initial_value"
            );
        }
    }

    #[test]
    fn test_fbm_autocovariance_at_zero() {
        // gamma(0) = sigma^2
        let cov = fbm_autocovariance(0, 0.7, 2.0);
        assert!(
            (cov - 4.0).abs() < 1e-10,
            "gamma(0) should be sigma^2=4, got {cov}"
        );
    }

    // --- Gaussian Process ---

    #[test]
    fn test_gaussian_process_rbf_finite() {
        let mut out = vec![0.0; N];
        gaussian_process(&mut out, 0, 10.0, 1.0, 1.0, 0.0, 0.01, 42);
        assert_finite(&out, "gp_rbf");
    }

    #[test]
    fn test_gaussian_process_matern_kernels() {
        for kernel_id in 1..=3 {
            let mut out = vec![0.0; N];
            gaussian_process(&mut out, kernel_id, 10.0, 1.0, 1.0, 0.0, 0.01, 42);
            assert_finite(&out, &format!("gp_matern_{kernel_id}"));
        }
    }

    #[test]
    fn test_gaussian_process_periodic() {
        let mut out = vec![0.0; N];
        gaussian_process(&mut out, 4, 5.0, 1.0, 20.0, 0.0, 0.01, 42);
        assert_finite(&out, "gp_periodic");
    }

    #[test]
    fn test_gaussian_process_mean_offset() {
        let mean = 50.0;
        let mut out = vec![0.0; N];
        gaussian_process(&mut out, 0, 10.0, 0.001, 1.0, mean, 0.0001, 42);
        // With tiny amplitude and noise, values should be near mean
        let avg: f64 = out.iter().sum::<f64>() / N as f64;
        assert!((avg - mean).abs() < 1.0, "mean should be ~50, got {avg}");
    }

    #[test]
    fn test_gaussian_process_deterministic() {
        assert_deterministic(
            |o| gaussian_process(o, 0, 10.0, 1.0, 1.0, 0.0, 0.01, 42),
            "gp",
        );
    }

    #[test]
    fn test_gp_kernel_rbf_at_zero() {
        // k(0) = amplitude^2 for RBF
        let val = gp_kernel(0.0, 0, 10.0, 2.0, 1.0);
        assert!((val - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gp_kernel_rbf_decays() {
        let k0 = gp_kernel(0.0, 0, 10.0, 1.0, 1.0);
        let k1 = gp_kernel(5.0, 0, 10.0, 1.0, 1.0);
        let k2 = gp_kernel(20.0, 0, 10.0, 1.0, 1.0);
        assert!(k0 > k1 && k1 > k2, "RBF kernel should decay with distance");
    }
}
