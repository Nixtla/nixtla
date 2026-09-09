//! Fourier transforms: radix-2 Cooley-Tukey, Bluestein for arbitrary lengths,
//! and RealFFT for the real half-spectrum. Bluestein (1968), "A linear filtering
//! approach to the computation of the discrete Fourier transform", NEREM 10,
//! pp. 218-219; bibliography: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.CZT.html
//! See GENERATORS.md for references and THIRD_PARTY_NOTICES.md for dependencies.

use num_complex::Complex64;
use realfft::{RealFftPlanner, RealToComplex};
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

const REAL_FFT_PLAN_CACHE_CAPACITY: usize = 16;
type RealFftPlan = Arc<dyn RealToComplex<f64>>;

#[derive(Default)]
struct RealFftPlanCache {
    plans: HashMap<usize, RealFftPlan>,
    order: VecDeque<usize>,
}

fn real_fft_plan(len: usize) -> RealFftPlan {
    static CACHE: OnceLock<Mutex<RealFftPlanCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(RealFftPlanCache::default()));
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if let Some(plan) = cache.plans.get(&len).cloned() {
        cache.order.retain(|cached_len| *cached_len != len);
        cache.order.push_back(len);
        return plan;
    }

    // Plans own their shared internal data, so the planner can be dropped.
    // Creating it per cache miss keeps this cache genuinely bounded instead
    // of retaining the planner's own unbounded history of requested lengths.
    let mut planner = RealFftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(len);
    if cache.plans.len() == REAL_FFT_PLAN_CACHE_CAPACITY {
        if let Some(evicted) = cache.order.pop_front() {
            cache.plans.remove(&evicted);
        }
    }
    cache.plans.insert(len, Arc::clone(&plan));
    cache.order.push_back(len);
    plan
}

/// Minimal radix-2 Cooley-Tukey FFT implementation.
pub fn fft_radix2(x: &mut [Complex64], inverse: bool) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            x.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * 2.0 * PI / len as f64;
        let wn = Complex64::new(angle.cos(), angle.sin());
        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for jj in 0..len / 2 {
                let u = x[i + jj];
                let v = x[i + jj + len / 2] * w;
                x[i + jj] = u + v;
                x[i + jj + len / 2] = u - v;
                w *= wn;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for val in x.iter_mut() {
            *val *= inv_n;
        }
    }
}

/// Forward DFT of arbitrary length via Bluestein's chirp-z algorithm.
///
/// Reduces a length-`n` DFT to a convolution evaluated with a radix-2 FFT of
/// size `next_pow2(2n - 1)`, so the cost is O(n log n) for every `n`.
pub fn fft_bluestein(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    let m = next_pow2(2 * n - 1);
    let chirp: Vec<Complex64> = (0..n)
        .map(|k| {
            // Reduce k^2 mod 2n before scaling to keep the angle small.
            let k2 = (k * k) % (2 * n);
            let angle = -PI * k2 as f64 / n as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();
    let mut a = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..n {
        a[k] = x[k] * chirp[k];
    }
    let mut b = vec![Complex64::new(0.0, 0.0); m];
    b[0] = chirp[0].conj();
    for k in 1..n {
        b[k] = chirp[k].conj();
        b[m - k] = chirp[k].conj();
    }
    fft_radix2(&mut a, false);
    fft_radix2(&mut b, false);
    for (av, bv) in a.iter_mut().zip(&b) {
        *av *= bv;
    }
    fft_radix2(&mut a, true);
    (0..n).map(|k| a[k] * chirp[k]).collect()
}

/// Real-to-complex FFT for any length (radix-2 when `n` is a power of two).
pub fn rfft(data: &[f64]) -> Vec<Complex64> {
    let n = data.len();
    let mut x: Vec<Complex64> = data.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    if n.is_power_of_two() {
        fft_radix2(&mut x, false);
        x
    } else {
        fft_bluestein(&x)
    }
}

/// Real-to-complex FFT retaining only the non-redundant half spectrum.
///
/// All lengths use a cached RealFFT plan, with output and scratch storage allocated
/// per call so concurrent feature computations never share mutable buffers.
pub fn rfft_half(data: &mut [f64]) -> Result<Vec<Complex64>, String> {
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let plan = real_fft_plan(n);
    let mut spectrum = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();
    plan.process_with_scratch(data, &mut spectrum, &mut scratch)
        .map_err(|error| format!("real FFT failed: {error}"))?;
    Ok(spectrum)
}

/// Complex-to-real IFFT
pub fn irfft(x: &mut Vec<Complex64>, out: &mut [f64]) {
    let n = out.len();
    x.resize(n, Complex64::new(0.0, 0.0));
    fft_radix2(x, true);
    for i in 0..n {
        out[i] = x[i].re;
    }
}

/// Next power of 2 >= n
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    n.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(8), 8);
        assert_eq!(next_pow2(9), 16);
    }

    #[test]
    fn test_fft_radix2_single_element() {
        let mut x = vec![Complex64::new(5.0, 0.0)];
        fft_radix2(&mut x, false);
        assert!((x[0].re - 5.0).abs() < TOL);
        assert!(x[0].im.abs() < TOL);
    }

    #[test]
    fn test_fft_radix2_roundtrip() {
        // forward then inverse should recover the original signal
        let original = [1.0, 2.0, 3.0, 4.0];
        let mut x: Vec<Complex64> = original.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        fft_radix2(&mut x, false);
        fft_radix2(&mut x, true);
        for (i, &val) in original.iter().enumerate() {
            assert!(
                (x[i].re - val).abs() < TOL,
                "mismatch at index {i}: got {}, expected {val}",
                x[i].re
            );
            assert!(
                x[i].im.abs() < TOL,
                "imaginary part nonzero at index {i}: {}",
                x[i].im
            );
        }
    }

    #[test]
    fn test_fft_dc_component() {
        // FFT of constant signal: DC = N * value, all others = 0
        let n = 8;
        let val = 3.0;
        let mut x: Vec<Complex64> = vec![Complex64::new(val, 0.0); n];
        fft_radix2(&mut x, false);
        assert!((x[0].re - (n as f64) * val).abs() < TOL);
        for (k, value) in x.iter().enumerate().skip(1) {
            assert!(value.norm() < TOL, "non-zero at bin {k}: {value:?}");
        }
    }

    #[test]
    fn test_fft_parseval_theorem() {
        // Sum of |x[n]|^2 = (1/N) * Sum of |X[k]|^2
        let signal = [1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -2.0, 1.5];
        let time_energy: f64 = signal.iter().map(|&v| v * v).sum();
        let mut x: Vec<Complex64> = signal.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        fft_radix2(&mut x, false);
        let freq_energy: f64 = x.iter().map(|c| c.norm_sqr()).sum::<f64>() / signal.len() as f64;
        assert!(
            (time_energy - freq_energy).abs() < 1e-8,
            "Parseval: time={time_energy}, freq={freq_energy}"
        );
    }

    #[test]
    fn test_bluestein_matches_naive_dft() {
        for n in [3usize, 5, 7, 12, 100, 4095] {
            let data: Vec<f64> = (0..n).map(|i| ((i * 7919) % 13) as f64 - 6.0).collect();
            let fast = rfft(&data);
            for k in [0, 1, n / 3, n / 2, n - 1] {
                let mut acc = Complex64::new(0.0, 0.0);
                for (i, &v) in data.iter().enumerate() {
                    let angle = -2.0 * PI * (k as f64) * (i as f64) / n as f64;
                    acc += Complex64::new(v * angle.cos(), v * angle.sin());
                }
                assert!(
                    (fast[k] - acc).norm() < 1e-7 * n as f64,
                    "n={n} k={k}: {:?} vs {:?}",
                    fast[k],
                    acc
                );
            }
        }
    }

    #[test]
    fn test_half_spectrum_matches_full_transform() {
        for n in [1usize, 2, 3, 63, 64, 1000, 1001, 4093, 4095, 4096] {
            let data: Vec<f64> = (0..n).map(|i| ((i * 7919) % 13) as f64 - 6.0).collect();
            let expected = rfft(&data);
            let actual = rfft_half(&mut data.clone()).unwrap();
            assert_eq!(actual.len(), n / 2 + 1);
            for (bin, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).norm() < 1e-7 * n as f64,
                    "n={n} bin={bin}: {actual:?} vs {expected:?}",
                );
            }
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut spectrum = rfft(&original);
        let mut recovered = vec![0.0; original.len()];
        irfft(&mut spectrum, &mut recovered);
        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < TOL,
                "mismatch at {i}: orig={orig}, rec={rec}"
            );
        }
    }
}
