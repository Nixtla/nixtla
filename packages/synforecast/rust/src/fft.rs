use num_complex::Complex64;
use std::f64::consts::PI;

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

/// Real-to-complex FFT
pub fn rfft(data: &[f64]) -> Vec<Complex64> {
    let n = data.len();
    let mut x: Vec<Complex64> = data.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    fft_radix2(&mut x, false);
    x.truncate(n); // keep full spectrum for irfft compatibility
    x
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
