use std::f64::consts::PI;

const EPS: f64 = 1e-14;
const MAX_ITER: i32 = 200;

// ---------------------------------------------------------------------------
// 1. erf_approx  --  Abramowitz & Stegun formula 7.1.26
// ---------------------------------------------------------------------------
#[inline]
pub fn erf_approx(x: f64) -> f64 {
    const P: f64 = 0.3275911;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

// ---------------------------------------------------------------------------
// 2. norm_cdf
// ---------------------------------------------------------------------------
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    0.5 * (1.0 + erf_approx(x * inv_sqrt2))
}

// ---------------------------------------------------------------------------
// 3. norm_ppf  --  Normal inverse CDF (Wichura 1988 / AS 241)
// ---------------------------------------------------------------------------
#[inline]
pub fn norm_ppf(p: f64) -> f64 {
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    if (P_LOW..=P_HIGH).contains(&p) {
        let q = p - 0.5;
        let r = q * q;
        let num = ((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5];
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num * q / den
    } else if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -(num / den)
    }
}

// ---------------------------------------------------------------------------
// 4. lgamma_approx  --  Lanczos approximation (g = 7)
// ---------------------------------------------------------------------------
pub fn lgamma_approx(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.9999999999998099,
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    let log_sqrt_2pi = 0.5 * (2.0 * PI).ln();

    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - lgamma_approx(1.0 - x);
    }

    let y = x - 1.0;
    let mut ser = COEFFS[0];
    for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
        ser += coeff / (y + i as f64);
    }
    let t = y + 7.0 + 0.5;
    log_sqrt_2pi + (y + 0.5) * t.ln() - t + ser.ln()
}

// ---------------------------------------------------------------------------
// 5. betainc  --  Regularized incomplete beta (continued fraction / Lentz)
// ---------------------------------------------------------------------------
pub fn betainc(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - betainc(1.0 - x, b, a);
    }

    let lbeta = lgamma_approx(a) + lgamma_approx(b) - lgamma_approx(a + b);
    let prefix = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;

    // Modified Lentz's algorithm (Numerical Recipes style)
    // Initialization handles the first coefficient d_1 = -(a+b)*x/(a+1)
    let mut c = 1.0_f64;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < EPS {
        d = EPS;
    }
    d = 1.0 / d;
    let mut h = d;

    // Loop with even+odd pairs: m=1,2,... computes d_{2m} then d_{2m+1}
    for m in 1..=(MAX_ITER / 2) {
        let mf = m as f64;
        let m2 = 2.0 * mf;

        // Even coefficient: d_{2m} = m(b-m)x / ((a+2m-1)(a+2m))
        let aa_even = mf * (b - mf) * x / ((a + m2 - 1.0) * (a + m2));
        d = 1.0 + aa_even * d;
        if d.abs() < EPS {
            d = EPS;
        }
        c = 1.0 + aa_even / c;
        if c.abs() < EPS {
            c = EPS;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd coefficient: d_{2m+1} = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
        let aa_odd = -((a + mf) * (a + b + mf) * x) / ((a + m2) * (a + m2 + 1.0));
        d = 1.0 + aa_odd * d;
        if d.abs() < EPS {
            d = EPS;
        }
        c = 1.0 + aa_odd / c;
        if c.abs() < EPS {
            c = EPS;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }

    prefix * h
}

// ---------------------------------------------------------------------------
// 6. t_cdf  --  Student's t CDF via regularized incomplete beta
// ---------------------------------------------------------------------------
pub fn t_cdf(x: f64, df: f64) -> f64 {
    let t_val = df / (df + x * x);
    let mut f = 1.0 - 0.5 * betainc(t_val, df / 2.0, 0.5);
    if x < 0.0 {
        f = 1.0 - f;
    }
    f
}

// ---------------------------------------------------------------------------
// 7. expon_ppf
// ---------------------------------------------------------------------------
pub fn expon_ppf(u: f64, scale: f64) -> f64 {
    -scale * (1.0 - u).ln()
}

// ---------------------------------------------------------------------------
// 8. lognorm_ppf
// ---------------------------------------------------------------------------
pub fn lognorm_ppf(u: f64, s: f64, scale: f64) -> f64 {
    scale * (s * norm_ppf(u)).exp()
}

// ---------------------------------------------------------------------------
// 9. uniform_ppf
// ---------------------------------------------------------------------------
pub fn uniform_ppf(u: f64, loc: f64, scale: f64) -> f64 {
    loc + scale * u
}

// ---------------------------------------------------------------------------
// 10. gammainc_series  --  Lower incomplete gamma (series)
// ---------------------------------------------------------------------------
pub fn gammainc_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;

    for n in 1..=MAX_ITER {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < EPS * sum.abs() {
            break;
        }
    }

    sum * (-x + a * x.ln() - lgamma_approx(a)).exp()
}

// ---------------------------------------------------------------------------
// 11. gammainc_cf  --  Upper incomplete gamma (continued fraction / Lentz)
// ---------------------------------------------------------------------------
pub fn gammainc_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / EPS;
    let mut d = 1.0 / b;
    let mut h = d;

    for n in 1..=MAX_ITER {
        let an = -(n as f64) * (n as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < EPS {
            d = EPS;
        }
        c = b + an / c;
        if c.abs() < EPS {
            c = EPS;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }

    h * (-x + a * x.ln() - lgamma_approx(a)).exp()
}

// ---------------------------------------------------------------------------
// 12. gammainc_lower  --  Regularized lower incomplete gamma P(a,x)
// ---------------------------------------------------------------------------
pub fn gammainc_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gammainc_series(a, x)
    } else {
        1.0 - gammainc_cf(a, x)
    }
}

// ---------------------------------------------------------------------------
// 13. gamma_ppf  --  Gamma PPF via Newton-Raphson (Wilson-Hilferty init)
// ---------------------------------------------------------------------------
pub fn gamma_ppf(u: f64, a: f64, scale: f64) -> f64 {
    const NR_MAX: i32 = 50;
    const TOL: f64 = 1e-12;

    let inv9a = 1.0 / (9.0 * a);
    let wh = 1.0 - inv9a + norm_ppf(u) * inv9a.sqrt();
    let mut x = a * wh * wh * wh * scale;
    if x <= 0.0 {
        x = TOL;
    }

    let lga = lgamma_approx(a);

    for _ in 0..NR_MAX {
        let p_val = gammainc_lower(a, x / scale);
        let err = p_val - u;
        let log_pdf = (a - 1.0) * x.ln() - x / scale - a * scale.ln() - lga;
        let pdf = log_pdf.exp();
        if pdf < 1e-300 {
            break;
        }
        let dx = err / pdf;
        x -= dx;
        if x <= 0.0 {
            x = TOL;
        }
        if dx.abs() < TOL * x.abs() {
            break;
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_erf_approx_known_values() {
        assert!((erf_approx(0.0)).abs() < TOL);
        assert!((erf_approx(1.0) - 0.842700793).abs() < TOL);
        assert!((erf_approx(-1.0) + 0.842700793).abs() < TOL);
        assert!((erf_approx(2.0) - 0.995322265).abs() < TOL);
        // Symmetry
        assert!((erf_approx(0.5) + erf_approx(-0.5)).abs() < TOL);
    }

    #[test]
    fn test_norm_cdf_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < TOL);
        assert!((norm_cdf(1.0) - 0.841344746).abs() < TOL);
        assert!((norm_cdf(-1.0) - 0.158655254).abs() < TOL);
        assert!((norm_cdf(2.0) - 0.977249868).abs() < TOL);
        assert!((norm_cdf(-2.0) - 0.022750132).abs() < TOL);
    }

    #[test]
    fn test_norm_ppf_known_values() {
        // norm_ppf should be the inverse of norm_cdf
        assert!(norm_ppf(0.5).abs() < TOL);
        assert!((norm_ppf(0.841344746) - 1.0).abs() < 1e-4);
        assert!((norm_ppf(0.158655254) + 1.0).abs() < 1e-4);
        assert!((norm_ppf(0.977249868) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_norm_ppf_cdf_roundtrip() {
        // ppf(cdf(x)) ≈ x
        for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let roundtrip = norm_ppf(norm_cdf(x));
            assert!(
                (roundtrip - x).abs() < 1e-4,
                "roundtrip failed for x={x}: got {roundtrip}"
            );
        }
    }

    #[test]
    fn test_lgamma_approx_known_values() {
        // lgamma(1) = 0, lgamma(2) = 0, lgamma(0.5) = ln(sqrt(pi))
        assert!(lgamma_approx(1.0).abs() < TOL);
        assert!(lgamma_approx(2.0).abs() < TOL);
        let expected_half = 0.5 * PI.ln(); // ln(sqrt(pi)) = 0.5 * ln(pi)
        assert!((lgamma_approx(0.5) - expected_half).abs() < TOL);
        // lgamma(5) = ln(24) ≈ 3.178054
        assert!((lgamma_approx(5.0) - 24.0_f64.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_betainc_boundary_values() {
        assert!((betainc(0.0, 2.0, 3.0)).abs() < TOL);
        assert!((betainc(1.0, 2.0, 3.0) - 1.0).abs() < TOL);
        // betainc(0.5, 1.0, 1.0) = 0.5 (uniform case)
        assert!((betainc(0.5, 1.0, 1.0) - 0.5).abs() < TOL);
    }

    #[test]
    fn test_t_cdf_symmetry() {
        // t distribution is symmetric around 0
        let df = 10.0;
        assert!((t_cdf(0.0, df) - 0.5).abs() < TOL);
        assert!((t_cdf(1.0, df) + t_cdf(-1.0, df) - 1.0).abs() < TOL);
        assert!((t_cdf(2.0, df) + t_cdf(-2.0, df) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_expon_ppf() {
        assert!(expon_ppf(0.0, 1.0).abs() < TOL);
        assert!((expon_ppf(0.5, 1.0) - std::f64::consts::LN_2).abs() < 1e-4);
        assert!((expon_ppf(0.5, 2.0) - 1.3862944).abs() < 1e-4); // 2*ln(2)
    }

    #[test]
    fn test_lognorm_ppf() {
        // lognorm_ppf(0.5, s, scale) = scale (median of lognormal)
        assert!((lognorm_ppf(0.5, 1.0, 1.0) - 1.0).abs() < 1e-4);
        assert!((lognorm_ppf(0.5, 1.0, 2.0) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_uniform_ppf() {
        assert!((uniform_ppf(0.0, 0.0, 1.0)).abs() < TOL);
        assert!((uniform_ppf(0.5, 0.0, 1.0) - 0.5).abs() < TOL);
        assert!((uniform_ppf(1.0, 0.0, 1.0) - 1.0).abs() < TOL);
        assert!((uniform_ppf(0.5, 2.0, 4.0) - 4.0).abs() < TOL); // 2 + 4*0.5 = 4
    }

    #[test]
    fn test_gammainc_lower_known_values() {
        // P(1, 1) = 1 - e^{-1} ≈ 0.632121
        assert!((gammainc_lower(1.0, 1.0) - 0.632121).abs() < 1e-4);
        assert!(gammainc_lower(1.0, 0.0).abs() < TOL);
    }

    #[test]
    fn test_gamma_ppf_roundtrip() {
        // gamma_ppf(gammainc_lower(a, x/scale), a, scale) ≈ x
        let a = 2.0;
        let scale = 1.0;
        let x = 3.0;
        let p = gammainc_lower(a, x / scale);
        let roundtrip = gamma_ppf(p, a, scale);
        assert!(
            (roundtrip - x).abs() < 1e-3,
            "gamma roundtrip: got {roundtrip}, expected {x}"
        );
    }
}
