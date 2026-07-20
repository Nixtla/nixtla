use rand::Rng;
use rand_distr::{Distribution, Exp, Normal, StandardNormal, Uniform};
use rand_mt::Mt64;
use std::f64::consts::PI;

/// RNG wrapper using Mersenne Twister 19937-64.
/// Uses Mersenne Twister 19937-64 (same algorithm as std::mt19937_64).
pub struct SfRng {
    gen: Mt64,
}

impl SfRng {
    pub fn new(seed: u64) -> Self {
        Self {
            gen: Mt64::new(seed),
        }
    }

    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let d = Normal::new(mean, std).expect("invalid normal params: std must be finite and >= 0");
        d.sample(&mut self.gen)
    }

    pub fn normal_array(&mut self, out: &mut [f64], mean: f64, std: f64) {
        let d = Normal::new(mean, std).expect("invalid normal params: std must be finite and >= 0");
        for v in out.iter_mut() {
            *v = d.sample(&mut self.gen);
        }
    }

    pub fn standard_normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.gen)
    }

    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        let d = Uniform::new(a, b).expect("invalid uniform params: requires a < b and finite");
        d.sample(&mut self.gen)
    }

    pub fn uniform_array(&mut self, out: &mut [f64], a: f64, b: f64) {
        let d = Uniform::new(a, b).expect("invalid uniform params: requires a < b and finite");
        for v in out.iter_mut() {
            *v = d.sample(&mut self.gen);
        }
    }

    pub fn uniform01(&mut self) -> f64 {
        self.gen.random::<f64>()
    }

    pub fn poisson(&mut self, lambda: f64) -> i32 {
        if lambda <= 0.0 {
            return 0;
        }
        let d: rand_distr::Poisson<f64> =
            rand_distr::Poisson::new(lambda).expect("invalid poisson param: lambda must be finite");
        d.sample(&mut self.gen) as i32
    }

    pub fn poisson_array(&mut self, out: &mut [i32], lambda: f64) {
        if lambda <= 0.0 {
            for v in out.iter_mut() {
                *v = 0;
            }
            return;
        }
        let d: rand_distr::Poisson<f64> =
            rand_distr::Poisson::new(lambda).expect("invalid poisson param: lambda must be finite");
        for v in out.iter_mut() {
            *v = d.sample(&mut self.gen) as i32;
        }
    }

    pub fn integers(&mut self, low: i32, high: i32) -> i32 {
        let d = Uniform::new(low, high).expect("invalid integer range: requires low < high");
        d.sample(&mut self.gen)
    }

    pub fn exponential(&mut self, lambda: f64) -> f64 {
        let d = Exp::new(lambda).expect("invalid exponential param: lambda must be > 0 and finite");
        d.sample(&mut self.gen)
    }

    pub fn bernoulli(&mut self, p: f64) -> bool {
        let d =
            rand_distr::Bernoulli::new(p).expect("invalid bernoulli param: p must be in [0, 1]");
        d.sample(&mut self.gen)
    }

    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        let d = rand_distr::Gamma::new(shape, scale)
            .expect("invalid gamma params: shape and scale must be > 0 and finite");
        d.sample(&mut self.gen)
    }

    pub fn lognormal(&mut self, mean: f64, std: f64) -> f64 {
        let d = rand_distr::LogNormal::new(mean, std)
            .expect("invalid lognormal params: std must be finite and >= 0");
        d.sample(&mut self.gen)
    }

    pub fn binomial(&mut self, n: i32, p: f64) -> i32 {
        if n <= 0 || p <= 0.0 {
            return 0;
        }
        if p >= 1.0 {
            return n;
        }
        let d = rand_distr::Binomial::new(n as u64, p)
            .expect("invalid binomial params: n >= 0 and p in [0, 1]");
        d.sample(&mut self.gen) as i32
    }

    pub fn negative_binomial(&mut self, r: i32, p: f64) -> i32 {
        // Gamma-Poisson mixture for negative binomial
        let g = rand_distr::Gamma::new(r as f64, (1.0 - p) / p)
            .expect("invalid negative binomial params: r > 0 and p in (0, 1)");
        let lambda: f64 = g.sample(&mut self.gen);
        if lambda <= 0.0 {
            return 0;
        }
        let pois: rand_distr::Poisson<f64> = rand_distr::Poisson::new(lambda)
            .expect("negative binomial: gamma sample produced invalid lambda");
        pois.sample(&mut self.gen) as i32
    }

    /// Geometric distribution (numpy convention): number of trials until
    /// first success, with min value 1.
    pub fn geometric(&mut self, p: f64) -> i32 {
        let d =
            rand_distr::Geometric::new(p).expect("invalid geometric param: p must be in (0, 1]");
        d.sample(&mut self.gen) as i32 + 1
    }

    /// Shuffle indices [0, n)
    pub fn permutation(&mut self, n: i32) -> Vec<i32> {
        let mut indices: Vec<i32> = (0..n).collect();
        // Fisher-Yates shuffle
        for i in (1..n as usize).rev() {
            let j = Uniform::new(0usize, i + 1)
                .expect("permutation: invalid shuffle range")
                .sample(&mut self.gen);
            indices.swap(i, j);
        }
        indices
    }

    /// Choose k unique indices from [0, n) without replacement.
    ///
    /// Uses Floyd's algorithm, which draws exactly `k` random numbers and
    /// runs in O(k) time and space, rather than shuffling all `n` elements
    /// and truncating. The result is a uniformly random size-`k` subset; its
    /// order is not a uniform permutation of that subset, which is fine for
    /// callers that only need the set of indices (missingness/anomalies) or
    /// assign per-index attributes afterwards.
    pub fn choice(&mut self, n: i32, k: i32) -> Vec<i32> {
        let k = k.min(n).max(0) as usize;
        let n = n.max(0) as usize;
        let mut selected = std::collections::HashSet::with_capacity(k);
        let mut result = Vec::with_capacity(k);
        for j in (n - k)..n {
            // Inclusive draw from [0, j]; new_inclusive keeps the rand 0.9
            // idiom used elsewhere in this file.
            let t = Uniform::new_inclusive(0usize, j)
                .expect("choice: invalid inclusive range")
                .sample(&mut self.gen);
            let picked = if selected.contains(&t) { j } else { t };
            selected.insert(picked);
            result.push(picked as i32);
        }
        result
    }

    /// Innovation distribution sampling.
    /// dist encoding: 0=normal, 1=student_t, 2=laplace, 3=uniform, 4=skew_normal
    pub fn sample_innovation(&mut self, scale: f64, dist: i32, param: f64) -> f64 {
        match dist {
            1 => {
                // Student's t
                let df = param;
                assert!(df > 2.0, "Student's t df must be > 2 for finite variance");
                let t_scale = scale * ((df - 2.0) / df).sqrt();
                let d = rand_distr::StudentT::new(df)
                    .expect("invalid Student's t param: df must be > 0 and finite");
                d.sample(&mut self.gen) * t_scale
            }
            2 => {
                // Laplace
                let b = scale / 2.0_f64.sqrt();
                let d = Exp::new(1.0 / b).expect("invalid laplace param: scale must be > 0");
                d.sample(&mut self.gen) - d.sample(&mut self.gen)
            }
            3 => {
                // Uniform
                let a = scale * 3.0_f64.sqrt();
                let d = Uniform::new(-a, a)
                    .expect("invalid uniform innovation param: scale must be > 0");
                d.sample(&mut self.gen)
            }
            4 => {
                // Skew-normal
                let alpha = param;
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                let u0 = self.normal(0.0, 1.0).abs();
                let v = self.normal(0.0, 1.0);
                let raw = delta * u0 + (1.0 - delta * delta).sqrt() * v;
                let mn = delta * (2.0 / PI).sqrt();
                let sd = (1.0 - 2.0 * delta * delta / PI).sqrt();
                if sd > 0.0 {
                    (raw - mn) * (scale / sd)
                } else {
                    raw * scale
                }
            }
            _ => {
                // 0 = normal (default)
                self.normal(0.0, scale)
            }
        }
    }

    pub fn sample_innovations(&mut self, out: &mut [f64], scale: f64, dist: i32, param: f64) {
        match dist {
            1 => {
                let df = param;
                assert!(df > 2.0, "Student's t df must be > 2 for finite variance");
                let t_scale = scale * ((df - 2.0) / df).sqrt();
                let d = rand_distr::StudentT::new(df)
                    .expect("invalid Student's t param: df must be > 0 and finite");
                for v in out.iter_mut() {
                    *v = d.sample(&mut self.gen) * t_scale;
                }
            }
            2 => {
                let b = scale / 2.0_f64.sqrt();
                let d = Exp::new(1.0 / b).expect("invalid laplace param: scale must be > 0");
                for v in out.iter_mut() {
                    *v = d.sample(&mut self.gen) - d.sample(&mut self.gen);
                }
            }
            3 => {
                let a = scale * 3.0_f64.sqrt();
                let d = Uniform::new(-a, a)
                    .expect("invalid uniform innovation param: scale must be > 0");
                for v in out.iter_mut() {
                    *v = d.sample(&mut self.gen);
                }
            }
            4 => {
                for v in out.iter_mut() {
                    *v = self.sample_innovation(scale, dist, param);
                }
            }
            _ => {
                let d = Normal::new(0.0, scale)
                    .expect("invalid normal innovation param: scale must be finite and >= 0");
                for v in out.iter_mut() {
                    *v = d.sample(&mut self.gen);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeded_reproducibility() {
        let mut rng1 = SfRng::new(42);
        let mut rng2 = SfRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.uniform01(), rng2.uniform01());
        }
    }

    #[test]
    fn test_normal_finite() {
        let mut rng = SfRng::new(123);
        for _ in 0..1000 {
            let v = rng.normal(0.0, 1.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_normal_array_fills_correctly() {
        let mut rng = SfRng::new(99);
        let mut buf = vec![0.0; 50];
        rng.normal_array(&mut buf, 5.0, 0.1);
        // All values should be close to 5.0
        for v in &buf {
            assert!(v.is_finite());
            assert!((v - 5.0).abs() < 5.0, "value {v} too far from mean 5.0");
        }
    }

    #[test]
    fn test_uniform_bounds() {
        let mut rng = SfRng::new(7);
        for _ in 0..1000 {
            let v = rng.uniform(2.0, 5.0);
            assert!((2.0..5.0).contains(&v), "uniform out of bounds: {v}");
        }
    }

    #[test]
    fn test_uniform_array_bounds() {
        let mut rng = SfRng::new(7);
        let mut buf = vec![0.0; 100];
        rng.uniform_array(&mut buf, -1.0, 1.0);
        for v in &buf {
            assert!(*v >= -1.0 && *v < 1.0);
        }
    }

    #[test]
    fn test_uniform01_bounds() {
        let mut rng = SfRng::new(11);
        for _ in 0..1000 {
            let v = rng.uniform01();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_poisson_nonnegative() {
        let mut rng = SfRng::new(55);
        for _ in 0..500 {
            let v = rng.poisson(3.0);
            assert!(v >= 0);
        }
    }

    #[test]
    fn test_poisson_zero_lambda() {
        let mut rng = SfRng::new(1);
        assert_eq!(rng.poisson(0.0), 0);
        assert_eq!(rng.poisson(-1.0), 0);
    }

    #[test]
    fn test_poisson_array() {
        let mut rng = SfRng::new(10);
        let mut buf = vec![0_i32; 50];
        rng.poisson_array(&mut buf, 5.0);
        for v in &buf {
            assert!(*v >= 0);
        }
    }

    #[test]
    fn test_poisson_array_zero_lambda() {
        let mut rng = SfRng::new(10);
        let mut buf = vec![99_i32; 10];
        rng.poisson_array(&mut buf, 0.0);
        for v in &buf {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn test_integers_bounds() {
        let mut rng = SfRng::new(3);
        for _ in 0..1000 {
            let v = rng.integers(0, 10);
            assert!((0..10).contains(&v));
        }
    }

    #[test]
    fn test_bernoulli_values() {
        let mut rng = SfRng::new(77);
        let mut trues = 0;
        let n = 1000;
        for _ in 0..n {
            if rng.bernoulli(0.3) {
                trues += 1;
            }
        }
        // Expect ~300 trues, allow wide margin
        assert!(
            trues > 200 && trues < 400,
            "bernoulli(0.3) gave {trues}/1000 trues"
        );
    }

    #[test]
    fn test_exponential_positive() {
        let mut rng = SfRng::new(8);
        for _ in 0..500 {
            let v = rng.exponential(1.0);
            assert!(v > 0.0 && v.is_finite());
        }
    }

    #[test]
    fn test_gamma_positive() {
        let mut rng = SfRng::new(22);
        for _ in 0..500 {
            let v = rng.gamma(2.0, 1.0);
            assert!(v > 0.0 && v.is_finite());
        }
    }

    #[test]
    fn test_binomial_bounds() {
        let mut rng = SfRng::new(33);
        for _ in 0..500 {
            let v = rng.binomial(10, 0.5);
            assert!((0..=10).contains(&v));
        }
    }

    #[test]
    fn test_binomial_edge_cases() {
        let mut rng = SfRng::new(1);
        assert_eq!(rng.binomial(0, 0.5), 0);
        assert_eq!(rng.binomial(10, 0.0), 0);
        assert_eq!(rng.binomial(10, 1.0), 10);
        assert_eq!(rng.binomial(-1, 0.5), 0);
    }

    #[test]
    fn test_negative_binomial_nonnegative() {
        let mut rng = SfRng::new(44);
        for _ in 0..500 {
            let v = rng.negative_binomial(5, 0.5);
            assert!(v >= 0);
        }
    }

    #[test]
    fn test_geometric_positive() {
        let mut rng = SfRng::new(66);
        for _ in 0..500 {
            let v = rng.geometric(0.3);
            assert!(v >= 1, "geometric should be >= 1, got {v}");
        }
    }

    #[test]
    fn test_permutation_is_permutation() {
        let mut rng = SfRng::new(88);
        let perm = rng.permutation(10);
        assert_eq!(perm.len(), 10);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<i32>>());
    }

    #[test]
    fn test_choice_correct_size() {
        let mut rng = SfRng::new(99);
        let c = rng.choice(20, 5);
        assert_eq!(c.len(), 5);
        for v in &c {
            assert!(*v >= 0 && *v < 20);
        }
    }

    #[test]
    fn test_choice_unique() {
        let mut rng = SfRng::new(1234);
        for _ in 0..50 {
            let c = rng.choice(50, 12);
            assert_eq!(c.len(), 12);
            let mut sorted = c.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 12, "choice returned duplicate indices: {c:?}");
            for v in &c {
                assert!(*v >= 0 && *v < 50);
            }
        }
    }

    #[test]
    fn test_choice_edge_cases() {
        let mut rng = SfRng::new(7);
        // k == 0 yields empty
        assert!(rng.choice(10, 0).is_empty());
        // k >= n selects all n distinct indices
        let mut all = rng.choice(6, 6);
        all.sort_unstable();
        assert_eq!(all, (0..6).collect::<Vec<i32>>());
        let mut clamped = rng.choice(4, 10);
        clamped.sort_unstable();
        assert_eq!(clamped, (0..4).collect::<Vec<i32>>());
        // n == 0 yields empty
        assert!(rng.choice(0, 3).is_empty());
    }

    #[test]
    fn test_sample_innovation_normal() {
        let mut rng = SfRng::new(1);
        for _ in 0..100 {
            let v = rng.sample_innovation(1.0, 0, 0.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_innovation_student_t() {
        let mut rng = SfRng::new(2);
        for _ in 0..100 {
            let v = rng.sample_innovation(1.0, 1, 5.0); // df=5
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_innovation_laplace() {
        let mut rng = SfRng::new(3);
        for _ in 0..100 {
            let v = rng.sample_innovation(1.0, 2, 0.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_innovation_uniform() {
        let mut rng = SfRng::new(4);
        let a = 3.0_f64.sqrt();
        for _ in 0..100 {
            let v = rng.sample_innovation(1.0, 3, 0.0);
            assert!(v >= -a && v <= a);
        }
    }

    #[test]
    fn test_sample_innovation_skew_normal() {
        let mut rng = SfRng::new(5);
        for _ in 0..100 {
            let v = rng.sample_innovation(1.0, 4, 3.0); // alpha=3
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_innovations_fills_array() {
        let mut rng = SfRng::new(6);
        let mut buf = vec![0.0; 50];
        rng.sample_innovations(&mut buf, 1.0, 0, 0.0);
        // Should not all be zero
        assert!(buf.iter().any(|&v| v != 0.0));
        for v in &buf {
            assert!(v.is_finite());
        }
    }
}
