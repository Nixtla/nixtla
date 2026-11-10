#include <algorithm>
#include <cmath>
#include <cstdint>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace reconciliation {

// ---------- OpenMP thread control ----------
// WARNING: Do NOT call Eigen matrix operations (e.g. matmul, decomposition)
// inside OpenMP parallel regions. Eigen has its own internal parallelism and
// nesting it within OpenMP threads can cause thread oversubscription, incorrect
// results, or deadlocks. Only use scalar/element-wise operations inside
// #pragma omp parallel blocks.

int get_num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void set_num_threads(int n) {
#ifdef _OPENMP
  omp_set_num_threads(n);
#else
  (void)n;
#endif
}
namespace py = pybind11;

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ---------- _ma_cov ----------
// Masked empirical covariance matrix.
// residuals: (n_ts, n_samples) row-major float64
// not_nan_mask: (n_ts, n_samples) bool row-major
// Returns: (n_ts, n_ts) symmetric float64 matrix (column-major = F-order)
MatrixXd ma_cov(
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &residuals_arr,
    const py::array_t<bool, py::array::c_style | py::array::forcecast>
        &mask_arr) {
  auto res = residuals_arr.unchecked<2>();
  auto mask = mask_arr.unchecked<2>();
  const Eigen::Index n_ts = res.shape(0);
  const Eigen::Index n_samples = res.shape(1);

  MatrixXd W = MatrixXd::Zero(n_ts, n_ts);

#pragma omp parallel for schedule(dynamic)
  for (Eigen::Index i = 0; i < n_ts; ++i) {
    for (Eigen::Index j = 0; j <= i; ++j) {
      // Count valid samples for this pair
      int64_t count = 0;
      for (Eigen::Index k = 0; k < n_samples; ++k) {
        if (mask(i, k) && mask(j, k)) {
          ++count;
        }
      }
      if (count > 1) {
        // Compute masked means
        double sum_i = 0.0, sum_j = 0.0;
        for (Eigen::Index k = 0; k < n_samples; ++k) {
          if (mask(i, k) && mask(j, k)) {
            sum_i += res(i, k);
            sum_j += res(j, k);
          }
        }
        double mean_i = sum_i / count;
        double mean_j = sum_j / count;
        // Empirical covariance
        double cov = 0.0;
        for (Eigen::Index k = 0; k < n_samples; ++k) {
          if (mask(i, k) && mask(j, k)) {
            cov += (res(i, k) - mean_i) * (res(j, k) - mean_j);
          }
        }
        double val = cov / (count - 1);
        W(i, j) = val;
        W(j, i) = val;
      }
    }
  }
  return W;
}

// ---------- _shrunk_covariance_schaferstrimmer_no_nans ----------
// Schafer-Strimmer shrinkage covariance (no NaN handling).
// Optimized: precompute per-row scalars, single-pass (i,j) inner loop.
// Key insight: Xs_mean = 0 for centered data, so standardization factors out.
// residuals: (n_ts, n_samples) row-major float64
// mint_shr_ridge: minimum diagonal value
// Returns: (n_ts, n_ts) symmetric float64 matrix (column-major = F-order)
MatrixXd shrunk_covariance_ss_no_nans(
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &residuals_arr,
    double mint_shr_ridge) {
  auto res = residuals_arr.unchecked<2>();
  const Eigen::Index n_ts = res.shape(0);
  const Eigen::Index T = res.shape(1);

  constexpr double eps = 2e-8;
  const double factor_emp_cov = 1.0 / (T - 1);
  const double factor_shrinkage =
      1.0 / (static_cast<double>(T) * (T - 1));

  // Precompute per-row scalars — O(n_ts * T) time, O(n_ts) storage
  VectorXd means(n_ts), inv_stds_eps(n_ts);
  for (Eigen::Index i = 0; i < n_ts; ++i) {
    double s = 0.0;
    for (Eigen::Index k = 0; k < T; ++k)
      s += res(i, k);
    means(i) = s / T;

    double var = 0.0;
    for (Eigen::Index k = 0; k < T; ++k) {
      double d = res(i, k) - means(i);
      var += d * d;
    }
    inv_stds_eps(i) = 1.0 / (std::sqrt(var / T) + eps);
  }

  MatrixXd W = MatrixXd::Zero(n_ts, n_ts);
  double sum_var_emp_corr = 0.0;
  double sum_sq_emp_corr = 0.0;

#pragma omp parallel for schedule(dynamic)                                     \
    reduction(+ : sum_var_emp_corr, sum_sq_emp_corr)
  for (Eigen::Index i = 0; i < n_ts; ++i) {
    for (Eigen::Index j = 0; j <= i; ++j) {
      // Single pass: accumulate cov and cov^2
      double cov = 0.0, cov_sq = 0.0;
      for (Eigen::Index k = 0; k < T; ++k) {
        double ck = (res(i, k) - means(i)) * (res(j, k) - means(j));
        cov += ck;
        if (i != j)
          cov_sq += ck * ck;
      }
      double val = factor_emp_cov * cov;
      W(i, j) = val;
      if (i != j) {
        W(j, i) = val;
        // Shrinkage terms (Xs_mean = 0, so standardization factors out)
        // var_w = s^2 * (sum(cov_k^2) - T * cov_mean^2)
        double cov_mean = cov / T;
        double s = inv_stds_eps(i) * inv_stds_eps(j);
        double var_w = s * s * (cov_sq - T * cov_mean * cov_mean);
        sum_var_emp_corr += var_w;
        double w_mean = s * cov_mean;
        sum_sq_emp_corr += w_mean * w_mean;
      }
    }
  }

  // Calculate shrinkage intensity
  double shrinkage =
      1.0 - std::clamp((factor_shrinkage * sum_var_emp_corr) /
                            (sum_sq_emp_corr + eps),
                        0.0, 1.0);

  // Apply shrinkage to off-diagonal, ridge to diagonal
  for (Eigen::Index i = 0; i < n_ts; ++i) {
    for (Eigen::Index j = 0; j < i; ++j) {
      double v = shrinkage * W(i, j);
      W(i, j) = v;
      W(j, i) = v;
    }
    W(i, i) = std::max(W(i, i), mint_shr_ridge);
  }
  return W;
}

// ---------- _shrunk_covariance_schaferstrimmer_with_nans ----------
// Schafer-Strimmer shrinkage covariance with NaN masking.
// residuals: (n_ts, n_samples) row-major float64
// not_nan_mask: (n_ts, n_samples) bool row-major
// mint_shr_ridge: minimum diagonal value
// Returns: (n_ts, n_ts) symmetric float64 matrix (column-major = F-order)
MatrixXd shrunk_covariance_ss_with_nans(
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &residuals_arr,
    const py::array_t<bool, py::array::c_style | py::array::forcecast>
        &mask_arr,
    double mint_shr_ridge) {
  auto res = residuals_arr.unchecked<2>();
  auto mask = mask_arr.unchecked<2>();
  const Eigen::Index n_ts = res.shape(0);
  const Eigen::Index n_samples = res.shape(1);

  MatrixXd W = MatrixXd::Zero(n_ts, n_ts);
  constexpr double epsilon = 2e-8;

  double sum_var_emp_corr = 0.0;
  double sum_sq_emp_corr = 0.0;

#pragma omp parallel for schedule(dynamic)                                     \
    reduction(+ : sum_var_emp_corr, sum_sq_emp_corr)
  for (Eigen::Index i = 0; i < n_ts; ++i) {
    for (Eigen::Index j = 0; j <= i; ++j) {
      // Build joint mask and count valid samples
      int64_t count = 0;
      for (Eigen::Index k = 0; k < n_samples; ++k) {
        if (mask(i, k) && mask(j, k)) {
          ++count;
        }
      }
      if (count > 1) {
        // Compute masked means
        double sum_i = 0.0, sum_j = 0.0;
        for (Eigen::Index k = 0; k < n_samples; ++k) {
          if (mask(i, k) && mask(j, k)) {
            sum_i += res(i, k);
            sum_j += res(j, k);
          }
        }
        double mean_i = sum_i / count;
        double mean_j = sum_j / count;

        // Compute masked centered residuals and empirical covariance
        std::vector<double> X_i, X_j;
        X_i.reserve(count);
        X_j.reserve(count);
        double cov = 0.0;
        for (Eigen::Index k = 0; k < n_samples; ++k) {
          if (mask(i, k) && mask(j, k)) {
            double xi = res(i, k) - mean_i;
            double xj = res(j, k) - mean_j;
            X_i.push_back(xi);
            X_j.push_back(xj);
            cov += xi * xj;
          }
        }
        double factor_emp_cov = 1.0 / (count - 1);
        W(i, j) = factor_emp_cov * cov;

        // Off-diagonal sums for shrinkage estimation
        if (i != j) {
          double factor_var_emp_cor =
              static_cast<double>(count) /
              (static_cast<double>((count - 1)) * (count - 1) * (count - 1));

          // Compute population std (ddof=0) for masked residuals_i and
          // residuals_j
          double var_i = 0.0, var_j = 0.0;
          for (int64_t m = 0; m < count; ++m) {
            var_i += X_i[m] * X_i[m];
            var_j += X_j[m] * X_j[m];
          }
          double std_i = std::sqrt(var_i / count) + epsilon;
          double std_j = std::sqrt(var_j / count) + epsilon;

          // Standardize
          std::vector<double> Xs_i(count), Xs_j(count);
          double Xs_i_sum = 0.0, Xs_j_sum = 0.0;
          for (int64_t m = 0; m < count; ++m) {
            Xs_i[m] = X_i[m] / (std_i + epsilon);
            Xs_j[m] = X_j[m] / (std_j + epsilon);
            Xs_i_sum += Xs_i[m];
            Xs_j_sum += Xs_j[m];
          }
          double Xs_i_mean = Xs_i_sum / count;
          double Xs_j_mean = Xs_j_sum / count;

          // w = (Xs_i - Xs_i_mean) * (Xs_j - Xs_j_mean)
          double w_sum = 0.0;
          for (int64_t m = 0; m < count; ++m) {
            w_sum += (Xs_i[m] - Xs_i_mean) * (Xs_j[m] - Xs_j_mean);
          }
          double w_mean = w_sum / count;

          // sum_var_emp_corr += factor_var_emp_cor * sum((w - w_mean)^2)
          double var_w = 0.0;
          for (int64_t m = 0; m < count; ++m) {
            double wk = (Xs_i[m] - Xs_i_mean) * (Xs_j[m] - Xs_j_mean);
            double d = wk - w_mean;
            var_w += d * d;
          }
          sum_var_emp_corr += factor_var_emp_cor * var_w;
          // sum_sq_emp_corr += (factor_emp_cov * n_samples * w_mean)^2
          double sq_term = factor_emp_cov * count * w_mean;
          sum_sq_emp_corr += sq_term * sq_term;
        }
      }
    }
  }

  // Calculate shrinkage intensity (no factor_shrinkage, unlike no_nans)
  double shrinkage = 1.0 - std::clamp(sum_var_emp_corr /
                                           (sum_sq_emp_corr + epsilon),
                                       0.0, 1.0);

  // Shrink the empirical covariance
  for (Eigen::Index i = 0; i < n_ts; ++i) {
    for (Eigen::Index j = 0; j < i; ++j) {
      double val = shrinkage * W(i, j);
      W(i, j) = val;
      W(j, i) = val;
    }
    W(i, i) = std::max(W(i, i), mint_shr_ridge);
  }
  return W;
}

// ---------- _lasso ----------
// Lasso cyclic coordinate descent (sequential, no OpenMP).
// Optimized: column-major X for cache-friendly column access,
// Eigen vectorized dot products and axpy updates.
// X: (n, feats) column-major float64
// y: (n,) float64
// Returns: (feats,) float64 beta coefficients
VectorXd lasso(const Eigen::Ref<const MatrixXd> &X,
               const Eigen::Ref<const VectorXd> &y, double lambda_reg,
               int64_t max_iters, double tol) {
  const Eigen::Index n = X.rows();
  const Eigen::Index feats = X.cols();

  // Column norms: sum(X^2, axis=0)
  VectorXd norms = X.colwise().squaredNorm();
  VectorXd beta = VectorXd::Zero(feats);
  VectorXd beta_changes = VectorXd::Zero(feats);
  VectorXd residuals = y; // copy

  for (int64_t it = 0; it < max_iters; ++it) {
    for (Eigen::Index i = 0; i < feats; ++i) {
      double norms_i = norms(i);
      if (norms_i < 1e-8)
        continue;

      double inv_norms_i = 1.0 / norms_i;

      // Eigen-vectorized dot product (SIMD: SSE/AVX)
      double rho = beta(i) + X.col(i).dot(residuals) * inv_norms_i;

      // Soft threshold
      double threshold = lambda_reg * n * inv_norms_i;
      double sign_rho = (rho > 0.0) ? 1.0 : ((rho < 0.0) ? -1.0 : 0.0);
      double beta_i_next =
          sign_rho * std::max(std::abs(rho) - threshold, 0.0);
      double beta_delta = beta(i) - beta_i_next;
      beta_changes(i) = std::abs(beta_delta);

      if (beta_delta != 0.0) {
        // Eigen-vectorized axpy update (SIMD: SSE/AVX)
        residuals.noalias() += beta_delta * X.col(i);
        beta(i) = beta_i_next;
      }
    }
    if (beta_changes.maxCoeff() < tol)
      break;
  }
  return beta;
}

// ---------- Module init ----------
void init(py::module_ &m) {
  py::module_ recon = m.def_submodule("reconciliation");
  recon.def("get_num_threads", &get_num_threads,
            "Return the maximum number of OpenMP threads.");
  recon.def("set_num_threads", &set_num_threads, py::arg("n"),
            "Set the number of OpenMP threads.");
  recon.def("_ma_cov", &ma_cov, py::call_guard<py::gil_scoped_release>());
  recon.def("_shrunk_covariance_schaferstrimmer_no_nans",
            &shrunk_covariance_ss_no_nans,
            py::call_guard<py::gil_scoped_release>());
  recon.def("_shrunk_covariance_schaferstrimmer_with_nans",
            &shrunk_covariance_ss_with_nans,
            py::call_guard<py::gil_scoped_release>());
  recon.def("_lasso", &lasso, py::arg("X"), py::arg("y"),
            py::arg("lambda_reg"), py::arg("max_iters") = 1000,
            py::arg("tol") = 1e-4, py::call_guard<py::gil_scoped_release>());
}

} // namespace reconciliation
