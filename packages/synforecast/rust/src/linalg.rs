use nalgebra::{DMatrix, DVector};

/// Cholesky decomposition with eigendecomposition fallback.
/// Returns the lower triangular factor L such that mat ≈ L * L^T.
pub fn cholesky_with_fallback(mat: &DMatrix<f64>) -> DMatrix<f64> {
    match mat.clone().cholesky() {
        Some(chol) => chol.l(),
        None => {
            // Fallback: symmetric eigendecomposition with clamped eigenvalues
            let eig = mat.clone().symmetric_eigen();
            let n = eig.eigenvalues.len();
            let mut sqrt_diag = DVector::zeros(n);
            for i in 0..n {
                sqrt_diag[i] = eig.eigenvalues[i].max(1e-10).sqrt();
            }
            &eig.eigenvectors * DMatrix::from_diagonal(&sqrt_diag)
        }
    }
}

/// Build a DMatrix from a flat row-major slice
pub fn flat_to_matrix(data: &[f64], rows: usize, cols: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, cols, data)
}

/// Matrix-vector multiply
pub fn mat_vec_mul(mat: &DMatrix<f64>, vec: &DVector<f64>) -> DVector<f64> {
    mat * vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_to_matrix_identity() {
        let data = [1.0, 0.0, 0.0, 1.0];
        let m = flat_to_matrix(&data, 2, 2);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 0.0);
        assert_eq!(m[(1, 0)], 0.0);
        assert_eq!(m[(1, 1)], 1.0);
    }

    #[test]
    fn test_flat_to_matrix_row_major() {
        // Row-major: [[1, 2], [3, 4]]
        let data = [1.0, 2.0, 3.0, 4.0];
        let m = flat_to_matrix(&data, 2, 2);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_mat_vec_mul_identity() {
        let data = [1.0, 0.0, 0.0, 1.0];
        let m = flat_to_matrix(&data, 2, 2);
        let v = DVector::from_vec(vec![3.0, 7.0]);
        let result = mat_vec_mul(&m, &v);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_vec_mul_general() {
        // [[1, 2], [3, 4]] * [5, 6] = [17, 39]
        let data = [1.0, 2.0, 3.0, 4.0];
        let m = flat_to_matrix(&data, 2, 2);
        let v = DVector::from_vec(vec![5.0, 6.0]);
        let result = mat_vec_mul(&m, &v);
        assert!((result[0] - 17.0).abs() < 1e-10);
        assert!((result[1] - 39.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_positive_definite() {
        // Known PD matrix: [[4, 2], [2, 3]]
        let data = [4.0, 2.0, 2.0, 3.0];
        let m = flat_to_matrix(&data, 2, 2);
        let l = cholesky_with_fallback(&m);
        // Verify L * L^T ≈ original
        let reconstructed = &l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[(i, j)] - m[(i, j)]).abs() < 1e-10,
                    "mismatch at ({i},{j}): {} vs {}",
                    reconstructed[(i, j)],
                    m[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cholesky_identity() {
        let data = [1.0, 0.0, 0.0, 1.0];
        let m = flat_to_matrix(&data, 2, 2);
        let l = cholesky_with_fallback(&m);
        // L should be identity for identity matrix
        assert!((l[(0, 0)] - 1.0).abs() < 1e-10);
        assert!(l[(0, 1)].abs() < 1e-10);
        assert!(l[(1, 0)].abs() < 1e-10);
        assert!((l[(1, 1)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_fallback_non_pd() {
        // Non-PD matrix (has negative eigenvalue): [[1, 2], [2, 1]]
        let data = [1.0, 2.0, 2.0, 1.0];
        let m = flat_to_matrix(&data, 2, 2);
        let l = cholesky_with_fallback(&m);
        // Should not panic; L should produce a valid (approximate) factorization
        let reconstructed = &l * l.transpose();
        // The reconstructed matrix should be PSD (eigenvalues >= 0)
        let eig = reconstructed.symmetric_eigen();
        for ev in eig.eigenvalues.iter() {
            assert!(*ev >= -1e-8, "negative eigenvalue in reconstructed: {ev}");
        }
    }

    #[test]
    fn test_cholesky_3x3() {
        // 3x3 PD matrix: correlation matrix with off-diag 0.5
        let data = [1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0];
        let m = flat_to_matrix(&data, 3, 3);
        let l = cholesky_with_fallback(&m);
        let reconstructed = &l * l.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reconstructed[(i, j)] - m[(i, j)]).abs() < 1e-10,
                    "3x3 mismatch at ({i},{j})"
                );
            }
        }
    }
}
