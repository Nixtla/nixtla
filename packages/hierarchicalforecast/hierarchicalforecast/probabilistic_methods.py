__all__ = ['Normality', 'CovarianceType']


import warnings
from enum import Enum

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder

from .utils import (
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
    is_strictly_hierarchical,
)


class CovarianceType(str, Enum):
    """Covariance estimation method for Normality probabilistic reconciliation.

    Attributes:
        DIAGONAL: Uses the W matrix diagonal with correlation scaling.
            This is the default and backward-compatible option.
            The W matrix is required.
        FULL: Uses full empirical covariance computed from residuals.
            Requires the `residuals` parameter. The W matrix is ignored.
            Warning: May produce non-positive-definite matrices when
            n_series > n_observations.
        SHRINK: Uses Schäfer-Strimmer shrinkage estimator for better numerical
            stability. Requires the `residuals` parameter. The W matrix is ignored.
            Recommended when n_series is large relative to n_observations.
    """

    DIAGONAL = "diagonal"
    FULL = "full"
    SHRINK = "shrink"


class Normality:
    r"""Normality Probabilistic Reconciliation Class.

    The Normality method leverages the Gaussian Distribution linearity, to
    generate hierarchically coherent prediction distributions. This class is
    meant to be used as the `sampler` input as other `HierarchicalForecast` [reconciliation classes](./methods).

    Given base forecasts under a normal distribution:
    ```math
    \hat{y}_{h} \sim \mathrm{N}(\hat{\boldsymbol{\mu}}, \hat{\mathbf{W}}_{h})
    ```

    The reconciled forecasts are also normally distributed:

    ```math
    \tilde{y}_{h} \sim \mathrm{N}(\mathbf{S}\mathbf{P}\hat{\boldsymbol{\mu}},
    \mathbf{S}\mathbf{P}\hat{\mathbf{W}}_{h} \mathbf{P}^{\intercal} \mathbf{S}^{\intercal})
    ```

    Args:
        S (Union[np.ndarray, sp.spmatrix]): Summing matrix of size (`base`, `bottom`).
        P (Union[np.ndarray, sp.spmatrix]): Reconciliation matrix of size (`bottom`, `base`).
        y_hat (np.ndarray): Point forecasts values of size (`base`, `horizon`).
        sigmah (np.ndarray): Forecast standard dev. of size (`base`, `horizon`).
        W (Union[np.ndarray, sp.spmatrix], optional): Hierarchical covariance matrix of size
            (`base`, `base`). Required when `covariance_type='diagonal'` (default).
            **Ignored** when `covariance_type` is `'full'` or `'shrink'` (covariance is
            computed from residuals instead). Default is None.
        seed (int, optional): Random seed for numpy generator's replicability. Default is 0.
        covariance_type (Union[str, CovarianceType], optional): Type of covariance estimator.
            Can be a string or CovarianceType enum. Options are:

            - `'diagonal'` / `CovarianceType.DIAGONAL`: Uses the W matrix diagonal with
              correlation scaling (default, backward compatible). W is required.
            - `'full'` / `CovarianceType.FULL`: Uses full empirical covariance from residuals.
              W is ignored. Warning: may be non-positive-definite if n_series > n_observations.
            - `'shrink'` / `CovarianceType.SHRINK`: Uses Schäfer-Strimmer shrinkage estimator.
              W is ignored. Recommended for numerical stability with many series.

            Default is `'diagonal'`.
        residuals (np.ndarray, optional): Insample residuals of size (`base`, `obs`).
            Required when `covariance_type` is `'full'` or `'shrink'`. Default is None.
        shrinkage_ridge (float, optional): Ridge parameter for shrinkage covariance estimator.
            Only used when `covariance_type='shrink'`. A warning is issued if provided
            with other covariance types. Default is 2e-8.

    Raises:
        ValueError: If `covariance_type` is invalid.
        ValueError: If `covariance_type='diagonal'` and `W` is None.
        ValueError: If `covariance_type` is `'full'` or `'shrink'` and `residuals` is None.
        ValueError: If `residuals` shape doesn't match expected (`base`, `obs`).
        ValueError: If `residuals` has fewer than 2 observations.
        ValueError: If `residuals` is empty.
        ValueError: If any series in `residuals` has all NaN values.

    Warnings:
        UserWarning: If `shrinkage_ridge` is provided but `covariance_type` is not `'shrink'`.
        UserWarning: If `W` is provided but `covariance_type` is not `'diagonal'` (W is ignored).
        UserWarning: If any series has zero or near-zero variance (may affect correlation estimates).
        UserWarning: If `covariance_type='full'` and n_series > n_observations (non-PSD risk).

    References:
        - [Panagiotelis A., Gamakumara P. Athanasopoulos G., and Hyndman R. J. (2022).
          "Probabilistic forecast reconciliation: Properties, evaluation and score optimisation".
          European Journal of Operational Research.](https://www.sciencedirect.com/science/article/pii/S0377221722006087)
        - [Schäfer, Juliane, and Korbinian Strimmer. "A Shrinkage Approach to Large-Scale
          Covariance Matrix Estimation". Statistical Applications in Genetics and Molecular
          Biology 4, no. 1 (2005).](https://doi.org/10.2202/1544-6115.1175)

    Examples:
        >>> # Using diagonal covariance (default, backward compatible)
        >>> normality = Normality(S=S, P=P, y_hat=y_hat, sigmah=sigmah, W=W)
        >>> samples = normality.get_samples(num_samples=100)

        >>> # Using full empirical covariance from residuals
        >>> normality = Normality(
        ...     S=S, P=P, y_hat=y_hat, sigmah=sigmah,
        ...     covariance_type="full", residuals=residuals
        ... )

        >>> # Using shrinkage covariance (recommended for stability)
        >>> normality = Normality(
        ...     S=S, P=P, y_hat=y_hat, sigmah=sigmah,
        ...     covariance_type=CovarianceType.SHRINK, residuals=residuals
        ... )
    """

    # Numerical stability constants
    # Minimum standard deviation threshold to avoid division by zero when
    # converting covariance to correlation. Values below this are clamped.
    # Chosen as sqrt(machine epsilon) for float64 ≈ 1.5e-8, rounded to 1e-8.
    _MIN_STD_THRESHOLD: float = 1e-8

    # Default ridge parameter for shrinkage covariance estimator.
    # Provides numerical stability by ensuring diagonal elements are bounded away from zero.
    # Based on Schäfer-Strimmer (2005) recommendations for genomic data.
    _DEFAULT_SHRINKAGE_RIDGE: float = 2e-8

    def __init__(
        self,
        S: np.ndarray | sp.spmatrix,
        P: np.ndarray | sp.spmatrix,
        y_hat: np.ndarray,
        sigmah: np.ndarray,
        W: np.ndarray | sp.spmatrix | None = None,
        seed: int = 0,
        covariance_type: str | CovarianceType = "diagonal",
        residuals: np.ndarray | None = None,
        shrinkage_ridge: float = _DEFAULT_SHRINKAGE_RIDGE,
    ):
        # Normalize covariance_type to CovarianceType enum
        if isinstance(covariance_type, str):
            covariance_type_lower = covariance_type.lower()
            try:
                covariance_type = CovarianceType(covariance_type_lower)
            except ValueError as e:
                valid_types = [t.value for t in CovarianceType]
                raise ValueError(
                    f"Unknown covariance_type `{covariance_type}`. "
                    f"Choose from {valid_types} or use CovarianceType enum."
                ) from e
        elif not isinstance(covariance_type, CovarianceType):
            raise ValueError(
                f"covariance_type must be a string or CovarianceType enum, "
                f"got {type(covariance_type).__name__}."
            )

        # Get number of series from S matrix
        n_series = S.shape[0]

        # Validate W for diagonal covariance type
        if covariance_type == CovarianceType.DIAGONAL:
            if W is None:
                raise ValueError(
                    "covariance_type='diagonal' requires `W` parameter. "
                    "Provide hierarchical covariance matrix of size (`base`, `base`), "
                    "or use covariance_type='full' or 'shrink' with residuals."
                )
            # Check W has valid diagonal
            w_diag = W.diagonal() if hasattr(W, 'diagonal') else np.diag(W)
            if np.any(w_diag <= 0):
                raise ValueError(
                    "W matrix has non-positive diagonal elements. "
                    "Covariance diagonal must be strictly positive."
                )
            if np.any(np.isnan(w_diag)):
                raise ValueError("W matrix contains NaN values on the diagonal.")

        # Warn if W is provided but will be ignored
        if covariance_type != CovarianceType.DIAGONAL and W is not None:
            warnings.warn(
                f"W parameter is ignored when covariance_type='{covariance_type.value}'. "
                "Covariance will be computed from residuals instead.",
                UserWarning,
                stacklevel=2,
            )

        # Warn if shrinkage_ridge is provided but not used
        if covariance_type != CovarianceType.SHRINK and shrinkage_ridge != self._DEFAULT_SHRINKAGE_RIDGE:
            warnings.warn(
                f"shrinkage_ridge parameter is only used when covariance_type='shrink'. "
                f"Current covariance_type='{covariance_type.value}', shrinkage_ridge will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Validate residuals for full/shrink types
        if covariance_type in (CovarianceType.FULL, CovarianceType.SHRINK):
            if residuals is None:
                raise ValueError(
                    f"covariance_type='{covariance_type.value}' requires `residuals` parameter. "
                    "Provide insample residuals of size (`base`, `obs`)."
                )

            # Validate residuals is a 2D array
            if residuals.ndim != 2:
                raise ValueError(
                    f"residuals must be a 2D array of shape (`base`, `obs`), "
                    f"got {residuals.ndim}D array with shape {residuals.shape}."
                )

            # Validate residuals shape matches number of series
            if residuals.shape[0] != n_series:
                raise ValueError(
                    f"residuals shape mismatch: first dimension ({residuals.shape[0]}) "
                    f"must match number of series in S ({n_series}). "
                    f"Expected residuals shape: ({n_series}, obs)."
                )

            # Validate residuals has enough observations
            if residuals.shape[1] == 0:
                raise ValueError(
                    "residuals is empty (0 observations). "
                    "Provide residuals with at least 2 observations."
                )

            if residuals.shape[1] < 2:
                raise ValueError(
                    f"residuals has only {residuals.shape[1]} observation(s). "
                    "At least 2 observations are required to compute covariance."
                )

            # Check for all-NaN series
            nan_mask = np.isnan(residuals)
            nan_counts_per_series = nan_mask.sum(axis=1)
            all_nan_series = nan_counts_per_series == residuals.shape[1]
            if np.any(all_nan_series):
                all_nan_indices = np.where(all_nan_series)[0]
                raise ValueError(
                    f"residuals contains series with all NaN values at indices: "
                    f"{all_nan_indices.tolist()}. Each series must have at least "
                    "2 non-NaN observations."
                )

            # Check for series with insufficient non-NaN observations
            non_nan_counts = residuals.shape[1] - nan_counts_per_series
            insufficient_obs = non_nan_counts < 2
            if np.any(insufficient_obs):
                insufficient_indices = np.where(insufficient_obs)[0]
                raise ValueError(
                    f"residuals contains series with fewer than 2 non-NaN observations "
                    f"at indices: {insufficient_indices.tolist()}. Each series must have "
                    "at least 2 non-NaN observations to compute covariance."
                )

            # Warn about n_series > n_observations for full covariance
            n_obs = residuals.shape[1] - nan_counts_per_series.max()
            if covariance_type == CovarianceType.FULL and n_series > n_obs:
                warnings.warn(
                    f"covariance_type='full' with n_series ({n_series}) > n_observations ({n_obs}) "
                    "may produce a non-positive-definite covariance matrix. "
                    "Consider using covariance_type='shrink' for better numerical stability.",
                    UserWarning,
                    stacklevel=2,
                )

        self.S = S
        self.P = P
        self.y_hat = y_hat
        self.covariance_type = covariance_type
        self.residuals = residuals
        self.shrinkage_ridge = shrinkage_ridge

        if isinstance(P, sp.linalg.LinearOperator) and sp.issparse(S):
            self.SP = sp.linalg.aslinearoperator(self.S) @ self.P
        else:
            self.SP = self.S @ self.P
        self.W = W
        self.sigmah = sigmah
        self.seed = seed

        # Compute correlation matrix based on covariance_type
        R1 = self._compute_correlation_matrix(covariance_type, W, residuals, shrinkage_ridge)

        # Store for potential inspection
        self._correlation_matrix = R1

        # Using elementwise multiplication
        cov_recs = []
        sigmah_recs = []
        for sigma in self.sigmah.T:
            # Broadcast sigma to create a matrix of pairwise products
            sigma_matrix = np.outer(sigma, sigma)
            # Element-wise multiplication with correlation matrix
            if sp.issparse(R1):
                cov_matrix = R1.multiply(sigma_matrix).toarray()
                cov_rec = self.SP @ cov_matrix @ self.SP.T
            else:
                # If R1 is dense, use numpy multiplication
                cov_matrix = R1 * sigma_matrix
                cov_rec = self.SP @ cov_matrix @ self.SP.T
            cov_recs.append(cov_rec)
            diag_cov = cov_rec.diagonal()
            if np.any(diag_cov < 0):
                n_neg = np.count_nonzero(diag_cov < 0)
                warnings.warn(
                    f"Detected {n_neg} negative variance value(s) in reconciled covariance "
                    "matrix diagonal. This indicates a non-positive-definite covariance "
                    "matrix. Negative variances will be clamped to zero.",
                    RuntimeWarning,
                )
            sigmah_recs.append(np.sqrt(np.maximum(diag_cov, 0)))

        self.sigmah_rec = np.hstack(sigmah_recs).reshape(-1, self.sigmah.shape[0]).T
        self.cov_rec = cov_recs

    def _compute_correlation_matrix(
        self,
        covariance_type: CovarianceType,
        W: np.ndarray | sp.spmatrix | None,
        residuals: np.ndarray | None,
        shrinkage_ridge: float,
    ) -> np.ndarray:
        """Compute correlation matrix based on covariance type.

        Args:
            covariance_type: The covariance estimation method to use.
            W: Hierarchical covariance matrix (used for diagonal type).
            residuals: Insample residuals (used for full/shrink types).
            shrinkage_ridge: Ridge parameter for shrinkage estimator.

        Returns:
            Correlation matrix of size (n_series, n_series).
        """
        if covariance_type == CovarianceType.DIAGONAL and W is not None:
            # Original behavior: use W diagonal with correlation scaling
            diag = W.diagonal() if hasattr(W, "diagonal") else np.diag(W)
            std_ = np.sqrt(diag)
            R1 = W / np.outer(std_, std_)
        elif covariance_type == CovarianceType.FULL:
            R1 = self._compute_full_correlation(residuals)
        elif covariance_type == CovarianceType.SHRINK:
            R1 = self._compute_shrink_correlation(residuals, shrinkage_ridge)

        return R1

    def _compute_full_correlation(self, residuals: np.ndarray) -> np.ndarray:
        """Compute full empirical correlation matrix from residuals.

        Args:
            residuals: Insample residuals of size (n_series, n_obs).

        Returns:
            Correlation matrix of size (n_series, n_series).
        """
        nan_mask = np.isnan(residuals)
        if np.any(nan_mask):
            cov_matrix = _ma_cov(residuals, ~nan_mask)
        else:
            cov_matrix = np.cov(residuals)

        # Handle edge case: np.cov returns scalar for single series
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])

        return self._covariance_to_correlation(cov_matrix)

    def _compute_shrink_correlation(
        self, residuals: np.ndarray, shrinkage_ridge: float
    ) -> np.ndarray:
        """Compute shrinkage correlation matrix using Schäfer-Strimmer method.

        Args:
            residuals: Insample residuals of size (n_series, n_obs).
            shrinkage_ridge: Ridge parameter for numerical stability.

        Returns:
            Correlation matrix of size (n_series, n_series).
        """
        nan_mask = np.isnan(residuals)
        if np.any(nan_mask):
            cov_matrix = _shrunk_covariance_schaferstrimmer_with_nans(
                residuals, ~nan_mask, shrinkage_ridge
            )
        else:
            cov_matrix = _shrunk_covariance_schaferstrimmer_no_nans(
                residuals, shrinkage_ridge
            )

        return self._covariance_to_correlation(cov_matrix)

    def _covariance_to_correlation(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix with numerical safeguards.

        Args:
            cov_matrix: Covariance matrix of size (n_series, n_series).

        Returns:
            Correlation matrix of size (n_series, n_series).

        Warns:
            UserWarning: If any series has zero or near-zero variance.
        """
        std_ = np.sqrt(np.diag(cov_matrix))

        # Check for zero/near-zero variance series
        low_variance_mask = std_ < self._MIN_STD_THRESHOLD
        if np.any(low_variance_mask):
            low_variance_indices = np.where(low_variance_mask)[0]
            warnings.warn(
                f"Series at indices {low_variance_indices.tolist()} have zero or "
                f"near-zero variance (std < {self._MIN_STD_THRESHOLD}). "
                "Correlation estimates for these series may be unreliable. "
                "Consider removing constant series or using regularization.",
                UserWarning,
                stacklevel=4,
            )
            # Clamp to minimum threshold to avoid division by zero
            std_[low_variance_mask] = self._MIN_STD_THRESHOLD

        # Convert to correlation
        correlation = cov_matrix / np.outer(std_, std_)

        # Ensure diagonal is exactly 1.0 (numerical precision fix)
        np.fill_diagonal(correlation, 1.0)

        # Clamp off-diagonal to [-1, 1] for numerical stability
        correlation = np.clip(correlation, -1.0, 1.0)

        return correlation

    def get_samples(self, num_samples: int):
        """Normality Coherent Samples.

        Obtains coherent samples under the Normality assumptions.

        Args:
            num_samples (int): number of samples generated from coherent distribution.

        Returns:
            samples (np.ndarray): Coherent samples of size (`base`, `horizon`, `num_samples`).
        """
        rng = np.random.default_rng(self.seed)
        n_series, n_horizon = self.y_hat.shape
        samples = np.empty(shape=(num_samples, n_series, n_horizon))
        for t in range(n_horizon):
            with warnings.catch_warnings():
                # Avoid 'RuntimeWarning: covariance is not positive-semidefinite.'
                # By definition the multivariate distribution is not full-rank
                partial_samples = rng.multivariate_normal(
                    mean=self.SP @ self.y_hat[:, t],
                    cov=self.cov_rec[t],
                    size=num_samples,
                )
            samples[:, :, t] = partial_samples

        # [samples, N, H] -> [N, H, samples]
        samples = samples.transpose((1, 2, 0))
        return samples

    def get_prediction_levels(self, res, level):
        """Adds reconciled forecast levels to results dictionary

        Args:
            res (dict): Results dictionary to update.
            level (list): Confidence levels.

        Returns:
            dict: Updated results dictionary.
        """
        res["sigmah"] = self.sigmah_rec
        level = np.asarray(level)
        z = norm.ppf(0.5 + level / 200)
        for zs, lv in zip(z, level, strict=False):
            res[f"lo-{lv}"] = res["mean"] - zs * self.sigmah_rec
            res[f"hi-{lv}"] = res["mean"] + zs * self.sigmah_rec
        return res

    def get_prediction_quantiles(self, res, quantiles):
        """Adds reconciled forecast quantiles to results dictionary

        Args:
            res (dict): Results dictionary to update.
            quantiles (np.ndarray): Quantiles to compute.

        Returns:
            dict: Updated results dictionary.
        """
        # [N,H,None] + [None None,Q] * [N,H,None] -> [N,H,Q]
        z = norm.ppf(quantiles)
        res["sigmah"] = self.sigmah_rec
        res["quantiles"] = (
            res["mean"][:, :, None] + z[None, None, :] * self.sigmah_rec[:, :, None]
        )
        return res


class Bootstrap:
    r"""Bootstrap Probabilistic Reconciliation Class.

    This method goes beyond the normality assumption for the base forecasts,
    the technique simulates future sample paths and uses them to generate
    base sample paths that are latered reconciled. This clever idea and its
    simplicity allows to generate coherent bootstraped prediction intervals
    for any reconciliation strategy. This class is meant to be used as the `sampler`
    input as other `HierarchicalForecast` [reconciliation classes](./methods).

    Given a boostraped set of simulated sample paths:
    ```math
    \hat{\mathbf{y}}^{[1]}_{\\tau}, \dots ,\hat{\mathbf{y}}^{[B]}_{\\tau})
    ```

    The reconciled sample paths allow for reconciled distributional forecasts:
    ```math
    (\mathbf{S}\mathbf{P}\hat{\mathbf{y}}^{[1]}_{\\tau}, \dots ,\mathbf{S}\mathbf{P}\hat{\mathbf{y}}^{[B]}_{\\tau})
    ```

    Args:
        S: np.array, summing matrix of size (`base`, `bottom`).
        P: np.array, reconciliation matrix of size (`bottom`, `base`).
        y_hat: Point forecasts values of size (`base`, `horizon`).
        y_insample: Insample values of size (`base`, `insample_size`).
        y_hat_insample: Insample point forecasts of size (`base`, `insample_size`).
        num_samples: int, number of bootstraped samples generated.
        seed: int, random seed for numpy generator's replicability.

    References:
        - [Puwasala Gamakumara Ph. D. dissertation. Monash University, Econometrics and Business Statistics (2020). "Probabilistic Forecast Reconciliation"](https://bridges.monash.edu/articles/thesis/Probabilistic_Forecast_Reconciliation_Theory_and_Applications/11869533)
        - [Panagiotelis A., Gamakumara P. Athanasopoulos G., and Hyndman R. J. (2022). "Probabilistic forecast reconciliation: Properties, evaluation and score optimisation". European Journal of Operational Research.](https://www.sciencedirect.com/science/article/pii/S0377221722006087)
    """

    def __init__(
        self,
        S: np.ndarray | sp.spmatrix,
        P: np.ndarray | sp.spmatrix,
        y_hat: np.ndarray,
        y_insample: np.ndarray,
        y_hat_insample: np.ndarray,
        num_samples: int = 100,
        seed: int = 0,
        W: np.ndarray | sp.spmatrix = None,
    ):
        self.S = S
        self.P = P
        self.W = W
        self.y_hat = y_hat
        self.y_insample = y_insample
        self.y_hat_insample = y_hat_insample
        self.num_samples = num_samples
        self.seed = seed

    def get_samples(self, num_samples: int):
        """Bootstrap Sample Reconciliation Method.

        Applies Bootstrap sample reconciliation method as defined by Gamakumara 2020.
        Generating independent sample paths and reconciling them with Bootstrap.

        Args:
            num_samples: int, number of samples generated from coherent distribution.

        Returns:
            samples: Coherent samples of size (`base`, `horizon`, `num_samples`).
        """
        residuals = self.y_insample - self.y_hat_insample
        h = self.y_hat.shape[1]

        # removing nas from residuals
        residuals = residuals[:, np.isnan(residuals).sum(axis=0) == 0]
        sample_idx = np.arange(residuals.shape[1] - h)
        rng = np.random.default_rng(self.seed)
        samples_idx = rng.choice(sample_idx, size=num_samples)
        samples = [self.y_hat + residuals[:, idx : (idx + h)] for idx in samples_idx]
        SP = self.S @ self.P
        samples = np.apply_along_axis(lambda path: SP @ path, axis=1, arr=samples)
        samples_np = np.stack(samples)

        # [samples, N, H] -> [N, H, samples]
        samples_np = samples_np.transpose((1, 2, 0))
        return samples_np

    def get_prediction_levels(self, res, level):
        """Adds reconciled forecast levels to results dictionary"""
        samples = self.get_samples(num_samples=self.num_samples)
        for lv in level:
            min_q = (100 - lv) / 200
            max_q = min_q + lv / 100
            res[f"lo-{lv}"] = np.quantile(samples, min_q, axis=2)
            res[f"hi-{lv}"] = np.quantile(samples, max_q, axis=2)
        return res

    def get_prediction_quantiles(self, res, quantiles):
        """Adds reconciled forecast quantiles to results dictionary"""
        samples = self.get_samples(num_samples=self.num_samples)

        # [Q, N, H] -> [N, H, Q]
        sample_quantiles = np.quantile(samples, quantiles, axis=2)
        res["quantiles"] = sample_quantiles.transpose((1, 2, 0))
        return res


class PERMBU:
    r"""PERMBU Probabilistic Reconciliation Class.

    The PERMBU method leverages empirical bottom-level marginal distributions
    with empirical copula functions (describing bottom-level dependencies) to
    generate the distribution of aggregate-level distributions using BottomUp
    reconciliation. The sample reordering technique in the PERMBU method reinjects
    multivariate dependencies into independent bottom-level samples. $\hat{\epsilon}_{i,t}$

    Algorithm:
    1.   For all series compute conditional marginals distributions.
    2.   Compute residuals $\hat{\epsilon}_{i,t}$ and obtain rank permutations.
    3.   Obtain K-sample from the bottom-level series predictions.
    4.   Apply recursively through the hierarchical structure:
        4.1.   For a given aggregate series $i$ and its children series:
        4.2.   Obtain children's empirical joint using sample reordering copula.
        4.2.   From the children's joint obtain the aggregate series's samples.

    Args:
        S (np.array): summing matrix of size (`base`, `bottom`).
        tags (dict[str, np.ndarray]): Each key is a level and each value its `S` indices.
        y_insample (np.array): Insample values of size (`base`, `insample_size`).
        y_hat_insample (np.array): Insample point forecasts of size (`base`, `insample_size`).
        sigmah (np.array): forecast standard dev. of size (`base`, `horizon`).
        num_samples (int): number of normal prediction samples generated. Default is None
        seed (int): random seed for numpy generator's replicability. Default is 0.

    References:
        - [Taieb, Souhaib Ben and Taylor, James W and Hyndman, Rob J. (2017). "Coherent probabilistic forecasts for hierarchical time series. International conference on machine learning ICML."](https://proceedings.mlr.press/v70/taieb17a.html)
    """

    def __init__(
        self,
        S: np.ndarray | sp.spmatrix,
        tags: dict[str, np.ndarray],
        y_hat: np.ndarray,
        y_insample: np.ndarray,
        y_hat_insample: np.ndarray,
        sigmah: np.ndarray,
        num_samples: int | None = None,
        seed: int = 0,
        P: np.ndarray | sp.spmatrix = None,
    ):
        # PERMBU only works for strictly hierarchical structures
        if not is_strictly_hierarchical(S, tags):
            raise ValueError(
                "PERMBU probabilistic reconciliation requires strictly hierarchical structures."
            )
        self.S = S
        self.P = P
        self.y_hat = y_hat
        self.y_insample = y_insample
        self.y_hat_insample = y_hat_insample
        self.sigmah = sigmah
        self.num_samples = num_samples
        self.seed = seed

    def _obtain_ranks(self, array):
        """Vector ranks

        Efficiently obtain vector ranks.
        Example `array=[4,2,7,1]` -> `ranks=[2, 1, 3, 0]`.

        Args:
            array (np.ndarray): Matrix with floats or integers on which the ranks will be computed on the second dimension.

        Returns:
            np.ndarray: Matrix with ranks along the second dimension.
        """
        temp = array.argsort(axis=1)
        ranks = np.empty_like(temp)
        a_range = np.arange(temp.shape[1])
        for i_row in range(temp.shape[0]):
            ranks[i_row, temp[i_row, :]] = a_range
        return ranks

    def _permutate_samples(self, samples, permutations):
        """Permutate Samples

        Applies efficient vectorized permutation on the samples.

        Args:
            samples (np.ndarray): Independent base samples.
            permutations (np.ndarray): Permutation ranks with which `samples` dependence will be restored see `_obtain_ranks`.

        Returns:
            np.ndarray: Permutated samples.
        """
        # Generate auxiliary and flat permutation indexes
        n_rows, n_cols = permutations.shape
        aux_row_idx = np.arange(n_rows)[:, None] * n_cols
        aux_row_idx = np.repeat(aux_row_idx, repeats=n_cols, axis=1)
        permutate_idxs = permutations.flatten() + aux_row_idx.flatten()

        # Apply flat permutation indexes and recover original shape
        permutated_samples = samples.flatten()
        permutated_samples = permutated_samples[permutate_idxs]
        permutated_samples = permutated_samples.reshape(n_rows, n_cols)
        return permutated_samples

    def _permutate_predictions(self, prediction_samples, permutations):
        """Permutate Prediction Samples

        Applies permutations to prediction_samples across the horizon.

        Args:
            prediction_samples (np.ndarray): Independent base prediction samples.
            permutations (np.ndarray): Permutation ranks with which `samples` dependence will be restored see `_obtain_ranks`.

        Returns:
            np.ndarray: Permutated prediction samples.
        """
        # Apply permutation throughout forecast horizon
        permutated_prediction_samples = prediction_samples.copy()

        _, n_horizon, _ = prediction_samples.shape
        for t in range(n_horizon):
            permutated_prediction_samples[:, t, :] = self._permutate_samples(
                prediction_samples[:, t, :], permutations
            )
        return permutated_prediction_samples

    def _nonzero_indexes_by_row(self, M):
        nonzeros = M.nonzero()[1]
        nnz_per_row = int(M[0, :].sum())
        return nonzeros.reshape(-1, nnz_per_row)

    def get_samples(self, num_samples: int | None = None):
        """PERMBU Sample Reconciliation Method.

        Applies PERMBU reconciliation method as defined by Taieb et. al 2017.
        Generating independent base prediction samples, restoring its multivariate
        dependence using estimated copula with reordering and applying the BottomUp
        aggregation to the new samples.

        Args:
            num_samples (int): number of samples generated from coherent distribution.

        Returns:
            samples (np.ndarray): Coherent samples of size (`base`, `horizon`, `num_samples`).
        """
        # Compute residuals and rank permutations
        residuals = self.y_insample - self.y_hat_insample
        residuals = residuals[:, np.isnan(residuals).sum(axis=0) == 0]

        # Sample h step-ahead base marginal distributions
        if num_samples is None:
            num_samples = residuals.shape[1]

        # Expand residuals to match num_samples [(a,b),T] -> [(a,b),num_samples]
        rng = np.random.default_rng(self.seed)
        if num_samples > residuals.shape[1]:
            residuals_idxs = rng.choice(residuals.shape[1], size=num_samples)
        else:
            residuals_idxs = rng.choice(
                residuals.shape[1], size=num_samples, replace=False
            )
        residuals = residuals[:, residuals_idxs]
        rank_permutations = self._obtain_ranks(residuals)

        n_series, n_horizon = self.y_hat.shape

        base_samples = np.array(
            [
                rng.normal(loc=m, scale=s, size=num_samples)
                for m, s in zip(self.y_hat.flatten(), self.sigmah.flatten(), strict=False)
            ]
        )
        base_samples = base_samples.reshape(n_series, n_horizon, num_samples)

        # Initialize PERMBU utility
        rec_samples = base_samples.copy()
        try:
            encoder = OneHotEncoder(sparse_output=False, dtype=np.float64)
        except TypeError:
            encoder = OneHotEncoder(sparse=False, dtype=np.float64)
        hier_links = self._nonzero_indexes_by_row(self.S.T)

        # BottomUp hierarchy traversing
        hier_levels = hier_links.shape[1] - 1
        for level_idx in reversed(range(hier_levels)):
            # Obtain aggregation matrix from parent/children links
            children_links = np.unique(hier_links[:, level_idx : level_idx + 2], axis=0)
            children_idxs = np.unique(children_links[:, 1])
            parent_idxs = np.unique(children_links[:, 0])
            Agg = encoder.fit_transform(children_links).T
            Agg = Agg[: len(parent_idxs), :]

            # Permute children_samples for each prediction step
            children_permutations = rank_permutations[children_idxs, :]
            children_samples = rec_samples[children_idxs, :, :]
            children_samples = self._permutate_predictions(
                prediction_samples=children_samples, permutations=children_permutations
            )

            # Overwrite hier_samples with BottomUp aggregation
            # and randomly shuffle parent predictions after aggregation
            parent_samples = np.einsum("ab,bhs->ahs", Agg, children_samples)
            random_permutation = np.array(
                [
                    rng.permutation(np.arange(num_samples))
                    for serie in range(len(parent_samples))
                ]
            )
            parent_samples = self._permutate_predictions(
                prediction_samples=parent_samples, permutations=random_permutation
            )

            rec_samples[parent_idxs, :, :] = parent_samples
        return rec_samples

    def get_prediction_levels(self, res, level):
        """Adds reconciled forecast levels to results dictionary"""
        samples = self.get_samples(num_samples=self.num_samples)
        for lv in level:
            min_q = (100 - lv) / 200
            max_q = min_q + lv / 100
            res[f"lo-{lv}"] = np.quantile(samples, min_q, axis=2)
            res[f"hi-{lv}"] = np.quantile(samples, max_q, axis=2)
        return res

    def get_prediction_quantiles(self, res, quantiles):
        """Adds reconciled forecast quantiles to results dictionary"""
        samples = self.get_samples(num_samples=self.num_samples)

        # [Q, N, H] -> [N, H, Q]
        sample_quantiles = np.quantile(samples, quantiles, axis=2)
        res["quantiles"] = sample_quantiles.transpose((1, 2, 0))
        return res
