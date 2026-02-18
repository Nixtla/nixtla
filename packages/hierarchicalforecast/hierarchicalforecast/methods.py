__all__ = ['BottomUp', 'BottomUpSparse', 'TopDown', 'TopDownSparse', 'MiddleOut', 'MiddleOutSparse', 'MinTrace', 'MinTraceSparse',
           'OptimalCombination', 'ERM']


import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import clarabel
import numpy as np
from qpsolvers import solve_qp
from scipy import sparse

from hierarchicalforecast.utils import (
    _construct_adjacency_matrix,
    _is_strictly_hierarchical,
    _lasso,
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
    is_strictly_hierarchical,
)

from .probabilistic_methods import PERMBU, Bootstrap, Conformal, Normality


class HReconciler:
    """Base class for hierarchical forecast reconciliation methods.

    Class Attributes:
        fitted (bool): Whether the reconciler has been fitted to data.
        is_sparse_method (bool): Whether this method uses sparse matrix operations.
        insample (bool): Whether this method requires insample data.
        is_strictly_hierarchical (bool): Whether this method requires a strictly
            hierarchical structure (tree-like) or supports grouped hierarchies.
            - True: Method requires each node to have exactly one parent (e.g., TopDown, MiddleOut)
            - False: Method works with any hierarchy structure, including grouped hierarchies
              where nodes can have multiple parents (e.g., BottomUp, MinTrace, ERM)
        P (np.ndarray | None): Projection matrix computed by the reconciliation method.
        sampler: Probabilistic sampler for generating prediction intervals.
        _init_params (dict | None): Stores initialization parameters for method naming.
    """
    fitted = False
    is_sparse_method = False
    insample = False
    is_strictly_hierarchical = False  # Default: supports grouped hierarchies
    P = None
    sampler = None
    _init_params: dict | None = None  # Stores initialization parameters for naming

    def _get_sampler(
        self,
        intervals_method,
        S,
        P,
        y_hat,
        y_insample,
        y_hat_insample,
        W,
        sigmah,
        num_samples,
        seed,
        tags,
    ):
        if intervals_method == "normality":
            sampler = Normality(S=S, P=P, y_hat=y_hat, W=W, sigmah=sigmah, seed=seed)
        elif intervals_method == "permbu":
            sampler = PERMBU(
                S=S,
                P=P,
                y_hat=(S @ (P @ y_hat)),
                tags=tags,
                y_insample=y_insample,
                y_hat_insample=y_hat_insample,
                sigmah=sigmah,
                num_samples=num_samples,
                seed=seed,
            )
        elif intervals_method == "bootstrap":
            sampler = Bootstrap(
                S=S,
                P=P,
                y_hat=y_hat,
                y_insample=y_insample,
                y_hat_insample=y_hat_insample,
                num_samples=num_samples,
                seed=seed,
            )
        elif intervals_method == "conformal":
            sampler = Conformal(
                S=S,
                P=P,
                y_hat=y_hat,
                y_cal=y_insample,
                y_hat_cal=y_hat_insample,
                seed=seed,
            )
        else:
            sampler = None
        return sampler

    def _reconcile(
        self,
        S: np.ndarray,
        P: np.ndarray,
        y_hat: np.ndarray,
        SP: np.ndarray = None,
        level: list[int] | None = None,
        sampler: Normality | PERMBU | Bootstrap | None = None,
    ):

        # Mean reconciliation
        res = {"mean": (S @ (P @ y_hat))}

        # Probabilistic reconciliation
        if (level is not None) and (sampler is not None):
            # Update results dictionary within
            # Vectorized quantiles
            quantiles = np.concatenate(
                [[(100 - lv) / 200, ((100 - lv) / 200) + lv / 100] for lv in level]
            )
            quantiles = np.sort(quantiles)
            res = sampler.get_prediction_quantiles(res, quantiles)

        return res

    def predict(
        self, S: np.ndarray, y_hat: np.ndarray, level: list[int] | None = None
    ):
        """Predict using reconciler.

        Predict using fitted mean and probabilistic reconcilers.

        Args:
            S (np.ndarray): Summing matrix of size (`base`, `bottom`).
            y_hat (np.ndarray): Forecast values of size (`base`, `horizon`).
            level (Optional[list[int]], optional): float list 0-100, confidence levels for prediction intervals. Default is None.

        Returns:
            y_tilde (dict): Reconciliated predictions.
        """
        if not self.fitted:
            raise Exception("This model instance is not fitted yet, Call fit method.")

        return self._reconcile(
            S=S, P=self.P, y_hat=y_hat, sampler=self.sampler, level=level
        )

    def sample(self, num_samples: int):
        """Sample probabilistic coherent distribution.

        Generates n samples from a probabilistic coherent distribution.
        The method uses fitted mean and probabilistic reconcilers, defined by
        the `intervals_method` selected during the reconciler's
        instantiation. Currently available: `normality`, `bootstrap`, `permbu`.

        Args:
            num_samples (int): number of samples generated from coherent distribution.

        Returns:
            samples (np.ndarray): Coherent samples of size (`num_series`, `horizon`, `num_samples`).
        """
        if not self.fitted:
            raise Exception("This model instance is not fitted yet, Call fit method.")
        if self.sampler is None:
            raise ValueError(
                "This model instance does not have sampler. Call fit with `intervals_method`."
            )

        samples = self.sampler.get_samples(num_samples=num_samples)
        return samples

    def fit(self, *args, **kwargs):

        raise NotImplementedError("This method is not implemented yet.")

    def fit_predict(self, *args, **kwargs):

        raise NotImplementedError("This method is not implemented yet.")

    __call__ = fit_predict


class BottomUp(HReconciler):
    r"""Bottom Up Reconciliation Class.

    The most basic hierarchical reconciliation is performed using an Bottom-Up strategy. It was proposed for
    the first time by Orcutt in 1968.
    The corresponding hierarchical \"projection\" matrix is defined as:
    ```math
    \mathbf{P}_{\\text{BU}} = [\mathbf{0}_{\mathrm{[b],[a]}}\;|\;\mathbf{I}_{\mathrm{[b][b]}}]
    ```

    Args:
        None

    References:
    - [Orcutt, G.H., Watts, H.W., & Edwards, J.B.(1968). "Data aggregation and
    information loss". The American Economic Review, 58 , 773(787)](http://www.jstor.org/stable/1815532).
    """

    insample = False
    is_strictly_hierarchical = False

    def __init__(self):
        self._init_params = {}

    def _get_PW_matrices(self, S):
        n_hiers, n_bottom = S.shape
        P = np.eye(n_bottom, n_hiers, n_hiers - n_bottom, np.float64)
        if getattr(self, "intervals_method", False) is None:
            W = None
        else:
            W = np.eye(n_hiers, dtype=np.float64)
        return P, W

    def fit(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ):
        """Bottom Up Fit Method.

        Args:
            S (np.ndarray): Summing matrix of size (`base`, `bottom`).
            y_hat (np.ndarray): Forecast values of size (`base`, `horizon`).
            y_insample (Optional[np.ndarray], optional): In-sample values of size (`base`, `horizon`). Default is None.
            y_hat_insample (Optional[np.ndarray], optional): In-sample forecast values of size (`base`, `horizon`). Default is None.
            sigmah (Optional[np.ndarray], optional): Estimated standard deviation of the conditional marginal distribution. Default is None.
            intervals_method (Optional[str], optional): Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`. Default is None.
            num_samples (Optional[int], optional): Number of samples for probabilistic coherent distribution. Default is None.
            seed (Optional[int], optional): Seed for reproducibility. Default is None.
            tags (Optional[dict[str, np.ndarray]], optional): Tags for hierarchical structure. Default is None.

        Returns:
            BottomUp (object): fitted reconciler.
        """
        self.intervals_method = intervals_method
        self.P, self.W = self._get_PW_matrices(S=S)
        self.sampler = self._get_sampler(
            S=S,
            P=self.P,
            W=self.W,
            y_hat=y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            sigmah=sigmah,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags,
        )
        self.fitted = True
        return self

    def fit_predict(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        level: list[int] | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ):
        """BottomUp Reconciliation Method.

        Args:
            S (np.ndarray): Summing matrix of size (`base`, `bottom`).
            y_hat (np.ndarray): Forecast values of size (`base`, `horizon`).
            y_insample (Optional[np.ndarray], optional): In-sample values of size (`base`, `insample_size`). Default is None.
            y_hat_insample (Optional[np.ndarray], optional): In-sample forecast values of size (`base`, `insample_size`). Default is None.
            sigmah (Optional[np.ndarray], optional): Estimated standard deviation of the conditional marginal distribution. Default is None.
            level (Optional[list[int]], optional): float list 0-100, confidence levels for prediction intervals. Default is None.
            intervals_method (Optional[str], optional): Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`. Default is None.
            num_samples (Optional[int], optional): Number of samples for probabilistic coherent distribution. Default is None.
            seed (Optional[int], optional): Seed for reproducibility. Default is None.
            tags (Optional[dict[str, np.ndarray]], optional): Tags for hierarchical structure. Default is None.

        Returns:
            y_tilde (dict): Reconciliated y_hat using the Bottom Up approach.
        """
        # Fit creates P, W and sampler attributes
        self.fit(
            S=S,
            y_hat=y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            sigmah=sigmah,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags,
        )

        return self._reconcile(
            S=S, P=self.P, y_hat=y_hat, sampler=self.sampler, level=level
        )

    __call__ = fit_predict


class BottomUpSparse(BottomUp):
    """BottomUpSparse Reconciliation Class.

    This is the implementation of a Bottom Up reconciliation using the sparse
    matrix approach. It works much more efficient on datasets with many time series.
    [makoren: At least I hope so, I only checked up until ~20k time series, and
    there's no real improvement, it would be great to check for smth like 1M time
    series, where the dense S matrix really stops fitting in memory]

    See the parent class for more details.
    """

    is_sparse_method = True
    is_strictly_hierarchical = False

    def __init__(self):
        self._init_params = {}

    def _get_PW_matrices(self, S):
        n_hiers, n_bottom = S.shape
        P = sparse.eye(n_bottom, n_hiers, n_hiers - n_bottom, np.float64, "csr")
        if getattr(self, "intervals_method", False) is None:
            W = None
        else:
            W = sparse.eye(n_hiers, dtype=np.float64, format="csr")
        return P, W


def _get_child_nodes(
    S: np.ndarray | sparse.csr_matrix, tags: dict[str, np.ndarray]
):
    if isinstance(S, sparse.spmatrix):
        S = S.toarray()
    level_names = list(tags.keys())
    nodes = OrderedDict()
    for i_level, level in enumerate(level_names[:-1]):
        parent = tags[level]
        child = np.zeros_like(S)
        idx_child = tags[level_names[i_level + 1]]
        child[idx_child] = S[idx_child]
        nodes_level = {}
        for idx_parent_node in parent:
            parent_node = S[idx_parent_node]
            idx_node = child * parent_node.astype(bool)
            (idx_node,) = np.where(idx_node.sum(axis=1) > 0)
            nodes_level[idx_parent_node] = [idx for idx in idx_child if idx in idx_node]
        nodes[level] = nodes_level
    return nodes


def _reconcile_fcst_proportions(
    S: np.ndarray,
    y_hat: np.ndarray,
    tags: dict[str, np.ndarray],
    nodes: dict[str, dict[int, np.ndarray]],
    idxs_top: np.ndarray,
):
    reconciled = np.zeros_like(y_hat)
    level_names = list(tags.keys())
    for idx_top in idxs_top:
        reconciled[idx_top] = y_hat[idx_top]
        for i_level, level in enumerate(level_names[:-1]):
            nodes_level = nodes[level]
            for idx_parent, idx_childs in nodes_level.items():
                fcst_parent = reconciled[idx_parent]
                childs_sum = y_hat[idx_childs].sum()
                for idx_child in idx_childs:
                    if np.abs(childs_sum) < 1e-8:
                        n_children = len(idx_childs)
                        reconciled[idx_child] = fcst_parent / n_children
                    else:
                        reconciled[idx_child] = (
                            y_hat[idx_child] * fcst_parent / childs_sum
                        )
    return reconciled


def _reconcile_fcst_proportions_bootstrap(
    S: np.ndarray | sparse.csr_matrix,
    y_hat: np.ndarray,
    tags: dict[str, np.ndarray],
    y_insample: np.ndarray,
    y_hat_insample: np.ndarray,
    num_samples: int,
    seed: int,
    level: list[int],
    nodes: dict[str, dict[int, np.ndarray]] | None = None,
    idxs_top: np.ndarray | None = None,
    A: sparse.csr_matrix | None = None,
):
    """Generate prediction intervals for forecast_proportions using bootstrap.

    This function generates bootstrap samples by adding resampled residuals to
    the base forecasts, then reconciles each sample using forecast proportions.
    Supports both dense and sparse summing matrices.

    Args:
        S: Summing matrix of size (`base`, `bottom`). Can be dense or sparse.
        y_hat: Forecast values of size (`base`, `horizon`).
        tags: Each key is a level and each value its `S` indices.
        y_insample: Insample values of size (`base`, `insample_size`).
        y_hat_insample: Insample forecast values of size (`base`, `insample_size`).
        num_samples: Number of bootstrap samples.
        seed: Random seed for reproducibility.
        level: Confidence levels for prediction intervals.
        nodes: Child nodes structure from _get_child_nodes (required for dense S).
        idxs_top: Indices of top-level nodes (required for dense S).
        A: Adjacency matrix (required for sparse S).

    Returns:
        quantiles: Array of shape (`base`, `horizon`, `num_quantiles`).
    """
    is_sparse = sparse.issparse(S)

    # Compute residuals
    residuals = y_insample - y_hat_insample
    h = y_hat.shape[1]

    # Remove NaN columns from residuals
    residuals = residuals[:, np.isnan(residuals).sum(axis=0) == 0]

    # Get valid sample indices
    sample_idx = np.arange(residuals.shape[1] - h)
    rng = np.random.default_rng(seed)

    # Generate bootstrap samples
    samples_idx = rng.choice(sample_idx, size=num_samples)
    bootstrap_samples = []

    for idx in samples_idx:
        # Add residual block to forecasts
        y_hat_sample = y_hat + residuals[:, idx : (idx + h)]

        if is_sparse:
            # Reconcile the bootstrap sample using sparse forecast proportions
            reconciled_sample = _reconcile_fcst_proportions_sparse(
                S=S,
                y_hat=y_hat_sample,
                A=A,
                tags=tags,
            )
        else:
            # Reconcile the bootstrap sample using dense forecast proportions
            reconciled_sample = np.hstack(
                [
                    _reconcile_fcst_proportions(
                        S=S,
                        y_hat=y_hat_sample_col[:, None],
                        tags=tags,
                        nodes=nodes, # type: ignore[arg-type]
                        idxs_top=idxs_top,
                    )
                    for y_hat_sample_col in y_hat_sample.T
                ]
            )
        bootstrap_samples.append(reconciled_sample)

    # Stack samples: [num_samples, n_series, horizon]
    samples = np.stack(bootstrap_samples)
    # Transpose to [n_series, horizon, num_samples]
    samples = samples.transpose((1, 2, 0))

    # Compute quantiles
    quantiles = np.concatenate(
        [[(100 - lv) / 200, ((100 - lv) / 200) + lv / 100] for lv in level]
    )
    quantiles = np.sort(quantiles)

    # [Q, N, H] -> [N, H, Q]
    sample_quantiles = np.quantile(samples, quantiles, axis=2)
    return sample_quantiles.transpose((1, 2, 0))


def _reconcile_fcst_proportions_sparse(
    S: sparse.csr_matrix,
    y_hat: np.ndarray,
    A: sparse.csr_matrix,
    tags: dict[str, np.ndarray],
):
    """Reconcile forecasts using sparse forecast proportions.

    This is a helper that implements the sparse forecast proportions reconciliation
    for a single forecast matrix.

    Args:
        S: Sparse summing matrix of size (`base`, `bottom`).
        y_hat: Forecast values of size (`base`, `horizon`).
        A: Adjacency matrix.
        tags: Each key is a level and each value its `S` indices.

    Returns:
        y_tilde: Reconciled forecasts of size (`base`, `horizon`).
    """
    # Make a copy to avoid modifying the original
    y_hat = y_hat.copy()
    # As we may have zero sibling sums, replace any zeroes with eps.
    y_hat[y_hat == 0.0] = np.finfo(np.float64).eps
    # Calculate the relative proportions for each node.
    with np.errstate(divide="ignore"):
        P = y_hat / ((A.T @ A) @ y_hat)
    # Get the number of root nodes.
    n = len(next(iter(tags.values())))
    # Set the relative proportion(s) of the root node(s).
    P[:n, :] = 1.0
    # Precompute the transpose of the summing matrix.
    S_T = S.T
    # Propagate the relative proportions for the nodes along each leaf
    # node's disaggregation pathway.
    y_tilde = np.array(
        [
            S
            @ (
                S_T[:, :n].multiply(
                    np.prod(
                        np.vstack(S_T.multiply(P[:, i]).tolil().data), 1
                    ).reshape(-1, 1)
                )
                @ y_hat[:n, i]
            )
            for i in range(y_hat.shape[1])
        ]
    ).T
    return y_tilde


class TopDown(HReconciler):
    r"""Top Down Reconciliation Class.

    The Top Down hierarchical reconciliation method, distributes the total aggregate predictions and decomposes
    it down the hierarchy using proportions $\mathbf{p}_{\mathrm{[b]}}$ that can be actual historical values
    or estimated.

    ```math
    \mathbf{P}=[\mathbf{p}_{\mathrm{[b]}}\;|\;\mathbf{0}_{\mathrm{[b][a,b\;-1]}}]
    ```

    Args:
        method (str): One of `forecast_proportions`, `average_proportions` and `proportion_averages`.

    References:
    - [CW. Gross (1990). "Disaggregation methods to expedite product line forecasting". Journal of Forecasting, 9 , 233-254. doi:10.1002/for.3980090304](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3980090304).
    - [G. Fliedner (1999). "An investigation of aggregate variable time series forecast strategies with specific subaggregate time series statistical correlation". Computers and Operations Research, 26 , 1133-1149. doi:10.1016/S0305-0548(99)00017-9](https://doi.org/10.1016/S0305-0548(99)00017-9).
    """

    is_strictly_hierarchical = True

    def __init__(self, method: str):
        if method not in [
            "forecast_proportions",
            "average_proportions",
            "proportion_averages",
        ]:
            raise ValueError(
                f"Unknown method `{method}`. Choose from `forecast_proportions`, `average_proportions`, `proportion_averages`."
            )
        self.method = method
        self.insample = method in ["average_proportions", "proportion_averages"]
        self._init_params = {"method": method}

    def _get_PW_matrices(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray,
        tags: dict[str, np.ndarray] | None = None,
    ):

        n_hiers, n_bottom = S.shape

        # Check if the data structure is strictly hierarchical.
        if tags is not None:
            if not is_strictly_hierarchical(S, tags):
                raise ValueError(
                    "Top-down reconciliation requires strictly hierarchical structures."
                )
            idx_top = int(S.sum(axis=1).argmax())
            levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
            idx_bottom = levels_[list(levels_)[-1]]
            y_btm = y_insample[idx_bottom]
        else:
            idx_top = 0
            y_btm = y_insample[(n_hiers - n_bottom) :]

        y_top = y_insample[idx_top]

        if self.method == "average_proportions":
            prop = np.nanmean(y_btm / y_top, axis=1)
        elif self.method == "proportion_averages":
            prop = np.nanmean(y_btm, axis=1) / np.nanmean(y_top)
        else:
            raise ValueError(f"Unknown method {self.method}")

        if np.isnan(y_btm).any() or np.isnan(y_top).any():
            warnings.warn(
                """
                Warning: There are NaN values in one or more levels of Y_df.
                This may lead to unexpected behavior when computing average proportions and proportion averages.
                """
            )

        P = np.zeros_like(
            S, np.float64
        ).T  # float 64 if prop is too small, happens with wiki2
        P[:, idx_top] = prop
        W = np.eye(n_hiers, dtype=np.float64)
        return P, W

    def fit(
        self,
        S,
        y_hat,
        y_insample: np.ndarray,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ):
        """TopDown Fit Method.

        Args:
            S (np.ndarray): Summing matrix of size (`base`, `bottom`).
            y_hat (np.ndarray): Forecast values of size (`base`, `horizon`).
            y_insample (np.ndarray): Insample values of size (`base`, `insample_size`). Optional for `forecast_proportions` method.
            y_hat_insample (np.ndarray): Insample forecast values of size (`base`, `insample_size`). Optional for `forecast_proportions` method.
            sigmah (np.ndarray): Estimated standard deviation of the conditional marginal distribution.
            intervals_method (str): Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`.
            num_samples (int): Number of samples for probabilistic coherent distribution.
            seed (int): Seed for reproducibility.
            tags (dict[str, np.ndarray]): Each key is a level and each value its `S` indices.

        Returns:
            TopDown (object): fitted reconciler.
        """
        self.intervals_method = intervals_method
        self.P, self.W = self._get_PW_matrices(
            S=S, y_hat=y_hat, tags=tags, y_insample=y_insample
        )
        self.sampler = self._get_sampler(
            S=S,
            P=self.P,
            W=self.W,
            y_hat=y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            sigmah=sigmah,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags,
        )
        self.fitted = True
        return self

    def fit_predict(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        tags: dict[str, np.ndarray],
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        level: list[int] | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
    ):
        """Top Down Reconciliation Method.

        Args:
            S (np.ndarray): Summing matrix of size (`base`, `bottom`).
            y_hat (np.ndarray): Forecast values of size (`base`, `horizon`).
            tags (dict[str, np.ndarray]): Each key is a level and each value its `S` indices.
            y_insample (np.ndarray): Insample values of size (`base`, `insample_size`). Optional for `forecast_proportions` method. Default is None.
            y_hat_insample (np.ndarray): Insample forecast values of size (`base`, `insample_size`). Optional for `forecast_proportions` method. Default is None.
            sigmah (np.ndarray): Estimated standard deviation of the conditional marginal distribution. Default is None.
            level (list[int]): float list 0-100, confidence levels for prediction intervals. Default is None.
            intervals_method (str): Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`. Default is None.
            num_samples (int): Number of samples for probabilistic coherent distribution. Default is None.
            seed (int): Seed for reproducibility.

        Returns:
            y_tilde (np.ndarray): Reconciliated y_hat using the Top Down approach.
        """
        if self.method == "forecast_proportions":
            if not getattr(self, '_hierarchy_validated', False):
                # Check if the data structure is strictly hierarchical.
                if tags is not None and not is_strictly_hierarchical(S, tags):
                    raise ValueError(
                        "Top-down reconciliation requires strictly hierarchical structures."
                    )
            S_sum = np.sum(S, axis=1)
            if S.shape[1] > 1:
                S_max_idxs = np.argsort(S_sum)[::-1]
                idxs_top = S_max_idxs[np.cumsum(S_sum[S_max_idxs]) <= S.shape[1]]
            else:
                idxs_top = np.array([np.argmax(S_sum)])
            levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
            nodes = _get_child_nodes(S=S, tags=levels_)
            reconciled = [
                _reconcile_fcst_proportions(
                    S=S,
                    y_hat=y_hat_[:, None],
                    tags=levels_,
                    nodes=nodes,
                    idxs_top=idxs_top,
                )
                for y_hat_ in y_hat.T
            ]
            reconciled = np.hstack(reconciled)
            res = {"mean": reconciled}

            # Compute prediction intervals using bootstrap if requested
            if level is not None:
                if y_insample is None or y_hat_insample is None:
                    raise ValueError(
                        "Prediction intervals for `forecast_proportions` require "
                        "`y_insample` and `y_hat_insample`."
                    )
                if intervals_method != "bootstrap":
                    raise ValueError(
                        "Only `bootstrap` intervals_method is implemented for "
                        "`forecast_proportions`."
                    )
                if num_samples is None:
                    num_samples = 100
                if seed is None:
                    seed = 0
                res["quantiles"] = _reconcile_fcst_proportions_bootstrap(
                    S=S,
                    y_hat=y_hat,
                    tags=levels_,
                    y_insample=y_insample,
                    y_hat_insample=y_hat_insample,
                    num_samples=num_samples,
                    seed=seed,
                    level=level,
                    nodes=nodes,
                    idxs_top=idxs_top,
                )
            return res
        else:
            # Fit creates P, W and sampler attributes
            self.fit(
                S=S,
                y_hat=y_hat,
                y_insample=y_insample,
                y_hat_insample=y_hat_insample,
                sigmah=sigmah,
                intervals_method=intervals_method,
                num_samples=num_samples,
                seed=seed,
                tags=tags,
            )
            return self._reconcile(
                S=S, P=self.P, y_hat=y_hat, level=level, sampler=self.sampler
            )

    __call__ = fit_predict


class TopDownSparse(TopDown):
    """TopDownSparse Reconciliation Class.

    This is an implementation of top-down reconciliation using the sparse matrix
    approach. It works much more efficiently on data sets with many time series.

    See the parent class for more details.
    """

    is_sparse_method = True
    is_strictly_hierarchical = True

    def _get_PW_matrices(
        self,
        S: sparse.csr_matrix,
        y_hat: np.ndarray,
        y_insample: np.ndarray,
        tags: dict[str, np.ndarray] | None = None,
    ):
        # Avoid a redundant check during middle-out reconciliation.
        if not getattr(self, '_hierarchy_validated', False):
            # Check if the data structure is strictly hierarchical.
            if tags is not None and not _is_strictly_hierarchical(
                _construct_adjacency_matrix(S, tags), tags
            ):
                raise ValueError(
                    "Top-down reconciliation requires strictly hierarchical structures."
                )

        # Get the dimensions of the "summing" matrix.
        n_hiers, n_bottom = S.shape

        # Get the in-sample values of the top node and bottom nodes.
        y_top = y_insample[0]
        y_btm = y_insample[(n_hiers - n_bottom) :]

        # Calculate the disaggregation proportions.
        if self.method == "average_proportions":
            prop = np.mean(y_btm / y_top, 1)
        elif self.method == "proportion_averages":
            prop = np.mean(y_btm, 1) / np.mean(y_top)
        else:
            raise ValueError(f"{self.method} is an unknown disaggregation method.")

        # Instantiate and allocate the "projection" matrix to distribute the
        # disaggregated base forecast of the top node to the bottom nodes.
        P = sparse.csr_matrix(
            (
                prop,
                np.zeros_like(prop, np.uint8),
                np.arange(len(prop) + 1, dtype=np.min_scalar_type(n_bottom)),
            ),
            shape=(n_bottom, n_hiers),
            dtype=np.float64,
        )

        # Instantiate and allocate the "weight" matrix.
        if getattr(self, "intervals_method", False) is None:
            W = None
        else:
            W = sparse.eye(n_hiers, dtype=np.float64, format="csr")

        return P, W

    def fit_predict(
        self,
        S: sparse.csr_matrix,
        y_hat: np.ndarray,
        tags: dict[str, np.ndarray],
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        level: list[int] | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        if self.method == "forecast_proportions":
            # Construct the adjacency matrix.
            A = _construct_adjacency_matrix(S, tags)
            # Avoid a redundant check during middle-out reconciliation.
            if not getattr(self, '_hierarchy_validated', False):
                # Check if the data structure is strictly hierarchical.
                if tags is not None and not _is_strictly_hierarchical(A, tags):
                    raise ValueError(
                        "Top-down reconciliation requires strictly hierarchical structures."
                    )
            # Reconcile point forecasts using sparse forecast proportions
            y_tilde = _reconcile_fcst_proportions_sparse(
                S=S,
                y_hat=y_hat,
                A=A,
                tags=tags,
            )
            res = {"mean": y_tilde}

            # Compute prediction intervals using bootstrap if requested
            if level is not None:
                if y_insample is None or y_hat_insample is None:
                    raise ValueError(
                        "Prediction intervals for `forecast_proportions` require "
                        "`y_insample` and `y_hat_insample`."
                    )
                if intervals_method != "bootstrap":
                    raise ValueError(
                        "Only `bootstrap` intervals_method is implemented for "
                        "`forecast_proportions` with sparse matrices."
                    )
                if num_samples is None:
                    num_samples = 100
                if seed is None:
                    seed = 0
                res["quantiles"] = _reconcile_fcst_proportions_bootstrap(
                    S=S,
                    y_hat=y_hat,
                    tags=tags,
                    y_insample=y_insample,
                    y_hat_insample=y_hat_insample,
                    num_samples=num_samples,
                    seed=seed,
                    level=level,
                    A=A,
                )
            return res
        else:
            # Fit creates the P, W, and sampler attributes.
            self.fit(
                S=S,
                y_hat=y_hat,
                y_insample=y_insample,
                y_hat_insample=y_hat_insample,
                sigmah=sigmah,
                intervals_method=intervals_method,
                num_samples=num_samples,
                seed=seed,
                tags=tags,
            )
            return self._reconcile(
                S=S, P=self.P, y_hat=y_hat, level=level, sampler=self.sampler
            )

    __call__ = fit_predict


class MiddleOut(HReconciler):
    """Middle Out Reconciliation Class.

    This method is only available for **strictly hierarchical structures**. It anchors the base predictions
    in a middle level. The levels above the base predictions use the Bottom-Up approach, while the levels
    below use a Top-Down.

    Args:
        middle_level: Middle level.
        top_down_method: One of `forecast_proportions`, `average_proportions` and `proportion_averages`.

    References:
    - [Hyndman, R.J., & Athanasopoulos, G. (2021). "Forecasting: principles and
    practice, 3rd edition: Chapter 11: Forecasting hierarchical and grouped series".
    OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed on July 2022.](https://otexts.com/fpp3/hierarchical.html)
    """

    is_strictly_hierarchical = True

    def __init__(self, middle_level: str, top_down_method: str):
        if top_down_method not in [
            "forecast_proportions",
            "average_proportions",
            "proportion_averages",
        ]:
            raise ValueError(
                f"Unknown top_down_method `{top_down_method}`. Choose from `forecast_proportions`, `average_proportions`, `proportion_averages`."
            )
        self.middle_level = middle_level
        self.top_down_method = top_down_method
        self.insample = top_down_method in [
            "average_proportions",
            "proportion_averages",
        ]
        self._init_params = {"middle_level": middle_level, "top_down_method": top_down_method}

    def _get_PW_matrices(self, **kwargs):
        raise NotImplementedError("Not implemented")

    def fit(self, **kwargs):
        raise NotImplementedError("Not implemented")

    def predict(self, **kwargs):
        raise NotImplementedError("Not implemented")

    def fit_predict(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        tags: dict[str, np.ndarray],
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        level: list[int] | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
    ):
        """Middle Out Reconciliation Method.

        Args:
            S (np.ndarray): Summing matrix of size (`base`, `bottom`).
            y_hat (np.ndarray): Forecast values of size (`base`, `horizon`).
            tags (dict[str, np.ndarray]): Each key is a level and each value its `S` indices.
            y_insample (np.ndarray): Insample values of size (`base`, `insample_size`). Only used for `forecast_proportions`. Default is None.
            y_hat_insample (np.ndarray): In-sample forecast values of size (`base`, `insample_size`). Only used for `forecast_proportions`. Default is None.
            sigmah (np.ndarray): Estimated standard deviation of the conditional marginal distribution. Default is None.
            level (list[int]): Confidence levels for prediction intervals. Default is None.
            intervals_method (str): Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`. Default is None.
            num_samples (int): Number of samples for probabilistic coherent distribution. Default is None.
            seed (int): Seed for reproducibility. Default is None.

        Returns:
            y_tilde: Reconciliated y_hat using the Middle Out approach.
        """
        if not is_strictly_hierarchical(S, tags):
            raise ValueError(
                "Middle out reconciliation requires strictly hierarchical structures."
            )
        if self.middle_level not in tags.keys():
            raise KeyError(f"{self.middle_level} is not a key in `tags`.")

        levels = dict(sorted(tags.items(), key=lambda x: len(x[1])))
        # Allocate an array to store the reconciled point forecasts.
        y_tilde = np.full_like(y_hat, np.nan)
        y_tilde_quantiles = None
        # Find the nodes that constitute the middle level.
        cut_nodes = levels[self.middle_level]

        # Calculate the cut that separates the middle level from the lower levels.
        cut_idx = max(cut_nodes) + 1

        tags_bu = {}
        for node, idx_node in levels.items():
            tags_bu[node] = idx_node
            if node == self.middle_level:
                break

        # Perform bottom-up reconciliation from the middle level.
        bu = BottomUp().fit_predict(
            S=np.fliplr(np.unique(S[:cut_idx], axis=1)),
            y_hat=y_hat[:cut_idx],
            y_insample=y_insample[:cut_idx] if y_insample is not None else None,
            y_hat_insample=(
                y_hat_insample[:cut_idx] if y_hat_insample is not None else None
            ),
            sigmah=sigmah[:cut_idx] if sigmah is not None else None,
            level=level,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags_bu,
        )
        y_tilde[:cut_idx] = bu["mean"]
        if level is not None:
            if "quantiles" not in bu:
                raise ValueError("Quantiles not found in BottomUp output.")
            y_tilde_quantiles = np.full(
                (y_tilde.shape[0], y_tilde.shape[1], bu["quantiles"].shape[-1]),
                np.nan,
                dtype=np.float64,
            )
            y_tilde_quantiles[:cut_idx] = bu["quantiles"]

        # Set up the reconciler for top-down reconciliation.
        cls_top_down = TopDown(self.top_down_method)
        cls_top_down._hierarchy_validated = True # type: ignore[attr-defined]

        # Perform top-down reconciliation from the middle level.
        for cut_node in cut_nodes:
            # Find the leaf nodes of the subgraph for the cut node.
            leaf_idx = np.flatnonzero(S[cut_node])
            # Find all the nodes in the subgraph for the cut node.
            sub_idx = np.hstack(
                (cut_node, cut_idx + np.flatnonzero(np.any(S[cut_idx:, leaf_idx], 1)))
            )

            # Construct the "tags" argument for the cut node.
            if self.insample and level is None:
                # It is not required for in-sample disaggregation methods.
                sub_tags = None
            else:
                # Disaggregating using forecast proportions requires the "tags" for
                # the subgraph.
                sub_tags = {}
                acc = 0
                for level_, nodes in levels.items():
                    # Find all the nodes in the subgraph for the level.
                    nodes = np.intersect1d(nodes, sub_idx, True)
                    # Get the number of nodes in the level.
                    n = len(nodes)
                    # Exclude any levels above the cut node or empty ones below.
                    if len(nodes) > 0:
                        sub_tags[level_] = np.arange(acc, n + acc)
                        acc += n

            # Perform top-down reconciliation from the cut node.
            td_node = cls_top_down.fit_predict(
                S=S[sub_idx[:, None], leaf_idx],
                y_hat=y_hat[sub_idx],
                y_insample=y_insample[sub_idx] if y_insample is not None else None,
                y_hat_insample=(
                    y_hat_insample[sub_idx] if y_hat_insample is not None else None
                ),
                sigmah=sigmah[sub_idx] if sigmah is not None else None,
                level=level,
                intervals_method=intervals_method,
                num_samples=num_samples,
                seed=seed,
                tags=sub_tags,
            )
            y_tilde[sub_idx] = td_node["mean"]
            if level is not None and y_tilde_quantiles is not None:
                if "quantiles" not in td_node:
                    raise ValueError("Quantiles not found in TopDown output.")
                y_tilde_quantiles[sub_idx] = td_node["quantiles"]

        return {"mean": y_tilde, "quantiles": y_tilde_quantiles}

    __call__ = fit_predict


class MiddleOutSparse(MiddleOut):
    """MiddleOutSparse Reconciliation Class.

    This is an implementation of middle-out reconciliation using the sparse matrix
    approach. It works much more efficiently on data sets with many time series.

    See the parent class for more details.
    """

    # Although this is a sparse method, as we need to a dense representation of the
    # "summing" matrix for the required transformations in the fit_predict method
    # prior to bottom-up and top-down reconciliation, we can avoid a redundant
    # conversion.
    is_sparse_method = False
    is_strictly_hierarchical = True

    def fit_predict(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        tags: dict[str, np.ndarray],
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        level: list[int] | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        # Check if the middle level exists in the level to nodes mapping.
        if self.middle_level not in tags.keys():
            raise KeyError(f"{self.middle_level} is not a key in `tags`.")
        # Check if the data structure is strictly hierarchical.
        if not _is_strictly_hierarchical(
            _construct_adjacency_matrix(sparse.csr_matrix(S), tags), tags
        ):
            raise ValueError(
                "Middle-out reconciliation requires strictly hierarchical structures."
            )

        # Sort the levels by the number of nodes.
        levels = dict(sorted(tags.items(), key=lambda x: len(x[1])))
        # Allocate an array to store the reconciled point forecasts.
        y_tilde = np.full_like(y_hat, np.nan)
        y_tilde_quantiles = None
        # Find the nodes that constitute the middle level.
        cut_nodes = levels[self.middle_level]

        # Calculate the cut that separates the middle level from the lower levels.
        cut_idx = max(cut_nodes) + 1

        tags_bu = {}
        for node, idx_node in levels.items():
            tags_bu[node] = idx_node
            if node == self.middle_level:
                break

        # Perform sparse bottom-up reconciliation from the middle level.
        bu_sparse = BottomUpSparse().fit_predict(
            S=sparse.csr_matrix(np.fliplr(np.unique(S[:cut_idx], axis=1))),
            y_hat=y_hat[:cut_idx],
            y_insample=y_insample[:cut_idx] if y_insample is not None else None,
            y_hat_insample=(
                y_hat_insample[:cut_idx] if y_hat_insample is not None else None
            ),
            sigmah=sigmah[:cut_idx] if sigmah is not None else None,
            level=level,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags_bu,
        )
        y_tilde[:cut_idx] = bu_sparse["mean"]
        if level is not None:
            if "quantiles" not in bu_sparse:
                raise ValueError("Quantiles not found in BottomUpSparse output.")
            y_tilde_quantiles = np.full(
                (y_tilde.shape[0], y_tilde.shape[1], bu_sparse["quantiles"].shape[-1]),
                np.nan,
                dtype=np.float64,
            )
            y_tilde_quantiles[:cut_idx] = bu_sparse["quantiles"]

        # Set up the reconciler for top-down reconciliation.
        cls_top_down = TopDownSparse(self.top_down_method)
        cls_top_down._hierarchy_validated = True # type: ignore[attr-defined]

        # Perform sparse top-down reconciliation from the middle level.
        for cut_node in cut_nodes:
            # Find the leaf nodes of the subgraph for the cut node.
            leaf_idx = np.flatnonzero(S[cut_node])
            # Find all the nodes in the subgraph for the cut node.
            sub_idx = np.hstack(
                (cut_node, cut_idx + np.flatnonzero(np.any(S[cut_idx:, leaf_idx], 1)))
            )

            # Construct the "tags" argument for the cut node.
            if self.insample and level is None:
                # It is not required for in-sample disaggregation methods.
                sub_tags = None
            else:
                # Disaggregating using forecast proportions requires the "tags" for
                # the subgraph.
                sub_tags = {}
                acc = 0
                for level_, nodes in levels.items():
                    # Find all the nodes in the subgraph for the level.
                    nodes = np.intersect1d(nodes, sub_idx, True)
                    # Get the number of nodes in the level.
                    n = len(nodes)
                    # Exclude any levels above the cut node or empty ones below.
                    if len(nodes) > 0:
                        sub_tags[level_] = np.arange(acc, n + acc)
                        acc += n

            # Perform sparse top-down reconciliation from the cut node.
            td_node = cls_top_down.fit_predict(
                S=sparse.csr_matrix(S[sub_idx[:, None], leaf_idx]),
                y_hat=y_hat[sub_idx],
                y_insample=y_insample[sub_idx] if y_insample is not None else None,
                y_hat_insample=(
                    y_hat_insample[sub_idx] if y_hat_insample is not None else None
                ),
                sigmah=sigmah[sub_idx] if sigmah is not None else None,
                level=level,
                intervals_method=intervals_method,
                num_samples=num_samples,
                seed=seed,
                tags=sub_tags,
            )
            y_tilde[sub_idx] = td_node["mean"]
            if level is not None and y_tilde_quantiles is not None:
                if "quantiles" not in td_node:
                    raise ValueError("Quantiles not found in TopDownSparse output.")
                y_tilde_quantiles[sub_idx] = td_node["quantiles"]

        return {"mean": y_tilde, "quantiles": y_tilde_quantiles}

    __call__ = fit_predict


class MinTrace(HReconciler):
    r"""MinTrace Reconciliation Class.

    This reconciliation algorithm proposed by Wickramasuriya et al. depends on a generalized least squares estimator
    and an estimator of the covariance matrix of the coherency errors $\mathbf{W}_{h}$. The Min Trace algorithm
    minimizes the squared errors for the coherent forecasts under an unbiasedness assumption; the solution has a
    closed form.

    ```math
    \mathbf{P}_{\text{MinT}}=\left(\mathbf{S}^{\intercal}\mathbf{W}_{h}\mathbf{S}\right)^{-1}\mathbf{S}^{\intercal}\mathbf{W}^{-1}_{h}
    ```

    Args:
        method (str): One of `ols`, `wls_struct`, `wls_var`, `mint_shrink`, `mint_cov`, `emint`.
        nonnegative (bool): Reconciled forecasts should be nonnegative?
        mint_shr_ridge (float): Ridge numeric protection to MinTrace-shr covariance estimator.
        num_threads (int): Number of threads to use for solving the optimization problems (when nonnegative=True).

    References:
    - [Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization". Journal of the American Statistical Association, 114 , 804-819. doi:10.1080/01621459.2018.1448825.](https://robjhyndman.com/publications/mint/).
    - [Wickramasuriya, S.L., Turlach, B.A. & Hyndman, R.J. (2020). "Optimal non-negative forecast reconciliation". Stat Comput 30, 1167-1182. https://doi.org/10.1007/s11222-020-09930-0](https://robjhyndman.com/publications/nnmint/).
    - [Wickramasuriya, S.L. (2021). Properties of point forecast reconciliation approaches. arXiv:2103.11129](https://arxiv.org/abs/2103.11129).
    - [Wang, X., Hyndman, R.J., & Wickramasuriya, S.L. (2025). Optimal forecast reconciliation with time series selection. European Journal of Operational Research, 323, 455-470.](https://doi.org/10.1016/j.ejor.2024.12.004)
    """

    is_strictly_hierarchical = False

    def __init__(
        self,
        method: str,
        nonnegative: bool = False,
        mint_shr_ridge: float | None = 2e-8,
        num_threads: int = 1,
    ):
        if method not in ["ols", "wls_struct", "wls_var", "mint_cov", "mint_shrink", "emint"]:
            raise ValueError(
                f"Unknown method `{method}`. Choose from `ols`, `wls_struct`, `wls_var`, `mint_cov`, `mint_shrink`, `emint`."
            )
        self.method = method
        self.nonnegative = nonnegative
        self.insample = method in ["wls_var", "mint_cov", "mint_shrink", "emint"]
        if method == "mint_shrink":
            self.mint_shr_ridge = mint_shr_ridge
        self.num_threads = num_threads
        if not self.nonnegative and self.num_threads > 1:
            warnings.warn("`num_threads` is only used when `nonnegative=True`")
        # Store init params for naming (excluding internal flags like insample, num_threads)
        self._init_params = {"method": method, "nonnegative": nonnegative}
        if method == "mint_shrink":
            self._init_params["mint_shr_ridge"] = mint_shr_ridge

    def _get_PW_matrices(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
    ):
        # Handle emint method separately
        if self.method == "emint":
            return self._get_PW_matrices_emint(
                S=S,
                y_hat=y_hat,
                y_insample=y_insample,
                y_hat_insample=y_hat_insample,
            )

        # shape residuals_insample (n_hiers, obs)
        res_methods = ["wls_var", "mint_cov", "mint_shrink"]
        if self.method in res_methods and (
            y_insample is None or y_hat_insample is None
        ):
            raise ValueError(
                f"Check `Y_df`. For method `{self.method}` you need to pass insample predictions and insample values."
            )
        n_hiers, n_bottom = S.shape
        n_aggs = n_hiers - n_bottom
        # Construct J and U.T
        J = np.concatenate(
            (np.zeros((n_bottom, n_aggs), dtype=np.float64), S[n_aggs:]), axis=1
        )
        Ut = np.concatenate((np.eye(n_aggs, dtype=np.float64), -S[:n_aggs]), axis=1)
        if self.method == "ols":
            W = np.eye(n_hiers)
            UtW = Ut
        elif self.method == "wls_struct":
            Wdiag = np.sum(S, axis=1, dtype=np.float64)
            UtW = Ut * Wdiag
            W = np.diag(Wdiag)
        elif (
            self.method in res_methods
            and y_insample is not None
            and y_hat_insample is not None
        ):
            # Residuals with shape (obs, n_hiers)
            residuals = (y_insample - y_hat_insample).T
            n, _ = residuals.shape

            # Protection: against overfitted model
            residuals_sum = np.sum(residuals, axis=0)
            zero_residual_prc = np.abs(residuals_sum) < 1e-4
            zero_residual_prc = np.mean(zero_residual_prc)
            if zero_residual_prc > 0.98:
                raise Exception(
                    f"Insample residuals close to 0, zero_residual_prc={zero_residual_prc}. Check `Y_df`"
                )

            if self.method == "wls_var":
                Wdiag = (
                    np.nansum(residuals**2, axis=0, dtype=np.float64)
                    / residuals.shape[0]
                )
                Wdiag += np.full(n_hiers, 2e-8, dtype=np.float64)
                W = np.diag(Wdiag)
                UtW = Ut * Wdiag
            elif self.method == "mint_cov":
                # Compute nans
                nan_mask = np.isnan(residuals.T)
                if np.any(nan_mask):
                    W = _ma_cov(residuals.T, ~nan_mask)
                else:
                    W = np.cov(residuals.T)

                UtW = Ut @ W
            elif self.method == "mint_shrink":
                # Compute nans
                nan_mask = np.isnan(residuals.T)
                # Compute shrunk empirical covariance
                if np.any(nan_mask):
                    W = _shrunk_covariance_schaferstrimmer_with_nans(
                        residuals.T, ~nan_mask, self.mint_shr_ridge
                    )
                else:
                    W = _shrunk_covariance_schaferstrimmer_no_nans(
                        residuals.T, self.mint_shr_ridge
                    )

                UtW = Ut @ W
        else:
            raise ValueError(f"Unknown reconciliation method {self.method}")

        try:
            # improve stability of linalg.solve
            coef = UtW[:, n_aggs:] @ Ut.T[n_aggs:] + UtW[:, :n_aggs]
            coef[np.abs(coef) < 1e-10] = 0.0
            dep = UtW[:, n_aggs:] @ J.T[n_aggs:]
            dep[np.abs(dep) < 1e-10] = 0.0

            P = J - np.linalg.solve(coef, dep).T @ Ut
        except np.linalg.LinAlgError:
            if self.method == "mint_shrink":
                raise Exception(
                    f"min_trace ({self.method}) is ill-conditioned. Increase the value of parameter 'mint_shr_ridge' or use another reconciliation method."
                )
            else:
                raise Exception(
                    f"min_trace ({self.method}) is ill-conditioned. Please use another reconciliation method."
                )

        return P, W

    def _get_PW_matrices_emint(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray,
        y_hat_insample: np.ndarray,
    ):
        """Compute the projection matrix P and weight matrix W for EMinT.

        Args:
            S: Summing matrix of size (base, bottom).
            y_hat: Forecast values of size (base, horizon).
            y_insample: In-sample values of size (base, insample_size).
            y_hat_insample: In-sample fitted values of size (base, insample_size).

        Returns:
            P: Projection matrix of size (bottom, base).
            W: Weight matrix (identity for EMinT).
        """
        n_hiers, n_bottom = S.shape

        if y_insample is None or y_hat_insample is None:
            raise ValueError(
                "Check `Y_df`. For method `emint` you need to pass insample predictions and insample values."
            )

        # Infer bottom level indices as last n_bottom rows
        idx_bottom = list(range(n_hiers - n_bottom, n_hiers))

        # Remove observations with nan values
        nan_idx = np.isnan(y_hat_insample).any(axis=0)
        # Raise warning if any nan values are found
        if np.any(nan_idx):
            warnings.warn(
                f"Found {np.sum(nan_idx)} out of {y_hat_insample.shape[1]} insample observations with NaN values. These will be ignored when computing the EMinT projection matrix."
            )
        y_insample_clean = y_insample[:, ~nan_idx]
        y_hat_insample_clean = y_hat_insample[:, ~nan_idx]

        # Extract bottom-level observations (use ALL available data)
        # Note: G-matrix computed once for h=1 holds for all h > 1
        # B_h shape: (T, n_bottom)
        B_h = y_insample_clean[idx_bottom, :].T

        # Y_hat_h shape: (T, n_hiers)
        Y_hat_h = y_hat_insample_clean.T

        # Compute P = B_h^T @ Y_hat_h @ (Y_hat_h^T @ Y_hat_h)^(-1)
        # P shape: (n_bottom, n_hiers)
        P = B_h.T @ Y_hat_h @ np.linalg.pinv(Y_hat_h.T @ Y_hat_h)

        # Weight matrix is identity for EMinT
        W = np.eye(n_hiers, dtype=np.float64)

        return P, W

    def fit(
        self,
        S,
        y_hat,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ):
        """MinTrace Fit Method.

        Args:
            S: Summing matrix of size (`base`, `bottom`).
            y_hat: Forecast values of size (`base`, `horizon`).
            y_insample: Insample values of size (`base`, `insample_size`). Only used with "wls_var", "mint_cov", "mint_shrink".
            y_hat_insample: Insample forecast values of size (`base`, `insample_size`). Only used with "wls_var", "mint_cov", "mint_shrink"
            sigmah: Estimated standard deviation of the conditional marginal distribution.
            intervals_method: Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`.
            num_samples: Number of samples for probabilistic coherent distribution.
            seed: Seed for reproducibility.
            tags: Each key is a level and each value its `S` indices.

        Returns:
            self: object, fitted reconciler.
        """
        self.P, self.W = self._get_PW_matrices(
            S=S,
            y_hat=y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
        )
        self.y_hat = y_hat

        if self.nonnegative:
            _, n_bottom = S.shape
            W_inv = np.linalg.pinv(self.W)
            negatives = y_hat < 0
            if negatives.any():
                warnings.warn("Replacing negative forecasts with zero.")
                y_hat = np.copy(y_hat)
                y_hat[negatives] = 0.0

            a = S.T @ W_inv
            P = a @ S
            q = -(a @ y_hat)
            try:
                _ = np.linalg.cholesky(P)
            except np.linalg.LinAlgError:
                raise Exception(
                    f"min_trace ({self.method}) is ill-conditioned. Try setting nonnegative=False or use another reconciliation method."
                )
            G = -np.eye(n_bottom)
            h = np.zeros(n_bottom)
            # the quadratic programming problem
            # returns the forecasts of the bottom series
            if self.num_threads == 1:
                bottom_fcts = np.zeros_like(q)
                for i in range(y_hat.shape[1]):
                    bottom_fcts[:, i] = solve_qp(P=P, q=q[:, i], G=G, h=h, solver="clarabel")
            else:
                futures = []
                with ThreadPoolExecutor(self.num_threads) as executor:
                    for i in range(y_hat.shape[1]):
                        future = executor.submit(
                            solve_qp, P=P, q=q[:, i], G=G, h=h, solver="clarabel"
                        )
                        futures.append(future)
                    bottom_fcts = np.hstack([f.result()[:, None] for f in futures])
            if not np.all(bottom_fcts > -1e-8):
                raise Exception("nonnegative optimization failed")
            # remove negative values close to zero
            bottom_fcts = np.clip(np.float32(bottom_fcts), a_min=0, a_max=None)
            self.y_hat = S @ bottom_fcts  # Hack

            # Overwrite P, W and sampler attributes with BottomUp's
            self.P, self.W = BottomUp()._get_PW_matrices(S=S)

        self.sampler = self._get_sampler(
            S=S,
            P=self.P,
            W=self.W,
            y_hat=self.y_hat,  # self.y_hat contains nonnegative-constrained forecasts when nonnegative=True
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            sigmah=sigmah,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags,
        )
        self.fitted = True
        return self

    def fit_predict(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        level: list[int] | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ):
        """MinTrace Reconciliation Method.

        Args:
            S: Summing matrix of size (`base`, `bottom`).
            y_hat: Forecast values of size (`base`, `horizon`).
            y_insample: Insample values of size (`base`, `insample_size`). Only used by `wls_var`, `mint_cov`, `mint_shrink`
            y_hat_insample: Insample fitted values of size (`base`, `insample_size`). Only used by `wls_var`, `mint_cov`, `mint_shrink`
            sigmah: Estimated standard deviation of the conditional marginal distribution.
            level: float list 0-100, confidence levels for prediction intervals.
            intervals_method: Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`.
            num_samples: Number of samples for probabilistic coherent distribution.
            seed: Seed for reproducibility.
            tags: Each key is a level and each value its `S` indices.

        Returns:
            y_tilde: Reconciliated y_hat using the MinTrace approach.
        """
        if self.nonnegative:
            if (level is not None) and intervals_method in ["bootstrap", "permbu"]:
                raise ValueError(
                    "nonnegative reconciliation is not compatible with bootstrap or permbu forecasts"
                )

        # Fit creates P, W and sampler attributes
        self.fit(
            S=S,
            y_hat=y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            sigmah=sigmah,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags,
        )

        return self._reconcile(
            S=S, P=self.P, y_hat=self.y_hat, level=level, sampler=self.sampler
        )

    __call__ = fit_predict


class MinTraceSparse(MinTrace):
    """MinTraceSparse Reconciliation Class.

    This is the implementation of OLS and WLS estimators using sparse matrices. It is not guaranteed
    to give identical results to the non-sparse version, but works much more efficiently on data sets
    with many time series.

    See the parent class for more details.

    Args:
        method (str): One of `ols`, `wls_struct`, or `wls_var`.
        nonnegative (bool): Return non-negative reconciled forecasts.
        num_threads (int): Number of threads to execute non-negative quadratic programming calls.
        qp (bool): Implement non-negativity constraint with a quadratic programming approach. Setting
        this to True generally gives better results, but at the expense of higher cost to compute.
    """

    is_sparse_method = True
    is_strictly_hierarchical = False

    def __init__(
        self,
        method: str,
        nonnegative: bool = False,
        num_threads: int = 1,
        qp: bool = True,
    ) -> None:
        if method not in ["ols", "wls_struct", "wls_var"]:
            raise ValueError(
                f"Unknown method `{method}`. Choose from `ols`, `wls_struct`, or `wls_var`."
            )
        # Call the parent constructor.
        super().__init__(method, nonnegative, num_threads=num_threads)
        # Assign the attributes specific to the sparse class.
        self.qp = qp
        # Override _init_params to include sparse-specific params
        self._init_params = {"method": method, "nonnegative": nonnegative}

    def _get_PW_matrices(
        self,
        S: np.ndarray | sparse.spmatrix,
        y_hat: np.ndarray,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
    ):
        # shape residuals_insample (n_hiers, obs)
        res_methods = ["wls_var", "mint_cov", "mint_shrink"]

        S = sparse.csr_matrix(S)

        if self.method in res_methods and (
            y_insample is None or y_hat_insample is None
        ):
            raise ValueError(
                f"Check `Y_df`. For method `{self.method}` you need to pass insample predictions and insample values."
            )
        n_hiers, n_bottom = S.shape

        if self.method == "ols":
            W_diag = np.ones(n_hiers)
        elif self.method == "wls_struct":
            W_diag = S @ np.ones((n_bottom,))
        elif (
            self.method == "wls_var"
            and y_insample is not None
            and y_hat_insample is not None
        ):
            # Residuals with shape (obs, n_hiers)
            residuals = (y_insample - y_hat_insample).T
            n, _ = residuals.shape

            # Protection: against overfitted model
            residuals_sum = np.sum(residuals, axis=0)
            zero_residual_prc = np.abs(residuals_sum) < 1e-4
            zero_residual_prc = np.mean(zero_residual_prc)
            if zero_residual_prc > 0.98:
                raise Exception(
                    f"Insample residuals close to 0, zero_residual_prc={zero_residual_prc}. Check `Y_df`"
                )

            # Protection: cases where data is unavailable/nan
            # makoren: this masking stuff causes more harm than good, I found the results in the presence
            # of nan-s can often be rubbish, I'd argue it's better to fail than give rubbish results, here
            # the code is simply failing if it encounters nan in the variance vector.
            # masked_res = np.ma.array(residuals, mask=np.isnan(residuals))
            # covm = np.ma.cov(masked_res, rowvar=False, allow_masked=True).data

            W_diag = np.nanvar(residuals, axis=0, ddof=1)
        else:
            raise ValueError(f"Unknown reconciliation method {self.method}")

        if any(W_diag < 1e-8):
            raise Exception(
                f"min_trace ({self.method}) needs covariance matrix to be positive definite."
            )

        if any(np.isnan(W_diag)):
            raise Exception(
                f"min_trace ({self.method}) needs covariance matrix to be positive definite (not nan)."
            )

        M = sparse.spdiags(np.reciprocal(W_diag), 0, W_diag.size, W_diag.size)
        R = sparse.csr_matrix(S.T @ M)

        # The implementation of P acting on a vector:
        def get_P_action(y):
            b = R @ y

            A = sparse.linalg.LinearOperator(
                (b.size, b.size), matvec=lambda v: R @ (S @ v)
            )

            x_tilde, exit_code = sparse.linalg.bicgstab(A, b, atol=1e-5)

            return x_tilde

        P = sparse.linalg.LinearOperator(
            (S.shape[1], y_hat.shape[0]), matvec=get_P_action
        )
        W = sparse.spdiags(W_diag, 0, W_diag.size, W_diag.size)

        return P, W

    def fit(
        self,
        S: sparse.csr_matrix,
        y_hat: np.ndarray,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ) -> "MinTraceSparse":
        """MinTraceSparse Fit Method.

        Args:
            S: Summing matrix of size (`base`, `bottom`).
            y_hat: Forecast values of size (`base`, `horizon`).
            y_insample: Insample values of size (`base`, `insample_size`). Only used with "wls_var".
            y_hat_insample: Insample forecast values of size (`base`, `insample_size`). Only used with "wls_var"
            sigmah: Estimated standard deviation of the conditional marginal distribution.
            intervals_method: Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`.
            num_samples: Number of samples for probabilistic coherent distribution.
            seed: Seed for reproducibility.
            tags: Each key is a level and each value its `S` indices.

        Returns:
            self: object, fitted reconciler.
        """
        if self.nonnegative:
            # Clip the base forecasts to align them with their use in practice.
            self.y_hat = np.clip(y_hat, 0, None)
            # Get the number of nodes, leaf nodes, and parent nodes.
            n, n_b = S.shape
            n_a = n - n_b
            # Find the optimal non-negative forecasts.
            if self.qp:
                # Get the diagonal weight matrix, i.e., precision matrix, for
                # the problem.
                if self.method == "ols":
                    W = sparse.eye(n, format="csc")
                elif self.method == "wls_struct":
                    W = sparse.csc_matrix(
                        (
                            (1.0 / S.sum(axis=1)).A1,
                            np.arange(n, dtype=np.min_scalar_type(n - 1)),
                            np.arange(n + 1, dtype=np.min_scalar_type(n)),
                        )
                    )
                elif self.method == "wls_var":
                    # Check that we have the in-sample values.
                    if y_insample is None or y_hat_insample is None:
                        raise ValueError(
                            "`y_insample` and `y_hat_insample` are required to calculate residuals."
                        )
                    # Add a small jitter to the variance to improve the condition
                    # of the variance matrix.
                    W = sparse.csc_matrix(
                        (
                            1.0
                            / (
                                np.nanvar(y_insample - y_hat_insample, 1, ddof=1) + 2e-8
                            ),
                            np.arange(n, dtype=np.min_scalar_type(n - 1)),
                            np.arange(n + 1, dtype=np.min_scalar_type(n)),
                        )
                    )
                # Get the linear constraints matrix by vertically stacking the
                # (n_a x n) constraint matrix in the zero-constrained
                # represenation, which has the set of all reconciled forecasts
                # in its null space, and a horizontally stacked (n_b x n_a)
                # zero matrix and a negated (n_b x n_b) identity matrix.
                A = sparse.vstack(
                    (
                        sparse.hstack(
                            (sparse.eye(n_a, format="csc"), -S[:n_a, :].tocsc())
                        ),
                        -sparse.eye(n_b, n, n_a, format="csc"),
                    )
                )
                # Get the linear constraints vector.
                b = np.zeros(n)
                # Get the composition of convex cones to solve the problem.
                cones = [clarabel.ZeroConeT(n_a), clarabel.NonnegativeConeT(n_b)]
                # Set up the settings for the solver.
                settings = clarabel.DefaultSettings()
                settings.verbose = False

                def solve_clarabel(
                    y: np.ndarray,
                    P: sparse.csc_matrix,
                    A: sparse.csc_matrix,
                    b: np.ndarray,
                    cones: list,
                    settings: clarabel.DefaultSettings,
                    n_b: int,
                ) -> tuple[bool, np.ndarray | None]:
                    # Get the linear coefficients, i.e., the cost vector.
                    q = P @ -y
                    # Set up the Clarabel solver.
                    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
                    # Solve the problem.
                    solution = solver.solve()
                    # Resolve the solver exit status.
                    if status := solution.status == clarabel.SolverStatus.Solved:
                        # Return the slice of the primal solution that
                        # represents the optimal non-negative reconciled
                        # bottom level forecasts.
                        return status, solution.x[-n_b:]
                    else:
                        # As the solver failed, discard the empty primal
                        # solution.
                        return status, None

                with ThreadPoolExecutor(self.num_threads) as executor:
                    # Dispatch the jobs.
                    futures = [
                        executor.submit(
                            solve_clarabel, y, W, A, b, cones, settings, n_b
                        )
                        for y in self.y_hat.transpose()
                    ]
                    # Yield the futures as they complete.
                    for future in as_completed(futures):
                        # Return the exit status of the solver and the primal
                        # solution.
                        status, x = future.result()
                        # Check that the problem is successfully solved and the
                        # primal solution is within tolerance.
                        if not (status and np.min(x) > -1e-8):
                            raise Exception("Non-negative optimisation failed.")

                # Extract the optimal non-negative reconciled bottom level
                # forecasts.
                x = np.vstack([future.result()[1] for future in futures]).transpose()
                # Clip the negative forecasts within tolerance.
                x = np.clip(x, 0, None)
                # Aggregate the clipped bottom level forecasts and overwrite
                # the base forecasts with the solution.
                self.y_hat = S @ x
                # Overwrite the attributes for the P and W matrices with those
                # for bottom-up reconciliation to force projection onto the
                # non-negative coherent subspace.
                self.P, self.W = BottomUpSparse()._get_PW_matrices(S=S)
            else:
                # Get the reconciliation matrices.
                self.P, self.W = self._get_PW_matrices(
                    S=S,
                    y_hat=self.y_hat,
                    y_insample=y_insample,
                    y_hat_insample=y_hat_insample,
                )
                # Although it is now sufficient to ensure that all of the
                # entries in P are positive, as it is implemented as a linear
                # operator for the iterative method to solve the sparse linear
                # system, we need to reconcile to find if any of the coherent
                # bottom level point forecasts are negative.
                y_tilde = self._reconcile(
                    S=S, P=self.P, y_hat=self.y_hat, level=None, sampler=None
                )["mean"][-n_b:, :]
                # Find if any of the forecasts are negative.
                if np.any(y_tilde < 0):
                    # Clip the negative forecasts.
                    y_tilde = np.clip(y_tilde, 0, None)
                    # Force non-negative coherence by overwriting the base
                    # forecasts with the aggregated, clipped bottom level
                    # forecasts.
                    self.y_hat = S @ y_tilde
                    # Overwrite the attributes for the P and W matrices with
                    # those for bottom-up reconciliation to force projection
                    # onto the non-negative coherent subspace.
                    self.P, self.W = BottomUpSparse()._get_PW_matrices(S=S)
        else:
            # Get the reconciliation matrices.
            self.y_hat = y_hat
            self.P, self.W = self._get_PW_matrices(
                S=S,
                y_hat=self.y_hat,
                y_insample=y_insample,
                y_hat_insample=y_hat_insample,
            )

        # Get the sampler for probabilistic reconciliation.
        self.sampler = self._get_sampler(
            S=S,
            P=self.P,
            W=self.W,
            y_hat=self.y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            sigmah=sigmah,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags,
        )
        # Set the instance as fitted.
        self.fitted = True
        return self


class OptimalCombination(MinTrace):
    r"""Optimal Combination Reconciliation Class.

    This reconciliation algorithm was proposed by Hyndman et al. 2011, the method uses generalized least squares
    estimator using the coherency errors covariance matrix. Consider the covariance of the base forecast
    $\\textrm{Var}(\epsilon_{h}) = \Sigma_{h}$, the $\mathbf{P}$ matrix of this method is defined by:
    $$ \mathbf{P} = \\left(\mathbf{S}^{\intercal}\Sigma_{h}^{\dagger}\mathbf{S}\\right)^{-1}\mathbf{S}^{\intercal}\Sigma^{\dagger}_{h}$$
    where $\Sigma_{h}^{\dagger}$ denotes the variance pseudo-inverse. The method was later proven equivalent to
    `MinTrace` variants.

    Args:
        method: str, allowed optimal combination methods: 'ols', 'wls_struct'.
        nonnegative: bool, reconciled forecasts should be nonnegative?

    References:
        - [Rob J. Hyndman, Roman A. Ahmed, George Athanasopoulos, Han Lin Shang (2010). "Optimal Combination Forecasts for Hierarchical Time Series".](https://robjhyndman.com/papers/Hierarchical6.pdf).
        - [Shanika L. Wickramasuriya, George Athanasopoulos and Rob J. Hyndman (2010). "Optimal Combination Forecasts for Hierarchical Time Series".](https://robjhyndman.com/papers/MinT.pdf).
        - [Wickramasuriya, S.L., Turlach, B.A. & Hyndman, R.J. (2020). "Optimal non-negative forecast reconciliation". Stat Comput 30, 1167-1182. https://doi.org/10.1007/s11222-020-09930-0](https://robjhyndman.com/publications/nnmint/).
    """

    is_strictly_hierarchical = False

    def __init__(self, method: str, nonnegative: bool = False, num_threads: int = 1):
        if method not in ["ols", "wls_struct"]:
            raise ValueError(
                f"Unknown method `{method}`. Choose from `ols`, `wls_struct`."
            )
        warnings.warn(
            "OptimalCombination is deprecated and will be removed in a future version. "
            "Use MinTrace(method='ols') or MinTrace(method='wls_struct') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            method=method, nonnegative=nonnegative, num_threads=num_threads
        )
        self.insample = False
        # Override _init_params for OptimalCombination
        self._init_params = {"method": method, "nonnegative": nonnegative}


class ERM(HReconciler):
    r"""Empirical Risk Minimization Reconciliation Class.

    The Empirical Risk Minimization reconciliation strategy relaxes the unbiasedness assumptions from
    previous reconciliation methods like MinT and optimizes square errors between the reconciled predictions
    and the validation data to obtain an optimal reconciliation matrix P.

    The exact solution for $\mathbf{P}$ (`method='closed'`) follows the expression:
    $$\mathbf{P}^{*} = \\left(\mathbf{S}^{\intercal}\mathbf{S}\\right)^{-1}\mathbf{Y}^{\intercal}\hat{\mathbf{Y}}\\left(\hat{\mathbf{Y}}\hat{\mathbf{Y}}\\right)^{-1}$$

    The alternative Lasso regularized $\mathbf{P}$ solution (`method='reg_bu'`) is useful when the observations
    of validation data is limited or the exact solution has low numerical stability.
    $$\mathbf{P}^{*} = \\text{argmin}_{\mathbf{P}} ||\mathbf{Y}-\mathbf{S} \mathbf{P} \hat{Y} ||^{2}_{2} + \lambda ||\mathbf{P}-\mathbf{P}_{\\text{BU}}||_{1}$$

    Args:
        method: str, one of `closed`, `reg` and `reg_bu`.
        lambda_reg: float, l1 regularizer for `reg` and `reg_bu`.

    References:
        - [Ben Taieb, S., & Koo, B. (2019). Regularized regression for hierarchical forecasting without unbiasedness conditions. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining KDD '19 (p. 1337-1347). New York, NY, USA: Association for Computing Machinery.](https://doi.org/10.1145/3292500.3330976).
    """

    is_strictly_hierarchical = False

    def __init__(self, method: str, lambda_reg: float = 1e-2):
        if method not in ["closed", "reg", "reg_bu"]:
            raise ValueError(
                f"Unknown method `{method}`. Choose from `closed`, `reg`, `reg_bu`."
            )
        self.method = method
        self.lambda_reg = lambda_reg
        self.insample = True
        self._init_params = {"method": method, "lambda_reg": lambda_reg}

    def _get_PW_matrices(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray,
        y_hat_insample: np.ndarray,
    ):
        n_hiers, n_bottom = S.shape
        # y_hat_insample shape (n_hiers, obs)
        if y_insample is None or y_hat_insample is None:
            raise ValueError(
                "Check `Y_df`. For method `ERM` you need to pass insample predictions and insample values."
            )

        # Infer bottom level indices as last n_bottom rows
        idx_bottom = list(range(n_hiers - n_bottom, n_hiers))

        # remove obs with nan values
        nan_idx = np.isnan(y_hat_insample).any(axis=0)
        # Raise warning if any nan values are found
        if np.any(nan_idx):
            warnings.warn(
                f"Found {np.sum(nan_idx)} out of {y_hat_insample.shape[1]} insample observations with NaN values. These will be ignored when computing the EMinT projection matrix."
            )
        y_insample = y_insample[:, ~nan_idx]
        y_hat_insample = y_hat_insample[:, ~nan_idx]
        # only using h validation steps to avoid
        # computational burden
        h = min(y_hat.shape[1], y_hat_insample.shape[1])
        y_hat_insample = y_hat_insample[:, -h:]  # shape (h, n_hiers)
        y_insample = y_insample[:, -h:]
        if self.method == "closed":
            B = np.linalg.inv(S.T @ S) @ S.T @ y_insample
            B = B.T
            P = np.linalg.pinv(y_hat_insample.T) @ B
            P = P.T
        elif self.method == "reg":
            X = np.kron(S, y_hat_insample.T)
            z = y_insample.reshape(-1)

            if self.lambda_reg is None:
                lambda_reg = np.max(np.abs(X.T.dot(z)))
            else:
                lambda_reg = self.lambda_reg

            beta = _lasso(X, z, lambda_reg, max_iters=1000, tol=1e-4)
            P = beta.reshape(S.shape).T
        elif self.method == "reg_bu":
            X = np.kron(S, y_hat_insample.T)
            Pbu = np.zeros_like(S)
            Pbu[idx_bottom] = S[idx_bottom]
            z = y_insample.reshape(-1) - X @ Pbu.reshape(-1)

            if self.lambda_reg is None:
                lambda_reg = np.max(np.abs(X.T.dot(z)))
            else:
                lambda_reg = self.lambda_reg

            beta = _lasso(X, z, lambda_reg, max_iters=1000, tol=1e-4)
            P = beta + Pbu.reshape(-1)
            P = P.reshape(S.shape).T
        else:
            raise ValueError(f"Unknown reconciliation method {self.method}")

        W = np.eye(n_hiers, dtype=np.float64)

        return P, W

    def fit(
        self,
        S,
        y_hat,
        y_insample,
        y_hat_insample,
        sigmah: np.ndarray | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ):
        """ERM Fit Method.

        Args:
            S: Summing matrix of size (`base`, `bottom`).
            y_hat: Forecast values of size (`base`, `horizon`).
            y_insample: Train values of size (`base`, `insample_size`).
            y_hat_insample: Insample train predictions of size (`base`, `insample_size`).
            sigmah: Estimated standard deviation of the conditional marginal distribution.
            intervals_method: Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`.
            num_samples: Number of samples for probabilistic coherent distribution.
            seed: Seed for reproducibility.
            tags: Each key is a level and each value its `S` indices.

        Returns:
            self: object, fitted reconciler.
        """
        self.P, self.W = self._get_PW_matrices(
            S=S,
            y_hat=y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
        )
        self.sampler = self._get_sampler(
            S=S,
            P=self.P,
            W=self.W,
            y_hat=y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            sigmah=sigmah,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags,
        )
        self.fitted = True
        return self

    def fit_predict(
        self,
        S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
        sigmah: np.ndarray | None = None,
        level: list[int] | None = None,
        intervals_method: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        tags: dict[str, np.ndarray] | None = None,
    ):
        """ERM Reconciliation Method.

        Args:
            S: Summing matrix of size (`base`, `bottom`).
            y_hat: Forecast values of size (`base`, `horizon`).
            y_insample: Train values of size (`base`, `insample_size`).
            y_hat_insample: Insample train predictions of size (`base`, `insample_size`).
            sigmah: Estimated standard deviation of the conditional marginal distribution.
            level: float list 0-100, confidence levels for prediction intervals.
            intervals_method: Sampler for prediction intervals, one of `normality`, `bootstrap`, `permbu`, `conformal`.
            num_samples: Number of samples for probabilistic coherent distribution.
            seed: Seed for reproducibility.
            tags: Each key is a level and each value its `S` indices.

        Returns:
            y_tilde: Reconciliated y_hat using the ERM approach.
        """
        # Fit creates P, W and sampler attributes
        self.fit(
            S=S,
            y_hat=y_hat,
            y_insample=y_insample,
            y_hat_insample=y_hat_insample,
            sigmah=sigmah,
            intervals_method=intervals_method,
            num_samples=num_samples,
            seed=seed,
            tags=tags,
        )

        return self._reconcile(
            S=S, P=self.P, y_hat=y_hat, level=level, sampler=self.sampler
        )

    __call__ = fit_predict
