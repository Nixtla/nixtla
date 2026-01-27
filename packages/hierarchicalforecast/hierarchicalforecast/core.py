__all__ = ['HierarchicalReconciliation']


import re
import reprlib
import time
from inspect import signature

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import Frame, FrameT
from scipy import sparse
from scipy.stats import norm

from .methods import HReconciler
from .utils import (
    _construct_adjacency_matrix,
    _is_strictly_hierarchical,
    is_strictly_hierarchical,
)


def _compute_coherence_residual(
    y: np.ndarray,
    S: np.ndarray,
    idx_bottom: np.ndarray,
) -> np.ndarray:
    """Compute coherence residual: y - S @ y_bottom.

    The coherence residual measures how much the forecasts violate
    the hierarchical aggregation constraint. For coherent forecasts,
    this should be zero (or near-zero due to numerical precision).

    Args:
        y: Forecasts array of shape (n_series,) or (n_series, horizon).
        S: Summing matrix of shape (n_series, n_bottom).
        idx_bottom: Indices of bottom-level series.

    Returns:
        Coherence residual array with same shape as y.
    """
    y_bottom = y[idx_bottom]
    y_implied = S @ y_bottom
    return y - y_implied


def _compute_diagnostics_for_level(
    y_before: np.ndarray,
    y_after: np.ndarray,
    residual_before: np.ndarray,
    residual_after: np.ndarray,
    level_indices: np.ndarray,
) -> dict[str, float]:
    """Compute all diagnostic metrics for a single hierarchical level.

    Args:
        y_before: Base forecasts array of shape (n_series, horizon).
        y_after: Reconciled forecasts array of shape (n_series, horizon).
        residual_before: Coherence residuals before reconciliation.
        residual_after: Coherence residuals after reconciliation.
        level_indices: Indices of series belonging to this level.

    Returns:
        Dictionary of metric names to values.
    """
    # Extract level data
    y_before_level = y_before[level_indices].flatten()
    y_after_level = y_after[level_indices].flatten()
    residual_before_level = residual_before[level_indices].flatten()
    residual_after_level = residual_after[level_indices].flatten()

    # Adjustments
    adjustment = y_after_level - y_before_level

    metrics = {
        "coherence_residual_mae_before": np.mean(np.abs(residual_before_level), dtype=np.float64),
        "coherence_residual_rmse_before": np.sqrt(np.mean(residual_before_level**2), dtype=np.float64),
        "coherence_residual_mae_after": np.mean(np.abs(residual_after_level), dtype=np.float64),
        "coherence_residual_rmse_after": np.sqrt(np.mean(residual_after_level**2), dtype=np.float64),
        "adjustment_mae": np.mean(np.abs(adjustment), dtype=np.float64),
        "adjustment_rmse": np.sqrt(np.mean(adjustment**2), dtype=np.float64),
        "adjustment_max": np.max(np.abs(adjustment, dtype=np.float64)),
        "adjustment_mean": np.mean(adjustment, dtype=np.float64),
        "negative_count_before": np.sum(y_before_level < 0, dtype=np.int64),
        "negative_count_after": np.sum(y_after_level < 0, dtype=np.int64),
        "negative_introduced": np.sum((y_before_level >= 0) & (y_after_level < 0), dtype=np.int64),
        "negative_removed": np.sum((y_before_level < 0) & (y_after_level >= 0), dtype=np.int64),
    }
    return metrics


def _aggregate_diagnostics(
    diagnostics_per_model: dict[str, dict[str, dict[str, float]]],
    backend,
) -> Frame:
    """Aggregate per-model, per-level diagnostics into a DataFrame.

    Args:
        diagnostics_per_model: Nested dict {model_name: {level: {metric: value}}}.
        backend: Narwhals backend for DataFrame creation.

    Returns:
        DataFrame with columns [level, metric, model1, model2, ...].
    """
    models = list(diagnostics_per_model.keys())
    first_model_data = next(iter(diagnostics_per_model.values()))
    levels = list(first_model_data.keys())

    # Build output data - iterate through each level's metrics since
    # different levels may have different metrics (e.g., is_coherent only for Overall)
    data: dict[str, list] = {"level": [], "metric": []}
    for model in models:
        data[model] = []

    for level in levels:
        level_metrics = list(first_model_data[level].keys())
        for metric in level_metrics:
            data["level"].append(level)
            data["metric"].append(metric)
            for model in models:
                data[model].append(diagnostics_per_model[model][level][metric])

    df = nw.from_dict(data, backend=backend)
    return df.to_native()


def _build_fn_name(fn) -> str:
    fn_name = type(fn).__name__
    # Use _init_params if available, otherwise fall back to __dict__ for backwards compatibility
    func_params = getattr(fn, "_init_params", None)
    if func_params is None:
        func_params = fn.__dict__

    # Take default parameter out of names
    args_to_remove = ["insample", "num_threads"]
    if not func_params.get("nonnegative", False):
        args_to_remove.append("nonnegative")

    if fn_name == "MinTrace" and func_params.get("method") == "mint_shrink":
        if func_params.get("mint_shr_ridge") == 2e-8:
            args_to_remove.append("mint_shr_ridge")

    func_params = [
        f"{name}-{value}"
        for name, value in func_params.items()
        if name not in args_to_remove
    ]
    if func_params:
        fn_name += "_" + "_".join(func_params)
    return fn_name


def _reverse_engineer_sigmah(
    Y_hat_df: Frame,
    y_hat: np.ndarray,
    model_name: str,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    num_samples: int = 200,
) -> np.ndarray:
    r"""Reverse engineer sigma_h from prediction intervals.

    This function assumes that the model creates prediction intervals
    under a normality with the following the Equation:
    $\hat{y}_{t+h} + c \hat{sigma}_{h}$

    In the future, we might deprecate this function in favor of a
    direct usage of an estimated $\hat{sigma}_{h}$
    """

    drop_cols = [time_col]
    if target_col in Y_hat_df.columns:
        drop_cols.append(target_col)
    if model_name + "-median" in Y_hat_df.columns:
        drop_cols.append(model_name + "-median")
    model_names = [c for c in Y_hat_df.columns if c not in drop_cols]
    pi_model_names = [name for name in model_names if ("-lo" in name or "-hi" in name)]
    pi_model_name = [pi_name for pi_name in pi_model_names if model_name in pi_name]
    pi = len(pi_model_name) > 0

    n_series = Y_hat_df[id_col].n_unique()

    if not pi:
        raise ValueError(
            f"Please include `{model_name}` prediction intervals in `Y_hat_df`"
        )

    pi_col = pi_model_name[0]
    sign = -1 if "lo" in pi_col else 1
    level_cols = re.findall(r"[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+", pi_col)
    level_col = float(level_cols[-1])
    z = norm.ppf(0.5 + level_col / num_samples)
    sigmah = Y_hat_df[pi_col].to_numpy().reshape(n_series, -1)
    sigmah = sign * (sigmah - y_hat) / z

    return sigmah


class HierarchicalReconciliation:
    r"""Hierarchical Reconciliation Class.

    The `core.HierarchicalReconciliation` class allows you to efficiently fit multiple
    HierarchicaForecast methods for a collection of time series and base predictions stored in
    pandas DataFrames. The `Y_df` dataframe identifies series and datestamps with the unique_id and ds columns while the
    y column denotes the target time series variable. The `Y_h` dataframe stores the base predictions,
    example ([AutoARIMA](../../../statsforecast/docs/models/AutoARIMA#autoarima-model), [ETS](../../../statsforecast/docs/models/AutoETS#autoets-model), etc.).

    Args:
        - reconcilers (list[HReconciler]): A list of instantiated classes of the [reconciliation methods](./methods) module.

    References:
        - [Rob J. Hyndman and George Athanasopoulos (2018). "Forecasting principles and practice, Hierarchical and Grouped Series"](https://otexts.com/fpp3/hierarchical.html).

    """

    def __init__(self, reconcilers: list[HReconciler]):
        self.reconcilers = reconcilers

    def _prepare_fit(
        self,
        Y_hat_nw: Frame,
        S_nw: Frame,
        Y_nw: Frame | None,
        tags: dict[str, np.ndarray],
        level: list[int] | None = None,
        intervals_method: str = "normality",
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        id_time_col: str = "temporal_id",
        temporal: bool = False,
    ) -> tuple[FrameT, FrameT, FrameT, list[str], str]:
        """Performs preliminary wrangling and protections."""
        Y_hat_nw_cols = Y_hat_nw.columns
        S_nw_cols = S_nw.columns

        # Check if Y_hat_df has the necessary columns for temporal
        if temporal:
            # We don't support insample methods, so Y_df must be None
            if Y_nw is not None:
                raise NotImplementedError(
                    "Temporal reconciliation requires `Y_df` to be None."
                )
            # If Y_nw is None, we need to check if the reconcilers are not insample methods
            for reconciler in self.reconcilers:
                if reconciler.insample:
                    reconciler_name = _build_fn_name(reconciler)
                    raise NotImplementedError(
                        f"Temporal reconciliation is not supported for `{reconciler_name}`."
                    )
            # Hence we also don't support bootstrap or permbu (rely on insample values)
            if intervals_method in ["bootstrap", "permbu"]:
                raise NotImplementedError(
                    f"Temporal reconciliation is not supported for intervals_method=`{intervals_method}`."
                )

            # Validate horizon against max row sum in temporal spec
            S_cols = [c for c in S_nw.columns if c != id_time_col]
            max_agg_factor = S_nw.with_columns(
                sum=nw.sum_horizontal(S_cols)
            )["sum"].max()

           # Compute horizon: number of unique timestamps per unique_id
            # This represents the number of bottom-level periods available for reconciliation
            horizon = Y_hat_nw.group_by(id_col).agg(nw.col(time_col).n_unique().alias("count"))["count"].max()
            if horizon < max_agg_factor:
                raise ValueError(
                    f"The forecast horizon ({horizon}) must be greater than or equal to "
                    f"the maximum aggregation factor in temporal_spec ({max_agg_factor}). "
                    f"Ensure you have at least {max_agg_factor} forecast periods."
                )

            missing_cols_temporal = set([id_col, time_col, id_time_col]) - set(
                Y_hat_nw_cols
            )
            if missing_cols_temporal:
                raise ValueError(
                    f"Check `Y_hat_df` columns, for temporal reconciliation {reprlib.repr(missing_cols_temporal)} must be in `Y_hat_df` columns."
                )
            if id_time_col not in S_nw_cols:
                raise ValueError(
                    f"Check `S_df` columns, {reprlib.repr(id_time_col)} must be in `S_df` columns."
                )
            id_cols = [id_col, time_col, target_col, id_time_col]
            id_col = id_time_col
        else:
            id_cols = [id_col, time_col, target_col]
            if id_col not in S_nw_cols:
                raise ValueError(
                    f"Check `S_df` columns, {reprlib.repr(id_col)} must be in `S_df` columns."
                )

        # Check if Y_hat_df has the right shape
        if len(Y_hat_nw.group_by(id_col).agg(nw.len()).unique(subset="len")) != 1:
            raise ValueError(
                "Check `Y_hat_df`, there are missing timestamps. All series should have the same number of predictions."
            )

        # -------------------------------- Match Y_hat/Y/S index order --------------------------------#
        # TODO: This is now a bit slow as we always sort.
        S_nw = S_nw.with_columns(**{f"{id_col}_id": np.arange(len(S_nw))})

        Y_hat_nw = Y_hat_nw.join(S_nw[[id_col, f"{id_col}_id"]], on=id_col, how="left")
        Y_hat_nw = Y_hat_nw.sort(by=[f"{id_col}_id", time_col])
        Y_hat_nw = Y_hat_nw[Y_hat_nw_cols]

        if Y_nw is not None:
            Y_nw_cols = Y_nw.columns
            Y_nw = Y_nw.join(S_nw[[id_col, f"{id_col}_id"]], on=id_col, how="left")
            Y_nw = Y_nw.sort(by=[f"{id_col}_id", time_col])
            Y_nw = Y_nw[Y_nw_cols]

        S_nw = S_nw[S_nw_cols]

        # ----------------------------------- Check Input's Validity ----------------------------------#

        # Check input's validity
        if intervals_method not in ["normality", "bootstrap", "permbu", "conformal"]:
            raise ValueError(f"Unknown interval method: {intervals_method}")

        # Check absence of Y_nw for insample reconcilers
        if Y_nw is None:
            for reconciler in self.reconcilers:
                if reconciler.insample:
                    reconciler_name = _build_fn_name(reconciler)
                    raise ValueError(
                        f"You need to provide `Y_df` for reconciler {reconciler_name}"
                    )
            if intervals_method in ["bootstrap", "permbu", "conformal"]:
                raise ValueError(
                    f"You need to provide `Y_df` when using intervals_method=`{intervals_method}`."
                )

        # Protect level list
        if level is not None:
            level_outside_domain = not all(0 <= x < 100 for x in level)
            if level_outside_domain and (intervals_method in ["normality", "permbu"]):
                raise ValueError(
                    "Level must be a list containing floating values in the interval [0, 100)."
                )

        # Declare output names
        model_names = [col for col in Y_hat_nw.columns if col not in id_cols]

        # Ensure numeric columns
        for model in model_names:
            if not Y_hat_nw.schema[model].is_numeric():
                raise ValueError(
                    f"Column `{model}` in `Y_hat_df` contains non-numeric values. Make sure no column in `Y_hat_df` contains non-numeric values."
                )
            if Y_hat_nw[model].is_null().any():
                raise ValueError(
                    f"Column `{model}` in `Y_hat_df` contains null values. Make sure no column in `Y_hat_df` contains null values."
                )

        # TODO: Complete y_hat_insample protection
        model_names = [
            name
            for name in model_names
            if not ("-lo" in name or "-hi" in name or "-median" in name)
        ]
        if intervals_method in ["bootstrap", "permbu"] and Y_nw is not None:
            missing_models = set(model_names) - set(Y_nw.columns)
            if missing_models:
                raise ValueError(
                    f"Check `Y_df` columns, {reprlib.repr(missing_models)} must be in `Y_df` columns."
                )

        # Assert S is an identity matrix at the bottom
        S_nw_cols.remove(id_col)
        # Check if S_nw is backed by a sparse pandas DataFrame (check value columns only)
        S_bottom_nw = S_nw[S_nw_cols][-len(S_nw_cols) :]
        S_bottom = S_bottom_nw.to_native()
        is_sparse_df = hasattr(S_bottom, "sparse") and hasattr(S_bottom, "dtypes") and all(
            str(dtype).startswith("Sparse") for dtype in S_bottom.dtypes
        )
        if is_sparse_df:
            # Sparse-aware identity check: verify diagonal is 1 and off-diagonal is 0
            # by checking nnz equals n and all non-zero values are 1
            S_bottom_coo = S_bottom.sparse.to_coo()
            n = S_bottom_coo.shape[0]
            is_identity = (
                S_bottom_coo.shape[0] == S_bottom_coo.shape[1]
                and S_bottom_coo.nnz == n
                and np.allclose(S_bottom_coo.data, 1.0)
                and np.array_equal(S_bottom_coo.row, S_bottom_coo.col)  # diagonal only
            )
            if not is_identity:
                raise ValueError(
                    f"The bottom {n}x{n} part of S must be an identity matrix."
                )
        else:
            # Dense path (original)
            if not np.allclose(
                S_bottom_nw, np.eye(len(S_nw_cols))
            ):
                raise ValueError(
                    f"The bottom {S_nw.shape[1]}x{S_nw.shape[1]} part of S must be an identity matrix."
                )

        # Check Y_hat_df\S_df series difference
        # TODO: this logic should be method specific
        S_diff = set(S_nw[id_col]) - set(Y_hat_nw[id_col])
        Y_hat_diff = set(Y_hat_nw[id_col]) - set(S_nw[id_col])
        if S_diff:
            raise ValueError(
                f"There are unique_ids in S_df that are not in Y_hat_df: {reprlib.repr(S_diff)}"
            )
        if Y_hat_diff:
            raise ValueError(
                f"There are unique_ids in Y_hat_df that are not in S_df: {reprlib.repr(Y_hat_diff)}"
            )

        if Y_nw is not None:
            Y_diff = set(Y_nw[id_col]) - set(Y_hat_nw[id_col])
            Y_hat_diff = set(Y_hat_nw[id_col]) - set(Y_nw[id_col])
            if Y_diff:
                raise ValueError(
                    f"There are unique_ids in Y_df that are not in Y_hat_df: {reprlib.repr(Y_diff)}"
                )
            if Y_hat_diff:
                raise ValueError(
                    f"There are unique_ids in Y_hat_df that are not in Y_df: {reprlib.repr(Y_hat_diff)}"
                )

        # Same Y_hat_df/S_df/Y_df's unique_ids. Order is guaranteed by sorting.
        # TODO: this logic should be method specific
        unique_ids = Y_hat_nw[id_col].unique().to_numpy()
        S_nw = S_nw.filter(nw.col(id_col).is_in(unique_ids))

        return Y_hat_nw, S_nw, Y_nw, model_names, id_col

    def _prepare_Y(
        self,
        Y_nw: Frame,
        S_nw: Frame,
        is_balanced: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> np.ndarray:
        """Prepare Y data."""
        if is_balanced:
            Y = Y_nw[target_col].to_numpy().reshape(len(S_nw), -1)
        else:
            Y_pivot = Y_nw.pivot(
                on=time_col, index=id_col, values=target_col, sort_columns=True
            ).sort(by=id_col)
            Y_pivot_cols_ex_id_col = Y_pivot.columns
            Y_pivot_cols_ex_id_col.remove(id_col)

            # TODO: check if this is the best way to do it - it's reasonably fast to ensure Y_pivot has same order as S_nw
            pos_in_Y = np.searchsorted(
                Y_pivot[id_col].to_numpy(), S_nw[id_col].to_numpy()
            )
            Y_pivot = Y_pivot.select(nw.col(Y_pivot_cols_ex_id_col))
            Y_pivot = Y_pivot[pos_in_Y]
            Y = Y_pivot.to_numpy()

        # TODO: the result is a Fortran contiguous array, see if we can avoid the below copy (I don't think so)
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        return Y

    def reconcile(
        self,
        Y_hat_df: Frame,
        tags: dict[str, np.ndarray],
        S_df: Frame = None,
        Y_df: Frame | None = None,
        level: list[int] | None = None,
        intervals_method: str = "normality",
        num_samples: int = -1,
        seed: int = 0,
        is_balanced: bool = False,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        id_time_col: str = "temporal_id",
        temporal: bool = False,
        S: Frame = None,  # For compatibility with the old API, S_df is now S
        diagnostics: bool = False,
        diagnostics_atol: float = 1e-6,
    ) -> FrameT:
        r"""Hierarchical Reconciliation Method.

        The `reconcile` method is analogous to SKLearn `fit_predict` method, it
        applies different reconciliation techniques instantiated in the `reconcilers` list.

        Most reconciliation methods can be described by the following convenient
        linear algebra notation:

        ```math
        \tilde{\mathbf{y}}_{[a,b],\\tau} = \mathbf{S}_{[a,b][b]} \mathbf{P}_{[b][a,b]} \hat{\mathbf{y}}_{[a,b],\\tau}
        ```

        where $a, b$ represent the aggregate and bottom levels, $\mathbf{S}_{[a,b][b]}$ contains
        the hierarchical aggregation constraints, and $\mathbf{P}_{[b][a,b]}$ varies across
        reconciliation methods. The reconciled predictions are

        ```math
        \tilde{\mathbf{y}}_{[a,b],\tau}
        ```

        and the base predictions

        ```math
        \hat{\mathbf{y}}_{[a,b],\tau}
        ```

        Args:
            Y_hat_df (Frame): DataFrame, base forecasts with columns ['unique_id', 'ds'] and models to reconcile.
            tags (dict[str, np.ndarray]): Each key is a level and its value contains tags associated to that level.
            S_df (Frame, optional): DataFrame with summing matrix of size `(base, bottom)`, see [aggregate method](./utils#function-aggregate). Default is None.
            Y_df (Optional[Frame], optional): DataFrame, training set of base time series with columns `['unique_id', 'ds', 'y']`.
                If a class of `self.reconciles` receives `y_hat_insample`, `Y_df` must include them as columns. Default is None.
            level (Optional[list[int]], optional): positive float list [0,100), confidence levels for prediction intervals. Default is None.
            intervals_method (str, optional): method used to calculate prediction intervals, one of `normality`, `bootstrap`, `permbu`. Default is "normality".
            num_samples (int, optional): if positive return that many probabilistic coherent samples. Default is -1.
            seed (int, optional): random seed for numpy generator's replicability. Default is 0.
            is_balanced (bool, optional): wether `Y_df` is balanced, set it to True to speed things up if `Y_df` is balanced. Default is False.
            id_col (str, optional): column that identifies each serie. Default is "unique_id".
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is "ds".
            target_col (str, optional): column that contains the target. Default is "y".
            id_time_col (str, optional): column that identifies each temporal aggregation level (required when `temporal=True`). Default is "temporal_id".
            temporal (bool, optional): if True, perform temporal reconciliation. Default is False.
            diagnostics (bool, optional): if True, compute coherence diagnostics and store in `self.diagnostics`. Default is False.
            diagnostics_atol (float, optional): absolute tolerance for numerical coherence check. Default is 1e-6.

        Returns:
            (FrameT): DataFrame, with reconciled predictions.

        Note:
            When `diagnostics=True`, after reconciliation completes, `self.diagnostics` will contain
            a DataFrame with coherence metrics per hierarchical level, including:
            - `coherence_residual_mae_before/after`: Mean absolute coherence residual before/after reconciliation
            - `adjustment_mae/rmse/max/mean`: Statistics on the adjustments made by reconciliation
            - `negative_count_before/after`: Count of negative values before/after reconciliation
            - `is_coherent`: Whether reconciled forecasts satisfy aggregation constraints (Overall level only)
            - `coherence_max_violation`: Maximum coherence violation (Overall level only)
        """
        # Handle deprecated S parameter
        if S is not None:
            import warnings

            if S_df is not None:
                raise ValueError(
                    "Both 'S' and 'S_df' parameters were provided. Please use only 'S_df'."
                )
            warnings.warn(
                "The 'S' parameter is deprecated and will be removed in a future version. "
                "Please use 'S_df' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            S_df = S

        # To Narwhals
        Y_hat_nw = nw.from_native(Y_hat_df)
        S_nw = nw.from_native(S_df)
        if Y_df is not None:
            Y_nw = nw.from_native(Y_df)
        else:
            Y_nw = None

        # Check input's validity and sort dataframes
        Y_hat_nw, S_nw, Y_nw, self.model_names, id_col = self._prepare_fit(
            Y_hat_nw=Y_hat_nw,
            S_nw=S_nw,
            Y_nw=Y_nw,
            tags=tags,
            level=level,
            intervals_method=intervals_method,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            id_time_col=id_time_col,
            temporal=temporal,
        )

        # Initialize reconciler arguments
        reconciler_args = dict(
            idx_bottom=np.arange(len(S_nw))[-S_nw.shape[1] + 1:],
            tags={
                key: S_nw.with_columns(nw.col(id_col).is_in(val).alias("in_cols"))[
                    "in_cols"
                ]
                .to_numpy()
                .nonzero()[0]
                for key, val in tags.items()
            },
        )

        any_sparse = any([method.is_sparse_method for method in self.reconcilers])
        S_nw_cols_ex_id_col = S_nw.columns
        S_nw_cols_ex_id_col.remove(id_col)
        if any_sparse:
            try:
                S_for_sparse = sparse.csr_matrix(
                    S_nw.select(nw.col(S_nw_cols_ex_id_col)).to_native().sparse.to_coo()
                )
            except AttributeError:
                S_for_sparse = sparse.csr_matrix(
                    S_nw.select(nw.col(S_nw_cols_ex_id_col))
                    .to_numpy()
                    .astype(np.float64, copy=False)
                )
        if Y_nw is not None:
            y_insample = self._prepare_Y(
                Y_nw=Y_nw,
                S_nw=S_nw,
                is_balanced=is_balanced,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            reconciler_args["y_insample"] = y_insample

        # Check if any reconciler requires a strictly hierarchical structure (fail fast)
        # Single traversal to collect all strict reconcilers
        strict_reconcilers = [
            r for r in self.reconcilers
            if getattr(r, 'is_strictly_hierarchical', False)
        ]

        if strict_reconcilers:
            # Check if any strict reconciler uses sparse methods
            has_sparse_strict_method = any(r.is_sparse_method for r in strict_reconcilers)

            # Perform the appropriate hierarchy validation
            is_valid_hierarchy = False
            if has_sparse_strict_method:
                # Use sparse check with adjacency matrix
                A = _construct_adjacency_matrix(S_for_sparse, reconciler_args["tags"])
                is_valid_hierarchy = _is_strictly_hierarchical(A, reconciler_args["tags"])
            else:
                # Use dense check with summing matrix
                S_numpy = (
                    S_nw.select(nw.col(S_nw_cols_ex_id_col))
                    .to_numpy()
                    .astype(np.float64, copy=False)
                )
                is_valid_hierarchy = is_strictly_hierarchical(S_numpy, reconciler_args["tags"])

            # Raise error if hierarchy is not valid
            if not is_valid_hierarchy:
                strict_method_names = [_build_fn_name(r) for r in strict_reconcilers]
                methods_str = "', '".join(strict_method_names)
                raise ValueError(
                    f"The reconciliation method(s) '{methods_str}' require a strictly hierarchical structure. "
                    f"The provided hierarchy contains nodes with multiple parents (grouped structure), "
                    f"which is not supported by these methods. Please use a different reconciliation method "
                    f"(e.g., BottomUp, MinTrace, or ERM) that supports grouped hierarchies."
                )

        Y_tilde_nw = nw.maybe_reset_index(Y_hat_nw.clone())
        self.execution_times = {}
        self.level_names = {}
        self.sample_names = {}
        for reconciler in self.reconcilers:
            reconcile_fn_name = _build_fn_name(reconciler)

            if reconciler.is_sparse_method:
                reconciler_args["S"] = S_for_sparse
            else:
                reconciler_args["S"] = (
                    S_nw.select(nw.col(S_nw_cols_ex_id_col))
                    .to_numpy()
                    .astype(np.float64, copy=False)
                )

            for model_name in self.model_names:
                start = time.time()
                recmodel_name = f"{model_name}/{reconcile_fn_name}"

                model_cols = [id_col, time_col, model_name]

                # TODO: the below should be method specific
                y_hat = self._prepare_Y(
                    Y_nw=Y_hat_nw[model_cols],
                    S_nw=S_nw,
                    is_balanced=True,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=model_name,
                )
                reconciler_args["y_hat"] = y_hat

                if Y_nw is not None and model_name in Y_nw.columns:
                    y_hat_insample = self._prepare_Y(
                        Y_nw=Y_nw[model_cols],
                        S_nw=S_nw,
                        is_balanced=is_balanced,
                        id_col=id_col,
                        time_col=time_col,
                        target_col=model_name,
                    )
                    reconciler_args["y_hat_insample"] = y_hat_insample

                if level is not None:
                    reconciler_args["intervals_method"] = intervals_method
                    reconciler_args["num_samples"] = 200
                    reconciler_args["seed"] = seed

                    if intervals_method in ["normality", "permbu"]:
                        sigmah = _reverse_engineer_sigmah(
                            Y_hat_df=Y_hat_nw,
                            y_hat=y_hat,
                            model_name=model_name,
                            id_col=id_col,
                            time_col=time_col,
                            target_col=target_col,
                            num_samples=reconciler_args["num_samples"],
                        )
                        reconciler_args["sigmah"] = sigmah

                # Mean and Probabilistic reconciliation
                kwargs_ls = [
                    key
                    for key in signature(reconciler.fit_predict).parameters
                    if key in reconciler_args.keys()
                ]
                kwargs = {key: reconciler_args[key] for key in kwargs_ls}

                fcsts_model = reconciler(**kwargs, level=level)

                # Validate reconciler state for sampling (fail fast before processing results)
                if num_samples > 0 and level is not None:
                    if not getattr(reconciler, "fitted", False):
                        raise ValueError(
                            f"Reconciler {reconcile_fn_name} does not support sampling. "
                            "Set num_samples=0 or use a different reconciler."
                        )
                    if getattr(reconciler, "sampler", None) is None:
                        raise ValueError(
                            f"Reconciler {reconcile_fn_name} does not have a sampler configured. "
                            "Ensure intervals_method is set correctly."
                        )

                # Parse final outputs
                Y_tilde_nw = Y_tilde_nw.with_columns(
                    **{recmodel_name: fcsts_model["mean"].flatten()}
                )

                if (
                    intervals_method in ["bootstrap", "normality", "permbu", "conformal"]
                    and level is not None
                ):
                    level.sort()
                    lo_names = [f"{recmodel_name}-lo-{lv}" for lv in reversed(level)]
                    hi_names = [f"{recmodel_name}-hi-{lv}" for lv in level]
                    self.level_names[recmodel_name] = lo_names + hi_names
                    sorted_quantiles = np.reshape(
                        fcsts_model["quantiles"], (len(Y_tilde_nw), -1)
                    )
                    y_tilde = dict(
                        zip(self.level_names[recmodel_name], sorted_quantiles.T, strict=False)
                    )
                    Y_tilde_nw = Y_tilde_nw.with_columns(**y_tilde)

                    if num_samples > 0:
                        samples = reconciler.sample(num_samples=num_samples)
                        self.sample_names[recmodel_name] = [
                            f"{recmodel_name}-sample-{i}" for i in range(num_samples)
                        ]
                        samples = np.reshape(samples, (len(Y_tilde_nw), -1))
                        y_tilde = dict(zip(self.sample_names[recmodel_name], samples.T, strict=False))
                        Y_tilde_nw = Y_tilde_nw.with_columns(**y_tilde)

                end = time.time()
                self.execution_times[f"{model_name}/{reconcile_fn_name}"] = end - start

        # Compute diagnostics if requested
        if diagnostics:
            native_namespace = nw.get_native_namespace(Y_hat_nw)

            # Prepare S matrix as dense numpy array for diagnostics
            S_numpy = (
                S_nw.select(nw.col(S_nw_cols_ex_id_col))
                .to_numpy()
                .astype(np.float64, copy=False)
            )

            # Get indices - note: S_numpy has shape (n_series, n_bottom)
            # idx_bottom should refer to the last n_bottom rows of y
            n_series = S_numpy.shape[0]
            n_bottom = S_numpy.shape[1]
            idx_bottom_diag = np.arange(n_series)[-n_bottom:]
            tags_numeric = reconciler_args["tags"]

            # Add "Overall" level
            tags_with_overall = {
                **tags_numeric,
                "Overall": np.arange(n_series),
            }

            diagnostics_per_model: dict[str, dict[str, dict[str, float]]] = {}

            for recmodel_name in self.execution_times.keys():
                # Extract base model name
                base_model_name = recmodel_name.split("/")[0]

                # Get base forecasts as numpy array (n_series, horizon)
                y_before = (
                    Y_hat_nw[base_model_name].to_numpy().reshape(n_series, -1)
                )
                # Get reconciled forecasts
                y_after = (
                    Y_tilde_nw[recmodel_name].to_numpy().reshape(n_series, -1)
                )

                # Compute coherence residuals for the entire array
                residual_before = _compute_coherence_residual(y_before, S_numpy, idx_bottom_diag)
                residual_after = _compute_coherence_residual(y_after, S_numpy, idx_bottom_diag)

                model_diagnostics: dict[str, dict[str, float]] = {}

                for level_name, level_indices in tags_with_overall.items():
                    level_metrics = _compute_diagnostics_for_level(
                        y_before=y_before,
                        y_after=y_after,
                        residual_before=residual_before,
                        residual_after=residual_after,
                        level_indices=level_indices,
                    )
                    model_diagnostics[level_name] = level_metrics

                # Add overall coherence check metrics
                max_violation = float(np.max(np.abs(residual_after)))
                is_coherent = float(max_violation <= diagnostics_atol)
                model_diagnostics["Overall"]["is_coherent"] = is_coherent
                model_diagnostics["Overall"]["coherence_max_violation"] = max_violation

                diagnostics_per_model[recmodel_name] = model_diagnostics

            self.diagnostics = _aggregate_diagnostics(
                diagnostics_per_model, backend=native_namespace
            )
        else:
            self.diagnostics = None

        Y_tilde_df = Y_tilde_nw.to_native()

        return Y_tilde_df

    def bootstrap_reconcile(
        self,
        Y_hat_df: Frame,
        S_df: Frame,
        tags: dict[str, np.ndarray],
        Y_df: Frame | None = None,
        level: list[int] | None = None,
        intervals_method: str = "normality",
        num_samples: int = -1,
        num_seeds: int = 1,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> FrameT:
        """Bootstraped Hierarchical Reconciliation Method.

        Applies N times, based on different random seeds, the `reconcile` method
        for the different reconciliation techniques instantiated in the `reconcilers` list.

        Args:
            Y_hat_df (Frame): DataFrame, base forecasts with columns ['unique_id', 'ds'] and models to reconcile.
            S_df (Frame): DataFrame with summing matrix of size `(base, bottom)`, see [aggregate method](./utils#function-aggregate).
            tags (dict[str, np.ndarray]): Each key is a level and its value contains tags associated to that level.
            Y_df (Optional[Frame], optional): DataFrame, training set of base time series with columns `['unique_id', 'ds', 'y']`.
                If a class of `self.reconciles` receives `y_hat_insample`, `Y_df` must include them as columns. Default is None.
            level (Optional[list[int]], optional): positive float list [0,100), confidence levels for prediction intervals. Default is None.
            intervals_method (str, optional): method used to calculate prediction intervals, one of `normality`, `bootstrap`, `permbu`. Default is "normality".
            num_samples (int, optional): if positive return that many probabilistic coherent samples. Default is -1.
            num_seeds (int, optional): random seed for numpy generator's replicability. Default is 1.
            id_col (str, optional): column that identifies each serie. Default is "unique_id".
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is "ds".
            target_col (str, optional): column that contains the target. Default is "y".

        Returns:
            (FrameT): DataFrame, with bootstraped reconciled predictions.
        """
        # Bootstrap reconciled predictions
        Y_tilde_list = []
        for seed in range(num_seeds):
            Y_tilde_df = self.reconcile(
                Y_hat_df=Y_hat_df,
                S_df=S_df,
                tags=tags,
                Y_df=Y_df,
                level=level,
                intervals_method=intervals_method,
                num_samples=num_samples,
                seed=seed,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            Y_tilde_nw = nw.from_native(Y_tilde_df)
            Y_tilde_nw = Y_tilde_nw.with_columns(nw.lit(seed).alias("seed"))

            # TODO: fix broken recmodel_names
            if seed == 0:
                first_columns = Y_tilde_nw.columns
            Y_tilde_nw = Y_tilde_nw.rename(
                {col: first_columns[i] for i, col in enumerate(first_columns)}
            )
            Y_tilde_list.append(Y_tilde_nw)

        Y_bootstrap_nw = nw.concat(Y_tilde_list, how="vertical")
        Y_bootstrap_df = Y_bootstrap_nw.to_native()

        return Y_bootstrap_df
