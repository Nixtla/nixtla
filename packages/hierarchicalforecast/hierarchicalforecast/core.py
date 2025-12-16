__all__ = ['HierarchicalReconciliation']


import copy
import re
import reprlib
import time
from inspect import signature
from typing import Optional

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import Frame, FrameT
from scipy import sparse
from scipy.stats import norm

from .methods import HReconciler


def _build_fn_name(fn) -> str:
    fn_name = type(fn).__name__
    func_params = fn.__dict__

    # Take default parameter out of names
    args_to_remove = ["insample", "num_threads"]
    if not func_params.get("nonnegative", False):
        args_to_remove.append("nonnegative")

    if fn_name == "MinTrace" and func_params["method"] == "mint_shrink":
        if func_params["mint_shr_ridge"] == 2e-8:
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
    """Reverse engineer sigma_h from prediction intervals.

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
    level_cols = re.findall("[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+", pi_col)
    level_col = float(level_cols[-1])
    z = norm.ppf(0.5 + level_col / num_samples)
    sigmah = Y_hat_df[pi_col].to_numpy().reshape(n_series, -1)
    sigmah = sign * (sigmah - y_hat) / z

    return sigmah


class HierarchicalReconciliation:
    """Hierarchical Reconciliation Class.

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
        self.orig_reconcilers = copy.deepcopy(reconcilers)  # TODO: elegant solution

    def _prepare_fit(
        self,
        Y_hat_nw: Frame,
        S_nw: Frame,
        Y_nw: Optional[Frame],
        tags: dict[str, np.ndarray],
        level: Optional[list[int]] = None,
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
            for reconciler in self.orig_reconcilers:
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
        if intervals_method not in ["normality", "bootstrap", "permbu"]:
            raise ValueError(f"Unknown interval method: {intervals_method}")

        # Check absence of Y_nw for insample reconcilers
        if Y_nw is None:
            for reconciler in self.orig_reconcilers:
                if reconciler.insample:
                    reconciler_name = _build_fn_name(reconciler)
                    raise ValueError(
                        f"You need to provide `Y_df` for reconciler {reconciler_name}"
                    )
            if intervals_method in ["bootstrap", "permbu"]:
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
        if not np.allclose(S_nw[S_nw_cols][-len(S_nw_cols) :], np.eye(len(S_nw_cols))):
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
        Y_df: Optional[Frame] = None,
        level: Optional[list[int]] = None,
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

        Returns:
            (FrameT): DataFrame, with reconciled predictions.
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
            idx_bottom=np.arange(len(S_nw))[-S_nw.shape[1] :],
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

        Y_tilde_nw = nw.maybe_reset_index(Y_hat_nw.clone())
        self.execution_times = {}
        self.level_names = {}
        self.sample_names = {}
        for reconciler, name_copy in zip(self.reconcilers, self.orig_reconcilers):
            reconcile_fn_name = _build_fn_name(name_copy)

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

                if (level is not None) and (num_samples > 0):
                    # Store reconciler's memory to generate samples
                    reconciler = reconciler.fit(**kwargs)
                    fcsts_model = reconciler.predict(
                        S=reconciler_args["S"],
                        y_hat=reconciler_args["y_hat"],
                        level=level,
                    )
                else:
                    # Memory efficient reconciler's fit_predict
                    fcsts_model = reconciler(**kwargs, level=level)

                # Parse final outputs
                Y_tilde_nw = Y_tilde_nw.with_columns(
                    **{recmodel_name: fcsts_model["mean"].flatten()}
                )

                if (
                    intervals_method in ["bootstrap", "normality", "permbu"]
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
                        zip(self.level_names[recmodel_name], sorted_quantiles.T)
                    )
                    Y_tilde_nw = Y_tilde_nw.with_columns(**y_tilde)

                    if num_samples > 0:
                        samples = reconciler.sample(num_samples=num_samples)
                        self.sample_names[recmodel_name] = [
                            f"{recmodel_name}-sample-{i}" for i in range(num_samples)
                        ]
                        samples = np.reshape(samples, (len(Y_tilde_nw), -1))
                        y_tilde = dict(zip(self.sample_names[recmodel_name], samples.T))
                        Y_tilde_nw = Y_tilde_nw.with_columns(**y_tilde)

                end = time.time()
                self.execution_times[f"{model_name}/{reconcile_fn_name}"] = end - start

        Y_tilde_df = Y_tilde_nw.to_native()

        return Y_tilde_df

    def bootstrap_reconcile(
        self,
        Y_hat_df: Frame,
        S_df: Frame,
        tags: dict[str, np.ndarray],
        Y_df: Optional[Frame] = None,
        level: Optional[list[int]] = None,
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
