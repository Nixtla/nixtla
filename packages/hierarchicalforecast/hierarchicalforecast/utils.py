__all__ = ['aggregate', 'aggregate_temporal', 'make_future_dataframe', 'get_cross_temporal_tags', 'HierarchicalPlot']


import itertools
import reprlib
import sys
import timeit
from collections.abc import Sequence
from typing import Optional, Union

import matplotlib.pyplot as plt
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
import utilsforecast.validation as ufv
from narwhals.typing import Frame, FrameT
from numba import njit, prange
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

# Global variables
NUMBA_NOGIL = True
NUMBA_CACHE = True
NUMBA_PARALLEL = True
NUMBA_FASTMATH = True


class CodeTimer:
    def __init__(self, name=None, verbose=True):
        self.name = " '" + name + "'" if name else ""
        self.verbose = verbose

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = timeit.default_timer() - self.start
        if self.verbose:
            print(
                "Code block"
                + self.name
                + " took:\t{0:.5f}".format(self.took)
                + " seconds"
            )


def _construct_adjacency_matrix(
    S: sparse.csr_matrix, tags: dict[str, np.ndarray]
) -> sparse.csr_matrix:
    """Construct a disaggregation adjacency matrix.

    Args:
        S (sparse.csr_matrix): A summing matrix for hierarchical or grouped time series.
        tags (dict[str, np.ndarray]): A mapping of level name to node indices.

    Returns:
        sparse.csr_matrix: The disaggregation adjacency matrix for the structure.
    """
    # Get the nodes in each level.
    l = sorted(tags.values(), key=lambda x: len(x))
    # Get the number of aggregation nodes.
    n_a = S.shape[0] - S.shape[1]
    # Copy and cast the summing matrix to bool.
    S = S.astype(bool)
    # Precompute the transpose of the boolean summing matrix.
    S_T = S.T
    # Find the affinity, i.e., connectivity, between nodes in successive
    # levels, construct a sparse block diagonal matrix from these blocks, and
    # return the truncated disaggregation adjacency matrix.
    return sparse.hstack(
        (
            sparse.csr_matrix((n_a, len(l[0])), dtype=bool),
            sparse.block_diag(
                [S[l[i]] * S_T[:, l[i + 1]] for i in range(len(l) - 1)], "csr"
            ),
        )
    )


def is_strictly_hierarchical(S: np.ndarray, tags: dict[str, np.ndarray]) -> bool:
    # main idea:
    # if S represents a strictly hierarchical structure
    # the number of paths before the bottom level
    # should be equal to the number of nodes
    # of the previuos level
    levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
    # removing bottom level
    levels_.popitem()
    # making S categorical
    hiers = [np.argmax(S[idx], axis=0) + 1 for _, idx in levels_.items()]
    hiers = np.vstack(hiers)
    paths = np.unique(hiers, axis=1).shape[1]
    nodes = levels_.popitem()[1].size
    return paths == nodes


def _is_strictly_hierarchical(
    A: sparse.csr_matrix, tags: dict[str, np.ndarray]
) -> bool:
    """Check if a disaggregation structure is strictly hierarchical.

    The nodes in a strictly hierarchical disaggregation structure, except for
    the root node(s), should have exactly one incoming edge.

    Args:
        A (sparse.csr_matrix): A disaggregation adjacency matrix.
        tags (dict[str, np.ndarray]): A mapping of level name to node indices.

    Returns:
        bool: `True` if strictly hierarchical, otherwise `False`.
    """
    return np.all(A.sum(axis=0).A1[len(next(iter(tags.values()))) :] == 1)


def _to_upper_hierarchy(
    bottom_split: list[str], bottom_values: str, upper_key: str
) -> list[str]:
    upper_split = upper_key.split("/")
    upper_idxs = [bottom_split.index(i) for i in upper_split]

    def join_upper(bottom_value):
        bottom_parts = bottom_value.split("/")
        return "/".join(bottom_parts[i] for i in upper_idxs)

    return [join_upper(val) for val in bottom_values]


def aggregate(
    df: Frame,
    spec: list[list[str]],
    exog_vars: Optional[dict[str, Union[str, list[str]]]] = None,
    sparse_s: bool = False,
    id_col: str = "unique_id",
    time_col: str = "ds",
    id_time_col: Optional[str] = None,
    target_cols: Sequence[str] = ("y",),
) -> tuple[FrameT, FrameT, dict]:
    """Utils Aggregation Function.
    
    Aggregates bottom level series contained in the DataFrame `df` according
    to levels defined in the `spec` list.

    Args:
        df (Frame): Dataframe with columns `[time_col, *target_cols]`, columns to aggregate and optionally exog_vars.
        spec (list[list[str]]): list of levels. Each element of the list should contain a list of columns of `df` to aggregate.
        exog_vars (Optional[dict[str, Union[str, list[str]]]], optional): dictionary of string keys & values that can either be a list of strings or a single string
            keys correspond to column names and the values represent the aggregation(s) that will be applied to each column. Accepted values are those from Pandas or Polars aggregation Functions, check the respective docs for guidance. Default is None.
        sparse_s (bool, optional): Return `S_df` as a sparse Pandas dataframe. Default is False.
        id_col (str, optional): Column that will identify each serie after aggregation. Default is "unique_id".
        time_col (str, optional): Column that identifies each timestep, its values can be timestamps or integers. Default is "ds".
        id_time_col (Optional[str], optional): Column that will identify each timestep after temporal aggregation. If provided, aggregate will operate temporally. Default is None.
        target_cols (Sequence[str], optional): list of columns that contains the targets to aggregate. Default is ("y",).

    Returns:
        tuple[FrameT, FrameT, dict]: Y_df, S_df, tags
            Y_df: Hierarchically structured series.
            S_df: Summing dataframe.
            tags: Aggregation indices.
    """
    # To Narwhals
    target_cols = list(target_cols)
    df_nw = nw.from_native(df)
    backend = df_nw.implementation

    # Check if we do temporal aggregation
    if id_time_col is not None:
        temporal_agg = True
        group_col = id_col
        _id_col = id_time_col
    else:
        temporal_agg = False
        group_col = time_col
        _id_col = id_col

    # Checks
    # Generate order-preserving list of unique cols based on spec
    seen = set()
    spec_cols = [col for cols in spec for col in cols if col not in seen and not seen.add(col)]  # type: ignore[func-returns-value]

    # Check if last level in spec contains all levels
    missing_cols_in_bottom_spec = set(spec_cols) - set(spec[-1])
    if missing_cols_in_bottom_spec and not temporal_agg:
        raise ValueError(
            f"Check the last (bottom) level of spec, it has missing columns: {reprlib.repr(missing_cols_in_bottom_spec)}"
        )

    if sparse_s and not nw.dependencies.is_pandas_dataframe(df):
        raise ValueError("Sparse output is only supported for Pandas DataFrames.")

    for col in df_nw.columns:
        if df_nw[col].is_null().any():
            raise ValueError(
                f"Column {col} contains null values. Make sure no column in the DataFrame contains null values."
            )

    # Check whether all columns in the spec are in the df
    aggregation_cols_in_spec = list(
        dict.fromkeys([col for cols in spec for col in cols])
    )
    for col in aggregation_cols_in_spec:
        if col not in df_nw.columns:
            raise ValueError(f"Column {col} in spec not present in df")

    # Prepare the aggregation dictionary
    agg_dict = dict(
        zip(target_cols, tuple(zip(target_cols, len(target_cols) * ["sum"])))
    )

    # Check if exog_vars are present in df & add to the aggregation dictionary if it is not None
    exog_var_names = []
    if exog_vars is not None:
        missing_vars = [var for var in exog_vars.keys() if var not in df.columns]
        if missing_vars:
            raise ValueError(
                f"The following exogenous variables are not present in the DataFrame: {', '.join(missing_vars)}"
            )
        else:
            # Update agg_dict to handle multiple aggregations for each exog_vars key
            for key, agg_func in exog_vars.items():
                # Ensure agg_func is a list
                if isinstance(
                    agg_func, str
                ):  # If it's a single string, convert to list
                    agg_func = [agg_func]
                elif not isinstance(agg_func, list):  # Raise an error if it's neither
                    raise ValueError(
                        f"Aggregation functions for '{key}' must be a string or a list of strings."
                    )

                for func in agg_func:
                    agg_dict[f"{key}_{func}"] = (
                        key,
                        func,
                    )  # Update the agg_dict with the new naming structure
                    exog_var_names.append(f"{key}_{func}")

    # compute aggregations and tags
    spec = sorted(spec, key=len)

    tags = {}
    Y_nws = []
    category_list = []
    level_sep = "/"
    # Perform the aggregation
    for level in spec:
        level_name = level_sep.join(level)

        # Create Y_df
        Y_level = df_nw.with_columns(
            *[
                getattr(nw.col(col), agg)().over(level + [group_col]).alias(col_name)
                for col_name, (col, agg) in agg_dict.items()
            ]
        )
        Y_level = Y_level.select(
            nw.concat_str([nw.col(col) for col in level], separator=level_sep).alias(
                _id_col
            ),
            nw.all(),
        )
        # For temporal aggregation, we need to keep the time column
        if temporal_agg:
            Y_level = Y_level.sort(by=[id_col, time_col])
            Y_level = Y_level.unique(
                subset=level + [group_col], maintain_order=temporal_agg, keep="last"
            ).select([_id_col, group_col, time_col, *target_cols] + exog_var_names)
            tags[level_name] = (
                Y_level[_id_col].unique(maintain_order=temporal_agg).to_numpy()
            )
        else:
            Y_level = Y_level.unique(
                subset=level + [group_col], maintain_order=temporal_agg, keep="last"
            ).select([_id_col, group_col, *target_cols] + exog_var_names)
            Y_level = Y_level.sort(by=[_id_col, group_col])
            tags[level_name] = (
                Y_level[_id_col].unique(maintain_order=temporal_agg).sort().to_numpy()
            )

        Y_nws.append(Y_level)
        category_list.extend(tags[level_name])

    Y_nw = nw.concat(Y_nws, how="vertical")
    Y_nw = nw.maybe_reset_index(Y_nw)
    Y_df = Y_nw.to_native()

    # construct S
    if temporal_agg:
        bottom = spec_cols
        bottom_levels = (
            df_nw.select(
                nw.concat_str(
                    [nw.col(col) for col in bottom], separator=level_sep
                ).alias(_id_col)
            )
            .unique(subset=[_id_col], maintain_order=temporal_agg, keep="last")[_id_col]
            .to_numpy()
        )
        categories = list([v for k, v in tags.items() if k != level_name])
        categories += [bottom_levels]
    else:
        bottom = spec[-1]
        bottom_levels = tags[level_name]
        categories = list(tags.values())

    S = np.empty((len(bottom_levels), len(spec)), dtype=object)

    for j, levels in enumerate(spec[:-1]):
        S[:, j] = _to_upper_hierarchy(bottom, bottom_levels, level_sep.join(levels))
    S[:, -1] = bottom_levels

    encoder = OneHotEncoder(
        categories=categories, sparse_output=sparse_s, dtype=np.float64
    )
    S_dum = encoder.fit_transform(S)

    if not sparse_s:
        S_nw = nw.from_dict(
            {
                **{_id_col: category_list},
                **dict(zip(tags[level_name], S_dum)),
            },
            backend=backend,
        )
        S_nw = nw.maybe_reset_index(S_nw)
        S_df = S_nw.to_native()
    else:
        S_df = pd.DataFrame.sparse.from_spmatrix(
            S_dum.T, columns=list(bottom_levels), index=category_list
        )
        S_df = S_df.reset_index(names=_id_col)

    return Y_df, S_df, tags


def aggregate_temporal(
    df: Frame,
    spec: dict[str, int],
    exog_vars: Optional[dict[str, Union[str, list[str]]]] = None,
    sparse_s: bool = False,
    id_col: str = "unique_id",
    time_col: str = "ds",
    id_time_col: str = "temporal_id",
    target_cols: Sequence[str] = ("y",),
    aggregation_type: str = "local",
) -> tuple[FrameT, FrameT, dict]:
    """Utils Aggregation Function for Temporal aggregations.
    
    Aggregates bottom level timesteps contained in the DataFrame `df` according
    to temporal levels defined in the `spec` list.

    Args:
        df (Frame): Dataframe with columns `[time_col, target_cols]` and columns to aggregate.
        spec (dict[str, int]): Dictionary of temporal levels. Each key should be a string with the value representing the number of bottom-level timesteps contained in the aggregation.
        exog_vars (Optional[dict[str, Union[str, list[str]]]], optional): dictionary of string keys & values that can either be a list of strings or a single string
            keys correspond to column names and the values represent the aggregation(s) that will be applied to each column. Accepted values are those from Pandas or Polars aggregation Functions, check the respective docs for guidance. Default is None.
        sparse_s (bool, optional): Return `S_df` as a sparse Pandas dataframe. Default is False.
        id_col (str, optional): Column that will identify each serie after aggregation. Default is 'unique_id'.
        time_col (str, optional): Column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
        id_time_col (str, optional): Column that will identify each timestep after aggregation. Default is 'temporal_id'.
        target_cols (Sequence[str], optional): List of columns that contain the targets to aggregate. Default is ('y',).
        aggregation_type (str, optional): If 'local' the aggregation will be performed on the timestamps of each timeseries independently. If 'global' the aggregation will be performed on the unique timestamps of all timeseries. Default is 'local'.

    Returns:
        tuple[FrameT, FrameT, dict]: Y_df, S_df, tags
            Y_df: Temporally hierarchically structured series.
            S_df: Temporal summing dataframe.
            tags: Temporal aggregation indices.
    """
    # Check if ds column is a timestamp or integer, if not raise an error
    target_cols = list(target_cols)
    df = ufv.ensure_time_dtype(df=df, time_col=time_col)
    df = ufp.ensure_sorted(df=df, id_col=id_col, time_col=time_col)
    df_nw = nw.from_native(df)

    # We add a cumulative count column to the dataframe to be able to compute the aggregations
    if aggregation_type == "local":
        df_nw = df_nw.with_columns(
            nw.col(time_col).cum_count().over([id_col]).alias(f"{time_col}_count")
        )
    elif aggregation_type == "global":
        unique_ts = df_nw.select(nw.col(time_col)).unique(maintain_order=True)
        unique_ts = unique_ts.with_columns(
            nw.col(time_col).cum_count().alias(f"{time_col}_count")
        )
        df_nw = df_nw.join(unique_ts, on=[time_col], how="left")
        df_nw = df_nw.sort(by=[id_col, time_col])
    else:
        raise ValueError("aggregation_type must be either 'local' or 'global'.")
    df_nw = nw.maybe_reset_index(df_nw)

    # Check spec that lowest level with seasonality of 1 has been defined
    if 1 not in spec.values():
        raise ValueError(
            "The spec must contain a level with a seasonality of 1. This represents the lowest level in the temporal aggregation."
        )

    # Check for duplicate values in spec
    if len(spec) != len(set(spec.values())):
        raise ValueError(
            "The spec must contain unique values for each level. Each value represents the seasonality of the aggregation."
        )

    # Check for duplicate keys in spec
    if len(spec) != len(set(spec.keys())):
        raise ValueError(
            "The spec must contain unique keys for each level. Each key represents the name of the aggregation."
        )

    # Loop over the spec and create the aggregation columns
    spec_agg: list = []
    for agg, seasonality in spec.items():
        df_nw = df_nw.with_columns(
            nw.concat_str(
                [
                    nw.lit(agg),
                    nw.lit("-"),
                    (((nw.col(f"{time_col}_count") - 1) // seasonality) + 1).cast(
                        nw.String
                    ),
                ]
            ).alias(agg)
        )
        spec_agg.append([agg])

    # If target_cols is not in df, we add a placeholder column so that we can compute the aggregations
    add_placeholder = False
    if set(df.columns) == set([time_col, id_col]):
        add_placeholder = True
        df_nw = df_nw.with_columns(nw.lit(0).alias("y"))

    # Create the aggregation
    df = df_nw.to_native()
    Y_df, S_df, tags = aggregate(
        df=df,
        spec=spec_agg,
        exog_vars=exog_vars,
        sparse_s=sparse_s,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        id_time_col=id_time_col,
    )
    Y_nw = nw.from_native(Y_df)

    # Drop the placeholder column if it was added
    if add_placeholder:
        Y_nw = Y_nw.drop("y")

    Y_df = Y_nw.to_native()

    return Y_df, S_df, tags


def make_future_dataframe(
    df: Frame,
    freq: Union[str, int],
    h: int,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> FrameT:
    """Create future dataframe for forecasting.

    Args:
        df (Frame): Dataframe with ids, times and values for the exogenous regressors.
        freq (Union[str, int]): Frequency of the data. Must be a valid pandas or polars offset alias, or an integer.
        h (int): Forecast horizon.
        id_col (str, optional): Column that identifies each serie. Default is 'unique_id'.
        time_col (str, optional): Column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.

    Returns:
        FrameT: DataFrame with future values.
    """
    times_by_id = ufp.group_by_agg(df, id_col, {time_col: "max"}, maintain_order=True)
    times_by_id = ufp.sort(times_by_id, id_col)
    future_df = ufp.make_future_dataframe(
        uids=times_by_id[id_col],
        last_times=times_by_id[time_col],
        freq=freq,
        h=h,
        id_col=id_col,
        time_col=time_col,
    )
    return future_df


def get_cross_temporal_tags(
    df: Frame,
    tags_cs: dict[str, np.ndarray],
    tags_te: dict[str, np.ndarray],
    sep: str = "//",
    id_col: str = "unique_id",
    id_time_col: str = "temporal_id",
    cross_temporal_id_col: str = "cross_temporal_id",
) -> tuple[FrameT, dict[str, np.ndarray]]:
    """Get cross-temporal tags.

    Args:
        df (Frame): DataFrame with temporal ids.
        tags_cs (dict[str, np.ndarray]): Tags for the cross-sectional hierarchies.
        tags_te (dict[str, np.ndarray]): Tags for the temporal hierarchies.
        sep (str, optional): Separator for the cross-temporal tags. Default is "//".
        id_col (str, optional): Column that identifies each serie. Default is 'unique_id'.
        id_time_col (str, optional): Column that identifies each (aggregated) timestep. Default is 'temporal_id'.
        cross_temporal_id_col (str, optional): Column that will identify each cross-temporal aggregation. Default is 'cross_temporal_id'.

    Returns:
        tuple[FrameT, dict[str, np.ndarray]]: df, tags_ct
            df: DataFrame with cross-temporal ids.
            tags_ct: Tags for the cross-temporal hierarchies.
    """
    df_nw = nw.from_native(df)

    # Check if relevant columns are present
    if id_col not in df_nw.columns:
        raise ValueError(f"Column '{id_col}' not present in df")
    if id_time_col not in df_nw.columns:
        raise ValueError(f"Column '{id_time_col}' not present in df")

    # Create cross-temporal tags
    tags_ct = {}
    for key_cs, value_cs in tags_cs.items():
        for key_te, value_te in tags_te.items():
            key_ct = key_cs + sep + key_te
            value_ct = list(sep.join(s) for s in itertools.product(value_cs, value_te))
            tags_ct[key_ct] = value_ct

    df_nw = df_nw.with_columns(
        **{cross_temporal_id_col: df_nw[id_col] + sep + df_nw[id_time_col]}
    )
    df = df_nw.to_native()

    return df, tags_ct


class HierarchicalPlot:
    """Hierarchical Plot

    This class contains a collection of matplotlib visualization methods, suited for small
    to medium sized hierarchical series.

    Args:
        S (Frame): DataFrame with summing matrix of size `(base, bottom)`, see [aggregate function](./utils#function-aggregate).
        tags (dict[str, np.ndarray]): hierarchical aggregation indexes, where
            each key is a level and its value contains tags associated to that level.
        S_id_col (str, optional): column that identifies each aggregation. Default is 'unique_id'.
    """

    def __init__(
        self,
        S: Frame,
        tags: dict[str, np.ndarray],
        S_id_col: str = "unique_id",
    ):

        self.S = nw.from_native(S)
        S_cols_ex_id_col = self.S.columns
        S_cols_ex_id_col.remove(S_id_col)
        self.S_cols_ex_id_col = S_cols_ex_id_col
        self.tags = tags

    def plot_summing_matrix(self):
        """Summation Constraints plot

        This method simply plots the hierarchical aggregation
        constraints matrix $\mathbf{S}$.

        Returns:
            matplotlib.figure.Figure: figure object containing the plot of the summing matrix.
        """
        fig = plt.figure(num=1, figsize=(4, 6), dpi=80, facecolor="w")
        plt.spy(self.S[self.S_cols_ex_id_col].to_numpy())
        plt.show()
        return fig

    def plot_series(
        self,
        series: str,
        Y_df: Frame,
        models: Optional[list[str]] = None,
        level: Optional[list[int]] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        """Single Series plot

        Args:
            series (str): string identifying the `'unique_id'` any-level series to plot.
            Y_df (Frame): hierarchically structured series ($\mathbf{y}_{[a,b]}$).
                    It contains columns `['unique_id', 'ds', 'y']`, it may have `'models'`.
            models (Optional[list[str]], optional): string identifying filtering model columns. Default is None.
            level (Optional[list[int]], optional): confidence levels for prediction intervals available in `Y_df`. Default is None.
            id_col (str, optional): column that identifies each serie. Default is 'unique_id'.
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
            target_col (str, optional): column that contains the target. Default is 'y'.

        Returns:
            matplotlib.figure.Figure: figure object containing the plot of the single series.
        """
        Y_nw = nw.from_native(Y_df)

        if series not in self.S[id_col]:
            raise Exception(f"time series {series} not found")
        fig, ax = plt.subplots(1, 1, figsize=(20, 7))
        df_plot = Y_nw.filter(nw.col(id_col) == series)
        cols = (
            models
            if models is not None
            else [col for col in df_plot.columns if col not in [id_col, time_col]]
        )
        cols_wo_levels = [
            col for col in cols if ("-lo-" not in col and "-hi-" not in col)
        ]
        try:
            cmap = plt.get_cmap("tab10", 10)
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10", 10)
        cmap = [cmap(i) for i in range(10)][: len(cols_wo_levels)]
        cmap_dict = dict(zip(cols_wo_levels, cmap))
        for col in cols_wo_levels:
            ax.plot(
                df_plot[time_col].to_numpy(),
                df_plot[col].to_numpy(),
                linewidth=2,
                label=col,
                color=cmap_dict[col],
            )
            if level is not None and col != target_col:
                for lv in level:
                    if f"{col}-lo-{lv}" not in df_plot.columns:
                        # if model
                        # doesnt have levels
                        continue
                    ax.fill_between(
                        df_plot.select(nw.col(time_col))[time_col].to_numpy(),
                        df_plot.select(nw.col(f"{col}-lo-{lv}"))[
                            f"{col}-lo-{lv}"
                        ].to_numpy(),
                        df_plot.select(nw.col(f"{col}-hi-{lv}"))[
                            f"{col}-hi-{lv}"
                        ].to_numpy(),
                        alpha=-lv / 100 + 1,
                        color=cmap_dict[col],
                        label=f"{col}_level_{lv}",
                    )
        ax.set_title(f"{series} Forecast", fontsize=22)
        ax.set_xlabel("Timestamp [t]", fontsize=20)
        ax.legend(prop={"size": 15})
        ax.grid()
        ax.xaxis.set_major_locator(plt.MaxNLocator(min(max(len(df_plot) // 10, 1), 10)))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(20)

        plt.show()
        return fig

    def plot_hierarchically_linked_series(
        self,
        bottom_series: str,
        Y_df: Frame,
        models: Optional[list[str]] = None,
        level: Optional[list[int]] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        """Hierarchically Linked Series plot

        Args:
            bottom_series (str): string identifying the `'unique_id'` bottom-level series to plot.
            Y_df (Frame): hierarchically structured series ($\mathbf{y}_{[a,b]}$).
                    It contains columns ['unique_id', 'ds', 'y'] and models.
            models (Optional[list[str]], optional): string identifying filtering model columns. Default is None.
            level (Optional[list[int]], optional): confidence levels for prediction intervals available in `Y_df`. Default is None.
            id_col (str, optional): column that identifies each serie. Default is 'unique_id'.
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
            target_col (str, optional): column that contains the target. Default is 'y'.

        Returns:
            matplotlib.figure.Figure: figure object containing the plots of the hierarchilly linked series.
        """
        Y_nw = nw.from_native(Y_df)

        if bottom_series not in self.S.columns:
            raise Exception(f"bottom time series {bottom_series} not found")

        linked_series = (
            self.S[[id_col, bottom_series]]
            .filter(nw.col(bottom_series) == 1)[id_col]
            .to_numpy()
        )
        fig, axs = plt.subplots(
            len(linked_series), 1, figsize=(20, 2 * len(linked_series))
        )
        cols = (
            models
            if models is not None
            else [col for col in Y_nw.columns if col not in [id_col, time_col]]
        )
        cols_wo_levels = [
            col for col in cols if ("-lo-" not in col and "-hi-" not in col)
        ]
        cmap = plt.cm.get_cmap("tab10", 10)
        cmap = [cmap(i) for i in range(10)][: len(cols_wo_levels)]
        cmap_dict = dict(zip(cols_wo_levels, cmap))
        for idx, series in enumerate(linked_series):
            df_plot = Y_nw.filter(nw.col(id_col) == series)
            for col in cols_wo_levels:
                axs[idx].plot(
                    df_plot[time_col].to_numpy(),
                    df_plot[col].to_numpy(),
                    linewidth=2,
                    label=col,
                    color=cmap_dict[col],
                )
                if level is not None and col != target_col:
                    for lv in level:
                        if f"{col}-lo-{lv}" not in df_plot.columns:
                            # if model
                            # doesnt have levels
                            continue
                        axs[idx].fill_between(
                            df_plot.select(nw.col(time_col))[time_col].to_numpy(),
                            df_plot.select(nw.col(f"{col}-lo-{lv}"))[
                                f"{col}-lo-{lv}"
                            ].to_numpy(),
                            df_plot.select(nw.col(f"{col}-hi-{lv}"))[
                                f"{col}-hi-{lv}"
                            ].to_numpy(),
                            alpha=-lv / 100 + 1,
                            color=cmap_dict[col],
                            label=f"{col}_level_{lv}",
                        )
            axs[idx].set_title(f"{series}", fontsize=10)
            axs[idx].grid()
            axs[idx].get_xaxis().label.set_visible(False)
            axs[idx].legend().set_visible(False)
            axs[idx].xaxis.set_major_locator(
                plt.MaxNLocator(min(max(len(df_plot) // 10, 1), 10))
            )
            for label in axs[idx].get_xticklabels() + axs[idx].get_yticklabels():
                label.set_fontsize(10)
        plt.subplots_adjust(hspace=0.4)
        handles, labels = axs[0].get_legend_handles_labels()
        kwargs = dict(
            loc="lower center", prop={"size": 10}, bbox_to_anchor=(0, 0.05, 1, 1)
        )
        if sys.version_info.minor > 7:
            kwargs["ncols"] = np.max([2, np.ceil(len(labels) / 2)])
        fig.legend(handles, labels, **kwargs)
        return fig

    def plot_hierarchical_predictions_gap(
        self,
        Y_df: Frame,
        models: Optional[list[str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        """Hierarchically Predictions Gap plot

        Args:
            Y_df (Frame): hierarchically structured series ($\mathbf{y}_{[a,b]}$).
                    It contains columns ['unique_id', 'ds', 'y'] and models.
            models (Optional[list[str]], optional): string identifying filtering model columns. Default is None.
            xlabel (Optional[str], optional): string for the plot's x axis label. Default is None.
            ylabel (Optional[str], optional): string for the plot's y axis label. Default is None.
            id_col (str, optional): column that identifies each serie. Default is 'unique_id'.
            time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
            target_col (str, optional): column that contains the target. Default is 'y'.

        Returns:
            matplotlib.figure.Figure: figure object containing the plot of the aggregated predictions at different levels of the hierarchical structure.
        """
        Y_nw = nw.from_native(Y_df)

        # Parse predictions dataframe
        horizon_dates = Y_nw["ds"].unique().to_numpy()
        cols = (
            models
            if models is not None
            else [col for col in Y_nw.columns if col not in [id_col, time_col]]
        )

        # Plot predictions across tag levels
        fig, ax = plt.subplots(figsize=(8, 5))

        if target_col in Y_nw.columns:
            idx_top = (
                self.S.with_columns(sum_cols=nw.sum_horizontal(cols))
                .sort(by="sum_cols", descending=True)[0][id_col]
                .to_numpy()
            )
            y_plot = Y_nw.filter(nw.col(id_col) == idx_top)[target_col].to_numpy()
            plt.plot(horizon_dates, y_plot, label="True")

        ys = []
        for tag in self.tags:
            y_plot = sum(
                [
                    Y_nw.filter(nw.col(id_col) == idx)[cols].to_numpy()
                    for idx in self.tags[tag]
                ]
            )
            plt.plot(horizon_dates, y_plot, label=f"Level: {tag}")

            ys.append(y_plot[:, None])

        plt.title("Predictions Accumulated Difference")
        if ylabel is not None:
            plt.ylabel(ylabel)
        if xlabel is not None:
            plt.xlabel(xlabel)

        plt.legend()
        plt.grid()
        plt.show()
        return fig


# convert levels to output quantile names
def level_to_outputs(level: list[int]) -> tuple[list[float], list[str]]:
    """Converts list of levels into output names matching StatsForecast and NeuralForecast methods.

    Args:
        level (list[int]): Probability levels for prediction intervals [0,100].

    Returns:
        tuple[list[float], list[str]]: quantiles and output_names
            quantiles: quantiles derived from levels.
            output_names: String list with output column names.
    """
    qs = sum([[50 - l / 2, 50 + l / 2] for l in level], [])
    output_names = sum([[f"-lo-{l}", f"-hi-{l}"] for l in level], [])

    sort_idx = np.argsort(qs)
    quantiles = np.array(qs)[sort_idx]

    # Add default median
    quantiles = np.concatenate([np.array([50]), quantiles]) / 100
    output_names = list(np.array(output_names)[sort_idx])
    output_names.insert(0, "-median")

    return quantiles, output_names


# convert quantiles to output quantile names
def quantiles_to_outputs(quantiles: list[float]) -> tuple[list[float], list[str]]:
    """Converts list of quantiles into output names matching StatsForecast and NeuralForecast methods.

    Args:
        quantiles (list[float]): Alternative to level, quantiles to estimate from y distribution [0., 1.].

    Returns:
        tuple[list[float], list[str]]: quantiles and output_names
            quantiles: quantiles to estimate from y distribution.
            output_names: String list with output column names.
    """
    output_names = []
    for q in quantiles:
        if q < 0.50:
            output_names.append(f"-lo-{np.round(100-200*q,2)}")
        elif q > 0.50:
            output_names.append(f"-hi-{np.round(100-200*(1-q),2)}")
        else:
            output_names.append("-median")
    return quantiles, output_names


# given input array of sample forecasts and inptut quantiles/levels,
# output a Pandas Dataframe with columns of quantile predictions
def samples_to_quantiles_df(
    samples: np.ndarray,
    unique_ids: Sequence[str],
    dates: list[str],
    quantiles: Optional[list[float]] = None,
    level: Optional[list[int]] = None,
    model_name: str = "model",
    id_col: str = "unique_id",
    time_col: str = "ds",
    backend: str = "pandas",
) -> tuple[list[float], FrameT]:
    """Transform Random Samples into HierarchicalForecast input.

    Auxiliary function to create compatible HierarchicalForecast input `Y_hat_df` dataframe.

    Args:
        samples (np.ndarray): Samples from forecast distribution of shape [n_series, n_samples, horizon].
        unique_ids (Sequence[str]): Unique identifiers for each time series.
        dates (list[str]): list of forecast dates.
        quantiles (Optional[list[float]], optional): Alternative to level, quantiles to estimate from y distribution [0., 1.]. Default is None.
        level (Optional[list[int]], optional): Probability levels for prediction intervals [0,100]. Default is None.
        model_name (str, optional): Name of forecasting model. Default is "model".
        id_col (str, optional): column that identifies each serie. Default is 'unique_id'.
        time_col (str, optional): column that identifies each timestep, its values can be timestamps or integers. Default is 'ds'.
        backend (str, optional): backend to use for the output dataframe, either 'pandas' or 'polars'. Default is 'pandas'.

    Returns:
        tuple[list[float], FrameT]: quantiles and Y_hat_df
            quantiles: quantiles to estimate from y distribution [0., 1.].
            Y_hat_df: DataFrame with base quantile forecasts with columns ds and models to reconcile indexed by unique_id.
    """

    # Get the shape of the array
    n_series, n_samples, horizon = samples.shape

    if n_series != len(unique_ids):
        raise ValueError(
            f"Number of unique_ids ({len(unique_ids)}) must match the number of series ({n_series})."
        )
    if horizon != len(dates):
        raise ValueError(
            f"Number of dates ({len(dates)}) must match third dimension of samples array ({horizon})."
        )
    if not ((quantiles is None) ^ (level is None)):
        raise ValueError("Either quantiles or level must be provided, but not both.")

    namespace = sys.modules.get(backend, None)
    if namespace is None:
        raise ValueError(f"DataFrame backend {backend} not installed.")

    # create initial dictionary
    forecasts_mean = np.mean(samples, axis=1).flatten()
    unique_ids = np.repeat(unique_ids, horizon)
    ds = np.tile(dates, n_series)

    # create quantiles and quantile names
    if level is not None:
        _quantiles, quantile_names = level_to_outputs(level)
    elif quantiles is not None:
        _quantiles, quantile_names = quantiles_to_outputs(quantiles)

    percentiles = [quantile * 100 for quantile in _quantiles]
    col_names = np.array(
        [model_name + quantile_name for quantile_name in quantile_names]
    )

    # add quantiles to dataframe
    forecasts_quantiles = np.percentile(samples, percentiles, axis=1)

    forecasts_quantiles = np.transpose(
        forecasts_quantiles, (1, 2, 0)
    )  # [Q,H,N] -> [N,H,Q]
    forecasts_quantiles = forecasts_quantiles.reshape(-1, len(_quantiles))

    df_nw = nw.from_dict(
        {
            **{id_col: unique_ids, time_col: ds, model_name: forecasts_mean},
            **dict(zip(col_names, forecasts_quantiles.T)),
        },
        backend=backend,
    )

    return _quantiles, df_nw.to_native()


# Masked empirical covariance matrix
@njit(
    "Array(float64, 2, 'F')(Array(float64, 2, 'C'), Array(bool_, 2, 'C'))",
    nogil=NUMBA_NOGIL,
    cache=NUMBA_CACHE,
    parallel=NUMBA_PARALLEL,
    fastmath=NUMBA_FASTMATH,
    error_model="numpy",
)
def _ma_cov(residuals: np.ndarray, not_nan_mask: np.ndarray):
    """Masked empirical covariance matrix.

    :meta private:
    """
    n_timeseries = residuals.shape[0]
    W = np.zeros((n_timeseries, n_timeseries), dtype=np.float64).T
    for i in prange(n_timeseries):
        not_nan_mask_i = not_nan_mask[i]
        for j in range(i + 1):
            not_nan_mask_j = not_nan_mask[j]
            not_nan_mask_ij = not_nan_mask_i & not_nan_mask_j
            n_samples = np.sum(not_nan_mask_ij)
            # Only compute if we have enough non-nan samples in the time series pair
            if n_samples > 1:
                # Masked residuals
                residuals_i = residuals[i][not_nan_mask_ij]
                residuals_j = residuals[j][not_nan_mask_ij]
                residuals_i_mean = np.mean(residuals_i)
                residuals_j_mean = np.mean(residuals_j)
                X_i = residuals_i - residuals_i_mean
                X_j = residuals_j - residuals_j_mean
                # Empirical covariance
                factor_emp_cov = np.float64(1 / (n_samples - 1))
                W[i, j] = W[j, i] = factor_emp_cov * np.sum(X_i * X_j)

    return W


# Shrunk covariance matrix using the Schafer-Strimmer method


@njit(
    "Array(float64, 2, 'F')(Array(float64, 2, 'C'), float64)",
    nogil=NUMBA_NOGIL,
    cache=NUMBA_CACHE,
    parallel=NUMBA_PARALLEL,
    fastmath=NUMBA_FASTMATH,
    error_model="numpy",
)
def _shrunk_covariance_schaferstrimmer_no_nans(
    residuals: np.ndarray, mint_shr_ridge: float
):
    """Shrink empirical covariance according to the following method:
        Schäfer, Juliane, and Korbinian Strimmer.
        "A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and
        Implications for Functional Genomics". Statistical Applications in
        Genetics and Molecular Biology 4, no. 1 (14 January 2005).
        https://doi.org/10.2202/1544-6115.1175.

    :meta private:
    """
    n_timeseries = residuals.shape[0]
    n_samples = residuals.shape[1]

    # We need the empirical covariance, the off-diagonal sum of the variance of
    # the empirical correlation matrix and the off-diagonal sum of the squared
    # empirical correlation matrix.
    W = np.zeros((n_timeseries, n_timeseries), dtype=np.float64).T
    sum_var_emp_corr = np.float64(0.0)
    sum_sq_emp_corr = np.float64(0.0)
    factor_emp_cov = np.float64(1 / (n_samples - 1))
    factor_shrinkage = np.float64(1 / (n_samples * (n_samples - 1)))
    epsilon = np.float64(2e-8)
    for i in prange(n_timeseries):
        # Mean of the standardized residuals
        X_i = residuals[i] - np.mean(residuals[i])
        Xs_i = X_i / (np.std(residuals[i]) + epsilon)
        Xs_i_mean = np.mean(Xs_i)
        for j in range(i + 1):
            # Empirical covariance
            X_j = residuals[j] - np.mean(residuals[j])
            W[i, j] = factor_emp_cov * np.sum(X_i * X_j)
            # Off-diagonal sums
            if i != j:
                Xs_j = X_j / (np.std(residuals[j]) + epsilon)
                Xs_j_mean = np.mean(Xs_j)
                # Sum off-diagonal variance of empirical correlation
                w = (Xs_i - Xs_i_mean) * (Xs_j - Xs_j_mean)
                w_mean = np.mean(w)
                sum_var_emp_corr += np.sum(np.square(w - w_mean))
                # Sum squared empirical correlation
                sum_sq_emp_corr += w_mean**2

    # Calculate shrinkage intensity
    shrinkage = 1.0 - max(
        min((factor_shrinkage * sum_var_emp_corr) / (sum_sq_emp_corr + epsilon), 1.0),
        0.0,
    )
    # Shrink the empirical covariance
    for i in prange(n_timeseries):
        for j in range(i + 1):
            if i != j:
                W[i, j] = W[j, i] = shrinkage * W[i, j]
            else:
                W[i, j] = W[j, i] = max(W[i, j], mint_shr_ridge)
    return W


@njit(
    "Array(float64, 2, 'F')(Array(float64, 2, 'C'), Array(bool_, 2, 'C'), float64)",
    nogil=NUMBA_NOGIL,
    cache=NUMBA_CACHE,
    parallel=NUMBA_PARALLEL,
    fastmath=NUMBA_FASTMATH,
    error_model="numpy",
)
def _shrunk_covariance_schaferstrimmer_with_nans(
    residuals: np.ndarray, not_nan_mask: np.ndarray, mint_shr_ridge: float
):
    """Shrink empirical covariance according to the following method:
        Schäfer, Juliane, and Korbinian Strimmer.
        "A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and
        Implications for Functional Genomics". Statistical Applications in
        Genetics and Molecular Biology 4, no. 1 (14 January 2005).
        https://doi.org/10.2202/1544-6115.1175.

    :meta private:
    """
    n_timeseries = residuals.shape[0]

    # We need the empirical covariance, the off-diagonal sum of the variance of
    # the empirical correlation matrix and the off-diagonal sum of the squared
    # empirical correlation matrix.
    W = np.zeros((n_timeseries, n_timeseries), dtype=np.float64).T
    sum_var_emp_corr = np.float64(0.0)
    sum_sq_emp_corr = np.float64(0.0)
    epsilon = np.float64(2e-8)
    for i in prange(n_timeseries):
        not_nan_mask_i = not_nan_mask[i]
        for j in range(i + 1):
            not_nan_mask_j = not_nan_mask[j]
            not_nan_mask_ij = not_nan_mask_i & not_nan_mask_j
            n_samples = np.sum(not_nan_mask_ij)
            # Only compute if we have enough non-nan samples in the time series pair
            if n_samples > 1:
                # Masked residuals
                residuals_i = residuals[i][not_nan_mask_ij]
                residuals_j = residuals[j][not_nan_mask_ij]
                residuals_i_mean = np.mean(residuals_i)
                residuals_j_mean = np.mean(residuals_j)
                X_i = residuals_i - residuals_i_mean
                X_j = residuals_j - residuals_j_mean
                # Empirical covariance
                factor_emp_cov = np.float64(1 / (n_samples - 1))
                W[i, j] = factor_emp_cov * np.sum(X_i * X_j)
                # Off-diagonal sums
                if i != j:
                    factor_var_emp_cor = np.float64(n_samples / (n_samples - 1) ** 3)
                    residuals_i_std = np.std(residuals_i) + epsilon
                    residuals_j_std = np.std(residuals_j) + epsilon
                    Xs_i = X_i / (residuals_i_std + epsilon)
                    Xs_j = X_j / (residuals_j_std + epsilon)
                    Xs_im_mean = np.mean(Xs_i)
                    Xs_jm_mean = np.mean(Xs_j)
                    # Sum off-diagonal variance of empirical correlation
                    w = (Xs_i - Xs_im_mean) * (Xs_j - Xs_jm_mean)
                    w_mean = np.mean(w)
                    sum_var_emp_corr += factor_var_emp_cor * np.sum(
                        np.square(w - w_mean)
                    )
                    # Sum squared empirical correlation
                    sum_sq_emp_corr += np.square(factor_emp_cov * n_samples * w_mean)

    # Calculate shrinkage intensity
    shrinkage = 1.0 - max(
        min((sum_var_emp_corr) / (sum_sq_emp_corr + epsilon), 1.0), 0.0
    )

    # Shrink the empirical covariance
    for i in prange(n_timeseries):
        for j in range(i + 1):
            if i != j:
                W[i, j] = W[j, i] = shrinkage * W[i, j]
            else:
                W[i, j] = W[j, i] = max(W[i, j], mint_shr_ridge)

    return W


# Lasso cyclic coordinate descent
@njit(
    "Array(float64, 1, 'C')(Array(float64, 2, 'C'), Array(float64, 1, 'C'), float64, int64, float64)",
    nogil=NUMBA_NOGIL,
    cache=NUMBA_CACHE,
    fastmath=NUMBA_FASTMATH,
    error_model="numpy",
)
def _lasso(
    X: np.ndarray,
    y: np.ndarray,
    lambda_reg: float,
    max_iters: int = 1_000,
    tol: float = 1e-4,
):
    # lasso cyclic coordinate descent
    n, feats = X.shape
    norms = np.sum(X**2, axis=0)
    beta = np.zeros(feats, dtype=np.float64)
    beta_changes = np.zeros(feats, dtype=np.float64)
    residuals = y.copy()

    for it in range(max_iters):
        for i in range(feats):
            norms_i = norms[i]
            # is feature is close to zero, we
            # continue to the next.
            # in this case is optimal betai= 0
            if abs(norms_i) < 1e-8:
                continue
            beta_i = beta[i]

            # we calculate the normalized derivative
            rho = beta_i
            for j in range(n):
                rho += X[j, i] * residuals[j] / norms_i

            # soft threshold
            beta_i_next = np.sign(rho) * max(
                np.abs(rho) - lambda_reg * n / norms_i, 0.0
            )  # (norms[i] + 1e-3), 0.)
            beta_delta = beta_i - beta_i_next
            beta_changes[i] = np.abs(beta_delta)
            if beta_delta != 0.0:
                for j in range(n):
                    residuals[j] += beta_delta * X[j, i]

                beta[i] = beta_i_next

        if max(beta_changes) < tol:
            break

    return beta
