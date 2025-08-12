__all__ = ['transform_exog']


from typing import Optional

import utilsforecast.processing as ufp
from utilsforecast.compat import DFType
from utilsforecast.validation import validate_format

from .core import Lags, LagTransforms, _parse_transforms
from .grouped_array import GroupedArray


def transform_exog(
    df: DFType,
    lags: Optional[Lags] = None,
    lag_transforms: Optional[LagTransforms] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    num_threads: int = 1,
) -> DFType:
    """Compute lag features for dynamic exogenous regressors.

    Args:
        df (pandas or polars DataFrame): Dataframe with ids, times and values for the exogenous regressors.
        lags (list of int, optional): Lags of the target to use as features. Defaults to None.
        lag_transforms (dict of int to list of functions, optional): Mapping of target lags to their transformations. Defaults to None.
        id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
        time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
        num_threads (int): Number of threads to use when computing the features. Defaults to 1.

    Returns:
        (pandas or polars DataFrame): Original DataFrame with the computed features
    """
    if lags is None and lag_transforms is None:
        raise ValueError("At least one of `lags` or `lag_transforms` is required.")
    if lags is None:
        lags = []
    if lag_transforms is None:
        lag_transforms = {}
    tfms = _parse_transforms(lags, lag_transforms)
    targets = [c for c in df.columns if c not in (id_col, time_col)]
    # this is just a dummy target because process_df requires one
    target_col = targets[0]
    validate_format(df, id_col, time_col, target_col)
    processed = ufp.process_df(df, id_col, time_col, target_col)
    results = {}
    cols = []
    for j, target in enumerate(targets):
        ga = GroupedArray(processed.data[:, j], processed.indptr)
        named_tfms = {f"{target}_{k}": v for k, v in tfms.items()}
        if num_threads == 1 or len(named_tfms) == 1:
            computed_tfms = ga.apply_transforms(
                transforms=named_tfms, updates_only=False
            )
        else:
            computed_tfms = ga.apply_multithreaded_transforms(
                transforms=named_tfms, num_threads=num_threads, updates_only=False
            )
        results.update(computed_tfms)
        cols.extend(list(named_tfms.keys()))
    if processed.sort_idxs is not None:
        base_df = ufp.take_rows(df, processed.sort_idxs)
    else:
        base_df = df
    base_df = ufp.drop_index_if_pandas(base_df)
    return ufp.horizontal_concat([base_df, type(df)(results)[cols]])
