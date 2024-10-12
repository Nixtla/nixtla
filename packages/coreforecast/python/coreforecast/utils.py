import numpy as np

_indptr_dtype = np.int32


def _diffs_to_indptr(diffs: np.ndarray) -> np.ndarray:
    diffs = diffs.astype(_indptr_dtype, copy=False)
    return np.append(
        _indptr_dtype(0),
        diffs.cumsum(dtype=_indptr_dtype),
    )
