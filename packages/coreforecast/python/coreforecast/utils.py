import numpy as np

# what GroupedArray stores; any integer input is widened to it
_indptr_dtype = np.int64


def _diffs_to_indptr(diffs: np.ndarray) -> np.ndarray:
    diffs = diffs.astype(_indptr_dtype, copy=False)
    return np.append(
        _indptr_dtype(0),
        diffs.cumsum(dtype=_indptr_dtype),
    )
