## Motivation
At Nixtla we have implemented several libraries to deal with time series data. We often have to apply some transformation over all of the series, which can prove time consuming even for simple operations like performing some kind of scaling.

We've used [numba](https://numba.pydata.org/) to speed up our expensive computations, however that comes with other issues such as cold starts and more dependencies (LLVM). That's why we developed this library, which implements several operators in C++ to transform time series data (or other kind of data that can be thought of as independent groups), with the possibility to use multithreading to get the best performance possible.

You probably won't need to use this library directly but rather use one of our higher level libraries like [mlforecast](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/lag_transforms_guide.html#built-in-transformations-experimental), which will use this library under the hood. If you're interested on using this library directly (only depends on numpy) you should continue reading.

## Installation

### PyPI
```python
pip install coreforecast
```

### conda-forge
```python
conda install -c conda-forge coreforecast
```

## Minimal example
The base data structure is the "grouped array" which holds two numpy 1d arrays:

* **data**: values of the series.
* **indptr**: series boundaries such that `data[indptr[i] : indptr[i + 1]]` returns the `i-th` series. For example, if you have two series of sizes 5 and 10 the indptr would be [0, 5, 15].

```python
import numpy as np
from coreforecast.grouped_array import GroupedArray

data = np.arange(10)
indptr = np.array([0, 3, 10])
ga = GroupedArray(data, indptr)
```

Once you have this structure you can run any of the provided transformations, for example:

```python
from coreforecast.lag_transforms import ExpandingMean
from coreforecast.scalers import LocalStandardScaler

exp_mean = ExpandingMean(lag=1).transform(ga)
scaler = LocalStandardScaler().fit(ga)
standardized = scaler.transform(ga)
```

## Multithreading
Every `GroupedArray` operation can split the groups across threads with `num_threads`, e.g. `GroupedArray(data, indptr, num_threads=4)`. Each call starts its own threads, so calling it from several Python threads at once runs `num_threads` threads per call.

coreforecast also runs without the GIL on free-threaded Python (3.14t and later). A `GroupedArray` can be shared across threads as long as none of them modifies `data` or `indptr` while another one is using it. Setting `num_threads` is safe at any time; calls already running keep the thread count they started with. Transformations and scalers store what they learn in `fit` and `transform`, so give each thread its own instance.

## Single-array functions
We've also implemented some functions that work on single arrays, you can refer to the following pages:

* [differences](https://nixtlaverse.nixtla.io/coreforecast/differences)
* [scalers](https://nixtlaverse.nixtla.io/coreforecast/scalers)
* [seasonal](https://nixtlaverse.nixtla.io/coreforecast/seasonal)
* [rolling](https://nixtlaverse.nixtla.io/coreforecast/rolling)
* [expanding](https://nixtlaverse.nixtla.io/coreforecast/expanding)
* [exponentially weighted](https://nixtlaverse.nixtla.io/coreforecast/exponentially_weighted)
