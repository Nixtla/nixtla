# TSBenchmarks

This folder contains the code that executes the API called `tsbenchmarks`. The main code is located in `benchmarks/compute_benchmarks.py`. The module computes performance based on M4 and M5 competitions.

You can use the code locally as follows:

- `python -m benchmarks.compute_benchmarks [PARAMS]`

# Test code locally

1. Build the image using `make init`.
2. Test tsforecast code using `make test_tsbenchmarks`.
