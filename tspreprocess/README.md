# TSPreprocess

This folder contains the code that executes the API called `tspreprocess`. The main code is located in `preprocess/make_preprocess.py`. The module creates one-hot-encoded static variables and is also capable of balancing the panel for missing values.

You can use the code locally as follows:

- `python -m preprocess.make_preprocess [PARAMS]`

# Test code locally

1. Build the image using `make init`.
2. Test tsforecast code using `make test_tspreprocess`.
