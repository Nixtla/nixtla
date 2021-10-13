# TSForecast

This folder contains the code that executes the API called `tsforecast`. The main code is located in `forecast/make_forecast.py`. It's a module wrapping the library [mlforecast](https://github.com/Nixtla/mlforecast). 

You can use the code locally as follows:

- `python -m forecast.make_forecast [PARAMS]`

# Test code locally

1. Build the image using `make init`.
2. Test tsforecast code using `make test_tsforecast`.
