# TSFeatures

This folder contains the code that executes the API called `tsfeatures` and `calendartsfeatures`. The main code executed by `tsfeatures` is located in `features/make_features.py`. It's a module wrapping the library [tsfeatures](https://github.com/Nixtla/tsfeatures) and [tsfresh](https://github.com/blue-yonder/tsfresh). 

By the other hand, the main code executed by `calendartsfeatures` is `calendar/make_holidays.py`. 

You can run the modules using:

- `python -m calendar.make_holidays [PARAMS]`.

# Test code locally

1. Build the image using `make init`.
2. Test tsforecast code using `make test_tsfeatures`.
