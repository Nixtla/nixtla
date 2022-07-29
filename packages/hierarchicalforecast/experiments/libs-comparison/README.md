# Hierarchical Methods Comparison

## Main results

### Performance

[!image](./results.png)

### Time

| Dataset      |   statsforecast |   fable |   sktime |
|:-------------|----------------:|--------:|---------:|
| Labour       |           1.982 |  11.233 |   44.51  |
| TourismSmall |           0.627 |   7.61  |   19.294 |
| Wiki2        |           1.822 |  47.626 |  118.772 |

## Reproducibility

To reproduce the main results you have:

1. Execute `conda env create -f environment.yml`. 
2. Activate the environment using `conda activate hts-comparison`.
3. Run the experiments using `python -m src.[model] --group [group]` where `[model]` can be `statsforecast`, and `[group]` can be `Labour`, `Wiki2`, and `TourismSmall`.
4. To run R experiments you have to prepare the data using `python -m src.data --group [group]` for each `[group]`. Once it is done, just run `Rscript src/fable.R [group]`.
5. To parse the results, use `nbs/parse-results.ipynb`.

