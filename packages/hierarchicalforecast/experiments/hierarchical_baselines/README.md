# HierarchicalForecast Baselines

This study showcases the HierarchicalForecast library's statistical baselines, following established experimental protocols from previous research Rangapuram et al. [2021], Olivares et al. [2023] and Kamarthi et al. [2022]. The benchmark datasets utilized include Australian Monthly Labour, SF Bay Area daily Traffic, Quarterly Australian Tourism Visits, Monthly Australian Tourism visits, and daily Wikipedia article views.

In our experiment, we compare the predictions of several state-of-the-art probabilistic coherent methods. The statistical baselines encompass different variants, such as (1) BOOTSTRAP [Panagiotelis et al., 2023], (2) NORMALITY [Wickramasuriya et al., 2022], and (3) PERMBU probabilistic reconciliation [Ben Taieb et al., 2017], combined with mean reconcilers including BottomUp [Orcutt et al., 1968], TopDown [Gross and Sohl, 1990], and MinTrace [Wickramasuriya et al., 2019].

## Probabilistic Coherent Forecast Accuracy

To evaluate the accuracy of probabilistic coherent forecasts, we employ the Mean scaled Continuous Ranked Probability Scores (sCRPS) integral. This evaluation metric utilizes a Riemann approximation to the sCRPS, considering quantile intervals of 1 percent denoted as $dq$. We report here the best performing method across the BottomUp, TopDown and MinTrace reconciliations.
```math
\mathrm{sCRPS}(\mathbb{P}, \mathbf{y}_{[i],\tau}) = \frac{2}{|[i\,]|} \sum_{i}
    \frac{\int^{1}_{0} \mathrm{QL}(\mathbb{P}_{i,\tau}, y_{i,\tau})_{q} dq }{\sum_{i} | y_{i,\tau} |}
```

|              |  **BOOTSTRAP**              ||| **NORMALITY**               ||| **PERMBU**                  |||
|--------------|----------|---------|----------|----------|---------|----------|----------|---------|----------|
| **Dataset**  | BottomUp | TopDown | MinTrace | BottomUp | TopDown | MinTrace | BottomUp | TopDown | MinTrace |
| Labour       | 0.0078   | 0.0668  | 0.0073   | 0.0076   | 0.0656  | 0.0069   | 0.0077   | 0.0623  | 0.0069   |
| Traffic      | 0.0736   | 0.0741  | 0.0608   | 0.0845   | 0.0738  | 0.0630   | 0.0849   | 0.0708  | 0.0651   |
| TourismS     | 0.0682   | 0.1040  | 0.0703   | 0.0649   | 0.1000  | 0.6830   | 0.0649   | 0.0898  | 0.0680   |
| TourismL     | 0.1375   | -       | 0.1313   | 0.1620   | -       | 0.1338   | -        | -       | -        |
| Wiki2        | 0.2894   | 0.3231  | 0.2808   | 0.3914   | 0.3385  | 0.3385   | 0.3920   | 0.4269  | 0.3821   |
 <br>


## Hierarchically Coherent Forecast Accuracy

To assess the accuracy of hierarchically coherent mean forecasts, we employ the relative mean squared error that compares the Naive forecast, with reconciled StatsForecast's base AutoARIMA forecasts.
```math
\mathrm{relMSE}(\mathbf{y}_{[i]}, \hat{\mathbf{y}}_{[i]}, \mathbf{\check{y}}_{[i]}) =
    \frac{\mathrm{MSE}(\mathbf{y}_{[i]}, \mathbf{\hat{y}}_{[i]})}{\mathrm{MSE}(\mathbf{y}_{[i]}, \mathbf{\check{y}}_{[i]})}
```

| Dataset  | BottomUp | TopDown | MinTrace |
|----------|----------|---------|----------|
| Labour   | 0.5382   | 16.8204 | 0.3547   |
| Traffic  | 0.1394   | 0.0614  | 0.0744   |
| TourismS | 0.1002   | 0.1919  | 0.1235   |
| TourismL | 0.3070   | -       | 0.1375   |
| Wiki2    | 1.0163   | 1.4482  | 1.0068   |
<br>

## Reproducibility

1. Create a conda environment `hierarchical_baselines` using the `environment.yml` file.
  ```shell
  conda env create -f environment.yml
  ```

3. Activate the conda environment using 
  ```shell
  conda activate hierarchical_baselines
  ```

4. Run the experiments for each dataset and each model using with 
- `--intervals_method` parameter in `['bootstrap', 'normality', 'permbu']`
- `--dataset` parameter in `['Labour', 'Traffic', 'OldTraffic', 'TourismSmall', 'TourismLarge', 'OldTourismLarge', 'Wikitwo']`
  ```shell
  python src/run_baselines.py --intervals_method 'bootstrap' --dataset 'OldTourismLarge'
  ```

You can access the final reconciled forecasts from the `./data/{dataset}/{intervals_method}_rec.csv` file. Example: `./data/TourismLarge/bootstrap_rec.csv`.
The file gives you access to the mean and quantile forecasts for all series in the dataset. Example `AutoARIMA/BottomUp`, `AutoARIMA/BottomUp-lo-90`, `AutoARIMA/BottomUp-hi-90`, that correspond to mean, q5 and q95.

<br><br>

## References
- [Kin G. Olivares, Federico Garza, David Luo, Cristian Challú, Max Mergenthaler, Souhaib Ben Taieb, Shanika Wickramasuriya, and Artur Dubrawski 2023. "HierarchicalForecast: A reference framework for hierarchical forecasting". Journal of Machine Learning Research, submitted URL https://arxiv.org/abs/2207.03517](https://arxiv.org/abs/2207.03517)

### Datasets
- [Australian Bureau of Statistics. Labour force, Australia. Accessed Online, 2019. URL https://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/6202.0Dec%202019?OpenDocument.](https://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/6202.0Dec%202019?OpenDocument)
- [Dheeru Dua and Casey Graff. Traffic dataset. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.](http://archive.ics.uci.edu/ml.)
- [Tourism Australia, Canberra. Tourism Research Australia (2005), Travel by Australians. URL https://www.kaggle.com/luisblanche/quarterly-tourism-in-australia/](https://www.kaggle.com/luisblanche/quarterly-tourism-in-australia/)
- [Tourism Australia, Canberra. Detailed tourism Australia (2005), Travel by Australians, Sep 2019. URL https://robjhyndman.com/publications/hierarchical-tourism/.](https://robjhyndman.com/publications/hierarchical-tourism/)
- [Oren Anava, Vitaly Kuznetsov, and (Google Inc. Sponsorship). Web traffic time series forecasting, forecast future traffic to wikipedia pages. Kaggle Competition, 2018. URL https://www.kaggle.com/c/web-traffic-time-series-forecasting/.](https://www.kaggle.com/c/web-traffic-time-series-forecasting/)

### Baselines
- [BOOTSTRAP: Anastasios Panagiotelis, Puwasala Gamakumara, George Athanasopoulos, and Rob J. Hyndman (2022). "Probabilistic forecast reconciliation: Properties, evaluation and score optimisation". European Journal of Operational Research, 306(2):693–706, 2023. ISSN 0377-2217. doi: https://doi.org/10. 1016/j.ejor.2022.07.040. URL https://www.sciencedirect.com/science/article/pii/S0377221722006087.](https://www.sciencedirect.com/science/article/pii/S0377221722006087)
- [PERMBU: Souhaib Ben Taieb, James W. Taylor, and Rob J. Hyndman. Coherent probabilistic forecasts for hierarchical time series. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3348–3357. PMLR, 06–11 Aug 2017. URL http://proceedings.mlr.press/v70/taieb17a.html.](http://proceedings.mlr.press/v70/taieb17a.html)
- [NORMALITY: Shanika L. Wickramasuriya. Probabilistic forecast reconciliation under the Gaussian framework. Accepted at Journal of Business and Economic Statistics, 2023.]()
- [Orcutt, G.H., Watts, H.W., & Edwards, J.B.(1968). \"Data aggregation and information loss\". The American Economic Review, 58 , 773{787)](http://www.jstor.org/stable/1815532).
- [CW. Gross (1990). \"Disaggregation methods to expedite product line forecasting\". Journal of Forecasting, 9 , 233–254. doi:10.1002/for.3980090304](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3980090304).
- [Wickramasuriya, S.L., Turlach, B.A. & Hyndman, R.J. (2020). "Optimal non-negative forecast reconciliation". Stat Comput 30, 1167–1182, https://doi.org/10.1007/s11222-020-09930-0](https://robjhyndman.com/publications/nnmint/).
