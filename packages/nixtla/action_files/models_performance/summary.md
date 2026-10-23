<details><summary>Experiment Results</summary>

## Experiment 1: air-passengers

### Description:
| variable      | experiment   |
|:--------------|:-------------|
| h             | 12           |
| season_length | 12           |
| freq          | MS           |
| level         | None         |
| n_windows     | 1            |

### Results:
| metric     |   timegpt-1 |   timegpt-1-long-horizon |   SeasonalNaive |      Naive |
|:-----------|------------:|-------------------------:|----------------:|-----------:|
| mae        |     12.6793 |                  11.0623 |         47.8333 |    76      |
| mape       |      0.027  |                   0.0232 |          0.0999 |     0.1425 |
| mse        |    213.936  |                 199.132  |       2571.33   | 10604.2    |
| total_time |      2.4918 |                   1.5065 |          0.0046 |     0.0045 |

### Plot:
![](/action_files/models_performance/plots/plot_air-passengers_12_12_MS_None_1.png)

## Experiment 2: air-passengers

### Description:
| variable      | experiment   |
|:--------------|:-------------|
| h             | 24           |
| season_length | 12           |
| freq          | MS           |
| level         | None         |
| n_windows     | 1            |

### Results:
| metric     |   timegpt-1 |   timegpt-1-long-horizon |   SeasonalNaive |      Naive |
|:-----------|------------:|-------------------------:|----------------:|-----------:|
| mae        |     58.1031 |                  58.4587 |         71.25   |   115.25   |
| mape       |      0.1257 |                   0.1267 |          0.1552 |     0.2358 |
| mse        |   4040.21   |                4110.79   |       5928.17   | 18859.2    |
| total_time |      0.5508 |                   0.5551 |          0.0032 |     0.0028 |

### Plot:
![](/action_files/models_performance/plots/plot_air-passengers_24_12_MS_None_1.png)

## Experiment 3: electricity-multiple-series

### Description:
| variable      | experiment   |
|:--------------|:-------------|
| h             | 24           |
| season_length | 24           |
| freq          | H            |
| level         | None         |
| n_windows     | 1            |

### Results:
| metric     |   timegpt-1 |   timegpt-1-long-horizon |   SeasonalNaive |          Naive |
|:-----------|------------:|-------------------------:|----------------:|---------------:|
| mae        |    178.293  |                 268.129  |        269.23   | 1331.02        |
| mape       |      0.0234 |                   0.0311 |          0.0304 |    0.1692      |
| mse        | 121586      |              219467      |     213677      |    4.68961e+06 |
| total_time |      1.2402 |                   1.5986 |          0.0046 |    0.0036      |

### Plot:
![](/action_files/models_performance/plots/plot_electricity-multiple-series_24_24_H_None_1.png)

## Experiment 4: electricity-multiple-series

### Description:
| variable      | experiment   |
|:--------------|:-------------|
| h             | 168          |
| season_length | 24           |
| freq          | H            |
| level         | None         |
| n_windows     | 1            |

### Results:
| metric     |   timegpt-1 |   timegpt-1-long-horizon |   SeasonalNaive |          Naive |
|:-----------|------------:|-------------------------:|----------------:|---------------:|
| mae        |    465.496  |                 346.976  |        398.956  | 1119.26        |
| mape       |      0.062  |                   0.0436 |          0.0512 |    0.1583      |
| mse        | 835064      |              403762      |     656723      |    3.17316e+06 |
| total_time |      0.6668 |                   0.637  |          0.0046 |    0.0037      |

### Plot:
![](/action_files/models_performance/plots/plot_electricity-multiple-series_168_24_H_None_1.png)

## Experiment 5: electricity-multiple-series

### Description:
| variable      | experiment   |
|:--------------|:-------------|
| h             | 336          |
| season_length | 24           |
| freq          | H            |
| level         | None         |
| n_windows     | 1            |

### Results:
| metric     |     timegpt-1 |   timegpt-1-long-horizon |   SeasonalNaive |          Naive |
|:-----------|--------------:|-------------------------:|----------------:|---------------:|
| mae        | 558.702       |                 459.769  |   602.926       | 1340.95        |
| mape       |   0.0697      |                   0.0565 |     0.0787      |    0.17        |
| mse        |   1.22728e+06 |              739162      |     1.61572e+06 |    6.04619e+06 |
| total_time |   1.1054      |                   1.3368 |     0.0046      |    0.0038      |

### Plot:
![](/action_files/models_performance/plots/plot_electricity-multiple-series_336_24_H_None_1.png)

</details>
