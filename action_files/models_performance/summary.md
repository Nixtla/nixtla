## Experiment 1
### Description:
| variable      | experiment     |
|:--------------|:---------------|
| h             | 12             |
| season_length | 12             |
| freq          | MS             |
| level         | None           |
| n_windows     | 1              |
| experiment    | air-passengers |

### Results:
| metric     |   TimeGPT |   SeasonalNaive |      Naive |
|:-----------|----------:|----------------:|-----------:|
| mae        |   12.6793 |         47.8333 |    76      |
| mape       |    0.027  |          0.0999 |     0.1425 |
| mse        |  213.936  |       2571.33   | 10604.2    |
| total_time |    5.0026 |          5.0874 |     0.4146 |

### Plot:
![](/action_files/models_performance/plots/plot_12_12_MS_None_1.png)

## Experiment 2
### Description:
| variable      | experiment     |
|:--------------|:---------------|
| h             | 24             |
| season_length | 12             |
| freq          | MS             |
| level         | None           |
| n_windows     | 1              |
| experiment    | air-passengers |

### Results:
| metric     |   TimeGPT |   SeasonalNaive |      Naive |
|:-----------|----------:|----------------:|-----------:|
| mae        |   58.1031 |         71.25   |   115.25   |
| mape       |    0.1257 |          0.1552 |     0.2358 |
| mse        | 4040.21   |       5928.17   | 18859.2    |
| total_time |    2.5503 |          0.004  |     0.0036 |

### Plot:
![](/action_files/models_performance/plots/plot_24_12_MS_None_1.png)

## Experiment 3
### Description:
| variable      | experiment                  |
|:--------------|:----------------------------|
| h             | 24                          |
| season_length | 24                          |
| freq          | H                           |
| level         | None                        |
| n_windows     | 1                           |
| experiment    | electricity-multiple-series |

### Results:
| metric     |   TimeGPT |   SeasonalNaive |   Naive |
|:-----------|----------:|----------------:|--------:|
| mae        |    4.8617 |          5.6289 |  5.2381 |
| mape       |    0.6816 |          0.7654 |  0.6328 |
| mse        |   40.5749 |         63.023  | 50.8454 |
| total_time |    3.1519 |          0.0092 |  0.0087 |

### Plot:
![](/action_files/models_performance/plots/plot_24_24_H_None_1.png)

## Experiment 4
### Description:
| variable      | experiment                               |
|:--------------|:-----------------------------------------|
| h             | 24                                       |
| season_length | 24                                       |
| freq          | H                                        |
| level         | None                                     |
| n_windows     | 1                                        |
| experiment    | electricity-multiple-series-with-ex-vars |

### Results:
| metric     |   TimeGPT |   SeasonalNaive |   Naive |
|:-----------|----------:|----------------:|--------:|
| mae        |    5.5206 |          5.6289 |  5.2381 |
| mape       |    0.4674 |          0.7654 |  0.6328 |
| mse        |   40.9623 |         63.023  | 50.8454 |
| total_time |    3.2932 |          0.4221 |  0.4013 |

### Plot:
![](/action_files/models_performance/plots/plot_24_24_H_None_1.png)

## Experiment 5
### Description:
| variable      | experiment                  |
|:--------------|:----------------------------|
| h             | 168                         |
| season_length | 24                          |
| freq          | H                           |
| level         | None                        |
| n_windows     | 1                           |
| experiment    | electricity-multiple-series |

### Results:
| metric     |   TimeGPT |   SeasonalNaive |    Naive |
|:-----------|----------:|----------------:|---------:|
| mae        |    8.3102 |          9.3176 |  12.6464 |
| mape       |    1.382  |          1.0786 |   1.4208 |
| mse        |  165.13   |        202.596  | 336.086  |
| total_time |    3.5833 |          0.0107 |   0.0102 |

### Plot:
![](/action_files/models_performance/plots/plot_168_24_H_None_1.png)

## Experiment 6
### Description:
| variable      | experiment                               |
|:--------------|:-----------------------------------------|
| h             | 168                                      |
| season_length | 24                                       |
| freq          | H                                        |
| level         | None                                     |
| n_windows     | 1                                        |
| experiment    | electricity-multiple-series-with-ex-vars |

### Results:
| metric     |   TimeGPT |   SeasonalNaive |    Naive |
|:-----------|----------:|----------------:|---------:|
| mae        |    7.5605 |          9.3176 |  12.6464 |
| mape       |    1.371  |          1.0786 |   1.4208 |
| mse        |  135.78   |        202.596  | 336.086  |
| total_time |    5.5679 |          0.0123 |   0.0118 |

### Plot:
![](/action_files/models_performance/plots/plot_168_24_H_None_1.png)

