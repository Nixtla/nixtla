library(tsibble)
library(fable)
library(dplyr)
library(readr)
library(future)

df <- read_csv('data/prepared-data-train.csv')

plan(multiprocess, gc = TRUE)
forecasts <- df %>%
  as_tsibble(key = unique_id, index = ds) %>%
  mutate(ds = yearmonth(ds)) %>% 
  model(arima = ARIMA(y, stepwise=FALSE, approximation=FALSE)) %>% 
  forecast(h = "4 month") %>%
  as_tibble() %>% 
  select(-y, -.model) %>%
  rename(yhat = .mean) %>% 
  write_csv('data/arima-forecasts.csv')
