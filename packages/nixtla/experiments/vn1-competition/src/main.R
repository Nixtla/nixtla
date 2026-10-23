
# VN1 Forecasting Competition Solution with nixtlar ---- 

install.packages(c("nixtlar", "tidyverse", "data.table"))

library(nixtlar)
library(tidyverse)
library(data.table)

source("functions.R") # same directory as main.R

## Load Data ---- 
sales0 <- read_and_prepare_data("phase0_sales")
sales1 <- read_and_prepare_data("phase1_sales")
test_df <- read_and_prepare_data("phase2_sales")

## Prepare Training Dataset ---- 
train_df <- get_train_data(sales0, sales1)

## Generate TimeGPT Forecast  ----

# nixtla_client_setup(api_key = "Your API key here") 
# Learn how to set up your API key here: https://nixtla.github.io/nixtlar/articles/setting-up-your-api-key.html

fc <- nixtla_client_forecast(train_df, h=13, model="timegpt-1-long-horizon")

## Visualize TimeGPT Forecast ----
nixtla_client_plot(train_df, fc)

## Evaluate TimeGPT & Top 5 Competition Solutions ----
timegpt_score <- vn1_competition_evaluation(test_df, fc, "TimeGPT")

scores <- lapply(1:5, function(i){ # Top 5 
  winner_df <- read_and_prepare_data(paste0("winners", i))
  vn1_competition_evaluation(test_df, winner_df, model = paste0("winners", i))
})

scores_df <- data.frame(
  "Result" = c(paste0("Place #", 1:5), "TimeGPT"), 
  "Score" = c(as.numeric(scores), timegpt_score)
)

scores_df <- scores_df |> arrange(Score)
print(scores_df) # TimeGPT places 2nd! 
