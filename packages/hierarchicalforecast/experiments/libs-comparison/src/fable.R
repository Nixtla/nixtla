library(fable)
library(tsibble)
library(tsibbledata)
library(lubridate)
library(dplyr)
library(readr)
library(future)
library(stringr)

args <- commandArgs(trailingOnly=TRUE)
meta <- list(
	TourismSmall=list(ds_fn=yearquarter, cutoff=yearquarter('2004-12-31'), 
			  key=c("Purpose", "State", "CityNonCity")),
	Labour=list(ds_fn=yearmonth, cutoff=yearmonth('2019-04-01'), 
		    key=c('Region', 'Employment', 'Gender')),
	Wiki2=list(ds_fn=ymd, cutoff=ymd('2016-12-17'), 
		   key=c('Country', 'Access', 'Agent', 'Topic'))
)
group <- args[1]
ds_fn <- meta[[group]][['ds_fn']]
cutoff <- meta[[group]][['cutoff']]
key <- meta[[group]][['key']]

plan(multiprocess, gc=TRUE)


Y_df <- read_csv(str_glue('./data/{group}.csv')) %>%
	mutate(ds = ds_fn(ds)) %>%
	as_tsibble(
		index = ds,
		key = key,
	) 
if(group == 'TourismSmall'){
	Y_df <- aggregate_key(Y_df, Purpose / State / CityNonCity, y = sum(y))
} else if (group == 'Labour') {
	Y_df <- aggregate_key(Y_df, Region / Employment / Gender, y = sum(y))
} else if (group == 'Wiki2') {
	Y_df <- aggregate_key(Y_df, Country / Access / Agent / Topic, y = sum(y))
}

#split train/test sets
Y_df_train <- Y_df %>%
	filter(ds <= cutoff)
Y_df_test <- Y_df %>%
	filter(ds > cutoff)

#forecaster
start <- Sys.time()
ets_fit <- Y_df_train %>%
	model(ets = ETS(y), naive = NAIVE(y)) 
end <- Sys.time()

ets_fit <- ets_fit %>%
	reconcile(
		bu = bottom_up(ets),
		ols = min_trace(ets, method='ols'),
		wls_struct = min_trace(ets, method='wls_struct'),
		wls_var = min_trace(ets, method='wls_var'),
		mint_shrink = min_trace(ets, method='mint_shrink'),
	)
fc <- ets_fit %>%
	forecast(Y_df_test)

fc <- fc %>%
	as_tibble() %>%
	select(-y) %>%
	left_join(Y_df_test, by=c(key, 'ds'))

errors <- fc %>%
	mutate(error = (y - .mean) ** 2) %>%
	group_by_at(c(key, '.model')) %>%
	summarise(rmse = sqrt(mean(error))) %>%
	ungroup()

naive_errors <- errors %>%
	filter(.model == 'naive') %>%
	select(-.model) %>%
	rename(naive_rmse = rmse)

errors <- errors %>%
	filter(.model != 'naive') %>%
	left_join(naive_errors, by=key) %>%
	group_by(.model) %>%
	summarise(rmsse = mean(rmse / naive_rmse))

write_csv(errors, 
	  str_glue('./results/{group}/fable.csv'))
tibble(group = group, 
       time = difftime(end, start, units='secs')) %>%
	write_csv(str_glue('./results/{group}/fable-time.csv'))


