
# Functions for VN1 Forecasting Competition ---- 

read_and_prepare_data <- function(dataset){
  # Reads data in wide format and returns it in long format with columns `unique_id`, `ds`, and `y`
  url <- get_dataset_url(dataset)
  df_wide <- fread(url)
  df_wide <- df_wide |> 
    mutate(unique_id = paste0(Client, "/", Warehouse, "/", Product)) |> 
    select(c(unique_id, everything())) |> 
    select(-c(Client, Warehouse, Product))
  
  df <- pivot_longer(
    data = df_wide, 
    cols = -unique_id, 
    names_to = "ds", 
    values_to = "y"
  )
  
  if(startsWith(dataset, "winners")){
    names(df)[which(names(df) == "y")] <- dataset
  }
  
  return(df)
}

get_train_data <- function(df0, df1){
  # Merges training data from phase 0 and phase 1 and removes leading zeros 
  df <- rbind(df0, df1) |> 
    arrange(unique_id, ds)
  
  df_clean <- df |> 
    group_by(unique_id) |> 
    mutate(cumsum = cumsum(y)) |>
    filter(cumsum > 0) |> 
    select(-cumsum) |> 
    ungroup()
  
  return(df_clean)
}

vn1_competition_evaluation <- function(test, forecast, model){
  # Computes competition evaluation 
  if(!is.character(forecast$ds)){
    forecast$ds <- as.character(forecast$ds) # nixtlar returns timestamps for plotting 
  }
  
  res <- merge(forecast, test, by=c("unique_id", "ds"))
  
  res <- res |> 
    mutate(abs_err = abs(res[[model]]-res$y)) |> 
    mutate(err = res[[model]]-res$y) 
  
  abs_err = sum(res$abs_err, na.rm = TRUE)
  err = sum(res$err, na.rm = TRUE)    
  score = abs_err+abs(err)
  score = score/sum(res$y)
  score = round(score, 4)
  
  return(score)
}

get_dataset_url <- function(dataset){
  # Returns the url of the given competition dataset 
  urls <- list(
    phase0_sales = "https://www.datasource.ai/attachments/eyJpZCI6Ijk4NDYxNjE2NmZmZjM0MGRmNmE4MTczOGMyMzI2ZWI2LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMCAtIFNhbGVzLmNzdiIsInNpemUiOjEwODA0NjU0LCJtaW1lX3R5cGUiOiJ0ZXh0L2NzdiJ9fQ", 
    phase1_sales = "https://www.datasource.ai/attachments/eyJpZCI6ImM2OGQxNGNmNTJkZDQ1MTUyZTg0M2FkMDAyMjVlN2NlLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMSAtIFNhbGVzLmNzdiIsInNpemUiOjEwMTgzOTYsIm1pbWVfdHlwZSI6InRleHQvY3N2In19",
    phase2_sales = "https://www.datasource.ai/attachments/eyJpZCI6IjhlNmJmNmU3ZTlhNWQ4NTcyNGVhNTI4YjAwNTk3OWE1LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMiAtIFNhbGVzLmNzdiIsInNpemUiOjEwMTI0MzcsIm1pbWVfdHlwZSI6InRleHQvY3N2In19", 
    winners1 = "https://www.datasource.ai/attachments/eyJpZCI6IjI1NDQxYmMyMTQ3MTA0MjJhMDcyYjllODcwZjEyNmY4LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoicGhhc2UgMiBzdWJtaXNzaW9uIGV4YW1pbmUgc21vb3RoZWQgMjAyNDEwMTcgRklOQUwuY3N2Iiwic2l6ZSI6MTk5MzAzNCwibWltZV90eXBlIjoidGV4dC9jc3YifX0",
    winners2 = "https://www.datasource.ai/attachments/eyJpZCI6IjU3ODhjZTUwYTU3MTg3NjFlYzMzOWU0ZTg3MWUzNjQxLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoidm4xX3N1Ym1pc3Npb25fanVzdGluX2Z1cmxvdHRlLmNzdiIsInNpemUiOjM5MDkzNzksIm1pbWVfdHlwZSI6InRleHQvY3N2In19",
    winners3 = "https://www.datasource.ai/attachments/eyJpZCI6ImE5NzcwNTZhMzhhMTc2ZWJjODFkMDMwMTM2Y2U2MTdlLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiYXJzYW5pa3phZF9zdWIuY3N2Iiwic2l6ZSI6Mzg4OTcyNCwibWltZV90eXBlIjoidGV4dC9jc3YifX0",
    winners4 = "https://www.datasource.ai/attachments/eyJpZCI6ImVlZmUxYWY2NDFjOWMwM2IxMzRhZTc2MzI1Nzg3NzIxLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiVEZUX3R1bmVkX1YyX3NlZWRfNDIuY3N2Iiwic2l6ZSI6NjA3NDgzLCJtaW1lX3R5cGUiOiJ0ZXh0L2NzdiJ9fQ",
    winners5 = "https://www.datasource.ai/attachments/eyJpZCI6IjMwMDEwMmY3NTNhMzlhN2YxNTk3ODYxZTI1N2Q2NzRmLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiZGl2aW5lb3B0aW1pemVkd2VpZ2h0c2Vuc2VtYmxlLmNzdiIsInNpemUiOjE3OTU0NzgsIm1pbWVfdHlwZSI6InRleHQvY3N2In19"
  )
  
  return(urls[[dataset]])
}

