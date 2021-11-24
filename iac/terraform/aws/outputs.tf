output "api_key" {
  value = module.apigateway.api_key
}

output "benchmarks_invoke_url" {
  value = "${module.apigateway.invoke_url}/tsbenchmarks"
}

output "tsfeatures_invoke_url" {
  value = "${module.apigateway.invoke_url}/tsfeatures"
}

output "tsforecast_invoke_url" {
  value = "${module.apigateway.invoke_url}/tsforecast"
}