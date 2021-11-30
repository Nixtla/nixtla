output "invoke_url" {
    value   = aws_api_gateway_stage.apigateway_stage.invoke_url
}

output "api_key" {
    value   = aws_api_gateway_api_key.apigateway_api_key
}