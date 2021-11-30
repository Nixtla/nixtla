output "arn_repositories" {
    value = [ 
        aws_ecr_repository.tsbenchmarks_ecr.arn,
        aws_ecr_repository.tsfeatures_ecr.arn,
        aws_ecr_repository.tsforecast_ecr.arn,
        aws_ecr_repository.tspreprocess_ecr.arn
    ]
}

output "tsbenchmarks_repository_url" {
    value = aws_ecr_repository.tsbenchmarks_ecr.repository_url
}

output "tsfeatures_repository_url" {
    value = aws_ecr_repository.tsfeatures_ecr.repository_url
}

output "tsforecast_repository_url" {
    value = aws_ecr_repository.tsforecast_ecr.repository_url
}

output "tspreprocess_repository_url" {
    value = aws_ecr_repository.tspreprocess_ecr.repository_url
}

output "registry_id" {
    value = aws_ecr_repository.tsbenchmarks_ecr.registry_id
}