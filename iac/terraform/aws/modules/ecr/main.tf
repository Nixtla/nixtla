variable "prefix" {}
variable "region" {}

resource "aws_ecr_repository" "tsbenchmarks_ecr" {
  name                 = "tsbenchmarks"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  provisioner "local-exec" {
    working_dir = "${path.module}"
    command     = "/bin/bash docker_images.sh"

    environment = {
      SRC_DOCKER_IMAGE  = aws_ecr_repository.tspreprocess_ecr.registry_id
      ROUTE             = "tsbenchmarks"
      AWS_REGION        = var.region
    }
  }
}

resource "aws_ecr_repository" "tsfeatures_ecr" {
  name                 = "tsfeatures"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  provisioner "local-exec" {
    working_dir = "${path.module}"
    command     = "/bin/bash docker_images.sh"

    environment = {
      SRC_DOCKER_IMAGE  = aws_ecr_repository.tspreprocess_ecr.registry_id
      ROUTE             = "tsfeatures"
      AWS_REGION        = var.region
    }
  }
  
}

resource "aws_ecr_repository" "tsforecast_ecr" {
  name                 = "tsforecast"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  provisioner "local-exec" {
    working_dir = "${path.module}"
    command     = "/bin/bash docker_images.sh"

    environment = {
      SRC_DOCKER_IMAGE  = aws_ecr_repository.tspreprocess_ecr.registry_id
      ROUTE             = "tsforecast"
      AWS_REGION        = var.region
    }
  }
}

resource "aws_ecr_repository" "tspreprocess_ecr" {
  name                 = "tspreprocess"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  provisioner "local-exec" {
    working_dir = "${path.module}"
    command     = "/bin/bash docker_images.sh"

    environment = {
      SRC_DOCKER_IMAGE  = aws_ecr_repository.tspreprocess_ecr.registry_id
      ROUTE             = "tspreprocess"
      AWS_REGION        = var.region
    }
  }
}