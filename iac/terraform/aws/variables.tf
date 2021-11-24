variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region"
}

variable "prefix" {
  type        = string
  default     = "lab"
  description = "Prefix word for all services"
}

variable "instance_count" {
  type    = number
  default = 1
}

variable "instace_type" {
  type    = string
  default = "ml.t3.2xlarge"
}