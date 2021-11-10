variable "my_access_key" {
  description = "Access-key-for-AWS"
  default = "no_access_key_value_found"
}

variable "my_secret_key" {
  description = "Secret-key-for-AWS"
  default = "no_secret_key_value_found"
}

variable "my_key_name_pem" {
  description = "Key-name-pem-AWS"
  default = "no_secret_key_value_found"
}

provider "aws" {
  region = "us-east-1"
  access_key = var.my_access_key
  secret_key = var.my_secret_key
}

resource "aws_instance" "ec2_gpu" {
  ami = "ami-05e329519be512f1b"
  instance_type = "g4dn.2xlarge"
  associate_public_ip_address = true
  key_name = var.my_key_name_pem

  user_data = <<-EOL
    #!/bin/bash -xe

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
    rm -rf Miniconda3-latest-Linux-x86_64.sh
    echo 'export PATH="/miniconda/bin:$PATH"' >> /home/ubuntu/.bashrc
    PATH="/miniconda/bin:$PATH"
    conda init

  EOL

  tags = {
    Name = "EC2 GPU"
  }

  vpc_security_group_ids = [aws_security_group.ec2_gpu.id]
}

resource "aws_security_group" "ec2_gpu" {
  name = "terraform-tcp-security-group"

  ingress {
    from_port = 22
	  to_port = 22
	  protocol = "tcp"
	  cidr_blocks = ["0.0.0.0/0"]
	}

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
output "instance_id" {
  description = "ID of the EC2 instance"
  value = aws_instance.ec2_gpu.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value = aws_instance.ec2_gpu.public_ip
}
