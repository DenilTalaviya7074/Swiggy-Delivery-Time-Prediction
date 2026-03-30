#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
# CHANGE 1: Replace YOUR_12_DIGIT_AWS_ID with your actual AWS Account ID
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 691537867280.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
# CHANGE 2: Replace YOUR_12_DIGIT_AWS_ID with your actual AWS Account ID
docker pull 691537867280.dkr.ecr.ap-south-1.amazonaws.com/food_delivery_time_prediction:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=delivery_time_pred)" ]; then
    echo "Stopping existing container..."
    docker stop delivery_time_pred
fi

if [ "$(docker ps -aq -f name=delivery_time_pred)" ]; then
    echo "Removing existing container..."
    docker rm delivery_time_pred
fi

echo "Starting new container..."
# CHANGE 3: Replace YOUR_DAGSHUB_TOKEN with your actual DagsHub Token
# CHANGE 4: Replace YOUR_12_DIGIT_AWS_ID with your actual AWS Account ID
docker run -d -p 80:8000 --name delivery_time_pred -e DAGSHUB_USER_TOKEN=73ca3a03d48870387ac1298f9e5d4a1f643110f0 691537867280.dkr.ecr.ap-south-1.amazonaws.com/food_delivery_time_prediction:latest

echo "Container started successfully."