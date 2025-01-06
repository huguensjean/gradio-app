#!/bin/bash
# Build and deploy the image upscaler app to Cloud Run
# Please run `gcloud auth login` first to authenticate yourself

set -xe

REPO="vertex-vision-model-garden-dockers"
IMAGE_TAG="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/cloud-run-gradio-app-base:latest"

gcloud auth configure-docker us-docker.pkg.dev

docker build -f base.Dockerfile . -t "${IMAGE_TAG}"
docker push "${IMAGE_TAG}"