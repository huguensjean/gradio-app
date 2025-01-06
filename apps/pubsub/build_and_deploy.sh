#!/bin/bash
# Build and deploy the pubsub app to Cloud Run.
# Run `gcloud auth login` first to authenticate yourself

set -x

PROJECT="automl-migration-test"
REGION="us-central1"
REPO="vertex-vision-model-garden-dockers"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/cloud-run-gradio-app-pubsub:${USER}-test"

gcloud auth configure-docker ${REGION}-docker.pkg.dev

docker build -f ../docker-base/cloud_run.Dockerfile . -t "${IMAGE_TAG}"
docker push "${IMAGE_TAG}"

# Deploy to Cloud Run
gcloud run deploy app-pubsub \
    --port 7860 \
    --image=${IMAGE_TAG} \
    --project=${PROJECT} \
    --region=${REGION} \
    --platform=managed \
    --allow-unauthenticated \
    --memory=1024Mi \
    --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=keyfile.json