#!/bin/bash

# Build Docker image using gcloud
# This script builds the image using Cloud Build instead of local Docker

set -e

# Configuration
SERVICE_ACCOUNT_KEY="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
PROJECT_ID="electric-vision-463705-f6"
SERVICE_NAME="financial-analyst"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🏗️ Building Docker image using Cloud Build..."
echo "Project: ${PROJECT_ID}"
echo "Image: ${IMAGE_NAME}"

# Authenticate with service account
echo "🔐 Authenticating with service account..."
gcloud auth activate-service-account --key-file="${SERVICE_ACCOUNT_KEY}"
gcloud config set project ${PROJECT_ID}

# Get current timestamp for tagging
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")

echo "📦 Submitting build to Cloud Build..."
echo "Tags: ${COMMIT_SHA}, ${TIMESTAMP}, latest"

# Submit build to Cloud Build
# This builds the image remotely without needing local Docker
gcloud builds submit \
    --tag "${IMAGE_NAME}:${COMMIT_SHA}" \
    --timeout=20m \
    --machine-type=E2_HIGHCPU_8 \
    --project=${PROJECT_ID} \
    .

# Tag additional versions
echo "🏷️ Adding additional tags..."
gcloud container images add-tag \
    "${IMAGE_NAME}:${COMMIT_SHA}" \
    "${IMAGE_NAME}:${TIMESTAMP}" \
    --quiet

gcloud container images add-tag \
    "${IMAGE_NAME}:${COMMIT_SHA}" \
    "${IMAGE_NAME}:latest" \
    --quiet

echo "✅ Build complete!"
echo ""
echo "📋 Image details:"
echo "Repository: ${IMAGE_NAME}"
echo "Tags:"
echo "  - ${COMMIT_SHA}"
echo "  - ${TIMESTAMP}"
echo "  - latest"
echo ""
echo "🔍 View image details:"
echo "gcloud container images list-tags ${IMAGE_NAME}"
echo ""
echo "🚀 Deploy to Cloud Run:"
echo "gcloud run deploy ${SERVICE_NAME} --image ${IMAGE_NAME}:latest --region us-central1"