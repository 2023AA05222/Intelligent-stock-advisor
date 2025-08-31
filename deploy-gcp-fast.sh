#!/bin/bash

# Fast deployment script for GCP Cloud Run using pre-built image
# This uses gcloud builds submit for faster deployment

set -e

# Configuration
SERVICE_ACCOUNT_KEY="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
PROJECT_ID="electric-vision-463705-f6"
REGION="us-central1"
SERVICE_NAME="financial-analyst"

echo "🚀 Fast Deployment to GCP Cloud Run..."
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"

# Authenticate with service account
echo "🔐 Authenticating with service account..."
gcloud auth activate-service-account --key-file="${SERVICE_ACCOUNT_KEY}"
gcloud config set project ${PROJECT_ID}

# Submit build to Cloud Build (faster than local Docker build)
echo "📦 Submitting build to Cloud Build..."
gcloud builds submit \
    --config cloudbuild.yaml \
    --project=${PROJECT_ID} \
    --timeout=30m

# Get service URL
echo "🔍 Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project=${PROJECT_ID})

echo ""
echo "✅ Deployment Complete!"
echo "================================"
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "📋 Useful commands:"
echo "View logs: gcloud run logs read --service=${SERVICE_NAME} --region=${REGION}"
echo "Check status: gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
echo "Stream logs: gcloud run logs tail --service=${SERVICE_NAME} --region=${REGION}"
echo ""
echo "🌐 Your app is now live at: ${SERVICE_URL}"