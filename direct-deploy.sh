#!/bin/bash

# Direct deployment using pre-built Python image
set -e

# Configuration
SERVICE_ACCOUNT_KEY="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
PROJECT_ID="electric-vision-463705-f6"
REGION="us-central1"
SERVICE_NAME="financial-analyst"

echo "🚀 Direct Deployment to Cloud Run..."

# Authenticate
gcloud auth activate-service-account --key-file="${SERVICE_ACCOUNT_KEY}"
gcloud config set project ${PROJECT_ID}

# Use a pre-built streamlit image as base and mount our code
echo "🏗️ Deploying using Cloud Run source deploy..."

# Deploy directly from source
gcloud run deploy ${SERVICE_NAME} \
    --source . \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 5 \
    --min-instances 0 \
    --timeout 3600 \
    --set-env-vars "STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0,STREAMLIT_SERVER_HEADLESS=true,STREAMLIT_BROWSER_GATHER_USAGE_STATS=false,TOKENIZERS_PARALLELISM=false" \
    --set-secrets "GOOGLE_AI_API_KEY=google-ai-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,NEO4J_PASSWORD=neo4j-password:latest" \
    --project=${PROJECT_ID}

# Get URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project=${PROJECT_ID})

echo "✅ Deployment Complete!"
echo "Service URL: ${SERVICE_URL}"