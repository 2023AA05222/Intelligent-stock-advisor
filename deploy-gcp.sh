#!/bin/bash

# Deployment script for GCP Cloud Run
# This script sets up secrets and deploys the application

set -e

# Configuration
SERVICE_ACCOUNT_KEY="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
PROJECT_ID="electric-vision-463705-f6"
REGION="us-central1"
SERVICE_NAME="financial-analyst"

# API Keys (from .env file)
GOOGLE_AI_API_KEY=""
OPENAI_API_KEY=""
NEO4J_PASSWORD="financialpass"

echo "🚀 Deploying Financial Analyst to GCP..."
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"

# Authenticate with service account
echo "🔐 Authenticating with service account..."
gcloud auth activate-service-account --key-file="${SERVICE_ACCOUNT_KEY}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com \
    cloudresourcemanager.googleapis.com \
    artifactregistry.googleapis.com

# Create or update secrets in Secret Manager
echo "🔑 Setting up secrets in Secret Manager..."

# Function to create or update secret
create_or_update_secret() {
    SECRET_NAME=$1
    SECRET_VALUE=$2
    
    if ! gcloud secrets describe ${SECRET_NAME} --project=${PROJECT_ID} &> /dev/null; then
        echo "Creating secret: ${SECRET_NAME}"
        echo -n "${SECRET_VALUE}" | gcloud secrets create ${SECRET_NAME} \
            --data-file=- \
            --project=${PROJECT_ID}
    else
        echo "Updating secret: ${SECRET_NAME}"
        echo -n "${SECRET_VALUE}" | gcloud secrets versions add ${SECRET_NAME} \
            --data-file=- \
            --project=${PROJECT_ID}
    fi
}

# Create/update all secrets
create_or_update_secret "google-ai-api-key" "${GOOGLE_AI_API_KEY}"
create_or_update_secret "openai-api-key" "${OPENAI_API_KEY}"
create_or_update_secret "neo4j-password" "${NEO4J_PASSWORD}"

echo "✅ Secrets configured"

# Grant Cloud Run service account access to secrets
echo "🔒 Setting up IAM permissions..."
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
CLOUD_RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Grant Secret Manager access to Cloud Run service account
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${CLOUD_RUN_SA}" \
    --role="roles/secretmanager.secretAccessor"

echo "✅ IAM permissions configured"

# Build and push Docker image
echo "📦 Building Docker image..."
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

# Configure Docker for GCR
gcloud auth configure-docker

# Build image
docker build -t ${IMAGE_NAME} .

echo "⬆️ Pushing Docker image to GCR..."
docker push ${IMAGE_NAME}

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 10 \
    --min-instances 0 \
    --timeout 3600 \
    --set-env-vars "STREAMLIT_SERVER_HEADLESS=true,STREAMLIT_BROWSER_GATHER_USAGE_STATS=false,TOKENIZERS_PARALLELISM=false" \
    --set-secrets "GOOGLE_AI_API_KEY=google-ai-api-key:latest,OPENAI_API_KEY=openai-api-key:latest,NEO4J_PASSWORD=neo4j-password:latest" \
    --project=${PROJECT_ID}

# Get service URL
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
