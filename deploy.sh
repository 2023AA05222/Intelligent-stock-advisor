#!/bin/bash

# Deploy script for Google Cloud Run using service account
# Usage: ./deploy.sh

set -e

# Configuration
SERVICE_ACCOUNT_KEY="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
PROJECT_ID="electric-vision-463705-f6"
REGION="us-central1"
SERVICE_NAME="financial-analyst"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Starting deployment to Google Cloud Run..."
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Authenticate with service account
echo "🔐 Authenticating with service account..."
gcloud auth activate-service-account --key-file="${SERVICE_ACCOUNT_KEY}"

# Set the project
echo "📋 Setting project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Build the Docker image
echo "🏗️ Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Configure Docker authentication
echo "🔐 Configuring Docker authentication..."
gcloud auth configure-docker

# Push the image
echo "📤 Pushing image to Container Registry..."
docker push ${IMAGE_NAME}

# Create secret for API key (if not exists)
echo "🔑 Managing secrets..."
if ! gcloud secrets describe google-ai-api-key --project=${PROJECT_ID} &> /dev/null; then
    echo "Creating secret for Google AI API key..."
    echo -n "AIzaSyDkxK4UwL9GHdCSBoAs_m6FB1z-3degaCI" | gcloud secrets create google-ai-api-key \
        --data-file=- \
        --project=${PROJECT_ID}
else
    echo "Secret 'google-ai-api-key' already exists"
    # Update the secret with new value
    echo -n "AIzaSyDkxK4UwL9GHdCSBoAs_m6FB1z-3degaCI" | gcloud secrets versions add google-ai-api-key \
        --data-file=- \
        --project=${PROJECT_ID}
fi

# Create Neo4j secrets (if environment variables are set)
if [ ! -z "${NEO4J_URI}" ]; then
    echo "Creating/updating Neo4j URI secret..."
    if ! gcloud secrets describe neo4j-uri --project=${PROJECT_ID} &> /dev/null; then
        echo -n "${NEO4J_URI}" | gcloud secrets create neo4j-uri \
            --data-file=- \
            --project=${PROJECT_ID}
    else
        echo -n "${NEO4J_URI}" | gcloud secrets versions add neo4j-uri \
            --data-file=- \
            --project=${PROJECT_ID}
    fi
fi

if [ ! -z "${NEO4J_USERNAME}" ]; then
    echo "Creating/updating Neo4j username secret..."
    if ! gcloud secrets describe neo4j-username --project=${PROJECT_ID} &> /dev/null; then
        echo -n "${NEO4J_USERNAME}" | gcloud secrets create neo4j-username \
            --data-file=- \
            --project=${PROJECT_ID}
    else
        echo -n "${NEO4J_USERNAME}" | gcloud secrets versions add neo4j-username \
            --data-file=- \
            --project=${PROJECT_ID}
    fi
fi

if [ ! -z "${NEO4J_PASSWORD}" ]; then
    echo "Creating/updating Neo4j password secret..."
    if ! gcloud secrets describe neo4j-password --project=${PROJECT_ID} &> /dev/null; then
        echo -n "${NEO4J_PASSWORD}" | gcloud secrets create neo4j-password \
            --data-file=- \
            --project=${PROJECT_ID}
    else
        echo -n "${NEO4J_PASSWORD}" | gcloud secrets versions add neo4j-password \
            --data-file=- \
            --project=${PROJECT_ID}
    fi
fi

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars="STREAMLIT_SERVER_HEADLESS=true,STREAMLIT_BROWSER_GATHER_USAGE_STATS=false,TOKENIZERS_PARALLELISM=false,NEO4J_URI=neo4j+s://98b288d4.databases.neo4j.io,NEO4J_USERNAME=neo4j" \
    --set-secrets="GOOGLE_AI_API_KEY=google-ai-api-key:latest,NEO4J_PASSWORD=neo4j-password:latest" \
    --project=${PROJECT_ID}

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project=${PROJECT_ID})

echo "✅ Deployment complete!"
echo "🌐 Your app is available at: ${SERVICE_URL}"
echo ""
echo "📊 Next steps:"
echo "1. Visit ${SERVICE_URL} to access your Financial Analyst app"
echo "2. Monitor logs: gcloud run logs read --service=${SERVICE_NAME}"
echo "3. View metrics in Cloud Console: https://console.cloud.google.com/run"