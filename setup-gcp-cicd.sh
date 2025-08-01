#!/bin/bash

# Setup script for GCP CI/CD pipeline using service account
# This script configures Cloud Build, Secret Manager, and Cloud Run

set -e

# Configuration
SERVICE_ACCOUNT_KEY="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
PROJECT_ID="electric-vision-463705-f6"  # Extracted from service account filename
REGION="us-central1"
SERVICE_NAME="financial-analyst"
REPO_NAME="fl-financial-advisor"
GITHUB_OWNER="sathishm7432"  # Update with actual GitHub username

echo "🚀 Setting up GCP CI/CD pipeline..."
echo "Project ID: ${PROJECT_ID}"
echo "Service Account: ${SERVICE_ACCOUNT_KEY}"

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
    cloudresourcemanager.googleapis.com

# Create Google AI API key secret
echo "🔑 Setting up Secret Manager..."
if ! gcloud secrets describe google-ai-api-key --project=${PROJECT_ID} &> /dev/null; then
    echo "Creating secret for Google AI API key..."
    echo -n "AIzaSyDkxK4UwL9GHdCSBoAs_m6FB1z-3degaCI" | gcloud secrets create google-ai-api-key \
        --data-file=- \
        --project=${PROJECT_ID}
    echo "✅ Secret created successfully"
else
    echo "Secret 'google-ai-api-key' already exists"
    # Update the secret with new value
    echo -n "AIzaSyDkxK4UwL9GHdCSBoAs_m6FB1z-3degaCI" | gcloud secrets versions add google-ai-api-key \
        --data-file=- \
        --project=${PROJECT_ID}
    echo "✅ Secret updated successfully"
fi

# Grant Cloud Build service account necessary permissions
echo "🔒 Setting up IAM permissions..."
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

# Grant Cloud Run Admin permission to Cloud Build
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/run.admin"

# Grant Service Account User permission
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/iam.serviceAccountUser"

# Grant Secret Manager access
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/secretmanager.secretAccessor"

# Grant Container Registry access
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/storage.admin"

echo "✅ IAM permissions configured"

# Create Cloud Build trigger for GitHub
echo "📦 Setting up Cloud Build trigger..."

# First, check if repository is already connected
REPO_CONNECTION=$(gcloud alpha builds repositories list --filter="name:${REPO_NAME}" --format="value(name)" 2>/dev/null || echo "")

if [ -z "$REPO_CONNECTION" ]; then
    echo "⚠️  GitHub repository not connected to Cloud Build"
    echo "Please follow these steps:"
    echo "1. Go to https://console.cloud.google.com/cloud-build/triggers?project=${PROJECT_ID}"
    echo "2. Click 'Connect Repository'"
    echo "3. Select GitHub and authenticate"
    echo "4. Select repository: ${GITHUB_OWNER}/${REPO_NAME}"
    echo "5. Run this script again after connecting"
else
    # Create trigger if it doesn't exist
    TRIGGER_NAME="deploy-financial-analyst"
    
    if ! gcloud builds triggers describe ${TRIGGER_NAME} --project=${PROJECT_ID} &> /dev/null; then
        echo "Creating Cloud Build trigger..."
        gcloud builds triggers create github \
            --repo-name="${REPO_NAME}" \
            --repo-owner="${GITHUB_OWNER}" \
            --branch-pattern="^main$" \
            --build-config="cloudbuild.yaml" \
            --name="${TRIGGER_NAME}" \
            --project=${PROJECT_ID}
        echo "✅ Cloud Build trigger created"
    else
        echo "Cloud Build trigger already exists"
    fi
fi

# Create initial Cloud Run service (if doesn't exist)
echo "🏃 Creating Cloud Run service..."
if ! gcloud run services describe ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID} &> /dev/null; then
    # Build and deploy initial image
    echo "Building initial Docker image..."
    docker build -t gcr.io/${PROJECT_ID}/${SERVICE_NAME}:initial .
    
    echo "Pushing initial image..."
    gcloud auth configure-docker
    docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}:initial
    
    echo "Deploying initial Cloud Run service..."
    gcloud run deploy ${SERVICE_NAME} \
        --image gcr.io/${PROJECT_ID}/${SERVICE_NAME}:initial \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10 \
        --min-instances 0 \
        --set-env-vars "STREAMLIT_SERVER_HEADLESS=true" \
        --set-secrets "GOOGLE_AI_API_KEY=google-ai-api-key:latest" \
        --project=${PROJECT_ID}
else
    echo "Cloud Run service already exists"
fi

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project=${PROJECT_ID})

echo ""
echo "✅ GCP CI/CD Setup Complete!"
echo "================================"
echo "Project ID: ${PROJECT_ID}"
echo "Cloud Run Service: ${SERVICE_NAME}"
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "📋 Next Steps:"
echo "1. Connect GitHub repository in Cloud Console (if not done)"
echo "2. Push changes to main branch to trigger deployment"
echo "3. Monitor builds at: https://console.cloud.google.com/cloud-build/builds?project=${PROJECT_ID}"
echo "4. View logs: gcloud run logs read --service=${SERVICE_NAME} --region=${REGION}"
echo ""
echo "🔧 Manual deployment command:"
echo "gcloud builds submit --config cloudbuild.yaml --project=${PROJECT_ID}"