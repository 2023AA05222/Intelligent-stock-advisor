#!/bin/bash

# Build Docker image using Cloud Build with cloudbuild.yaml
# This gives more control over the build process

set -e

# Configuration
SERVICE_ACCOUNT_KEY="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
PROJECT_ID="electric-vision-463705-f6"

echo "🏗️ Building Docker image using Cloud Build..."
echo "Project: ${PROJECT_ID}"
echo "Using: cloudbuild.yaml"

# Authenticate with service account
echo "🔐 Authenticating with service account..."
gcloud auth activate-service-account --key-file="${SERVICE_ACCOUNT_KEY}"
gcloud config set project ${PROJECT_ID}

# Check if we're in a git repository and get commit SHA
if git rev-parse --git-dir > /dev/null 2>&1; then
    COMMIT_SHA=$(git rev-parse --short HEAD)
    echo "📝 Git commit: ${COMMIT_SHA}"
else
    COMMIT_SHA="manual-$(date +%Y%m%d-%H%M%S)"
    echo "📝 No git repository, using timestamp: ${COMMIT_SHA}"
fi

# Submit build using cloudbuild.yaml
echo "📦 Submitting build to Cloud Build..."
gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions=COMMIT_SHA=${COMMIT_SHA} \
    --project=${PROJECT_ID} \
    .

echo ""
echo "✅ Build submitted!"
echo ""
echo "📊 Monitor build progress:"
echo "https://console.cloud.google.com/cloud-build/builds?project=${PROJECT_ID}"
echo ""
echo "🔍 View build logs:"
echo "gcloud builds log --stream \$(gcloud builds list --limit=1 --format='value(id)')"