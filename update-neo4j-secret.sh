#!/bin/bash

# Update Neo4j password in GCP Secret Manager

set -e

SERVICE_ACCOUNT_KEY="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
PROJECT_ID="electric-vision-463705-f6"
REGION="us-central1"
SERVICE_NAME="financial-analyst"

# Correct Neo4j cloud password
NEO4J_PASSWORD="wENxe9nIG0Kl8ysrxijtMEUTOoMD1HwrHqua5iOUk0o"

echo "🔐 Authenticating..."
gcloud auth activate-service-account --key-file="${SERVICE_ACCOUNT_KEY}"
gcloud config set project ${PROJECT_ID}

echo "🔑 Updating Neo4j password in Secret Manager..."
echo -n "${NEO4J_PASSWORD}" | gcloud secrets versions add neo4j-password \
    --data-file=- \
    --project=${PROJECT_ID}

echo "✅ Secret updated successfully"

echo "🔄 Triggering service redeployment to apply new credentials..."
gcloud run services update ${SERVICE_NAME} \
    --region=${REGION} \
    --update-env-vars="FORCE_REDEPLOY=$(date +%s)" \
    --project=${PROJECT_ID}

echo "✅ Service redeployed with new credentials"

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project=${PROJECT_ID})

echo ""
echo "✅ Update Complete!"
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "The Neo4j connection should now work with the updated password."