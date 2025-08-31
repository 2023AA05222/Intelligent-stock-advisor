# 🚀 GCP Deployment & CI/CD Setup Guide

## Overview

This guide provides step-by-step instructions for deploying the Financial Analyst Streamlit app with Neo4j integration to Google Cloud Platform with automated CI/CD using Cloud Build.

## Prerequisites

1. **Google Cloud SDK** installed locally
2. **Docker** installed locally
3. **Service Account JSON** file: `/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json`
4. **GitHub repository** for the code

## Project Details

- **Project ID**: `electric-vision-463705-f6`
- **Service Name**: `financial-analyst`
- **Region**: `us-central1`
- **Service Account**: Already configured in the provided JSON file

## Quick Start

### 1. Initial Setup (One-time)

```bash
# Make scripts executable
chmod +x setup-gcp-cicd.sh deploy.sh

# Run the setup script
./setup-gcp-cicd.sh
```

This script will:
- Authenticate using the service account
- Enable required GCP APIs
- Create secrets in Secret Manager
- Set up IAM permissions
- Create initial Cloud Run service

### 2. Connect GitHub Repository

1. Go to [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers?project=electric-vision-463705-f6)
2. Click **"Connect Repository"**
3. Select **GitHub** and authenticate
4. Choose your repository: `sathishm7432/fl_financial_advisor`
5. Click **"Connect"**

### 3. Create Build Trigger

After connecting the repository, run:

```bash
# Create the build trigger
gcloud builds triggers create github \
    --repo-name="fl_financial_advisor" \
    --repo-owner="sathishm7432" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml" \
    --name="deploy-financial-analyst" \
    --project="electric-vision-463705-f6"
```

## Manual Deployment

If you need to deploy manually:

```bash
# Option 1: Using deploy script
./deploy.sh

# Option 2: Using Cloud Build
gcloud builds submit --config cloudbuild.yaml --project=electric-vision-463705-f6
```

## CI/CD Pipeline

### How it Works

1. **Push to main branch** → Triggers Cloud Build
2. **Cloud Build** → Builds Docker image
3. **Container Registry** → Stores the image
4. **Cloud Run** → Deploys the new version

### Build Process

The `cloudbuild.yaml` file defines these steps:
1. Build Docker image with commit SHA tag
2. Push image to Container Registry
3. Deploy to Cloud Run with secrets
4. Tag image as 'latest'

### Monitoring Builds

- **View builds**: [Cloud Build History](https://console.cloud.google.com/cloud-build/builds?project=electric-vision-463705-f6)
- **View logs**: 
  ```bash
  gcloud builds log <BUILD_ID> --project=electric-vision-463705-f6
  ```

## Secrets Management

The Google AI API key is stored in Secret Manager:

```bash
# View secret
gcloud secrets describe google-ai-api-key --project=electric-vision-463705-f6

# Update secret (if needed)
echo -n "NEW_API_KEY" | gcloud secrets versions add google-ai-api-key \
    --data-file=- --project=electric-vision-463705-f6
```

## Application URLs

After deployment, your app will be available at:
- **Cloud Run URL**: `https://financial-analyst-<hash>-uc.a.run.app`

To get the exact URL:
```bash
gcloud run services describe financial-analyst \
    --region=us-central1 \
    --format='value(status.url)' \
    --project=electric-vision-463705-f6
```

## Monitoring & Logs

### View Application Logs
```bash
# Stream logs
gcloud run logs tail financial-analyst \
    --region=us-central1 \
    --project=electric-vision-463705-f6

# Read recent logs
gcloud run logs read financial-analyst \
    --region=us-central1 \
    --limit=50 \
    --project=electric-vision-463705-f6
```

### View Metrics
1. Go to [Cloud Run Console](https://console.cloud.google.com/run?project=electric-vision-463705-f6)
2. Click on `financial-analyst` service
3. View metrics, logs, and revisions

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Re-authenticate
   gcloud auth activate-service-account --key-file="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
   ```

2. **Build Fails**
   - Check Cloud Build logs
   - Ensure Dockerfile is correct
   - Verify all dependencies in requirements.txt

3. **App Crashes**
   - Check Cloud Run logs
   - Verify environment variables
   - Check memory/CPU limits

4. **Secret Access Issues**
   ```bash
   # Grant secret access to Cloud Run
   gcloud secrets add-iam-policy-binding google-ai-api-key \
       --member="serviceAccount:electric-vision-463705-f6@appspot.gserviceaccount.com" \
       --role="roles/secretmanager.secretAccessor" \
       --project=electric-vision-463705-f6
   ```

## Neo4j Database Deployment

### Option 1: Neo4j AuraDB (Currently Configured)

✅ **Neo4j AuraDB Free Instance is already configured**:
- **URI**: `neo4j+s://98b288d4.databases.neo4j.io`
- **Username**: `neo4j`
- **Password**: Stored securely in Google Secret Manager as `neo4j-password`
- **Configuration**: Automatically deployed via `cloudbuild.yaml`

The Cloud Run deployment is now configured to use your Neo4j Aura instance. All graph database features will be enabled on the next deployment.

**Manual Update (if needed)**:
   ```bash
   # Update Neo4j configuration
   gcloud run services update financial-analyst \
     --set-env-vars="NEO4J_URI=neo4j+s://98b288d4.databases.neo4j.io" \
     --set-env-vars="NEO4J_USERNAME=neo4j" \
     --set-secrets="NEO4J_PASSWORD=neo4j-password:latest" \
     --region=us-central1 \
     --project=electric-vision-463705-f6
   ```

### Option 2: Self-Hosted Neo4j on GCP

1. **Create Compute Engine Instance**
   ```bash
   # Create VM for Neo4j
   gcloud compute instances create neo4j-server \
     --zone=us-central1-a \
     --machine-type=e2-standard-2 \
     --boot-disk-size=50GB \
     --boot-disk-type=pd-standard \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --project=electric-vision-463705-f6
   ```

2. **Install Neo4j on VM**
   ```bash
   # SSH to VM and install Neo4j
   gcloud compute ssh neo4j-server --zone=us-central1-a
   
   # Install Neo4j (same as local installation)
   wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
   echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
   sudo apt update && sudo apt install neo4j -y
   ```

3. **Configure Network Access**
   ```bash
   # Create firewall rule for Neo4j
   gcloud compute firewall-rules create allow-neo4j \
     --allow tcp:7474,tcp:7687 \
     --source-ranges 0.0.0.0/0 \
     --description "Allow Neo4j HTTP and Bolt" \
     --project=electric-vision-463705-f6
   ```

### Option 3: Graceful Degradation (No Neo4j)

The application is designed to work without Neo4j. If no graph database is available:
- Relationship features will be disabled
- Standard financial analysis will continue to work
- All other features remain fully functional

## Cost Optimization

1. **Cloud Run Settings**:
   - Min instances: 0 (scale to zero)
   - Max instances: 10
   - Memory: 2Gi
   - CPU: 2

2. **Estimated Monthly Costs**:
   - Cloud Run: ~$5-20 (depends on usage)
   - Container Registry: <$1
   - Secret Manager: <$1
   - Cloud Build: 120 free minutes/day
   - **Neo4j AuraDB Professional**: $65+/month (if using managed Neo4j)
   - **Self-hosted Neo4j on Compute Engine**: $20-50/month (e2-standard-2)

## Rollback Procedure

To rollback to a previous version:

```bash
# List all revisions
gcloud run revisions list --service=financial-analyst \
    --region=us-central1 \
    --project=electric-vision-463705-f6

# Rollback to specific revision
gcloud run services update-traffic financial-analyst \
    --to-revisions=<REVISION_NAME>=100 \
    --region=us-central1 \
    --project=electric-vision-463705-f6
```

## Security Best Practices

1. **Never commit secrets** - Use Secret Manager
2. **Use least privilege** - IAM roles are minimal
3. **Enable audit logs** - Track all changes
4. **Regular updates** - Keep dependencies updated

## Next Steps

1. **Custom Domain**: Map a custom domain to Cloud Run service
2. **Monitoring**: Set up alerts and uptime checks
3. **Backup**: Regular exports of configurations
4. **Performance**: Optimize Streamlit caching

## Support

- **GCP Console**: https://console.cloud.google.com/?project=electric-vision-463705-f6
- **Cloud Run Dashboard**: https://console.cloud.google.com/run?project=electric-vision-463705-f6
- **Cloud Build History**: https://console.cloud.google.com/cloud-build/builds?project=electric-vision-463705-f6