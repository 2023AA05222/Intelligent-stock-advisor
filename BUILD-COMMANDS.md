# 🏗️ Building Docker Image with gcloud

## Quick Commands

### 1. Authenticate First
```bash
gcloud auth activate-service-account --key-file="/home/rajan/CREDENTIALS/electric-vision-463705-f6-1aea2195d199.json"
gcloud config set project electric-vision-463705-f6
```

### 2. Build Image Using Cloud Build

#### Option A: Simple Build (Quickest)
```bash
# Build with automatic tagging
gcloud builds submit --tag gcr.io/electric-vision-463705-f6/financial-analyst:latest .
```

#### Option B: Build with Specific Tags
```bash
# Build with commit SHA
gcloud builds submit --tag gcr.io/electric-vision-463705-f6/financial-analyst:$(git rev-parse --short HEAD) .

# Build with timestamp
gcloud builds submit --tag gcr.io/electric-vision-463705-f6/financial-analyst:$(date +%Y%m%d-%H%M%S) .
```

#### Option C: Use Cloud Build Configuration (Recommended)
```bash
# This uses cloudbuild.yaml and handles everything
gcloud builds submit --config=cloudbuild.yaml --substitutions=COMMIT_SHA=$(git rev-parse --short HEAD) .
```

### 3. Using the Build Scripts

#### For Simple Build:
```bash
./build-image.sh
```

#### For Full CI/CD Build:
```bash
./build-with-cloudbuild.sh
```

## Build Options

### Specify Build Machine Type (Faster Builds)
```bash
gcloud builds submit \
  --tag gcr.io/electric-vision-463705-f6/financial-analyst:latest \
  --machine-type=E2_HIGHCPU_8 \
  --timeout=30m \
  .
```

### Build with Specific Dockerfile
```bash
# For MCP server
gcloud builds submit \
  --tag gcr.io/electric-vision-463705-f6/mcp-server:latest \
  --file=Dockerfile.mcp \
  .

# For unified build
gcloud builds submit \
  --tag gcr.io/electric-vision-463705-f6/financial-analyst-unified:latest \
  --file=Dockerfile.unified \
  .
```

## View Build Results

### List Recent Builds
```bash
gcloud builds list --limit=5
```

### Stream Build Logs
```bash
# Get the latest build ID and stream logs
gcloud builds log --stream $(gcloud builds list --limit=1 --format='value(id)')
```

### View Specific Build
```bash
gcloud builds describe BUILD_ID
```

## Container Registry Commands

### List Images
```bash
gcloud container images list --repository=gcr.io/electric-vision-463705-f6
```

### List Tags for Image
```bash
gcloud container images list-tags gcr.io/electric-vision-463705-f6/financial-analyst
```

### Delete Old Images
```bash
# Delete specific tag
gcloud container images delete gcr.io/electric-vision-463705-f6/financial-analyst:TAG_NAME --quiet

# Delete untagged images
gcloud container images list-tags gcr.io/electric-vision-463705-f6/financial-analyst \
  --filter='-tags:*' --format='get(digest)' | \
  xargs -I {} gcloud container images delete gcr.io/electric-vision-463705-f6/financial-analyst@{} --quiet
```

## Deploy After Building

### Deploy to Cloud Run
```bash
gcloud run deploy financial-analyst \
  --image gcr.io/electric-vision-463705-f6/financial-analyst:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-secrets "GOOGLE_AI_API_KEY=google-ai-api-key:latest"
```

## Troubleshooting

### Build Fails
1. Check Cloud Build logs
2. Ensure APIs are enabled:
   ```bash
   gcloud services enable cloudbuild.googleapis.com containerregistry.googleapis.com
   ```

### Permission Issues
```bash
# Grant Cloud Build permissions
PROJECT_NUMBER=$(gcloud projects describe electric-vision-463705-f6 --format="value(projectNumber)")
gcloud projects add-iam-policy-binding electric-vision-463705-f6 \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/storage.admin"
```

### View Build History in Console
https://console.cloud.google.com/cloud-build/builds?project=electric-vision-463705-f6