# 🚀 GCP Deployment Guide for Financial Analyst App

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed locally
3. **Docker** installed (for Cloud Run)
4. **Project ID** from GCP Console

## Option 1: Cloud Run Deployment (Recommended)

### Step 1: Initial Setup
```bash
# Install Google Cloud SDK (if not installed)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login to GCP
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Prepare Secrets
```bash
# Create secret for API key
echo -n "AIzaSyDkxK4UwL9GHdCSBoAs_m6FB1z-3degaCI" | \
  gcloud secrets create google-ai-api-key --data-file=-

# Grant Cloud Run access to the secret
gcloud secrets add-iam-policy-binding google-ai-api-key \
  --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Step 3: Deploy Using Script
```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment (replace with your project ID)
./deploy.sh YOUR_PROJECT_ID us-central1
```

### Step 4: Manual Deployment (Alternative)
```bash
# Build and push image
docker build -t gcr.io/YOUR_PROJECT_ID/financial-analyst .
docker push gcr.io/YOUR_PROJECT_ID/financial-analyst

# Deploy to Cloud Run
gcloud run deploy financial-analyst \
  --image gcr.io/YOUR_PROJECT_ID/financial-analyst \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets="GOOGLE_AI_API_KEY=google-ai-api-key:latest"
```

## Option 2: App Engine Deployment

### Step 1: Deploy to App Engine
```bash
# Deploy directly (uses app.yaml)
gcloud app deploy

# Set the API key as environment variable
gcloud app deploy --set-env-vars GOOGLE_AI_API_KEY="AIzaSyDkxK4UwL9GHdCSBoAs_m6FB1z-3degaCI"
```

## Option 3: Compute Engine Deployment

### Step 1: Create VM Instance
```bash
gcloud compute instances create financial-analyst-vm \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --tags=http-server,https-server
```

### Step 2: SSH and Install
```bash
# SSH into instance
gcloud compute ssh financial-analyst-vm

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git

# Clone your repo
git clone https://github.com/YOUR_USERNAME/fl_financial_advisor.git
cd fl_financial_advisor

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_AI_API_KEY=AIzaSyDkxK4UwL9GHdCSBoAs_m6FB1z-3degaCI" > .env

# Run with systemd (create service file)
sudo tee /etc/systemd/system/financial-analyst.service > /dev/null <<EOF
[Unit]
Description=Financial Analyst Streamlit App
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/fl_financial_advisor
Environment="PATH=/home/YOUR_USERNAME/fl_financial_advisor/venv/bin"
ExecStart=/home/YOUR_USERNAME/fl_financial_advisor/venv/bin/streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl enable financial-analyst
sudo systemctl start financial-analyst
```

## CI/CD with Cloud Build

### Automatic Deployment on Git Push
1. Connect your GitHub repo to Cloud Build
2. Cloud Build will use `cloudbuild.yaml` automatically
3. Every push to main will trigger deployment

### Manual Trigger
```bash
gcloud builds submit --config cloudbuild.yaml
```

## Cost Optimization Tips

1. **Cloud Run**
   - Set min instances to 0 (scale to zero)
   - Use concurrency settings
   - Set appropriate memory/CPU limits

2. **App Engine**
   - Use automatic scaling
   - Set min_instances to 0
   - Use F1 instance class for testing

3. **Compute Engine**
   - Use preemptible instances for dev/test
   - Set up instance scheduling
   - Use committed use discounts

## Monitoring & Logging

### View Logs
```bash
# Cloud Run logs
gcloud run logs read --service=financial-analyst

# App Engine logs
gcloud app logs read

# Compute Engine logs
gcloud compute ssh financial-analyst-vm --command="journalctl -u financial-analyst -f"
```

### Set Up Monitoring
1. Enable Cloud Monitoring API
2. Create uptime checks
3. Set up alerts for errors/downtime

## Security Best Practices

1. **Never commit API keys** - Always use Secret Manager
2. **Enable IAP** for internal apps
3. **Use VPC** for Compute Engine
4. **Set up Cloud Armor** for DDoS protection
5. **Enable audit logs**

## Domain Setup (Optional)

### Map Custom Domain
```bash
# For Cloud Run
gcloud run domain-mappings create \
  --service=financial-analyst \
  --domain=your-domain.com \
  --region=us-central1

# For App Engine
gcloud app domain-mappings create your-domain.com
```

## Troubleshooting

### Common Issues

1. **"Permission denied" errors**
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
     --role="roles/run.admin"
   ```

2. **Memory issues**
   - Increase memory in deployment config
   - Optimize Streamlit caching

3. **Slow startup**
   - Use smaller base images
   - Optimize requirements.txt
   - Enable Cloud Run CPU boost

## Estimated Costs

### Cloud Run (Recommended)
- **Free tier**: 2 million requests/month
- **Estimated**: $5-20/month for moderate usage
- **Pay per use**: Only charged when running

### App Engine
- **Free tier**: 28 instance hours/day
- **Estimated**: $10-50/month
- **Always some baseline cost**

### Compute Engine
- **e2-micro**: ~$6/month (always on)
- **e2-medium**: ~$24/month (always on)
- **Can use preemptible**: 80% cheaper

## Next Steps

1. Set up Cloud Build for CI/CD
2. Configure monitoring and alerts
3. Set up backup strategies
4. Implement rate limiting
5. Add authentication if needed