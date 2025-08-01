# Deployment Guide

## Overview

This guide covers deployment strategies and infrastructure setup for the Financial Advisor Application with Neo4j graph database integration across different environments.

## Table of Contents
1. [Local Development](#local-development)
2. [Neo4j Database Setup](#neo4j-database-setup)
3. [Docker Deployment](#docker-deployment)
4. [Google Cloud Run](#google-cloud-run)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Environment Configuration](#environment-configuration)
7. [Monitoring & Observability](#monitoring--observability)
8. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites
- Python 3.11+
- Git
- Make (optional, for convenience)
- **Neo4j 5.0+** (optional, but recommended for full graph features)
- Docker (for containerized Neo4j deployment)

### Setup Instructions
```bash
# Clone repository
git clone <repository-url>
cd fl_financial_advisor

# Install dependencies
make install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and Neo4j configuration

# Start Neo4j (optional, for graph features)
make docker-neo4j

# Run locally
make run-streamlit
```

### Environment Variables
```bash
# Required for AI features
GOOGLE_AI_API_KEY=your_gemini_api_key

# Neo4j Configuration (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=financialpass
```

### Environment Variables
```bash
# .env file
GOOGLE_AI_API_KEY=your_gemini_api_key_here
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Development Workflow
```bash
# Install development tools
make install-dev

# Run quality checks
make check  # format, lint, type-check

# Run tests
make test

# Watch for changes
make watch
```

---

## Neo4j Database Setup

### Option 1: Docker Deployment (Recommended for Development)

```bash
# Start Neo4j with Docker Compose
make docker-neo4j

# Start full stack (Neo4j + Application)
make docker-up

# Access Neo4j Browser
open http://localhost:7474
# Username: neo4j, Password: financialpass
```

### Option 2: Local Installation

#### Ubuntu/Debian
```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt update
sudo apt install neo4j

# Start and enable service
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Set initial password
cypher-shell -u neo4j -p neo4j
> ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'financialpass';
```

#### macOS
```bash
# Install with Homebrew
brew install neo4j

# Start Neo4j
brew services start neo4j

# Set password via browser
open http://localhost:7474
```

### Option 3: Neo4j Desktop
1. Download from https://neo4j.com/download/
2. Create a new project and database
3. Set password to `financialpass`
4. Start the database
5. Note the connection URI (usually `bolt://localhost:7687`)

### Verifying Installation
```bash
# Test connection
cypher-shell -u neo4j -p financialpass "RETURN 'Neo4j is working!' as message"

# Test from application
make docker-test-graph
```

---

## Docker Deployment

### Building the Image
```bash
# Build with integrated RAG system and Neo4j support
make docker-build

# Or manually
docker build -t financial-advisor:latest .
```

### Running with Docker Compose (Recommended)
```bash
# Start full stack (Neo4j + Application)
make docker-up

# View logs
make docker-logs

# Stop all services
make docker-down
```

### Running Individual Containers
```bash
# Start Neo4j only
make docker-neo4j

# Run application container
make docker-run

# Or manually
docker run -p 8080:8080 \
  -e GOOGLE_AI_API_KEY="your-api-key" \
  --name financial-advisor \
  financial-advisor:latest
```

### Docker Compose (Optional)
```yaml
# docker-compose.yml
version: '3.8'
services:
  financial-advisor:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GOOGLE_AI_API_KEY=${GOOGLE_AI_API_KEY}
      - STREAMLIT_SERVER_HEADLESS=true
    volumes:
      - rag_cache:/app/rag_cache
    restart: unless-stopped

volumes:
  rag_cache:
```

### Container Specifications
```dockerfile
Base Image: python:3.11-slim
Port: 8080
Memory Limit: 4Gi
CPU Limit: 2 cores
User: Non-root (streamlit:1000)
```

---

## Google Cloud Run

### Architecture Overview
```
Internet → Cloud Load Balancer → Cloud Run → Container Registry
                                      ↓
                              Secret Manager ← Service Account
```

### Prerequisites
- Google Cloud Project
- gcloud CLI installed
- Docker installed
- Billing enabled

### Quick Deployment
```bash
# One-time setup
make gcp-setup

# Deploy to Cloud Run
make gcp-deploy

# Or use Cloud Build
make gcp-build
```

### Manual Deployment Steps

#### 1. Set up Authentication
```bash
# Using service account key
gcloud auth activate-service-account \
  --key-file="/path/to/service-account.json"

# Set project
gcloud config set project electric-vision-463705-f6
```

#### 2. Create Secrets
```bash
# Store API key in Secret Manager
echo -n "your-gemini-api-key" | gcloud secrets create google-ai-api-key \
  --data-file=- \
  --project=electric-vision-463705-f6
```

#### 3. Build and Push Image
```bash
# Build image
gcloud builds submit --tag gcr.io/electric-vision-463705-f6/financial-analyst

# Or using local build
docker build -t gcr.io/electric-vision-463705-f6/financial-analyst .
docker push gcr.io/electric-vision-463705-f6/financial-analyst
```

#### 4. Deploy to Cloud Run
```bash
gcloud run deploy financial-analyst \
  --image gcr.io/electric-vision-463705-f6/financial-analyst \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 10 \
  --execution-environment gen2 \
  --no-cpu-throttling \
  --set-env-vars "STREAMLIT_SERVER_HEADLESS=true,STREAMLIT_BROWSER_GATHER_USAGE_STATS=false,TOKENIZERS_PARALLELISM=false" \
  --set-secrets "GOOGLE_AI_API_KEY=google-ai-api-key:latest" \
  --project electric-vision-463705-f6
```

### Resource Configuration
```yaml
Cloud Run Service:
  CPU: 2 cores
  Memory: 4Gi
  Timeout: 300s
  Concurrency: 10
  Min Instances: 0
  Max Instances: 10
  Execution Environment: gen2
  CPU Throttling: Disabled
```

### Custom Domain Setup (Optional)
```bash
# Map custom domain
gcloud run domain-mappings create \
  --service financial-analyst \
  --domain yourdomain.com \
  --region us-central1 \
  --project electric-vision-463705-f6
```

---

## CI/CD Pipeline

### GitHub Actions (Alternative)
```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - id: 'auth'
      uses: 'google-github-actions/auth@v1'
      with:
        credentials_json: '${{ secrets.GCP_SA_KEY }}'
    
    - name: 'Set up Cloud SDK'
      uses: 'google-github-actions/setup-gcloud@v1'
    
    - name: 'Build and Deploy'
      run: |
        gcloud builds submit --config cloudbuild.yaml
```

### Cloud Build Trigger
```bash
# Create trigger for main branch
gcloud builds triggers create github \
  --repo-name="fl_financial_advisor" \
  --repo-owner="your-username" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml" \
  --name="deploy-financial-analyst" \
  --project="electric-vision-463705-f6"
```

### Build Configuration (`cloudbuild.yaml`)
```yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/financial-analyst:$COMMIT_SHA', '.']
    timeout: '1200s'
  
  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/financial-analyst:$COMMIT_SHA']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'financial-analyst'
      - '--image'
      - 'gcr.io/$PROJECT_ID/financial-analyst:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'

options:
  machineType: 'E2_HIGHCPU_8'
  diskSizeGb: 100
  logging: CLOUD_LOGGING_ONLY

timeout: '1800s'
```

---

## Environment Configuration

### Environment Variables by Environment

#### Development
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_SERVER_HEADLESS=false
STREAMLIT_BROWSER_GATHER_USAGE_STATS=true
LOG_LEVEL=DEBUG
```

#### Staging
```bash
STREAMLIT_SERVER_PORT=8080
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
LOG_LEVEL=INFO
TOKENIZERS_PARALLELISM=false
```

#### Production
```bash
STREAMLIT_SERVER_PORT=8080
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
TOKENIZERS_PARALLELISM=false
LOG_LEVEL=WARNING
```

### Secret Management

#### Development
- Use `.env` file
- Never commit secrets to git

#### Production
- Google Secret Manager
- Environment variable injection
- Service account authentication

```bash
# Create secret
gcloud secrets create google-ai-api-key --data-file=-

# Grant access to service account
gcloud secrets add-iam-policy-binding google-ai-api-key \
  --member="serviceAccount:electric-vision-463705-f6@appspot.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Monitoring & Observability

### Health Checks
```python
# Streamlit health endpoint
GET /_stcore/health
Response: 200 OK
```

### Logging Configuration
```python
import logging
import json

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(self.JsonFormatter())
        self.logger.addHandler(handler)
    
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                'timestamp': record.created,
                'level': record.levelname,
                'message': record.getMessage(),
                'component': getattr(record, 'component', 'unknown')
            })
```

### Metrics Collection
```bash
# View Cloud Run metrics
gcloud run services describe financial-analyst \
  --region us-central1 \
  --project electric-vision-463705-f6

# Stream logs
gcloud run logs tail financial-analyst \
  --region us-central1 \
  --project electric-vision-463705-f6
```

### Custom Monitoring
```python
# Application metrics
METRICS = {
    'requests_total': Counter('requests_total', 'Total requests'),
    'request_duration': Histogram('request_duration_seconds', 'Request duration'),
    'rag_queries': Counter('rag_queries_total', 'RAG queries'),
    'ai_requests': Counter('ai_requests_total', 'AI requests')
}
```

---

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Symptoms
- Application crashes
- "Out of memory" errors
- Slow performance

# Solutions
- Increase memory allocation
- Optimize RAG chunk sizes
- Implement data streaming
```

#### 2. Cold Starts
```bash
# Symptoms
- Slow initial response
- Timeout errors
- Model loading delays

# Solutions
- Use gen2 execution environment
- Pre-load models in Docker image
- Set min-instances > 0
```

#### 3. API Rate Limits
```bash
# Symptoms
- 429 status codes
- "Quota exceeded" errors
- Intermittent failures

# Solutions
- Implement exponential backoff
- Add request caching
- Use multiple API keys
```

### Debugging Commands
```bash
# View service status
make gcp-status

# Check logs
make gcp-logs

# Get service URL
make gcp-url

# Test locally with production config
docker run -p 8080:8080 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  -e GOOGLE_AI_API_KEY="your-key" \
  financial-advisor:latest
```

### Performance Optimization
```bash
# Resource optimization
- Use multi-stage Docker builds
- Minimize base image size
- Cache pip dependencies
- Pre-download ML models

# Application optimization
- Implement lazy loading
- Use efficient data structures
- Optimize database queries
- Enable compression
```

### Rollback Procedures
```bash
# List all revisions
gcloud run revisions list \
  --service financial-analyst \
  --region us-central1

# Rollback to specific revision
gcloud run services update-traffic financial-analyst \
  --to-revisions REVISION_NAME=100 \
  --region us-central1
```

---

## Cost Optimization

### Resource Right-sizing
```yaml
Development:
  CPU: 1 core
  Memory: 2Gi
  Min Instances: 0
  Max Instances: 2

Production:
  CPU: 2 cores
  Memory: 4Gi
  Min Instances: 0
  Max Instances: 10
```

### Cost Monitoring
```bash
# Enable billing alerts
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Cloud Run Budget" \
  --budget-amount=50 \
  --threshold-rule=amount=40,basis=current-spend

# Monitor costs
gcloud billing budgets list \
  --billing-account=BILLING_ACCOUNT_ID
```

### Estimated Monthly Costs
- **Cloud Run**: $5-20 (based on usage)
- **Container Registry**: <$1
- **Secret Manager**: <$1
- **Cloud Build**: Free tier (120 minutes/day)
- **Total**: ~$10-25/month for moderate usage

---

## Security Considerations

### Network Security
- HTTPS enforcement
- VPC-native networking (future)
- IAM-based access control

### Application Security
- Input validation and sanitization
- Rate limiting via Cloud Run
- Secrets management via Secret Manager

### Compliance
- Data encryption in transit and at rest
- Audit logging enabled
- Regular security updates