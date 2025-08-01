# Makefile for Financial Advisor Project

# Python interpreter
PYTHON := python
PIP := pip
VENV := venv
VENV_BIN := $(VENV)/bin

# Default target
.DEFAULT_GOAL := help

# Help command
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install          - Create venv and install all dependencies (includes RAG)"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make install-models   - Download optional NLP models (spaCy)"
	@echo "  make run-mcp          - Run the MCP financial server"
	@echo "  make run-streamlit    - Run the Streamlit web app"
	@echo "  make test             - Run all tests"
	@echo "  make format           - Format code with black"
	@echo "  make lint             - Run linting checks"
	@echo "  make type-check       - Run type checking with mypy"
	@echo "  make clean            - Remove build artifacts and cache"
	@echo "  make clean-all        - Remove venv, build artifacts, and cache"
	@echo "  make freeze           - Update requirements.txt with current dependencies"
	@echo "  make check            - Run all quality checks (format, lint, type-check)"
	@echo "  make docker-build     - Build Docker image with RAG system"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-stop      - Stop Docker container"
	@echo "  make docker-clean     - Clean Docker resources"
	@echo "  make docker-neo4j     - Start Neo4j database only"
	@echo "  make docker-up        - Start full stack (Neo4j + App)"
	@echo "  make docker-down      - Stop all Docker services"
	@echo "  make docker-logs      - View Docker service logs"
	@echo "  make gcp-setup        - Setup GCP deployment environment"
	@echo "  make gcp-deploy       - Deploy to GCP Cloud Run"
	@echo "  make gcp-build        - Build and deploy via Cloud Build"
	@echo "  make gcp-logs         - View Cloud Run logs"
	@echo "  make gcp-status       - Check Cloud Run service status"
	@echo "  make gcp-url          - Get Cloud Run service URL"

# Create virtual environment
.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created."

# Install dependencies
.PHONY: install
install: venv
	@echo "Installing dependencies..."
	@$(VENV_BIN)/pip install --upgrade pip
	@$(VENV_BIN)/pip install -e .
	@$(VENV_BIN)/pip install -r requirements.txt
	@echo "Dependencies installed."

# Install development dependencies
.PHONY: install-dev
install-dev: install
	@echo "Installing development dependencies..."
	@$(VENV_BIN)/pip install -r requirements-dev.txt
	@echo "Development dependencies installed."

# Download optional NLP models
.PHONY: install-models
install-models:
	@echo "Downloading optional NLP models..."
	@$(VENV_BIN)/python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || echo "NLTK data download failed (optional)"
	@$(VENV_BIN)/python -m spacy download en_core_web_sm || echo "spaCy model download failed (optional)"
	@echo "Optional models installed."

# Run MCP financial server
.PHONY: run-mcp
run-mcp:
	@echo "Starting MCP Financial Server..."
	@$(VENV_BIN)/python -m src.mcp_financial_server

# Run Streamlit app
.PHONY: run-streamlit
run-streamlit:
	@echo "Starting Streamlit app..."
	@$(VENV_BIN)/streamlit run streamlit_app.py

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	@$(VENV_BIN)/python test_server.py
	@if [ -d "tests" ]; then \
		echo "Running pytest..."; \
		$(VENV_BIN)/pytest tests/ -v; \
	fi

# Format code
.PHONY: format
format:
	@echo "Formatting code with black..."
	@$(VENV_BIN)/black src/ *.py

# Lint code
.PHONY: lint
lint:
	@echo "Running flake8..."
	@$(VENV_BIN)/flake8 src/ *.py --max-line-length=100 --extend-ignore=E203,W503

# Type check
.PHONY: type-check
type-check:
	@echo "Running mypy..."
	@$(VENV_BIN)/mypy src/ --ignore-missing-imports

# Run all quality checks
.PHONY: check
check: format lint type-check
	@echo "All quality checks completed."

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean completed."

# Clean everything including venv
.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "All artifacts removed."

# Update requirements.txt
.PHONY: freeze
freeze:
	@echo "Updating requirements.txt..."
	@$(VENV_BIN)/pip freeze | grep -E "^(mcp|yfinance|pandas|numpy|streamlit|plotly)" > requirements.txt
	@echo "requirements.txt updated."

# Development shortcuts
.PHONY: dev
dev: install-dev
	@echo "Development environment ready."

# Quick test of MCP server
.PHONY: test-mcp
test-mcp:
	@echo "Testing MCP server..."
	@$(VENV_BIN)/python test_server.py

# Check if venv is activated (helper)
.PHONY: check-venv
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠️  Virtual environment not activated. Run 'source venv/bin/activate'"; \
		exit 1; \
	fi

# Watch for changes and restart Streamlit
.PHONY: watch
watch:
	@echo "Watching for changes..."
	@$(VENV_BIN)/streamlit run streamlit_app.py --server.runOnSave true

# Build distribution
.PHONY: build
build:
	@echo "Building distribution..."
	@$(VENV_BIN)/pip install build
	@$(VENV_BIN)/python -m build
	@echo "Build completed. Check dist/ directory."

# Install pre-commit hooks (if using git)
.PHONY: hooks
hooks:
	@echo "Setting up git hooks..."
	@$(VENV_BIN)/pip install pre-commit
	@$(VENV_BIN)/pre-commit install
	@echo "Git hooks installed."

# Docker commands
.PHONY: docker-build
docker-build:
	@echo "Building Docker image with integrated RAG system..."
	@docker build -t financial-advisor:latest .
	@echo "Docker image built successfully."

.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	@docker run -p 8080:8080 \
		-e GOOGLE_AI_API_KEY="$(GOOGLE_AI_API_KEY)" \
		--name financial-advisor \
		financial-advisor:latest

.PHONY: docker-stop
docker-stop:
	@echo "Stopping Docker container..."
	@docker stop financial-advisor || true
	@docker rm financial-advisor || true

.PHONY: docker-clean
docker-clean:
	@echo "Cleaning Docker resources..."
	@docker stop financial-advisor || true
	@docker rm financial-advisor || true
	@docker rmi financial-advisor:latest || true

# GCP Cloud Run deployment commands
.PHONY: gcp-setup
gcp-setup:
	@echo "Setting up GCP deployment environment..."
	@chmod +x setup-gcp-cicd.sh deploy.sh
	@./setup-gcp-cicd.sh

.PHONY: gcp-deploy
gcp-deploy:
	@echo "Deploying to GCP Cloud Run..."
	@./deploy.sh

.PHONY: gcp-build
gcp-build:
	@echo "Building and deploying via Cloud Build..."
	@gcloud builds submit --config cloudbuild.yaml --project=electric-vision-463705-f6

.PHONY: gcp-logs
gcp-logs:
	@echo "Viewing Cloud Run logs..."
	@gcloud run logs tail financial-analyst --region=us-central1 --project=electric-vision-463705-f6

.PHONY: gcp-status
gcp-status:
	@echo "Checking Cloud Run service status..."
	@gcloud run services describe financial-analyst --region=us-central1 --project=electric-vision-463705-f6 --format='table(metadata.name,status.url,status.conditions[0].type,spec.template.spec.containers[0].image)'

.PHONY: gcp-url
gcp-url:
	@echo "Getting Cloud Run service URL..."
	@gcloud run services describe financial-analyst --region=us-central1 --format='value(status.url)' --project=electric-vision-463705-f6

# Docker Compose commands for Neo4j integration
.PHONY: docker-neo4j
docker-neo4j:
	@echo "Starting Neo4j database..."
	@docker-compose up -d neo4j
	@echo "Neo4j started. Access at http://localhost:7474 (neo4j/financialpass)"

.PHONY: docker-up
docker-up:
	@echo "Starting full stack (Neo4j + Financial App)..."
	@docker-compose up -d
	@echo "Stack started. App: http://localhost:8501, Neo4j: http://localhost:7474"

.PHONY: docker-down
docker-down:
	@echo "Stopping all Docker services..."
	@docker-compose down
	@echo "All services stopped."

.PHONY: docker-logs
docker-logs:
	@echo "Viewing Docker service logs..."
	@docker-compose logs -f

.PHONY: docker-neo4j-shell
docker-neo4j-shell:
	@echo "Connecting to Neo4j shell..."
	@docker-compose exec neo4j cypher-shell -u neo4j -p financialpass

.PHONY: docker-test-graph
docker-test-graph:
	@echo "Testing Neo4j graph connection..."
	@docker-compose exec financial-app python -c "from src.neo4j_client import get_graph_db; db = get_graph_db(); print('Neo4j Health:', db.health_check() if db else 'Not available')" || echo "Graph test failed - make sure services are running"