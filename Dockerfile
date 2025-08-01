# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including for ML/NLP libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies (including RAG system)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download optional NLP models (with fallback if download fails)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || echo "NLTK data download failed (optional)"
RUN python -m spacy download en_core_web_sm || echo "spaCy model download failed (optional)"

# Download and cache sentence transformer model to reduce startup time
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Model downloaded successfully')" || echo "Sentence transformer model download failed (optional)"

# Also try downloading with explicit cache directory
RUN python -c "import os; os.environ['TRANSFORMERS_CACHE']='/app/.cache/transformers'; os.environ['HF_HOME']='/app/.cache/huggingface'; from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || echo "Cached model download failed (optional)"

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Expose Streamlit port
EXPOSE 8080

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set environment variables for ML libraries (performance optimization)
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Neo4j configuration environment variables
ENV NEO4J_URI=bolt://localhost:7687
ENV NEO4J_USERNAME=neo4j
ENV NEO4J_PASSWORD=financialpass

# Create cache directories
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface /app/rag_cache /app/data /app/logs

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]