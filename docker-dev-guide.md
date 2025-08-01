# Docker Development Guide for Financial Advisor with Neo4j

This guide explains how to set up and run the Financial Advisor application with Neo4j graph database using Docker.

## Quick Start

### 1. Start Neo4j Database Only
```bash
# Start only Neo4j database
docker-compose up neo4j -d

# View logs
docker-compose logs -f neo4j
```

### 2. Start Full Application Stack
```bash
# Start both Neo4j and Financial App
docker-compose up -d

# View all logs
docker-compose logs -f
```

### 3. Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (deletes all data)
docker-compose down -v
```

## Neo4j Database Access

### Web Interface
- **URL**: http://localhost:7474
- **Username**: neo4j
- **Password**: financialpass

### Bolt Connection
- **URI**: bolt://localhost:7687
- **Username**: neo4j
- **Password**: financialpass

## Application Access

### Streamlit Web App
- **URL**: http://localhost:8501
- **Features**: Full financial analysis with Neo4j graph relationships

## Environment Configuration

### Required Environment Variables
Create a `.env` file in the project root:

```bash
# Google AI API Key (required for AI chat)
GOOGLE_AI_API_KEY=your-api-key-here

# Neo4j Configuration (optional - defaults provided)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=financialpass
```

## Development Workflow

### 1. Code Changes
When you modify code, the application will automatically reload thanks to volume mounting.

### 2. Installing New Dependencies
```bash
# Rebuild application container after adding dependencies
docker-compose build financial-app
docker-compose up financial-app -d
```

### 3. Database Development
```bash
# Access Neo4j shell
docker exec -it financial_neo4j cypher-shell -u neo4j -p financialpass

# Example Cypher queries
MATCH (n) RETURN count(n);  # Count all nodes
MATCH (c:Company) RETURN c LIMIT 10;  # View companies
```

### 4. Data Persistence
- Neo4j data is persisted in Docker volumes
- Application cache is mounted from host filesystem
- Logs are accessible via `docker-compose logs`

## Troubleshooting

### Neo4j Connection Issues
```bash
# Check Neo4j health
docker-compose exec neo4j cypher-shell --username neo4j --password financialpass "RETURN 1"

# View Neo4j logs
docker-compose logs neo4j
```

### Application Issues
```bash
# View application logs
docker-compose logs financial-app

# Restart application
docker-compose restart financial-app
```

### Memory Issues
If you encounter memory issues, you can adjust Neo4j memory settings in `docker-compose.yml`:

```yaml
environment:
  - NEO4J_dbms_memory_heap_max__size=4G  # Increase heap size
  - NEO4J_dbms_memory_pagecache_size=2G  # Increase page cache
```

## Production Considerations

### 1. Security
- Change default passwords in production
- Use Docker secrets for sensitive data
- Enable Neo4j authentication and authorization

### 2. Performance
- Adjust Neo4j memory settings based on available resources
- Use SSD storage for Neo4j data volumes
- Monitor resource usage with `docker stats`

### 3. Backup
```bash
# Backup Neo4j data
docker exec financial_neo4j neo4j-admin dump --database=neo4j --to=/var/lib/neo4j/backup.dump

# Copy backup from container
docker cp financial_neo4j:/var/lib/neo4j/backup.dump ./backup.dump
```

## Docker Commands Reference

```bash
# Build specific service
docker-compose build neo4j

# View running containers
docker-compose ps

# Execute commands in containers
docker-compose exec neo4j bash
docker-compose exec financial-app python -c "from src.neo4j_client import get_graph_db; print(get_graph_db().health_check())"

# View resource usage
docker stats

# Clean up everything
docker-compose down -v --rmi all
```

## Integration Testing

### Test Neo4j Connection
```bash
# From host
docker-compose exec financial-app python -c "
from src.neo4j_client import get_graph_db
db = get_graph_db()
print('Neo4j Health:', db.health_check() if db else 'Not available')
"
```

### Test Graph Features
```bash
# Test graph-enhanced RAG
docker-compose exec financial-app python -c "
from src.rag_graph_integration import get_graph_enhanced_rag
rag = get_graph_enhanced_rag()
print('Graph RAG Health:', rag.health_check() if rag else 'Not available')
"
```

This Docker setup provides a complete development environment for the Financial Advisor application with Neo4j graph database integration.