# Financial Advisor - Intelligent Stock Portfolio Manager

A comprehensive financial analysis platform that combines real-time market data, advanced AI analysis, and Neo4j knowledge graphs to provide intelligent investment insights and portfolio management.

## 🚀 Key Features

### 📊 **Core Financial Analysis**
- Real-time stock data and historical price analysis
- Interactive candlestick charts and technical indicators
- Financial statements analysis (Income, Balance Sheet, Cash Flow)
- Statistical analysis (volatility, Sharpe ratio, drawdown analysis)
- News aggregation with sentiment analysis

### 🤖 **AI-Powered Insights**
- Google Gemini-powered financial chat assistant
- Advanced RAG (Retrieval-Augmented Generation) system
- Graph-enhanced contextual analysis
- Investment recommendations and risk assessment

### 🔗 **Neo4j Knowledge Graph Integration**
- **Company Relationship Mapping**: Discover ownership, supply chain, and competitive relationships
- **Portfolio Correlation Analysis**: Identify hidden correlations and risk concentrations
- **Market Influence Networks**: Map how events propagate through related companies
- **Supply Chain Risk Assessment**: Understand dependency vulnerabilities
- **News Impact Propagation**: Track how news affects interconnected companies

### 🌐 **Multi-Protocol Integration**
- **MCP Server**: Model Context Protocol for AI tool integration
- **RESTful APIs**: Yahoo Finance integration for real-time data
- **Graph Database**: Neo4j for relationship storage and analysis
- **Vector Search**: Advanced RAG with HNSW indexing

## 🛠 Quick Start

### Prerequisites
- Python 3.8+
- Neo4j 5.0+ (optional, but recommended for full features)
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd fl_financial_advisor

# Install dependencies (includes Neo4j drivers and advanced RAG system)
make install

# Activate virtual environment
source venv/bin/activate
```

### Neo4j Setup (Recommended)

**Option 1: Docker (Easiest)**
```bash
# Start Neo4j database
make docker-neo4j

# Access Neo4j Browser: http://localhost:7474
# Username: neo4j, Password: financialpass
```

**Option 2: Local Installation**
```bash
# Ubuntu/Debian
sudo apt install neo4j

# macOS  
brew install neo4j

# Start service
sudo systemctl start neo4j
```

### Running the Application

```bash
# Start Streamlit web application
make run-streamlit

# Or run MCP server
make run-mcp
```

### Environment Configuration

Create `.env` file:
```bash
# Google AI API Key (for AI chat features)
GOOGLE_AI_API_KEY=your_api_key_here

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=financialpass
```

## 💡 Usage Examples

### Streamlit Web Application
1. **Stock Analysis**: Enter any stock symbol (e.g., AAPL, TSLA) to get comprehensive analysis
2. **AI Chat**: Ask questions like "What are the risks of investing in Tesla?" or "How does Apple's supply chain affect its stock?"
3. **Relationship Analysis**: Use the Relationships tab to discover company connections and portfolio correlations
4. **Portfolio Analysis**: Enter multiple symbols to analyze hidden correlations

### Graph Database Features
- **Index Companies**: Click "Index Company" to add companies to the knowledge graph
- **Find Relationships**: Discover how companies are connected through ownership, supply chains, or competition
- **Portfolio Correlations**: Analyze risk concentrations in your portfolio
- **Market Networks**: Visualize how market events propagate through company relationships

### MCP Server Tools

**Financial Data Tools:**
1. **get_stock_history** - Historical price data with configurable periods
2. **get_stock_info** - Detailed company information and metrics  
3. **get_stock_quote** - Real-time stock quotes
4. **get_news** - Latest financial news with sentiment analysis

**Graph Database Tools (New):**
5. **get_company_relationships** - Find related companies through graph traversal
6. **analyze_portfolio_relationships** - Discover portfolio correlation risks
7. **get_market_influence_network** - Map market influence propagation
8. **find_supply_chain_risks** - Identify supply chain vulnerabilities

### MCP Configuration

Add to your MCP settings file:

```json
{
  "mcpServers": {
    "financial-server": {
      "command": "python",
      "args": ["-m", "mcp_financial_server"]
    }
  }
}
```

## 🏗 Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│            (Interactive UI + Relationship Visualization)    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                Application Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ RAG System  │  │ MCP Server  │  │ Neo4j Graph Engine  │ │
│  │ (Enhanced)  │  │ (Extended)  │  │ (Relationship AI)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────┬───────────┬─────────────────┬─────────────────┘
              │           │                 │
┌─────────────┴───┐  ┌────┴─────┐  ┌───────┴──────────┐
│ Vector Store    │  │ Yahoo    │  │ Neo4j Database   │
│ (HNSW)         │  │ Finance  │  │ (Graph Store)    │
└─────────────────┘  └──────────┘  └──────────────────┘
```

### Technology Stack

**Frontend & UI**
- **Streamlit** - Interactive web application framework
- **Plotly** - Advanced financial charting and visualization
- **Custom CSS** - Dark theme optimized for financial data

**AI & Machine Learning**  
- **Google Gemini** - Large language model for financial analysis
- **Sentence Transformers** - Semantic embeddings for RAG
- **FAISS/HNSW** - Vector similarity search
- **spaCy/NLTK** - Natural language processing

**Graph Database**
- **Neo4j 5.x** - Knowledge graph storage and querying
- **Cypher** - Graph query language for relationship analysis
- **APOC** - Neo4j procedures for advanced graph algorithms

**Data & APIs**
- **Yahoo Finance (yfinance)** - Real-time and historical market data
- **Model Context Protocol (MCP)** - AI tool integration framework
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

**Infrastructure**
- **Docker** - Containerized deployment
- **Google Cloud Run** - Serverless deployment platform
- **Cloud Build** - CI/CD pipeline
- **Secret Manager** - Secure credential management

## 📊 Example Outputs

### Stock Analysis with Relationships
```json
{
  "symbol": "AAPL",
  "analysis": {
    "price_data": {...},
    "relationships": [
      {
        "connected_company": "MSFT",
        "relationship_type": "COMPETES_WITH",
        "strength": 0.85
      }
    ],
    "ai_insights": "Apple shows strong correlation with...",
    "risk_assessment": {...}
  }
}
```

### Portfolio Correlation Analysis
```json
{
  "portfolio": ["AAPL", "MSFT", "GOOGL"],
  "correlations": [
    {
      "stock1": "AAPL", "stock2": "MSFT",
      "relationship_type": "COMPETES_WITH",
      "correlation": 0.73
    }
  ],
  "risk_insights": "High correlation detected between...",
  "recommendations": [...]
}
```

## 🚀 Deployment

### Local Development
```bash
# Full stack with Neo4j
make docker-up

# Streamlit only
make run-streamlit

# MCP server only  
make run-mcp
```

### Production Deployment (Google Cloud)
```bash
# One-time setup
make gcp-setup

# Deploy application
make gcp-deploy

# View deployed application
make gcp-url
```

## 🧪 Testing

```bash
# Run all tests
make test

# Test MCP server
make test-mcp

# Test Neo4j integration
make docker-test-graph

# Code quality checks
make check
```

## 📚 Documentation

- **[Technical Specifications](docs/TECHNICAL_SPECIFICATIONS.md)** - Detailed technical documentation
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development setup and contribution guidelines  
- **[User Guide](docs/USER_GUIDE.md)** - Complete user manual
- **[Neo4j Integration Plan](docs/NEO4J_INTEGRATION_PLAN.md)** - Graph database strategy and implementation
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Production deployment instructions
- **[GCP Deployment](GCP-DEPLOYMENT.md)** - Google Cloud specific deployment guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for comprehensive financial data APIs
- **Neo4j** for powerful graph database capabilities
- **Google Gemini** for advanced AI analysis capabilities
- **Streamlit** for the excellent web framework
- **Model Context Protocol** for AI tool integration standards