# Technical Specifications

## Component Specifications

### 1. MCP Financial Server (Enhanced with Graph Integration)

#### Interface Specification
```python
class FinancialServer:
    """MCP-compliant financial data server"""
    
    @tool("get_stock_history")
    async def get_stock_history(
        symbol: str,
        period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        interval: Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    ) -> Dict[str, Any]:
        """Fetch historical stock data"""
        pass
    
    @tool("get_stock_info")
    async def get_stock_info(symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information"""
        pass
    
    @tool("get_stock_quote")
    async def get_stock_quote(symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote"""
        pass
    
    @tool("get_news")
    async def get_news(symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get latest financial news"""
        pass
    
    # Graph Database Tools (New)
    @tool("get_company_relationships")
    async def get_company_relationships(
        symbol: str, 
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """Find related companies through graph traversal"""
        pass
    
    @tool("analyze_portfolio_relationships")
    async def analyze_portfolio_relationships(
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Analyze relationship correlations in portfolio"""
        pass
    
    @tool("get_market_influence_network")
    async def get_market_influence_network(
        symbol: str, 
        depth: int = 2
    ) -> Dict[str, Any]:
        """Map market influence propagation networks"""
        pass
    
    @tool("find_supply_chain_risks")
    async def find_supply_chain_risks(symbol: str) -> Dict[str, Any]:
        """Identify supply chain vulnerabilities"""
        pass
```

#### Error Handling
```python
class FinancialDataError(Exception):
    """Base exception for financial data errors"""
    pass

class SymbolNotFoundError(FinancialDataError):
    """Raised when stock symbol is not found"""
    pass

class APILimitExceededError(FinancialDataError):
    """Raised when API rate limits are exceeded"""
    pass
```

### 2. RAG System Architecture

#### Semantic Chunker
```python
class SemanticChunker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.coherence_threshold = 0.7
        self.max_chunk_size = 512
        self.min_chunk_size = 50
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Create semantically coherent chunks"""
        # Implementation details in main code
        pass
```

#### HNSW Vector Store
```python
class HNSWVectorStore:
    def __init__(self, dimension: int = 384, max_elements: int = 100000):
        self.dimension = dimension
        self.max_elements = max_elements
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=max_elements, ef_construction=400, M=32)
    
    def add_chunks(self, chunks: List[SemanticChunk]) -> None:
        """Add chunks to vector store"""
        pass
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[SemanticChunk, float]]:
        """Search for similar chunks"""
        pass
```

#### Query Rewriter
```python
class QueryRewriter:
    def __init__(self):
        self.financial_synonyms = {
            "profit": ["earnings", "income", "net income", "profit margin"],
            "revenue": ["sales", "turnover", "top line", "gross revenue"],
            # ... more synonyms
        }
    
    def rewrite_query(self, query: str) -> QueryRewrite:
        """Generate multiple query variants"""
        pass
```

### 3. Neo4j Graph Database

#### Schema Specification
```python
# Node Types
class CompanyNode:
    symbol: str
    name: str
    sector: str
    industry: str
    marketCap: float
    employees: int
    headquarters: str
    description: str
    currency: str
    country: str
    website: str

class PersonNode:
    id: str
    name: str
    title: str
    biography: str

class SectorNode:
    name: str
    description: str

class IndustryNode:
    name: str
    sector: str
    description: str

class NewsEventNode:
    id: str
    title: str
    summary: str
    date: datetime
    sentiment: float
    source: str
    impact_score: float
    url: str

# Relationship Types
class RelationshipTypes:
    OWNS = "OWNS"  # {percentage: float, since: date}
    SUPPLIES = "SUPPLIES"  # {volume: str, since: date}
    COMPETES_WITH = "COMPETES_WITH"  # {intensity: float}
    CORRELATED_WITH = "CORRELATED_WITH"  # {coefficient: float, timeframe: str}
    AFFECTS = "AFFECTS"  # {impact_score: float}
    BELONGS_TO = "BELONGS_TO"  # Company -> Industry
    PART_OF = "PART_OF"  # Industry -> Sector
    EXECUTIVE_OF = "EXECUTIVE_OF"  # Person -> Company
    BOARD_MEMBER_OF = "BOARD_MEMBER_OF"  # Person -> Company
```

#### Graph Client Interface
```python
class FinancialGraphDB:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection"""
        pass
    
    def create_company(self, company_data: Dict[str, Any]) -> bool:
        """Create or update company node"""
        pass
    
    def create_relationship(
        self, 
        from_symbol: str, 
        to_symbol: str,
        rel_type: str, 
        properties: Dict = None
    ) -> bool:
        """Create relationship between companies"""
        pass
    
    def find_related_companies(
        self, 
        symbol: str, 
        max_hops: int = 2, 
        limit: int = 20
    ) -> List[Dict]:
        """Find companies related to given symbol"""
        pass
    
    def analyze_portfolio_correlations(self, symbols: List[str]) -> List[Dict]:
        """Analyze correlations between portfolio holdings"""
        pass
    
    def get_market_influence_network(
        self, 
        symbol: str, 
        depth: int = 2
    ) -> Dict[str, Any]:
        """Get network of market influence around a company"""
        pass
    
    def find_supply_chain_risk(self, symbol: str) -> List[Dict]:
        """Find potential supply chain risks"""
        pass
```

#### Performance Specifications
```python
class GraphPerformanceSpecs:
    MAX_QUERY_TIME = 500  # milliseconds
    MAX_RELATIONSHIP_HOPS = 3
    INDEX_REFRESH_INTERVAL = 3600  # seconds
    CACHE_TTL = 1800  # seconds for relationship cache
    
    # Index specifications
    REQUIRED_INDEXES = [
        "CREATE INDEX company_symbol FOR (c:Company) ON (c.symbol)",
        "CREATE INDEX company_sector FOR (c:Company) ON (c.sector)",
        "CREATE INDEX news_date FOR (n:NewsEvent) ON (n.date)",
        "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
    ]
    
    # Constraints
    REQUIRED_CONSTRAINTS = [
        "CREATE CONSTRAINT company_symbol FOR (c:Company) REQUIRE c.symbol IS UNIQUE",
        "CREATE CONSTRAINT person_id FOR (p:Person) REQUIRE p.id IS UNIQUE",
    ]
```

### 4. Graph-Enhanced RAG System

#### Enhanced RAG Interface
```python
class GraphEnhancedRAG:
    def __init__(self, rag_system, graph_db: FinancialGraphDB):
        """Initialize graph-enhanced RAG system"""
        pass
    
    def expand_query_with_relationships(
        self, 
        query: str, 
        symbol: str
    ) -> List[str]:
        """Expand query to include related entities"""
        pass
    
    def get_relationship_context(self, symbol: str) -> str:
        """Generate context about company relationships"""
        pass
    
    def enhanced_retrieval(
        self, 
        query: str, 
        symbol: str, 
        k: int = 10
    ) -> Dict[str, Any]:
        """RAG retrieval enhanced with graph relationships"""
        pass
```

#### Query Enhancement Strategies
```python
class QueryEnhancementStrategies:
    RELATIONSHIP_EXPANSION = "relationship_expansion"
    MULTI_HOP_TRAVERSAL = "multi_hop_traversal"
    CORRELATION_CONTEXT = "correlation_context"
    SUPPLY_CHAIN_CONTEXT = "supply_chain_context"
    NEWS_IMPACT_CONTEXT = "news_impact_context"
    
    def apply_strategy(
        self, 
        strategy: str, 
        query: str, 
        symbol: str
    ) -> EnhancedQuery:
        """Apply enhancement strategy to query"""
        pass
```

### 5. Data Models

#### Stock Data Model
```python
@dataclass
class StockData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
```

#### Company Info Model
```python
@dataclass
class CompanyInfo:
    symbol: str
    long_name: str
    sector: str
    industry: str
    market_cap: int
    enterprise_value: int
    trailing_pe: float
    forward_pe: float
    price_to_book: float
    dividend_yield: Optional[float]
    beta: float
    website: str
    business_summary: str
```

#### Financial Statement Model
```python
@dataclass
class FinancialStatement:
    symbol: str
    period_type: Literal["quarterly", "annual"]
    end_date: datetime
    total_revenue: int
    gross_profit: int
    operating_income: int
    net_income: int
    total_assets: int
    total_liabilities: int
    shareholders_equity: int
    operating_cash_flow: int
    free_cash_flow: int
```

## Performance Specifications

### Response Time Requirements
- Stock data fetch: < 2 seconds
- AI analysis: < 15 seconds
- RAG query: < 3 seconds
- Chart rendering: < 1 second

### Memory Usage
- Base application: 500MB
- With RAG system: 2GB
- Peak usage (10 stocks): 4GB

### Throughput
- Concurrent users: 100+
- Requests per second: 50+
- Daily active users: 1000+

## Data Specifications

### Supported Stock Exchanges
- NYSE (New York Stock Exchange)
- NASDAQ
- LSE (London Stock Exchange)
- TSE (Tokyo Stock Exchange)
- TSX (Toronto Stock Exchange)

### Data Refresh Intervals
- Real-time quotes: 1 minute
- Historical data: 5 minutes
- Financial statements: Daily
- News data: 15 minutes

### Data Retention
- Price data: 10 years maximum
- News articles: 30 days
- RAG index: Session-based (ephemeral)
- User sessions: 24 hours

## Integration Specifications

### External APIs

#### Yahoo Finance API
```python
BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
RATE_LIMIT = "2000 requests/hour"
TIMEOUT = 30  # seconds

# Example endpoint
GET /v8/finance/chart/{symbol}?interval={interval}&period1={start}&period2={end}
```

#### Google Gemini API
```python
MODEL = "gemini-1.5-flash"
MAX_TOKENS = 1000000  # Context window
TEMPERATURE = 0.7
TOP_P = 0.9
RATE_LIMIT = "60 requests/minute"
```

### Internal APIs

#### Streamlit Session State
```python
SessionState = {
    "stock_data": pd.DataFrame,
    "stock_info": Dict[str, Any],
    "chat_history": List[Dict[str, str]],
    "current_symbol": str,
    "financial_statements": Dict[str, pd.DataFrame],
    "rag_index": Optional[RAGSystem]
}
```

## Security Specifications

### API Key Management
```yaml
Environment Variables:
  - GOOGLE_AI_API_KEY: Gemini API access
  
Secret Manager:
  - google-ai-api-key: Production key storage
  
Access Control:
  - Service Account: electric-vision-463705-f6@appspot.gserviceaccount.com
  - Roles: 
    - secretmanager.secretAccessor
    - run.developer
```

### Input Validation
```python
def validate_stock_symbol(symbol: str) -> str:
    """Validate and sanitize stock symbol"""
    if not symbol or len(symbol) > 10:
        raise ValueError("Invalid symbol length")
    
    # Remove special characters except dots and dashes
    sanitized = re.sub(r'[^A-Za-z0-9.-]', '', symbol.upper())
    
    if not sanitized:
        raise ValueError("Invalid symbol format")
    
    return sanitized

def validate_period(period: str) -> str:
    """Validate time period parameter"""
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    if period not in valid_periods:
        raise ValueError(f"Invalid period. Must be one of: {valid_periods}")
    return period
```

## Quality Assurance Specifications

### Testing Requirements
- Unit test coverage: >90%
- Integration test coverage: >80%
- End-to-end test coverage: >70%
- Performance test scenarios: Load, stress, spike

### Monitoring Requirements
```python
Metrics:
  - Response time percentiles (p50, p95, p99)
  - Error rates by endpoint
  - Memory usage patterns
  - CPU utilization
  - Cache hit rates
  - RAG retrieval accuracy

Alerts:
  - Response time > 10s
  - Error rate > 5%
  - Memory usage > 3.5GB
  - API rate limit approaching
```

### Logging Specifications
```python
LOG_FORMAT = {
    "timestamp": "ISO 8601",
    "level": "INFO|WARN|ERROR",
    "component": "mcp|rag|streamlit|ai",
    "message": "string",
    "user_id": "optional_uuid",
    "request_id": "uuid",
    "duration_ms": "integer",
    "error_code": "optional_string"
}

# Example log entry
{
    "timestamp": "2024-12-29T10:30:00Z",
    "level": "INFO",
    "component": "rag",
    "message": "Query processed successfully",
    "request_id": "abc-123-def",
    "duration_ms": 250,
    "query_tokens": 15,
    "retrieved_chunks": 5
}
```