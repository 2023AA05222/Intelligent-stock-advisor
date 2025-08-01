# Neo4j Integration Plan for Financial Advisor Application

## Overview

This document outlines the integration of Neo4j graph database to enhance the Financial Advisor Application with advanced relationship analysis, pattern discovery, and contextual financial intelligence.

## Table of Contents
1. [Integration Architecture](#integration-architecture)
2. [Graph Schema Design](#graph-schema-design)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Use Cases and Queries](#use-cases-and-queries)
5. [Technical Implementation](#technical-implementation)
6. [Deployment Strategy](#deployment-strategy)
7. [Performance Considerations](#performance-considerations)

---

## Integration Architecture

### Current System + Neo4j
```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                Application Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ RAG System  │  │ MCP Server  │  │ Neo4j Graph Engine  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────┬───────────┬─────────────────┬─────────────────┘
              │           │                 │
┌─────────────┴───┐  ┌────┴─────┐  ┌───────┴──────────┐
│ Vector Store    │  │ Yahoo    │  │ Neo4j Database   │
│ (HNSW)         │  │ Finance  │  │ (Graph Store)    │
└─────────────────┘  └──────────┘  └──────────────────┘
```

### Integration Points

1. **Data Ingestion Pipeline**
   - Extend existing MCP server to populate Neo4j
   - Real-time relationship discovery from financial data
   - News event relationship mapping

2. **Enhanced RAG System**
   - Graph-based context retrieval
   - Relationship-aware chunking
   - Multi-hop query expansion

3. **New Analysis Capabilities**
   - Market correlation analysis
   - Portfolio risk assessment through relationships
   - Industry impact modeling

---

## Graph Schema Design

### Core Node Types

#### 1. Company Nodes
```cypher
(:Company {
  symbol: string,
  name: string,
  sector: string,
  industry: string,
  marketCap: float,
  employees: int,
  founded: date,
  headquarters: string,
  description: string
})
```

#### 2. Person Nodes (Executives, Board Members)
```cypher
(:Person {
  name: string,
  title: string,
  linkedIn: string,
  biography: string
})
```

#### 3. Sector/Industry Nodes
```cypher
(:Sector {
  name: string,
  description: string
})

(:Industry {
  name: string,
  sector: string,
  description: string
})
```

#### 4. Financial Event Nodes
```cypher
(:NewsEvent {
  title: string,
  summary: string,
  date: datetime,
  sentiment: float,
  source: string,
  impact_score: float
})

(:EarningsEvent {
  date: datetime,
  quarter: string,
  revenue: float,
  eps: float,
  guidance: string
})
```

#### 5. Geographic Nodes
```cypher
(:Country {
  code: string,
  name: string,
  gdp: float,
  currency: string
})

(:Market {
  name: string,
  country: string,
  timezone: string,
  trading_hours: string
})
```

### Core Relationship Types

#### 1. Corporate Structure
```cypher
// Ownership and control
(parent:Company)-[:OWNS {percentage: float, since: date}]->(subsidiary:Company)
(company:Company)-[:LISTED_ON]->(market:Market)
(company:Company)-[:OPERATES_IN]->(country:Country)

// Industry classification
(company:Company)-[:BELONGS_TO]->(industry:Industry)
(industry:Industry)-[:PART_OF]->(sector:Sector)
```

#### 2. Business Relationships
```cypher
// Supply chain and partnerships
(supplier:Company)-[:SUPPLIES {volume: string, since: date}]->(customer:Company)
(company1:Company)-[:PARTNERS_WITH {type: string, since: date}]->(company2:Company)
(company1:Company)-[:COMPETES_WITH {intensity: float}]->(company2:Company)
```

#### 3. People Relationships
```cypher
// Leadership and governance
(person:Person)-[:EXECUTIVE_OF {title: string, since: date}]->(company:Company)
(person:Person)-[:BOARD_MEMBER_OF {since: date, committee: string}]->(company:Company)
(person:Person)-[:PREVIOUSLY_WORKED_AT {title: string, duration: string}]->(company:Company)
```

#### 4. Event Relationships
```cypher
// Market events and impacts
(event:NewsEvent)-[:AFFECTS {impact_score: float}]->(company:Company)
(event:NewsEvent)-[:MENTIONS]->(company:Company)
(earnings:EarningsEvent)-[:REPORTED_BY]->(company:Company)
(company:Company)-[:INFLUENCES {correlation: float}]->(other:Company)
```

#### 5. Financial Relationships
```cypher
// Market correlations and dependencies
(company1:Company)-[:CORRELATED_WITH {coefficient: float, timeframe: string}]->(company2:Company)
(company:Company)-[:SENSITIVE_TO {beta: float}]->(sector:Sector)
(event:NewsEvent)-[:IMPACTS_SECTOR {severity: float}]->(sector:Sector)
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Neo4j database (local development)
- [ ] Design and implement core schema
- [ ] Create basic company and relationship data ingestion
- [ ] Develop Neo4j integration module

### Phase 2: Data Population (Weeks 3-4)
- [ ] Extend MCP server to write to Neo4j
- [ ] Build relationship discovery algorithms
- [ ] Import historical company data and relationships
- [ ] Implement news event relationship mapping

### Phase 3: RAG Enhancement (Weeks 5-6)
- [ ] Integrate graph queries into RAG system
- [ ] Develop graph-based context generation
- [ ] Create relationship-aware query expansion
- [ ] Implement multi-hop retrieval strategies

### Phase 4: Advanced Analytics (Weeks 7-8)
- [ ] Build correlation analysis features
- [ ] Implement risk assessment through relationships
- [ ] Create industry impact modeling
- [ ] Develop portfolio relationship analysis

### Phase 5: UI Integration (Weeks 9-10)
- [ ] Add graph visualization to Streamlit app
- [ ] Create relationship exploration interface
- [ ] Implement graph-based search and filtering
- [ ] Add advanced analytics dashboards

### Phase 6: Production Deployment (Weeks 11-12)
- [ ] Set up Neo4j on Google Cloud
- [ ] Implement production data pipeline
- [ ] Performance optimization and monitoring
- [ ] Documentation and training

---

## Use Cases and Queries

### 1. Market Impact Analysis
**Question**: "How might a semiconductor shortage affect Apple's business?"

```cypher
// Find supply chain dependencies
MATCH path = (apple:Company {symbol: 'AAPL'})
-[:SUPPLIES|PARTNERS*1..3]-(supplier:Company)
-[:BELONGS_TO]->(industry:Industry)
WHERE industry.name CONTAINS 'semiconductor'
RETURN path, supplier.name, supplier.symbol
```

### 2. Portfolio Risk Assessment
**Question**: "What hidden correlations exist in my tech portfolio?"

```cypher
// Find shared dependencies and correlations
MATCH (stock1:Company)-[:CORRELATED_WITH|SHARES_SUPPLIER|SAME_MARKET*1..2]-(stock2:Company)
WHERE stock1.symbol IN ['AAPL', 'MSFT', 'GOOGL']
  AND stock2.symbol IN ['AAPL', 'MSFT', 'GOOGL']
  AND stock1 <> stock2
RETURN stock1.symbol, stock2.symbol, 
       relationships(path) as connection_types,
       length(path) as connection_strength
```

### 3. Industry Leadership Analysis
**Question**: "Who are the key people connecting different tech companies?"

```cypher
// Find influential people across companies
MATCH (person:Person)-[:EXECUTIVE_OF|BOARD_MEMBER_OF]->(company:Company)
WHERE company.sector = 'Technology'
WITH person, collect(company.symbol) as companies
WHERE size(companies) > 1
RETURN person.name, companies, size(companies) as influence_score
ORDER BY influence_score DESC
```

### 4. News Impact Propagation
**Question**: "How does news about Tesla affect the broader EV market?"

```cypher
// Trace news impact through relationships
MATCH (tesla:Company {symbol: 'TSLA'})
<-[:AFFECTS]-(news:NewsEvent)
-[:IMPACTS_SECTOR]->(sector:Sector)
-[:CONTAINS]->(affected:Company)
WHERE news.date > date() - duration('P7D')
RETURN news.title, affected.symbol, affected.name,
       news.impact_score, news.sentiment
ORDER BY news.impact_score DESC
```

### 5. Competitive Landscape Mapping
**Question**: "What companies are in direct competition with Microsoft?"

```cypher
// Map competitive relationships
MATCH (msft:Company {symbol: 'MSFT'})
-[:COMPETES_WITH|OPERATES_IN_SAME_MARKET|SIMILAR_BUSINESS_MODEL]-(competitor:Company)
OPTIONAL MATCH (competitor)-[:BELONGS_TO]->(industry:Industry)
RETURN competitor.symbol, competitor.name, industry.name,
       competitor.marketCap, competitor.revenue
ORDER BY competitor.marketCap DESC
```

---

## Technical Implementation

### 1. Neo4j Driver Integration

```python
# neo4j_client.py
from neo4j import GraphDatabase
import pandas as pd
from typing import Dict, List, Any
import logging

class FinancialGraphDB:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = logging.getLogger(__name__)
    
    def close(self):
        self.driver.close()
    
    def create_company(self, company_data: Dict[str, Any]) -> bool:
        """Create or update company node"""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (c:Company {symbol: $symbol})
                SET c.name = $name,
                    c.sector = $sector,
                    c.industry = $industry,
                    c.marketCap = $marketCap,
                    c.employees = $employees,
                    c.headquarters = $headquarters,
                    c.description = $description,
                    c.updated = datetime()
                RETURN c.symbol as symbol
            """, **company_data)
            return bool(result.single())
    
    def create_relationship(self, from_symbol: str, to_symbol: str, 
                          rel_type: str, properties: Dict = None) -> bool:
        """Create relationship between companies"""
        properties = properties or {}
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (a:Company {{symbol: $from_symbol}})
                MATCH (b:Company {{symbol: $to_symbol}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $properties, r.updated = datetime()
                RETURN r
            """, from_symbol=from_symbol, to_symbol=to_symbol, properties=properties)
            return bool(result.single())
    
    def find_related_companies(self, symbol: str, max_hops: int = 2) -> List[Dict]:
        """Find companies related to given symbol"""
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH path = (start:Company {{symbol: $symbol}})
                -[*1..{max_hops}]-(related:Company)
                WHERE start <> related
                RETURN DISTINCT related.symbol as symbol,
                       related.name as name,
                       length(path) as distance,
                       relationships(path) as connections
                ORDER BY distance, related.marketCap DESC
                LIMIT 20
            """, symbol=symbol)
            return [record.data() for record in result]
    
    def analyze_portfolio_correlations(self, symbols: List[str]) -> List[Dict]:
        """Analyze correlations between portfolio holdings"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s1:Company)-[r:CORRELATED_WITH|SHARES_SUPPLIER|COMPETES_WITH]-(s2:Company)
                WHERE s1.symbol IN $symbols AND s2.symbol IN $symbols
                  AND s1.symbol < s2.symbol  // Avoid duplicates
                RETURN s1.symbol as stock1, s2.symbol as stock2,
                       type(r) as relationship_type,
                       r.coefficient as correlation,
                       r.strength as strength
                ORDER BY abs(r.coefficient) DESC
            """, symbols=symbols)
            return [record.data() for record in result]
```

### 2. RAG System Enhancement

```python
# rag_graph_integration.py
from .neo4j_client import FinancialGraphDB
from typing import List, Dict, Any

class GraphEnhancedRAG:
    def __init__(self, rag_system, graph_db: FinancialGraphDB):
        self.rag_system = rag_system
        self.graph_db = graph_db
    
    def expand_query_with_relationships(self, query: str, symbol: str) -> List[str]:
        """Expand query to include related entities"""
        # Find related companies
        related = self.graph_db.find_related_companies(symbol, max_hops=2)
        
        # Generate expanded queries
        expanded_queries = [query]
        
        for rel_company in related[:5]:  # Limit to top 5 related
            expanded_queries.extend([
                f"{query} {rel_company['name']}",
                f"{query} related to {rel_company['symbol']}",
                f"How does {query} affect {rel_company['symbol']}"
            ])
        
        return expanded_queries
    
    def get_relationship_context(self, symbol: str) -> str:
        """Generate context about company relationships"""
        related = self.graph_db.find_related_companies(symbol)
        
        if not related:
            return ""
        
        context_parts = [f"=== RELATIONSHIP CONTEXT FOR {symbol} ==="]
        
        for rel in related:
            distance = rel['distance']
            connections = [conn['type'] for conn in rel['connections']]
            context_parts.append(
                f"{rel['symbol']} ({rel['name']}) - "
                f"Distance: {distance}, "
                f"Connected via: {', '.join(set(connections))}"
            )
        
        return "\n".join(context_parts)
    
    def enhanced_retrieval(self, query: str, symbol: str, k: int = 10) -> Dict[str, Any]:
        """RAG retrieval enhanced with graph relationships"""
        # Standard RAG retrieval
        standard_result = self.rag_system.query(query, k=k)
        
        # Get relationship context
        relationship_context = self.get_relationship_context(symbol)
        
        # Expand query with relationships
        expanded_queries = self.expand_query_with_relationships(query, symbol)
        
        # Enhanced context
        enhanced_context = "\n\n".join([
            standard_result['context'],
            relationship_context
        ])
        
        return {
            **standard_result,
            'context': enhanced_context,
            'relationship_context': relationship_context,
            'expanded_queries': expanded_queries,
            'graph_enhanced': True
        }
```

### 3. MCP Server Extension

```python
# Add to src/mcp_financial_server/server.py

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """Extended tool list with graph capabilities"""
    return [
        # ... existing tools ...
        Tool(
            name="get_company_relationships",
            description="Get relationship network for a company",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol"},
                    "max_hops": {"type": "integer", "default": 2, "description": "Maximum relationship hops"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="analyze_portfolio_relationships",
            description="Analyze relationships between portfolio holdings",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "List of stock symbols"}
                },
                "required": ["symbols"]
            }
        )
    ]

async def get_company_relationships(symbol: str, max_hops: int = 2) -> List[TextContent]:
    """Get company relationship network"""
    try:
        graph_db = get_graph_db_connection()
        relationships = graph_db.find_related_companies(symbol, max_hops)
        
        result = {
            "symbol": symbol,
            "relationship_count": len(relationships),
            "relationships": relationships,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
```

---

## Deployment Strategy

### Local Development
```yaml
# docker-compose.yml addition
version: '3.8'
services:
  neo4j:
    image: neo4j:5.13
    ports:
      - "7474:7474"  # Browser interface
      - "7687:7687"  # Bolt protocol
    environment:
      - NEO4J_AUTH=neo4j/financialpass
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    
  financial-advisor:
    build: .
    depends_on:
      - neo4j
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=financialpass

volumes:
  neo4j_data:
  neo4j_logs:
```

### Google Cloud Production
```bash
# Neo4j AuraDB Professional (Managed)
# Or Google Cloud Compute Engine with Neo4j

# Update Cloud Run environment variables
gcloud run services update financial-analyst \
  --set-env-vars="NEO4J_URI=neo4j+s://xxx.databases.neo4j.io,NEO4J_USERNAME=neo4j" \
  --set-secrets="NEO4J_PASSWORD=neo4j-password:latest"
```

### Infrastructure Requirements
- **Memory**: Additional 1-2GB for graph operations
- **Storage**: 10-50GB for relationship data (depending on scope)
- **Network**: Persistent connection to Neo4j database

---

## Performance Considerations

### Indexing Strategy
```cypher
// Essential indexes for performance
CREATE INDEX company_symbol FOR (c:Company) ON (c.symbol);
CREATE INDEX company_sector FOR (c:Company) ON (c.sector);
CREATE INDEX news_date FOR (n:NewsEvent) ON (n.date);
CREATE INDEX person_name FOR (p:Person) ON (p.name);

// Composite indexes for complex queries
CREATE INDEX company_sector_market_cap FOR (c:Company) ON (c.sector, c.marketCap);
```

### Query Optimization
- **Limit relationship depth** (max 3 hops for most queries)
- **Use LIMIT clauses** to prevent expensive full-graph scans
- **Implement query caching** for frequently accessed relationships
- **Batch relationship updates** during data ingestion

### Caching Strategy
- **Relationship cache**: 1 hour TTL for company relationships
- **Graph query cache**: 30 minutes for complex analyses
- **Real-time updates**: Only for news events and market data

---

## Success Metrics

### Technical Metrics
- **Query performance**: <500ms for relationship queries
- **Data freshness**: Relationships updated within 1 hour
- **Coverage**: 80%+ of S&P 500 companies with full relationship data

### Business Value Metrics
- **Enhanced insights**: 40%+ more comprehensive AI responses
- **Risk discovery**: Early identification of portfolio correlations
- **Market understanding**: Better prediction of sector impacts

---

## Next Steps

1. **Immediate (Week 1)**:
   - Set up local Neo4j development environment
   - Design initial schema for top 100 companies
   - Create basic data ingestion pipeline

2. **Short-term (Weeks 2-4)**:
   - Integrate with existing MCP server
   - Build relationship discovery algorithms
   - Enhance RAG system with graph queries

3. **Medium-term (Weeks 5-8)**:
   - Add graph visualization to Streamlit UI
   - Implement advanced analytics features
   - Performance optimization and testing

4. **Long-term (Weeks 9-12)**:
   - Production deployment on Google Cloud
   - Full market coverage (5000+ companies)
   - Advanced AI models for relationship prediction

---

**Last Updated**: June 2025  
**Document Version**: 1.0  
**Compatible with**: Financial Advisor Application v2.0+ with Neo4j integration