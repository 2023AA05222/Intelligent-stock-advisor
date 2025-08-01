"""
Neo4j Graph Database Client for Financial Analysis
Provides graph-based storage and retrieval for financial relationships
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available. Install with: pip install neo4j")

logger = logging.getLogger(__name__)

class FinancialGraphDB:
    """Neo4j client for financial relationship management"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """Initialize Neo4j connection"""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not installed. Install with: pip install neo4j")
        
        # Default connection parameters
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'financialpass')
        
        self.driver = None
        self.is_connected = False
        
        try:
            self.connect()
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
    
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=basic_auth(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            self.is_connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Create indexes and constraints
            self._setup_schema()
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.is_connected = False
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            logger.info("Neo4j connection closed")
    
    def _setup_schema(self):
        """Create indexes and constraints for optimal performance"""
        schema_queries = [
            # Constraints (unique identifiers)
            "CREATE CONSTRAINT company_symbol IF NOT EXISTS FOR (c:Company) REQUIRE c.symbol IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT industry_name IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector)",
            "CREATE INDEX company_market_cap IF NOT EXISTS FOR (c:Company) ON (c.marketCap)",
            "CREATE INDEX news_date IF NOT EXISTS FOR (n:NewsEvent) ON (n.date)",
            "CREATE INDEX news_sentiment IF NOT EXISTS FOR (n:NewsEvent) ON (n.sentiment)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            
            # Composite indexes
            "CREATE INDEX company_sector_market_cap IF NOT EXISTS FOR (c:Company) ON (c.sector, c.marketCap)",
        ]
        
        with self.driver.session() as session:
            for query in schema_queries:
                try:
                    session.run(query)
                except Exception as e:
                    logger.debug(f"Schema setup query failed (may already exist): {query} - {e}")
    
    def create_company(self, company_data: Dict[str, Any]) -> bool:
        """Create or update company node"""
        if not self.is_connected:
            return False
        
        try:
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
                        c.currency = $currency,
                        c.country = $country,
                        c.website = $website,
                        c.updated = datetime()
                    RETURN c.symbol as symbol
                """, **company_data)
                
                return bool(result.single())
        except Exception as e:
            logger.error(f"Error creating company {company_data.get('symbol')}: {e}")
            return False
    
    def create_sector_industry(self, sector: str, industry: str) -> bool:
        """Create sector and industry nodes with relationship"""
        if not self.is_connected:
            return False
        
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (s:Sector {name: $sector})
                    MERGE (i:Industry {name: $industry})
                    MERGE (i)-[:PART_OF]->(s)
                """, sector=sector, industry=industry)
                return True
        except Exception as e:
            logger.error(f"Error creating sector/industry {sector}/{industry}: {e}")
            return False
    
    def create_news_event(self, news_data: Dict[str, Any], affected_symbols: List[str] = None) -> bool:
        """Create news event and link to affected companies"""
        if not self.is_connected:
            return False
        
        try:
            with self.driver.session() as session:
                # Create news event
                result = session.run("""
                    CREATE (n:NewsEvent {
                        id: $id,
                        title: $title,
                        summary: $summary,
                        date: datetime($date),
                        sentiment: $sentiment,
                        source: $source,
                        impact_score: $impact_score,
                        url: $url
                    })
                    RETURN n.id as id
                """, **news_data)
                
                if not result.single():
                    return False
                
                # Link to affected companies
                if affected_symbols:
                    for symbol in affected_symbols:
                        session.run("""
                            MATCH (n:NewsEvent {id: $news_id})
                            MATCH (c:Company {symbol: $symbol})
                            MERGE (n)-[:AFFECTS {impact_score: $impact_score}]->(c)
                        """, news_id=news_data['id'], symbol=symbol, 
                             impact_score=news_data.get('impact_score', 0.5))
                
                return True
        except Exception as e:
            logger.error(f"Error creating news event: {e}")
            return False
    
    def create_relationship(self, from_symbol: str, to_symbol: str, 
                          rel_type: str, properties: Dict = None) -> bool:
        """Create relationship between companies"""
        if not self.is_connected:
            return False
        
        properties = properties or {}
        try:
            with self.driver.session() as session:
                result = session.run(f"""
                    MATCH (a:Company {{symbol: $from_symbol}})
                    MATCH (b:Company {{symbol: $to_symbol}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r += $properties, r.updated = datetime()
                    RETURN r
                """, from_symbol=from_symbol, to_symbol=to_symbol, properties=properties)
                return bool(result.single())
        except Exception as e:
            logger.error(f"Error creating relationship {from_symbol}-[{rel_type}]->{to_symbol}: {e}")
            return False
    
    def create_correlation(self, symbol1: str, symbol2: str, correlation: float, 
                          timeframe: str = "1y") -> bool:
        """Create correlation relationship between companies"""
        return self.create_relationship(
            symbol1, symbol2, "CORRELATED_WITH",
            {"coefficient": correlation, "timeframe": timeframe, "strength": abs(correlation)}
        )
    
    def find_related_companies(self, symbol: str, max_hops: int = 2, limit: int = 20) -> List[Dict]:
        """Find companies related to given symbol"""
        if not self.is_connected:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(f"""
                    MATCH path = (start:Company {{symbol: $symbol}})
                    -[*1..{max_hops}]-(related:Company)
                    WHERE start <> related
                    WITH related, path, 
                         [r IN relationships(path) | type(r)] as rel_types,
                         length(path) as distance
                    RETURN DISTINCT related.symbol as symbol,
                           related.name as name,
                           related.sector as sector,
                           related.marketCap as marketCap,
                           distance,
                           rel_types as connection_types
                    ORDER BY distance, related.marketCap DESC
                    LIMIT $limit
                """, symbol=symbol, limit=limit)
                
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error finding related companies for {symbol}: {e}")
            return []
    
    def find_sector_companies(self, sector: str, limit: int = 50) -> List[Dict]:
        """Find all companies in a sector"""
        if not self.is_connected:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Company)
                    WHERE c.sector = $sector
                    RETURN c.symbol as symbol,
                           c.name as name,
                           c.industry as industry,
                           c.marketCap as marketCap
                    ORDER BY c.marketCap DESC
                    LIMIT $limit
                """, sector=sector, limit=limit)
                
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error finding companies in sector {sector}: {e}")
            return []
    
    def analyze_portfolio_correlations(self, symbols: List[str]) -> List[Dict]:
        """Analyze correlations and relationships between portfolio holdings"""
        if not self.is_connected or not symbols:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (s1:Company)-[r]-(s2:Company)
                    WHERE s1.symbol IN $symbols AND s2.symbol IN $symbols
                      AND s1.symbol < s2.symbol  // Avoid duplicates
                    RETURN s1.symbol as stock1, 
                           s1.name as name1,
                           s2.symbol as stock2,
                           s2.name as name2,
                           type(r) as relationship_type,
                           r.coefficient as correlation,
                           r.strength as strength,
                           r.impact_score as impact_score
                    ORDER BY coalesce(
                        CASE WHEN r.coefficient IS NOT NULL THEN abs(r.coefficient) END,
                        r.strength, 
                        r.impact_score, 
                        0
                    ) DESC
                """, symbols=symbols)
                
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error analyzing portfolio correlations: {e}")
            return []
    
    def find_news_impact(self, symbol: str, days: int = 30) -> List[Dict]:
        """Find news events affecting a company in recent days"""
        if not self.is_connected:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n:NewsEvent)-[r:AFFECTS]->(c:Company {symbol: $symbol})
                    WHERE n.date > datetime() - duration({days: $days})
                    RETURN n.title as title,
                           n.summary as summary,
                           n.date as date,
                           n.sentiment as sentiment,
                           n.source as source,
                           r.impact_score as impact_score
                    ORDER BY n.date DESC
                    LIMIT 20
                """, symbol=symbol, days=days)
                
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error finding news impact for {symbol}: {e}")
            return []
    
    def find_supply_chain_risk(self, symbol: str) -> List[Dict]:
        """Find potential supply chain risks for a company"""
        if not self.is_connected:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH path = (company:Company {symbol: $symbol})
                    <-[:SUPPLIES]-(supplier:Company)
                    -[:BELONGS_TO]->(industry:Industry)
                    OPTIONAL MATCH (supplier)-[:OPERATES_IN]->(country:Country)
                    RETURN supplier.symbol as supplier_symbol,
                           supplier.name as supplier_name,
                           industry.name as supplier_industry,
                           country.name as supplier_country,
                           length(path) as dependency_level
                    ORDER BY dependency_level, supplier.marketCap DESC
                    LIMIT 20
                """, symbol=symbol)
                
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error finding supply chain risks for {symbol}: {e}")
            return []
    
    def get_market_influence_network(self, symbol: str, depth: int = 2) -> Dict[str, Any]:
        """Get network of market influence around a company"""
        if not self.is_connected:
            return {}
        
        try:
            with self.driver.session() as session:
                # Get the influence network
                result = session.run(f"""
                    MATCH path = (center:Company {{symbol: $symbol}})
                    -[*1..{depth}]-(connected:Company)
                    WHERE center <> connected
                    WITH path, connected, center,
                         [r IN relationships(path) | {{type: type(r), properties: properties(r)}}] as rels
                    RETURN center.symbol as center_symbol,
                           connected.symbol as connected_symbol,
                           connected.name as connected_name,
                           connected.sector as connected_sector,
                           connected.marketCap as market_cap,
                           length(path) as distance,
                           rels as relationships
                    ORDER BY distance, market_cap DESC
                    LIMIT 100
                """, symbol=symbol)
                
                network_data = [record.data() for record in result]
                
                # Calculate network statistics
                total_connections = len(network_data)
                sectors = set(record['connected_sector'] for record in network_data if record['connected_sector'])
                avg_market_cap = np.mean([record['market_cap'] for record in network_data if record['market_cap']])
                
                return {
                    "center_company": symbol,
                    "total_connections": total_connections,
                    "unique_sectors": len(sectors),
                    "sectors": list(sectors),
                    "average_market_cap": float(avg_market_cap) if not np.isnan(avg_market_cap) else 0,
                    "network_data": network_data,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting market influence network for {symbol}: {e}")
            return {}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics"""
        if not self.is_connected:
            return {}
        
        try:
            with self.driver.session() as session:
                # Count nodes by type
                node_counts = {}
                for node_type in ['Company', 'Person', 'Sector', 'Industry', 'NewsEvent']:
                    result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                    node_counts[node_type] = result.single()['count']
                
                # Count relationships by type
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                    ORDER BY count DESC
                """)
                relationship_counts = {record['rel_type']: record['count'] for record in rel_result}
                
                # Get largest companies
                top_companies = session.run("""
                    MATCH (c:Company)
                    WHERE c.marketCap IS NOT NULL
                    RETURN c.symbol, c.name, c.marketCap
                    ORDER BY c.marketCap DESC
                    LIMIT 10
                """)
                top_companies_list = [record.data() for record in top_companies]
                
                return {
                    "node_counts": node_counts,
                    "relationship_counts": relationship_counts,
                    "top_companies": top_companies_list,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(relationship_counts.values()),
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity"""
        try:
            if not self.driver:
                return {"status": "disconnected", "error": "No driver instance"}
            
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()['test']
                
                if test_value == 1:
                    stats = self.get_graph_statistics()
                    return {
                        "status": "healthy",
                        "connection": "active",
                        "uri": self.uri,
                        "total_nodes": stats.get('total_nodes', 0),
                        "total_relationships": stats.get('total_relationships', 0)
                    }
                else:
                    return {"status": "unhealthy", "error": "Unexpected test result"}
                    
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global instance for easy access
_graph_db_instance = None

def get_graph_db(uri: str = None, username: str = None, password: str = None) -> Optional[FinancialGraphDB]:
    """Get or create global graph database instance"""
    global _graph_db_instance
    
    if not NEO4J_AVAILABLE:
        return None
    
    if _graph_db_instance is None:
        try:
            _graph_db_instance = FinancialGraphDB(uri, username, password)
        except Exception as e:
            logger.warning(f"Could not initialize graph database: {e}")
            return None
    
    return _graph_db_instance if _graph_db_instance.is_connected else None

def close_graph_db():
    """Close global graph database connection"""
    global _graph_db_instance
    if _graph_db_instance:
        _graph_db_instance.close()
        _graph_db_instance = None