"""
RAG System Integration with Neo4j Graph Database
Enhances retrieval-augmented generation with graph-based context
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np

try:
    from .neo4j_client import get_graph_db, FinancialGraphDB
    from .rag_system import AdvancedRAGSystem, SemanticChunk
    GRAPH_RAG_AVAILABLE = True
except ImportError as e:
    GRAPH_RAG_AVAILABLE = False
    logging.warning(f"Graph-RAG integration not available: {e}")

logger = logging.getLogger(__name__)

class GraphEnhancedRAG:
    """RAG system enhanced with graph database relationships"""
    
    def __init__(self, rag_system: 'AdvancedRAGSystem' = None, graph_db: FinancialGraphDB = None):
        self.rag_system = rag_system
        self.graph_db = graph_db or get_graph_db()
        self.is_available = GRAPH_RAG_AVAILABLE and self.graph_db and self.graph_db.is_connected
        
        if not self.is_available:
            logger.warning("Graph-enhanced RAG not available - falling back to standard RAG")
    
    def expand_query_with_relationships(self, query: str, symbol: str, max_related: int = 5) -> List[str]:
        """Expand query to include related entities from graph"""
        if not self.is_available:
            return [query]
        
        try:
            # Find related companies
            related = self.graph_db.find_related_companies(symbol, max_hops=2, limit=max_related)
            
            # Generate expanded queries
            expanded_queries = [query]  # Include original
            
            for rel_company in related:
                company_name = rel_company.get('name', rel_company.get('symbol', ''))
                company_symbol = rel_company.get('symbol', '')
                
                # Add relationship-aware query variations
                expanded_queries.extend([
                    f"{query} {company_name}",
                    f"{query} related to {company_symbol}",
                    f"How does {query} affect {company_symbol}",
                    f"{company_name} impact on {query}",
                    f"Relationship between {symbol} and {company_symbol} regarding {query}"
                ])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in expanded_queries:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            
            return unique_queries[:10]  # Limit to prevent too many queries
            
        except Exception as e:
            logger.error(f"Error expanding query with relationships: {e}")
            return [query]
    
    def get_relationship_context(self, symbol: str, include_news: bool = True, include_supply_chain: bool = True) -> str:
        """Generate comprehensive context about company relationships"""
        if not self.is_available:
            return ""
        
        try:
            context_parts = [f"=== GRAPH-BASED RELATIONSHIP CONTEXT FOR {symbol} ==="]
            
            # Get related companies
            related = self.graph_db.find_related_companies(symbol, max_hops=2, limit=15)
            if related:
                context_parts.append("\n--- RELATED COMPANIES ---")
                for rel in related:
                    distance = rel.get('distance', 0)
                    connection_types = rel.get('connection_types', [])
                    market_cap = rel.get('marketCap', 0)
                    sector = rel.get('sector', 'Unknown')
                    
                    context_parts.append(
                        f"{rel['symbol']} ({rel['name']}) - "
                        f"Sector: {sector}, "
                        f"Market Cap: ${market_cap:,.0f}, "
                        f"Relationship Distance: {distance}, "
                        f"Connected via: {', '.join(set(connection_types))}"
                    )
            
            # Get sector companies for context
            if related:
                primary_sector = related[0].get('sector')
                if primary_sector:
                    sector_companies = self.graph_db.find_sector_companies(primary_sector, limit=10)
                    if sector_companies:
                        context_parts.append(f"\n--- {primary_sector.upper()} SECTOR COMPANIES ---")
                        for company in sector_companies[:5]:  # Top 5 by market cap
                            context_parts.append(
                                f"{company['symbol']} ({company['name']}) - "
                                f"Industry: {company.get('industry', 'Unknown')}, "
                                f"Market Cap: ${company.get('marketCap', 0):,.0f}"
                            )
            
            # Get recent news impact if requested
            if include_news:
                news_impacts = self.graph_db.find_news_impact(symbol, days=14)
                if news_impacts:
                    context_parts.append("\n--- RECENT NEWS IMPACT ---")
                    for news in news_impacts[:3]:  # Most recent 3
                        sentiment_text = "Positive" if news.get('sentiment', 0) > 0 else "Negative" if news.get('sentiment', 0) < 0 else "Neutral"
                        context_parts.append(
                            f"• {news.get('title', 'No title')} "
                            f"(Sentiment: {sentiment_text}, "
                            f"Impact Score: {news.get('impact_score', 0):.2f})"
                        )
            
            # Get supply chain risks if requested
            if include_supply_chain:
                supply_risks = self.graph_db.find_supply_chain_risk(symbol)
                if supply_risks:
                    context_parts.append("\n--- SUPPLY CHAIN RELATIONSHIPS ---")
                    countries = set()
                    industries = set()
                    for risk in supply_risks[:5]:  # Top 5 suppliers
                        if risk.get('supplier_country'):
                            countries.add(risk['supplier_country'])
                        if risk.get('supplier_industry'):
                            industries.add(risk['supplier_industry'])
                        
                        context_parts.append(
                            f"Supplier: {risk.get('supplier_name', 'Unknown')} ({risk.get('supplier_symbol', 'N/A')}) - "
                            f"Industry: {risk.get('supplier_industry', 'Unknown')}, "
                            f"Country: {risk.get('supplier_country', 'Unknown')}"
                        )
                    
                    if countries:
                        context_parts.append(f"Geographic Exposure: {', '.join(countries)}")
                    if industries:
                        context_parts.append(f"Supplier Industries: {', '.join(industries)}")
            
            # Get market influence network summary
            network = self.graph_db.get_market_influence_network(symbol, depth=1)
            if network and network.get('total_connections', 0) > 0:
                context_parts.append("\n--- MARKET INFLUENCE NETWORK ---")
                context_parts.append(f"Total Connections: {network.get('total_connections', 0)}")
                context_parts.append(f"Unique Sectors: {network.get('unique_sectors', 0)}")
                context_parts.append(f"Connected Sectors: {', '.join(network.get('sectors', []))}")
                context_parts.append(f"Average Connected Market Cap: ${network.get('average_market_cap', 0):,.0f}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error generating relationship context: {e}")
            return f"Error retrieving graph relationships for {symbol}: {str(e)}"
    
    def enhanced_retrieval(self, query: str, symbol: str, k: int = 10, 
                          use_graph_expansion: bool = True, 
                          include_relationship_context: bool = True) -> Dict[str, Any]:
        """RAG retrieval enhanced with graph relationships"""
        
        # Start with standard RAG retrieval if available
        if self.rag_system:
            try:
                standard_result = self.rag_system.query(query, k=k)
            except Exception as e:
                logger.error(f"Standard RAG query failed: {e}")
                standard_result = {
                    'context': '',
                    'chunks': [],
                    'num_chunks_retrieved': 0,
                    'retrieval_scores': []
                }
        else:
            standard_result = {
                'context': '',
                'chunks': [],
                'num_chunks_retrieved': 0,
                'retrieval_scores': []
            }
        
        # Enhanced context components
        enhanced_context_parts = []
        
        # Add standard RAG context if available
        if standard_result.get('context'):
            enhanced_context_parts.append("=== DOCUMENT-BASED CONTEXT ===")
            enhanced_context_parts.append(standard_result['context'])
        
        # Add graph-based relationship context
        if include_relationship_context and self.is_available:
            relationship_context = self.get_relationship_context(symbol)
            if relationship_context:
                enhanced_context_parts.append(relationship_context)
        
        # Perform graph-enhanced query expansion
        expanded_queries = []
        if use_graph_expansion and self.is_available:
            expanded_queries = self.expand_query_with_relationships(query, symbol)
            
            # If we have RAG system, query with expanded queries
            if self.rag_system and len(expanded_queries) > 1:
                try:
                    for expanded_query in expanded_queries[1:3]:  # Try 2 additional queries
                        expanded_result = self.rag_system.query(expanded_query, k=max(k//2, 3))
                        if expanded_result.get('context'):
                            enhanced_context_parts.append(f"=== EXPANDED QUERY CONTEXT: {expanded_query} ===")
                            enhanced_context_parts.append(expanded_result['context'])
                except Exception as e:
                    logger.warning(f"Expanded query processing failed: {e}")
        
        # Combine all context
        enhanced_context = "\n\n".join(enhanced_context_parts)
        
        # Portfolio analysis if multiple symbols detected in query
        portfolio_analysis = None
        if self.is_available:
            # Simple heuristic to detect multiple symbols in query
            words = query.upper().split()
            detected_symbols = [word for word in words if len(word) <= 5 and word.isalpha()]
            if len(detected_symbols) > 1:
                try:
                    portfolio_analysis = self.graph_db.analyze_portfolio_correlations(detected_symbols[:5])
                except Exception as e:
                    logger.warning(f"Portfolio analysis failed: {e}")
        
        # Prepare enhanced result
        result = {
            **standard_result,
            'context': enhanced_context,
            'graph_enhanced': self.is_available,
            'relationship_context_included': include_relationship_context and self.is_available,
            'query_expansion_used': use_graph_expansion and self.is_available,
            'expanded_queries': expanded_queries if use_graph_expansion else [],
            'portfolio_analysis': portfolio_analysis,
            'enhancement_timestamp': datetime.now().isoformat()
        }
        
        # Add graph-specific metadata
        if self.is_available:
            try:
                related_count = len(self.graph_db.find_related_companies(symbol, max_hops=1, limit=5))
                result['graph_metadata'] = {
                    'direct_relationships': related_count,
                    'graph_database_connected': True
                }
            except Exception as e:
                result['graph_metadata'] = {
                    'error': str(e),
                    'graph_database_connected': False
                }
        
        return result
    
    def analyze_cross_company_impact(self, primary_symbol: str, query: str, 
                                   impact_depth: int = 2) -> Dict[str, Any]:
        """Analyze how a query/event might impact related companies"""
        if not self.is_available:
            return {"error": "Graph database not available"}
        
        try:
            # Get related companies
            related = self.graph_db.find_related_companies(primary_symbol, max_hops=impact_depth, limit=20)
            
            impact_analysis = {
                "primary_company": primary_symbol,
                "query": query,
                "impact_depth": impact_depth,
                "potentially_affected_companies": len(related),
                "impact_analysis": []
            }
            
            for company in related:
                # Determine impact likelihood based on relationship distance and type
                distance = company.get('distance', 3)
                connection_types = company.get('connection_types', [])
                
                # Simple impact scoring
                impact_score = 1.0 / distance  # Closer relationships have higher impact
                if 'SUPPLIES' in connection_types:
                    impact_score *= 1.5  # Supply relationships are more impactful
                if 'COMPETES_WITH' in connection_types:
                    impact_score *= 1.2  # Competitive relationships matter
                if 'CORRELATED_WITH' in connection_types:
                    impact_score *= 1.3  # Correlated stocks move together
                
                impact_analysis["impact_analysis"].append({
                    "symbol": company.get('symbol'),
                    "name": company.get('name'),
                    "sector": company.get('sector'),
                    "relationship_distance": distance,
                    "connection_types": connection_types,
                    "estimated_impact_score": round(impact_score, 3),
                    "impact_likelihood": "High" if impact_score > 0.7 else "Medium" if impact_score > 0.4 else "Low"
                })
            
            # Sort by impact score
            impact_analysis["impact_analysis"].sort(key=lambda x: x["estimated_impact_score"], reverse=True)
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cross-company impact: {e}")
            return {"error": str(e)}
    
    def get_sector_context(self, symbol: str, max_companies: int = 10) -> str:
        """Get sector-wide context for better analysis"""
        if not self.is_available:
            return ""
        
        try:
            # First get the company's sector
            related = self.graph_db.find_related_companies(symbol, max_hops=1, limit=1)
            if not related:
                return ""
            
            sector = related[0].get('sector')
            if not sector:
                return ""
            
            # Get sector companies
            sector_companies = self.graph_db.find_sector_companies(sector, limit=max_companies)
            
            if not sector_companies:
                return ""
            
            context_parts = [f"=== {sector.upper()} SECTOR CONTEXT ==="]
            context_parts.append(f"Total Companies Analyzed: {len(sector_companies)}")
            
            # Calculate sector statistics
            market_caps = [c.get('marketCap', 0) for c in sector_companies if c.get('marketCap')]
            if market_caps:
                total_market_cap = sum(market_caps)
                avg_market_cap = total_market_cap / len(market_caps)
                context_parts.append(f"Sector Total Market Cap: ${total_market_cap:,.0f}")
                context_parts.append(f"Average Market Cap: ${avg_market_cap:,.0f}")
            
            # List top companies
            context_parts.append("\nTop Companies by Market Cap:")
            for i, company in enumerate(sector_companies[:5], 1):
                context_parts.append(
                    f"{i}. {company['symbol']} ({company['name']}) - "
                    f"${company.get('marketCap', 0):,.0f}"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting sector context: {e}")
            return ""
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of graph-enhanced RAG system"""
        result = {
            "graph_rag_available": self.is_available,
            "rag_system_available": self.rag_system is not None,
            "graph_db_available": self.graph_db is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.graph_db:
            result["graph_db_health"] = self.graph_db.health_check()
        
        if self.rag_system:
            try:
                # Test RAG system with a simple query
                test_result = self.rag_system.query("test", k=1)
                result["rag_system_responsive"] = True
            except Exception as e:
                result["rag_system_responsive"] = False
                result["rag_system_error"] = str(e)
        
        return result


# Global instance for easy access
_graph_rag_instance = None

def get_graph_enhanced_rag(rag_system: 'AdvancedRAGSystem' = None, 
                          graph_db: FinancialGraphDB = None) -> Optional[GraphEnhancedRAG]:
    """Get or create global graph-enhanced RAG instance"""
    global _graph_rag_instance
    
    if not GRAPH_RAG_AVAILABLE:
        return None
    
    if _graph_rag_instance is None:
        try:
            _graph_rag_instance = GraphEnhancedRAG(rag_system, graph_db)
        except Exception as e:
            logger.warning(f"Could not initialize graph-enhanced RAG: {e}")
            return None
    
    return _graph_rag_instance if _graph_rag_instance.is_available else None