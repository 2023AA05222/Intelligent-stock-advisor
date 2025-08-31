"""
MCP Server Extension for Neo4j Graph Database Integration
Adds graph-based financial analysis tools to the MCP server
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from mcp.types import Tool, TextContent

# Import our Neo4j client
try:
    from .neo4j_client import get_graph_db, FinancialGraphDB
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logging.warning("Graph database client not available")

logger = logging.getLogger(__name__)

class GraphMCPExtension:
    """MCP extension for graph database operations"""
    
    def __init__(self):
        self.graph_db = get_graph_db() if GRAPH_AVAILABLE else None
    
    def get_graph_tools(self) -> List[Tool]:
        """Get list of graph-related MCP tools"""
        if not GRAPH_AVAILABLE:
            return []
        
        return [
            Tool(
                name="get_company_relationships",
                description="Get relationship network for a company using graph database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                        },
                        "max_hops": {
                            "type": "integer",
                            "default": 2,
                            "description": "Maximum relationship hops (1-3)",
                            "minimum": 1,
                            "maximum": 3
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Maximum number of related companies to return",
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="analyze_portfolio_relationships",
                description="Analyze relationships and correlations between portfolio holdings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock ticker symbols",
                            "minItems": 2,
                            "maxItems": 20
                        }
                    },
                    "required": ["symbols"]
                }
            ),
            Tool(
                name="find_sector_companies",
                description="Find all companies in a specific sector using graph database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sector": {
                            "type": "string",
                            "description": "Sector name (e.g., Technology, Healthcare, Finance)"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 50,
                            "description": "Maximum number of companies to return",
                            "minimum": 1,
                            "maximum": 200
                        }
                    },
                    "required": ["sector"]
                }
            ),
            Tool(
                name="get_news_impact_analysis",
                description="Get news events affecting a company with impact analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "days": {
                            "type": "integer",
                            "default": 30,
                            "description": "Number of days to look back for news",
                            "minimum": 1,
                            "maximum": 365
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="analyze_supply_chain_risk",
                description="Analyze potential supply chain risks for a company",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_market_influence_network",
                description="Get network of market influence around a company",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "depth": {
                            "type": "integer",
                            "default": 2,
                            "description": "Network depth (1-3)",
                            "minimum": 1,
                            "maximum": 3
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_graph_statistics",
                description="Get overall graph database statistics and health",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            ),
            Tool(
                name="index_company_data",
                description="Index company data into the graph database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "auto_discover_relationships": {
                            "type": "boolean",
                            "default": True,
                            "description": "Automatically discover and create relationships"
                        }
                    },
                    "required": ["symbol"]
                }
            )
        ]
    
    async def handle_graph_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle graph tool execution"""
        if not GRAPH_AVAILABLE or not self.graph_db:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Graph database not available or not connected",
                    "suggestion": "Check Neo4j connection and ensure database is running"
                })
            )]
        
        try:
            if name == "get_company_relationships":
                return await self._get_company_relationships(arguments)
            elif name == "analyze_portfolio_relationships":
                return await self._analyze_portfolio_relationships(arguments)
            elif name == "find_sector_companies":
                return await self._find_sector_companies(arguments)
            elif name == "get_news_impact_analysis":
                return await self._get_news_impact_analysis(arguments)
            elif name == "analyze_supply_chain_risk":
                return await self._analyze_supply_chain_risk(arguments)
            elif name == "get_market_influence_network":
                return await self._get_market_influence_network(arguments)
            elif name == "get_graph_statistics":
                return await self._get_graph_statistics(arguments)
            elif name == "index_company_data":
                return await self._index_company_data(arguments)
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown graph tool: {name}"})
                )]
        
        except Exception as e:
            logger.error(f"Error executing graph tool {name}: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Graph tool execution failed: {str(e)}",
                    "tool": name
                })
            )]
    
    async def _get_company_relationships(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get company relationship network"""
        symbol = arguments.get("symbol", "").upper()
        max_hops = arguments.get("max_hops", 2)
        limit = arguments.get("limit", 20)
        
        if not symbol:
            return [TextContent(type="text", text=json.dumps({"error": "Symbol is required"}))]
        
        relationships = self.graph_db.find_related_companies(symbol, max_hops, limit)
        
        result = {
            "symbol": symbol,
            "max_hops": max_hops,
            "relationship_count": len(relationships),
            "relationships": relationships,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def _analyze_portfolio_relationships(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Analyze portfolio relationships"""
        symbols = [s.upper() for s in arguments.get("symbols", [])]
        
        if len(symbols) < 2:
            return [TextContent(type="text", text=json.dumps({
                "error": "At least 2 symbols required for portfolio analysis"
            }))]
        
        correlations = self.graph_db.analyze_portfolio_correlations(symbols)
        
        # Calculate portfolio risk metrics
        total_relationships = len(correlations)
        high_correlation_count = len([c for c in correlations 
                                    if c.get('correlation') and abs(c['correlation']) > 0.7])
        
        result = {
            "portfolio_symbols": symbols,
            "total_relationships": total_relationships,
            "high_correlation_pairs": high_correlation_count,
            "risk_level": "High" if high_correlation_count > len(symbols) / 2 else "Medium" if high_correlation_count > 0 else "Low",
            "relationships": correlations,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def _find_sector_companies(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Find companies in a sector"""
        sector = arguments.get("sector", "")
        limit = arguments.get("limit", 50)
        
        if not sector:
            return [TextContent(type="text", text=json.dumps({"error": "Sector is required"}))]
        
        companies = self.graph_db.find_sector_companies(sector, limit)
        
        result = {
            "sector": sector,
            "company_count": len(companies),
            "companies": companies,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def _get_news_impact_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get news impact analysis"""
        symbol = arguments.get("symbol", "").upper()
        days = arguments.get("days", 30)
        
        if not symbol:
            return [TextContent(type="text", text=json.dumps({"error": "Symbol is required"}))]
        
        news_impacts = self.graph_db.find_news_impact(symbol, days)
        
        # Calculate impact summary
        total_news = len(news_impacts)
        positive_sentiment = len([n for n in news_impacts if n.get('sentiment', 0) > 0])
        negative_sentiment = len([n for n in news_impacts if n.get('sentiment', 0) < 0])
        avg_impact = sum(n.get('impact_score', 0) for n in news_impacts) / max(total_news, 1)
        
        result = {
            "symbol": symbol,
            "analysis_period_days": days,
            "total_news_events": total_news,
            "sentiment_distribution": {
                "positive": positive_sentiment,
                "negative": negative_sentiment,
                "neutral": total_news - positive_sentiment - negative_sentiment
            },
            "average_impact_score": round(avg_impact, 3),
            "news_events": news_impacts,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def _analyze_supply_chain_risk(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Analyze supply chain risks"""
        symbol = arguments.get("symbol", "").upper()
        
        if not symbol:
            return [TextContent(type="text", text=json.dumps({"error": "Symbol is required"}))]
        
        supply_chain_risks = self.graph_db.find_supply_chain_risk(symbol)
        
        # Analyze risk factors
        unique_countries = set(r.get('supplier_country') for r in supply_chain_risks if r.get('supplier_country'))
        unique_industries = set(r.get('supplier_industry') for r in supply_chain_risks if r.get('supplier_industry'))
        
        result = {
            "symbol": symbol,
            "total_suppliers": len(supply_chain_risks),
            "unique_countries": len(unique_countries),
            "countries": list(unique_countries),
            "unique_supplier_industries": len(unique_industries),
            "supplier_industries": list(unique_industries),
            "geographic_risk": "High" if len(unique_countries) < 3 else "Medium" if len(unique_countries) < 6 else "Low",
            "industry_concentration_risk": "High" if len(unique_industries) < 3 else "Medium" if len(unique_industries) < 6 else "Low",
            "supply_chain_details": supply_chain_risks,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def _get_market_influence_network(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get market influence network"""
        symbol = arguments.get("symbol", "").upper()
        depth = arguments.get("depth", 2)
        
        if not symbol:
            return [TextContent(type="text", text=json.dumps({"error": "Symbol is required"}))]
        
        network = self.graph_db.get_market_influence_network(symbol, depth)
        
        return [TextContent(type="text", text=json.dumps(network, indent=2))]
    
    async def _get_graph_statistics(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get graph database statistics"""
        stats = self.graph_db.get_graph_statistics()
        health = self.graph_db.health_check()
        
        result = {
            "database_health": health,
            "statistics": stats,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def _index_company_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Index company data into graph database"""
        symbol = arguments.get("symbol", "").upper()
        auto_discover = arguments.get("auto_discover_relationships", True)
        
        if not symbol:
            return [TextContent(type="text", text=json.dumps({"error": "Symbol is required"}))]
        
        try:
            # Import here to avoid circular imports
            import yfinance as yf
            
            # Fetch company data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return [TextContent(type="text", text=json.dumps({
                    "error": f"Could not fetch data for symbol: {symbol}"
                }))]
            
            # Prepare company data for graph
            company_data = {
                "symbol": symbol,
                "name": info.get("longName", info.get("shortName", symbol)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "marketCap": info.get("marketCap"),
                "employees": info.get("fullTimeEmployees"),
                "headquarters": f"{info.get('city', '')}, {info.get('country', '')}".strip(", "),
                "description": info.get("longBusinessSummary", "")[:500],  # Truncate long descriptions
                "currency": info.get("currency", "USD"),
                "country": info.get("country", "Unknown"),
                "website": info.get("website", "")
            }
            
            # Create company node
            success = self.graph_db.create_company(company_data)
            
            # Create sector and industry if they don't exist
            if company_data["sector"] and company_data["industry"]:
                self.graph_db.create_sector_industry(company_data["sector"], company_data["industry"])
            
            result = {
                "symbol": symbol,
                "indexed": success,
                "company_data": company_data,
                "auto_discover_enabled": auto_discover,
                "indexing_timestamp": datetime.now().isoformat()
            }
            
            if auto_discover:
                result["note"] = "Relationship discovery will be implemented in future updates"
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Failed to index company data: {str(e)}",
                "symbol": symbol
            }))]