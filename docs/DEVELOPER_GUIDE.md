# Developer Guide - Financial Advisor Application

## Table of Contents
1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Neo4j Graph Database Integration](#neo4j-graph-database-integration)
4. [Main Application Analysis](#main-application-analysis)
5. [MCP Server Implementation](#mcp-server-implementation)
6. [Graph-Enhanced RAG System](#graph-enhanced-rag-system)
7. [Integration Patterns](#integration-patterns)
8. [Development Workflows](#development-workflows)
9. [Testing and Quality](#testing-and-quality)

---

## Project Overview

### Architecture Summary
The Financial Advisor Application is built on a modular architecture with four main components:
1. **Streamlit Frontend** (`streamlit_app.py`) - User interface with relationship visualization
2. **MCP Financial Server** (`src/mcp_financial_server/`) - Data access layer with graph tools
3. **Neo4j Graph Database** (`src/neo4j_client.py`) - Relationship storage and analysis
4. **Graph-Enhanced RAG System** (`src/rag_graph_integration.py`) - AI enhancement with relationship context

### Technology Stack
- **Frontend**: Streamlit 1.28+, Plotly for charts and network visualization
- **Backend**: Python 3.11, FastAPI for MCP server
- **Graph Database**: Neo4j 5.13+, Cypher query language
- **AI/ML**: Google Gemini, Sentence Transformers, HNSW, Graph-RAG integration
- **Data**: Yahoo Finance (yfinance), Pandas, NumPy, Neo4j relationships

---

## Codebase Structure

```
fl_financial_advisor/
├── streamlit_app.py              # Main application entry point (with Relationships tab)
├── src/
│   ├── mcp_financial_server/     # MCP server implementation
│   │   ├── __init__.py          # Package initialization
│   │   ├── __main__.py          # CLI entry point
│   │   └── server.py            # MCP server logic
│   ├── neo4j_client.py          # Neo4j graph database client
│   ├── rag_graph_integration.py # Graph-enhanced RAG system
│   ├── graph_mcp_extension.py   # MCP server graph tools
│   ├── rag_integration.py       # RAG-Streamlit integration
│   └── rag_system.py           # Core RAG implementation
├── test_server.py              # MCP server tests
├── test_chat.py               # AI chat functionality test
├── requirements.txt           # Dependencies
├── Dockerfile                # Container configuration
├── Makefile                  # Build automation
└── docs/                     # Documentation
```

---

## Neo4j Graph Database Integration

### Core Components

#### 1. Neo4j Client (`src/neo4j_client.py`)

The Neo4j client provides a comprehensive interface to the graph database:

```python
from src.neo4j_client import FinancialGraphDB, get_graph_db

# Initialize connection
graph_db = FinancialGraphDB(
    uri="bolt://localhost:7687",
    username="neo4j", 
    password="financialpass"
)

# Basic operations
graph_db.create_company(company_data)
graph_db.create_relationship(from_symbol, to_symbol, rel_type, properties)
relationships = graph_db.find_related_companies(symbol, max_hops=2)
```

**Key Features:**
- **Graceful Degradation**: Application works without Neo4j connection
- **Connection Pooling**: Efficient resource management
- **Schema Management**: Automatic index and constraint creation
- **Performance Optimization**: Strategic caching and query limits

#### 2. Graph Schema Design

**Node Types:**
```cypher
(:Company {symbol, name, sector, industry, marketCap, ...})
(:Person {name, title, biography, ...})
(:Sector {name, description})
(:Industry {name, sector, description})
(:NewsEvent {title, date, sentiment, impact_score, ...})
```

**Relationship Types:**
```cypher
(company1)-[:OWNS {percentage, since}]->(company2)
(company1)-[:SUPPLIES {volume, since}]->(company2)
(company1)-[:COMPETES_WITH {intensity}]->(company2)
(company1)-[:CORRELATED_WITH {coefficient, timeframe}]->(company2)
(news)-[:AFFECTS {impact_score}]->(company)
```

#### 3. Graph-Enhanced RAG Integration (`src/rag_graph_integration.py`)

```python
from src.rag_graph_integration import GraphEnhancedRAG

# Initialize with existing RAG system and graph database
enhanced_rag = GraphEnhancedRAG(rag_system, graph_db)

# Enhanced retrieval with relationship context
result = enhanced_rag.enhanced_retrieval(
    query="What are Apple's supply chain risks?",
    symbol="AAPL",
    k=10
)
```

**Enhancement Strategies:**
- **Relationship Expansion**: Include related companies in context
- **Multi-hop Traversal**: Navigate complex relationship networks
- **Context Enrichment**: Add relationship metadata to RAG context
- **Query Rewriting**: Expand queries with relationship-aware terms

### Development Patterns

#### 1. Graph Query Development

```python
# Example: Finding supply chain dependencies
def find_supply_chain_risks(self, symbol: str) -> List[Dict]:
    with self.driver.session() as session:
        result = session.run("""
            MATCH path = (company:Company {symbol: $symbol})
            <-[:SUPPLIES]-(supplier:Company)
            -[:BELONGS_TO]->(industry:Industry)
            RETURN supplier.symbol, supplier.name, industry.name,
                   length(path) as dependency_level
            ORDER BY dependency_level, supplier.marketCap DESC
            LIMIT 20
        """, symbol=symbol)
        return [record.data() for record in result]
```

#### 2. Relationship Discovery

```python
# Example: Creating correlations from price data
def create_correlation_relationships(self, symbols: List[str], timeframe: str = "1y"):
    # Fetch price data for all symbols
    price_data = self.get_price_data(symbols, timeframe)
    
    # Calculate correlation matrix
    correlation_matrix = price_data.corr()
    
    # Store significant correlations in graph
    for symbol1 in symbols:
        for symbol2 in symbols:
            if symbol1 != symbol2:
                correlation = correlation_matrix.loc[symbol1, symbol2]
                if abs(correlation) > 0.5:  # Significant correlation
                    self.create_correlation(symbol1, symbol2, correlation, timeframe)
```

#### 3. Performance Optimization

```python
# Caching for frequently accessed relationships
@lru_cache(maxsize=100)
def get_cached_relationships(self, symbol: str, max_hops: int = 2):
    return self.find_related_companies(symbol, max_hops)

# Batch operations for efficiency
def batch_create_companies(self, companies_data: List[Dict]):
    with self.driver.session() as session:
        session.run("""
            UNWIND $companies as company
            MERGE (c:Company {symbol: company.symbol})
            SET c += company
        """, companies=companies_data)
```

---

## Main Application Analysis

### File: `streamlit_app.py`

This is the main application file that orchestrates the entire user experience. Let's analyze it section by section:

#### Import Section (Lines 1-48)
```python
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import numpy as np
import io
import os
import google.generativeai as genai
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
```

**Analysis:**
- **Core Dependencies**: Streamlit for UI, yfinance for data, plotly for charts
- **AI Integration**: Google Generative AI for LLM capabilities
- **Data Processing**: Pandas/NumPy for numerical operations
- **Environment**: dotenv for configuration management

#### RAG System Integration (Lines 16-48)
```python
# RAG System Integration
RAG_AVAILABLE = False
try:
    from src.rag_integration import (
        get_rag_integration, display_rag_status_sidebar, 
        auto_index_on_data_fetch, get_rag_enhanced_context,
        rag_enhanced_prompt_modification
    )
    RAG_AVAILABLE = True
except ImportError as e:
    # RAG system is optional - app works without it
    import logging
    logging.info(f"RAG system not available (running in basic mode): {e}")
    
    # Create fallback functions so the app doesn't break
    def get_rag_integration():
        return None
    
    def display_rag_status_sidebar():
        with st.sidebar:
            st.markdown("### 🔍 RAG System")
            st.info("⚠️ Basic Mode (Install dependencies for enhanced RAG)")
    
    def auto_index_on_data_fetch(*args, **kwargs):
        pass  # No-op
    
    def get_rag_enhanced_context(symbol, stock_data, stock_info, news_data, financial_statements, user_question):
        context = create_financial_context(symbol, stock_data, stock_info, news_data, financial_statements)
        return context, False
    
    def rag_enhanced_prompt_modification(base_prompt, context, user_question):
        return base_prompt.format(context=context, user_question=user_question)
```

**Design Pattern: Graceful Degradation**
- The application can run with or without RAG dependencies
- Uses try/catch to detect RAG availability
- Provides fallback functions that maintain API compatibility
- Logs the degradation for debugging purposes

#### Currency Symbol Mapping (Lines 52-79)
```python
def get_currency_symbol(currency_code):
    """Get currency symbol for given currency code"""
    currency_symbols = {
        'USD': '$',
        'INR': '₹',
        'EUR': '€',
        # ... more currencies
    }
    return currency_symbols.get(currency_code, currency_code + ' ')
```

**Purpose**: Provides user-friendly currency display across different markets

#### Gemini AI Initialization (Lines 81-116)
```python
@st.cache_resource
def initialize_gemini():
    """Initialize Gemini AI with API key or service account credentials"""
    try:
        # Try API key first (more straightforward for Gemini)
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            return model
        
        # Fallback to service account credentials
        service_account_path = "/home/rajan/CREDENTIALS/rtc-lms-ef961e47471d.json"
        
        if os.path.exists(service_account_path):
            # Load service account credentials
            credentials = Credentials.from_service_account_file(
                service_account_path,
                scopes=['https://www.googleapis.com/auth/generative-language.retriever']
            )
            # ... credential refresh logic
```

**Key Design Decisions:**
- **Caching**: `@st.cache_resource` ensures single initialization
- **Multiple Auth Methods**: API key preferred, service account fallback
- **Error Handling**: Graceful degradation if no credentials available

#### Financial Context Creation (Lines 118-233)
```python
def create_financial_context(symbol, stock_data, stock_info, news_data, financial_statements):
    """Create comprehensive context from all available data sources"""
    context_parts = []
    
    # Complete stock information
    if stock_info:
        context_parts.append(f"=== COMPLETE STOCK INFORMATION FOR {symbol} ===")
        # Include all available stock info fields
        for key, value in stock_info.items():
            if value is not None:
                context_parts.append(f"{key}: {value}")
        context_parts.append("")
    
    # All stock price data
    if stock_data is not None and not stock_data.empty:
        context_parts.append("=== COMPLETE PRICE DATA ===")
        context_parts.append(f"Total data points: {len(stock_data)}")
        context_parts.append(f"Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
        context_parts.append("")
        
        # Include all price data
        context_parts.append("Full price history:")
        for idx, row in stock_data.iterrows():
            context_parts.append(f"{idx.strftime('%Y-%m-%d %H:%M:%S')}: Open={row['Open']:.2f}, High={row['High']:.2f}, Low={row['Low']:.2f}, Close={row['Close']:.2f}, Volume={row['Volume']:,}")
        context_parts.append("")
```

**Critical Function**: This creates the complete context that gets passed to the AI
- **Comprehensive Data**: Includes ALL available data, not just summaries
- **Structured Format**: Clear sections with headers for AI parsing
- **Data Validation**: Checks for None/empty data before processing

#### Session State Management (Lines 234-262)
```python
# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'stock_info' not in st.session_state:
    st.session_state.stock_info = None
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = "AAPL"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'financial_statements' not in st.session_state:
    st.session_state.financial_statements = {}
```

**Streamlit Pattern**: Session state maintains data across user interactions
- **Persistence**: Data survives page refreshes and user actions
- **Default Values**: Sensible defaults prevent None errors
- **Type Consistency**: Each variable has a consistent expected type

#### News Fetching (Lines 264-301)
```python
@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_news(symbol):
    """Fetch news for a given stock symbol"""
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        
        # Process news articles
        processed_news = []
        for article in news:
            # Extract and clean news data
            processed_article = {
                'title': article.get('title', 'No title'),
                'summary': article.get('summary', 'No summary available'),
                'publisher': article.get('publisher', 'Unknown'),
                'published': article.get('providerPublishTime', 0),
                'link': article.get('link', '')
            }
            processed_news.append(processed_article)
        
        return processed_news, None
    except Exception as e:
        return [], str(e)
```

**Caching Strategy**: 15-minute TTL balances freshness with performance
**Error Handling**: Returns empty list and error message on failure
**Data Cleaning**: Ensures consistent data structure with fallback values

#### Financial Statement Fetching (Lines 303-355)
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_financial_statements(symbol):
    """Fetch comprehensive financial statements"""
    try:
        stock = yf.Ticker(symbol)
        
        statements = {}
        
        # Try to fetch each type of financial statement
        try:
            statements['quarterly_financials'] = stock.quarterly_financials
        except:
            statements['quarterly_financials'] = pd.DataFrame()
        
        try:
            statements['financials'] = stock.financials
        except:
            statements['financials'] = pd.DataFrame()
        
        # ... similar for other statements
        
        return statements, None
    except Exception as e:
        return {}, str(e)
```

**Robust Error Handling**: Each statement type is fetched independently
**Longer Caching**: Financial statements change less frequently (1 hour TTL)
**Comprehensive Coverage**: Quarterly and annual data for all statement types

#### Sidebar Configuration (Lines 400-453)
```python
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Stock symbol input
    symbol = st.text_input(
        "Stock Symbol (e.g., AAPL, MSFT, GOOGL)",
        value=st.session_state.current_symbol,
        help="Enter the stock ticker symbol"
    ).upper()
    
    st.markdown("### Time Period")
    period_options = {
        "1 Day": "1d",
        "5 Days": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    selected_period = st.selectbox(
        "Select Period",
        options=list(period_options.keys()),
        index=7  # Default to 5 Years
    )
```

**UI Design Patterns**:
- **Clear Labels**: Descriptive text and help tooltips
- **Sensible Defaults**: 5 Years provides good long-term perspective
- **Input Validation**: `.upper()` standardizes symbol format

#### Data Fetching Logic (Lines 455-487)
```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period, interval):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        info = stock.info
        return data, info, None
    except Exception as e:
        return None, None, str(e)

# Main content area
if fetch_button or (st.session_state.stock_data is None and symbol):
    if not symbol:
        st.error("Please enter a stock symbol")
    else:
        with st.spinner(f"Fetching data for {symbol}..."):
            data, info, error = fetch_stock_data(symbol, period_options[selected_period], interval_options[selected_interval])
            
            if error:
                st.error(f"Error fetching data: {error}")
            elif data is None or data.empty:
                st.error(f"No data found for symbol: {symbol}")
            else:
                st.session_state.stock_data = data
                st.session_state.stock_info = info
                st.session_state.current_symbol = symbol
                st.success(f"Data fetched successfully for {symbol}")
                
                # Auto-index data for RAG if available
                if RAG_AVAILABLE:
                    auto_index_on_data_fetch(symbol, data, info, [])
```

**User Experience Focus**:
- **Loading Indicators**: Spinner shows progress during data fetch
- **Error Feedback**: Clear error messages for different failure modes
- **Success Confirmation**: Positive feedback when data loads successfully
- **Automatic RAG Indexing**: Transparently enhances AI capabilities when available

#### Tab Structure (Lines 500-659)
```python
# Display data if available
if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
    data = st.session_state.stock_data
    info = st.session_state.stock_info or {}
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🤖 AI Chat", "📈 Chart", "📋 Data Table", "📊 Statistics", 
        "💼 Financial Statements", "📰 News", "📤 Export"
    ])
```

**Tab Organization**: Logical flow from AI insights to raw data to export

#### AI Chat Implementation (Lines 520-659)
```python
with tab1:
    st.subheader("🤖 AI Financial Analyst")
    
    # Get or initialize Gemini model
    gemini_model = initialize_gemini()
    
    if gemini_model:
        # Chat input
        user_question = st.text_input(
            "Ask me anything about this stock:",
            placeholder="e.g., What's the trend analysis? Is this a good buy?",
            key="chat_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send", type="primary")
        
        # Process user input
        if send_button and user_question:
            with st.spinner("🤖 Analyzing..."):
                # Fetch financial statements for comprehensive analysis
                financial_statements_data, fin_error = fetch_financial_statements(symbol)
                if fin_error:
                    st.warning(f"Could not fetch financial statements: {fin_error}")
                
                # Fetch latest news
                news_data, news_error = fetch_news(symbol)
                if news_error:
                    st.warning(f"Could not fetch news: {news_error}")
                
                # Get enhanced context (RAG or regular)
                if RAG_AVAILABLE:
                    context, is_rag_enhanced = get_rag_enhanced_context(
                        symbol,
                        st.session_state.stock_data,
                        st.session_state.stock_info,
                        news_data,
                        financial_statements_data,
                        user_question
                    )
                    
                    if is_rag_enhanced:
                        st.info("🔍 Using RAG-enhanced context for better analysis")
                else:
                    context = create_financial_context(
                        symbol, 
                        st.session_state.stock_data, 
                        st.session_state.stock_info, 
                        news_data,
                        financial_statements_data
                    )
                    is_rag_enhanced = False
                
                # Get AI response with potentially enhanced prompt
                if RAG_AVAILABLE and is_rag_enhanced:
                    # Use RAG-enhanced prompt
                    enhanced_prompt = rag_enhanced_prompt_modification(
                        get_base_prompt(), context, user_question
                    )
                    response = gemini_model.generate_content(enhanced_prompt)
                    ai_response = response.text
                else:
                    # Use original method
                    ai_response = get_gemini_response(gemini_model, user_question, context)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'response': ai_response,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                
                # Clear input
                st.rerun()
```

**AI Integration Architecture**:
1. **Data Aggregation**: Fetches all available data (prices, statements, news)
2. **Context Enhancement**: Uses RAG if available, falls back to basic context
3. **Prompt Engineering**: Different prompts for RAG vs basic mode
4. **Response Management**: Stores chat history with timestamps
5. **UI Updates**: Forces rerun to clear input and show new response

#### Chart Visualization (Lines 661-739)
```python
with tab2:
    # Create candlestick chart with volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} Price', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume bar chart
    colors = ['#ef5350' if row['Close'] < row['Open'] else '#26a69a' 
              for idx, row in data.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
```

**Professional Charting**:
- **Subplot Layout**: Price chart above, volume below (standard financial layout)
- **Color Coding**: Green for up days, red for down days
- **Interactive Features**: Plotly provides zoom, pan, hover details
- **Volume Correlation**: Volume bars match price movement colors

#### Data Table (Lines 740-789)
```python
with tab3:
    st.subheader("📋 Historical Data Table")
    
    # Display options
    col1, col2 = st.columns([1, 3])
    with col1:
        show_rows = st.selectbox("Show rows", [10, 25, 50, 100, "All"], index=1)
    
    # Prepare display data
    display_data = data.copy()
    display_data.index = display_data.index.strftime('%Y-%m-%d %H:%M:%S')
    
    # Format numeric columns
    for col in ['Open', 'High', 'Low', 'Close']:
        display_data[col] = display_data[col].round(2)
    
    display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{x:,}")
    
    # Apply row limit
    if show_rows != "All":
        display_data = display_data.tail(show_rows)
    
    # Display with custom styling
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )
```

**Data Presentation**:
- **User Control**: Configurable row limits
- **Formatting**: Rounded prices, comma-separated volume
- **Responsive**: Full-width container adaptation
- **Recent Focus**: Shows most recent data first

#### Statistics Calculation (Lines 791-854)
```python
with tab4:
    st.subheader("📊 Statistical Analysis")
    
    # Calculate returns
    returns = data['Close'].pct_change().dropna()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📈 Total Return", f"{((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}%")
        st.metric("📊 Volatility (Annualized)", f"{returns.std() * np.sqrt(252) * 100:.2f}%")
    
    with col2:
        st.metric("🎯 Sharpe Ratio", f"{(returns.mean() / returns.std()) * np.sqrt(252):.2f}")
        st.metric("📉 Max Drawdown", f"{((data['Close'] / data['Close'].cummax()) - 1).min() * 100:.2f}%")
    
    with col3:
        st.metric("📅 Trading Days", f"{len(data)}")
        st.metric("💰 Average Daily Volume", f"{data['Volume'].mean():,.0f}")
```

**Financial Analytics**:
- **Risk Metrics**: Volatility, Sharpe ratio, maximum drawdown
- **Return Analysis**: Total return over the period
- **Trading Statistics**: Volume patterns and trading frequency
- **Annualization**: Standard 252 trading days per year assumption

---

## MCP Server Implementation

### File: `src/mcp_financial_server/server.py`

The MCP (Model Context Protocol) server provides a standardized interface for financial data access.

#### Server Initialization (Lines 1-35)
```python
"""
MCP (Model Context Protocol) Server for Financial Data
Provides standardized access to stock market data and analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import yfinance as yf
import pandas as pd
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource, LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financial-server")

# Initialize the MCP server
server = Server("financial-server")
```

**MCP Architecture**:
- **Standardized Protocol**: Follows MCP specifications for tool-based AI interactions
- **Async Design**: All operations are async for better performance
- **Logging**: Comprehensive logging for debugging and monitoring

#### Tool Definitions (Lines 37-120)
```python
@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available financial analysis tools"""
    return [
        Tool(
            name="get_stock_history",
            description="Get historical stock price data with customizable time periods and intervals",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                        "description": "Data period to retrieve",
                        "default": "1y"
                    },
                    "interval": {
                        "type": "string",
                        "enum": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                        "description": "Data interval granularity",
                        "default": "1d"
                    }
                },
                "required": ["symbol"]
            }
        ),
        # ... more tool definitions
    ]
```

**Tool Design Pattern**:
- **JSON Schema Validation**: Input parameters are strictly validated
- **Enumerated Options**: Prevents invalid period/interval combinations
- **Default Values**: Sensible defaults reduce complexity for users
- **Documentation**: Rich descriptions for AI model understanding

#### Stock History Implementation (Lines 122-180)
```python
@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool execution requests"""
    
    if name == "get_stock_history":
        return await get_stock_history(
            arguments.get("symbol"),
            arguments.get("period", "1y"),
            arguments.get("interval", "1d")
        )
    # ... other tool handlers

async def get_stock_history(symbol: str, period: str = "1y", interval: str = "1d") -> List[TextContent]:
    """Fetch historical stock data using yfinance"""
    try:
        logger.info(f"Fetching stock history for {symbol} (period: {period}, interval: {interval})")
        
        # Create yfinance Ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch historical data
        hist_data = ticker.history(period=period, interval=interval)
        
        if hist_data.empty:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"No data found for symbol: {symbol}",
                    "symbol": symbol,
                    "period": period,
                    "interval": interval
                })
            )]
        
        # Process and format data
        result = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data_points": len(hist_data),
            "date_range": {
                "start": hist_data.index[0].isoformat(),
                "end": hist_data.index[-1].isoformat()
            },
            "latest_price": {
                "date": hist_data.index[-1].isoformat(),
                "open": float(hist_data['Open'].iloc[-1]),
                "high": float(hist_data['High'].iloc[-1]),
                "low": float(hist_data['Low'].iloc[-1]),
                "close": float(hist_data['Close'].iloc[-1]),
                "volume": int(hist_data['Volume'].iloc[-1])
            },
            "historical_data": []
        }
        
        # Add historical data points
        for date, row in hist_data.iterrows():
            result["historical_data"].append({
                "date": date.isoformat(),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"Error fetching stock history for {symbol}: {str(e)}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Failed to fetch data: {str(e)}",
                "symbol": symbol
            })
        )]
```

**Data Processing Pipeline**:
1. **Input Validation**: Symbol, period, interval validation
2. **External API Call**: yfinance integration with error handling
3. **Data Transformation**: Pandas DataFrame to JSON structure
4. **Type Conversion**: Ensures JSON serializable types (float, int)
5. **Error Response**: Consistent error format for client handling

#### Company Information Tool (Lines 182-240)
```python
async def get_stock_info(symbol: str) -> List[TextContent]:
    """Get comprehensive stock information and metrics"""
    try:
        logger.info(f"Fetching stock info for {symbol}")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"No information found for symbol: {symbol}",
                    "symbol": symbol
                })
            )]
        
        # Extract key metrics with safe access
        key_metrics = {}
        
        # Basic company info
        for field in ['longName', 'sector', 'industry', 'website', 'city', 'country']:
            if field in info:
                key_metrics[field] = info[field]
        
        # Financial metrics
        financial_fields = [
            'marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE', 
            'priceToBook', 'dividendYield', 'beta', 'sharesOutstanding'
        ]
        
        for field in financial_fields:
            if field in info and info[field] is not None:
                key_metrics[field] = info[field]
        
        # Business summary
        if 'longBusinessSummary' in info:
            key_metrics['businessSummary'] = info['longBusinessSummary']
        
        result = {
            "symbol": symbol,
            "company_info": key_metrics,
            "data_timestamp": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Failed to fetch company info: {str(e)}",
                "symbol": symbol
            })
        )]
```

**Data Reliability Patterns**:
- **Safe Dictionary Access**: Checks key existence before access
- **Null Handling**: Explicit None checks for financial metrics
- **Selective Extraction**: Only includes relevant fields to reduce noise
- **Timestamp Addition**: Tracks when data was retrieved

#### News Integration (Lines 300-360)
```python
async def get_news(symbol: str, count: int = 10) -> List[TextContent]:
    """Get latest financial news for a stock"""
    try:
        logger.info(f"Fetching news for {symbol}")
        
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "message": f"No recent news found for {symbol}",
                    "symbol": symbol,
                    "articles": []
                })
            )]
        
        # Process and limit news articles
        processed_news = []
        for article in news[:count]:
            news_item = {
                "title": article.get('title', 'No title'),
                "summary": article.get('summary', 'No summary available'),
                "publisher": article.get('publisher', 'Unknown'),
                "link": article.get('link', ''),
                "published_time": article.get('providerPublishTime', 0)
            }
            
            # Convert timestamp to readable format
            if news_item['published_time']:
                news_item['published_date'] = datetime.fromtimestamp(
                    news_item['published_time']
                ).isoformat()
            
            processed_news.append(news_item)
        
        result = {
            "symbol": symbol,
            "articles_count": len(processed_news),
            "articles": processed_news,
            "fetched_at": datetime.now().isoformat()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Failed to fetch news: {str(e)}",
                "symbol": symbol
            })
        )]
```

**News Processing Strategy**:
- **Count Limiting**: Prevents overwhelming responses
- **Data Cleaning**: Handles missing fields with defaults
- **Timestamp Conversion**: Unix timestamps to ISO format
- **Metadata Tracking**: Includes fetch timestamp and article count

---

## RAG System Deep Dive

### File: `src/rag_system.py`

The RAG (Retrieval-Augmented Generation) system enhances AI responses by finding and using the most relevant information from indexed financial data.

#### Core Architecture Overview
The RAG system consists of several interconnected components:

1. **SemanticChunker**: Intelligently segments text into coherent chunks
2. **HNSWVectorStore**: Efficient similarity search using HNSW algorithm
3. **QueryRewriter**: Expands queries using financial domain knowledge
4. **HybridRetriever**: Combines vector search with keyword matching
5. **Reranker**: Advanced relevance scoring using cross-encoder models
6. **AutocutGenerator**: Dynamic context selection based on relevance

#### Data Models (Lines 63-82)
```python
@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""
    source: str = ""
    timestamp: datetime = None
    semantic_score: float = 0.0

@dataclass
class QueryRewrite:
    """Represents a rewritten query with metadata"""
    original_query: str
    rewritten_queries: List[str]
    query_type: str  # "financial", "technical", "news", "company"
    intent: str
    entities: List[str]
```

**Design Principles**:
- **Immutable Data**: Dataclasses provide structured, type-safe data containers
- **Rich Metadata**: Each chunk carries context about its source and processing
- **Optional Fields**: Gradual enhancement of data as it flows through pipeline

#### Semantic Chunking Implementation (Lines 83-147)
```python
class SemanticChunker:
    """Advanced semantic chunking using sentence embeddings and coherence scoring"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.coherence_threshold = 0.7
        self.max_chunk_size = 512
        self.min_chunk_size = 50
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Create semantically coherent chunks from text"""
        # Split into sentences
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return [SemanticChunk(content=text, metadata=metadata)]
        
        # Get sentence embeddings
        sentence_embeddings = self.model.encode(sentences)
        
        # Calculate semantic similarity between adjacent sentences
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = [sentence_embeddings[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            chunk_embedding = np.mean(current_embedding, axis=0)
            similarity = cosine_similarity([chunk_embedding], [sentence_embeddings[i]])[0][0]
            
            # Check chunk size
            current_text = " ".join(current_chunk + [sentences[i]])
            
            if (similarity >= self.coherence_threshold and 
                len(current_text.split()) <= self.max_chunk_size):
                # Add to current chunk
                current_chunk.append(sentences[i])
                current_embedding.append(sentence_embeddings[i])
            else:
                # Create new chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunk_embedding = np.mean(current_embedding, axis=0)
                    chunks.append(SemanticChunk(
                        content=chunk_text,
                        metadata=metadata.copy(),
                        embedding=chunk_embedding,
                        semantic_score=similarity
                    ))
                
                # Start new chunk
                current_chunk = [sentences[i]]
                current_embedding = [sentence_embeddings[i]]
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunk_embedding = np.mean(current_embedding, axis=0)
                chunks.append(SemanticChunk(
                    content=chunk_text,
                    metadata=metadata.copy(),
                    embedding=chunk_embedding
                ))
        
        return chunks
```

**Semantic Chunking Algorithm**:
1. **Sentence Tokenization**: Uses NLTK to split text into sentences
2. **Embedding Generation**: Creates vector representations for each sentence
3. **Similarity Calculation**: Measures semantic coherence between adjacent content
4. **Threshold-Based Grouping**: Groups sentences above coherence threshold
5. **Size Constraints**: Ensures chunks are neither too small nor too large
6. **Progressive Building**: Incrementally builds chunks while maintaining coherence

#### HNSW Vector Store (Lines 149-233)
```python
class HNSWVectorStore:
    """HNSW-based vector store for efficient similarity search"""
    
    def __init__(self, dimension: int = 384, max_elements: int = 100000):
        self.dimension = dimension
        self.max_elements = max_elements
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=max_elements, ef_construction=400, M=32)
        self.index.set_ef(200)  # Set ef for search
        self.chunks: Dict[int, SemanticChunk] = {}
        self.next_id = 0
        
    def add_chunks(self, chunks: List[SemanticChunk]):
        """Add chunks to the vector store"""
        embeddings = []
        ids = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                chunk.chunk_id = str(self.next_id)
                self.chunks[self.next_id] = chunk
                embeddings.append(chunk.embedding)
                ids.append(self.next_id)
                self.next_id += 1
        
        if embeddings:
            self.index.add_items(np.array(embeddings), ids)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[SemanticChunk, float]]:
        """Search for similar chunks"""
        if self.next_id == 0:
            return []
        
        try:
            # Ensure we don't search for more items than we have
            actual_k = min(k, self.next_id)
            self.index.set_ef(max(actual_k * 2, 50))  # Dynamic ef for better recall
            labels, distances = self.index.knn_query(query_embedding.reshape(1, -1), k=actual_k)
            
            results = []
            for label, distance in zip(labels[0], distances[0]):
                if label in self.chunks:
                    similarity = 1 - distance  # Convert distance to similarity
                    results.append((self.chunks[label], similarity))
            
            return results
        except Exception as e:
            logging.warning(f"Error in HNSW search: {e}")
            return []
```

**HNSW Optimization**:
- **High Performance Parameters**: ef_construction=400, M=32 for quality/speed balance
- **Dynamic Search Tuning**: Adjusts ef parameter based on requested results
- **Robust Error Handling**: Graceful degradation when search fails
- **Memory Efficiency**: Stores only embeddings in index, full data separately

#### Query Rewriting System (Lines 234-337)
```python
class QueryRewriter:
    """Advanced query rewriting for financial domain"""
    
    def __init__(self):
        self.financial_synonyms = {
            "profit": ["earnings", "income", "net income", "profit margin"],
            "revenue": ["sales", "turnover", "top line", "gross revenue"],
            "debt": ["liabilities", "borrowings", "leverage", "debt ratio"],
            "growth": ["expansion", "increase", "development", "appreciation"],
            "performance": ["returns", "results", "metrics", "KPIs"],
            "risk": ["volatility", "uncertainty", "exposure", "downside"],
            "valuation": ["price", "worth", "market cap", "enterprise value"]
        }
        
        self.query_templates = {
            "trend": [
                "What is the {metric} trend over time?",
                "How has {metric} changed recently?",
                "Show me {metric} historical performance"
            ],
            "comparison": [
                "How does {metric} compare to industry average?",
                "Compare {metric} with competitors",
                "{metric} vs peer analysis"
            ],
            "analysis": [
                "Analyze {metric} performance",
                "What factors affect {metric}?",
                "Explain {metric} implications"
            ]
        }
    
    def rewrite_query(self, query: str) -> QueryRewrite:
        """Rewrite query with financial domain knowledge"""
        query_lower = query.lower()
        rewritten_queries = [query]  # Include original
        
        # Identify query type and intent
        query_type = self._classify_query_type(query_lower)
        intent = self._extract_intent(query_lower)
        entities = self._extract_entities(query_lower)
        
        # Generate synonym-based rewrites
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    rewritten = query_lower.replace(term, synonym)
                    rewritten_queries.append(rewritten)
        
        # Generate template-based rewrites
        if intent in self.query_templates:
            for entity in entities:
                for template in self.query_templates[intent]:
                    rewritten = template.format(metric=entity)
                    rewritten_queries.append(rewritten)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_rewrites = []
        for q in rewritten_queries:
            if q not in seen:
                seen.add(q)
                unique_rewrites.append(q)
        
        return QueryRewrite(
            original_query=query,
            rewritten_queries=unique_rewrites,
            query_type=query_type,
            intent=intent,
            entities=entities
        )
```

**Query Enhancement Strategy**:
- **Domain-Specific Synonyms**: Financial terminology expansion
- **Intent Classification**: Identifies user goals (trend, comparison, analysis)
- **Template Generation**: Creates structured queries for better matching
- **Entity Extraction**: Identifies financial metrics and concepts

#### Hybrid Retrieval System (Lines 339-423)
```python
class HybridRetriever:
    """Hybrid search combining vector similarity and BM25"""
    
    def __init__(self, vector_store: HNSWVectorStore, embedding_model: SentenceTransformer):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25 = None
        self.chunks_text = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.tfidf_matrix = None
        
    def index_for_keyword_search(self, chunks: List[SemanticChunk]):
        """Index chunks for keyword-based search"""
        self.chunks_text = [chunk.content for chunk in chunks]
        
        # BM25 indexing
        tokenized_chunks = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # TF-IDF indexing
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks_text)
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> List[Tuple[SemanticChunk, float]]:
        """Perform hybrid search combining semantic and keyword search"""
        # Vector search
        query_embedding = self.embedding_model.encode([query])
        vector_results = self.vector_store.search(query_embedding[0], k=k*2)
        
        # Keyword search (BM25)
        bm25_scores = self.bm25.get_scores(query.lower().split()) if self.bm25 else []
        
        # TF-IDF search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Combine scores
        combined_results = {}
        
        # Add vector results
        for chunk, score in vector_results:
            chunk_id = int(chunk.chunk_id) if chunk.chunk_id else 0
            combined_results[chunk_id] = {
                'chunk': chunk,
                'vector_score': score,
                'bm25_score': 0,
                'tfidf_score': 0
            }
        
        # Add BM25 scores
        for i, score in enumerate(bm25_scores):
            if i in combined_results:
                combined_results[i]['bm25_score'] = score
            elif score > 0:  # Only add if there's a meaningful BM25 score
                # Find corresponding chunk
                chunk = self.vector_store.chunks.get(i)
                if chunk:
                    combined_results[i] = {
                        'chunk': chunk,
                        'vector_score': 0,
                        'bm25_score': score,
                        'tfidf_score': 0
                    }
        
        # Add TF-IDF scores
        for i, score in enumerate(tfidf_similarities):
            if i in combined_results:
                combined_results[i]['tfidf_score'] = score
        
        # Calculate final hybrid scores
        final_results = []
        for result in combined_results.values():
            # Normalize scores
            vector_score = result['vector_score']
            bm25_score = result['bm25_score'] / max(bm25_scores + [1]) if bm25_scores else 0
            tfidf_score = result['tfidf_score']
            
            # Weighted combination
            hybrid_score = (alpha * vector_score + 
                           (1 - alpha) * 0.5 * (bm25_score + tfidf_score))
            
            final_results.append((result['chunk'], hybrid_score))
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]
```

**Hybrid Search Architecture**:
1. **Vector Search**: Semantic similarity using embeddings
2. **BM25 Search**: Statistical keyword matching (TF-IDF variant)
3. **TF-IDF Search**: Term frequency analysis
4. **Score Fusion**: Weighted combination of all three approaches
5. **Alpha Parameter**: Controls semantic vs keyword balance

#### Cross-Encoder Reranking (Lines 425-457)
```python
class Reranker:
    """Advanced reranking using cross-encoder models"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(model_name)
            self.available = True
        except ImportError:
            self.available = False
            logging.warning("CrossEncoder not available, using fallback reranking")
    
    def rerank(self, query: str, chunks_with_scores: List[Tuple[SemanticChunk, float]], 
              top_k: int = 5) -> List[Tuple[SemanticChunk, float]]:
        """Rerank chunks using cross-encoder"""
        if not self.available or len(chunks_with_scores) <= 1:
            return chunks_with_scores[:top_k]
        
        # Prepare query-chunk pairs
        pairs = [(query, chunk.content) for chunk, _ in chunks_with_scores]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Combine with original scores
        reranked_results = []
        for i, (chunk, original_score) in enumerate(chunks_with_scores):
            combined_score = 0.6 * cross_scores[i] + 0.4 * original_score
            reranked_results.append((chunk, combined_score))
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]
```

**Reranking Strategy**:
- **Cross-Encoder Model**: Deep interaction between query and candidate text
- **Score Combination**: Balances cross-encoder and original retrieval scores
- **Fallback Handling**: Graceful degradation if cross-encoder unavailable

### File: `src/rag_integration.py`

This file provides the integration layer between the RAG system and the Streamlit application.

#### Integration Architecture (Lines 21-54)
```python
class RAGIntegration:
    """Integration layer between RAG system and Streamlit app"""
    
    def __init__(self, cache_dir: str = "rag_cache"):
        self.cache_dir = cache_dir
        self.rag_system = None
        self.is_initialized = False
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        if RAG_AVAILABLE:
            self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize the RAG system"""
        try:
            self.rag_system = AdvancedRAGSystem()
            
            # Try to load existing index (disabled for now due to pickle issues)
            # index_path = os.path.join(self.cache_dir, "financial_rag")
            # if os.path.exists(f"{index_path}_index.bin"):
            #     try:
            #         self.rag_system.load_index(index_path)
            #         st.success("🔍 RAG system loaded from cache")
            #     except Exception as e:
            #         st.warning(f"Could not load RAG cache: {e}")
            #         self.rag_system = AdvancedRAGSystem()  # Reinitialize
            
            self.is_initialized = True
            
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            self.is_initialized = False
```

**Integration Design**:
- **Lazy Initialization**: RAG system only initializes if dependencies available
- **Cache Management**: Directory structure for persistent storage
- **Error Isolation**: RAG failures don't crash main application

#### Data Indexing Pipeline (Lines 60-97)
```python
def index_stock_data(self, symbol: str, stock_data: pd.DataFrame, 
                    stock_info: Dict[str, Any], news_data: List[Dict], 
                    financial_statements: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, int]:
    """Index all available data for a stock symbol"""
    if not self.is_available():
        return {"error": "RAG system not available"}
    
    indexed_counts = {}
    
    try:
        # Index stock price data
        if stock_data is not None and not stock_data.empty:
            count = self.rag_system.index_financial_data(stock_data, symbol, 'stock_data')
            indexed_counts['stock_data'] = count
        
        # Index company information
        if stock_info:
            count = self.rag_system.index_financial_data(stock_info, symbol, 'company_info')
            indexed_counts['company_info'] = count
        
        # Index news data
        if news_data:
            count = self.rag_system.index_financial_data(news_data, symbol, 'news')
            indexed_counts['news'] = count
        
        # Index financial statements
        if financial_statements:
            count = self.rag_system.index_financial_data(financial_statements, symbol, 'financial_statements')
            indexed_counts['financial_statements'] = count
        
        # Save the updated index (disabled for now due to pickle issues)
        # self._save_index()
        
        return indexed_counts
        
    except Exception as e:
        st.error(f"Error indexing data: {e}")
        return {"error": str(e)}
```

**Indexing Strategy**:
- **Multi-Modal Data**: Handles different data types (prices, news, statements)
- **Incremental Indexing**: Can index partial data sets
- **Progress Tracking**: Returns counts of indexed chunks
- **Error Resilience**: Continues indexing even if some data types fail

#### Query Processing (Lines 99-134)
```python
def query_rag(self, question: str, k: int = 5) -> Dict[str, Any]:
    """Query the RAG system"""
    if not self.is_available():
        return {
            "error": "RAG system not available",
            "context": "",
            "rag_enhanced": False
        }
    
    try:
        # Check if we have any indexed data
        total_chunks = len(self.rag_system.vector_store.chunks)
        if total_chunks == 0:
            return {
                "error": "No data indexed yet",
                "context": "",
                "rag_enhanced": False
            }
        
        # Limit k to available chunks
        actual_k = min(k, total_chunks)
        result = self.rag_system.query(question, k=actual_k, use_autocut=True)
        
        # Add metadata for the UI
        result['rag_enhanced'] = True
        result['query_timestamp'] = datetime.now()
        
        return result
        
    except Exception as e:
        logging.warning(f"Error querying RAG system: {e}")
        return {
            "error": str(e),
            "context": "",
            "rag_enhanced": False
        }
```

**Query Processing Pipeline**:
1. **Availability Check**: Ensures RAG system is ready
2. **Data Validation**: Verifies indexed data exists
3. **Parameter Adjustment**: Prevents requests exceeding available data
4. **Metadata Enhancement**: Adds UI-relevant information
5. **Error Handling**: Returns consistent error format

---

## Integration Patterns

### Error Handling Strategy

Throughout the codebase, a consistent error handling pattern is used:

1. **Graceful Degradation**: Features fail gracefully without crashing the app
2. **User Feedback**: Clear error messages shown to users
3. **Logging**: Technical details logged for debugging
4. **Fallback Behavior**: Alternative functionality when primary fails

Example from `streamlit_app.py`:
```python
try:
    data, info, error = fetch_stock_data(symbol, period, interval)
    if error:
        st.error(f"Error fetching data: {error}")
    elif data is None or data.empty:
        st.error(f"No data found for symbol: {symbol}")
    else:
        # Success path
        st.success(f"Data fetched successfully for {symbol}")
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    logging.error(f"Stock data fetch failed: {e}")
```

### Caching Strategy

The application uses multiple caching layers:

1. **Streamlit Caching**: `@st.cache_data` and `@st.cache_resource`
2. **Session State**: Persistent data across user interactions
3. **RAG Index Caching**: Vector store persistence (when enabled)

Cache TTL (Time To Live) values:
- Stock data: 5 minutes (frequent updates needed)
- News data: 15 minutes (moderate update frequency)
- Financial statements: 1 hour (infrequent updates)
- AI model: Resource cache (loaded once per session)

### State Management

The application maintains state through:

1. **Session State Variables**:
   ```python
   st.session_state = {
       'stock_data': pd.DataFrame,
       'stock_info': Dict[str, Any],
       'current_symbol': str,
       'chat_history': List[Dict],
       'financial_statements': Dict[str, pd.DataFrame]
   }
   ```

2. **RAG System State**:
   - Vector store with indexed chunks
   - Query rewriter with financial domain knowledge
   - Hybrid retriever with multiple search indices

### Data Flow Architecture

```
User Input → Streamlit UI → Data Fetching → Processing → Storage
                ↓
RAG Indexing ← Session State ← Validation ← Transformation
                ↓
AI Query → RAG Retrieval → Context Enhancement → LLM → Response
```

---

## Development Workflows

### Adding New Features

1. **Data Sources**: Add new fetching functions with caching
2. **UI Components**: Create new tabs or sections in Streamlit
3. **RAG Integration**: Add data type to indexing pipeline
4. **AI Enhancement**: Update context creation functions
5. **Testing**: Add tests for new functionality

### Code Organization Principles

1. **Separation of Concerns**: UI, data, AI, and integration layers
2. **Dependency Injection**: Components receive dependencies rather than creating them
3. **Interface Consistency**: Similar patterns across different data types
4. **Error Boundaries**: Isolated error handling for each component

### Performance Optimization

1. **Lazy Loading**: Components only load when needed
2. **Batch Processing**: Group related operations
3. **Memory Management**: Clear unused data from session state
4. **Async Operations**: Use async/await for I/O operations

---

## Testing and Quality

### Test Structure

The codebase includes several testing approaches:

1. **Unit Tests**: `test_server.py` for MCP server functionality
2. **Integration Tests**: `test_chat.py` for AI functionality
3. **Manual Testing**: Through Streamlit interface

### Quality Assurance

Development workflow includes:

```bash
make check  # Runs all quality checks
make format # Code formatting with black
make lint   # Linting with flake8
make type-check # Type checking with mypy
```

### Code Quality Patterns

1. **Type Hints**: All functions include type annotations
2. **Docstrings**: Comprehensive documentation for all classes/functions
3. **Error Handling**: Explicit exception handling with logging
4. **Code Formatting**: Consistent style enforced by black
5. **Import Organization**: Organized imports with clear dependencies

---

This developer guide provides a comprehensive understanding of the codebase architecture, implementation patterns, and development workflows. Each component is designed with modularity, reliability, and maintainability in mind, following modern Python development best practices.