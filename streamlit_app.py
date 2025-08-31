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
from src.ai_providers import AIModelManager

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

# Neo4j Graph Database Integration
GRAPH_AVAILABLE = False
try:
    from src.neo4j_client import get_graph_db
    from src.rag_graph_integration import get_graph_enhanced_rag
    GRAPH_AVAILABLE = True
except ImportError as e:
    import logging
    logging.info(f"Graph database not available: {e}")
    
    def get_graph_db():
        return None
    
    def get_graph_enhanced_rag():
        return None

# Load environment variables from .env file
load_dotenv()

# Currency symbol mapping
def get_currency_symbol(currency_code):
    """Get currency symbol for given currency code"""
    currency_symbols = {
        'USD': '$',
        'INR': '₹',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CAD': 'C$',
        'AUD': 'A$',
        'CHF': 'CHF ',
        'CNY': '¥',
        'HKD': 'HK$',
        'SGD': 'S$',
        'KRW': '₩',
        'BRL': 'R$',
        'MXN': 'MX$',
        'RUB': '₽',
        'SEK': 'kr',
        'NOK': 'kr',
        'DKK': 'kr',
        'PLN': 'zł',
        'TRY': '₺',
        'ZAR': 'R',
        'NZD': 'NZ$'
    }
    return currency_symbols.get(currency_code, currency_code + ' ')

# Initialize Gemini AI
@st.cache_resource
def initialize_ai_model(model_name: str = None):
    """Initialize selected AI model"""
    try:
        if model_name:
            provider = AIModelManager.get_provider(model_name)
            if provider:
                return provider
        
        # Get default model if none specified
        default_model = AIModelManager.get_default_model()
        if default_model:
            return AIModelManager.get_provider(default_model)
        
        return None
    except Exception as e:
        st.error(f"Error initializing AI model: {str(e)}")
        return None

def create_financial_context(symbol, stock_data, stock_info, news_data, financial_statements, max_context_size=50000):
    """Create comprehensive context from all available data sources with size limits"""
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
        
        # Include price data summary and recent data only
        context_parts.append("Recent price history (last 20 data points):")
        recent_data = stock_data.tail(20)
        for idx, row in recent_data.iterrows():
            context_parts.append(f"{idx.strftime('%Y-%m-%d %H:%M:%S')}: Open={row['Open']:.2f}, High={row['High']:.2f}, Low={row['Low']:.2f}, Close={row['Close']:.2f}, Volume={row['Volume']:,}")
        
        # Add summary statistics
        context_parts.append("\nPrice Statistics:")
        context_parts.append(f"Current Price: ${stock_data['Close'].iloc[-1]:.2f}")
        context_parts.append(f"Period High: ${stock_data['High'].max():.2f}")
        context_parts.append(f"Period Low: ${stock_data['Low'].min():.2f}")
        context_parts.append(f"Average Volume: {stock_data['Volume'].mean():,.0f}")
        context_parts.append(f"Price Change: {((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100:.2f}%")
        context_parts.append("")
    
    # Complete financial statements
    if financial_statements:
        context_parts.append("=== COMPLETE FINANCIAL STATEMENTS ===")
        
        # Recent quarterly financials (Income Statement) - last 4 quarters only
        quarterly_financials = financial_statements.get('quarterly_financials')
        if quarterly_financials is not None and not quarterly_financials.empty:
            context_parts.append("\n--- RECENT QUARTERLY INCOME STATEMENTS (Last 4 Quarters) ---")
            # Include only last 4 quarters
            recent_quarters = quarterly_financials.columns[:4] if len(quarterly_financials.columns) >= 4 else quarterly_financials.columns
            for col in recent_quarters:
                context_parts.append(f"\nPeriod: {col.strftime('%Y-%m-%d')}")
                # Include only key metrics
                key_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EBITDA', 'Operating Expense']
                for metric in quarterly_financials.index:
                    if any(key in metric for key in key_metrics):
                        value = quarterly_financials.loc[metric, col]
                        if pd.notna(value):
                            context_parts.append(f"{metric}: {value:,.0f}")
        
        # Recent annual financials (Income Statement) - last 2 years only
        annual_financials = financial_statements.get('financials')
        if annual_financials is not None and not annual_financials.empty:
            context_parts.append("\n--- RECENT ANNUAL INCOME STATEMENTS (Last 2 Years) ---")
            recent_years = annual_financials.columns[:2] if len(annual_financials.columns) >= 2 else annual_financials.columns
            for col in recent_years:
                context_parts.append(f"\nYear: {col.strftime('%Y-%m-%d')}")
                # Include only key metrics
                key_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EBITDA', 'Operating Expense']
                for metric in annual_financials.index:
                    if any(key in metric for key in key_metrics):
                        value = annual_financials.loc[metric, col]
                        if pd.notna(value):
                            context_parts.append(f"{metric}: {value:,.0f}")
        
        # Recent quarterly balance sheet - last 2 quarters only
        quarterly_balance_sheet = financial_statements.get('quarterly_balance_sheet')
        if quarterly_balance_sheet is not None and not quarterly_balance_sheet.empty:
            context_parts.append("\n--- RECENT QUARTERLY BALANCE SHEETS (Last 2 Quarters) ---")
            recent_quarters = quarterly_balance_sheet.columns[:2] if len(quarterly_balance_sheet.columns) >= 2 else quarterly_balance_sheet.columns
            for col in recent_quarters:
                context_parts.append(f"\nPeriod: {col.strftime('%Y-%m-%d')}")
                # Include only key items
                key_items = ['Total Assets', 'Total Liabilities', 'Total Equity', 'Cash', 'Total Debt', 'Current Assets', 'Current Liabilities']
                for item in quarterly_balance_sheet.index:
                    if any(key in item for key in key_items):
                        value = quarterly_balance_sheet.loc[item, col]
                        if pd.notna(value):
                            context_parts.append(f"{item}: {value:,.0f}")
        
        # Skip detailed annual balance sheet to save space (quarterly is sufficient)
        
        # Cash flow
        quarterly_cashflow = financial_statements.get('quarterly_cashflow')
        if quarterly_cashflow is not None and not quarterly_cashflow.empty:
            context_parts.append("\n--- QUARTERLY CASH FLOW STATEMENTS ---")
            context_parts.append("All Quarterly Cash Flows:")
            for col in quarterly_cashflow.columns:
                context_parts.append(f"\nPeriod: {col.strftime('%Y-%m-%d')}")
                for item in quarterly_cashflow.index:
                    value = quarterly_cashflow.loc[item, col]
                    if pd.notna(value):
                        context_parts.append(f"{item}: {value:,.0f}")
        
        # Annual cash flow
        annual_cashflow = financial_statements.get('cashflow')
        if annual_cashflow is not None and not annual_cashflow.empty:
            context_parts.append("\n--- ANNUAL CASH FLOW STATEMENTS ---")
            context_parts.append("All Annual Cash Flows:")
            for col in annual_cashflow.columns:
                context_parts.append(f"\nYear: {col.strftime('%Y-%m-%d')}")
                for item in annual_cashflow.index:
                    value = annual_cashflow.loc[item, col]
                    if pd.notna(value):
                        context_parts.append(f"{item}: {value:,.0f}")
        
        context_parts.append("")
    
    # All news articles
    if news_data:
        context_parts.append("=== ALL NEWS ARTICLES ===")
        context_parts.append(f"Total articles: {len(news_data)}")
        for i, article in enumerate(news_data, 1):
            context_parts.append(f"\n{i}. Title: {article.get('title', 'No title')}")
            if article.get('published'):
                context_parts.append(f"   Published: {article['published']}")
            if article.get('publisher'):
                context_parts.append(f"   Publisher: {article['publisher']}")
            if article.get('summary'):
                context_parts.append(f"   Summary: {article['summary']}")
            if article.get('content'):
                context_parts.append(f"   Content: {article['content']}")
            if article.get('link'):
                context_parts.append(f"   Link: {article['link']}")
        context_parts.append("")
    
    # Join and check size
    full_context = "\n".join(context_parts)
    
    # Truncate if too large
    if len(full_context) > max_context_size:
        # Try to keep the most important parts
        truncated_context = full_context[:max_context_size]
        # Find a good breaking point
        last_section = truncated_context.rfind("\n===")
        if last_section > max_context_size * 0.7:
            truncated_context = truncated_context[:last_section]
        truncated_context += "\n\n[Note: Additional context truncated due to size limits]"
        return truncated_context
    
    return full_context

def get_base_prompt():
    """Get the base prompt template for Gemini"""
    return """You are an expert financial analyst AI assistant with deep expertise in technical analysis, fundamental analysis, and industry research. You have access to comprehensive financial data including:
- Complete stock price history with OHLCV data
- Comprehensive company information and metrics
- Full financial statements (Income Statement, Balance Sheet, Cash Flow)
- Recent news articles and market sentiment
- Market and industry context

RESPONSE STRUCTURE:
Your response must follow this exact structure:

## 💬 DIRECT ANSWER
First, provide a direct, concise answer to the user's specific question. Address exactly what they asked about using the available data.

## COMPREHENSIVE ANALYSIS
Then, provide detailed analysis in these four sections:

## 📈 TECHNICAL ANALYSIS
You have complete OHLCV (Open, High, Low, Close, Volume) data with timestamps. Use this raw data to CALCULATE and analyze:

- **Moving Averages**: From the closing prices provided, calculate 20-day, 50-day, and 200-day simple moving averages. Compare current price to these MAs and identify golden cross/death cross patterns.

- **RSI (14-period)**: Use the closing prices to calculate RSI = 100 - (100 / (1 + (Average Gain / Average Loss))). Identify overbought (>70) or oversold (<30) conditions.

- **MACD**: Calculate MACD using 12-period EMA, 26-period EMA, and 9-period signal line from closing prices. Look for bullish/bearish crossovers.

- **Support & Resistance**: From the price data, identify key levels where price has repeatedly bounced (support) or been rejected (resistance).

- **Fibonacci Retracements**: Identify the most recent significant high and low from the price data, then calculate retracement levels at 23.6%, 38.2%, 50%, 61.8%, and 78.6%.

- **Volume Analysis**: Analyze volume spikes, volume trends, and volume-price relationships using the provided volume data.

- **Trend Analysis**: Based on price action and moving averages, determine if the trend is bullish, bearish, or sideways for short-term (1-4 weeks), medium-term (1-3 months), and long-term (3+ months).

- **Price Patterns**: Look for chart patterns in the price data such as triangles, flags, head & shoulders, double tops/bottoms.

IMPORTANT: You MUST calculate these indicators from the raw OHLCV data provided. Do not say data is missing - use the closing prices, highs, lows, and volume data to compute all technical indicators.

## 📊 FUNDAMENTAL ANALYSIS
Conduct thorough fundamental analysis using financial statements and company data:
- **Financial Health**: Analyze revenue growth, profit margins, debt-to-equity ratio, current ratio, ROE, ROA.
- **Valuation Metrics**: Calculate and assess P/E ratio, P/B ratio, PEG ratio, EV/EBITDA, price-to-sales.
- **Growth Analysis**: Examine revenue growth, earnings growth, and growth sustainability.
- **Profitability**: Analyze gross margin, operating margin, net margin trends over time.
- **Cash Flow**: Evaluate operating cash flow, free cash flow, and cash flow stability.
- **Balance Sheet Strength**: Assess assets, liabilities, working capital, and debt levels.
- **Competitive Position**: Analyze market cap, market share, and competitive advantages.
- **Management Efficiency**: Calculate asset turnover, inventory turnover, and other efficiency metrics.
- **Dividend Analysis**: If applicable, analyze dividend yield, payout ratio, and dividend sustainability.
- **Fundamental Outlook**: Provide intrinsic value estimate and fundamental recommendation.

## 🏭 INDUSTRY & MARKET ANALYSIS
Provide broader market and industry perspective:
- **Industry Position**: Analyze the company's position within its industry and sector.
- **Industry Trends**: Identify key industry trends, growth drivers, and challenges.
- **Market Conditions**: Assess overall market sentiment and economic factors affecting the stock.
- **Sector Performance**: Compare performance relative to sector peers and benchmarks.
- **Regulatory Environment**: Consider any regulatory changes or risks affecting the industry.
- **Economic Indicators**: Factor in relevant economic indicators (interest rates, inflation, GDP growth).
- **News Impact**: Analyze how recent news affects the company and industry outlook.
- **Risk Factors**: Identify industry-specific and company-specific risks.
- **Market Outlook**: Provide perspective on how broader market trends may impact the stock.

## 🎯 INTEGRATED RECOMMENDATION
Synthesize all three analyses to provide:
- **Overall Rating**: Strong Buy / Buy / Hold / Sell / Strong Sell
- **Price Target**: Based on technical and fundamental analysis
- **Risk Assessment**: High/Medium/Low risk with key risk factors
- **Investment Horizon**: Short-term (1-3 months), Medium-term (3-12 months), Long-term (1+ years)
- **Key Catalysts**: What could drive the stock higher or lower

FINANCIAL DATA CONTEXT:
{context}

USER QUESTION: {user_question}

IMPORTANT: 
1. ALWAYS start with a "💬 DIRECT ANSWER" section that directly addresses the user's question
2. Then provide "COMPREHENSIVE ANALYSIS" with the four structured sections: Technical Analysis, Fundamental Analysis, Industry & Market Analysis, and Integrated Recommendation
3. Use specific calculations and data points from the provided context
4. Calculate all technical indicators from the raw OHLCV data provided - do not claim data is missing"""

def get_ai_response(provider, user_question, context):
    """Get response from AI provider"""
    try:
        prompt = get_base_prompt().format(context=context, user_question=user_question)
        response_text = provider.generate_response(prompt)
        return response_text
    except Exception as e:
        return f"Error generating response: {str(e)}"

st.set_page_config(
    page_title="Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stSidebar {
        background-color: #262730;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #262730;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: transparent;
        border-radius: 4px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #ff4444;
        color: white;
    }
    .stButton > button {
        background-color: #ff4444;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #ff6666;
    }
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #3d3d3d;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stDataFrame {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'stock_info' not in st.session_state:
    st.session_state.stock_info = None
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = "AAPL"

# Sidebar configuration
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
    
    st.markdown("### Data Interval")
    interval_options = {
        "1 Minute": "1m",
        "5 Minutes": "5m",
        "15 Minutes": "15m",
        "30 Minutes": "30m",
        "1 Hour": "1h",
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo"
    }
    
    # Filter intervals based on period
    if period_options[selected_period] in ["1d", "5d"]:
        available_intervals = ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"]
    else:
        available_intervals = ["1 Day", "1 Week", "1 Month"]
    
    selected_interval = st.selectbox(
        "Select Interval",
        options=available_intervals,
        index=0
    )
    
    # Fetch Data button
    fetch_button = st.button("🔄 Fetch Data", type="primary", use_container_width=True)

# Function to fetch stock data
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
                    
# Display RAG status in sidebar
if RAG_AVAILABLE:
    display_rag_status_sidebar()
                    
# Display title
st.title(f"{symbol} - Financial Analyst")

# Display data if available
if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
    data = st.session_state.stock_data
    info = st.session_state.stock_info or {}
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["🤖 AI Chat", "📊 Chart", "📋 Data Table", "📈 Statistics", "📰 News", "📊 Financial Statements", "🔗 Relationships", "💾 Export"])
                    
    with tab1:
        st.subheader("🤖 AI Financial Chat")
        
        # Get available AI models
        available_models = AIModelManager.get_available_models()
        active_models = [model for model, is_available in available_models.items() if is_available]
        
        if not active_models:
            st.error("❌ No AI models are configured. Please set up API keys for Google AI or OpenAI.")
            
            with st.expander("🔧 How to set up AI Chat"):
                st.markdown("""
                **Option 1: Google Gemini API**
                1. Get a Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. Edit the `.env` file in the project root
                3. Set: `GOOGLE_AI_API_KEY=your-api-key-here`
                4. Restart the application
                
                **Option 2: OpenAI API**
                1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
                2. Edit the `.env` file in the project root
                3. Set: `OPENAI_API_KEY=your-api-key-here`
                4. Restart the application
                
                **Option 3: Both APIs**
                Configure both API keys to have access to all models
                
                **Test your setup:**
                ```bash
                # Edit .env file with your API keys
                make run-streamlit
                ```
                """)
        else:
            # Model selection
            col1, col2 = st.columns([2, 3])
            with col1:
                selected_model = st.selectbox(
                    "Select AI Model:",
                    options=active_models,
                    index=0,
                    help="Choose which AI model to use for analysis"
                )
            with col2:
                st.success(f"✅ {len(active_models)} model(s) available")
            
            # Initialize selected model
            ai_provider = initialize_ai_model(selected_model)
            
            if not ai_provider:
                st.error(f"Failed to initialize {selected_model}")
            else:
                # Initialize chat history in session state
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Chat interface
                st.markdown("### 💬 Ask questions about your stock data")
                st.markdown("*Ask about price trends, financial metrics, news analysis, or any insights from the data.*")
            
                # Sample questions
                with st.expander("💡 Sample Questions"):
                    st.markdown("""
                    - What's the trend in gross profit margins over the last few quarters?
                    - How has the stock price performed recently?
                    - What are the key insights from recent news?
                    - Is the company's revenue growing?
                    - What are the main financial strengths and weaknesses?
                    - How does the current valuation look?
                    - What risks should I be aware of?
                    """)
            
                # Chat input form (enables Enter key submission)
                with st.form("chat_form", clear_on_submit=True):
                    user_question = st.text_input(
                        "Ask your question:",
                        placeholder="e.g., What's the trend in revenue growth?",
                        key="chat_input"
                    )
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        send_button = st.form_submit_button("🚀 Send", use_container_width=True)
                    with col2:
                        pass  # Empty column for spacing
            
                # Clear Chat button (outside form)
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
                
                # Process user question
                if send_button and user_question.strip():
                    with st.spinner("🤔 Analyzing your data..."):
                        # Gather all available data for context
                        financial_statements_data = None
                        news_data = st.session_state.get('news_data', [])
                        
                        # Fetch financial statements if needed
                        try:
                            stock = yf.Ticker(symbol)
                            financial_statements_data = {
                                'quarterly_financials': stock.quarterly_financials,
                                'yearly_financials': stock.financials,
                                'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                                'yearly_balance_sheet': stock.balance_sheet,
                                'quarterly_cashflow': stock.quarterly_cashflow,
                                'yearly_cashflow': stock.cashflow
                            }
                        except Exception as e:
                            st.warning(f"Could not fetch complete financial statements: {str(e)}")
                        
                        # Create comprehensive context (RAG-enhanced if available)
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
                            ai_response = ai_provider.generate_response(enhanced_prompt)
                        else:
                            # Use original method
                            ai_response = get_ai_response(ai_provider, user_question, context)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'response': ai_response,
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        })
                        
                        # Clear input
                        st.rerun()
            
                # Display chat history
                if st.session_state.chat_history:
                    st.markdown("### 💭 Chat History")
                    
                    for i, chat in enumerate(reversed(st.session_state.chat_history)):
                        with st.container():
                            # User question
                            st.markdown(f"**🙋 You ({chat['timestamp']}):**")
                            st.markdown(f"*{chat['question']}*")
                            
                            # AI response
                            st.markdown("**🤖 AI Analyst:**")
                            st.markdown(chat['response'])
                            
                            if i < len(st.session_state.chat_history) - 1:
                                st.markdown("---")
                else:
                    st.info("💡 Start a conversation by asking a question about your stock data!")
    
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
        
        # Update layout
        fig.update_layout(
            height=700,
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        latest_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else latest_close
        change = latest_close - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        # Get currency from stock info
        currency = st.session_state.stock_info.get('currency', 'USD') if st.session_state.stock_info else 'USD'
        currency_symbol = get_currency_symbol(currency)
        
        with col1:
            st.metric("Latest Close", f"{currency_symbol}{latest_close:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
        with col2:
            st.metric("High", f"{currency_symbol}{data['High'].iloc[-1]:.2f}")
        with col3:
            st.metric("Low", f"{currency_symbol}{data['Low'].iloc[-1]:.2f}")
        with col4:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        with col5:
            st.metric("Average Volume", f"{data['Volume'].mean():,.0f}")
                    
    with tab3:
        st.subheader("📋 Historical Data Table")
        
        # Display options
        col1, col2 = st.columns([1, 3])
        with col1:
            show_rows = st.selectbox("Show rows", [10, 25, 50, 100, "All"], index=1)
        
        # Prepare display data
        display_data = data.copy()
        display_data.index = display_data.index.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format numerical columns with dynamic currency
        currency = st.session_state.stock_info.get('currency', 'USD') if st.session_state.stock_info else 'USD'
        currency_symbol = get_currency_symbol(currency)
        
        for col in ['Open', 'High', 'Low', 'Close']:
            display_data[col] = display_data[col].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{x:,.0f}")
        
        # Show data
        if show_rows == "All":
            st.dataframe(display_data, use_container_width=True, height=600)
        else:
            st.dataframe(display_data.tail(show_rows), use_container_width=True, height=600)
        
        st.info(f"Total records: {len(data)}")
                    
    with tab4:
        st.subheader("📈 Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Price Statistics")
            price_stats = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
                'Value': [
                    f"${data['Close'].mean():.2f}",
                    f"${data['Close'].median():.2f}",
                    f"${data['Close'].std():.2f}",
                    f"${data['Close'].min():.2f}",
                    f"${data['Close'].max():.2f}",
                    f"${data['Close'].quantile(0.25):.2f}",
                    f"${data['Close'].quantile(0.75):.2f}"
                ]
            })
            st.dataframe(price_stats, hide_index=True, use_container_width=True)
            
            st.markdown("### Returns Analysis")
            returns = data['Close'].pct_change().dropna()
            returns_stats = pd.DataFrame({
                'Metric': ['Daily Return', 'Volatility (Daily)', 'Sharpe Ratio (Annualized)', 'Max Drawdown'],
                'Value': [
                    f"{returns.mean()*100:.4f}%",
                    f"{returns.std()*100:.4f}%",
                    f"{(returns.mean()/returns.std())*np.sqrt(252):.2f}" if returns.std() != 0 else "N/A",
                    f"{((data['Close'].cummax() - data['Close'])/data['Close'].cummax()).max()*100:.2f}%"
                ]
            })
            st.dataframe(returns_stats, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("### Volume Statistics")
            volume_stats = pd.DataFrame({
                'Statistic': ['Mean Volume', 'Median Volume', 'Max Volume', 'Min Volume', 'Total Volume'],
                'Value': [
                    f"{data['Volume'].mean():,.0f}",
                    f"{data['Volume'].median():,.0f}",
                    f"{data['Volume'].max():,.0f}",
                    f"{data['Volume'].min():,.0f}",
                    f"{data['Volume'].sum():,.0f}"
                ]
            })
            st.dataframe(volume_stats, hide_index=True, use_container_width=True)
            
            st.markdown("### Price Movement")
            up_days = (data['Close'] > data['Open']).sum()
            down_days = (data['Close'] < data['Open']).sum()
            unchanged_days = (data['Close'] == data['Open']).sum()
            
            movement_stats = pd.DataFrame({
                'Movement': ['Up Days', 'Down Days', 'Unchanged Days', 'Up/Down Ratio'],
                'Value': [
                    f"{up_days} ({up_days/len(data)*100:.1f}%)",
                    f"{down_days} ({down_days/len(data)*100:.1f}%)",
                    f"{unchanged_days} ({unchanged_days/len(data)*100:.1f}%)",
                    f"{up_days/down_days:.2f}" if down_days > 0 else "N/A"
                ]
            })
            st.dataframe(movement_stats, hide_index=True, use_container_width=True)
    
    with tab5:
        st.subheader("📰 Latest News")
        
        # Function to fetch news using direct yfinance
        @st.cache_data(ttl=600)  # Cache for 10 minutes
        def fetch_news(symbol):
            try:
                stock = yf.Ticker(symbol)
                news = stock.news
                
                # Process and clean the news data
                clean_news = []
                st.write(f"🔍 Debug: Raw news count from Yahoo Finance: {len(news)}")
                
                for article in news:
                    # Extract title from different possible locations
                    title = ''
                    if 'content' in article and isinstance(article['content'], dict):
                        title = article['content'].get('title', '')
                    if not title:
                        title = article.get('title', '')
                    
                    # Extract content/summary from different possible locations
                    summary = ''
                    if 'content' in article and isinstance(article['content'], dict):
                        summary = article['content'].get('summary') or article['content'].get('description', '')
                    if not summary:
                        summary = article.get('summary', '')
                    
                    # Extract link from different possible locations
                    link = '#'
                    if 'content' in article and isinstance(article['content'], dict):
                        # Try clickThroughUrl first
                        click_through = article['content'].get('clickThroughUrl')
                        if click_through and isinstance(click_through, dict):
                            link = click_through.get('url', '#')
                        elif isinstance(click_through, str):
                            link = click_through
                        
                        # Try canonicalURL if clickThroughUrl didn't work
                        if link == '#':
                            canonical = article['content'].get('canonicalURL')
                            if canonical and isinstance(canonical, dict):
                                link = canonical.get('url', '#')
                            elif isinstance(canonical, str):
                                link = canonical
                    
                    # Fallback to top-level link if nested extraction failed
                    if link == '#':
                        link = article.get('link', '#')
                    
                    # Extract publisher from different possible locations
                    publisher = 'Unknown'
                    if 'content' in article and isinstance(article['content'], dict):
                        provider = article['content'].get('provider')
                        if provider and isinstance(provider, dict):
                            publisher = provider.get('displayName', 'Unknown')
                    
                    # Fallback to top-level publisher if nested extraction failed
                    if publisher == 'Unknown':
                        publisher = article.get('publisher', 'Unknown')
                    
                    # Extract publish time from different possible locations
                    publish_time = None
                    if 'content' in article and isinstance(article['content'], dict):
                        publish_time = article['content'].get('pubDate')
                    
                    # Fallback to top-level providerPublishTime if nested extraction failed
                    if not publish_time:
                        publish_time = article.get('providerPublishTime')
                    
                    # Format date
                    publish_date = "Unknown date"
                    if publish_time:
                        try:
                            from datetime import datetime
                            if isinstance(publish_time, (int, float)):
                                publish_date = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                            elif isinstance(publish_time, str):
                                # Handle string date formats
                                try:
                                    from dateutil.parser import parse
                                    parsed_date = parse(publish_time)
                                    publish_date = parsed_date.strftime('%Y-%m-%d %H:%M')
                                except:
                                    publish_date = str(publish_time)
                            else:
                                publish_date = str(publish_time)
                        except:
                            publish_date = "Unknown date"
                    
                    # Only include articles with title and content
                    if title and summary:
                        clean_news.append({
                            'title': title,
                            'content': summary,
                            'link': link,
                            'publisher': publisher,
                            'published_date': publish_date
                        })
                
                st.write(f"🔍 Debug: Articles with valid content: {len(clean_news)}")
                return clean_news, None
            except Exception as e:
                return None, str(e)
        
        # Fetch news data
        with st.spinner(f"Fetching latest news for {symbol}..."):
            news_data, news_error = fetch_news(symbol)
        
        if news_error:
            st.error(f"Error fetching news: {news_error}")
        elif not news_data:
            st.info(f"No recent news found for {symbol}")
        else:
            # Store news data in session state for AI Chat
            st.session_state.news_data = news_data
            # Show how many articles we actually have
            st.info(f"📰 Found {len(news_data)} news articles from Yahoo Finance")
            
            # Index news data for RAG if available
            if RAG_AVAILABLE:
                rag_integration = get_rag_integration()
                if rag_integration.is_available():
                    with st.spinner("🔍 Indexing news data for RAG..."):
                        indexed_counts = rag_integration.index_stock_data(
                            symbol, None, {}, news_data, None
                        )
                        if "error" not in indexed_counts:
                            # Calculate total news chunks across all models
                            total_news_chunks = 0
                            news_results = indexed_counts.get('news', {})
                            if isinstance(news_results, dict):
                                for model_key, count in news_results.items():
                                    if isinstance(count, int) and count > 0:
                                        total_news_chunks += count
                            
                            if total_news_chunks > 0:
                                st.success(f"📚 Indexed {total_news_chunks} news chunks across all models for enhanced AI analysis")
            
            # Display all news articles
            st.markdown(f"### Showing all {len(news_data)} articles")
            
            articles_to_show = news_data
            for i, article in enumerate(articles_to_show):
                # Create news article card
                with st.container():
                    # Article title as clickable link that opens in new tab
                    st.markdown(f"""
                    <h4><a href="{article['link']}" target="_blank" style="color: #1f77b4; text-decoration: none;">{article['title']}</a></h4>
                    """, unsafe_allow_html=True)
                    
                    # Article content
                    content = article['content']
                    if len(content) > 300:
                        content = content[:300] + "..."
                    st.write(content)
                    
                    # Metadata
                    st.caption(f"📰 {article['publisher']} | 📅 {article['published_date']}")
                    
                    st.divider()
            
            # Summary statistics
            st.markdown("### News Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Articles", len(news_data))
            
            with col2:
                publishers = set(article['publisher'] for article in news_data)
                st.metric("News Sources", len(publishers))
    
    with tab6:
        st.subheader("📊 Financial Statements")
        
        # Function to fetch financial statements
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def fetch_financial_statements(symbol):
            try:
                stock = yf.Ticker(symbol)
                
                # Get financial statements
                quarterly_financials = stock.quarterly_financials
                yearly_financials = stock.financials
                quarterly_balance_sheet = stock.quarterly_balance_sheet
                yearly_balance_sheet = stock.balance_sheet
                quarterly_cashflow = stock.quarterly_cashflow
                yearly_cashflow = stock.cashflow
                
                return {
                    'quarterly_financials': quarterly_financials,
                    'yearly_financials': yearly_financials,
                    'quarterly_balance_sheet': quarterly_balance_sheet,
                    'yearly_balance_sheet': yearly_balance_sheet,
                    'quarterly_cashflow': quarterly_cashflow,
                    'yearly_cashflow': yearly_cashflow
                }, None
            except Exception as e:
                return None, str(e)
        
        # Fetch financial statements
        with st.spinner(f"Fetching financial statements for {symbol}..."):
            statements, statements_error = fetch_financial_statements(symbol)
        
        if statements_error:
            st.error(f"Error fetching financial statements: {statements_error}")
        elif not statements or all(df.empty for df in statements.values()):
            st.info(f"No financial statements found for {symbol}")
        else:
            # Financial statements tabs
            stmt_tab1, stmt_tab2, stmt_tab3 = st.tabs(["📊 Income Statement", "🏦 Balance Sheet", "💰 Cash Flow"])
            
            with stmt_tab1:
                st.subheader("Income Statement")
                
                # Period selection for Income Statement
                period_col1, period_col2 = st.columns(2)
                with period_col1:
                    income_period = st.radio("Select Period", ["Quarterly", "Yearly"], key="income_period")
                
                # Select the appropriate data
                if income_period == "Quarterly":
                    income_data = statements['quarterly_financials']
                    period_label = "Quarterly"
                else:
                    income_data = statements['yearly_financials']
                    period_label = "Annual"
                
                if not income_data.empty:
                    # Display income statement
                    st.markdown(f"### {period_label} Income Statement")
                    
                    # Format the data for display
                    display_income = income_data.copy()
                    
                    # Convert to millions for better readability
                    display_income = display_income / 1_000_000
                    
                    # Round to 2 decimal places
                    display_income = display_income.round(2)
                    
                    # Format column names (dates)
                    display_income.columns = [col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col) for col in display_income.columns]
                    
                    # Display the dataframe
                    st.dataframe(display_income, use_container_width=True, height=600)
                    st.caption("All figures in millions")
                    
                    # Gross Profit Analysis Section
                    st.markdown("### 💰 Gross Profit Analysis")
                    
                    # Check for required fields
                    has_revenue = 'Total Revenue' in display_income.index
                    has_cost = 'Cost Of Revenue' in display_income.index
                    has_gross_profit = 'Gross Profit' in display_income.index
                    
                    if has_revenue and has_gross_profit:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Get latest period data
                        latest_period = display_income.columns[0]
                        
                        with col1:
                            revenue = display_income.loc['Total Revenue', latest_period]
                            st.metric("Total Revenue", f"${revenue:,.0f}M")
                        
                        with col2:
                            if has_cost:
                                cost = display_income.loc['Cost Of Revenue', latest_period]
                                st.metric("Cost of Revenue", f"${cost:,.0f}M")
                            
                        with col3:
                            gross_profit = display_income.loc['Gross Profit', latest_period]
                            st.metric("Gross Profit", f"${gross_profit:,.0f}M")
                        
                        with col4:
                            # Calculate gross margin
                            gross_margin = (gross_profit / revenue) * 100
                            st.metric("Gross Margin", f"{gross_margin:.1f}%")
                        
                        # Gross Profit Trend Chart
                        if len(display_income.columns) >= 2:
                            st.markdown("#### Gross Profit Trend")
                            
                            fig_gross = go.Figure()
                            
                            # Add revenue
                            fig_gross.add_trace(go.Scatter(
                                x=display_income.columns,
                                y=display_income.loc['Total Revenue'],
                                mode='lines+markers',
                                name='Total Revenue',
                                line=dict(color='#4CAF50', width=3)
                            ))
                            
                            # Add gross profit
                            fig_gross.add_trace(go.Scatter(
                                x=display_income.columns,
                                y=display_income.loc['Gross Profit'],
                                mode='lines+markers',
                                name='Gross Profit',
                                line=dict(color='#2196F3', width=3)
                            ))
                            
                            # Add cost of revenue if available
                            if has_cost:
                                fig_gross.add_trace(go.Scatter(
                                    x=display_income.columns,
                                    y=display_income.loc['Cost Of Revenue'],
                                    mode='lines+markers',
                                    name='Cost of Revenue',
                                    line=dict(color='#FF5722', width=3)
                                ))
                            
                            fig_gross.update_layout(
                                title=f"{symbol} - Revenue and Gross Profit Trend ({period_label})",
                                xaxis_title="Period",
                                yaxis_title="Amount (Millions)",
                                template='plotly_dark',
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(fig_gross, use_container_width=True)
                            
                            # Gross Margin Trend
                            st.markdown("#### Gross Margin Trend")
                            
                            # Calculate gross margin for all periods
                            gross_margins = []
                            periods = []
                            for period in display_income.columns:
                                revenue_val = display_income.loc['Total Revenue', period]
                                gross_profit_val = display_income.loc['Gross Profit', period]
                                if revenue_val != 0:
                                    margin = (gross_profit_val / revenue_val) * 100
                                    gross_margins.append(margin)
                                    periods.append(period)
                            
                            if gross_margins:
                                fig_margin = go.Figure()
                                fig_margin.add_trace(go.Scatter(
                                    x=periods,
                                    y=gross_margins,
                                    mode='lines+markers',
                                    name='Gross Margin %',
                                    line=dict(color='#FF9800', width=3),
                                    fill='tonexty'
                                ))
                                
                                fig_margin.update_layout(
                                    title=f"{symbol} - Gross Margin Trend ({period_label})",
                                    xaxis_title="Period",
                                    yaxis_title="Gross Margin (%)",
                                    template='plotly_dark',
                                    hovermode='x unified',
                                    height=300
                                )
                                
                                st.plotly_chart(fig_margin, use_container_width=True)
                    
                    # Key metrics visualization
                    if len(display_income.columns) >= 2:
                        st.markdown("### 📊 Key Metrics Trend")
                        
                        # Select key metrics to plot
                        key_metrics = []
                        for metric in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']:
                            if metric in display_income.index:
                                key_metrics.append(metric)
                        
                        if key_metrics:
                            fig_income = go.Figure()
                            
                            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
                            
                            for i, metric in enumerate(key_metrics):
                                fig_income.add_trace(go.Scatter(
                                    x=display_income.columns,
                                    y=display_income.loc[metric],
                                    mode='lines+markers',
                                    name=metric,
                                    line=dict(width=3, color=colors[i % len(colors)])
                                ))
                            
                            fig_income.update_layout(
                                title=f"{symbol} - Key Income Statement Metrics ({period_label})",
                                xaxis_title="Period",
                                yaxis_title="Amount (Millions)",
                                template='plotly_dark',
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(fig_income, use_container_width=True)
                else:
                    st.info(f"No {income_period.lower()} income statement data available for {symbol}")
            
            with stmt_tab2:
                st.subheader("Balance Sheet")
                
                # Period selection for Balance Sheet
                balance_period = st.radio("Select Period", ["Quarterly", "Yearly"], key="balance_period")
                
                # Select the appropriate data
                if balance_period == "Quarterly":
                    balance_data = statements['quarterly_balance_sheet']
                    period_label = "Quarterly"
                else:
                    balance_data = statements['yearly_balance_sheet']
                    period_label = "Annual"
                
                if not balance_data.empty:
                    st.markdown(f"### {period_label} Balance Sheet")
                    
                    # Format the data for display
                    display_balance = balance_data.copy()
                    
                    # Convert to millions for better readability
                    display_balance = display_balance / 1_000_000
                    
                    # Round to 2 decimal places
                    display_balance = display_balance.round(2)
                    
                    # Format column names (dates)
                    display_balance.columns = [col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col) for col in display_balance.columns]
                    
                    # Display the dataframe
                    st.dataframe(display_balance, use_container_width=True, height=600)
                    st.caption("All figures in millions")
                    
                    # Key balance sheet metrics
                    if len(display_balance.columns) >= 2:
                        st.markdown("### Key Balance Sheet Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        latest_col = display_balance.columns[0]
                        
                        with col1:
                            if 'Total Assets' in display_balance.index:
                                total_assets = display_balance.loc['Total Assets', latest_col]
                                st.metric("Total Assets", f"${total_assets:,.0f}M")
                        
                        with col2:
                            if 'Total Debt' in display_balance.index:
                                total_debt = display_balance.loc['Total Debt', latest_col]
                                st.metric("Total Debt", f"${total_debt:,.0f}M")
                        
                        with col3:
                            if 'Total Stockholder Equity' in display_balance.index:
                                equity = display_balance.loc['Total Stockholder Equity', latest_col]
                                st.metric("Stockholder Equity", f"${equity:,.0f}M")
                else:
                    st.info(f"No {balance_period.lower()} balance sheet data available for {symbol}")
            
            with stmt_tab3:
                st.subheader("Cash Flow Statement")
                
                # Period selection for Cash Flow
                cashflow_period = st.radio("Select Period", ["Quarterly", "Yearly"], key="cashflow_period")
                
                # Select the appropriate data
                if cashflow_period == "Quarterly":
                    cashflow_data = statements['quarterly_cashflow']
                    period_label = "Quarterly"
                else:
                    cashflow_data = statements['yearly_cashflow']
                    period_label = "Annual"
                
                if not cashflow_data.empty:
                    st.markdown(f"### {period_label} Cash Flow Statement")
                    
                    # Format the data for display
                    display_cashflow = cashflow_data.copy()
                    
                    # Convert to millions for better readability
                    display_cashflow = display_cashflow / 1_000_000
                    
                    # Round to 2 decimal places
                    display_cashflow = display_cashflow.round(2)
                    
                    # Format column names (dates)
                    display_cashflow.columns = [col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col) for col in display_cashflow.columns]
                    
                    # Display the dataframe
                    st.dataframe(display_cashflow, use_container_width=True, height=600)
                    st.caption("All figures in millions")
                    
                    # Cash flow visualization
                    if len(display_cashflow.columns) >= 2:
                        st.markdown("### Cash Flow Analysis")
                        
                        # Key cash flow metrics
                        key_cf_metrics = []
                        for metric in ['Operating Cash Flow', 'Free Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']:
                            if metric in display_cashflow.index:
                                key_cf_metrics.append(metric)
                        
                        if key_cf_metrics:
                            fig_cf = go.Figure()
                            
                            for metric in key_cf_metrics:
                                fig_cf.add_trace(go.Bar(
                                    x=display_cashflow.columns,
                                    y=display_cashflow.loc[metric],
                                    name=metric,
                                    opacity=0.8
                                ))
                            
                            fig_cf.update_layout(
                                title=f"{symbol} - Cash Flow Analysis ({period_label})",
                                xaxis_title="Period",
                                yaxis_title="Cash Flow (Millions)",
                                template='plotly_dark',
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig_cf, use_container_width=True)
                        
                        # Key cash flow metrics display
                        if len(display_cashflow.columns) >= 1:
                            st.markdown("### Latest Cash Flow Metrics")
                            
                            col1, col2, col3 = st.columns(3)
                            latest_col = display_cashflow.columns[0]
                            
                            with col1:
                                if 'Operating Cash Flow' in display_cashflow.index:
                                    ocf = display_cashflow.loc['Operating Cash Flow', latest_col]
                                    st.metric("Operating Cash Flow", f"${ocf:,.0f}M")
                            
                            with col2:
                                if 'Free Cash Flow' in display_cashflow.index:
                                    fcf = display_cashflow.loc['Free Cash Flow', latest_col]
                                    st.metric("Free Cash Flow", f"${fcf:,.0f}M")
                            
                            with col3:
                                if 'Capital Expenditures' in display_cashflow.index:
                                    capex = display_cashflow.loc['Capital Expenditures', latest_col]
                                    st.metric("Capital Expenditures", f"${capex:,.0f}M")
                else:
                    st.info(f"No {cashflow_period.lower()} cash flow data available for {symbol}")
    
    with tab7:
        st.subheader("🔗 Company Relationships")
        
        if GRAPH_AVAILABLE:
            graph_db = get_graph_db()
            
            if graph_db and graph_db.is_connected:
                st.success("✅ Graph Database Connected")
                
                # Index current company data
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("**Company Data Indexing**")
                with col2:
                    if st.button("🔄 Index Company", help="Index this company's data into the graph database"):
                        with st.spinner("Indexing company data..."):
                            try:
                                # Prepare company data
                                company_data = {
                                    "symbol": symbol,
                                    "name": info.get("longName", info.get("shortName", symbol)),
                                    "sector": info.get("sector", "Unknown"),
                                    "industry": info.get("industry", "Unknown"),
                                    "marketCap": info.get("marketCap"),
                                    "employees": info.get("fullTimeEmployees"),
                                    "headquarters": f"{info.get('city', '')}, {info.get('country', '')}".strip(", "),
                                    "description": info.get("longBusinessSummary", "")[:500],
                                    "currency": info.get("currency", "USD"),
                                    "country": info.get("country", "Unknown"),
                                    "website": info.get("website", "")
                                }
                                
                                success = graph_db.create_company(company_data)
                                if success:
                                    st.success(f"✅ {symbol} indexed successfully!")
                                else:
                                    st.error("❌ Failed to index company")
                            except Exception as e:
                                st.error(f"❌ Error indexing company: {e}")
                
                # Relationship Analysis
                st.markdown("### 🔍 Relationship Analysis")
                
                analysis_cols = st.columns(3)
                
                with analysis_cols[0]:
                    max_hops = st.selectbox("Relationship Depth", [1, 2, 3], index=1)
                
                with analysis_cols[1]:
                    limit = st.selectbox("Max Results", [10, 20, 50], index=1)
                
                with analysis_cols[2]:
                    if st.button("🔍 Find Relationships"):
                        with st.spinner("Analyzing relationships..."):
                            try:
                                relationships = graph_db.find_related_companies(symbol, max_hops, limit)
                                
                                if relationships:
                                    st.success(f"Found {len(relationships)} related companies")
                                    
                                    # Display relationships
                                    for rel in relationships:
                                        with st.expander(f"{rel['symbol']} - {rel['name']} (Distance: {rel['distance']})"):
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("Sector", rel.get('sector', 'Unknown'))
                                                st.metric("Distance", rel['distance'])
                                            with col2:
                                                market_cap = rel.get('marketCap', 0)
                                                if market_cap:
                                                    st.metric("Market Cap", f"${market_cap:,.0f}")
                                                connections = rel.get('connection_types', [])
                                                if connections:
                                                    st.write("**Connection Types:**")
                                                    for conn in set(connections):
                                                        st.write(f"• {conn}")
                                else:
                                    st.info("No relationships found. Try indexing more companies or check if the company exists in the graph.")
                            except Exception as e:
                                st.error(f"Error finding relationships: {e}")
                
                # Portfolio Analysis
                st.markdown("### 📊 Portfolio Correlation Analysis")
                
                portfolio_symbols = st.text_input(
                    "Enter portfolio symbols (comma-separated)",
                    placeholder="AAPL,MSFT,GOOGL,AMZN",
                    help="Enter 2-10 stock symbols to analyze correlations"
                )
                
                if portfolio_symbols and st.button("📊 Analyze Portfolio"):
                    symbols_list = [s.strip().upper() for s in portfolio_symbols.split(",") if s.strip()]
                    
                    if len(symbols_list) >= 2:
                        with st.spinner("Analyzing portfolio relationships..."):
                            try:
                                correlations = graph_db.analyze_portfolio_correlations(symbols_list)
                                
                                if correlations:
                                    st.success(f"Found {len(correlations)} relationship pairs")
                                    
                                    # Display correlation matrix style
                                    for corr in correlations:
                                        correlation_val = corr.get('correlation') or 0
                                        rel_type = corr.get('relationship_type', 'Unknown')
                                        
                                        color = "🔴" if abs(correlation_val) > 0.7 else "🟡" if abs(correlation_val) > 0.4 else "🟢"
                                        
                                        st.write(f"{color} **{corr['stock1']} ↔ {corr['stock2']}**")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.write(f"Type: {rel_type}")
                                        with col2:
                                            if correlation_val:
                                                st.write(f"Correlation: {correlation_val:.3f}")
                                        with col3:
                                            strength = corr.get('strength') or 0
                                            if strength:
                                                st.write(f"Strength: {strength:.3f}")
                                else:
                                    st.info("No correlations found between the portfolio holdings.")
                            except Exception as e:
                                st.error(f"Error analyzing portfolio: {e}")
                    else:
                        st.warning("Please enter at least 2 valid symbols")
                
                # Market Influence Network
                st.markdown("### 🌐 Market Influence Network")
                
                if st.button("🌐 Analyze Market Network"):
                    with st.spinner("Analyzing market influence network..."):
                        try:
                            network = graph_db.get_market_influence_network(symbol, depth=2)
                            
                            if network and network.get('total_connections', 0) > 0:
                                st.success("Market network analysis complete!")
                                
                                # Network metrics
                                metrics_cols = st.columns(4)
                                with metrics_cols[0]:
                                    st.metric("Total Connections", network.get('total_connections', 0))
                                with metrics_cols[1]:
                                    st.metric("Unique Sectors", network.get('unique_sectors', 0))
                                with metrics_cols[2]:
                                    avg_cap = network.get('average_market_cap', 0)
                                    st.metric("Avg Market Cap", f"${avg_cap:,.0f}")
                                with metrics_cols[3]:
                                    sectors = network.get('sectors', [])
                                    st.metric("Sectors", len(sectors))
                                
                                # Connected sectors
                                if sectors:
                                    st.write("**Connected Sectors:**")
                                    st.write(", ".join(sectors))
                                
                                # Top connections
                                network_data = network.get('network_data', [])
                                if network_data:
                                    st.write("**Top Connected Companies:**")
                                    for i, conn in enumerate(network_data[:10], 1):
                                        market_cap = conn.get('market_cap', 0)
                                        st.write(f"{i}. {conn['connected_symbol']} ({conn['connected_name']}) - "
                                               f"${market_cap:,.0f} - Distance: {conn['distance']}")
                            else:
                                st.info("No market network data found. The company may not be indexed yet.")
                        except Exception as e:
                            st.error(f"Error analyzing network: {e}")
                
                # Graph Statistics
                with st.expander("📊 Graph Database Statistics"):
                    if st.button("📊 Refresh Statistics"):
                        try:
                            stats = graph_db.get_graph_statistics()
                            health = graph_db.health_check()
                            
                            st.write("**Database Health:**")
                            st.json(health)
                            
                            st.write("**Node Counts:**")
                            node_counts = stats.get('node_counts', {})
                            for node_type, count in node_counts.items():
                                st.write(f"• {node_type}: {count:,}")
                            
                            st.write("**Relationship Counts:**")
                            rel_counts = stats.get('relationship_counts', {})
                            for rel_type, count in rel_counts.items():
                                st.write(f"• {rel_type}: {count:,}")
                            
                        except Exception as e:
                            st.error(f"Error getting statistics: {e}")
            
            else:
                st.warning("⚠️ Graph Database Not Connected")
                st.info("Neo4j database is not available. Please check your connection settings.")
                
                # Show connection info
                st.markdown("**Connection Settings:**")
                st.code(f"""
Neo4j URI: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}
Username: {os.getenv('NEO4J_USERNAME', 'neo4j')}
Password: {'*' * len(os.getenv('NEO4J_PASSWORD', 'financialpass'))}
                """)
        
        else:
            st.info("🔗 Graph Database Features Not Available")
            st.markdown("""
            **Graph database features require Neo4j to be installed and configured.**
            
            To enable relationship analysis:
            1. Install Neo4j dependencies: `pip install neo4j py2neo`
            2. Set up Neo4j database (Docker or local installation)
            3. Configure environment variables in `.env` file:
               ```
               NEO4J_URI=bolt://localhost:7687
               NEO4J_USERNAME=neo4j
               NEO4J_PASSWORD=your_password
               ```
            4. Restart the application
            """)
    
    with tab8:
        st.subheader("💾 Export Data")
        
        st.markdown("### Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Select Export Format",
                ["CSV", "JSON"],
                key="export_format_selectbox"
            )
            
            include_index = st.checkbox("Include Date/Time Index", value=True)
        
        # Prepare export data
        export_data = data.copy()
        if include_index:
            export_data.reset_index(inplace=True)
        
        # Export buttons
        if export_format == "CSV":
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{symbol}_{selected_period.replace(' ', '_')}_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        else:  # JSON
            json_data = export_data.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"{symbol}_{selected_period.replace(' ', '_')}_data.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Preview
        st.markdown("### Data Preview")
        st.dataframe(export_data.head(10), use_container_width=True)
        st.info(f"Total rows to export: {len(export_data)}")

else:
    st.info("👈 Please configure the settings in the sidebar and click 'Fetch Data' to get started.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        Built with Streamlit and Yahoo Finance API
    </div>
    """,
    unsafe_allow_html=True
)
