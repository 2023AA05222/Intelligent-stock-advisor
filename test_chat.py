#!/usr/bin/env python3
"""
Test script for AI Chat functionality
"""

import os
import yfinance as yf
import pandas as pd
from google.oauth2.service_account import Credentials
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
            credentials = Credentials.from_service_account_file(
                service_account_path,
                scopes=['https://www.googleapis.com/auth/generative-language.retriever']
            )
            genai.configure(credentials=credentials)
            model = genai.GenerativeModel('gemini-1.5-flash')
            return model
        else:
            print("Neither GOOGLE_AI_API_KEY environment variable nor service account file found.")
            return None
    except Exception as e:
        print(f"Error initializing Gemini AI: {str(e)}")
        return None

def create_financial_context(symbol, stock_data, stock_info, news_data, financial_statements):
    """Create comprehensive context from all available data sources"""
    context_parts = []
    
    # Basic stock information
    if stock_info:
        context_parts.append(f"=== STOCK INFORMATION FOR {symbol} ===")
        context_parts.append(f"Company: {stock_info.get('longName', 'N/A')}")
        context_parts.append(f"Sector: {stock_info.get('sector', 'N/A')}")
        context_parts.append(f"Industry: {stock_info.get('industry', 'N/A')}")
        context_parts.append(f"Market Cap: {stock_info.get('marketCap', 'N/A')}")
        context_parts.append(f"Currency: {stock_info.get('currency', 'USD')}")
        context_parts.append("")
    
    # Recent stock price data
    if stock_data is not None and not stock_data.empty:
        context_parts.append("=== RECENT PRICE DATA ===")
        latest_data = stock_data.tail(5)
        for idx, row in latest_data.iterrows():
            context_parts.append(f"{idx.strftime('%Y-%m-%d')}: Open={row['Open']:.2f}, High={row['High']:.2f}, Low={row['Low']:.2f}, Close={row['Close']:.2f}, Volume={row['Volume']:,}")
        context_parts.append("")
    
    return "\n".join(context_parts)

def get_gemini_response(model, user_question, context):
    """Get response from Gemini model"""
    try:
        prompt = f"""You are a financial analyst AI assistant. You have access to comprehensive financial data for a stock including:
- Stock price history
- Company information
- Financial statements
- Recent news

Please analyze the following data and answer the user's question in a helpful, accurate, and professional manner.

FINANCIAL DATA CONTEXT:
{context}

USER QUESTION: {user_question}

Please provide a detailed and insightful response based on the available data. If you cannot answer based on the provided data, please say so clearly."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def test_chat_functionality():
    """Test the complete chat functionality"""
    print("🧪 Testing AI Chat Functionality")
    print("=" * 50)
    
    # Initialize Gemini
    print("1. Initializing Gemini AI...")
    model = initialize_gemini()
    if not model:
        print("❌ Failed to initialize Gemini")
        return False
    print("✅ Gemini initialized")
    
    # Fetch sample stock data
    print("2. Fetching sample stock data (AAPL)...")
    try:
        ticker = yf.Ticker("AAPL")
        stock_data = ticker.history(period="1mo")
        stock_info = ticker.info
        print("✅ Stock data fetched")
    except Exception as e:
        print(f"❌ Failed to fetch stock data: {e}")
        return False
    
    # Create context
    print("3. Creating financial context...")
    context = create_financial_context("AAPL", stock_data, stock_info, [], None)
    print("✅ Context created")
    print(f"Context length: {len(context)} characters")
    
    # Test sample questions
    print("4. Testing sample questions...")
    test_questions = [
        "What's the recent price trend for this stock?",
        "What company is this and what sector is it in?",
        "How has the stock performed in the last week?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🤔 Question {i}: {question}")
        response = get_gemini_response(model, question, context)
        print(f"🤖 Response preview: {response[:200]}...")
        
        if "Error generating response" in response:
            print(f"❌ Failed to get response for question {i}")
            return False
    
    print("\n✅ All tests passed! AI Chat functionality is working properly.")
    return True

if __name__ == "__main__":
    success = test_chat_functionality()
    if success:
        print("\n🎉 AI Chat feature is ready for use in the Streamlit app!")
    else:
        print("\n❌ AI Chat feature needs debugging.")