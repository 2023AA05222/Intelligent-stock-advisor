#!/usr/bin/env python3
"""
Test script for the MCP Financial Server
"""

import asyncio
import json
from src.mcp_financial_server.server import FinancialServer


async def test_server():
    print("Testing MCP Financial Server...")
    server = FinancialServer()
    
    # Test get_stock_history
    print("\n1. Testing get_stock_history for AAPL...")
    result = await server._get_stock_history({"symbol": "AAPL", "period": "500d", "interval": "1d"})
    data = json.loads(result[0].text)
    print(f"   Found {len(data['data'])} days of data")
    print(f"   Latest close: ${data['data'][-1]['close']}")
    
    # Test get_stock_info
    print("\n2. Testing get_stock_info for MSFT...")
    result = await server._get_stock_info({"symbol": "MSFT"})
    info = json.loads(result[0].text)
    print(f"   Company: {info['longName']}")
    print(f"   Sector: {info['sector']}")
    print(f"   Industry: {info['industry']}")
    
    # Test get_stock_quote
    print("\n3. Testing get_stock_quote for GOOGL...")
    result = await server._get_stock_quote({"symbol": "GOOGL"})
    quote = json.loads(result[0].text)
    print(f"   Current price: ${quote['price']}")
    print(f"   Day change: {quote['dayChange']} ({quote['dayChangePercent']}%)")
    
    # Test get_news
    print("\n4. Testing get_news for AAPL...")
    result = await server._get_news({"symbol": "AAPL", "limit": 5})
    news_data = json.loads(result[0].text)
    
    if 'articles' in news_data and news_data['articles']:
        print(f"   Found {news_data['total_articles']} news articles")
        print(f"   Latest article: {news_data['articles'][0]['title'][:60]}...")
        print(f"   Publisher: {news_data['articles'][0]['publisher']}")
        print(f"   Published: {news_data['articles'][0]['published_date']}")
        
        # Show content preview
        content_preview = news_data['articles'][0]['content'][:100]
        print(f"   Content preview: {content_preview}...")
    else:
        print("   No news articles found or error occurred")
        if 'message' in news_data:
            print(f"   Message: {news_data['message']}")
    
    # Test with different symbol
    print("\n5. Testing get_news for TSLA...")
    result = await server._get_news({"symbol": "TSLA", "limit": 3})
    news_data = json.loads(result[0].text)
    
    if 'articles' in news_data and news_data['articles']:
        print(f"   Found {news_data['total_articles']} news articles for TSLA")
        for i, article in enumerate(news_data['articles'][:2]):
            print(f"   Article {i+1}: {article['title'][:50]}...")
    else:
        print("   No news articles found for TSLA")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_server())