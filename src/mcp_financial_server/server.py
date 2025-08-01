import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)


class FinancialServer:
    def __init__(self):
        self.server = Server("financial-server")
        self._setup_handlers()
        
    def _setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="get_stock_history",
                    description="Fetch historical price data for a stock symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                            },
                            "period": {
                                "type": "string",
                                "description": "Time period for data: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max",
                                "default": "1mo"
                            },
                            "interval": {
                                "type": "string",
                                "description": "Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo",
                                "default": "1d"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="get_stock_info",
                    description="Get detailed information about a stock",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="get_stock_quote",
                    description="Get real-time quote for a stock",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="get_news",
                    description="Fetch latest financial news for a stock symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of articles to fetch (default: 10, max: 20)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 20
                            }
                        },
                        "required": ["symbol"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[TextContent | ImageContent | EmbeddedResource]:
            if name == "get_stock_history":
                return await self._get_stock_history(arguments)
            elif name == "get_stock_info":
                return await self._get_stock_info(arguments)
            elif name == "get_stock_quote":
                return await self._get_stock_quote(arguments)
            elif name == "get_news":
                return await self._get_news(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def _get_stock_history(self, arguments: Dict[str, Any]) -> List[TextContent]:
        symbol = arguments.get("symbol")
        period = arguments.get("period", "1mo")
        interval = arguments.get("interval", "1d")
        
        if not symbol:
            return [TextContent(
                type="text",
                text="Error: Stock symbol is required"
            )]
        
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if history.empty:
                return [TextContent(
                    type="text",
                    text=f"No data found for symbol: {symbol}"
                )]
            
            # Convert DataFrame to JSON format
            data = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data": []
            }
            
            for index, row in history.iterrows():
                data["data"].append({
                    "date": index.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": round(row["Open"], 2),
                    "high": round(row["High"], 2),
                    "low": round(row["Low"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"])
                })
            
            return [TextContent(
                type="text",
                text=json.dumps(data, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error fetching data for {symbol}: {str(e)}"
            )]
    
    async def _get_stock_info(self, arguments: Dict[str, Any]) -> List[TextContent]:
        symbol = arguments.get("symbol")
        
        if not symbol:
            return [TextContent(
                type="text",
                text="Error: Stock symbol is required"
            )]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            relevant_info = {
                "symbol": symbol,
                "longName": info.get("longName", "N/A"),
                "shortName": info.get("shortName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "marketCap": info.get("marketCap", "N/A"),
                "currency": info.get("currency", "N/A"),
                "exchange": info.get("exchange", "N/A"),
                "quoteType": info.get("quoteType", "N/A"),
                "previousClose": info.get("previousClose", "N/A"),
                "dayHigh": info.get("dayHigh", "N/A"),
                "dayLow": info.get("dayLow", "N/A"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", "N/A"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", "N/A"),
                "volume": info.get("volume", "N/A"),
                "averageVolume": info.get("averageVolume", "N/A"),
                "beta": info.get("beta", "N/A"),
                "trailingPE": info.get("trailingPE", "N/A"),
                "forwardPE": info.get("forwardPE", "N/A"),
                "dividendYield": info.get("dividendYield", "N/A"),
                "payoutRatio": info.get("payoutRatio", "N/A"),
                "profitMargins": info.get("profitMargins", "N/A"),
                "grossMargins": info.get("grossMargins", "N/A"),
                "ebitdaMargins": info.get("ebitdaMargins", "N/A"),
                "operatingMargins": info.get("operatingMargins", "N/A")
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(relevant_info, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error fetching info for {symbol}: {str(e)}"
            )]
    
    async def _get_stock_quote(self, arguments: Dict[str, Any]) -> List[TextContent]:
        symbol = arguments.get("symbol")
        
        if not symbol:
            return [TextContent(
                type="text",
                text="Error: Stock symbol is required"
            )]
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get real-time quote
            quote = ticker.history(period="1d", interval="1m").tail(1)
            
            if quote.empty:
                return [TextContent(
                    type="text",
                    text=f"No quote data found for symbol: {symbol}"
                )]
            
            latest = quote.iloc[0]
            
            quote_data = {
                "symbol": symbol,
                "timestamp": quote.index[0].strftime("%Y-%m-%d %H:%M:%S"),
                "price": round(latest["Close"], 2),
                "open": round(latest["Open"], 2),
                "high": round(latest["High"], 2),
                "low": round(latest["Low"], 2),
                "volume": int(latest["Volume"])
            }
            
            # Get additional quote info
            info = ticker.info
            quote_data["previousClose"] = info.get("previousClose", "N/A")
            quote_data["dayChange"] = round(latest["Close"] - info.get("previousClose", latest["Close"]), 2) if info.get("previousClose") else "N/A"
            quote_data["dayChangePercent"] = round(((latest["Close"] - info.get("previousClose", latest["Close"])) / info.get("previousClose", 1)) * 100, 2) if info.get("previousClose") else "N/A"
            
            return [TextContent(
                type="text",
                text=json.dumps(quote_data, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error fetching quote for {symbol}: {str(e)}"
            )]
    
    async def _get_news(self, arguments: Dict[str, Any]) -> List[TextContent]:
        symbol = arguments.get("symbol")
        limit = arguments.get("limit", 10)
        
        if not symbol:
            return [TextContent(
                type="text",
                text="Error: Stock symbol is required"
            )]
        
        try:
            # Use yfinance to get news for the symbol
            ticker = yf.Ticker(symbol)
            yf_news = ticker.news
            
            if not yf_news:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "symbol": symbol,
                        "articles": [],
                        "message": f"No news found for {symbol}"
                    }, indent=2)
                )]
            
            # Process news articles and extract clean title and content
            articles = []
            for i, news_item in enumerate(yf_news[:limit]):
                try:
                    # Extract title from different possible locations
                    title = None
                    if 'content' in news_item and isinstance(news_item['content'], dict):
                        title = news_item['content'].get('title')
                    if not title:
                        title = news_item.get('title')
                    if not title:
                        title = 'No title available'
                    
                    # Extract content from different possible locations
                    content = None
                    if 'content' in news_item and isinstance(news_item['content'], dict):
                        content = news_item['content'].get('summary') or news_item['content'].get('description')
                    if not content:
                        content = news_item.get('summary')
                    if not content:
                        content = 'No content available'
                    
                    # Extract link from different possible locations
                    link = '#'
                    if 'content' in news_item and isinstance(news_item['content'], dict):
                        # Try clickThroughUrl first
                        click_through = news_item['content'].get('clickThroughUrl')
                        if click_through and isinstance(click_through, dict):
                            link = click_through.get('url', '#')
                        elif isinstance(click_through, str):
                            link = click_through
                        
                        # Try canonicalURL if clickThroughUrl didn't work
                        if link == '#':
                            canonical = news_item['content'].get('canonicalURL')
                            if canonical and isinstance(canonical, dict):
                                link = canonical.get('url', '#')
                            elif isinstance(canonical, str):
                                link = canonical
                    
                    # Fallback to top-level link if nested extraction failed
                    if link == '#':
                        link = news_item.get('link', '#')
                    
                    # Extract publisher from different possible locations
                    publisher = 'Unknown'
                    if 'content' in news_item and isinstance(news_item['content'], dict):
                        provider = news_item['content'].get('provider')
                        if provider and isinstance(provider, dict):
                            publisher = provider.get('displayName', 'Unknown')
                    
                    # Fallback to top-level publisher if nested extraction failed
                    if publisher == 'Unknown':
                        publisher = news_item.get('publisher', 'Unknown')
                    
                    # Extract publish time from different possible locations
                    publish_time = None
                    if 'content' in news_item and isinstance(news_item['content'], dict):
                        publish_time = news_item['content'].get('pubDate')
                    
                    # Fallback to top-level providerPublishTime if nested extraction failed
                    if not publish_time:
                        publish_time = news_item.get('providerPublishTime')
                    
                    published_date = "Unknown date"
                    if publish_time:
                        try:
                            if isinstance(publish_time, (int, float)):
                                published_date = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                            elif isinstance(publish_time, str):
                                # Handle string date formats
                                from dateutil.parser import parse
                                parsed_date = parse(publish_time)
                                published_date = parsed_date.strftime('%Y-%m-%d %H:%M')
                            else:
                                published_date = str(publish_time)
                        except:
                            published_date = "Unknown date"
                    
                    # Only include articles with actual content
                    if title and title != 'No title available' and content and content != 'No content available':
                        articles.append({
                            "title": title,
                            "content": content,
                            "link": link,
                            "publisher": publisher,
                            "published_date": published_date
                        })
                        
                except Exception as e:
                    continue  # Skip articles that can't be parsed
            
            # Return clean, simple news data
            news_data = {
                "symbol": symbol,
                "total_articles": len(articles),
                "articles": articles
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(news_data, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error fetching news for {symbol}: {str(e)}"
            )]
    
    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="financial-server",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    server = FinancialServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())