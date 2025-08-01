"""
Example of how to integrate MCP server functionality into Streamlit app
"""

import asyncio
from src.mcp_financial_server.server import (
    get_stock_history,
    get_stock_info,
    get_stock_quote,
    get_news
)

class MCPFinancialClient:
    """Client to interact with MCP financial server functions"""
    
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def fetch_stock_history(self, symbol, period="1mo", interval="1d"):
        """Fetch stock history using MCP server function"""
        return self.loop.run_until_complete(
            get_stock_history(symbol, period, interval)
        )
    
    def fetch_stock_info(self, symbol):
        """Fetch stock info using MCP server function"""
        return self.loop.run_until_complete(
            get_stock_info(symbol)
        )
    
    def fetch_stock_quote(self, symbol):
        """Fetch stock quote using MCP server function"""
        return self.loop.run_until_complete(
            get_stock_quote(symbol)
        )
    
    def fetch_news(self, symbol, count=10):
        """Fetch news using MCP server function"""
        return self.loop.run_until_complete(
            get_news(symbol, count)
        )

# Usage in Streamlit:
# mcp_client = MCPFinancialClient()
# history_data = mcp_client.fetch_stock_history("AAPL", "1mo", "1d")
# info_data = mcp_client.fetch_stock_info("AAPL")