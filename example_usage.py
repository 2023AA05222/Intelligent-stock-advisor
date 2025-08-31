#!/usr/bin/env python3
"""
Example usage of the MCP Financial Server
This demonstrates how to integrate the server with MCP clients
"""

import asyncio
import json
from datetime import datetime


async def demo_usage():
    """
    This simulates how an MCP client would interact with the server.
    In practice, the MCP client would handle the communication protocol.
    """
    
    print("=== MCP Financial Server Usage Examples ===\n")
    
    # Example 1: Get historical prices for Apple
    print("1. Fetching 1 month of Apple (AAPL) stock history:")
    print("   Tool: get_stock_history")
    print("   Arguments: {\"symbol\": \"AAPL\", \"period\": \"1mo\", \"interval\": \"1d\"}")
    print("   Response would contain daily OHLC data for the past month\n")
    
    # Example 2: Get stock information
    print("2. Getting detailed info for Tesla (TSLA):")
    print("   Tool: get_stock_info")
    print("   Arguments: {\"symbol\": \"TSLA\"}")
    print("   Response would contain company details, metrics, and fundamentals\n")
    
    # Example 3: Get real-time quote
    print("3. Getting real-time quote for Microsoft (MSFT):")
    print("   Tool: get_stock_quote")
    print("   Arguments: {\"symbol\": \"MSFT\"}")
    print("   Response would contain current price and day's trading data\n")
    
    # Example 4: Get intraday data
    print("4. Fetching intraday data for Google (GOOGL):")
    print("   Tool: get_stock_history")
    print("   Arguments: {\"symbol\": \"GOOGL\", \"period\": \"1d\", \"interval\": \"5m\"}")
    print("   Response would contain 5-minute candles for today\n")
    
    # Example 5: Get long-term historical data
    print("5. Fetching 5 years of Amazon (AMZN) history:")
    print("   Tool: get_stock_history")
    print("   Arguments: {\"symbol\": \"AMZN\", \"period\": \"5y\", \"interval\": \"1wk\"}")
    print("   Response would contain weekly candles for the past 5 years\n")
    
    print("=== Configuration for MCP Clients ===\n")
    print("Add to your MCP settings:")
    print(json.dumps({
        "mcpServers": {
            "financial-server": {
                "command": "python",
                "args": ["-m", "mcp_financial_server"]
            }
        }
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(demo_usage())