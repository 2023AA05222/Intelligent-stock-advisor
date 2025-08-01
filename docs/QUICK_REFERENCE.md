# Quick Reference Guide

## 🚀 Quick Start (30 seconds)
1. Enter stock symbol (e.g., `AAPL`) → Select period → Click **Fetch Data**
2. Go to **Relationships** tab → Click **"🔄 Index Company"** (optional, for graph features)
3. Go to **AI Chat** tab → Ask "Analyze this stock"
4. Explore other tabs for detailed data and relationship networks

---

## 📊 Essential Features

### Stock Symbol Input
```
✅ AAPL, MSFT, GOOGL (US stocks)
✅ NESN.SW, ASML.AS (International)
✅ SPY, QQQ (ETFs)
❌ Apple, Microsoft (Company names)
```

### Time Periods & Intervals
| Use Case | Period | Interval |
|----------|---------|----------|
| Day Trading | 1d-5d | 1m-15m |
| Swing Trading | 1mo-3mo | 1h-1d |
| Long-term Analysis | 1y-5y | 1d-1wk |

---

## 🤖 AI Chat Quick Commands

### Basic Analysis
- `"Analyze this stock"`
- `"What's the trend?"`
- `"Is it a good buy?"`

### Technical Analysis
- `"Show technical indicators"`
- `"What's the volatility?"`
- `"Risk assessment"`

### Financial Health
- `"Company financial health"`
- `"Revenue growth analysis"`
- `"Compare to competitors"`

### 🔗 Relationship Analysis (with Neo4j)
- `"What companies are connected to this one?"`
- `"Show me supply chain risks"`
- `"How might portfolio correlations affect my investments?"`
- `"Which companies would be impacted if this one has problems?"`

---

## 📈 Key Metrics Explained

### Price Metrics
- **Close**: Current/last trading price
- **Volume**: Shares traded (higher = more activity)
- **High/Low**: Daily price range

### Risk Metrics
- **Volatility**: Price swing magnitude (higher = riskier)
- **Sharpe Ratio**: >1 good, >2 very good
- **Max Drawdown**: Worst loss from peak

### Financial Ratios
- **P/E Ratio**: <15 cheap, >25 expensive (varies by sector)
- **Debt/Equity**: <0.5 conservative, >1.0 risky
- **ROE**: >15% excellent, <10% concerning

---

## 🔧 Troubleshooting Cheat Sheet

| Problem | Solution |
|---------|----------|
| "No data found" | Check ticker symbol spelling |
| Blank chart | Try longer time period |
| Slow performance | Use smaller date ranges |
| AI not responding | Ensure data is loaded first |
| Missing financials | Try larger companies (some small caps don't report) |

---

## 💡 Pro Tips

### Data Loading
- Start with **1 year + daily** for overview
- **5 years** for long-term trends
- **1 month + hourly** for day trading

### Chart Analysis
- **Green candles** = price up
- **Red candles** = price down
- **Long wicks** = volatility
- **High volume** = confirmation

### AI Usage
- Load data BEFORE asking questions
- Be specific: "3-month trend" vs "trend"
- Ask follow-up questions for deeper analysis
- **Index companies** in Relationships tab for enhanced AI insights

### 🔗 Graph Database Features
- **Index companies** to enable relationship analysis
- **Portfolio analysis** works best with 3-10 symbols
- **Relationship depth** of 2-3 hops provides good balance of detail vs performance
- **Neo4j connection** optional but enhances AI responses significantly

---

## 📋 Tab Functions

| Tab | Purpose | Best For |
|-----|---------|----------|
| **AI Chat** | Q&A analysis | Getting insights with relationship context |
| **Chart** | Price visualization | Technical analysis |
| **Data Table** | Raw numbers | Detailed inspection |
| **Statistics** | Calculated metrics | Risk assessment |
| **Statements** | Company financials | Fundamental analysis |
| **News** | Latest updates | Market context |
| **🔗 Relationships** | **Network analysis** | **Portfolio correlations, supply chain risks** |
| **Export** | Download data | External analysis |

---

## ⚠️ Important Reminders

### Investment Warnings
- 🚨 **Not financial advice** - do your own research
- 🚨 **Past performance ≠ future results**
- 🚨 **Markets are unpredictable** - use multiple sources
- 🚨 **Risk management** - never invest more than you can lose

### Data Limitations
- Real-time data during market hours only
- 5-minute cache - click "Fetch Data" for latest
- Some international stocks may have limited data
- Financial statements updated quarterly (1-3 month delay)

---

## 🎯 Common Use Cases

### Research Before Buying
1. Load 2-5 years of data
2. Check financial statements
3. Read recent news
4. Ask AI: "Is this a good long-term investment?"

### Day Trading Setup
1. Use 1-5 day period with 1-15 minute intervals
2. Monitor volume patterns
3. Ask AI: "What are the intraday levels to watch?"

### Portfolio Review
1. Load each stock with 1 year daily data
2. Check statistics tab for risk metrics
3. Ask AI: "How does this stock compare to the market?"

### Earnings Analysis
1. Load quarterly financial statements
2. Check recent news for earnings reports
3. Ask AI: "Analyze the latest earnings results"

---

## 📞 Quick Help

### Can't Find Feature?
- **Sidebar**: Stock input, period selection
- **Tabs**: Different analysis views
- **AI Chat**: Natural language queries
- **Export**: Download buttons in Export tab

### Performance Issues?
- Close other browser tabs
- Use shorter time periods
- Clear browser cache
- Check internet connection

---

## 📚 Learning Path

### Beginner (Week 1)
1. Learn to load basic stock data
2. Understand candlestick charts
3. Use AI for simple questions
4. Read financial statements basics

### Intermediate (Week 2-4)
1. Compare multiple stocks
2. Understand technical indicators
3. Analyze different time frames
4. Export and track analysis

### Advanced (Month 2+)
1. Sector and market analysis
2. Risk assessment techniques
3. News impact evaluation
4. Custom analysis workflows

---

**Print this page** for offline reference!

*Last updated: June 2025*