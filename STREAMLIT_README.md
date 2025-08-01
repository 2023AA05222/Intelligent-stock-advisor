# Stock Historical Price Data Viewer

A Streamlit web application for fetching and visualizing historical stock price data using Yahoo Finance.

## Features

- **Real-time Stock Data**: Fetch historical price data for any stock symbol
- **Interactive Charts**: Candlestick charts with volume overlay
- **Customizable Time Periods**: From 1 day to maximum available data
- **Multiple Intervals**: From 1-minute to 3-month intervals
- **Statistical Analysis**: View price statistics and distributions
- **Data Export**: Download data in JSON or CSV format

## Installation

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Install dependencies (if not already installed):
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Enter Stock Symbol**: Type a stock ticker symbol (e.g., AAPL, MSFT, GOOGL) in the sidebar
2. **Select Time Period**: Choose how far back you want to fetch data
3. **Select Interval**: Choose the granularity of the data (minutes, days, weeks, etc.)
4. **Click "Fetch Data"**: The app will retrieve and display the data

## Features Overview

### 📊 Chart Tab
- Interactive candlestick chart showing OHLC (Open, High, Low, Close) data
- Volume bars overlaid on the chart
- Zoom and pan functionality

### 📋 Data Table Tab
- Full historical data in tabular format
- Includes price change and percentage change calculations
- Sortable and searchable

### 📈 Statistics Tab
- Key metrics: current price, period high/low, returns
- Volatility measurements
- Price distribution histogram

### 💾 Export Tab
- Download data as JSON with structured format
- Download data as CSV for use in Excel or other tools
- Preview exported data format

## Supported Time Periods
- 1 Day, 5 Days
- 1 Month, 3 Months, 6 Months
- 1 Year, 2 Years, 5 Years, 10 Years
- Year to Date
- Maximum available data

## Supported Intervals
- Intraday: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
- Daily and above: 1d, 5d, 1wk, 1mo, 3mo

Note: Intraday intervals are only available for recent time periods (1d, 5d).