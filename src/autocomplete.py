import pandas as pd
import requests
import streamlit as st
import json
import os
from datetime import datetime, timedelta

# Cache the stock list for better performance
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_list():
    """
    Fetch a list of common stock tickers and company names
    """
    # Check if we have a cached file that's recent
    cache_file = 'stock_list_cache.json'
    if os.path.exists(cache_file):
        modified_time = os.path.getmtime(cache_file)
        if datetime.fromtimestamp(modified_time) > datetime.now() - timedelta(days=7):
            # Use cache if less than 7 days old
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass  # If loading fails, continue to fetch new data
    
    try:
        # First attempt: Yahoo Finance API for common stocks
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.json"
        response = requests.get(url)
        if response.status_code == 200:
            stocks = response.json()
            # Format the data
            stock_list = []
            for stock in stocks:
                if 'symbol' in stock and 'name' in stock:
                    stock_list.append({
                        'symbol': stock['symbol'],
                        'name': stock['name']
                    })
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(stock_list, f)
                
            return stock_list
    except Exception as e:
        print(f"Error fetching stock list from primary source: {e}")
    
    try:
        # Fallback: Use a hardcoded list of popular stocks
        popular_stocks = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "BAC", "name": "Bank of America Corporation"},
            {"symbol": "WMT", "name": "Walmart Inc."},
            {"symbol": "DIS", "name": "The Walt Disney Company"},
            {"symbol": "NFLX", "name": "Netflix Inc."},
            {"symbol": "INTC", "name": "Intel Corporation"},
            {"symbol": "AMD", "name": "Advanced Micro Devices Inc."},
            {"symbol": "CSCO", "name": "Cisco Systems Inc."},
            {"symbol": "IBM", "name": "International Business Machines"},
            {"symbol": "GOOG", "name": "Alphabet Inc."},
            {"symbol": "ADBE", "name": "Adobe Inc."},
            {"symbol": "ORCL", "name": "Oracle Corporation"},
            {"symbol": "UBER", "name": "Uber Technologies Inc."},
            {"symbol": "PYPL", "name": "PayPal Holdings Inc."},
            {"symbol": "QCOM", "name": "Qualcomm Inc."},
            {"symbol": "T", "name": "AT&T Inc."},
            {"symbol": "VZ", "name": "Verizon Communications Inc."},
            {"symbol": "CMCSA", "name": "Comcast Corporation"},
            {"symbol": "KO", "name": "The Coca-Cola Company"},
            {"symbol": "PEP", "name": "PepsiCo Inc."},
            {"symbol": "MCD", "name": "McDonald's Corporation"},
            {"symbol": "ABT", "name": "Abbott Laboratories"},
            {"symbol": "JNJ", "name": "Johnson & Johnson"},
            {"symbol": "PFE", "name": "Pfizer Inc."},
            {"symbol": "NKE", "name": "Nike Inc."},
            {"symbol": "HD", "name": "The Home Depot Inc."},
            {"symbol": "V", "name": "Visa Inc."},
            {"symbol": "MA", "name": "Mastercard Incorporated"},
            {"symbol": "SBUX", "name": "Starbucks Corporation"},
            {"symbol": "GM", "name": "General Motors Company"},
            {"symbol": "F", "name": "Ford Motor Company"},
            {"symbol": "GE", "name": "General Electric Company"},
            {"symbol": "BA", "name": "The Boeing Company"},
            {"symbol": "CVX", "name": "Chevron Corporation"},
            {"symbol": "XOM", "name": "Exxon Mobil Corporation"},
            {"symbol": "GS", "name": "The Goldman Sachs Group Inc."},
            {"symbol": "MS", "name": "Morgan Stanley"},
            {"symbol": "TGT", "name": "Target Corporation"},
            {"symbol": "COST", "name": "Costco Wholesale Corporation"},
            {"symbol": "AMC", "name": "AMC Entertainment Holdings Inc."},
            {"symbol": "GME", "name": "GameStop Corp."},
            {"symbol": "COIN", "name": "Coinbase Global Inc."}
        ]
        
        # Cache the fallback result
        with open(cache_file, 'w') as f:
            json.dump(popular_stocks, f)
            
        return popular_stocks
        
    except Exception as e:
        print(f"Error with fallback stock list: {e}")
        # Return a minimal list if all else fails
        return [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."}
        ]

def search_stocks(query):
    """
    Search for stocks matching a query string
    """
    stocks = get_stock_list()
    query = query.upper()
    
    # Search for matches in both symbol and name
    matches = []
    
    # First, look for exact symbol match (highest priority)
    for stock in stocks:
        if stock['symbol'] == query:
            matches.append(stock)
    
    # Then look for symbols starting with the query
    if len(matches) < 10:
        for stock in stocks:
            if stock['symbol'].startswith(query) and stock not in matches:
                matches.append(stock)
                if len(matches) >= 10:
                    break
    
    # Then look for company names containing the query
    if len(matches) < 10:
        for stock in stocks:
            if query in stock['name'].upper() and stock not in matches:
                matches.append(stock)
                if len(matches) >= 10:
                    break
    
    return matches[:10]  # Return at most 10 matches