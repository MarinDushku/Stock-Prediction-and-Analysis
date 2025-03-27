import yfinance as yf
import pandas as pd
import streamlit as st

def fetch_stock_data(ticker, period='1y'):
    """
    Fetch stock data for a given ticker
    
    Parameters:
    - ticker: Stock symbol (e.g., 'AAPL')
    - period: Time period for historical data (default: 1 year)
    
    Returns:
    - Pandas DataFrame with stock data
    """
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        
        # Get historical market data
        df = stock.history(period=period)
        
        # Add additional useful columns
        df['Daily_Return'] = df['Close'].pct_change() * 100
        
        # Calculate moving averages
        df['50_Day_MA'] = df['Close'].rolling(window=50).mean()
        df['200_Day_MA'] = df['Close'].rolling(window=200).mean()
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def get_stock_info(ticker):
    """
    Fetch additional stock information
    
    Parameters:
    - ticker: Stock symbol
    
    Returns:
    - Dictionary with stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key information
        stock_info = {
            'Company Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Current Price': info.get('currentPrice', 'N/A')
        }
        
        return stock_info
    
    except Exception as e:
        st.error(f"Error fetching stock info for {ticker}: {e}")
        return None

# Example usage (you can comment this out later)
if __name__ == "__main__":
    # Test the functions
    ticker = 'AAPL'
    data = fetch_stock_data(ticker)
    info = get_stock_info(ticker)
    
    print(f"Stock Data for {ticker}:")
    print(data.tail())
    print("\nStock Information:")
    print(info)