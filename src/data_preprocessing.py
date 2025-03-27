import pandas as pd
import numpy as np
import streamlit as st

def clean_stock_data(df):
    """
    Clean and prepare stock data for analysis
    
    Parameters:
    - df: Input DataFrame from yfinance
    
    Returns:
    - Cleaned DataFrame
    """
    # Check for missing values
    if df is None:
        st.error("No data to preprocess")
        return None
    
    # Fill missing values
    df = df.fillna(method='ffill')  # Forward fill missing values
    
    # Remove any remaining NaN rows
    df = df.dropna()
    
    return df

def calculate_technical_indicators(df):
    """
    Calculate additional technical indicators
    
    Parameters:
    - df: Cleaned stock DataFrame
    
    Returns:
    - DataFrame with additional technical indicators
    """
    if df is None:
        st.error("No data to calculate technical indicators")
        return None
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band'] = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def normalize_data(df):
    """
    Normalize data for comparison and machine learning
    
    Parameters:
    - df: Input DataFrame
    
    Returns:
    - Normalized DataFrame
    """
    if df is None:
        st.error("No data to normalize")
        return None
    
    # Select numeric columns for normalization
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Normalize using Min-Max scaling
    normalized_df = df.copy()
    for col in numeric_columns:
        normalized_df[f'{col}_Normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    return normalized_df

# Example usage (can be commented out later)
if __name__ == "__main__":
    from data_acquisition import fetch_stock_data
    
    ticker = 'AAPL'
    raw_data = fetch_stock_data(ticker)
    
    # Apply preprocessing steps
    cleaned_data = clean_stock_data(raw_data)
    technical_data = calculate_technical_indicators(cleaned_data)
    normalized_data = normalize_data(technical_data)
    
    print(normalized_data.tail())