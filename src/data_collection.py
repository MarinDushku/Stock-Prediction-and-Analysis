import yfinance as yf
import pandas as pd
import requests
from newsapi import NewsApiClient
import time

class DataCollector:
    def __init__(self, ticker='TSLA'):
        self.ticker = ticker
        # Replace with your actual NewsAPI key or leave empty
        self.newsapi_key = ""  # Add your API key here if you have one
        if self.newsapi_key:
            try:
                self.newsapi = NewsApiClient(api_key=self.newsapi_key)
            except Exception as e:
                print(f"Error initializing NewsAPI: {e}")
                self.newsapi = None
        else:
            self.newsapi = None
    
    def get_stock_data(self, period='1y'):
        """
        Fetch historical stock data for any ticker
        """
        try:
            stock = yf.Ticker(self.ticker)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"No data found for ticker: {self.ticker}")
                return None
            
            # Add additional features
            df['Daily_Return'] = df['Close'].pct_change()
            df['50_Day_MA'] = df['Close'].rolling(window=50).mean()
            df['200_Day_MA'] = df['Close'].rolling(window=200).mean()
            
            return df
        except Exception as e:
            print(f"Error fetching stock data for {self.ticker}: {e}")
            return None
    
    def get_recent_news(self, days=30):
        """
        Fetch recent news about the stock with fallback options
        """
        try:
            # First attempt: Try NewsAPI if key is available
            if self.newsapi:
                try:
                    company_info = self.get_company_info()
                    company_name = company_info.get('name', self.ticker)
                    
                    search_query = f"{company_name} OR {self.ticker}"
                    
                    articles = self.newsapi.get_everything(
                        q=search_query,
                        language='en',
                        sort_by='publishedAt',
                        page_size=10
                    )
                    
                    if 'articles' in articles and articles['articles']:
                        news_df = pd.DataFrame(articles['articles'])
                        news_df['published_date'] = pd.to_datetime(news_df['publishedAt'])
                        return news_df
                except Exception as e:
                    print(f"NewsAPI error: {e}")
            
            # Fallback: Use Yahoo Finance news
            stock = yf.Ticker(self.ticker)
            news = stock.news
            
            if news and len(news) > 0:
                news_df = pd.DataFrame(news)
                # Format the data
                if 'providerPublishTime' in news_df.columns:
                    news_df['published_date'] = pd.to_datetime(news_df['providerPublishTime'], unit='s')
                news_df['title'] = news_df.get('title', '')
                news_df['description'] = news_df.get('summary', '')
                return news_df
                
            # If still no news, return empty DataFrame with message
            print(f"No news found for {self.ticker}")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()
    
    def get_company_info(self):
        """
        Get detailed company information
        """
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            # Extract key company information
            company_info = {
                'name': info.get('longName', self.ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'US'),
                'employees': info.get('fullTimeEmployees', 'Unknown'),
                'website': info.get('website', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'description': info.get('longBusinessSummary', '')
            }
            
            return company_info
        except Exception as e:
            print(f"Error getting company info for {self.ticker}: {e}")
            return {
                'name': self.ticker,
                'sector': 'Unknown',
                'country': 'US'
            }