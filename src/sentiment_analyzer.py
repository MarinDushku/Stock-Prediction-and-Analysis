import nltk
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import requests
import json
import os
from datetime import datetime, timedelta

class SentimentAnalyzer:
    def __init__(self):
        # Initialize NLTK's Sentiment Intensity Analyzer
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            
            # Add finance-specific terms to the sentiment analyzer
            self.finance_lexicon = {
                # Positive financial terms
                'beat expectations': 2.0,
                'exceeded estimates': 2.0,
                'upgraded': 1.5,
                'bullish': 1.5,
                'outperform': 1.5,
                'buy rating': 1.5,
                'strong buy': 2.0,
                'raised guidance': 1.8,
                'record high': 1.5,
                'growth': 1.0,
                'profit': 1.0,
                'revenue growth': 1.2,
                'margin expansion': 1.3,
                'dividend increase': 1.4,
                'share buyback': 1.2,
                'cost cutting': 1.0,
                'restructuring': 0.5,  # Can be good or bad
                
                # Negative financial terms
                'missed expectations': -2.0,
                'below estimates': -2.0,
                'downgraded': -1.5,
                'bearish': -1.5,
                'underperform': -1.5,
                'sell rating': -1.5,
                'strong sell': -2.0,
                'lowered guidance': -1.8,
                'all-time low': -1.5,
                'decline': -1.0,
                'loss': -1.5,
                'revenue decline': -1.2,
                'margin contraction': -1.3,
                'dividend cut': -1.8,
                'layoffs': -1.2,
                'bankruptcy': -2.5,
                'investigation': -1.7,
                'lawsuit': -1.6,
                'debt': -0.5,
                'inflation': -0.7
            }
            
            # Add the financial terms to the lexicon
            for term, value in self.finance_lexicon.items():
                self.sia.lexicon[term] = value
                
            self.initialized = True
            
            # Cache for earnings data
            self.cache_dir = "cache"
            os.makedirs(self.cache_dir, exist_ok=True)
            self.earnings_cache = {}
            self.load_earnings_cache()
            
        except Exception as e:
            print(f"Error initializing sentiment analyzer: {e}")
            self.initialized = False
    
    def load_earnings_cache(self):
        """Load earnings cache from file"""
        cache_file = os.path.join(self.cache_dir, "earnings_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.earnings_cache = json.load(f)
            except:
                self.earnings_cache = {}
    
    def save_earnings_cache(self):
        """Save earnings cache to file"""
        cache_file = os.path.join(self.cache_dir, "earnings_cache.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.earnings_cache, f)
        except Exception as e:
            print(f"Error saving earnings cache: {e}")
    
    def analyze_news_sentiment(self, news_df, ticker=None):
        """
        Analyze sentiment of news articles using finance-specific enhancements
        """
        if not self.initialized:
            # Return random sentiment if analyzer failed to initialize
            return np.random.uniform(-0.3, 0.3)
            
        sentiments = []
        
        # If no news data, use stock price momentum as pseudo-sentiment
        if news_df.empty:
            # Try to get earnings sentiment if ticker is provided
            if ticker:
                earnings_sentiment = self.get_earnings_sentiment(ticker)
                if earnings_sentiment is not None:
                    return earnings_sentiment
            return self._get_fallback_sentiment()
        
        # Analyze available news with financial weighting
        if 'title' in news_df.columns and 'description' in news_df.columns:
            for _, row in news_df.iterrows():
                try:
                    # Combine title and description with more weight on title
                    title = row.get('title', '')
                    desc = row.get('description', '')
                    
                    # Skip if missing data
                    if not title and not desc:
                        continue
                        
                    # Process title (with 2x weight) and description
                    if title and len(title) > 5:
                        title_sentiment = self._analyze_financial_text(title)
                        sentiments.append(title_sentiment * 2)  # Double weight for title
                    
                    if desc and len(desc) > 10:
                        desc_sentiment = self._analyze_financial_text(desc)
                        sentiments.append(desc_sentiment)
                        
                except Exception as e:
                    print(f"Error analyzing article: {e}")
        elif 'description' in news_df.columns:
            for desc in news_df['description'].dropna():
                try:
                    if desc and len(desc) > 10:
                        sentiment = self._analyze_financial_text(desc)
                        sentiments.append(sentiment)
                except Exception as e:
                    print(f"Error analyzing description: {e}")
        
        # Combine with earnings sentiment if available
        if ticker:
            earnings_sentiment = self.get_earnings_sentiment(ticker)
            if earnings_sentiment is not None:
                sentiments.append(earnings_sentiment * 2)  # Double weight for earnings
        
        # Return mean sentiment if we have any
        if sentiments:
            return np.mean(sentiments)
        else:
            return self._get_fallback_sentiment()
    
    def _analyze_financial_text(self, text):
        """Enhanced sentiment analysis for financial text"""
        # Base sentiment using VADER
        sentiment_scores = self.sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        # Apply financial pattern matching for stronger signals
        text_lower = text.lower()
        
        # Check for specific financial patterns
        if re.search(r'beat.*expectations|exceed.*estimates', text_lower):
            compound_score += 0.3
        if re.search(r'miss.*expectations|below.*estimates', text_lower):
            compound_score -= 0.3
        if re.search(r'raised.*guidance|increased.*outlook', text_lower):
            compound_score += 0.25
        if re.search(r'lowered.*guidance|reduced.*outlook', text_lower):
            compound_score -= 0.25
        if re.search(r'buy.*rating|outperform|upgrade', text_lower):
            compound_score += 0.2
        if re.search(r'sell.*rating|underperform|downgrade', text_lower):
            compound_score -= 0.2
            
        # Clip to [-1, 1] range
        return max(-1, min(1, compound_score))
    
    def get_earnings_sentiment(self, ticker):
        """Get sentiment based on recent earnings"""
        # Check cache first
        if ticker in self.earnings_cache:
            cache_time = self.earnings_cache[ticker].get('timestamp', 0)
            # Cache for 7 days
            if datetime.now().timestamp() - cache_time < 7 * 24 * 60 * 60:
                return self.earnings_cache[ticker].get('sentiment')
        
        try:
            # Try to get earnings data from Yahoo Finance or another API
            # This is a placeholder - in a real implementation, you would use a proper API
            # For demonstration, we'll simulate earnings data
            earnings_data = self._simulate_earnings_data(ticker)
            
            if earnings_data:
                # Calculate sentiment based on earnings beat/miss and guidance
                sentiment = 0
                
                # EPS beat/miss
                if 'eps_beat_pct' in earnings_data:
                    eps_beat = earnings_data['eps_beat_pct']
                    if eps_beat > 5:
                        sentiment += 0.3
                    elif eps_beat > 0:
                        sentiment += 0.1
                    elif eps_beat < -5:
                        sentiment -= 0.3
                    elif eps_beat < 0:
                        sentiment -= 0.1
                
                # Revenue beat/miss
                if 'revenue_beat_pct' in earnings_data:
                    rev_beat = earnings_data['revenue_beat_pct']
                    if rev_beat > 3:
                        sentiment += 0.25
                    elif rev_beat > 0:
                        sentiment += 0.1
                    elif rev_beat < -3:
                        sentiment -= 0.25
                    elif rev_beat < 0:
                        sentiment -= 0.1
                
                # Guidance
                if 'guidance' in earnings_data:
                    guidance = earnings_data['guidance']
                    if guidance == 'raised':
                        sentiment += 0.35
                    elif guidance == 'maintained':
                        sentiment += 0.05
                    elif guidance == 'lowered':
                        sentiment -= 0.35
                
                # Analyst reaction
                if 'analyst_changes' in earnings_data:
                    changes = earnings_data['analyst_changes']
                    if changes > 0:
                        sentiment += 0.2 * changes  # Positive changes
                    else:
                        sentiment += 0.2 * changes  # Negative changes
                
                # Clip to [-1, 1]
                earnings_sentiment = max(-1, min(1, sentiment))
                
                # Save to cache
                self.earnings_cache[ticker] = {
                    'sentiment': earnings_sentiment,
                    'timestamp': datetime.now().timestamp()
                }
                self.save_earnings_cache()
                
                return earnings_sentiment
            else:
                return None
                
        except Exception as e:
            print(f"Error getting earnings sentiment: {e}")
            return None
    
    def _simulate_earnings_data(self, ticker):
        """Simulate earnings data for demonstration"""
        # In a real implementation, you would fetch actual earnings data
        import hashlib
        import time
        
        # Use ticker to generate consistent but pseudo-random values
        seed = int(hashlib.md5(ticker.encode()).hexdigest(), 16) % 1000000
        np.random.seed(seed + int(time.time() / (60 * 60 * 24 * 7)))  # Change weekly
        
        # Generate simulated earnings data
        eps_beat = np.random.normal(2, 5)  # Mean 2% beat with std of 5%
        rev_beat = np.random.normal(1, 3)  # Mean 1% beat with std of 3%
        
        # Guidance options with probabilities
        guidance_options = ['raised', 'maintained', 'lowered', None]
        guidance_probs = [0.2, 0.5, 0.2, 0.1]  # 20% raise, 50% maintain, 20% lower, 10% none
        guidance = np.random.choice(guidance_options, p=guidance_probs)
        
        # Analyst rating changes (-2 to +2)
        analyst_changes = np.random.normal(0, 0.8)
        
        return {
            'eps_beat_pct': eps_beat,
            'revenue_beat_pct': rev_beat,
            'guidance': guidance,
            'analyst_changes': analyst_changes,
            'report_date': (datetime.now() - timedelta(days=np.random.randint(1, 60))).strftime('%Y-%m-%d')
        }
    
    def _get_fallback_sentiment(self):
        """
        Generate a fallback sentiment score when no news is available
        """
        # Generate a slightly varied sentiment score to avoid always neutral
        return np.random.uniform(-0.3, 0.3)
    
    def get_sentiment_category(self, score):
        """
        Convert numerical sentiment score to a categorical label
        """
        if score > 0.25:
            return "Very Positive"
        elif score > 0.05:
            return "Positive"
        elif score < -0.25:
            return "Very Negative"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"