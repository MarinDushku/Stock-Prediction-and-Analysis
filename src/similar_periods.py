import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go

class SimilarPeriodAnalyzer:
    def __init__(self, stock_data, window_size=60):
        """
        Initialize period analyzer with historical stock data
        
        Args:
            stock_data: DataFrame with OHLCV data and technical indicators
            window_size: Size of period window to compare (days)
        """
        self.stock_data = stock_data
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def _normalize_window(self, window):
        """Normalize values in a window to range [0,1]"""
        return self.scaler.fit_transform(window.reshape(-1, 1)).flatten()
    
    def _calculate_pattern_similarity(self, current_window, historical_window):
        """
        Calculate similarity between current pattern and a historical pattern
        using dynamic time warping or euclidean distance
        """
        # Normalize both windows for fair comparison
        norm_current = self._normalize_window(current_window)
        norm_historical = self._normalize_window(historical_window)
        
        # Calculate Euclidean distance (lower is more similar)
        similarity = euclidean(norm_current, norm_historical)
        
        return similarity
    
    def find_similar_periods(self, top_n=3):
        """Find historical periods similar to current market conditions"""
        # Extract features for comparison
        features = self._extract_features()
        
        if len(features) < 2 * self.window_size:
            return []
        
        # Get current window (most recent window_size days)
        current_idx = len(features) - self.window_size
        current_window = features.iloc[current_idx:].values
        
        # Normalize current window
        current_window_norm = self._normalize_window(current_window)
        
        # Find similar windows in history
        similarities = []
        
        # Only look at windows that ended at least window_size days before today
        max_idx = len(features) - 2 * self.window_size
        
        for i in range(max_idx):
            historical_window = features.iloc[i:i+self.window_size].values
            historical_window_norm = self._normalize_window(historical_window)
            
            # Calculate similarity
            similarity = self._calculate_similarity(current_window_norm, historical_window_norm)
            
            # Store this period
            similarities.append({
                'start_idx': i,
                'end_idx': i + self.window_size - 1,
                'similarity': similarity,
                'start_date': features.index[i],
                'end_date': features.index[i + self.window_size - 1],
                'next_period_return': (features['Close'].iloc[i + 2*self.window_size - 1] / 
                                    features['Close'].iloc[i + self.window_size - 1]) - 1 
                    if i + 2*self.window_size - 1 < len(features) else None
            })
        
        # Sort by similarity (lower = more similar)
        similarities = sorted(similarities, key=lambda x: x['similarity'])
        
        # Take top n most similar
        top_similar = similarities[:top_n]
        
        # Calculate what happened next in similar periods
        for period in top_similar:
            start_idx = period['end_idx'] + 1
            
            # Calculate returns for 7, 30, and 90 days after similar period
            if start_idx + 7 < len(self.stock_data):
                period['next_7d_return'] = (
                    self.stock_data['Close'].iloc[start_idx + 7] / 
                    self.stock_data['Close'].iloc[start_idx] - 1
                )
            
            if start_idx + 30 < len(self.stock_data):
                period['next_30d_return'] = (
                    self.stock_data['Close'].iloc[start_idx + 30] / 
                    self.stock_data['Close'].iloc[start_idx] - 1
                )
            
            if start_idx + 90 < len(self.stock_data):
                period['next_90d_return'] = (
                    self.stock_data['Close'].iloc[start_idx + 90] / 
                    self.stock_data['Close'].iloc[start_idx] - 1
                )
            
            # Calculate max drawdown in the next 30 days
            max_drawdown = 0
            if start_idx + 30 < len(self.stock_data):
                prices = self.stock_data['Close'].iloc[start_idx:start_idx+31]
                peak = prices[0]
                for price in prices:
                    drawdown = (peak - price) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    if price > peak:
                        peak = price
                
                period['next_30d_max_drawdown'] = max_drawdown
        
        return top_similar
    
    def _extract_features(self):
        """Extract relevant features for period comparison"""
        df = self.stock_data.copy()
        
        # Calculate returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std().fillna(0)
        
        # Calculate Moving Averages and MA ratios
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['MA_Ratio'] = df['MA50'] / df['MA200']
        
        # Relative Strength Index
        delta = df['Close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = abs(loss.rolling(window=14).mean())
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume Trends
        if 'Volume' in df.columns:
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_Trend'] = df['Volume'].rolling(window=20).mean().pct_change(20)
        else:
            df['Volume_Change'] = 0
            df['Volume_Trend'] = 0
        
        # Fill NAs
        df = df.fillna(method='bfill').fillna(0)
        
        # Select features for comparison
        features = df[['Close', 'Returns', 'Volatility', 'MA_Ratio', 'RSI', 'Volume_Change', 'Volume_Trend']]
        
        return features
    
    def _calculate_similarity(self, window1, window2):
        """Calculate similarity between two time windows"""
        # Use euclidean distance
        return euclidean(window1, window2)
    
    def predict_based_on_similar_periods(self, forecast_days=30):
        """
        Make predictions based on what happened in similar historical periods
        
        Returns:
            DataFrame with predictions and confidence intervals
        """
        # Find similar periods
        similar_periods = self.find_similar_periods(top_n=5)
        
        if not similar_periods:
            return None, []
            
        # Create forecast dataframe
        last_date = self.stock_data.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        forecast_df = pd.DataFrame(index=forecast_dates)
        
        # Current price
        current_price = self.stock_data['Close'].iloc[-1]
        
        # Collect returns from similar periods
        similar_returns = []
        for period in similar_periods:
            start_idx = period['end_idx'] + 1
            
            if start_idx + forecast_days < len(self.stock_data):
                # Get actual returns for each day in the forecast period
                period_returns = []
                for i in range(forecast_days):
                    day_return = (
                        self.stock_data['Close'].iloc[start_idx + i] / 
                        self.stock_data['Close'].iloc[start_idx - 1] - 1
                    )
                    period_returns.append(day_return)
                
                similar_returns.append(period_returns)
        
        if not similar_returns:
            return None, similar_periods
            
        # Calculate mean and std of returns
        similar_returns_array = np.array(similar_returns)
        mean_returns = np.mean(similar_returns_array, axis=0)
        std_returns = np.std(similar_returns_array, axis=0)
        
        # Calculate forecast prices
        forecast_df['Predicted_Price'] = current_price * (1 + mean_returns)
        forecast_df['Upper_Bound'] = current_price * (1 + mean_returns + 1.96 * std_returns)
        forecast_df['Lower_Bound'] = current_price * (1 + mean_returns - 1.96 * std_returns)
        
        # Ensure lower bound is not negative
        forecast_df['Lower_Bound'] = forecast_df['Lower_Bound'].clip(lower=0)
        
        # Calculate confidence score
        confidence = 1.0 / (1.0 + std_returns)
        forecast_df['Confidence'] = np.minimum(confidence, 1.0)
        
        return forecast_df, similar_periods