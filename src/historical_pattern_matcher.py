import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go

class HistoricalPatternMatcher:
    def __init__(self, stock_data, window_size=20, pattern_count=5):
        """
        Initialize pattern matcher with historical stock data
        
        Args:
            stock_data: DataFrame with OHLCV data and technical indicators
            window_size: Size of pattern window to compare (days)
            pattern_count: Number of top matching patterns to return
        """
        self.stock_data = stock_data
        self.window_size = window_size
        self.pattern_count = pattern_count
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
    
    def find_similar_patterns(self):
        """Find historical patterns similar to current market conditions"""
        # Extract close prices
        prices = self.stock_data['Close'].values
        
        # Get current window (most recent window_size days)
        current_window = prices[-self.window_size:]
        
        # Prepare to store pattern matches
        pattern_matches = []
        
        # Look through historical data for similar patterns
        # Exclude the most recent window which would be comparing to itself
        for i in range(len(prices) - 2*self.window_size):
            # Get historical window
            historical_window = prices[i:i+self.window_size]
            
            # Calculate similarity
            similarity = self._calculate_pattern_similarity(current_window, historical_window)
            
            # Store this pattern and its similarity
            pattern_matches.append({
                'start_idx': i,
                'end_idx': i + self.window_size - 1,
                'similarity': similarity,
                'start_date': self.stock_data.index[i],
                'end_date': self.stock_data.index[i + self.window_size - 1],
                'next_period_return': (prices[i + 2*self.window_size - 1] / prices[i + self.window_size - 1]) - 1 
                    if i + 2*self.window_size - 1 < len(prices) else None
            })
        
        # Sort by similarity (lowest distance = most similar)
        pattern_matches = sorted(pattern_matches, key=lambda x: x['similarity'])
        
        # Take top matches
        return pattern_matches[:self.pattern_count]
    
    def predict_from_patterns(self, forecast_days=7):
        """
        Make predictions based on what happened after similar historical patterns
        
        Returns:
            DataFrame with predictions and confidence intervals
        """
        # Find similar patterns
        similar_patterns = self.find_similar_patterns()
        
        # Create output dataframe
        last_date = self.stock_data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        predictions_df = pd.DataFrame(index=future_dates)
        
        # Get current price
        current_price = self.stock_data['Close'].iloc[-1]
        
        # Extract what happened next in each similar pattern
        pattern_forecasts = []
        
        for pattern in similar_patterns:
            start_idx = pattern['end_idx'] + 1
            end_idx = min(start_idx + forecast_days, len(self.stock_data))
            
            if end_idx - start_idx > 0:
                # Get price changes after the pattern
                price_changes = []
                
                for i in range(start_idx, end_idx):
                    pct_change = (self.stock_data['Close'].iloc[i] / self.stock_data['Close'].iloc[start_idx-1]) - 1
                    price_changes.append(pct_change)
                
                # Extend if we don't have enough days
                while len(price_changes) < forecast_days:
                    price_changes.append(price_changes[-1] if price_changes else 0)
                
                pattern_forecasts.append(price_changes)
        
        # Calculate mean prediction and confidence intervals
        if pattern_forecasts:
            predictions_array = np.array(pattern_forecasts)
            
            # Calculate predictions and bounds
            mean_changes = np.mean(predictions_array, axis=0)
            std_changes = np.std(predictions_array, axis=0)
            lower_changes = mean_changes - 1.96 * std_changes
            upper_changes = mean_changes + 1.96 * std_changes
            
            # Convert to prices
            predictions_df['Predicted_Price'] = current_price * (1 + mean_changes)
            predictions_df['Lower_Bound'] = current_price * (1 + lower_changes)
            predictions_df['Upper_Bound'] = current_price * (1 + upper_changes)
            predictions_df['Pattern_Count'] = len(pattern_forecasts)
            predictions_df['Confidence'] = 1.0 / (1.0 + std_changes)  # Higher when std is lower
        else:
            # No patterns found
            predictions_df['Predicted_Price'] = current_price
            predictions_df['Lower_Bound'] = current_price * 0.95
            predictions_df['Upper_Bound'] = current_price * 1.05
            predictions_df['Pattern_Count'] = 0
            predictions_df['Confidence'] = 0
            
        return predictions_df, similar_patterns
    
    def visualize_patterns(self, similar_patterns):
        """Create visualization of similar patterns and current pattern"""
        fig = go.Figure()
        
        # Add current pattern
        current_window = self.stock_data['Close'].values[-self.window_size:]
        normalized_current = self._normalize_window(current_window)
        
        fig.add_trace(go.Scatter(
            x=list(range(self.window_size)),
            y=normalized_current,
            mode='lines',
            name='Current Pattern',
            line=dict(color='blue', width=3)
        ))
        
        # Add historical patterns
        colors = ['red', 'green', 'purple', 'orange', 'brown']
        
        for i, pattern in enumerate(similar_patterns):
            historical_window = self.stock_data['Close'].values[pattern['start_idx']:pattern['end_idx']+1]
            normalized_hist = self._normalize_window(historical_window)
            
            fig.add_trace(go.Scatter(
                x=list(range(self.window_size)),
                y=normalized_hist,
                mode='lines',
                name=f"Pattern {i+1} ({pattern['start_date'].strftime('%Y-%m-%d')})",
                line=dict(color=colors[i % len(colors)], dash='dash')
            ))
        
        fig.update_layout(
            title="Current Market Pattern vs. Similar Historical Patterns",
            xaxis_title="Days in Pattern",
            yaxis_title="Normalized Price",
            legend_title="Patterns"
        )
        
        return fig