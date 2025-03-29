import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, stock_data, sentiment_score, economic_score):
        self.stock_data = stock_data
        self.sentiment_score = sentiment_score
        self.economic_score = economic_score
    
    def prepare_features(self):
        """
        Prepare features for machine learning
        """
        df = self.stock_data.copy()
        
        # Feature engineering
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Diff'] = df['Close'] - df['Prev_Close']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Handle cases where we might not have enough data for volatility
        if len(df) >= 5:
            df['Return_Volatility'] = df['Daily_Return'].rolling(window=min(5, len(df)-1)).std()
        else:
            df['Return_Volatility'] = 0
            
        df['Sentiment_Score'] = self.sentiment_score
        df['Economic_Score'] = self.economic_score
        
        # Drop rows with NaN
        df.dropna(inplace=True)
        
        return df
    
    def train_model(self):
        """
        Train a Random Forest Regressor
        """
        df = self.prepare_features()
        
        if len(df) < 10:
            raise ValueError("Not enough data to train a model")
        
        # Select features
        features = ['Prev_Close', '50_Day_MA', '200_Day_MA', 
                   'Volume_Change', 'Return_Volatility',
                   'Sentiment_Score', 'Economic_Score']
        
        # Make sure all features are available
        for feature in features:
            if feature not in df.columns:
                if feature in ['50_Day_MA', '200_Day_MA']:
                    # For short histories, we might not have these features
                    df[feature] = df['Close']
                else:
                    df[feature] = 0
        
        target = 'Close'
        
        X = df[features]
        y = df[target]
        
        # Split data
        if len(df) < 30:
            # For small datasets, use a simple split
            train_size = max(int(len(df) * 0.8), 2)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        else:
            # For larger datasets, use sklearn's function
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        if len(X_test) > 0:
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            print(f"Model Root Mean Squared Error: {rmse}")
        
        return model
    
    def predict_future(self, model, days=7):
        """
        Predict stock prices for the next X days with dynamic updating
        """
        # Get the latest data
        df = self.prepare_features()
        last_date = df.index[-1]
        
        # Features for prediction
        features = ['Prev_Close', '50_Day_MA', '200_Day_MA', 
                   'Volume_Change', 'Return_Volatility',
                   'Sentiment_Score', 'Economic_Score']
        
        # Create a DataFrame for future predictions
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        future_df = pd.DataFrame(index=future_dates)
        future_df['Predicted_Price'] = 0.0
        
        # Current values
        last_close = df['Close'].iloc[-1]
        ma50 = df['50_Day_MA'].iloc[-1] if '50_Day_MA' in df.columns else last_close
        ma200 = df['200_Day_MA'].iloc[-1] if '200_Day_MA' in df.columns else last_close
        
        # Make predictions for each future day
        prev_close = last_close
        predicted_prices = []
        
        for i, date in enumerate(future_dates):
            # Update moving averages based on new predictions
            if i > 0:
                # Update 50-day MA with new predictions
                ma50 = (ma50 * 49 + prev_close) / 50
                # Update 200-day MA with new predictions
                ma200 = (ma200 * 199 + prev_close) / 200
            
            # Random noise to avoid flat predictions (small random component)
            volatility = df['Close'].pct_change().std() if len(df) > 10 else 0.01
            noise = np.random.normal(0, volatility * last_close * 0.2)
            
            # Create feature vector
            features_dict = {
                'Prev_Close': prev_close,
                '50_Day_MA': ma50,
                '200_Day_MA': ma200,
                'Volume_Change': df['Volume_Change'].mean() if 'Volume_Change' in df.columns else 0,
                'Return_Volatility': df['Return_Volatility'].mean() if 'Return_Volatility' in df.columns else 0.01,
                'Sentiment_Score': self.sentiment_score,
                'Economic_Score': self.economic_score
            }
            
            # Convert to DataFrame
            X_pred = pd.DataFrame([features_dict])
            
            # Predict with some noise to avoid flat line
            prediction = model.predict(X_pred)[0] + noise
            predicted_prices.append(prediction)
            
            # Store prediction
            future_df.loc[date, 'Predicted_Price'] = prediction
            
            # Update prev_close for next iteration
            prev_close = prediction
        
        # Calculate uncertainty (growing with time)
        price_std = df['Close'].std() if len(df) > 5 else last_close * 0.05
        
        for i, date in enumerate(future_dates):
            # Uncertainty grows with prediction distance
            uncertainty = price_std * (1 + i * 0.15)
            future_df.loc[date, 'Upper_Bound'] = future_df.loc[date, 'Predicted_Price'] + uncertainty
            future_df.loc[date, 'Lower_Bound'] = future_df.loc[date, 'Predicted_Price'] - uncertainty
            # Ensure lower bound is not negative
            future_df.loc[date, 'Lower_Bound'] = max(0, future_df.loc[date, 'Lower_Bound'])
        
        return future_df