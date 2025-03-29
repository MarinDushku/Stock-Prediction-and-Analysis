from data_collection import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from economic_analyzer import EconomicAnalyzer
from stock_predictor import StockPredictor
from advanced_prediction import AdvancedPredictionEngine
from autocomplete import search_stocks
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add import for the new pattern matcher
try:
    from historical_pattern_matcher import HistoricalPatternMatcher
    PATTERN_MATCHER_AVAILABLE = True
except ImportError:
    PATTERN_MATCHER_AVAILABLE = False
    print("Historical Pattern Matcher not available")

try:
    from similar_periods import SimilarPeriodAnalyzer
    SIMILAR_PERIODS_AVAILABLE = True
except ImportError:
    SIMILAR_PERIODS_AVAILABLE = False
    print("Similar Period Analyzer not available")
    # Create the class (will be used in the main code)
    class SimilarPeriodAnalyzer:
        def __init__(self, stock_data, window_size=60):
            self.stock_data = stock_data
            self.window_size = window_size
            
        def predict_based_on_similar_periods(self, forecast_days=30):
            return None, []

def get_momentum_sentiment(stock_data):
    """
    Calculate a sentiment score based on recent price momentum
    """
    if stock_data is None or len(stock_data) < 5:
        return 0
    
    # Calculate short-term momentum (5 days)
    recent_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-5] - 1)
    
    # Scale to a sentiment-like score (-1 to 1)
    momentum_score = max(-1, min(1, recent_return * 10))
    
    return momentum_score

def main():
    st.title("Advanced Stock Price Predictor")
    
    # Add stock ticker input with autocomplete
    ticker_input = st.text_input("Enter Stock Ticker Symbol or Company Name", value="TSLA")
    
    if ticker_input:
        # Search for matching stocks
        matching_stocks = search_stocks(ticker_input)
        
        if matching_stocks:
            # Create a selection box with matching stocks
            stock_options = [f"{stock['symbol']} - {stock['name']}" for stock in matching_stocks]
            selected_stock = st.selectbox("Select a stock", stock_options)
            
            # Extract ticker from selection
            ticker = selected_stock.split(" - ")[0] if selected_stock else ticker_input.upper()
        else:
            # No matches found, use the input as is
            ticker = ticker_input.upper()
            st.info(f"No matching stocks found for '{ticker_input}'. Using '{ticker}' as the ticker symbol.")
    else:
        st.warning("Please enter a valid stock ticker or company name.")
        return
    
    # Add customization options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            prediction_days = st.slider("Prediction Horizon (Days)", min_value=1, max_value=30, value=7)
            pattern_window = st.slider("Pattern Window Size (Days)", min_value=5, max_value=60, value=20)
        with col2:
            sentiment_weight = st.slider("Sentiment Analysis Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            technical_weight = st.slider("Technical Analysis Weight", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            
        # Add market regime override
        regime_options = ["Auto-detect", "Stable Bullish", "Volatile Bullish", "Range Bound", 
                          "Volatile Bearish", "Stable Bearish", "Rising Rate Regime", 
                          "Inflation Regime", "Growth Slowdown", "Recovery"]
        market_regime = st.selectbox("Market Regime", regime_options, index=0)
        regime_override = None if market_regime == "Auto-detect" else market_regime
    
    # Add prediction mode selection
    prediction_mode = st.radio(
        "Prediction Mode",
        ["Standard", "Advanced ML", "Pattern Matching", "Similar Periods", "Ensemble (All Methods)"],
        index=4
    )
    
    # Add a button to run the analysis
    if st.button(f"Analyze {ticker}"):
        try:
            st.subheader(f"Analyzing {ticker}...")
            
            # Collect data
            with st.spinner("Fetching stock data..."):
                data_collector = DataCollector(ticker=ticker)
                stock_data = data_collector.get_stock_data()
                
                if stock_data is None or stock_data.empty:
                    st.error(f"Failed to retrieve stock data for {ticker}")
                    return
            
            # Collect news data and perform sentiment analysis
            with st.spinner("Analyzing news and sentiment..."):
                news_data = data_collector.get_recent_news()
                
                # Enhanced Sentiment Analysis
                sentiment_analyzer = SentimentAnalyzer()
                sentiment_score = sentiment_analyzer.analyze_news_sentiment(news_data, ticker=ticker)
                sentiment_category = sentiment_analyzer.get_sentiment_category(sentiment_score)
            
            # Economic Analysis - get company country and sector
            company_info = data_collector.get_company_info()
            country = company_info.get('country', 'US')
            sector = company_info.get('sector', 'Unknown')
            
            # Use enhanced economic analyzer
            economic_analyzer = EconomicAnalyzer()
            economic_score = economic_analyzer.get_economic_score(country=country, sector=sector)
            economic_data = economic_analyzer.get_detailed_economic_data()
            
            # Stock Prediction - based on selected method
            predictions = {}
            
            if prediction_mode in ["Standard", "Ensemble (All Methods)"]:
                with st.spinner("Running standard prediction models..."):
                    predictor = StockPredictor(stock_data, sentiment_score, economic_score)
                    model = predictor.train_model()
                    predictions['standard'] = predictor.predict_future(model, days=prediction_days)
            
            if prediction_mode in ["Advanced ML", "Ensemble (All Methods)"]:
                with st.spinner("Training advanced machine learning models..."):
                    advanced_predictor = AdvancedPredictionEngine(
                        ticker=ticker,
                        stock_data=stock_data,
                        sentiment_score=sentiment_score,
                        economic_score=economic_score
                    )
                    advanced_predictor.prepare_data()
                    
                    # Apply regime override if specified
                    if regime_override:
                        advanced_predictor._detect_market_regime = lambda df: advanced_predictor._detect_market_regime(df, override=regime_override)
                    
                    # Add cross-asset correlation analysis
                    cross_asset_data = advanced_predictor._analyze_cross_asset_correlations()
                    
                    # Train models with dynamic weighting
                    advanced_predictor.train_models()
                    
                    # Update weights based on market conditions
                    advanced_predictor.prediction_weights = advanced_predictor._adjust_prediction_weights()
                    
                    predictions['advanced'] = advanced_predictor.predict_future(days=prediction_days)
            
            if prediction_mode in ["Pattern Matching", "Ensemble (All Methods)"]:
                with st.spinner("Analyzing historical patterns..."):
                    if PATTERN_MATCHER_AVAILABLE:
                        pattern_matcher = HistoricalPatternMatcher(stock_data, window_size=pattern_window)
                        pattern_predictions, similar_patterns = pattern_matcher.predict_from_patterns(forecast_days=prediction_days)
                        predictions['pattern'] = pattern_predictions
                    else:
                        # Simplified pattern matching if module not available
                        st.warning("Advanced pattern matching not available. Using simplified pattern analysis.")
                        # Create a basic pattern matching prediction
                        last_date = stock_data.index[-1]
                        future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                        pattern_df = pd.DataFrame(index=future_dates)
                        
                        # Find similar patterns (simplified)
                        current_returns = stock_data['Close'].pct_change().iloc[-pattern_window:].fillna(0)
                        
                        # Look for similar patterns in history
                        similar_returns = []
                        similar_indexes = []
                        
                        for i in range(len(stock_data) - 2*pattern_window):
                            historical_returns = stock_data['Close'].pct_change().iloc[i:i+pattern_window].fillna(0)
                            # Calculate similarity (correlation)
                            correlation = current_returns.corr(historical_returns)
                            if correlation > 0.7:  # Strong correlation
                                similar_returns.append(historical_returns)
                                similar_indexes.append(i)
                        
                        # If similar patterns found, predict based on what happened next
                        if similar_returns:
                            next_returns = []
                            for idx in similar_indexes:
                                if idx + pattern_window + prediction_days < len(stock_data):
                                    next_returns.append(stock_data['Close'].pct_change().iloc[idx+pattern_window:idx+pattern_window+prediction_days].fillna(0))
                            
                            if next_returns:
                                # Average the returns
                                avg_returns = pd.concat(next_returns, axis=1).mean(axis=1)
                                
                                # Current price
                                last_price = stock_data['Close'].iloc[-1]
                                
                                # Calculate future prices
                                future_prices = [last_price]
                                for ret in avg_returns:
                                    future_prices.append(future_prices[-1] * (1 + ret))
                                
                                future_prices = future_prices[1:]  # Remove initial price
                                
                                # Add predictions
                                pattern_df['Predicted_Price'] = future_prices
                                
                                # Add simple confidence intervals
                                std_returns = pd.concat(next_returns, axis=1).std(axis=1)
                                pattern_df['Upper_Bound'] = pattern_df['Predicted_Price'] * (1 + 1.96 * std_returns)
                                pattern_df['Lower_Bound'] = pattern_df['Predicted_Price'] * (1 - 1.96 * std_returns)
                                
                                predictions['pattern'] = pattern_df
                            else:
                                st.warning("Not enough historical data to make pattern-based predictions.")
                        else:
                            st.warning("No similar patterns found in historical data.")
            
            if prediction_mode in ["Similar Periods", "Ensemble (All Methods)"]:
                with st.spinner("Finding similar historical periods..."):
                    similar_analyzer = SimilarPeriodAnalyzer(stock_data, window_size=60)
                    similar_forecast, similar_periods = similar_analyzer.predict_based_on_similar_periods(
                        forecast_days=prediction_days
                    )
                    
                    if similar_forecast is not None:
                        predictions['similar_periods'] = similar_forecast
            
            # Create ensemble prediction if using all methods
            if prediction_mode == "Ensemble (All Methods)":
                with st.spinner("Creating ensemble prediction..."):
                    # Create new dataframe for ensemble predictions
                    last_date = stock_data.index[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                    ensemble_df = pd.DataFrame(index=future_dates)
                    
                    # Calculate weights based on current market conditions
                    if 'advanced' in predictions:
                        # Use advanced prediction weights
                        regime_data = advanced_predictor._detect_market_regime(stock_data)
                        regime = regime_data['regime']
                        
                        # Adjust weights based on regime
                        if regime in ["Volatile Bullish", "Volatile Bearish"]:
                            base_weights = {
                                'standard': 0.1,
                                'advanced': 0.4,
                                'pattern': 0.2,
                                'similar_periods': 0.3
                            }
                        elif regime in ["Stable Bullish", "Stable Bearish"]:
                            base_weights = {
                                'standard': 0.2,
                                'advanced': 0.5,
                                'pattern': 0.1,
                                'similar_periods': 0.2
                            }
                        elif regime == "Range Bound":
                            base_weights = {
                                'standard': 0.15,
                                'advanced': 0.3,
                                'pattern': 0.3,
                                'similar_periods': 0.25
                            }
                        else:
                            # Default weights
                            base_weights = {
                                'standard': 0.15,
                                'advanced': 0.4,
                                'pattern': 0.2,
                                'similar_periods': 0.25
                            }
                    else:
                        # Default weights when advanced not available
                        base_weights = {
                            'standard': 0.3,
                            'pattern': 0.3,
                            'similar_periods': 0.4
                        }
                    
                    # Filter to only available prediction types
                    weights = {k: v for k, v in base_weights.items() if k in predictions}
                    
                    # Normalize weights
                    weight_sum = sum(weights.values())
                    if weight_sum > 0:
                        weights = {k: v/weight_sum for k, v in weights.items()}
                    
                    # Initialize columns
                    ensemble_df['Predicted_Price'] = 0
                    ensemble_df['Upper_Bound'] = 0
                    ensemble_df['Lower_Bound'] = 0
                    
                    # Combine predictions with weights
                    for method, df in predictions.items():
                        if method in weights:
                            ensemble_df['Predicted_Price'] += df['Predicted_Price'] * weights[method]
                            ensemble_df['Upper_Bound'] += df['Upper_Bound'] * weights[method]
                            ensemble_df['Lower_Bound'] += df['Lower_Bound'] * weights[method]
                    
                    # Set as primary prediction
                    primary_prediction = ensemble_df
                

            # Economic Indicators Dashboard
            st.subheader("Economic Indicators")
            eco_data = economic_data.get('indicators', {})
            
            # Display economic indicators in a grid
            eco_cols = st.columns(3)
            eco_cols[0].metric("GDP Growth", f"{eco_data.get('GDP', 0):.1f}%")
            eco_cols[1].metric("Inflation", f"{eco_data.get('Inflation', 0):.1f}%")
            eco_cols[2].metric("Fed Funds Rate", f"{eco_data.get('Fed_Funds_Rate', 0):.2f}%")
            
            # Second row of economic indicators
            eco_cols2 = st.columns(3)
            eco_cols2[0].metric("10Y Treasury", f"{eco_data.get('Ten_Year_Treasury', 0):.2f}%")
            eco_cols2[1].metric("Yield Curve", f"{eco_data.get('Yield_Curve', 0):.2f}")
            eco_cols2[2].metric("Unemployment", f"{eco_data.get('Unemployment', 0):.1f}%")
            
            # Market regime indicators
            regime_signals = economic_data.get('regime_signals', {})
            
            # Show current regime
            if 'advanced' in predictions:
                regime_data = advanced_predictor._detect_market_regime(stock_data)
                st.write(f"**Detected Market Regime:** {regime_data['regime']}")
                
                # Show specific regimes if any
                if 'specific_regimes' in regime_data and regime_data['specific_regimes']:
                    st.write(f"**Specific Conditions:** {', '.join(regime_data['specific_regimes'])}")
            
            # Cross-Asset Correlations
            if 'advanced' in predictions and 'cross_asset_data' in locals():
                st.subheader("Cross-Asset Correlations")
                
                # Display key correlations in a table
                correlations = cross_asset_data.get('correlations', {})
                if correlations:
                    corr_data = []
                    for asset, corr in correlations.items():
                        asset_name = asset.replace('_', ' ').title()
                        corr_data.append({
                            "Asset": asset_name,
                            "Correlation": f"{corr:.2f}"
                        })
                    
                    st.table(pd.DataFrame(corr_data))
                
                # Display key signals
                signals = cross_asset_data.get('signals', {})
                active_signals = [k for k, v in signals.items() if v]
                
                if active_signals:
                    st.write("**Active Market Signals:**")
                    for signal in active_signals:
                        signal_name = signal.replace('_', ' ').title()
                        st.write(f"- {signal_name}")

            # Visualizations
            st.subheader(f"{ticker} Stock Price History & Prediction")
            
            # Combine historical data with predictions for visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=stock_data.index, 
                y=stock_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            # Prediction data
            fig.add_trace(go.Scatter(
                x=primary_prediction.index,
                y=primary_prediction['Predicted_Price'],
                mode='lines+markers',
                name='Price Prediction',
                line=dict(color='red', dash='dot'),
                marker=dict(size=8)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=primary_prediction.index,
                y=primary_prediction['Upper_Bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=primary_prediction.index,
                y=primary_prediction['Lower_Bound'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=True
            ))
            
            fig.update_layout(title=f'{ticker} Stock Price with {prediction_days}-Day Forecast')
            st.plotly_chart(fig)
            
            # Display Similar Periods Analysis if available
            if prediction_mode in ["Similar Periods", "Ensemble (All Methods)"] and 'similar_periods' in locals():
                st.subheader("Similar Historical Periods Analysis")
                
                if similar_periods:
                    # Display similar periods in a table
                    similar_data = []
                    for i, period in enumerate(similar_periods):
                        period_data = {
                            "Period": f"#{i+1}",
                            "Dates": f"{period['start_date'].strftime('%Y-%m-%d')} to {period['end_date'].strftime('%Y-%m-%d')}",
                            "Similarity": f"{period['similarity']:.4f}"
                        }
                        
                        # Add returns if available
                        if 'next_7d_return' in period:
                            period_data["7d Return"] = f"{period['next_7d_return']*100:.1f}%"
                        if 'next_30d_return' in period:
                            period_data["30d Return"] = f"{period['next_30d_return']*100:.1f}%"
                        if 'next_30d_max_drawdown' in period:
                            period_data["Max Drawdown"] = f"{period['next_30d_max_drawdown']*100:.1f}%"
                        
                        similar_data.append(period_data)
                    
                    st.table(pd.DataFrame(similar_data))
                    
                    # Create a visualization of similar periods
                    st.write("**Comparison of Current Period with Similar Historical Periods:**")
                    
                    # Plot historical periods vs current
                    comparison_fig = go.Figure()
                    
                    # Current period
                    current_data = stock_data['Close'].iloc[-60:].values
                    current_data_norm = current_data / current_data[0]
                    comparison_fig.add_trace(go.Scatter(
                        x=list(range(len(current_data_norm))),
                        y=current_data_norm,
                        mode='lines',
                        name='Current Period',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Add similar periods
                    colors = ['red', 'green', 'purple', 'orange', 'brown']
                    for i, period in enumerate(similar_periods[:5]):
                        start_idx = period['start_idx']
                        end_idx = period['end_idx']
                        
                        if start_idx >= 0 and end_idx < len(stock_data):
                            historical_data = stock_data['Close'].iloc[start_idx:end_idx+1].values
                            historical_data_norm = historical_data / historical_data[0]
                            
                            comparison_fig.add_trace(go.Scatter(
                                x=list(range(len(historical_data_norm))),
                                y=historical_data_norm,
                                mode='lines',
                                name=f"Period {i+1} ({period['start_date'].strftime('%Y-%m-%d')})",
                                line=dict(color=colors[i % len(colors)], dash='dot')
                            ))
                    
                    comparison_fig.update_layout(
                        title="Normalized Price Comparison with Similar Periods",
                        xaxis_title="Days",
                        yaxis_title="Normalized Price (Starting at 1.0)"
                    )
                    
                    st.plotly_chart(comparison_fig)
                    
                    # What happened after similar periods
                    st.write("**What Typically Happened After Similar Periods:**")
                    
                    # Calculate average outcomes
                    avg_7d = np.mean([p.get('next_7d_return', 0) for p in similar_periods if 'next_7d_return' in p])
                    avg_30d = np.mean([p.get('next_30d_return', 0) for p in similar_periods if 'next_30d_return' in p])
                    avg_dd = np.mean([p.get('next_30d_max_drawdown', 0) for p in similar_periods if 'next_30d_max_drawdown' in p])
                    
                    outcome_cols = st.columns(3)
                    outcome_cols[0].metric("Avg 7-Day Return", f"{avg_7d*100:.1f}%")
                    outcome_cols[1].metric("Avg 30-Day Return", f"{avg_30d*100:.1f}%")
                    outcome_cols[2].metric("Avg Max Drawdown", f"{avg_dd*100:.1f}%")
                else:
                    st.info("No similar historical periods found with sufficient data.")
            
            # Display pattern matching visualization if available
            if prediction_mode in ["Pattern Matching", "Ensemble (All Methods)"] and PATTERN_MATCHER_AVAILABLE:
                try:
                    st.subheader("Technical Pattern Analysis")
                    pattern_fig = pattern_matcher.visualize_patterns(similar_patterns)
                    st.plotly_chart(pattern_fig)
                    
                    # Display pattern details
                    st.write("**Historical Pattern Details:**")
                    pattern_details = []
                    for i, pattern in enumerate(similar_patterns):
                        pattern_details.append({
                            "Pattern": i+1,
                            "Start Date": pattern['start_date'].strftime('%Y-%m-%d'),
                            "End Date": pattern['end_date'].strftime('%Y-%m-%d'),
                            "Similarity Score": f"{pattern['similarity']:.4f}",
                            "Subsequent Return": f"{pattern['next_period_return']*100:.2f}%" if pattern['next_period_return'] is not None else "N/A"
                        })
                    
                    st.table(pd.DataFrame(pattern_details))
                except Exception as e:
                    st.error(f"Error displaying pattern analysis: {e}")
            
            # Display company info
            st.subheader("Company Information")
            if company_info:
                info_cols = st.columns(3)
                info_cols[0].metric("Company", company_info.get('name', ticker))
                info_cols[1].metric("Sector", company_info.get('sector', 'Unknown'))
                info_cols[2].metric("Country", company_info.get('country', 'Unknown'))
                
                # Add more company info in an expander
                with st.expander("More Company Information"):
                    st.write(f"**Market Cap:** {company_info.get('market_cap', 'N/A')}")
                    st.write(f"**P/E Ratio:** {company_info.get('pe_ratio', 'N/A')}")
                    st.write(f"**Employees:** {company_info.get('employees', 'N/A')}")
                    st.write(f"**Description:** {company_info.get('description', 'No description available.')}")
            
            # Enhanced prediction summary with more metrics
            st.subheader("Prediction Summary")
            latest_price = stock_data['Close'].iloc[-1]
            next_day_price = primary_prediction['Predicted_Price'].iloc[0]
            end_price = primary_prediction['Predicted_Price'].iloc[-1]
            
            price_change = ((next_day_price - latest_price) / latest_price) * 100
            period_change = ((end_price - latest_price) / latest_price) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${latest_price:.2f}")
            col2.metric("Next Day Prediction", f"${next_day_price:.2f}", f"{price_change:.2f}%")
            col3.metric(f"{prediction_days}-Day Prediction", f"${end_price:.2f}", f"{period_change:.2f}%")
            
            # Risk assessment
            st.subheader("Risk Assessment")
            
            # Calculate volatility and risk metrics
            daily_returns = stock_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(daily_returns, 5) * latest_price
            
            # Maximum Drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            max_return = cumulative_returns.cummax()
            drawdown = (cumulative_returns / max_return) - 1
            max_drawdown = drawdown.min() * 100
            
            # Prediction uncertainty (ratio of confidence interval to price)
            uncertainty = np.mean((primary_prediction['Upper_Bound'] - primary_prediction['Lower_Bound']) / primary_prediction['Predicted_Price'])
            
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            risk_col1.metric("Annual Volatility", f"{volatility*100:.2f}%")
            risk_col2.metric("Daily Value at Risk (95%)", f"${abs(var_95):.2f}")
            risk_col3.metric("Historical Max Drawdown", f"{abs(max_drawdown):.2f}%")
            risk_col4.metric("Prediction Uncertainty", f"{uncertainty*100:.2f}%")
            
            # Display prediction table
            st.subheader("Daily Price Predictions")
            st.dataframe(primary_prediction[['Predicted_Price', 'Lower_Bound', 'Upper_Bound']])
            
            # Advanced analysis components (only for advanced modes)
            if prediction_mode in ["Advanced ML", "Ensemble (All Methods)"]:
                st.subheader("Advanced Analysis Components")
                
                try:
                    # Display model weights with improved visualization
                    if 'advanced' in predictions:
                        st.write("**Prediction Model Weights:**")
                        advanced_predictor_weights = advanced_predictor.prediction_weights
                        weight_df = pd.DataFrame({'weight': advanced_predictor_weights}).sort_values('weight', ascending=False)
                        
                        # Create a horizontal bar chart
                        weights_fig = go.Figure()
                        weights_fig.add_trace(go.Bar(
                            x=list(weight_df['weight']),
                            y=list(weight_df.index),
                            orientation='h',
                            marker_color='darkblue'
                        ))
                        weights_fig.update_layout(
                            title="Model Contribution Weights",
                            xaxis_title="Weight",
                            yaxis_title="Model",
                            height=400
                        )
                        st.plotly_chart(weights_fig)
                    
                    # Display feature importance with improved visualization
                    if 'advanced' in predictions and 'random_forest' in advanced_predictor.models:
                        rf_model = advanced_predictor.models['random_forest']['model']
                        feature_names = advanced_predictor.models['random_forest']['features']
                        importances = rf_model.feature_importances_
                        
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        st.write("**Feature Importance:**")
                        
                        # Create a horizontal bar chart
                        importance_fig = go.Figure()
                        importance_fig.add_trace(go.Bar(
                            x=list(feature_importance['importance'])[:10],  # Top 10 features
                            y=list(feature_importance['feature'])[:10],
                            orientation='h',
                            marker_color='darkgreen'
                        ))
                        importance_fig.update_layout(
                            title="Top 10 Feature Importance",
                            xaxis_title="Importance",
                            yaxis_title="Feature",
                            height=400
                        )
                        st.plotly_chart(importance_fig)
                        
                    # Support and resistance levels
                    if 'advanced' in predictions:
                        mtf_data = advanced_predictor._multi_timeframe_analysis()
                        support_levels = mtf_data.get('support_levels', [])
                        resistance_levels = mtf_data.get('resistance_levels', [])
                        
                        if support_levels or resistance_levels:
                            st.write("**Key Price Levels:**")
                            
                            # Create a price levels chart
                            levels_fig = go.Figure()
                            
                            # Add historical prices
                            levels_fig.add_trace(go.Scatter(
                                x=stock_data.index[-30:],
                                y=stock_data['Close'].iloc[-30:],
                                mode='lines',
                                name='Price',
                                line=dict(color='blue')
                            ))
                            
                            # Add support levels
                            for level in support_levels:
                                levels_fig.add_shape(
                                    type="line",
                                    x0=stock_data.index[-30],
                                    y0=level,
                                    x1=stock_data.index[-1] + timedelta(days=prediction_days),
                                    y1=level,
                                    line=dict(color="green", width=2, dash="dash"),
                                )
                            
                            # Add resistance levels
                            for level in resistance_levels:
                                levels_fig.add_shape(
                                    type="line",
                                    x0=stock_data.index[-30],
                                    y0=level,
                                    x1=stock_data.index[-1] + timedelta(days=prediction_days),
                                    y1=level,
                                    line=dict(color="red", width=2, dash="dash"),
                                )
                            
                            levels_fig.update_layout(
                                title="Support and Resistance Levels",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                showlegend=True
                            )
                            
                            st.plotly_chart(levels_fig)
                except Exception as e:
                    st.error(f"Error displaying advanced analysis: {e}")
            
            # Display scores
            st.subheader("Analysis Factors")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment Score", f"{sentiment_score:.2f}")
            col2.metric("Sentiment Category", sentiment_category)
            col3.metric("Economic Score", f"{economic_score:.2f}")
            
            # Add explanation of scores
            with st.expander("Understanding the Analysis"):
                st.write("""
                ## Enhanced Prediction Methodology
                
                The prediction system has been upgraded with the following improvements:
                
                1. **Real-Time Economic Data**: Using current GDP, inflation, interest rates, and yield curve data
                
                2. **Finance-Specific Sentiment Analysis**: Analyzing news and earnings with financial context
                
                3. **Enhanced Market Regime Detection**: Identifying the current market environment
                
                4. **Cross-Asset Correlations**: Incorporating bonds, commodities, and currencies
                
                5. **Dynamic Model Weighting**: Adjusting model weights based on market conditions
                
                6. **Similar Period Analysis**: Finding historical patterns that match today's market
                
                The system now considers the current economic climate, particularly interest rates and inflation, which greatly impact stock valuations. It also adapts to different market regimes, reducing reliance on patterns that may not be relevant in the current environment.
                """)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.write(traceback.format_exc())
            st.write("Please try another stock ticker or check your internet connection.")

if __name__ == "__main__":
    main()