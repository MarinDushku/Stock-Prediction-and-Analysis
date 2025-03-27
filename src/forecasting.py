import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

def prepare_prophet_data(df):
    """
    Prepare data for Prophet forecasting
    
    Parameters:
    - df: Input DataFrame
    
    Returns:
    - Prepared DataFrame for Prophet
    """
    if df is None:
        st.error("No data to prepare for forecasting")
        return None
    
    # Create a DataFrame with date and closing price
    prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    return prophet_df

def forecast_stock_price(prophet_df, periods=30):
    """
    Forecast stock prices using Facebook Prophet
    
    Parameters:
    - prophet_df: Prepared DataFrame
    - periods: Number of days to forecast
    
    Returns:
    - Forecast DataFrame and Prophet model
    """
    if prophet_df is None:
        st.error("No data for forecasting")
        return None, None
    
    try:
        # Initialize and fit the Prophet model
        model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            daily_seasonality=False,
            uncertainty_samples=1000
        )
        model.fit(prophet_df)
        
        # Create future dataframe for prediction
        future = model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = model.predict(future)
        
        return forecast, model
    
    except Exception as e:
        st.error(f"Forecasting error: {e}")
        return None, None

def generate_forecast_visualizations(forecast, model):
    """
    Create forecast visualizations
    
    Parameters:
    - forecast: Prophet forecast DataFrame
    - model: Trained Prophet model
    
    Returns:
    - Plotly figure objects
    """
    if forecast is None or model is None:
        st.error("Cannot generate visualizations")
        return None
    
    # Main forecast plot
    fig1 = plot_plotly(model, forecast)
    fig1.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
    
    # Components plot
    fig2 = model.plot_components(forecast)
    
    # Create a Plotly figure from the Prophet components plot
    components_fig = go.Figure()
    
    # Trend component
    trend_trace = go.Scatter(
        x=forecast['ds'], 
        y=forecast['trend'], 
        mode='lines', 
        name='Trend'
    )
    components_fig.add_trace(trend_trace)
    
    # Yearly seasonality
    if 'yearly' in forecast.columns:
        yearly_trace = go.Scatter(
            x=forecast['ds'], 
            y=forecast['yearly'], 
            mode='lines', 
            name='Yearly Seasonality'
        )
        components_fig.add_trace(yearly_trace)
    
    components_fig.update_layout(
        title='Forecast Components', 
        xaxis_title='Date', 
        yaxis_title='Impact'
    )
    
    return {
        'main_forecast': fig1,
        'components': components_fig
    }

def calculate_forecast_metrics(forecast):
    """
    Calculate key forecast metrics
    
    Parameters:
    - forecast: Prophet forecast DataFrame
    
    Returns:
    - Dictionary of forecast metrics
    """
    if forecast is None:
        st.error("Cannot calculate forecast metrics")
        return None
    
    # Get the last few rows of the forecast
    recent_forecast = forecast.tail(30)
    
    metrics = {
        'Next_Day_Prediction': recent_forecast.iloc[-1]['yhat'],
        'Next_Week_Average': recent_forecast['yhat'].mean(),
        'Prediction_Interval_Low': recent_forecast['yhat_lower'].mean(),
        'Prediction_Interval_High': recent_forecast['yhat_upper'].mean(),
        'Forecast_Uncertainty': (recent_forecast['yhat_upper'] - recent_forecast['yhat_lower']).mean()
    }
    
    return metrics

# Example usage (can be commented out later)
if __name__ == "__main__":
    from data_acquisition import fetch_stock_data
    from data_preprocessing import clean_stock_data
    
    ticker = 'AAPL'
    raw_data = fetch_stock_data(ticker)
    cleaned_data = clean_stock_data(raw_data)
    
    # Prepare data for Prophet
    prophet_df = prepare_prophet_data(cleaned_data)
    
    # Generate forecast
    forecast, model = forecast_stock_price(prophet_df)
    
    # Generate visualizations
    visualizations = generate_forecast_visualizations(forecast, model)
    
    # Calculate metrics
    metrics = calculate_forecast_metrics(forecast)
    
    print("Forecast Metrics:")
    print(metrics)