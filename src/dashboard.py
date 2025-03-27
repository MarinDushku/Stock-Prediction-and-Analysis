import streamlit as st
import plotly.express as px
import plotly.graph_objs as go

# Import our custom modules
from data_acquisition import fetch_stock_data, get_stock_info
from data_preprocessing import clean_stock_data, calculate_technical_indicators
from forecasting import (
    prepare_prophet_data, 
    forecast_stock_price, 
    generate_forecast_visualizations,
    calculate_forecast_metrics
)

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Stock Market Analysis Dashboard", 
        page_icon=":chart_with_upwards_trend:", 
        layout="wide"
    )

    # Title and description
    st.title("📈 Stock Market Trend Analysis Dashboard")
    st.write("Analyze and Forecast Stock Performance in Real-Time")

    # Sidebar for user inputs
    st.sidebar.header("Stock Analysis Parameters")
    
    # Stock ticker input
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
    
    # Prediction horizon selection
    prediction_horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        ["Next Day", "Next Week", "Next Month"]
    )

    # Fetch and process data
    try:
        # Fetch raw stock data
        raw_data = fetch_stock_data(ticker)
        
        # Get additional stock info
        stock_info = get_stock_info(ticker)
        
        # Clean and preprocess data
        cleaned_data = clean_stock_data(raw_data)
        technical_data = calculate_technical_indicators(cleaned_data)
        
        # Prepare for forecasting
        prophet_df = prepare_prophet_data(cleaned_data)
        forecast, model = forecast_stock_price(prophet_df)
        
        # Generate visualizations
        forecast_viz = generate_forecast_visualizations(forecast, model)
        
        # Calculate forecast metrics
        forecast_metrics = calculate_forecast_metrics(forecast)

        # Dashboard Layout
        col1, col2, col3 = st.columns(3)
        
        # Company Info Column
        with col1:
            st.subheader("Company Information")
            if stock_info:
                for key, value in stock_info.items():
                    st.metric(label=key, value=value)
            else:
                st.warning("Could not fetch company information")

        # Stock Performance Column
        with col2:
            st.subheader("Stock Performance")
            # Recent price chart
            fig_price = px.line(
                technical_data, 
                y=['Close', '50_Day_MA', '200_Day_MA'], 
                title='Stock Price with Moving Averages'
            )
            st.plotly_chart(fig_price)

            # Technical Indicators
            st.subheader("Technical Indicators")
            st.metric("RSI", f"{technical_data['RSI'].iloc[-1]:.2f}")
            st.metric("MACD", f"{technical_data['MACD'].iloc[-1]:.2f}")

        # Forecast Column
        with col3:
            st.subheader("Price Forecast")
            if forecast_metrics:
                # Display forecast metrics based on selected horizon
                if prediction_horizon == "Next Day":
                    st.metric("Next Day Prediction", f"${forecast_metrics['Next_Day_Prediction']:.2f}")
                elif prediction_horizon == "Next Week":
                    st.metric("Next Week Average", f"${forecast_metrics['Next_Week_Average']:.2f}")
                
                # Confidence Interval
                st.metric(
                    "Prediction Range", 
                    f"${forecast_metrics['Prediction_Interval_Low']:.2f} - ${forecast_metrics['Prediction_Interval_High']:.2f}"
                )
                
                # Forecast Visualization
                if forecast_viz:
                    st.plotly_chart(forecast_viz['main_forecast'])
            else:
                st.warning("Forecast not available")

        # Additional Visualization Tabs
        st.subheader("Advanced Visualizations")
        tab1, tab2 = st.tabs(["Forecast Components", "Technical Analysis"])
        
        with tab1:
            if forecast_viz:
                st.plotly_chart(forecast_viz['components'])
            else:
                st.warning("Forecast components not available")
        
        with tab2:
            # RSI and Bollinger Bands visualization
            fig_technical = go.Figure()
            fig_technical.add_trace(go.Scatter(
                x=technical_data.index, 
                y=technical_data['Close'], 
                name='Close Price'
            ))
            fig_technical.add_trace(go.Scatter(
                x=technical_data.index, 
                y=technical_data['Upper_Band'], 
                name='Upper Bollinger Band'
            ))
            fig_technical.add_trace(go.Scatter(
                x=technical_data.index, 
                y=technical_data['Lower_Band'], 
                name='Lower Bollinger Band'
            ))
            fig_technical.update_layout(title='Bollinger Bands')
            st.plotly_chart(fig_technical)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Run the dashboard
if __name__ == "__main__":
    main()