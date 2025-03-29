import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import requests
import json
import os

# Try to import XGBoost but handle if it's not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using GradientBoostingRegressor instead")

# Try importing ta package, use fallback if not available
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Technical Analysis package not available, using simplified indicators")

# Try importing TensorFlow/Keras for LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, LSTM models will be skipped")

# Try importing HMM for market regime detection
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("hmmlearn not available, using simplified regime detection")

class AdvancedPredictionEngine:
    def __init__(self, ticker, stock_data, sentiment_score, economic_score):
        self.ticker = ticker
        self.stock_data = stock_data
        self.base_sentiment_score = sentiment_score
        self.base_economic_score = economic_score
        self.features = {}
        self.models = {}
        self.prediction_weights = {
            'random_forest': 0.25,
            'boosting': 0.20,  # 'xgboost' or 'gradient_boosting'
            'linear': 0.10,
            'technical': 0.15,
            'sentiment': 0.10,
            'lstm': 0.20
        }
        
        # Feature importance weights
        self.feature_weights = {
            'price_history': 0.30,
            'technical_indicators': 0.20,
            'sentiment_analysis': 0.15,
            'market_regime': 0.10,
            'sector_correlation': 0.10,
            'macroeconomic': 0.10,
            'volume_analysis': 0.05
        }
        
    def prepare_data(self):
        """Prepare and integrate all data sources and features"""
        # Clone base data
        df = self.stock_data.copy()
        
        # Basic price features (already in your system)
        df['Prev_Close'] = df['Close'].shift(1)
        df['Price_Diff'] = df['Close'] - df['Prev_Close']
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Return_Volatility'] = df['Daily_Return'].rolling(window=20).std().fillna(0)
        
        # Add technical indicators
        self._add_technical_indicators(df)
        
        # Detect market regime
        regime_data = self._detect_market_regime(df)
        df['Market_Regime'] = regime_data['regime_numeric']
        
        # Add sector correlation data
        sector_data = self._analyze_sector_correlations()
        df['Sector_Correlation'] = sector_data.get('correlation_score', 0)
        df['Sector_Relative_Strength'] = sector_data.get('relative_strength', 1)
        
        # Enhanced sentiment analysis
        sentiment_data = self._comprehensive_sentiment_analysis()
        df['Enhanced_Sentiment'] = sentiment_data.get('composite_score', self.base_sentiment_score)
        
        # Macroeconomic indicators
        macro_data = self._incorporate_macro_indicators()
        df['Macro_Score'] = macro_data.get('macro_score', self.base_economic_score)
        
        # Technical patterns
        pattern_data = self._detect_technical_patterns(df)
        df['Pattern_Score'] = pattern_data.get('pattern_score', 0)
        
        # Volume profile analysis
        volume_data = self._analyze_volume_profile(df)
        df['Volume_Profile_Score'] = volume_data.get('score', 0)
        
        # Multi-timeframe analysis
        timeframe_data = self._multi_timeframe_analysis()
        df['Timeframe_Alignment'] = timeframe_data.get('timeframe_alignment', 0)
        
        # Options market sentiment (if available)
        try:
            options_data = self._analyze_options_market()
            df['Options_Sentiment'] = options_data.get('options_sentiment', 0)
        except:
            df['Options_Sentiment'] = 0
            
        # Price momentum indicators
        df['Momentum_1D'] = df['Close'].pct_change(1)
        df['Momentum_5D'] = df['Close'].pct_change(5)
        df['Momentum_10D'] = df['Close'].pct_change(10)
        
        # Advanced volatility metrics
        df['ATR_Ratio'] = df['ATR'] / df['Close'] if 'ATR' in df.columns else 0
        
        # Market breadth indicators (simulated)
        df['Market_Breadth'] = np.random.normal(0, 0.1, size=len(df))  # In production would use real data
        
        # Elliott Wave pattern detection (simplified)
        df['Wave_Pattern'] = self._detect_elliott_wave_patterns(df)
        
        # Fibonacci retracement levels
        fib_levels = self._calculate_fibonacci_levels(df)
        for level_name, level_value in fib_levels.items():
            df[f'Fib_{level_name}'] = df['Close'] / level_value - 1  # Distance to Fibonacci level
        
        # Cross-asset correlation analysis
        cross_asset_data = self._analyze_cross_asset_correlations()
        df['Cross_Asset_Impact'] = cross_asset_data.get('impact_score', 0)
        
        # Fill any NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        # Store prepared data
        self.prepared_data = df
        return df
    
    def _add_technical_indicators(self, df):
        """Add comprehensive technical indicators"""
        # Moving Averages
        df['5_Day_MA'] = df['Close'].rolling(window=5).mean()
        df['10_Day_MA'] = df['Close'].rolling(window=10).mean()
        df['20_Day_MA'] = df['Close'].rolling(window=20).mean()
        df['50_Day_MA'] = df['Close'].rolling(window=50).mean()
        df['200_Day_MA'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['5_Day_EMA'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['13_Day_EMA'] = df['Close'].ewm(span=13, adjust=False).mean()
        df['26_Day_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Use ta library if available, otherwise calculate basic indicators
        if TA_AVAILABLE:
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_High'] = bollinger.bollinger_hband()
            df['BB_Low'] = bollinger.bollinger_lband()
            df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
            
            # ATR (Average True Range)
            df['ATR'] = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close']
            ).average_true_range()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                df['High'], df['Low'], df['Close']
            )
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Additional TA indicators
            # ADX (Average Directional Index)
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
            df['ADX'] = adx.adx()
            df['DI_plus'] = adx.adx_pos()
            df['DI_minus'] = adx.adx_neg()
            
            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            df['Ichimoku_a'] = ichimoku.ichimoku_a()
            df['Ichimoku_b'] = ichimoku.ichimoku_b()
            df['Ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['Ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
            
            # VWAP (Volume Weighted Average Price)
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Parabolic SAR
            try:
                df['SAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
            except:
                df['SAR'] = df['Close']  # Fallback
                
            # Awesome Oscillator
            df['AO'] = ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator()
            
        else:
            # Basic RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50)
            
            # Basic MACD
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
            
            # Basic Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            std_dev = df['Close'].rolling(window=20).std()
            df['BB_High'] = df['BB_Middle'] + 2 * std_dev
            df['BB_Low'] = df['BB_Middle'] - 2 * std_dev
            df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
            
            # Basic ATR calculation
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
        
        # Additional custom indicators
        
        # Hull Moving Average (HMA)
        def hma(series, period):
            return series.ewm(span=period, adjust=False).mean().ewm(span=int(period/2), adjust=False).mean()
        
        df['HMA_10'] = hma(df['Close'], 10)
        
        # Rate of Change (ROC)
        df['ROC_10'] = df['Close'].pct_change(10) * 100
        
        # Money Flow Index (simplified)
        if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            raw_money_flow = typical_price * df['Volume']
            
            pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
            neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
            
            pos_flow_sum = pos_flow.rolling(window=14).sum()
            neg_flow_sum = neg_flow.rolling(window=14).sum()
            
            money_ratio = pos_flow_sum / neg_flow_sum
            df['MFI'] = 100 - (100 / (1 + money_ratio))
        
        # TRIX (Triple Exponentially Smoothed Moving Average)
        ema1 = df['Close'].ewm(span=15, adjust=False).mean()
        ema2 = ema1.ewm(span=15, adjust=False).mean()
        ema3 = ema2.ewm(span=15, adjust=False).mean()
        df['TRIX'] = (ema3 / ema3.shift(1) - 1) * 100
        
        return df
    
    def _detect_market_regime(self, df, override=None):
        """Detect the current market regime using enhanced methods with override capability"""
        # Check if user provided a manual override
        if override is not None:
            if isinstance(override, str):
                # Map text override to numeric value
                regime_map = {
                    "Stable Bullish": 1.0,
                    "Volatile Bullish": 0.5,
                    "Range Bound": 0.0,
                    "Volatile Bearish": -0.5,
                    "Stable Bearish": -1.0,
                    "Inflation Regime": -0.3,
                    "Rising Rate Regime": -0.4,
                    "Growth Slowdown": -0.2,
                    "Recovery": 0.7
                }
                regime_numeric = regime_map.get(override, 0)
                return {
                    'regime': override,
                    'regime_numeric': regime_numeric,
                    'manual_override': True
                }
            elif isinstance(override, dict):
                # User provided a complete regime dictionary
                return override
        
        # If HMM available, use it for sophisticated regime detection
        if HMM_AVAILABLE and len(df) >= 100:
            try:
                # Extract features for regime detection
                returns = df['Close'].pct_change().fillna(0).values.reshape(-1, 1)
                
                # Add volatility as second dimension if we have enough data
                if len(df) >= 30:
                    volatility = df['Close'].pct_change().rolling(20).std().fillna(0).values.reshape(-1, 1)
                    X = np.column_stack([returns, volatility])
                else:
                    X = returns
                
                # Train HMM with 4 regimes
                model = hmm.GaussianHMM(n_components=4, covariance_type="diag", random_state=42)
                model.fit(X)
                
                # Predict regimes
                hidden_states = model.predict(X)
                
                # Map numeric states to meaningful regimes based on characteristics
                last_state = hidden_states[-1]
                
                # Calculate average return and volatility for each regime
                regime_returns = {}
                regime_vols = {}
                
                for i in range(4):
                    mask = (hidden_states == i)
                    if np.any(mask):
                        regime_returns[i] = np.mean(returns[mask])
                        if len(df) >= 30:
                            regime_vols[i] = np.mean(volatility[mask])
                        else:
                            regime_vols[i] = np.std(returns[mask])
                
                # Enhanced regime detection with more regimes
                regimes = {
                    'stable_bullish': {'idx': -1, 'score': -999},
                    'volatile_bullish': {'idx': -1, 'score': -999},
                    'range_bound': {'idx': -1, 'score': -999},
                    'volatile_bearish': {'idx': -1, 'score': -999},
                    'stable_bearish': {'idx': -1, 'score': -999},
                    'recovery': {'idx': -1, 'score': -999},
                    'correction': {'idx': -1, 'score': -999}
                }
                
                # Score each state based on return and volatility
                for i in range(4):
                    if i in regime_returns:
                        ret = regime_returns[i]
                        vol = regime_vols[i]
                        
                        # Bullish score: high return, low vol
                        bull_score = ret - vol
                        if bull_score > regimes['stable_bullish']['score']:
                            regimes['stable_bullish']['idx'] = i
                            regimes['stable_bullish']['score'] = bull_score
                        
                        # Volatile bullish: high return, high vol
                        vol_bull_score = ret
                        if vol > np.median(list(regime_vols.values())) and vol_bull_score > regimes['volatile_bullish']['score']:
                            regimes['volatile_bullish']['idx'] = i
                            regimes['volatile_bullish']['score'] = vol_bull_score
                        
                        # Range bound: low vol
                        range_score = -vol
                        if range_score > regimes['range_bound']['score']:
                            regimes['range_bound']['idx'] = i
                            regimes['range_bound']['score'] = range_score
                        
                        # Bearish: negative return, high vol
                        bear_score = -ret
                        if ret < 0 and bear_score > regimes['volatile_bearish']['score']:
                            regimes['volatile_bearish']['idx'] = i
                            regimes['volatile_bearish']['score'] = bear_score
                        
                        # Stable bearish: negative return, low vol
                        stable_bear_score = -ret - vol
                        if ret < 0 and stable_bear_score > regimes['stable_bearish']['score']:
                            regimes['stable_bearish']['idx'] = i
                            regimes['stable_bearish']['score'] = stable_bear_score
                        
                        # Recovery: was negative, becoming positive
                        recovery_score = ret
                        recovery_trend = np.mean(returns[-20:]) > np.mean(returns[-40:-20])
                        if ret > 0 and recovery_trend and recovery_score > regimes['recovery']['score']:
                            regimes['recovery']['idx'] = i
                            regimes['recovery']['score'] = recovery_score
                        
                        # Correction: was positive, becoming negative
                        correction_score = -ret
                        correction_trend = np.mean(returns[-20:]) < np.mean(returns[-40:-20])
                        if ret < 0 and correction_trend and correction_score > regimes['correction']['score']:
                            regimes['correction']['idx'] = i
                            regimes['correction']['score'] = correction_score
                
                # Determine current regime
                if last_state == regimes['stable_bullish']['idx']:
                    regime = "Stable Bullish"
                    regime_numeric = 1.0
                elif last_state == regimes['volatile_bullish']['idx']:
                    regime = "Volatile Bullish"
                    regime_numeric = 0.5
                elif last_state == regimes['range_bound']['idx']:
                    regime = "Range Bound"
                    regime_numeric = 0.0
                elif last_state == regimes['volatile_bearish']['idx']:
                    regime = "Volatile Bearish"
                    regime_numeric = -0.5
                elif last_state == regimes['stable_bearish']['idx']:
                    regime = "Stable Bearish"
                    regime_numeric = -1.0
                elif last_state == regimes['recovery']['idx']:
                    regime = "Recovery"
                    regime_numeric = 0.7
                elif last_state == regimes['correction']['idx']:
                    regime = "Correction"
                    regime_numeric = -0.3
                else:
                    # Fallback
                    volatility = df['Close'].pct_change().rolling(20).std().iloc[-1]
                    trend = df['Close'].iloc[-1] - df['Close'].iloc[-20]
                    
                    if trend > 0:
                        regime = "Stable Bullish"
                        regime_numeric = 1.0
                    else:
                        regime = "Stable Bearish"
                        regime_numeric = -1.0
                
                return {
                    'regime': regime,
                    'regime_numeric': regime_numeric,
                    'hidden_states': hidden_states,
                    'regime_returns': regime_returns,
                    'regime_volatilities': regime_vols
                }
                
            except Exception as e:
                print(f"Error in HMM regime detection: {e}")
                # Fall back to enhanced method
                pass
        
        # Enhanced method with more market indicators
        
        # 1. Volatility analysis
        volatility = df['Close'].pct_change().rolling(30).std().iloc[-1]
        historical_vol = df['Close'].pct_change().rolling(30).std().rolling(100).mean().iloc[-1]
        relative_vol = volatility / historical_vol if historical_vol > 0 else 1
        
        # 2. Trend analysis using multiple timeframes
        short_trend = df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1
        medium_trend = df['Close'].iloc[-1] / df['Close'].iloc[-30] - 1
        long_trend = df['Close'].iloc[-1] / df['Close'].iloc[-90] - 1
        
        # 3. Moving average analysis
        if '50_Day_MA' in df.columns and '200_Day_MA' in df.columns:
            ma_trend = df['50_Day_MA'].iloc[-1] - df['200_Day_MA'].iloc[-1]
            golden_cross = df['50_Day_MA'].iloc[-1] > df['200_Day_MA'].iloc[-1] and df['50_Day_MA'].iloc[-20] < df['200_Day_MA'].iloc[-20]
            death_cross = df['50_Day_MA'].iloc[-1] < df['200_Day_MA'].iloc[-1] and df['50_Day_MA'].iloc[-20] > df['200_Day_MA'].iloc[-20]
        else:
            ma_trend = long_trend
            golden_cross = False
            death_cross = False
        
        # 4. Volume analysis
        if 'Volume' in df.columns:
            recent_volume = df['Volume'].iloc[-10:].mean()
            historical_volume = df['Volume'].iloc[-50:-10].mean()
            relative_volume = recent_volume / historical_volume if historical_volume > 0 else 1
        else:
            relative_volume = 1
        
        # 5. RSI analysis
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            rsi_trend = df['RSI'].iloc[-1] - df['RSI'].iloc[-10]
        else:
            # Calculate RSI if not available
            delta = df['Close'].diff().dropna()
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            avg_gain = gains.rolling(window=14).mean().iloc[-1]
            avg_loss = -losses.rolling(window=14).mean().iloc[-1]
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            rsi_trend = 0
        
        # 6. Advanced trend detection - use rate of change in multiple timeframes
        price_roc = {
            '5d': df['Close'].pct_change(5).iloc[-1],
            '10d': df['Close'].pct_change(10).iloc[-1],
            '20d': df['Close'].pct_change(20).iloc[-1],
            '60d': df['Close'].pct_change(60).iloc[-1]
        }
        
        # 7. Identify distribution vs accumulation
        if 'Volume' in df.columns:
            # Accumulation: Rising prices on increasing volume
            # Distribution: Falling prices on increasing volume
            price_vol_correlation = np.corrcoef(
                df['Close'].pct_change().iloc[-20:].fillna(0),
                df['Volume'].pct_change().iloc[-20:].fillna(0)
            )[0, 1]
        else:
            price_vol_correlation = 0
        
        # Regime classification based on multiple indicators
        
        # Determine primary trend direction
        if (short_trend > 0 and medium_trend > 0 and long_trend > 0):
            trend_direction = "Strong Bullish"
            trend_score = 1.0
        elif (short_trend > 0 and medium_trend > 0):
            trend_direction = "Moderate Bullish"
            trend_score = 0.7
        elif (short_trend > 0 and medium_trend < 0 and long_trend < 0):
            trend_direction = "Potential Reversal Up"
            trend_score = 0.3
        elif (short_trend < 0 and medium_trend < 0 and long_trend < 0):
            trend_direction = "Strong Bearish"
            trend_score = -1.0
        elif (short_trend < 0 and medium_trend < 0):
            trend_direction = "Moderate Bearish"
            trend_score = -0.7
        elif (short_trend < 0 and medium_trend > 0 and long_trend > 0):
            trend_direction = "Potential Reversal Down"
            trend_score = -0.3
        elif (abs(short_trend) < 0.02 and abs(medium_trend) < 0.05):
            trend_direction = "Range Bound"
            trend_score = 0
        else:
            trend_direction = "Mixed"
            trend_score = 0.1 * (short_trend + medium_trend + long_trend) / 3
        
        # Determine volatility regime
        if relative_vol > 1.5:
            vol_regime = "High Volatility"
            vol_score = -0.5
        elif relative_vol < 0.7:
            vol_regime = "Low Volatility"
            vol_score = 0.3
        else:
            vol_regime = "Normal Volatility"
            vol_score = 0
        
        # Combined regime determination
        if trend_direction in ["Strong Bullish", "Moderate Bullish"]:
            if vol_regime == "High Volatility":
                regime = "Volatile Bullish"
                regime_numeric = 0.5
            else:
                regime = "Stable Bullish"
                regime_numeric = 1.0
        elif trend_direction in ["Strong Bearish", "Moderate Bearish"]:
            if vol_regime == "High Volatility":
                regime = "Volatile Bearish"
                regime_numeric = -0.5
            else:
                regime = "Stable Bearish"
                regime_numeric = -1.0
        elif trend_direction == "Range Bound":
            regime = "Range Bound"
            regime_numeric = 0.0
        elif trend_direction == "Potential Reversal Up":
            regime = "Potential Bottom"
            regime_numeric = 0.2
        elif trend_direction == "Potential Reversal Down":
            regime = "Potential Top"
            regime_numeric = -0.2
        else:
            regime = "Mixed"
            regime_numeric = 0
        
        # Detect specific market regimes
        specific_regimes = []
        
        # Check for bull/bear market
        if long_trend > 0.20:
            specific_regimes.append("Bull Market")
        elif long_trend < -0.20:
            specific_regimes.append("Bear Market")
        
        # Check for correction
        if medium_trend < -0.10 and long_trend > 0:
            specific_regimes.append("Correction")
        
        # Check for recovery/rally
        if short_trend > 0.05 and medium_trend < 0:
            specific_regimes.append("Relief Rally")
        
        # Check for consolidation
        if abs(short_trend) < 0.03 and abs(medium_trend) < 0.05:
            specific_regimes.append("Consolidation")
        
        # Check for breakout
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
            price_change = abs(df['Close'].iloc[-1] - df['Close'].iloc[-5])
            if price_change > 3 * atr:
                if short_trend > 0:
                    specific_regimes.append("Bullish Breakout")
                else:
                    specific_regimes.append("Bearish Breakdown")
        
        # Check for oversold/overbought
        if rsi < 30:
            specific_regimes.append("Oversold")
        elif rsi > 70:
            specific_regimes.append("Overbought")
        
        # Return detailed regime data
        return {
            'regime': regime,
            'regime_numeric': regime_numeric,
            'trend_direction': trend_direction,
            'volatility_regime': vol_regime,
            'specific_regimes': specific_regimes,
            'details': {
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'long_trend': long_trend,
                'relative_vol': relative_vol,
                'rsi': rsi,
                'ma_trend': ma_trend if 'ma_trend' in locals() else None,
                'golden_cross': golden_cross if 'golden_cross' in locals() else None,
                'death_cross': death_cross if 'death_cross' in locals() else None
            }
        }
    
    def _analyze_cross_asset_correlations(self):
        """Analyze correlations with other asset classes for enhanced prediction"""
        try:
            # Get data for key cross-asset indicators using yfinance
            tickers = {
                'treasury_10y': '^TNX',  # 10-year Treasury yield
                'treasury_2y': '^TWO',   # 2-year Treasury yield
                'gold': 'GC=F',          # Gold futures
                'oil': 'CL=F',           # Crude Oil futures
                'usd': 'DX=F',           # US Dollar Index
                'high_yield': 'HYG',     # High Yield Corporate Bond ETF
                'investment_grade': 'LQD', # Investment Grade Corporate Bond ETF
                'vix': '^VIX'            # Volatility Index
            }
            
            # Default period - try to match length of stock data
            stock_data_length = len(self.stock_data)
            if stock_data_length > 365:
                period = '2y'
            elif stock_data_length > 180:
                period = '1y'
            elif stock_data_length > 90:
                period = '6mo'
            else:
                period = '3mo'
            
            # Get cross-asset data
            asset_data = {}
            for name, ticker in tickers.items():
                try:
                    data = yf.download(ticker, period=period, interval='1d', progress=False)
                    if not data.empty:
                        asset_data[name] = data['Close']
                except Exception as e:
                    print(f"Error fetching {name} data: {e}")
            
            # Reindex the stock data for alignment
            stock_close = pd.Series(self.stock_data['Close'].values, index=self.stock_data.index)
            
            # Calculate correlations and recent changes
            correlations = {}
            changes = {}
            
            for name, data in asset_data.items():
                if len(data) > 20:
                    # Ensure date alignment
                    common_dates = stock_close.index.intersection(data.index)
                    if len(common_dates) > 20:
                        stock_aligned = stock_close.loc[common_dates]
                        asset_aligned = data.loc[common_dates]
                        
                        # Calculate correlation (last 60 days or all if shorter)
                        days = min(60, len(common_dates))
                        corr = stock_aligned[-days:].pct_change().corr(asset_aligned[-days:].pct_change())
                        correlations[name] = corr
                        
                        # Calculate recent change in asset (last 20 days)
                        change = (asset_aligned[-1] / asset_aligned[-20]) - 1 if len(asset_aligned) >= 20 else 0
                        changes[name] = change
            
            # Derive key metrics and signals
            signals = {}
            
            # Yield curve analysis
            if 'treasury_10y' in asset_data and 'treasury_2y' in asset_data:
                yield_curve = (asset_data['treasury_10y'] - asset_data['treasury_2y']).iloc[-1]
                signals['yield_curve'] = yield_curve
                signals['yield_curve_inversion'] = yield_curve < 0
            
            # Credit spread analysis
            if 'high_yield' in asset_data and 'investment_grade' in asset_data:
                hy_price_change = changes.get('high_yield', 0)
                ig_price_change = changes.get('investment_grade', 0)
                credit_spread_change = ig_price_change - hy_price_change  # Widening spread is negative
                signals['credit_spread_widening'] = credit_spread_change > 0.01
                signals['credit_spread_narrowing'] = credit_spread_change < -0.01
            
            # Dollar strength
            if 'usd' in changes:
                signals['dollar_strengthening'] = changes['usd'] > 0.02
                signals['dollar_weakening'] = changes['usd'] < -0.02
            
            # Safe haven demand
            if 'gold' in changes and 'vix' in changes:
                signals['risk_off'] = changes['gold'] > 0.03 or changes['vix'] > 0.2
                signals['risk_on'] = changes['gold'] < -0.03 and changes['vix'] < -0.2
            
            # Calculate impact score on stock based on correlations and changes
            impact_score = 0
            
            # Weights for different assets
            weights = {
                'treasury_10y': -0.3,  # Negative because higher yields generally bad for stocks
                'treasury_2y': -0.2,
                'gold': 0.1,
                'oil': 0.1,
                'usd': -0.1,  # Stronger dollar often negative for stocks
                'high_yield': 0.3,  # High yield bonds often correlate with stocks
                'investment_grade': 0.2,
                'vix': -0.3  # Higher VIX is negative
            }
            
            for name, change in changes.items():
                # Direction matters - multiply by correlation
                corr = correlations.get(name, 0)
                weight = weights.get(name, 0)
                
                # If correlation is strongly positive or negative, factor it in
                if abs(corr) > 0.3:
                    impact = change * weight * (corr / abs(corr))  # Preserve direction
                else:
                    impact = change * weight
                
                impact_score += impact
            
            # Cap the impact score
            impact_score = max(-1, min(1, impact_score * 5))  # Scale for impact
            
            return {
                'correlations': correlations,
                'changes': changes,
                'signals': signals,
                'impact_score': impact_score
            }
            
        except Exception as e:
            print(f"Error in cross-asset correlation analysis: {e}")
            return {
                'correlations': {},
                'changes': {},
                'signals': {},
                'impact_score': 0
            }
    
    def _adjust_prediction_weights(self):
        """Dynamically adjust prediction weights based on market conditions"""
        # Start with base weights
        base_weights = {
            'random_forest': 0.25,
            'boosting': 0.20,
            'linear': 0.10,
            'technical': 0.15,
            'sentiment': 0.10,
            'lstm': 0.20
        }
        
        # Get current market regime
        regime_data = self._detect_market_regime(self.stock_data)
        regime = regime_data['regime']
        
        # Get cross-asset correlations
        cross_asset_data = self._analyze_cross_asset_correlations()
        
        # Adjust weights based on market regime
        regime_adjustments = {
            'Stable Bullish': {
                'random_forest': 0.05,
                'boosting': 0.05,
                'linear': 0.05,
                'technical': 0,
                'sentiment': -0.05,
                'lstm': -0.10
            },
            'Volatile Bullish': {
                'random_forest': -0.05,
                'boosting': 0.05,
                'linear': -0.05,
                'technical': 0.05,
                'sentiment': 0.05,
                'lstm': -0.05
            },
            'Range Bound': {
                'random_forest': -0.05,
                'boosting': -0.05,
                'linear': 0.10,
                'technical': 0.10,
                'sentiment': -0.05,
                'lstm': -0.05
            },
            'Volatile Bearish': {
                'random_forest': -0.10,
                'boosting': 0.05,
                'linear': -0.05,
                'technical': 0.10,
                'sentiment': 0.10,
                'lstm': -0.10
            },
            'Stable Bearish': {
                'random_forest': -0.05,
                'boosting': 0.05,
                'linear': 0.05,
                'technical': 0,
                'sentiment': 0.05,
                'lstm': -0.10
            }
        }
        
        # Get adjustments for current regime
        current_adjustments = regime_adjustments.get(regime, {})
        
        # Apply regime adjustments
        adjusted_weights = {k: base_weights.get(k, 0) + current_adjustments.get(k, 0) 
                           for k in base_weights.keys()}
        
        # Adjust for specific market signals
        
        # If risk-off environment, increase technical weight
        if cross_asset_data['signals'].get('risk_off', False):
            adjusted_weights['technical'] += 0.05
            adjusted_weights['sentiment'] += 0.05
            adjusted_weights['random_forest'] -= 0.05
            adjusted_weights['lstm'] -= 0.05
        
        # If credit spreads widening, increase sentiment weight
        if cross_asset_data['signals'].get('credit_spread_widening', False):
            adjusted_weights['sentiment'] += 0.05
            adjusted_weights['boosting'] += 0.05
            adjusted_weights['linear'] -= 0.05
            adjusted_weights['lstm'] -= 0.05
        
        # Adjust based on model performance if available
        if hasattr(self, 'models') and self.models:
            performance_weights = {}
            total_accuracy = 0
            
            for model_name, model_info in self.models.items():
                if 'accuracy' in model_info and model_info['accuracy'] > 0:
                    performance_weights[model_name] = model_info['accuracy']
                    total_accuracy += model_info['accuracy']
            
            if total_accuracy > 0:
                # Allocate 50% of weight based on performance
                for model_name in adjusted_weights:
                    if model_name in performance_weights:
                        # 50% from regime adjustment, 50% from performance
                        adjusted_weights[model_name] = (
                            0.5 * adjusted_weights[model_name] + 
                            0.5 * (performance_weights[model_name] / total_accuracy)
                        )
        
        # Ensure no negative weights
        adjusted_weights = {k: max(0.01, v) for k, v in adjusted_weights.items()}
        
        # Normalize to sum to 1
        weight_sum = sum(adjusted_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in adjusted_weights.items()}
        
        return normalized_weights
    
    def _analyze_sector_correlations(self):
        """Analyze correlations with sector and peers"""
        try:
            # Get company info
            stock = yf.Ticker(self.ticker)
            sector = stock.info.get('sector', None)
            
            if not sector:
                return {'correlation_score': 0, 'relative_strength': 1.0}
            
            # Map sector to ETF (enhanced with more sectors)
            sector_etfs = {
                'Technology': 'XLK',
                'Information Technology': 'XLK',
                'Healthcare': 'XLV',
                'Health Care': 'XLV',
                'Consumer Cyclical': 'XLY',
                'Consumer Discretionary': 'XLY',
                'Financial Services': 'XLF',
                'Financials': 'XLF',
                'Communication Services': 'XLC',
                'Industrials': 'XLI',
                'Consumer Defensive': 'XLP',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Basic Materials': 'XLB',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Utilities': 'XLU'
            }
            
            sector_etf = sector_etfs.get(sector, 'SPY')  # Default to S&P 500
            
            # Get sector ETF data
            sector_data = yf.download(sector_etf, period='6mo')['Close']
            
            # Calculate correlation
            company_data = self.stock_data['Close']
            correlation = company_data.pct_change().corr(sector_data.pct_change())
            
            # Calculate relative strength
            company_return = (company_data.iloc[-1] / company_data.iloc[0]) - 1
            sector_return = (sector_data.iloc[-1] / sector_data.iloc[0]) - 1
            
            if sector_return != 0:
                relative_strength = company_return / sector_return
            else:
                relative_strength = 1.0
                
            # Get peer tickers (same sector)
            peer_tickers = []
            try:
                # In a production environment, you would use a proper API
                # This is a simplified approach
                if sector in ['Technology', 'Information Technology']:
                    peer_tickers = ['MSFT', 'AAPL', 'NVDA', 'ADBE', 'CRM']
                elif sector in ['Healthcare', 'Health Care']:
                    peer_tickers = ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK']
                elif sector in ['Financials', 'Financial Services']:
                    peer_tickers = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
                elif sector in ['Consumer Cyclical', 'Consumer Discretionary']:
                    peer_tickers = ['AMZN', 'HD', 'NKE', 'SBUX', 'MCD']
                else:
                    peer_tickers = []
                    
                # Remove the current ticker from peers if present
                if self.ticker in peer_tickers:
                    peer_tickers.remove(self.ticker)
            except:
                peer_tickers = []
                
            # Calculate peer correlation if available
            peer_correlation = 0
            if peer_tickers:
                try:
                    # Get peer data
                    peers_data = yf.download(' '.join(peer_tickers), period='6mo')['Close']
                    
                    # Calculate correlation with each peer
                    correlations = []
                    for peer in peer_tickers:
                        if peer in peers_data.columns:
                            corr = company_data.pct_change().corr(peers_data[peer].pct_change())
                            if not np.isnan(corr):
                                correlations.append(corr)
                    
                    if correlations:
                        peer_correlation = np.mean(correlations)
                except Exception as e:
                    print(f"Error calculating peer correlation: {e}")
                    peer_correlation = 0
                
            return {
                'correlation_score': correlation,
                'relative_strength': relative_strength,
                'sector': sector,
                'sector_etf': sector_etf,
                'peer_correlation': peer_correlation,
                'peer_tickers': peer_tickers
            }
            
        except Exception as e:
            print(f"Error in sector correlation analysis: {e}")
            return {'correlation_score': 0, 'relative_strength': 1.0}
    
    def _comprehensive_sentiment_analysis(self):
        """Enhanced sentiment analysis (using base sentiment with adjustments)"""
        # In production, would integrate social media, news API, etc.
        # For now, we'll use a simulation approach based on the base sentiment
        
        base_score = self.base_sentiment_score
        
        # Simulate additional sentiment sources
        simulated_sources = {
            'news': base_score + np.random.normal(0, 0.1),
            'social_media': base_score + np.random.normal(0, 0.2),
            'analyst_ratings': base_score + np.random.normal(0, 0.15),
            'options_sentiment': base_score + np.random.normal(0, 0.3),
            'insider_activity': base_score + np.random.normal(0, 0.25)
        }
        
        # Weight the sources
        composite_score = (
            0.3 * base_score +
            0.2 * simulated_sources['news'] +
            0.15 * simulated_sources['social_media'] +
            0.15 * simulated_sources['analyst_ratings'] +
            0.1 * simulated_sources['options_sentiment'] +
            0.1 * simulated_sources['insider_activity']
        )
        
        # Ensure in range [-1, 1]
        composite_score = max(-1, min(1, composite_score))
        
        return {
            'composite_score': composite_score,
            'source_scores': simulated_sources
        }
    
    def _incorporate_macro_indicators(self):
        """Incorporate macroeconomic indicators"""
        # In production, would use real economic data APIs
        # For now, using the base economic score with adjustments
        
        base_score = self.base_economic_score
        
        # Simulate some macro factors
        simulated_factors = {
            'interest_rates': np.random.normal(0, 0.1),
            'inflation': np.random.normal(0, 0.1),
            'gdp_growth': np.random.normal(0, 0.1),
            'unemployment': np.random.normal(0, 0.1),
            'consumer_confidence': np.random.normal(0, 0.1),
            'manufacturing_pmi': np.random.normal(0, 0.1),
            'housing_market': np.random.normal(0, 0.1)
        }
        
        # Calculate sector sensitivity (would be more sophisticated in production)
        stock = yf.Ticker(self.ticker)
        sector = stock.info.get('sector', 'Unknown')
        
        # Enhanced sector sensitivity map with more factors
        sector_sensitivities = {
            'Technology': {
                'interest_rates': -0.3, 
                'inflation': -0.1,
                'gdp_growth': 0.2,
                'consumer_confidence': 0.2
            },
            'Financial Services': {
                'interest_rates': 0.3, 
                'inflation': -0.1,
                'gdp_growth': 0.1,
                'unemployment': -0.2
            },
            'Consumer Cyclical': {
                'interest_rates': -0.2, 
                'gdp_growth': 0.3,
                'consumer_confidence': 0.3,
                'unemployment': -0.2
            },
            'Healthcare': {
                'interest_rates': -0.1, 
                'inflation': 0.0,
                'gdp_growth': 0.1
            },
            'Energy': {
                'interest_rates': -0.1, 
                'inflation': 0.2,
                'gdp_growth': 0.2,
                'manufacturing_pmi': 0.2
            },
            'Utilities': {
                'interest_rates': -0.4, 
                'inflation': -0.2,
                'gdp_growth': 0.0
            },
            'Real Estate': {
                'interest_rates': -0.4,
                'inflation': 0.1,
                'housing_market': 0.4,
                'unemployment': -0.2
            },
            'Consumer Defensive': {
                'interest_rates': -0.1,
                'inflation': -0.2,
                'gdp_growth': 0.1,
                'unemployment': -0.1
            },
            'Industrials': {
                'interest_rates': -0.2,
                'manufacturing_pmi': 0.3,
                'gdp_growth': 0.2
            },
            'Basic Materials': {
                'interest_rates': -0.1,
                'inflation': 0.2,
                'manufacturing_pmi': 0.3,
                'gdp_growth': 0.2
            }
        }
        
        # Get sensitivities for company sector (default to average sensitivity)
        default_sensitivities = {
            'interest_rates': -0.2, 
            'inflation': -0.1, 
            'gdp_growth': 0.15,
            'unemployment': -0.1,
            'consumer_confidence': 0.1
        }
        
        sensitivities = sector_sensitivities.get(sector, default_sensitivities)
        
        # Calculate macro impact
        macro_impact = 0
        factors_used = 0
        
        for factor, value in simulated_factors.items():
            if factor in sensitivities:
                macro_impact += value * sensitivities[factor]
                factors_used += 1
        
        # Normalize by number of factors used
        if factors_used > 0:
            macro_impact = macro_impact / factors_used
        
        # Adjust base score with macro impact
        adjusted_score = base_score + macro_impact
        
        # Ensure in range [-1, 1]
        adjusted_score = max(-1, min(1, adjusted_score))
        
        return {
            'macro_score': adjusted_score,
            'macro_impact': macro_impact,
            'factors': simulated_factors,
            'sensitivities': sensitivities
        }
    
    def _detect_technical_patterns(self, df):
        """Detect technical patterns"""
        patterns = {}
        pattern_score = 0
        
        # Simple pattern detection (for demonstration)
        # In production, would use more sophisticated pattern recognition
        
        # Check for potential bullish/bearish patterns
        close = df['Close'].values
        high = df['High'].values if 'High' in df.columns else close
        low = df['Low'].values if 'Low' in df.columns else close
        volume = df['Volume'].values if 'Volume' in df.columns else np.ones_like(close)
        
        n = len(close)
        
        if n < 20:
            return {'pattern_score': 0}
        # Higher highs and higher lows (uptrend)
        if (close[-1] > close[-5] > close[-10]) and (low[-1] > low[-5] > low[-10]):
            patterns['uptrend'] = True
            pattern_score += 0.2
        
        # Lower highs and lower lows (downtrend)
        if (close[-1] < close[-5] < close[-10]) and (high[-1] < high[-5] < high[-10]):
            patterns['downtrend'] = True
            pattern_score -= 0.2
            
        # Recent reversal - bullish
        if (close[-3] < close[-2] < close[-1]) and (close[-6] > close[-5] > close[-4]):
            patterns['reversal_bullish'] = True
            pattern_score += 0.15
            
        # Recent reversal - bearish
        if (close[-3] > close[-2] > close[-1]) and (close[-6] < close[-5] < close[-4]):
            patterns['reversal_bearish'] = True
            pattern_score -= 0.15
        
        # Double top
        if n > 40:
            recent_peak = max(close[-20:])
            peak_idx = np.where(close[-20:] == recent_peak)[0][0]
            peak_idx = len(close) - 20 + peak_idx
            
            # Look for previous similar peak
            prev_window = close[peak_idx-30:peak_idx-5]
            prev_peak = max(prev_window)
            prev_peak_idx = np.where(prev_window == prev_peak)[0][0] + peak_idx - 30
            
            # If peaks are similar and there's a trough between them
            if abs(recent_peak - prev_peak) / recent_peak < 0.05:
                min_between = min(close[prev_peak_idx:peak_idx])
                if min_between < min(recent_peak, prev_peak) * 0.95:
                    patterns['double_top'] = True
                    pattern_score -= 0.2
        
        # Double bottom
        if n > 40:
            recent_trough = min(close[-20:])
            trough_idx = np.where(close[-20:] == recent_trough)[0][0]
            trough_idx = len(close) - 20 + trough_idx
            
            # Look for previous similar trough
            prev_window = close[trough_idx-30:trough_idx-5]
            prev_trough = min(prev_window)
            prev_trough_idx = np.where(prev_window == prev_trough)[0][0] + trough_idx - 30
            
            # If troughs are similar and there's a peak between them
            if abs(recent_trough - prev_trough) / recent_trough < 0.05:
                max_between = max(close[prev_trough_idx:trough_idx])
                if max_between > max(recent_trough, prev_trough) * 1.05:
                    patterns['double_bottom'] = True
                    pattern_score += 0.2
        
        # Head and shoulders (bearish)
        if n > 60:
            try:
                # Find three recent peaks
                window = close[-60:]
                peaks = []
                for i in range(1, len(window)-1):
                    if window[i] > window[i-1] and window[i] > window[i+1]:
                        peaks.append((i, window[i]))
                
                # Need at least 3 peaks
                if len(peaks) >= 3:
                    # Get the highest 3 peaks
                    peaks.sort(key=lambda x: x[1], reverse=True)
                    top_peaks = sorted(peaks[:3], key=lambda x: x[0])
                    
                    # Check if middle peak is highest
                    if len(top_peaks) == 3 and top_peaks[1][1] > top_peaks[0][1] and top_peaks[1][1] > top_peaks[2][1]:
                        # Check if outer peaks are at similar levels
                        if abs(top_peaks[0][1] - top_peaks[2][1]) / top_peaks[0][1] < 0.1:
                            patterns['head_and_shoulders'] = True
                            pattern_score -= 0.25
            except:
                pass
        
        # Bullish engulfing
        if n > 5 and 'Open' in df.columns:
            opens = df['Open'].values
            if opens[-2] > close[-2] and opens[-1] < close[-1] and opens[-1] <= close[-2] and close[-1] > opens[-2]:
                patterns['bullish_engulfing'] = True
                pattern_score += 0.15
        
        # Bearish engulfing
        if n > 5 and 'Open' in df.columns:
            opens = df['Open'].values
            if opens[-2] < close[-2] and opens[-1] > close[-1] and opens[-1] >= close[-2] and close[-1] < opens[-2]:
                patterns['bearish_engulfing'] = True
                pattern_score -= 0.15
        
        # Volume confirmation
        if patterns.get('uptrend', False) and volume[-1] > np.mean(volume[-20:]):
            pattern_score += 0.1
        
        if patterns.get('downtrend', False) and volume[-1] > np.mean(volume[-20:]):
            pattern_score -= 0.1
        
        return {
            'detected_patterns': patterns,
            'pattern_score': pattern_score
        }
    
    def _analyze_volume_profile(self, df):
        """Analyze volume profile"""
        # Enhanced volume analysis
        if 'Volume' not in df.columns or len(df) < 20:
            return {'score': 0}
            
        try:
            # Check if current volume is above average
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price movement with volume confirmation
            price_change = df['Close'].pct_change().iloc[-1]
            
            # Volume price trend analysis
            vptrend_score = 0
            
            # Rising prices should have rising volume
            if price_change > 0:
                if volume_ratio > 1.2:  # Strong volume confirms uptrend
                    vptrend_score = 0.2
                elif volume_ratio < 0.8:  # Low volume suggests weak uptrend
                    vptrend_score = 0.05
            # Falling prices with high volume is bearish confirmation
            elif price_change < 0:
                if volume_ratio > 1.2:  # Strong volume confirms downtrend
                    vptrend_score = -0.2
                elif volume_ratio < 0.8:  # Low volume suggests weak downtrend
                    vptrend_score = -0.05
            
            # Volume divergence
            # Look for divergence between price trend and volume trend
            price_trend = np.mean([df['Close'].iloc[-1] / df['Close'].iloc[-i] - 1 for i in [5, 10, 15]])
            volume_trend = np.mean([df['Volume'].iloc[-1] / df['Volume'].iloc[-i] - 1 for i in [5, 10, 15]])
            
            # Positive divergence: price falling but volume declining (bullish)
            if price_trend < 0 and volume_trend < 0:
                vptrend_score += 0.1
            # Negative divergence: price rising but volume declining (bearish)
            elif price_trend > 0 and volume_trend < 0:
                vptrend_score -= 0.1
            
            # Advanced volume analysis: On-Balance Volume (OBV)
            obv = 0
            obv_values = []
            
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv += df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv -= df['Volume'].iloc[i]
                obv_values.append(obv)
            
            # Check if OBV trend confirms price trend
            if len(obv_values) > 20:
                obv_trend = (obv_values[-1] - obv_values[-20]) / abs(obv_values[-20]) if obv_values[-20] != 0 else 0
                price_trend_20 = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1)
                
                # OBV and price trends are aligned
                if (obv_trend > 0 and price_trend_20 > 0) or (obv_trend < 0 and price_trend_20 < 0):
                    vptrend_score += 0.1
                # OBV and price trends are diverging
                else:
                    vptrend_score -= 0.1
            
            return {
                'score': vptrend_score,
                'volume_ratio': volume_ratio,
                'obv_trend': obv_trend if 'obv_trend' in locals() else 0,
                'interpretation': 'Volume confirms trend' if vptrend_score > 0.1 else 
                                'Volume contradicts trend' if vptrend_score < -0.1 else
                                'Neutral volume'
            }
            
        except Exception as e:
            print(f"Error in volume profile analysis: {e}")
            return {'score': 0}
    
    def _multi_timeframe_analysis(self):
        """Analyze multiple timeframes"""
        try:
            # Get weekly and monthly data
            ticker_obj = yf.Ticker(self.ticker)
            weekly = ticker_obj.history(period='1y', interval='1wk')
            monthly = ticker_obj.history(period='3y', interval='1mo')
            
            # Determine trends across timeframes
            daily_trend = 1 if self.stock_data['Close'].iloc[-1] > self.stock_data['Close'].iloc[-10] else -1
            weekly_trend = 1 if weekly['Close'].iloc[-1] > weekly['Close'].iloc[-4] else -1
            monthly_trend = 1 if monthly['Close'].iloc[-1] > monthly['Close'].iloc[-3] else -1
            
            # Calculate alignment score (-1 to 1)
            alignment = (daily_trend + weekly_trend + monthly_trend) / 3
            
            # Calculate RSI across timeframes
            daily_rsi = None
            weekly_rsi = None
            monthly_rsi = None
            
            if 'RSI' in self.stock_data.columns:
                daily_rsi = self.stock_data['RSI'].iloc[-1]
            
            # Calculate RSI for weekly and monthly if not available
            if len(weekly) > 14:
                delta = weekly['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                weekly_rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            if len(monthly) > 14:
                delta = monthly['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                monthly_rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Identify support and resistance levels
            support_levels = []
            resistance_levels = []
            
            # Daily support/resistance
            daily_close = self.stock_data['Close'].iloc[-1]
            daily_lows = self.stock_data['Low'] if 'Low' in self.stock_data.columns else self.stock_data['Close']
            daily_highs = self.stock_data['High'] if 'High' in self.stock_data.columns else self.stock_data['Close']
            
            # Find major support levels
            for i in range(20, min(120, len(daily_lows))):
                if all(daily_lows.iloc[-i] < daily_lows.iloc[-i+j] for j in range(-5, 6) if j != 0):
                    level = daily_lows.iloc[-i]
                    if daily_close > level:  # Support level is below current price
                        support_levels.append(level)
            
            # Find major resistance levels
            for i in range(20, min(120, len(daily_highs))):
                if all(daily_highs.iloc[-i] > daily_highs.iloc[-i+j] for j in range(-5, 6) if j != 0):
                    level = daily_highs.iloc[-i]
                    if daily_close < level:  # Resistance level is above current price
                        resistance_levels.append(level)
            
            # Take the 3 closest levels
            support_levels = sorted(support_levels, reverse=True)[:3] if support_levels else []
            resistance_levels = sorted(resistance_levels)[:3] if resistance_levels else []
            
            return {
                'timeframe_alignment': alignment,
                'daily_trend': daily_trend,
                'weekly_trend': weekly_trend,
                'monthly_trend': monthly_trend,
                'daily_rsi': daily_rsi,
                'weekly_rsi': weekly_rsi,
                'monthly_rsi': monthly_rsi,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            print(f"Error in multi-timeframe analysis: {e}")
            return {'timeframe_alignment': 0}
    
    def _analyze_options_market(self):
        """Analyze options market sentiment"""
        # In production would use real options data
        # For now, returning simulated data based on price momentum
        
        recent_momentum = 0
        if len(self.stock_data) > 20:
            recent_momentum = (self.stock_data['Close'].iloc[-1] / self.stock_data['Close'].iloc[-20] - 1)
        
        # Simulate put/call ratio based on recent momentum
        # High put/call ratio (>1) is bearish, low (<0.7) is bullish
        put_call_ratio = 1.0 - (recent_momentum * 2)
        put_call_ratio = max(0.3, min(1.7, put_call_ratio))
        
        # Simulate options sentiment
        if put_call_ratio > 1.0:
            options_sentiment = -0.3  # Bearish sentiment
        elif put_call_ratio < 0.7:
            options_sentiment = 0.3   # Bullish sentiment
        else:
            options_sentiment = 0.0   # Neutral
            
        # Add random noise
        options_sentiment += np.random.normal(0, 0.1)
        options_sentiment = max(-1, min(1, options_sentiment))
        
        return {
            'options_sentiment': options_sentiment,
            'put_call_ratio': put_call_ratio
        }
    
    def _detect_elliott_wave_patterns(self, df):
        """Detect Elliott Wave patterns (simplified implementation)"""
        # This is a simplified approximation - a real implementation would be much more complex
        if len(df) < 30:
            return 0
        
        # Get price data
        prices = df['Close'].values
        
        # Look for potential 5-wave pattern
        # Check for alternating up and down movements
        wave_score = 0
        
        # Detect recent pivots
        pivots = []
        window_size = 5
        
        for i in range(window_size, len(prices) - window_size):
            # Local maximum
            if all(prices[i] > prices[i-j] for j in range(1, window_size+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, window_size+1)):
                pivots.append((i, 1))  # 1 for peak
            
            # Local minimum
            if all(prices[i] < prices[i-j] for j in range(1, window_size+1)) and \
               all(prices[i] < prices[i+j] for j in range(1, window_size+1)):
                pivots.append((i, -1))  # -1 for trough
        
        # Sort pivots by index
        pivots.sort(key=lambda x: x[0])
        
        # Need at least 5 pivots for a complete Elliott wave
        if len(pivots) >= 5:
            # Get most recent pivots
            recent_pivots = pivots[-9:]
            
            # Check for 5-wave impulse pattern (1-up, 2-down, 3-up, 4-down, 5-up)
            try:
                wave_pattern = [p[1] for p in recent_pivots]
                
                # Look for 5-3 wave pattern (5 up-down cycles followed by 3 correction cycles)
                patterns = []
                
                # Check various sequences
                for i in range(len(wave_pattern) - 4):
                    if wave_pattern[i:i+5] == [1, -1, 1, -1, 1]:  # Impulse up
                        wave_score = 0.2
                        break
                    elif wave_pattern[i:i+5] == [-1, 1, -1, 1, -1]:  # Impulse down
                        wave_score = -0.2
                        break
                
                # Check if price is in correction after 5-wave move
                if len(recent_pivots) >= 7:
                    if wave_pattern[:5] == [1, -1, 1, -1, 1] and wave_pattern[5:7] == [-1, 1]:
                        wave_score = -0.1  # Correction after bullish impulse
                    elif wave_pattern[:5] == [-1, 1, -1, 1, -1] and wave_pattern[5:7] == [1, -1]:
                        wave_score = 0.1   # Correction after bearish impulse
            except:
                wave_score = 0
        
        return wave_score
    
    def _calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement levels"""
        if len(df) < 20:
            return {}
        
        # Get high and low prices over the lookback period
        lookback = min(100, len(df))
        
        # Find significant high and low
        high = df['High'].iloc[-lookback:].max() if 'High' in df.columns else df['Close'].iloc[-lookback:].max()
        low = df['Low'].iloc[-lookback:].min() if 'Low' in df.columns else df['Close'].iloc[-lookback:].min()
        
        # Calculate direction (uptrend or downtrend)
        current = df['Close'].iloc[-1]
        start = df['Close'].iloc[-lookback]
        uptrend = current > start
        
        # Calculate Fibonacci levels
        diff = high - low
        
        if uptrend:
            # For uptrend, retracement is measured from low to high
            fib_levels = {
                '0.0': low,
                '0.236': low + 0.236 * diff,
                '0.382': low + 0.382 * diff,
                '0.5': low + 0.5 * diff, 
                '0.618': low + 0.618 * diff,
                '0.786': low + 0.786 * diff,
                '1.0': high,
                # Extension levels
                '1.272': high + 0.272 * diff,
                '1.618': high + 0.618 * diff
            }
        else:
            # For downtrend, retracement is measured from high to low
            fib_levels = {
                '0.0': high,
                '0.236': high - 0.236 * diff,
                '0.382': high - 0.382 * diff,
                '0.5': high - 0.5 * diff,
                '0.618': high - 0.618 * diff,
                '0.786': high - 0.786 * diff,
                '1.0': low,
                # Extension levels
                '1.272': low - 0.272 * diff,
                '1.618': low - 0.618 * diff
            }
        
        return fib_levels
    
    def train_models(self):
        """Train multiple prediction models"""
        # Prepare data if not already done
        if not hasattr(self, 'prepared_data'):
            self.prepare_data()
            
        df = self.prepared_data
        
        # Select features for modeling
        features = [
            'Prev_Close', '50_Day_MA', '200_Day_MA', 'Volume_Change', 
            'Return_Volatility', 'Market_Regime', 'Enhanced_Sentiment',
            'Sector_Correlation', 'Sector_Relative_Strength', 'Macro_Score',
            'Pattern_Score', 'RSI', 'MACD', 'BB_Width', 'Volume_Profile_Score',
            'Timeframe_Alignment', 'Momentum_1D', 'Momentum_5D', 'Wave_Pattern',
            'Cross_Asset_Impact'  # New feature for cross asset correlations
        ]
        
        # Make sure all features exist
        existing_features = [f for f in features if f in df.columns]
        
        if len(existing_features) < 5:
            raise ValueError("Not enough features available for prediction")
            
        # Prepare the dataset
        X = df[existing_features].dropna()
        y = df['Close'].loc[X.index]
        
        # Make sure we have enough data
        if len(X) < 10:
            raise ValueError("Not enough data points for training")
            
        # Split data
        if len(X) > 30:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            # For small datasets
            X_train, X_test = X.iloc[:-5], X.iloc[-5:]
            y_train, y_test = y.iloc[:-5], y.iloc[-5:]
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = {
            'model': rf_model,
            'features': existing_features,
            'accuracy': rf_model.score(X_test, y_test)
        }
        
        # Train Boosting Model - XGBoost if available, otherwise use GradientBoosting
        if XGBOOST_AVAILABLE:
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    random_state=42
                )
                xgb_model.fit(X_train, y_train)
                self.models['boosting'] = {
                    'model': xgb_model,
                    'features': existing_features,
                    'accuracy': xgb_model.score(X_test, y_test)
                }
            except:
                # Fallback to GradientBoosting if XGBoost fails
                gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gb_model.fit(X_train, y_train)
                self.models['boosting'] = {
                    'model': gb_model,
                    'features': existing_features,
                    'accuracy': gb_model.score(X_test, y_test)
                }
        else:
            # Use GradientBoosting as alternative to XGBoost
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train)
            self.models['boosting'] = {
                'model': gb_model,
                'features': existing_features,
                'accuracy': gb_model.score(X_test, y_test)
            }
        
        # Train Linear Regression (simpler model)
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        self.models['linear'] = {
            'model': linear_model,
            'features': existing_features,
            'accuracy': linear_model.score(X_test, y_test)
        }
        
        # Train LSTM model if TensorFlow is available and we have enough data
        if TENSORFLOW_AVAILABLE and len(df) >= 60:
            try:
                self.add_lstm_model()
            except Exception as e:
                print(f"Error adding LSTM model: {e}")
        
        # Adjust prediction weights based on market conditions
        self.prediction_weights = self._adjust_prediction_weights()
            
        return self.models
    
    def add_lstm_model(self):
        """Add an LSTM neural network to the model ensemble"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available, skipping LSTM model")
            return
            
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            # Prepare data in sequences
            sequence_length = 20
            X_sequences, y_sequences = self._prepare_sequences(sequence_length)
            
            if len(X_sequences) < 30:
                print("Not enough data for LSTM model")
                return
            
            # Split data
            split_idx = int(len(X_sequences) * 0.8)
            X_train = X_sequences[:split_idx]
            X_test = X_sequences[split_idx:]
            y_train = y_sequences[:split_idx]
            y_test = y_sequences[split_idx:]
            
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Evaluate
            mse = model.evaluate(X_test, y_test, verbose=0)
            accuracy = 1 / (1 + mse)  # Convert MSE to accuracy-like metric
            
            # Add to models dictionary
            self.models['lstm'] = {
                'model': model,
                'features': 'sequence',
                'accuracy': accuracy
            }
            
            # Update prediction weights
            self.prediction_weights['lstm'] = 0.25
            
            # Normalize weights
            weight_sum = sum(self.prediction_weights.values())
            self.prediction_weights = {k: v/weight_sum for k, v in self.prediction_weights.items()}
            
        except Exception as e:
            print(f"Error creating LSTM model: {e}")
    
    def _prepare_sequences(self, sequence_length):
        """Prepare data for sequence models like LSTM"""
        df = self.prepared_data
        
        # Select features for sequence
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Width']
        
        # Ensure all features exist
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 3:
            return np.array([]), np.array([])
        
        # Create sequences
        X_sequences = []
        y_values = []
        
        for i in range(len(df) - sequence_length):
            X_sequences.append(df[available_features].values[i:i+sequence_length])
            y_values.append(df['Close'].values[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_values)
    
    def predict_future(self, days=7):
        """Make ensemble predictions for future days"""
        # Train models if not already done
        if not self.models:
            self.train_models()
            
        df = self.prepared_data
        last_date = df.index[-1]
        
        # Create output dataframe
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        future_df = pd.DataFrame(index=future_dates)
        future_df['Predicted_Price'] = 0.0
        
        # Current values
        last_close = df['Close'].iloc[-1]
        
        # Make predictions for each day
        prev_close = last_close
        for i, date in enumerate(future_dates):
            # Create feature vector for each model
            model_predictions = {}
            
            # Get predictions from each ML model
            for model_name, model_info in self.models.items():
                # Skip LSTM for now, handle separately
                if model_name == 'lstm':
                    continue
                    
                model = model_info['model']
                features = model_info['features']
                
                # Prepare feature vector
                feature_vector = {}
                for feature in features:
                    if feature == 'Prev_Close':
                        feature_vector[feature] = prev_close
                    elif feature == '50_Day_MA':
                        feature_vector[feature] = df['50_Day_MA'].iloc[-1] if '50_Day_MA' in df.columns else prev_close
                    elif feature == '200_Day_MA':
                        feature_vector[feature] = df['200_Day_MA'].iloc[-1] if '200_Day_MA' in df.columns else prev_close
                    elif feature == 'Volume_Change':
                        feature_vector[feature] = df['Volume_Change'].mean() if 'Volume_Change' in df.columns else 0
                    elif feature == 'Return_Volatility':
                        feature_vector[feature] = df['Return_Volatility'].iloc[-1] if 'Return_Volatility' in df.columns else 0.01
                    elif feature == 'Enhanced_Sentiment':
                        feature_vector[feature] = self._comprehensive_sentiment_analysis().get('composite_score', 0)
                    elif feature == 'Macro_Score':
                        feature_vector[feature] = self._incorporate_macro_indicators().get('macro_score', 0)
                    elif feature == 'Market_Regime':
                        feature_vector[feature] = self._detect_market_regime(df).get('regime_numeric', 0)
                    elif feature == 'Pattern_Score':
                        feature_vector[feature] = self._detect_technical_patterns(df).get('pattern_score', 0)
                    elif feature == 'Cross_Asset_Impact':
                        feature_vector[feature] = self._analyze_cross_asset_correlations().get('impact_score', 0)
                    elif feature in df.columns:
                        feature_vector[feature] = df[feature].iloc[-1]
                    else:
                        feature_vector[feature] = 0
                
                X_pred = pd.DataFrame([feature_vector])
                
                # Predict
                try:
                    prediction = model.predict(X_pred)[0]
                    model_predictions[model_name] = prediction
                except Exception as e:
                    print(f"Error in {model_name} prediction: {e}")
                    model_predictions[model_name] = prev_close
            
            # LSTM prediction if available
            if 'lstm' in self.models:
                try:
                    model = self.models['lstm']['model']
                    
                    # Get recent sequence data
                    sequence_length = 20
                    available_features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Width']
                    available_features = [f for f in available_features if f in df.columns]
                    
                    if len(available_features) >= 3 and len(df) > sequence_length:
                        # Get the most recent sequence
                        recent_sequence = df[available_features].values[-sequence_length:]
                        
                        # If we're beyond the first prediction day, we need to update the sequence
                        if i > 0:
                            # Replace the oldest values with our new predictions and propagate features
                            # This is simplified; in production you'd update all features
                            sequence_copy = recent_sequence.copy()
                            sequence_copy = np.roll(sequence_copy, -1, axis=0)
                            sequence_copy[-1, 0] = prev_close  # Update Close price
                            recent_sequence = sequence_copy
                        
                        # Reshape for LSTM input [samples, time steps, features]
                        X_lstm = np.array([recent_sequence])
                        
                        # Predict
                        lstm_prediction = model.predict(X_lstm, verbose=0)[0][0]
                        model_predictions['lstm'] = lstm_prediction
                except Exception as e:
                    print(f"Error in LSTM prediction: {e}")
                    model_predictions['lstm'] = prev_close
            
            # Technical analysis adjustments
            technical_prediction = prev_close
            
            # RSI-based adjustment
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if rsi < 30:  # Oversold
                    technical_prediction *= 1.01
                elif rsi > 70:  # Overbought
                    technical_prediction *= 0.99
            
            # Pattern-based adjustment
            pattern_score = self._detect_technical_patterns(df).get('pattern_score', 0)
            technical_prediction *= (1 + (pattern_score * 0.02))
            
            # Volume confirmation
            volume_score = self._analyze_volume_profile(df).get('score', 0)
            technical_prediction *= (1 + (volume_score * 0.01))
            
            # Support and resistance levels
            mtf_data = self._multi_timeframe_analysis()
            support_levels = mtf_data.get('support_levels', [])
            resistance_levels = mtf_data.get('resistance_levels', [])
            
            # Check if price is near support or resistance
            if support_levels:
                closest_support = min(support_levels, key=lambda x: abs(prev_close - x))
                if 0.99 * prev_close <= closest_support <= prev_close:
                    technical_prediction *= 1.01  # Potential bounce from support
            
            if resistance_levels:
                closest_resistance = min(resistance_levels, key=lambda x: abs(prev_close - x))
                if prev_close <= closest_resistance <= 1.01 * prev_close:
                    technical_prediction *= 0.99  # Potential rejection at resistance
            
            # Cross-asset impact adjustment
            cross_asset_data = self._analyze_cross_asset_correlations()
            cross_asset_impact = cross_asset_data.get('impact_score', 0)
            technical_prediction *= (1 + (cross_asset_impact * 0.01))
            
            model_predictions['technical'] = technical_prediction
            
            # Sentiment-based prediction
            sentiment_prediction = prev_close
            sentiment_score = self._comprehensive_sentiment_analysis().get('composite_score', 0)
            sentiment_prediction *= (1 + (sentiment_score * 0.02))
            
            model_predictions['sentiment'] = sentiment_prediction
            
            # Add Fibonacci retracement-based adjustments
            fib_levels = self._calculate_fibonacci_levels(df)
            if fib_levels:
                fib_prediction = prev_close
                
                # Find closest Fibonacci level to current price
                levels = [(level, price) for level, price in fib_levels.items()]
                closest_level = min(levels, key=lambda x: abs(prev_close - x[1]))
                
                # Adjust prediction based on Fibonacci level proximity
                if abs(prev_close - closest_level[1]) / prev_close < 0.02:  # Within 2% of a Fibonacci level
                    if closest_level[0] in ['0.0', '0.236', '0.382']:  # Support levels in uptrend
                        fib_prediction *= 1.01  # Likely bounce
                    elif closest_level[0] in ['0.618', '0.786', '1.0']:  # Resistance levels in uptrend
                        fib_prediction *= 0.99  # Likely rejection
                
                model_predictions['fibonacci'] = fib_prediction
                self.prediction_weights['fibonacci'] = 0.1
                
                # Rebalance weights if necessary
                if sum(self.prediction_weights.values()) > 1:
                    weight_sum = sum(self.prediction_weights.values())
                    self.prediction_weights = {k: v/weight_sum for k, v in self.prediction_weights.items()}
            
            # Weighted ensemble prediction
            weighted_prediction = 0
            used_weights = 0
            
            for model_name, prediction in model_predictions.items():
                weight = self.prediction_weights.get(model_name, 0)
                weighted_prediction += prediction * weight
                used_weights += weight
            
            # Normalize if not all models were used
            if used_weights > 0 and used_weights < 1:
                weighted_prediction = weighted_prediction / used_weights
            
            # Add volatility-based noise
            volatility = df['Close'].pct_change().std()
            noise_factor = np.random.normal(0, volatility * last_close * 0.1)
            
            # Final prediction with noise
            final_prediction = weighted_prediction + noise_factor
            
            # Ensure prediction doesn't change too drastically
            max_change = 0.05 * (1 + i * 0.01)  # Allow larger changes further in the future
            if abs(final_prediction - prev_close) / prev_close > max_change:
                direction = 1 if final_prediction > prev_close else -1
                final_prediction = prev_close * (1 + direction * max_change)
            
            # Store prediction
            future_df.loc[date, 'Predicted_Price'] = final_prediction
            
            # Update for next iteration
            prev_close = final_prediction
            
        # Calculate confidence intervals using our adaptive method
        self._calculate_adaptive_confidence_intervals(future_df)
        
        return future_df
    
    def _calculate_adaptive_confidence_intervals(self, predictions):
        """Calculate adaptive confidence intervals"""
        # Calculate historical volatility
        price_std = self.stock_data['Close'].pct_change().std()
        
        # Get market regime data
        regime_data = self._detect_market_regime(self.stock_data)
        
        # Base uncertainty on historical volatility and current regime
        if regime_data['regime'] in ['Volatile Bullish', 'Volatile Bearish']:
            volatility_multiplier = 1.5
        elif regime_data['regime'] in ['Range Bound']:
            volatility_multiplier = 0.8
        else:
            volatility_multiplier = 1.0
            
        # Calculate uncertainty for each prediction
        last_close = self.stock_data['Close'].iloc[-1]
        
        # Bootstrap confidence intervals if we have enough data
        bootstrap_intervals = False
        if len(self.stock_data) > 100 and len(self.models) >= 3:
            try:
                bootstrap_intervals = True
                
                # Create bootstrapped predictions
                bootstrap_samples = 100
                bootstrap_predictions = np.zeros((len(predictions), bootstrap_samples))
                
                for i in range(bootstrap_samples):
                    # Sample with replacement from historical returns
                    historical_returns = self.stock_data['Close'].pct_change().dropna().values
                    sampled_returns = np.random.choice(historical_returns, size=len(predictions), replace=True)
                    
                    # Apply random walks starting from last close
                    bootstrap_predictions[:, i] = last_close * np.cumprod(1 + sampled_returns)
                
                # Calculate percentiles for confidence intervals
                lower_percentile = np.percentile(bootstrap_predictions, 5, axis=1)
                upper_percentile = np.percentile(bootstrap_predictions, 95, axis=1)
                
                # Add uncertainty from model predictions
                for i, date in enumerate(predictions.index):
                    # Combine bootstrap and analytical estimates
                    model_price = predictions.loc[date, 'Predicted_Price']
                    
                    # Weight more toward the models for shorter horizons, more toward bootstrap for longer horizons
                    time_factor = min(1.0, i / 5)  # Increases from 0 to 1 over first 5 days
                    
                    # Weighted combination
                    lower_bound = model_price - (model_price - lower_percentile[i]) * (0.5 + 0.5 * time_factor)
                    upper_bound = model_price + (upper_percentile[i] - model_price) * (0.5 + 0.5 * time_factor)
                    
                    predictions.loc[date, 'Lower_Bound'] = lower_bound
                    predictions.loc[date, 'Upper_Bound'] = upper_bound
                
            except Exception as e:
                print(f"Error in bootstrap confidence intervals: {e}")
                bootstrap_intervals = False
        
        # Fall back to analytical method if bootstrap failed or not enough data
        if not bootstrap_intervals:
            for i, date in enumerate(predictions.index):
                # Time factor - uncertainty increases with prediction distance
                time_factor = 1 + (i * 0.15)
                
                # Cross-asset adjustment - increase uncertainty in volatile market conditions
                cross_asset_data = self._analyze_cross_asset_correlations()
                is_risk_off = cross_asset_data['signals'].get('risk_off', False)
                is_credit_widening = cross_asset_data['signals'].get('credit_spread_widening', False)
                
                cross_asset_multiplier = 1.0
                if is_risk_off:
                    cross_asset_multiplier *= 1.2
                if is_credit_widening:
                    cross_asset_multiplier *= 1.1
                
                # Calculate uncertainty for this prediction
                uncertainty = price_std * last_close * time_factor * volatility_multiplier * cross_asset_multiplier
                
                # Add bounds
                predictions.loc[date, 'Upper_Bound'] = predictions.loc[date, 'Predicted_Price'] + uncertainty
                predictions.loc[date, 'Lower_Bound'] = predictions.loc[date, 'Predicted_Price'] - uncertainty
            
        # Ensure no negative prices
        for date in predictions.index:
            predictions.loc[date, 'Lower_Bound'] = max(0, predictions.loc[date, 'Lower_Bound'])
        
        # Calculate Value at Risk (VaR) at 95% confidence level
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        var_95 = np.percentile(daily_returns, 5)
        
        # Add VaR-based metrics for each prediction
        for date in predictions.index:
            predicted_price = predictions.loc[date, 'Predicted_Price']
            predictions.loc[date, 'VaR_95'] = predicted_price * var_95
            
        # Add prediction confidence score
        for date in predictions.index:
            price = predictions.loc[date, 'Predicted_Price']
            lower = predictions.loc[date, 'Lower_Bound']
            upper = predictions.loc[date, 'Upper_Bound']
            
            # Narrower bounds = higher confidence
            interval_width = (upper - lower) / price
            confidence_score = 1.0 / (1.0 + interval_width)
            predictions.loc[date, 'Confidence'] = confidence_score
        
        return predictions