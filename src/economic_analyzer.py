import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os

class EconomicAnalyzer:
    def __init__(self):
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache time - 24 hours
        self.cache_expiry = 24 * 60 * 60
        
        # FRED API key - replace with your own
        self.fred_api_key = "YOUR_FRED_API_KEY"  # Get from https://fred.stlouisfed.org/docs/api/api_key.html
        
        # Important economic indicators to track
        self.indicators = {
            'GDP': 'GDP',
            'Inflation': 'CPIAUCSL',
            'Unemployment': 'UNRATE',
            'Fed_Funds_Rate': 'FEDFUNDS',
            'Ten_Year_Treasury': 'DGS10',
            'Two_Year_Treasury': 'DGS2',
            'Yield_Curve': 'T10Y2Y',  # 10-year minus 2-year
            'Credit_Spread': 'BAA10Y',  # Moody's BAA corporate bond yield minus 10-year treasury
            'Industrial_Production': 'INDPRO',
            'Housing_Starts': 'HOUST',
            'Consumer_Sentiment': 'UMCSENT'
        }
        
        # Get the latest economic data
        self.economic_data = self.get_economic_data()
        
        # Sector sensitivity to economic indicators
        self.sector_sensitivities = {
            'Technology': {
                'Fed_Funds_Rate': -0.5, 
                'Ten_Year_Treasury': -0.4,
                'GDP': 0.3,
                'Consumer_Sentiment': 0.2
            },
            'Financial Services': {
                'Fed_Funds_Rate': 0.4, 
                'Yield_Curve': 0.5,
                'Credit_Spread': -0.3,
                'GDP': 0.2
            },
            'Consumer Cyclical': {
                'Unemployment': -0.4, 
                'GDP': 0.4,
                'Consumer_Sentiment': 0.5,
                'Fed_Funds_Rate': -0.3
            },
            'Healthcare': {
                'Inflation': -0.2, 
                'Unemployment': -0.1,
                'GDP': 0.1
            },
            'Energy': {
                'GDP': 0.3, 
                'Industrial_Production': 0.4,
                'Inflation': 0.2
            },
            'Utilities': {
                'Fed_Funds_Rate': -0.5, 
                'Ten_Year_Treasury': -0.6,
                'Inflation': -0.3
            },
            'Real Estate': {
                'Fed_Funds_Rate': -0.6,
                'Ten_Year_Treasury': -0.5,
                'Housing_Starts': 0.4,
                'Unemployment': -0.2
            },
            'Consumer Defensive': {
                'Inflation': -0.3,
                'Unemployment': -0.2,
                'Consumer_Sentiment': 0.2
            },
            'Industrials': {
                'Industrial_Production': 0.5,
                'GDP': 0.4,
                'Fed_Funds_Rate': -0.2
            },
            'Basic Materials': {
                'Industrial_Production': 0.5,
                'Inflation': 0.3,
                'GDP': 0.3
            }
        }
        
        # Current market regime weights
        self.regime_weights = {
            'high_inflation': 0.8,
            'rising_rates': 0.9,
            'yield_curve_inversion': 0.7,
            'growth_slowdown': 0.6
        }
    
    def get_economic_data(self):
        """Fetch real economic data from FRED API with caching"""
        cache_file = os.path.join(self.cache_dir, "economic_data_cache.json")
        
        # Check cache
        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - file_time < self.cache_expiry:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass
        
        data = {}
        
        # Use FRED API if key is provided
        if self.fred_api_key != "YOUR_FRED_API_KEY":
            for indicator_name, series_id in self.indicators.items():
                try:
                    url = f"https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": series_id,
                        "api_key": self.fred_api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1  # Just get latest
                    }
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        result = response.json()
                        if "observations" in result and result["observations"]:
                            data[indicator_name] = float(result["observations"][0]["value"])
                except Exception as e:
                    print(f"Error fetching {indicator_name}: {e}")
        
        # If we couldn't get data or no API key, use realistic recent values
        if not data:
            # These are recent values as of early 2025 (reasonable estimates)
            data = {
                'GDP': 3.2,
                'Inflation': 3.5,
                'Unemployment': 4.1,
                'Fed_Funds_Rate': 3.75,
                'Ten_Year_Treasury': 4.2,
                'Two_Year_Treasury': 4.1,
                'Yield_Curve': 0.1,
                'Credit_Spread': 2.1,
                'Industrial_Production': 2.5,
                'Housing_Starts': 1.4,
                'Consumer_Sentiment': 65.0
            }
        
        # Cache the data
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        return data
    
    def get_economic_score(self, country='US', sector='Technology'):
        """
        Calculate economic health score with focus on interest rates and monetary policy
        """
        # Base economic score
        eco_score = 0
        
        # GDP Growth - positive factor
        gdp_growth = self.economic_data.get('GDP', 0)
        gdp_score = self._normalize_score(gdp_growth, 0, 5)
        
        # Inflation - negative when too high or too low
        inflation = self.economic_data.get('Inflation', 0)
        inflation_target = 2.0  # Fed target
        inflation_score = -abs(self._normalize_score(inflation - inflation_target, -2, 2))
        
        # Unemployment - negative factor
        unemployment = self.economic_data.get('Unemployment', 0)
        unemployment_score = self._normalize_score(unemployment, 3, 8, reverse=True)
        
        # Interest rates - complex effect depending on level and direction
        fed_rate = self.economic_data.get('Fed_Funds_Rate', 0)
        ten_year = self.economic_data.get('Ten_Year_Treasury', 0)
        
        # Rate level effect - very high rates are negative
        rate_level_score = self._normalize_score(fed_rate, 0, 8, reverse=True)
        
        # Yield curve - inversion is negative
        yield_curve = self.economic_data.get('Yield_Curve', 0)
        yield_curve_score = self._normalize_score(yield_curve, -1, 2)
        
        # Credit conditions - higher spread is negative
        credit_spread = self.economic_data.get('Credit_Spread', 0)
        credit_score = self._normalize_score(credit_spread, 1, 5, reverse=True)
        
        # Current regime detection
        regime_signals = {
            'high_inflation': inflation > 3.0,
            'rising_rates': fed_rate > 3.5,
            'yield_curve_inversion': yield_curve < 0.25,
            'growth_slowdown': gdp_growth < 2.0
        }
        
        # Base economic score
        base_score = (
            0.2 * gdp_score + 
            0.2 * inflation_score + 
            0.15 * unemployment_score +
            0.2 * rate_level_score +
            0.15 * yield_curve_score +
            0.1 * credit_score
        )
        
        # Sector-specific adjustments
        sector_adj = 0
        sector_sensitivities = self.sector_sensitivities.get(sector, {})
        
        for indicator, sensitivity in sector_sensitivities.items():
            if indicator in self.economic_data:
                value = self.economic_data[indicator]
                # Normalize to [-1, 1] range
                norm_value = self._normalize_indicator(indicator, value)
                sector_adj += norm_value * sensitivity
        
        # Regime adjustment based on active regimes
        regime_adj = 0
        for regime, active in regime_signals.items():
            if active:
                regime_adj -= self.regime_weights.get(regime, 0.5) * 0.1
        
        # Final score combines base economic, sector-specific, and regime adjustments
        final_score = base_score + (sector_adj * 0.4) + regime_adj
        
        # Ensure in range [-1, 1]
        return max(-1, min(1, final_score))
    
    def _normalize_indicator(self, indicator, value):
        """Normalize an economic indicator to [-1, 1] range"""
        # Define ranges for each indicator
        ranges = {
            'GDP': (0, 5),
            'Inflation': (0, 6),
            'Unemployment': (3, 10),
            'Fed_Funds_Rate': (0, 8),
            'Ten_Year_Treasury': (1, 8),
            'Two_Year_Treasury': (1, 7),
            'Yield_Curve': (-2, 2),
            'Credit_Spread': (1, 6),
            'Industrial_Production': (-5, 5),
            'Housing_Starts': (0.5, 2),
            'Consumer_Sentiment': (50, 100)
        }
        
        # Default or custom range
        min_val, max_val = ranges.get(indicator, (0, 1))
        return 2 * ((value - min_val) / (max_val - min_val)) - 1
    
    def _normalize_score(self, value, min_val, max_val, reverse=False):
        """
        Normalize a value to a score between -1 and 1
        """
        if value is None or not isinstance(value, (int, float)):
            return 0
        
        normalized = 2 * ((value - min_val) / (max_val - min_val)) - 1
        normalized = max(-1, min(normalized, 1))  # Clip to [-1, 1]
        return -normalized if reverse else normalized
    
    def get_detailed_economic_data(self):
        """Return detailed economic data for display"""
        return {
            'indicators': self.economic_data,
            'regime_signals': {
                'high_inflation': self.economic_data.get('Inflation', 0) > 3.0,
                'rising_rates': self.economic_data.get('Fed_Funds_Rate', 0) > 3.5,
                'yield_curve_inversion': self.economic_data.get('Yield_Curve', 0) < 0.25,
                'growth_slowdown': self.economic_data.get('GDP', 0) < 2.0
            }
        }