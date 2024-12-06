# data_preparation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import  RISK_FREE_RATE, VOLATILITY_LOOKBACK, TARGET_PORTFOLIO_VOLATILITY, MAX_POSITION_SIZE, TREASURY_FALLBACK_RATES
from config import ALPHAVANTAGE_API_KEY
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho
from black_scholes import black_scholes_price
import ta
from options_model import OptionsPricingMLP
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import traceback
import logging
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time


import requests
from scipy.interpolate import CubicSpline
from column_utils import standardize_dataframe_columns, get_column_name
logger = logging.getLogger(__name__)




class BaseDataPipeline:
    """Base class for data preparation pipelines"""
    
    def __init__(self, alphavantage_key: str = ALPHAVANTAGE_API_KEY):
        self.scaler = StandardScaler()
        self.data_quality_metrics = {}
        self.feature_cache = {}
        self.treasury_cache = {}
        self.alphavantage_key = alphavantage_key
        self.data_cache = {}
        self.last_request_time = 0
        self.request_interval = 12.1  # AlphaVantage limit
        self.failed_symbols = set()

    def _enforce_rate_limit(self):
        """Enforce API rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()

    def _validate_data(self, data_start, required_start, actual_days, required_days,
                      coverage_ratio, price_quality):
        """Validate data quality"""
        reasons = []
        
        if data_start > required_start:
            reasons.append(f"Data starts too late (starts at {data_start}, need {required_start})")
            
        if actual_days < required_days:
            reasons.append(f"Insufficient trading days (has {actual_days}, need {required_days})")
            
        if coverage_ratio < 0.95:
            reasons.append(f"Poor data coverage ({coverage_ratio:.1%} < 95%)")
            
        if price_quality.get('major_gaps', 0) > 0:
            reasons.append(f"Has {price_quality['major_gaps']} major price gaps")
            
        return reasons

    def _check_price_quality(self, df):
        """Check quality of price data"""
        try:
            quality_metrics = {
                'missing_values': df.isnull().sum().to_dict(),
                'zero_values': (df == 0).sum().to_dict(),
                'major_gaps': 0,
                'outliers': 0
            }
            
            returns = df['close'].pct_change()
            major_gaps = returns[abs(returns) > 0.20]
            quality_metrics['major_gaps'] = len(major_gaps)
            
            z_scores = stats.zscore(returns.dropna())
            quality_metrics['outliers'] = len(z_scores[abs(z_scores) > 3])
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error checking price quality: {str(e)}")
            return None

    def _calculate_enhanced_features(self, df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
        """Calculate enhanced features using standardized column names"""
        try:
            if df is None or not col_map:
                return pd.DataFrame()

            close_col = col_map['close']
            volume_col = col_map['volume']
            
            # Calculate features using mapped column names
            df = df.copy()
            df['equity_return'] = df[close_col].pct_change()
            df['log_returns'] = np.log(df[close_col]).diff()
            
            # Moving averages
            for window in [5, 10, 20]:
                df[f'equity_ma_{window}'] = df[close_col].rolling(window=window, min_periods=1).mean()
                df[f'volume_ma_{window}'] = df[volume_col].rolling(window=window, min_periods=1).mean()
            
            # Technical indicators
            df['rsi_14'] = ta.momentum.RSIIndicator(close=df[close_col], window=14).rsi()
            
            macd = ta.trend.MACD(close=df[close_col])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            bb = ta.volatility.BollingerBands(close=df[close_col])
            df['bollinger_mavg'] = bb.bollinger_mavg()
            df['bollinger_hband'] = bb.bollinger_hband()
            df['bollinger_lband'] = bb.bollinger_lband()

            return df
            
        except Exception as e:
            logger.error(f"Error calculating enhanced features: {str(e)}")
            return pd.DataFrame()

    def preprocess_options_data(self, options_data, equity_data, treasury_data=None):
        """Preprocess options data with enhanced validation"""
        try:
            logger.info("\nPreprocessing options data...")
            all_data = []
            current_date = pd.Timestamp.now().tz_localize(None)
            
            for symbol, df_options in options_data.items():
                try:
                    df_equity = equity_data.get(symbol)
                    if df_equity is None or df_options.empty or df_equity.empty:
                        continue
                        
                    df = df_options.copy()
                    current_price = df_equity['close'].iloc[-1]
                    
                    # Add required features
                    df['moneyness'] = current_price / df['strike']
                    df['timeToExpiration'] = (pd.to_datetime(df['expirationDate']) - current_date).dt.days / 365.0
                    mid_price = (df['ask'] + df['bid']) / 2
                    df['bidAskSpread'] = (df['ask'] - df['bid']) / mid_price
                    
                    returns = df_equity['close'].pct_change().dropna()
                    hist_vol = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else returns.std() * np.sqrt(252)
                    df['historicalVolatility'] = hist_vol
                    
                    df['optionType'] = (df['optionType'].str.lower() == 'call').astype(int)
                    
                    all_data.append(df)
                    
                except Exception as e:
                    logger.error(f"Error processing options for {symbol}: {str(e)}")
                    continue
            
            if not all_data:
                logger.warning("No valid options data to process")
                return pd.DataFrame()
            
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"\nProcessed {len(combined_data)} options")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error preprocessing options data: {str(e)}")
            return pd.DataFrame()

class EnhancedDataPipeline(BaseDataPipeline):
    """Pipeline for regular equity data processing"""
    
    def prepare_features(self, price_data: Dict[str, pd.DataFrame], date: Optional[datetime] = None) -> Dict:
        """Prepare equity-specific features"""
        try:
            if date is not None:
                return self._prepare_features_for_date(price_data, date)
                
            all_features = {}
            for symbol, df in price_data.items():
                try:
                    if df is None or df.empty:
                        continue
                        
                    features = self._calculate_enhanced_features(df)
                    if not features.empty:
                        all_features[symbol] = features
                        
                except Exception as e:
                    logger.error(f"Error calculating features for {symbol}: {str(e)}")
                    continue
                    
            return all_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return {}

    def validate_data(self, price_data, start_date, minimum_history_years=3):
        """Validate data quality and coverage"""
        try:
            logger.info(f"\nValidating data from {start_date} with {minimum_history_years} years minimum history")
            
            validation_results = {
                'valid_symbols': [],
                'invalid_symbols': [],
                'coverage_stats': {},
                'data_quality': {}
            }
            
            equity_start = pd.to_datetime(start_date).tz_localize(None)
            required_start = equity_start - pd.DateOffset(years=minimum_history_years)
            
            for symbol, df in price_data.items():
                try:
                    if df is None or df.empty:
                        validation_results['invalid_symbols'].append({'symbol': symbol, 'reason': 'No data'})
                        continue
                        
                    df = df.copy()
                    if 'date' not in df.columns:
                        validation_results['invalid_symbols'].append({'symbol': symbol, 'reason': 'No date column'})
                        continue
                        
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                    
                    data_start = df['date'].min()
                    data_end = df['date'].max()
                    actual_days = len(df)
                    required_days = minimum_history_years * 252
                    
                    expected_dates = pd.date_range(data_start, data_end, freq='B')
                    coverage_ratio = 1 - (len(expected_dates.difference(df['date'])) / len(expected_dates))
                    
                    price_quality = self._check_price_quality(df)
                    
                    validation_results['coverage_stats'][symbol] = {
                        'start_date': data_start,
                        'end_date': data_end,
                        'trading_days': actual_days,
                        'coverage_ratio': coverage_ratio,
                        'price_quality': price_quality
                    }
                    
                    if (data_start > required_start or 
                        actual_days < required_days or 
                        coverage_ratio < 0.95):
                        reason = self._validate_data(data_start, required_start, actual_days, 
                                                   required_days, coverage_ratio, price_quality)
                        validation_results['invalid_symbols'].append({'symbol': symbol, 'reason': '; '.join(reason)})
                        continue
                    
                    validation_results['valid_symbols'].append(symbol)
                    
                except Exception as e:
                    logger.error(f"Error validating {symbol}: {str(e)}")
                    validation_results['invalid_symbols'].append({'symbol': symbol, 'reason': str(e)})
                    
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return None

class ETFDataPipeline(BaseDataPipeline):
    """Pipeline for ETF-specific data processing"""
    
    def prepare_etf_features(self, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """Prepare ETF-specific features including tracking metrics"""
        try:
            # Debug logging
            logger.info("\nDebugging ETF feature preparation:")
            sample_symbol = next(iter(price_data))
            logger.info(f"Sample symbol: {sample_symbol}")
            logger.info(f"Columns available: {price_data[sample_symbol].columns.tolist()}")
            logger.info(f"Index type: {type(price_data[sample_symbol].index)}")
            logger.info(f"First few rows:\n{price_data[sample_symbol].head()}")

            features_dict = {}
            
            for symbol, df in price_data.items():
                df, col_map = standardize_dataframe_columns(df)
                if df is None:
                    continue
                    
                # Get standardized column names
                close_col = col_map['close']
                volume_col = col_map['volume']
                high_col = col_map['high']
                low_col = col_map['low']
                
                # Calculate base features
                feature_data = self._calculate_enhanced_features(df, col_map)
                if feature_data.empty:
                    continue
                
                # Add ETF-specific features using mapped column names
                feature_data['dollar_volume'] = df[close_col] * df[volume_col]
                feature_data['spread'] = (df[high_col] - df[low_col]) / df[close_col]
                feature_data['illiquidity'] = feature_data.get('equity_return', 0) / feature_data['dollar_volume']
                feature_data['momentum_1m'] = self.calculate_momentum(df[close_col], periods=20)
                
                # Calculate volume stability
                volume_std = df[volume_col].rolling(20).std()
                volume_mean = df[volume_col].rolling(20).mean()
                feature_data['volume_stability'] = volume_std / volume_mean.replace(0, np.nan)
                
                # Fill any NaN values that might have been created
                feature_data = feature_data.ffill().bfill()
                
                features_dict[symbol] = feature_data
                    
            return features_dict
                
        except Exception as e:
            logger.error(f"Error preparing ETF features: {str(e)}")
            return {}
        
    def calculate_momentum(self, prices: pd.Series, periods: int) -> pd.Series:
        """Calculate momentum with proper NA handling"""
        cleaned_prices = prices.ffill().bfill()
        return (cleaned_prices / cleaned_prices.shift(periods) - 1).fillna(0)

    def prepare_features(self, price_data: Dict[str, pd.DataFrame], date: Optional[datetime] = None) -> Dict:
        """Main entry point for ETF feature preparation"""
        try:
            # First validate and normalize the input data
            normalized_data = {}
            for symbol, df in price_data.items():
                try:
                    if df is None or df.empty:
                        continue

                    # Ensure we have the required columns
                    required_columns = ['close', 'high', 'low', 'volume', 'date']
                    if not all(col.lower() in [c.lower() for c in df.columns] for col in required_columns):
                        logger.warning(f"Missing required columns for {symbol}")
                        continue

                    # Normalize column names to lowercase
                    df_normalized = df.copy()
                    df_normalized.columns = [c.lower() for c in df_normalized.columns]

                    # Filter data if date is provided
                    if date is not None:
                        df_normalized = df_normalized[df_normalized['date'] <= date]

                    normalized_data[symbol] = df_normalized

                except Exception as e:
                    logger.error(f"Error normalizing data for {symbol}: {str(e)}")
                    continue

            if not normalized_data:
                logger.error("No valid data after normalization")
                return {}

            # Now process the normalized data
            return self.prepare_etf_features(normalized_data)

        except Exception as e:
            logger.error(f"Error in prepare_features: {str(e)}")
            logger.error(traceback.format_exc())
            return {}


    def fetch_etf_holdings_alphavantage(self, symbol: str) -> Optional[Dict]:
        """Fetch ETF holdings data from AlphaVantage"""
        try:
            self._enforce_rate_limit()
            
            url = (
                f'https://www.alphavantage.co/query?'
                f'function=ETF_PROFILE'
                f'&symbol={symbol}'
                f'&apikey={self.alphavantage_key}'
            )
            
            logger.info(f"Fetching ETF data for {symbol}")
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}")
                return None
                
            data = response.json()
            logger.debug(f"API Response: {data}")
            
            # Check for error messages in response
            if 'Error Message' in data:
                logger.error(f"API returned error: {data['Error Message']}")
                return None
                
            # Validate required fields
            required_fields = ['holdings', 'sectors', 'net_assets', 'asset_allocation']
            if not all(field in data for field in required_fields):
                missing = [f for f in required_fields if f not in data]
                logger.error(f"Missing required fields: {missing}")
                return None
                
            # Convert holdings and sectors to DataFrames
            holdings_df = pd.DataFrame(data['holdings'])
            sectors_df = pd.DataFrame(data['sectors'])
            
            # Parse asset allocation
            asset_allocation = pd.Series(data['asset_allocation']).to_dict()
            
            # Create metadata dict
            metadata = {
                'net_assets': float(data['net_assets']),
                'net_expense_ratio': float(data['net_expense_ratio']),
                'portfolio_turnover': float(data['portfolio_turnover']),
                'dividend_yield': float(data['dividend_yield']),
                'inception_date': data['inception_date'],
                'leveraged': data['leveraged'],
                'asset_allocation': asset_allocation,
                'update_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            return {
                'holdings': holdings_df,
                'sectors': sectors_df,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error fetching ETF holdings for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
def fetch_treasury_rates():
    """Fetch current treasury rates with fallback"""
    try:
        return _get_fallback_rates()
    except Exception as e:
        logger.error(f"Error fetching treasury rates: {str(e)}")
        return _get_fallback_rates()

def _get_fallback_rates():
    """Return fallback treasury rates"""
    try:
        logger.warning("Using fallback treasury rates")
        return pd.DataFrame([TREASURY_FALLBACK_RATES])
    except Exception as e:
        logger.error(f"Error in fallback rates: {str(e)}")
        return None
