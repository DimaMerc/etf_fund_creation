# utils/data_utils.py

import logging
import traceback
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from collections import defaultdict
from data_preparation import ETFDataPipeline
from config import EQUITY_START_DATE, EQUITY_END_DATE, BACKTEST_START_DATE

from data_fetcher import (
    fetch_sp500_constituents,
    fetch_equity_data_for_symbols,
    fetch_options_data, 
    fetch_equity_data
    
)
from data_preparation import fetch_treasury_rates
from config import TREASURY_FALLBACK_RATES, MINIMUM_HISTORY_YEARS
from column_utils import standardize_dataframe_columns, get_column_name

logger = logging.getLogger(__name__)

def concatenate_metrics_arrays(arrays_list):
    try:
        # Check that all arrays have the same shape along the concatenation axis (e.g., axis 1)
        if len(arrays_list) == 0:
            logger.error("No arrays provided for metric calculation.")
            return None

        base_shape = arrays_list[0].shape
        for idx, array in enumerate(arrays_list):
            if len(array.shape) != len(base_shape) or array.shape[1] != base_shape[1]:
                logger.error(f"Array at index {idx} has mismatched shape {array.shape} compared to base shape {base_shape}")
                return None

        # Proceed with concatenation if all shapes match
        concatenated_array = np.concatenate(arrays_list, axis=1)
        return concatenated_array
    except ValueError as e:
        logger.error(f"Error calculating ETF metrics: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calculating ETF metrics: {str(e)}")
        return None


def fetch_and_validate_data(dates: Dict[str, pd.Timestamp], 
                          data_pipeline: ETFDataPipeline) -> Dict[str, Any]:
    """
    Fetch and validate all required data
    """
    try:
        logger.info("\nFetching and validating data...")
        
        # 1. Fetch symbols
        symbols, sector_map = fetch_sp500_constituents()
        if not symbols:
            logger.error("Failed to fetch symbols")
            return None
            
        logger.info(f"Fetched {len(symbols)} symbols")
        
        # 2. Fetch equity data
        equity_data = fetch_equity_data_for_symbols(
            symbols=symbols,
            start_date=dates['backtest_start'],
            end_date=dates['equity_end']
        )
        
        if not equity_data:
            logger.error("No equity data fetched")
            return None
            
        logger.info(f"Fetched equity data for {len(equity_data)} symbols")
        
        # 3. Fetch or create treasury data
        try:
            treasury_data = fetch_treasury_rates()
        except Exception as e:
            logger.warning(f"Error fetching treasury rates: {str(e)}")
            logger.info("Using fallback treasury rates")
            treasury_data = pd.DataFrame([TREASURY_FALLBACK_RATES])
        
       # 4. Process data through pipeline
        processed_data = data_pipeline.prepare_etf_features(equity_data)
        if not processed_data:
            logger.error("No data after processing") 
            return None

        
        
        # 5. Validate processed data
        validation_results = {
            'valid_symbols': list(processed_data.keys()),
            'coverage_stats': {},
            'data_quality': {}
        }
        
        for symbol, df in processed_data.items():
            try:
                stats = {
                    'start_date': df['date'].min(),
                    'end_date': df['date'].max(),
                    'trading_days': len(df),
                    'coverage_ratio': len(df) / 252  # Approximate trading days per year
                }
                validation_results['coverage_stats'][symbol] = stats
            except Exception as e:
                logger.warning(f"Error calculating stats for {symbol}: {str(e)}")
                continue
        
        # 6. Return validated data including treasury_data
        return {
            'equity_data': processed_data,
            'options_data': {},  # Empty dict for options data
            'valid_symbols': validation_results['valid_symbols'],
            'sector_map': sector_map,
            'coverage_stats': validation_results['coverage_stats'],
            'treasury_data': treasury_data  # Added treasury data
        }
        
    except Exception as e:
        logger.error(f"Error in data fetching and validation: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
def validate_date_ranges(equity_data: Dict, start_date: pd.Timestamp, 
                        end_date: pd.Timestamp) -> bool:
    """
    Validate data availability across specified date range
    
    Args:
        equity_data: Dictionary of equity price data
        start_date: Start date for validation
        end_date: End date for validation
        
    Returns:
        bool: True if date ranges are valid, False otherwise
    """
    try:
        logger.info("\nValidating date ranges...")
        
        for symbol, df in equity_data.items():
            df_start = pd.to_datetime(df['date'].min())
            df_end = pd.to_datetime(df['date'].max())
            
            if df_start > start_date:
                logger.warning(f"{symbol} data starts late: {df_start}")
                return False
                
            if df_end < end_date:
                logger.warning(f"{symbol} data ends early: {df_end}")
                return False
                
            # Check for large gaps
            dates = pd.to_datetime(df['date'])
            expected_dates = pd.date_range(df_start, df_end, freq='B')
            missing_dates = expected_dates.difference(dates)
            
            if len(missing_dates) > 10:  # Allow for some holidays
                logger.warning(f"{symbol} has {len(missing_dates)} missing dates")
                return False
                
        logger.info("Date range validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating date ranges: {str(e)}")
        return False
    

def validate_input_data(price_data, start_date, end_date, sector_map=None):
    """Validate input data before starting backtest with timezone handling"""
    try:
        if not price_data:
            logger.error("No price data provided")
            return False
            
        logger.info("\nValidating input data:")
        logger.info(f"Symbols received: {len(price_data)}")
        
        # Convert dates and ensure timezone-naive
        start_ts = pd.to_datetime(start_date).tz_localize(None)
        end_ts = pd.to_datetime(end_date).tz_localize(None)
        
        logger.info(f"Date range: {start_ts} to {end_ts}")
        
        valid_symbols = []
        coverage_stats = {}
        invalid_reasons = defaultdict(int)
        
        # Track sector distribution if provided
        sector_distribution = defaultdict(int) if sector_map else None
        
        for symbol, df in price_data.items():
            try:
                if df is None or df.empty:
                    invalid_reasons['empty_data'] += 1
                    continue
                
                # Basic data structure check
                required_columns = ['date', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    invalid_reasons['missing_columns'] += 1
                    continue
                    
                # Convert dates and ensure timezone-naive
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                
                # Check date coverage
                df_start = df['date'].min()
                df_end = df['date'].max()
                
                trading_days = len(df)
                expected_days = len(pd.date_range(start=df_start, end=df_end, freq='B'))
                coverage_ratio = trading_days / expected_days
                
                coverage_stats[symbol] = {
                    'start_date': df_start,
                    'end_date': df_end,
                    'trading_days': trading_days,
                    'coverage_ratio': coverage_ratio,
                    'sector': sector_map.get(symbol, 'Unknown') if sector_map else 'Unknown'
                }
                
                # More lenient validation criteria
                if (df_end >= start_ts and  # Has some data after start
                    df_start <= end_ts and  # Has some data before end
                    trading_days >= 30 and  # Minimum required history
                    coverage_ratio >= 0.5):  # More lenient coverage requirement
                    valid_symbols.append(symbol)
                    # Track sector distribution
                    if sector_map:
                        sector = sector_map.get(symbol, 'Unknown')
                        sector_distribution[sector] += 1
                else:
                    if df_end <= start_ts:
                        invalid_reasons['ends_too_early'] += 1
                    if df_start >= end_ts:
                        invalid_reasons['starts_too_late'] += 1
                    if trading_days < 30:
                        invalid_reasons['insufficient_history'] += 1
                    if coverage_ratio < 0.5:
                        invalid_reasons['poor_coverage'] += 1
                    
            except Exception as e:
                logger.error(f"Error validating {symbol}: {str(e)}")
                invalid_reasons['processing_error'] += 1
                continue
                
        logger.info(f"\nValidation Results:")
        logger.info(f"Valid symbols: {len(valid_symbols)}/{len(price_data)}")
        
        if invalid_reasons:
            logger.info("\nInvalid symbols breakdown:")
            for reason, count in invalid_reasons.items():
                logger.info(f"{reason}: {count}")

        # Log sector distribution if available
        if sector_distribution:
            logger.info("\nSector Distribution:")
            for sector, count in sector_distribution.items():
                percentage = (count / len(valid_symbols)) * 100 if valid_symbols else 0
                logger.info(f"{sector}: {count} stocks ({percentage:.1f}%)")
        
        if valid_symbols:
            # Log sample of valid symbols
            logger.info("\nSample of valid symbols:")
            for symbol in valid_symbols[:5]:
                stats = coverage_stats[symbol]
                logger.info(f"{symbol}:")
                logger.info(f"  Trading days: {stats['trading_days']}")
                logger.info(f"  Coverage ratio: {stats['coverage_ratio']:.2f}")
                if sector_map:
                    logger.info(f"  Sector: {stats['sector']}")
            
            # Create reduced price_data with only valid symbols
            for symbol in list(price_data.keys()):
                if symbol not in valid_symbols:
                    del price_data[symbol]
                    
            return True
        else:
            logger.error("No valid symbols found after validation")
            return False
            
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}")
        return False
    

