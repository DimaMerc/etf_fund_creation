# column_utils.py
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def standardize_dataframe_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Standardize DataFrame columns while preserving original data"""
    standard_cols = {
        'date': ['Date', 'date', 'DATE', 'timestamp'],
        'close': ['Close', 'close', 'CLOSE', 'Adj Close', 'adj_close', 'AdjClose'],
        'high': ['High', 'high', 'HIGH'],
        'low': ['Low', 'low', 'LOW'],
        'open': ['Open', 'open', 'OPEN'],
        'volume': ['Volume', 'volume', 'VOLUME']
    }
    
    col_map = {}
    df = df.copy()
    
    # Find matching columns
    for std_col, variants in standard_cols.items():
        for var in variants:
            if var in df.columns:
                col_map[std_col] = var
                break
                
    # Verify required columns
    required = ['close', 'volume', 'date']
    if not all(col in col_map for col in required):
        missing = [col for col in required if col not in col_map]
        logger.error(f"Missing required columns: {missing}")
        return None, {}
        
    return df, col_map

def get_column_name(df: pd.DataFrame, std_col: str) -> Optional[str]:
    """Get actual column name for standardized column"""
    _, col_map = standardize_dataframe_columns(df)
    return col_map.get(std_col)