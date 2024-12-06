# yfinance_cache.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from functools import lru_cache
import requests
from typing import Dict, Optional, Union
import pickle
import os

logger = logging.getLogger(__name__)

class YFinanceCache:
    def __init__(self, cache_dir: str = ".cache"):
        self._ticker_info_cache: Dict[str, Dict] = {}
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._last_request_time = 0
        self.min_request_interval = 0.2  # Increased to 200ms between requests
        self.max_retries = 3
        self.retry_delay = 1
        self.cache_dir = cache_dir
        self._initialize_cache_dir()

    def _initialize_cache_dir(self):
        """Initialize cache directory"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")

    def _enforce_rate_limit(self):
        """Enforce minimum time between requests with jitter"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.min_request_interval:
            # Add random jitter between 0-100ms
            jitter = np.random.uniform(0, 0.1)
            sleep_time = self.min_request_interval - time_since_last + jitter
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for a given key"""
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _save_to_disk_cache(self, key: str, data: Union[Dict, pd.DataFrame]):
        """Save data to disk cache"""
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {str(e)}")

    def _load_from_disk_cache(self, key: str) -> Optional[Union[Dict, pd.DataFrame]]:
        """Load data from disk cache"""
        try:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {str(e)}")
        return None

    def get_ticker_info(self, symbol: str) -> Optional[Dict]:
        """Get cached ticker info or fetch from yfinance with retries"""
        # Check memory cache
        if symbol in self._ticker_info_cache:
            return self._ticker_info_cache[symbol]

        # Check disk cache
        disk_cache = self._load_from_disk_cache(f"info_{symbol}")
        if disk_cache is not None:
            self._ticker_info_cache[symbol] = disk_cache
            return disk_cache

        # Fetch from yfinance with retries
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit()
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Cache successful result
                self._ticker_info_cache[symbol] = info
                self._save_to_disk_cache(f"info_{symbol}", info)
                return info

            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error fetching {symbol} (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
                
            except Exception as e:
                logger.error(f"Error fetching info for {symbol}: {str(e)}")
                break

        # Return None on failure
        return None

    def get_market_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        symbol: str = 'SPY',
        fallback_window: int = 60
    ) -> pd.DataFrame:
        """Get market data with fallback mechanisms"""
        cache_key = f"{symbol}_{start_date}_{end_date}"

        # Check memory cache
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        # Check disk cache
        disk_cache = self._load_from_disk_cache(cache_key)
        if disk_cache is not None:
            self._price_cache[cache_key] = disk_cache
            return disk_cache

        # Try to fetch with retries
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit()
                data = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date, 
                    progress=False
                )
                
                if not data.empty:
                    # Cache successful result
                    self._price_cache[cache_key] = data
                    self._save_to_disk_cache(cache_key, data)
                    return data

            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error fetching market data (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
                
            except Exception as e:
                logger.error(f"Error fetching market data: {str(e)}")
                break

        # Fallback: Try to fetch a smaller window
        logger.warning(f"Falling back to {fallback_window} day window")
        try:
            fallback_start = end_date - timedelta(days=fallback_window)
            self._enforce_rate_limit()
            data = yf.download(
                symbol,
                start=fallback_start,
                end=end_date,
                progress=False
            )
            if not data.empty:
                return data
        except Exception as e:
            logger.error(f"Fallback fetch failed: {str(e)}")

        # Return empty DataFrame if all attempts fail
        return pd.DataFrame()

    def clear_cache(self, clear_disk: bool = False):
        """Clear cached data"""
        self._ticker_info_cache.clear()
        self._price_cache.clear()
        
        if clear_disk and os.path.exists(self.cache_dir):
            try:
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
                logger.info("Disk cache cleared")
            except Exception as e:
                logger.error(f"Error clearing disk cache: {str(e)}")

# Global instance
yf_cache = YFinanceCache()