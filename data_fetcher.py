# data_fetcher.py

import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from config import EQUITY_START_DATE, EQUITY_END_DATE, BACKTEST_START_DATE
from typing import Dict, Optional, List, Tuple
import logging
import traceback
from collections import defaultdict



logger = logging.getLogger(__name__)



def fetch_sp500_constituents() -> Tuple[List[str], Dict[str, str]]:
    """Enhanced S&P 500 constituent fetching with retry logic"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                tables = pd.read_html(url)
                df = tables[0]
                
                symbols = []
                sector_map = {}
                market_caps = {}
                
                # Process constituents
                for _, row in df.iterrows():
                    symbol = str(row['Symbol']).strip().replace('.', '-')
                    sector = str(row['GICS Sector']).strip()
                    
                    if symbol and len(symbol) <= 5:  # Basic validation
                        symbols.append(symbol)
                        sector_map[symbol] = sector
                        
                        # Try to get market cap
                        try:
                            ticker = yf.Ticker(symbol)
                            market_caps[symbol] = ticker.info.get('marketCap', 0)
                            time.sleep(0.1)  # Rate limiting
                        except:
                            market_caps[symbol] = 0
                
                logger.info(f"Successfully fetched {len(symbols)} S&P 500 constituents")
                logger.info(f"Sectors represented: {len(set(sector_map.values()))}")
                
                # Log sector distribution
                sector_counts = defaultdict(int)
                for sector in sector_map.values():
                    sector_counts[sector] += 1
                
                logger.info("\nSector Distribution:")
                for sector, count in sorted(sector_counts.items()):
                    logger.info(f"{sector}: {count} stocks")
                
                return symbols, sector_map
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Failed to fetch S&P 500 constituents")
                    return [], {}
                    
        return [], {}
        
    except Exception as e:
        logger.error(f"Error fetching S&P 500 data: {str(e)}")
        return [], {}


def fetch_equity_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        # Convert dates and ensure timezone-naive
        start_ts = pd.to_datetime(start_date).tz_localize(None)
        end_ts = pd.to_datetime(end_date).tz_localize(None)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add delay between retries
                if attempt > 0:
                    time.sleep(2 ** attempt)
                    
                data = yf.download(
                    str(symbol).strip(),  # Ensure clean symbol string
                    start=start_ts,
                    end=end_ts,
                    progress=False,
                    ignore_tz=True
                )
                
                if data.empty:
                    continue
                    
                # Process the data
                data = data.reset_index()
                data.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                # Ensure date column is timezone naive
                data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
                
                return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"All retries failed for {symbol}: {str(e)}")
                    return pd.DataFrame()
                continue
                
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()
    
def fetch_equity_data_for_symbols(symbols: List[str],
                                start_date: str,
                                end_date: str,
                                include_market_cap: bool = True) -> Dict[str, pd.DataFrame]:
    try:
        logger.info(f"\nFetching data for {len(symbols)} symbols")
        
        # Debug timestamps
        start_ts = pd.to_datetime(start_date).tz_localize(None)
        end_ts = pd.to_datetime(end_date).tz_localize(None)
        logger.info(f"Date range: {start_ts} to {end_ts}")

        for symbol in symbols[:1]:  # Test with first symbol
            logger.info(f"\nTesting fetch for {symbol}")
            
            # Try direct single symbol download first
            test_data = yf.download(
                symbol,
                start=start_ts,
                end=end_ts,
                progress=False
            )
            
            logger.info(f"Direct download shape: {test_data.shape}")
            logger.info(f"First few rows:\n{test_data.head()}")
            logger.info(f"Data types:\n{test_data.dtypes}")
            logger.info(f"Missing values:\n{test_data.isna().sum()}")

        batch_size = 100
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        equity_data = {}
        total_fetched = 0
        missing_data = []
        
        for batch in symbol_batches:
            try:
                time.sleep(2)  # 2 second delay between batches
                logger.info(f"\nProcessing batch of {len(batch)} symbols")
                
                # Batch download
                data = yf.download(
                    tickers=batch,
                    start=start_ts,
                    end=end_ts,
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    prepost=False 
                )
                
                logger.info(f"Batch download shape: {data.shape}")
                if not data.empty:
                    logger.info(f"Sample from batch:\n{data.head()}")
                
                # Process current batch
                for symbol in batch:
                    try:
                        if isinstance(data, pd.DataFrame) and len(batch) == 1:
                            symbol_data = data.copy()
                        elif symbol in data.columns.levels[0]:
                            symbol_data = data[symbol].copy()
                        else:
                            logger.warning(f"No data found for {symbol}")
                            missing_data.append(symbol)
                            continue
                            
                        symbol_data = symbol_data.reset_index()
                        
                        # Debug for each symbol's data
                        logger.debug(f"\nProcessing {symbol}:")
                        logger.debug(f"Data shape: {symbol_data.shape}")
                        logger.debug(f"Missing values:\n{symbol_data.isna().sum()}")

                        df = pd.DataFrame()
                        df['date'] = pd.to_datetime(symbol_data['Date'])
                        df['close'] = symbol_data['Close']
                        df['volume'] = symbol_data['Volume']
                        df['high'] = symbol_data['High']
                        df['low'] = symbol_data['Low']
                        df['open'] = symbol_data['Open']
                        
                        # Make dates timezone-naive
                        df['date'] = df['date'].dt.tz_localize(None)
                        
                        if include_market_cap:
                            time.sleep(1)
                            try:
                                ticker = yf.Ticker(symbol)
                                info = ticker.info
                                logger.debug(f"Ticker info keys: {info.keys() if info else 'No info'}")
                                df['market_cap'] = info.get('marketCap', None) if info else None
                            except Exception as e:
                                logger.warning(f"Failed to get market cap for {symbol}: {str(e)}")
                                df['market_cap'] = None
                        
                        # Check final data quality
                        logger.debug(f"Final df shape: {df.shape}")
                        logger.debug(f"Final missing values:\n{df.isna().sum()}")
                        
                        if not df.empty:
                            equity_data[symbol] = df
                            total_fetched += 1
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        missing_data.append(symbol)
                        continue

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                missing_data.extend(batch)
                continue
        
        # Log results
        success_rate = (total_fetched / len(symbols)) * 100
        logger.info(f"\nFetch Results:")
        logger.info(f"Successfully fetched: {total_fetched}/{len(symbols)} ({success_rate:.1f}%)")
        
        if missing_data:
            logger.warning(f"Missing data for {len(missing_data)} symbols:")
            for symbol in missing_data[:5]:
                logger.warning(f"- {symbol}")
            if len(missing_data) > 5:
                logger.warning(f"... and {len(missing_data) - 5} more")
        
        return equity_data
        
    except Exception as e:
        logger.error(f"Error in batch download: {str(e)}")
        return {}
    
def validate_data_coverage(price_data: Dict[str, pd.DataFrame],
                         start_date: str,
                         end_date: str,
                         min_coverage: float = 0.95) -> List[str]:
    """Validate data coverage and return valid symbols"""
    try:
        valid_symbols = []
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        for symbol, df in price_data.items():
            try:
                # Check date range
                df_start = pd.to_datetime(df['date'].min())
                df_end = pd.to_datetime(df['date'].max())
                
                if df_start > start_ts or df_end < end_ts:
                    continue
                
                # Check data coverage
                expected_days = len(pd.date_range(start=start_ts, end=end_ts, freq='B'))
                actual_days = len(df)
                coverage = actual_days / expected_days
                
                if coverage >= min_coverage:
                    valid_symbols.append(symbol)
                    
            except Exception as e:
                logger.warning(f"Error validating {symbol}: {str(e)}")
                continue
        
        coverage_rate = (len(valid_symbols) / len(price_data)) * 100
        logger.info(f"\nData Coverage Analysis:")
        logger.info(f"Valid symbols: {len(valid_symbols)}/{len(price_data)} ({coverage_rate:.1f}%)")
        
        return valid_symbols
        
    except Exception as e:
        logger.error(f"Error validating data coverage: {str(e)}")
        return []

def fetch_options_data(symbols, start_date, end_date):
    """Fetch options data with improved error handling"""
    options_data = {}
    current_date = pd.Timestamp.now()
    
    for symbol in symbols:
        try:
            # Add delay between requests
            time.sleep(0.5)
            
            ticker = yf.Ticker(symbol)
            if not hasattr(ticker, 'options'):
                logger.debug(f"No options available for {symbol}")
                continue
                
            try:
                option_dates = ticker.options
                if not option_dates:
                    logger.debug(f"No option dates for {symbol}")
                    continue
            except (ValueError, AttributeError, requests.exceptions.RequestException) as e:
                logger.debug(f"Cannot get option dates for {symbol}: {str(e)}")
                continue

            # Get valid expirations (next 3 months)
            valid_expirations = []
            for exp in option_dates:
                try:
                    exp_date = pd.to_datetime(exp)
                    if current_date < exp_date <= current_date + pd.DateOffset(months=3):
                        valid_expirations.append(exp)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Invalid expiration date for {symbol}: {exp}")
                    continue
                    
            valid_expirations = valid_expirations[:3]
            
            if not valid_expirations:
                continue
            
            all_options = []
            for exp in valid_expirations:
                try:
                    time.sleep(0.2)  # Add delay between option chain requests
                    chain = ticker.option_chain(exp)
                    if chain is None or not hasattr(chain, 'calls') or not hasattr(chain, 'puts'):
                        logger.debug(f"Invalid option chain for {symbol} exp {exp}")
                        continue
                        
                    # Process calls
                    calls = chain.calls[
                        (chain.calls['volume'].fillna(0) > 100) &
                        (chain.calls['bid'].fillna(0) > 0.10) &
                        (chain.calls['openInterest'].fillna(0) > 50)
                    ].copy()
                    
                    if not calls.empty:
                        calls['optionType'] = 'call'
                        calls['bid_ask_spread'] = (calls['ask'] - calls['bid']) / calls['bid']
                        calls = calls[calls['bid_ask_spread'] < 0.10]
                    
                    # Process puts
                    puts = chain.puts[
                        (chain.puts['volume'].fillna(0) > 100) &
                        (chain.puts['bid'].fillna(0) > 0.10) &
                        (chain.puts['openInterest'].fillna(0) > 50)
                    ].copy()
                    
                    if not puts.empty:
                        puts['optionType'] = 'put'
                        puts['bid_ask_spread'] = (puts['ask'] - puts['bid']) / puts['bid']
                        puts = puts[puts['bid_ask_spread'] < 0.10]
                    
                    # Combine and add metadata
                    valid_options = pd.concat([calls, puts], ignore_index=True)
                    if not valid_options.empty:
                        valid_options['expirationDate'] = pd.to_datetime(exp)
                        valid_options['symbol'] = symbol
                        valid_options['days_to_expiry'] = (
                            valid_options['expirationDate'] - current_date
                        ).dt.days
                        all_options.append(valid_options)
                        
                except Exception as e:
                    logger.debug(f"Error processing expiration {exp} for {symbol}: {str(e)}")
                    continue
            
            if all_options:
                options_data[symbol] = pd.concat(all_options, ignore_index=True)
                
        except (ValueError, requests.exceptions.RequestException) as e:
            logger.debug(f"Network error fetching options for {symbol}: {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {str(e)}")
            continue
    
    return options_data

def fetch_sector_etfs() -> Dict[str, pd.DataFrame]:
    """Fetch data for sector ETFs"""
    try:
        sector_etfs = {
            'XLF': 'Financials',
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        etf_data = {}
        
        for symbol in sector_etfs.keys():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info:
                    etf_data[symbol] = {
                        'info': info,
                        'sector': sector_etfs[symbol]
                    }
                    
                    # Try to get holdings
                    try:
                        holdings = ticker.get_holdings()
                        if holdings is not None:
                            etf_data[symbol]['holdings'] = holdings
                    except:
                        pass
                    
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return etf_data
        
    except Exception as e:
        logger.error(f"Error fetching sector ETFs: {str(e)}")
        return {}
    
def fetch_spy_data(start_date: str,
                  end_date: str,
                  include_holdings: bool = False) -> Dict:
    """Fetch SPY data with holdings information"""
    try:
        spy_data = {}
        
        # Get price data
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        if not spy.empty:
            spy_data['prices'] = spy
        
        # Get holdings if requested
        if include_holdings:
            try:
                spy_ticker = yf.Ticker('SPY')
                holdings = spy_ticker.get_holdings()
                if holdings is not None:
                    spy_data['holdings'] = holdings
            except:
                logger.warning("Failed to fetch SPY holdings")
        
        return spy_data
        
    except Exception as e:
        logger.error(f"Error fetching SPY data: {str(e)}")
        return {}
    
def _fetch_spy_benchmark(start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.Series]:
    """
    Robust SPY data fetching with fallback mechanisms
    """
    try:
        # Method 1: Try direct download
        spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
        if not spy_data.empty:
            return spy_data['Adj Close']
            
        # Method 2: Try downloading in smaller chunks
        logger.info("Retrying SPY data in chunks...")
        chunk_size = pd.Timedelta(days=90)
        current_start = start_date
        chunks = []
        
        while current_start < end_date:
            chunk_end = min(current_start + chunk_size, end_date)
            try:
                chunk_data = yf.download('SPY', start=current_start, end=chunk_end, progress=False)
                if not chunk_data.empty:
                    chunks.append(chunk_data['Adj Close'])
            except Exception as e:
                logger.warning(f"Failed to fetch chunk {current_start} to {chunk_end}: {str(e)}")
            current_start = chunk_end
            
        if chunks:
            return pd.concat(chunks)
            
        # Method 3: Try using cached data
        logger.info("Trying cached SPY data...")
        from yfinance_cache import yf_cache
        return yf_cache.get_market_data(start_date, end_date, 'SPY')['Adj Close']
            
    except Exception as e:
        logger.error(f"Failed to fetch SPY benchmark: {str(e)}")
        return None

def create_spy_benchmark(portfolio_value: pd.Series, 
                        start_date: pd.Timestamp,
                        end_date: pd.Timestamp,
                        initial_value: float) -> Optional[pd.Series]:
    """
    Create SPY benchmark series aligned with portfolio value
    """
    try:
        spy_data = _fetch_spy_benchmark(start_date, end_date)
        if spy_data is not None and not spy_data.empty:
            # Align dates with portfolio value
            spy_data = spy_data.reindex(portfolio_value.index, method='ffill')
            # Normalize to initial value
            return (spy_data / spy_data.iloc[0]) * initial_value
        return None
    except Exception as e:
        logger.error(f"Error creating SPY benchmark: {str(e)}")
        return None
    
def process_option_chain(chain, symbol, expiration):
    """Process option chain data"""
    options = []
    
    # Process calls
    calls = chain.calls[
        (chain.calls['volume'] > 100) &
        (chain.calls['bid'] > 0.10) &
        (chain.calls['openInterest'] > 50)
    ].copy()
    
    if not calls.empty:
        calls['optionType'] = 'call'
        calls['bid_ask_spread'] = (calls['ask'] - calls['bid']) / calls['bid']
        calls = calls[calls['bid_ask_spread'] < 0.10]
        calls['symbol'] = symbol
        calls['expirationDate'] = pd.to_datetime(expiration)
        options.extend(calls.to_dict('records'))
    
    # Process puts
    puts = chain.puts[
        (chain.puts['volume'] > 100) &
        (chain.puts['bid'] > 0.10) &
        (chain.puts['openInterest'] > 50)
    ].copy()
    
    if not puts.empty:
        puts['optionType'] = 'put'
        puts['bid_ask_spread'] = (puts['ask'] - puts['bid']) / puts['bid']
        puts = puts[puts['bid_ask_spread'] < 0.10]
        puts['symbol'] = symbol
        puts['expirationDate'] = pd.to_datetime(expiration)
        options.extend(puts.to_dict('records'))
    
    return options