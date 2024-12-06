
import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict
from config import MIN_POSITIONS,MAX_POSITIONS
from yfinance_cache import yf_cache
from column_utils import standardize_dataframe_columns, get_column_name

import logging

logger = logging.getLogger(__name__)

def analyze_selected_stocks(selected_symbols, price_data, date):
    """Enhanced stock analysis with sector information"""
    try:
        analysis = {}
        lookback_days = 252
        
        # Use cached market data
        market_data = yf_cache.get_market_data(
            start_date=(date - pd.Timedelta(days=lookback_days)),
            end_date=date
        )['Adj Close']
        
        market_returns = market_data.pct_change()
        
        # Get sector information using cache
        sectors = {}
        for symbol in selected_symbols:
            try:
                info = yf_cache.get_ticker_info(symbol)
                sector = info.get('sector', 'Unknown') if info else 'Unknown'
                sectors[symbol] = sector
            except:
                sectors[symbol] = 'Unknown'
        
        for symbol in selected_symbols:
            df, col_map = standardize_dataframe_columns(price_data[symbol])
            if df is None:
                continue
                
            returns = df[col_map['close']].pct_change()
            volume = df[col_map['volume']]
            
            analysis[symbol] = {
                'volatility': returns.std() * np.sqrt(252),
                'momentum': df[col_map['close']].iloc[-1] / df[col_map['close']].iloc[-60] - 1,
                'avg_volume': volume.mean()
            }

            
            analysis[symbol] = {
                'volatility': returns.std() * np.sqrt(252),
                'momentum': df[col_map['close']].iloc[-1] / df[col_map['close']].iloc[-60] - 1,
                'win_rate': (returns > 0).mean(),
                'sortino_ratio': returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0,
                'relative_strength': (1 + (df[col_map['close']].iloc[-1] / df[col_map['close']].iloc[0] - 1)) / 
                                   (1 + (market_data.iloc[-1] / market_data.iloc[0] - 1)),
                'sector': sectors.get(symbol, 'Unknown')
            }
            
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing stocks: {str(e)}")
        return {}

def filter_selected_stocks(selected_symbols, price_data, date):

    try:
        analysis = analyze_selected_stocks(selected_symbols, price_data, date)
        
        # Log initial selection stats
        logger.info(f"\n=== Stock Selection Analysis for {date} ===")
        logger.info(f"Initial candidates: {len(selected_symbols)}")
        
        # Track sector distribution before filtering
        initial_sectors = defaultdict(int)
        for symbol in selected_symbols:
            if symbol in analysis:
                sector = analysis[symbol]['sector']
                initial_sectors[sector] += 1
                
        logger.info("\nInitial Sector Distribution:")
        for sector, count in initial_sectors.items():
            logger.info(f"{sector}: {count} stocks")
        
        # Original filtering logic
        scored_stocks = []
        for symbol, metrics in analysis.items():
            score = (
                metrics['momentum'] * 0.5 +
                metrics['win_rate'] * 0.3 +
                metrics['relative_strength'] * 0.2
            )
            scored_stocks.append((symbol, score, metrics))
            
        # Sort by score
        sorted_stocks = sorted(scored_stocks, key=lambda x: x[1], reverse=True)
        filtered_symbols = [s[0] for s in sorted_stocks[:MAX_POSITIONS]]
        
        # Log final selection details
        logger.info(f"\nSelected {len(filtered_symbols)} stocks:")
        final_sectors = defaultdict(int)
        sector_scores = defaultdict(list)
        
        for symbol, score, metrics in sorted_stocks[:MAX_POSITIONS]:
            sector = metrics['sector']
            final_sectors[sector] += 1
            sector_scores[sector].append(score)
            logger.info(f"{symbol}: Score={score:.3f}, Sector={sector}, "
                      f"Momentum={metrics['momentum']:.2%}, Win Rate={metrics['win_rate']:.2%}")
        
        # Log sector analysis
        logger.info("\nFinal Sector Distribution:")
        for sector, count in final_sectors.items():
            avg_score = np.mean(sector_scores[sector])
            logger.info(f"{sector}: {count} stocks, Avg Score={avg_score:.3f}")
        
        # Ensure minimum positions
        if len(filtered_symbols) < MIN_POSITIONS:
            remaining = [s[0] for s in sorted_stocks[MAX_POSITIONS:]]
            filtered_symbols.extend(remaining[:(MIN_POSITIONS - len(filtered_symbols))])
            logger.info(f"\nAdded {MIN_POSITIONS - len(filtered_symbols)} additional positions to meet minimum")
            
        return filtered_symbols
        
    except Exception as e:
        logger.error(f"Error filtering stocks: {str(e)}")
        return selected_symbols