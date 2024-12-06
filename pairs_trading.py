# pairs_trading.py

from statsmodels.tsa.stattools import coint
import numpy as np
from collections import defaultdict
import logging
import pandas as pd

logger = logging.getLogger(__name__)



class SectorPairsTrading:
    def __init__(self, zscore_threshold=2.0, min_half_life=5, max_half_life=60):
        self.zscore_threshold = zscore_threshold
        self.min_half_life = min_half_life  # Trading days
        self.max_half_life = max_half_life  # Trading days
        self.active_pairs = {}
        self.pair_stats = {}
        
    def _test_cointegration(self, price1, price2):
        """
        Test cointegration between two price series
        Returns: (is_cointegrated, hedge_ratio, half_life)
        """
        try:
            # Run cointegration test
            score, pvalue, _ = coint(price1, price2)
            
            if pvalue > 0.05:  # Not cointegrated at 95% confidence
                return False, None, None
                
            # Calculate hedge ratio using OLS
            hedge_ratio = np.polyfit(price2, price1, 1)[0]
            
            # Calculate half life of mean reversion
            spread = price1 - hedge_ratio * price2
            lag_spread = spread.shift(1)
            ret_spread = spread - lag_spread
            lag_spread = lag_spread[1:]
            ret_spread = ret_spread[1:]
            b = np.polyfit(lag_spread, ret_spread, 1)[0]
            half_life = -np.log(2) / b
            
            # Check if half life is within reasonable range
            if not (self.min_half_life <= half_life <= self.max_half_life):
                return False, None, None
                
            return True, hedge_ratio, half_life
            
        except Exception as e:
            logger.error(f"Error in cointegration test: {str(e)}")
            return False, None, None
            
    def find_pairs(self, price_data, sector_map):
        """
        Find cointegrated pairs within and across sectors
        """
        pairs = []
        sectors = defaultdict(list)
        
        # Group stocks by sector
        for symbol in price_data.keys():
            sector = sector_map.get(symbol)
            if sector:
                sectors[sector].append(symbol)
        
        logger.info(f"Finding pairs across {len(sectors)} sectors")
        
        # Test pairs within sectors first (likely to be more cointegrated)
        for sector, symbols in sectors.items():
            logger.info(f"Testing pairs in {sector}")
            for i, sym1 in enumerate(symbols):
                price1 = price_data[sym1]['close']
                for j in range(i+1, len(symbols)):
                    sym2 = symbols[j]
                    price2 = price_data[sym2]['close']
                    
                    is_cointegrated, hedge_ratio, half_life = self._test_cointegration(price1, price2)
                    
                    if is_cointegrated:
                        pair_info = {
                            'symbols': (sym1, sym2),
                            'sector': sector,
                            'type': 'intra_sector',
                            'hedge_ratio': hedge_ratio,
                            'half_life': half_life
                        }
                        pairs.append(pair_info)
                        logger.info(f"Found cointegrated pair: {sym1}-{sym2} in {sector}")
        
        # Test pairs across sectors
        sector_list = list(sectors.keys())
        for i, sector1 in enumerate(sector_list):
            for sector2 in sector_list[i+1:]:
                logger.info(f"Testing pairs between {sector1} and {sector2}")
                for sym1 in sectors[sector1]:
                    price1 = price_data[sym1]['close']
                    for sym2 in sectors[sector2]:
                        price2 = price_data[sym2]['close']
                        
                        is_cointegrated, hedge_ratio, half_life = self._test_cointegration(price1, price2)
                        
                        if is_cointegrated:
                            pair_info = {
                                'symbols': (sym1, sym2),
                                'sectors': (sector1, sector2),
                                'type': 'cross_sector',
                                'hedge_ratio': hedge_ratio,
                                'half_life': half_life
                            }
                            pairs.append(pair_info)
                            logger.info(f"Found cointegrated pair: {sym1}-{sym2} between {sector1}-{sector2}")
        
        return pairs
    
    def calculate_zscore(self, price1, price2, hedge_ratio):
        """Calculate z-score of the spread"""
        spread = price1 - hedge_ratio * price2
        zscore = (spread - spread.mean()) / spread.std()
        return zscore
    
    def generate_trading_signals(self, pairs, price_data, date):
        """Generate trading signals for pairs"""
        signals = []
        
        for pair in pairs:
            sym1, sym2 = pair['symbols']
            price1 = price_data[sym1].loc[price_data[sym1]['date'] <= date, 'close'].iloc[-1]
            price2 = price_data[sym2].loc[price_data[sym2]['date'] <= date, 'close'].iloc[-1]
            
            zscore = self.calculate_zscore(price1, price2, pair['hedge_ratio'])
            
            if abs(zscore) > self.zscore_threshold:
                # Generate signal
                signal = {
                    'pair': pair['symbols'],
                    'type': pair['type'],
                    'zscore': zscore,
                    'action': 'sell_spread' if zscore > 0 else 'buy_spread',
                    'hedge_ratio': pair['hedge_ratio'],
                    'half_life': pair['half_life']
                }
                signals.append(signal)
                
        return signals
    
    def execute_pairs_trades(self, signals, holdings, cash, price_data, date, transaction_cost):
        """Execute pairs trades based on signals"""
        trades_executed = []
        
        for signal in signals:
            sym1, sym2 = signal['pair']
            hedge_ratio = signal['hedge_ratio']
            
            # Calculate position sizes
            price1 = price_data[sym1].loc[price_data[sym1]['date'] == date, 'close'].iloc[0]
            price2 = price_data[sym2].loc[price_data[sym2]['date'] == date, 'close'].iloc[0]
            
            # Determine trade size based on available cash
            max_trade_value = cash * 0.1  # Use up to 10% of cash per pair
            
            if signal['action'] == 'buy_spread':
                # Buy sym1, sell sym2
                shares1 = int(max_trade_value / (price1 * (1 + transaction_cost)))
                shares2 = int(shares1 * hedge_ratio)
                
                trades_executed.append({
                    'date': date,
                    'symbol': sym1,
                    'action': 'buy',
                    'shares': shares1,
                    'price': price1,
                    'pair_trade': True,
                    'pair_symbol': sym2
                })
                
                trades_executed.append({
                    'date': date,
                    'symbol': sym2,
                    'action': 'sell',
                    'shares': -shares2,
                    'price': price2,
                    'pair_trade': True,
                    'pair_symbol': sym1
                })
                
            else:  # sell_spread
                # Sell sym1, buy sym2
                shares1 = int(max_trade_value / (price1 * (1 + transaction_cost)))
                shares2 = int(shares1 * hedge_ratio)
                
                trades_executed.append({
                    'date': date,
                    'symbol': sym1,
                    'action': 'sell',
                    'shares': -shares1,
                    'price': price1,
                    'pair_trade': True,
                    'pair_symbol': sym2
                })
                
                trades_executed.append({
                    'date': date,
                    'symbol': sym2,
                    'action': 'buy',
                    'shares': shares2,
                    'price': price2,
                    'pair_trade': True,
                    'pair_symbol': sym1
                })
        
        return trades_executed