# backtesting_module.py

import pandas as pd
import numpy as np
from data_preparation import  fetch_treasury_rates

from portfolio_state import PortfolioState
from portfolio_base import PortfolioState
from config import (
    RISK_FREE_RATE, TOP_N, STOP_LOSS, TRAILING_STOP, RE_ENTRY_THRESHOLD, 
    TREASURY_FALLBACK_RATES, MAX_POSITION_SIZE, MAX_OUT_OF_MARKET_DAYS, 
    FORCE_REENTRY_RECOVERY, MIN_EXPECTED_RETURN, MIN_POSITIONS, MIN_POSITION_SIZE, PREDICTION_THRESHOLD 
)
from config import (
    MAX_DRAWDOWN, MAX_VOLATILITY, VOL_WINDOW, 
    RISK_CHECK_FREQUENCY, FORCE_LIQUIDATION,
    MIN_RECOVERY, MAX_CONSECUTIVE_LOSSES, TRANSACTION_COST, MAX_RECOVERY_DAYS, EQUITY_START_DATE, EQUITY_END_DATE, BACKTEST_START_DATE, MINIMUM_HISTORY_YEARS
)
from config import ALPHAVANTAGE_API_KEY
import requests

from black_scholes import black_scholes_price
from portfolio_optimizer import EnhancedPortfolioOptimizer
from options_trading import EnhancedOptionsStrategy
from predictor import EnhancedPredictor
from options_report import OptionsReport
from data_utils import validate_date_ranges
from stock_analyzer import filter_selected_stocks

from options_model import OptionsPricingMLP
#from debug_utils import debug_ml_data, debug_stock_selection, debug_portfolio_optimization  

import yfinance as yf
from visualization import plot_portfolio_vs_market, save_results
from risk_management import ETFRiskManager
from model_builder import train_etf_model

import logging
import traceback
from datetime import datetime, date
from collections import defaultdict
import itertools 
from yfinance_cache import yf_cache
from pairs_trading import SectorPairsTrading
from data_fetcher import fetch_sp500_constituents

import time
from typing import List, Dict, Optional, Tuple, Union

from performance_monitor import PerformanceMonitor

#from rebalancing_debug import inject_rebalancing_debug
from column_utils import standardize_dataframe_columns, get_column_name

logger = logging.getLogger(__name__)

class RebalanceTracker:
    def __init__(self):
        self.attempts = []
        self.successes = []
        self.failures = []
        self.data_availability = {}
        
    def add_attempt(self, date, success, available_symbols, total_symbols, reason=None):
        attempt = {
            'date': date,
            'success': success,
            'available_symbols': available_symbols,
            'total_symbols': total_symbols,
            'availability_ratio': available_symbols / total_symbols,
            'reason': reason
        }
        self.attempts.append(attempt)
        if success:
            self.successes.append(attempt)
        else:
            self.failures.append(attempt)
            
    def print_summary(self):
        logger.info("\n=== Rebalancing Summary ===")
        logger.info(f"Total attempts: {len(self.attempts)}")
        logger.info(f"Successful: {len(self.successes)}")
        logger.info(f"Failed: {len(self.failures)}")
        
        if self.attempts:
            success_rate = (len(self.successes) / len(self.attempts)) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
            
            monthly_stats = self._calculate_monthly_stats()
            self._print_monthly_patterns(monthly_stats)

    def _calculate_monthly_stats(self):
        monthly_stats = {}
        for attempt in self.attempts:
            month = attempt['date'].month
            if month not in monthly_stats:
                monthly_stats[month] = {'total': 0, 'success': 0}
            monthly_stats[month]['total'] += 1
            if attempt['success']:
                monthly_stats[month]['success'] += 1
        return monthly_stats

    def _print_monthly_patterns(self, monthly_stats):
        logger.info("\nMonthly Success Patterns:")
        for month in sorted(monthly_stats.keys()):
            stats = monthly_stats[month]
            success_rate = (stats['success'] / stats['total']) * 100
            logger.info(f"Month {month}: {success_rate:.2f}% success rate "
                      f"({stats['success']}/{stats['total']} attempts)")
            

class ETFRebalanceTracker(RebalanceTracker):
    """ETF-specific rebalance tracking"""
    def __init__(self):
        super().__init__()
        self.sector_rebalances = defaultdict(list)
        self.etf_tracking_error = []

    def add_attempt(self, date, success, available_symbols, total_symbols, reason=None, sector_impact=None):
        # Call parent first
        super().add_attempt(date, success, available_symbols, total_symbols, reason)
        
        # Add ETF-specific tracking
        if sector_impact:
            for sector, change in sector_impact.items():
                self.sector_rebalances[sector].append({
                    'date': date,
                    'change': change
                })
    
    def print_summary(self):
        # First print parent summary
        super().print_summary()
        
        # Then add ETF-specific summary
        logger.info("\n=== ETF Rebalancing Summary ===")
        for sector, changes in self.sector_rebalances.items():
            logger.info(f"\nSector {sector}:")
            logger.info(f"Total changes: {len(changes)}")
            if changes:
                avg_change = np.mean([c['change'] for c in changes])
                logger.info(f"Average change: {avg_change:.2%}")


class RiskManager:
    """Fine-tuned risk manager with regime awareness"""
    
    def __init__(self, stop_loss=0.10, trailing_stop=0.15, re_entry_threshold=0.02):
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.re_entry_threshold = re_entry_threshold
        self.highest_value = None
        self.in_market = True
        self.exit_value = None
        self.exit_date = None
        self.lowest_value_since_exit = None
        self.market_conditions = {}
        self.recovery_attempts = []
        self.risk_events = [] 
        
        # Adjusted regime thresholds
        self.regime_adjustments = {
            'high_volatility': {
                'stop_loss': 0.12,      # More room in high vol
                'trailing_stop': 0.15,
                're_entry_threshold': 0.02
            },
            'strong_trend': {
                'stop_loss': 0.15,      # Even more room in trends
                'trailing_stop': 0.18,
                're_entry_threshold': 0.01  # Faster re-entry in trends
            },
            'normal': {
                'stop_loss': 0.13,      # Slightly more room than original
                'trailing_stop': 0.16,
                're_entry_threshold': 0.015
            }
        }
        self.volatility_cache = {}

    def _detect_regime(self, date):
        try:
            # Use AlphaVantage API with explicit comparisons
            vix_data = yf_cache.get_market_data(
                start_date=date - pd.Timedelta(days=30),
                end_date=date,
                symbol='^VIX'
            )['Close']
            
            if len(vix_data) == 0:
                return 'normal'
                
            current_vix = float(vix_data.iloc[-1])  # Convert to scalar
            vix_ma20 = float(vix_data.rolling(20).mean().iloc[-1])  # Convert to scalar
            
            # Get market trend data using cache
            market_data = yf_cache.get_market_data(
                start_date=date - pd.Timedelta(days=60),
                end_date=date,
                symbol='SPY'
            )['Close']
            
            trend_strength = 0
            if len(market_data) > 20:
                ma20 = float(market_data.rolling(20).mean().iloc[-1])  # Convert to scalar
                ma50 = float(market_data.rolling(50).mean().iloc[-1])  # Convert to scalar
                price = float(market_data.iloc[-1])  # Convert to scalar
                
                # Calculate trend score with explicit comparisons
                trend_strength = sum([
                    float(price > ma20),
                    float(ma20 > ma50),
                    float(price > float(market_data.shift(20).iloc[-1])),
                    float(ma20 > float(market_data.rolling(20).mean().shift(20).iloc[-1]))
                ]) / 4
            
            # Explicit scalar comparisons
            if (current_vix > 30) and (current_vix > vix_ma20 * 1.2):
                return 'high_volatility'
            elif trend_strength > 0.75 and current_vix < 20:
                return 'strong_trend'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return 'normal'

    def check_stops(self, current_value, date, holdings=None):
        """Enhanced stop check with smoothed regime transitions"""
        try:
            if not self.in_market:
                return self.check_reentry(current_value, date)
                
            # Ensure we're working with scalar values, not Series
            current_value = float(current_value)
            if self.highest_value is None or current_value > self.highest_value:
                self.highest_value = float(current_value)
                
            # Calculate drawdown (using scalar values)
            drawdown = (self.highest_value - current_value) / self.highest_value
            
            # Get market volatility for adjustment
            market_vol = self._get_market_volatility(date)
            if isinstance(market_vol, pd.Series):
                market_vol = float(market_vol.iloc[-1])
            baseline_vol = 0.20
            
            # Get current regime
            regime = self._detect_regime(date)
            thresholds = self.regime_adjustments[regime]
            
            # Adjust stops based on market conditions (all scalar operations)
            vol_adjustment = min(1.2, max(0.8, baseline_vol / market_vol))
            stop_loss = thresholds['stop_loss'] * vol_adjustment
            trailing_stop = thresholds['trailing_stop'] * vol_adjustment
            
            # Check stops with explicit scalar comparisons
            if drawdown >= stop_loss:
                # Hard stop loss hit
                self.in_market = False
                self.exit_value = float(current_value)
                self.exit_date = date
                self.lowest_value_since_exit = float(current_value)
                
                logger.info(f"\nStop loss triggered on {date}:")
                logger.info(f"Drawdown: {drawdown:.1%}")
                logger.info(f"Regime: {regime}")
                return False
                
            elif current_value <= (self.highest_value * (1 - trailing_stop)):
                # Trailing stop hit
                self.in_market = False
                self.exit_value = float(current_value)
                self.exit_date = date
                self.lowest_value_since_exit = float(current_value)
                
                logger.info(f"\nTrailing stop triggered on {date}:")
                logger.info(f"Decline from peak: {((self.highest_value - current_value) / self.highest_value):.1%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in check_stops: {str(e)}")
            return True

    def check_reentry(self, current_value, date, price_series, recovery_from_bottom, base_threshold, market_momentum, days_out):
        """Enhanced re-entry with faster triggers in strong trends"""
        try:
            days_out = (date - self.exit_date).days
            
            if self.lowest_value_since_exit is None or current_value < self.lowest_value_since_exit:
                self.lowest_value_since_exit = current_value
            
            # Get market condition metrics
            market_data = self._get_market_data(date)
            market_momentum = self._calculate_momentum(market_data)
            is_strong_uptrend = self._check_uptrend(market_data)
            regime = self.detect_regime(price_series)

            
            # Calculate recovery metrics
            recovery_from_exit = (current_value - self.exit_value) / self.exit_value
            recovery_from_bottom = (current_value - self.lowest_value_since_exit) / self.lowest_value_since_exit
            
            # Adjusted re-entry thresholds
            base_threshold = self.regime_adjustments[regime]['re_entry_threshold']
            
            # More aggressive re-entry in strong trends
            if regime == 'strong_trend':
                should_reenter = (
                    recovery_from_bottom > base_threshold * 0.5 or  # Lower threshold
                    market_momentum > 0.01                          # Any positive momentum
                ) and days_out >= 3                                # Minimum wait
                
            elif regime == 'high_volatility':
                should_reenter = (
                    recovery_from_bottom > base_threshold * 1.2 and  # Higher threshold
                    market_momentum > 0.02 and                       # Strong momentum
                    days_out >= 5                                    # Longer wait
                )
            else:  # Normal regime
                should_reenter = (
                    recovery_from_bottom > base_threshold and
                    market_momentum > 0 and
                    days_out >= 4
                )
            
            # Force re-entry conditions (more aggressive)
            force_reentry = (
                days_out >= 12 or                              # Shorter maximum wait
                recovery_from_bottom > 0.08 or                 # Lower recovery threshold
                (is_strong_uptrend and market_momentum > 0.03) # Faster trend following
            )
            
            if should_reenter or force_reentry:
                self.in_market = True
                self.highest_value = current_value
                self.exit_value = None
                self.exit_date = None
                self.lowest_value_since_exit = None
                
                logger.info(f"\nRe-entry triggered on {date}:")
                logger.info(f"Days out: {days_out}")
                logger.info(f"Recovery: {recovery_from_bottom:.1%}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in check_reentry: {str(e)}")
            return False

    def _get_market_volatility(self, date):
        """Get market volatility with caching"""
        try:
            if date not in self.volatility_cache:
                market_data = yf.download('SPY', 
                                        start=date - pd.Timedelta(days=60),
                                        end=date)['Close']
                returns = market_data.pct_change()
                vol = returns.std() * np.sqrt(252)
                self.volatility_cache[date] = vol
                
            return self.volatility_cache[date]
            
        except Exception as e:
            logger.error(f"Error getting market volatility: {str(e)}")
            return 0.15

    def _calculate_momentum(self, market_data):
        """Calculate market momentum"""
        try:
            return market_data.iloc[-1] / market_data.iloc[-20] - 1
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return 0

    def _check_uptrend(self, market_data):
        """Check for strong uptrend"""
        try:
            ma20 = market_data.rolling(20).mean()
            ma50 = market_data.rolling(50).mean()
            
            return (
                market_data.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1] and
                ma20.iloc[-1] > ma20.iloc[-20]
            )
        except Exception as e:
            logger.error(f"Error checking uptrend: {str(e)}")
            return False

    def _get_market_data(self, date):
        """Get market data for analysis"""
        try:
            return yf_cache.get_market_data(
                start_date=date - pd.Timedelta(days=60),
                end_date=date,
                symbol='SPY'
            )['Close']
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return pd.Series()

    def print_summary(self):
        """Print comprehensive risk management summary"""
        try:
            logger.info("\n=== Risk Management Summary ===")
            logger.info(f"Total Events: {len(self.events)}")
            
            # Analyze stops triggered
            stops = [e for e in self.events if e['type'] == 'stop_triggered']
            logger.info(f"\nStops Triggered: {len(stops)}")
            if stops:
                avg_drawdown = np.mean([s['drawdown'] for s in stops])
                logger.info(f"Average Drawdown at Stop: {avg_drawdown:.1%}")
            
            # Analyze recoveries
            logger.info(f"\nRecovery Attempts: {len(self.recovery_attempts)}")
            if self.recovery_attempts:
                avg_days_out = np.mean([r['days_out'] for r in self.recovery_attempts])
                avg_recovery = np.mean([r['recovery'] for r in self.recovery_attempts])
                logger.info(f"Average Days Out: {avg_days_out:.1f}")
                logger.info(f"Average Recovery: {avg_recovery:.1%}")
            
        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}")


class ETFRiskManager(RiskManager):
    """ETF-specific risk management"""
    def __init__(self, stop_loss=0.10, trailing_stop=0.15, re_entry_threshold=0.02):
        super().__init__(stop_loss, trailing_stop, re_entry_threshold)
        # Add ETF-specific risk parameters
        self.max_sector_deviation = 0.05
        self.max_tracking_error = 0.02
        self.sector_exposures = defaultdict(list)

    def check_stops(self, current_value, date, holdings=None, sector_weights=None):
        """Override with ETF-specific checks"""
        if not super().check_stops(current_value, date, holdings):
            return False
            
        #  ETF-specific checks
        if sector_weights:
            sector_deviation = self._check_sector_deviation(sector_weights)
            if sector_deviation > self.max_sector_deviation:
                return False
        return True

class TradeTracker:
    """Enhanced trade tracking with separate equity and options counts"""
    
    def __init__(self):
        self.trades = []
        self.total_transaction_costs = 0
        self.equity_trades = 0
        self.option_trades = 0
        self.monthly_activity = {}
        self.positions = {}
        self.risk_events = []
        
    def record_trade(self, date, action, symbol, shares, price, value, cost, 
                    trade_type='equity', pair_trade=False, pair_symbol=None,
                    expected_profit=None):
        """Record trade with enhanced tracking"""
        try:
            trade = {
                'date': pd.to_datetime(date),
                'action': action,
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'value': value,
                'cost': cost,
                'trade_type': trade_type,
                'pair_trade': pair_trade,
                'pair_symbol': pair_symbol,
                'expected_profit': expected_profit
            }
            
            self.trades.append(trade)
            self.total_transaction_costs += cost
            
            # Update trade counts
            if trade_type == 'option':
                self.option_trades += 1
            else:
                self.equity_trades += 1
                
            # Update monthly activity
            month_key = pd.to_datetime(date).strftime('%Y-%m')
            if month_key not in self.monthly_activity:
                self.monthly_activity[month_key] = {
                    'equity_trades': 0,
                    'option_trades': 0,
                    'equity_volume': 0,
                    'option_volume': 0
                }
                
            activity = self.monthly_activity[month_key]
            if trade_type == 'option':
                activity['option_trades'] += 1
                activity['option_volume'] += abs(value)
            else:
                activity['equity_trades'] += 1
                activity['equity_volume'] += abs(value)
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _update_activity_metrics(self, date, action, symbol, shares, value, cost):
        """Track activity metrics"""
        month_key = date.strftime('%Y-%m')
        if month_key not in self.monthly_activity:
            self.monthly_activity[month_key] = {'buys': 0, 'sells': 0, 'volume': 0}
        
        if action == 'buy':
            self._handle_buy(month_key, symbol, shares)
        elif action == 'sell':
            self._handle_sell(month_key, symbol, shares)
        
        self.monthly_activity[month_key]['volume'] += value
        self.total_transaction_costs += cost

    def _handle_buy(self, month_key, symbol, shares):
        self.buy_trades += 1
        self.monthly_activity[month_key]['buys'] += 1
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += shares

    def _handle_sell(self, month_key, symbol, shares):
        self.sell_trades += 1
        self.monthly_activity[month_key]['sells'] += 1
        if symbol in self.positions:
            self.positions[symbol] -= abs(shares)
            if self.positions[symbol] == 0:
                del self.positions[symbol]

    def print_summary(self):
        """Print comprehensive trading summary"""
        try:
            logger.info("\n=== Trading Summary ===")
            logger.info(f"Total Equity Trades: {self.equity_trades}")
            logger.info(f"Total Option Trades: {self.option_trades}")
            logger.info(f"Total Transaction Costs: ${self.total_transaction_costs:,.2f}")

            # Monthly breakdown
            logger.info("\nMonthly Trading Activity:")
            for month, activity in sorted(self.monthly_activity.items()):
                logger.info(f"\n{month}:")
                logger.info(f"Equity Trades: {activity['equity_trades']}")
                logger.info(f"Option Trades: {activity['option_trades']}")
                logger.info(f"Equity Volume: ${activity['equity_volume']:,.2f}")
                logger.info(f"Option Volume: ${activity['option_volume']:,.2f}")

            # Recent trades
            if self.trades:
                logger.info("\nLast 5 Trades:")
                for trade in self.trades[-5:]:
                    trade_type = 'OPTION' if trade['trade_type'] == 'option' else 'EQUITY'
                    logger.info(
                        f"{trade['date']}: {trade_type} {trade['action']} {abs(trade['shares'])} "
                        f"{trade['symbol']} @ ${trade['price']:.2f} "
                        f"(Value: ${trade['value']:,.2f}, Cost: ${trade['cost']:,.2f})"
                    )

        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}")

    def _print_trade_metrics(self):
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Buy Trades: {self.buy_trades}")
        logger.info(f"Sell Trades: {self.sell_trades}")
        logger.info(f"Total Transaction Costs: ${self.total_transaction_costs:,.2f}")

    def _print_monthly_activity(self):
        logger.info("\nMonthly Trading Activity:")
        for month, activity in sorted(self.monthly_activity.items()):
            logger.info(f"\n{month}:")
            logger.info(f"- Buys: {activity['buys']}")
            logger.info(f"- Sells: {activity['sells']}")
            logger.info(f"- Trading Volume: ${activity['volume']:,.2f}")

    def _print_recent_trades(self):
        if self.trades:
            logger.info("\nLast 5 Trades:")
            for trade in self.trades[-5:]:
                logger.info(
                    f"{trade['date']}: {trade['action']} {abs(trade['shares'])} "
                    f"{trade['symbol']} @ ${trade['price']:.2f} "
                    f"(Value: ${trade['value']:,.2f}, Cost: ${trade['cost']:,.2f})"
                )

    def _print_risk_events(self):
        """Print risk event summary"""
        if self.risk_events:
            logger.info("\nRisk Events Summary:")
            for event in self.risk_events:
                logger.info(f"Date: {event['date']}, Type: {event['type']}")
                if isinstance(event['details'], dict):
                    for key, value in event['details'].items():
                        logger.info(f"- {key}: {value}")
                else:
                    logger.info(f"- Details: {event['details']}")





def prepare_treasury_data(treasury_data):
    """Prepare treasury data with fallback"""
    if treasury_data is None:
        treasury_data = fetch_treasury_rates()
        if treasury_data is None:
            logger.warning("Using fallback treasury rates")
            treasury_data = pd.DataFrame([TREASURY_FALLBACK_RATES])
    return treasury_data




def prepare_price_data(self, price_data: Dict[str, pd.DataFrame], 
                      start_date: Union[str, datetime, pd.Timestamp], 
                      end_date: Union[str, datetime, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    try:
        combined_prices = pd.DataFrame()
        for symbol, df in price_data.items():
            try:
                df = df.copy()
                df, col_map = standardize_dataframe_columns(df)
                if df is None:
                    continue

                if 'date' in col_map and col_map['close'] in df.columns:
                    df_subset = df[[col_map['date'], col_map['close']]].copy()
                    df_subset = df_subset.rename(columns={col_map['close']: symbol})
                    df_subset.set_index(col_map['date'], inplace=True)

                    if combined_prices.empty:
                        combined_prices = df_subset
                    else:
                        combined_prices = combined_prices.join(df_subset, how='outer')

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

        if combined_prices.empty:
            logger.error("No valid price data")
            return pd.DataFrame(), pd.DatetimeIndex([])

        combined_prices.sort_index(inplace=True)
        date_range = combined_prices.index[
            (combined_prices.index >= pd.to_datetime(start_date)) &
            (combined_prices.index <= pd.to_datetime(end_date))
        ]

        return combined_prices, date_range

    except Exception as e:
        logger.error(f"Error preparing price data: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DatetimeIndex([])
    
def validate_data_availability(price_data, date_range):
    """Check data availability across key dates"""
    sample_dates = [date_range[0], date_range[len(date_range)//2], date_range[-1]]
    for date in sample_dates:
        logger.info(f"\nChecking data for {date}:")
        available_data = 0
        total_symbols = len(price_data)
        
        for symbol in price_data:
            if date in price_data[symbol]['date'].values:
                price = price_data[symbol].loc[
                    price_data[symbol]['date'] == date, 'close'
                ].iloc[0]
                logger.info(f"{symbol}: ${price:.2f}")
                available_data += 1
            else:
                logger.info(f"{symbol}: No data")
        
        coverage = (available_data / total_symbols) * 100
        logger.info(f"Data coverage: {coverage:.1f}%")
        
        if coverage < 90:
            logger.warning(f"Low data coverage on {date}")

def calculate_portfolio_metrics(self,portfolio_value_series: pd.Series,
                           initial_capital: float,
                           benchmark_series: pd.Series = None,
                           performance_monitor=None) -> dict:
    """Calculate portfolio metrics including benchmark comparison"""
    try:

        state = PortfolioState(initial_capital)
        # Calculate basic returns
        returns = state._state['portfolio_value'].pct_change().dropna()
        if len(returns) < 2:
            return self._create_empty_metrics()
        total_return = (state._state['portfolio_value'].iloc[-1]  / state._state['initial_capital'] - 1) * 100
        trading_days = len(portfolio_value_series)
        annual_return = ((1 + total_return/100) ** (252/trading_days) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Calculate Sharpe ratio
        excess_returns = returns - (RISK_FREE_RATE / 252)
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # Calculate maximum drawdown
        peak = portfolio_value_series.expanding(min_periods=1).max()
        drawdown = ((portfolio_value_series - peak) / peak)
        max_drawdown = drawdown.min() * 100
        
        # Calculate win rate
        win_rate = (returns > 0).mean() * 100

         # Avoid division by zero
        std_dev = returns.std()
        if std_dev == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (returns.mean() * 252 - RISK_FREE_RATE) / (std_dev * np.sqrt(252))

        
        metrics = {
            'Total Return (%)': total_return,
            'Annual Return (%)': annual_return,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Number of Trades': trading_days
        }
        
        # Add ETF-specific metrics
        if benchmark_series is not None:
            tracking_metrics = calculate_tracking_metrics(
                portfolio_value_series, 
                benchmark_series
            )
            metrics.update(tracking_metrics)
            
        if performance_monitor:
            sector_metrics = performance_monitor._calculate_sector_metrics()
            metrics.update({
                'Sector Metrics': sector_metrics
            })

        # Add benchmark comparison if available
        if benchmark_series is not None and not benchmark_series.empty:
            # Calculate benchmark metrics
            bench_returns = benchmark_series.pct_change().dropna()
            bench_total_return = (benchmark_series.iloc[-1] / initial_capital - 1) * 100
            bench_volatility = bench_returns.std() * np.sqrt(252) * 100
            
            # Calculate alpha and beta
            covariance = np.cov(returns, bench_returns)[0][1]
            variance = np.var(bench_returns)
            beta = covariance / variance if variance != 0 else 1
            
            alpha = annual_return - (RISK_FREE_RATE + beta * (bench_total_return - RISK_FREE_RATE))
            
            # Calculate tracking error
            tracking_error = (returns - bench_returns).std() * np.sqrt(252) * 100
            
            # Calculate information ratio
            excess_return = total_return - bench_total_return
            info_ratio = excess_return / tracking_error if tracking_error != 0 else 0
            
            metrics.update({
                'Alpha': alpha,
                'Beta': beta,
                'Tracking Error (%)': tracking_error,
                'Information Ratio': info_ratio,
                'Benchmark Return (%)': bench_total_return,
                'Active Return (%)': total_return - bench_total_return
            })
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        return {
            'Total Return (%)': 0.0,
            'Annual Return (%)': 0.0,
            'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown (%)': 0.0,
            'Win Rate (%)': 0.0,
            'Number of Trades': 0
        }
    

def calculate_tracking_metrics(portfolio_series: pd.Series, 
                             benchmark_series: pd.Series) -> Dict:
    """Calculate ETF tracking metrics"""
    try:
        # Align dates
        common_dates = portfolio_series.index.intersection(benchmark_series.index)
        portfolio_returns = portfolio_series[common_dates].pct_change()
        benchmark_returns = benchmark_series[common_dates].pct_change()
        
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        beta = np.cov(portfolio_returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
        
        return {
            'tracking_error': tracking_error,
            'beta': beta,
            'correlation': portfolio_returns.corr(benchmark_returns)
        }
    except Exception as e:
        logger.error(f"Error calculating tracking metrics: {str(e)}")
        return {}

def update_portfolio_state(state, date, combined_prices):
    """Update portfolio state with current values"""
    try:
        # Calculate equity value
        equity_value = sum(
            shares * combined_prices.loc[date, symbol]
            for symbol, shares in state['holdings']['equities'].items()
        )
        
        # Calculate options value (if any)
        options_value = 0  # Simplified for now
        
        # Update state values
        state['equity_value'] = equity_value
        state['options_value'] = options_value
        state['total_value'] = state['cash'] + equity_value + options_value
        
        # Update portfolio value series
        if not isinstance(state['portfolio_value'], pd.Series):
            state['portfolio_value'] = pd.Series(dtype=float)
        state['portfolio_value'].loc[date] = state['total_value']
        
        logger.debug(f"Updated state for {date}:")
        logger.debug(f"Cash: ${state['cash']:,.2f}")
        logger.debug(f"Equity: ${equity_value:,.2f}")
        logger.debug(f"Total: ${state['total_value']:,.2f}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error updating portfolio state: {str(e)}")
        logger.error(traceback.format_exc())
        return state




def execute_buy(state, date, symbol, shares, price, trade_value, trade_cost, trackers):
    """Execute a buy trade"""
    state['cash'] -= (trade_value + trade_cost)
    state['holdings']['equities'][symbol] = shares
    
    trackers['trade'].record_trade(
        date, 'buy', symbol, shares, price, trade_value, trade_cost
    )
    
    trackers['transactions'].append({
        'date': date,
        'symbol': symbol,
        'action': 'buy',
        'quantity': shares,
        'price': price,
        'value': trade_value,
        'cost': trade_cost,
        'reason': 'rebalance'
    })

def sell_all_positions(state, date, combined_prices, transaction_cost, trackers):
    """Sell all existing positions"""
    try:
        for symbol, shares in list(state['holdings']['equities'].items()):
            price = combined_prices.loc[date, symbol]
            sale_value = shares * price * (1 - transaction_cost)
            trade_cost = shares * price * transaction_cost
            
            state['cash'] += sale_value
            
            trackers['trade'].record_trade(
                date, 'sell', symbol, -shares, price, sale_value, trade_cost
            )
            
            trackers['transactions'].append({
                'date': date,
                'symbol': symbol,
                'action': 'sell',
                'quantity': -shares,
                'price': price,
                'value': sale_value,
                'cost': trade_cost,
                'reason': 'rebalance'
            })
        
        state['holdings']['equities'] = {}
        return state
    except Exception as e:
        logger.error(f"Error selling positions: {str(e)}")
        return state
    
def need_to_rebalance(date, last_rebalance, frequency):
    """Determines if rebalancing is needed based on the frequency"""
    try:
        logger.info(f"\nRebalance Check for {date}:")
        logger.info(f"Last rebalance: {last_rebalance}")
        logger.info(f"Frequency: {frequency}")
        logger.info(f"Current date components: Year={date.year}, Month={date.month}")
        logger.info(f"Last rebalance components: Year={last_rebalance.year}, Month={last_rebalance.month}")

        need_rebalance = False
        if frequency == 'M':
            current_period = date.year * 12 + date.month
            last_period = last_rebalance.year * 12 + last_rebalance.month
            need_rebalance = current_period > last_period
        elif frequency == 'W':
            need_rebalance = date.isocalendar()[1] != last_rebalance.isocalendar()[1]
        elif frequency == 'D':
            need_rebalance = True
            
        logger.info(f"Need rebalance: {need_rebalance}")
        return need_rebalance
        
    except Exception as e:
        logger.error(f"Error in need_to_rebalance: {str(e)}")
        logger.error(traceback.format_exc())
        return True  # Default to rebalance on error
    
def debug_rebalancing_step(date, state, df_ml_data, selected_symbols, expected_returns=None):
    """Debug helper to track rebalancing process"""
    logger.info("\nDEBUG REBALANCING STEP")
    logger.info(f"Date: {date}")
    logger.info(f"Portfolio Value: ${state['total_value']:,.2f}")
    logger.info(f"Cash: ${state['cash']:,.2f}")
    
    # Step 1: ML Data Check
    logger.info("\nStep 1: ML Data Preparation")
    if df_ml_data is None or df_ml_data.empty:
        logger.error("ML data preparation failed")
        logger.info(f"Features available: {df_ml_data.columns.tolist() if df_ml_data is not None else 'None'}")
        return False
    logger.info(f"ML data shape: {df_ml_data.shape}")
    
    # Step 2: Symbol Selection
    logger.info("\nStep 2: Symbol Selection")
    if not selected_symbols:
        logger.error("No symbols selected")
        return False
    logger.info(f"Selected symbols: {selected_symbols}")
    logger.info(f"Number of symbols: {len(selected_symbols)}")
    
    # Step 3: Expected Returns
    logger.info("\nStep 3: Expected Returns")
    if expected_returns is None:
        logger.error("No expected returns calculated")
        return False
    if isinstance(expected_returns, pd.Series):
        logger.info(f"Max expected return: {expected_returns.max():.4f}")
        logger.info(f"Min expected return: {expected_returns.min():.4f}")
        logger.info(f"Mean expected return: {expected_returns.mean():.4f}")
    
    return True



def log_portfolio_summary(state, date):
    """Log detailed portfolio summary"""
    logger.info(f"\nPortfolio Summary for {date}:")
    logger.info(f"Cash: ${state['cash']:,.2f}")
    logger.info(f"Equity Value: ${state['equity_value']:,.2f}")
    logger.info(f"Options Value: ${state['options_value']:,.2f}")
    logger.info(f"Total Value: ${state['total_value']:,.2f}")
    
    if len(state['holdings']['equities']) > 0:
        logger.info("\nHoldings:")
        for symbol, shares in state['holdings']['equities'].items():
            try:
                value = shares * state['combined_prices'].loc[date, symbol]
                weight = value / state['total_value']
                logger.info(f"- {symbol}: {shares} shares (${value:,.2f}, {weight:.2%})")
            except KeyError:
                logger.warning(f"Price data not available for {symbol} on {date}")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")


def select_initial_portfolio(price_data, start_date, top_n):
    """Select initial portfolio based on fundamental criteria"""
    try:
        initial_candidates = []
        
        for symbol, df in price_data.items():
            # Get data before start date
            hist_data = df[df['date'] < start_date].tail(252)  # 1 year of history
            if len(hist_data) < 252:
                continue
                
            # Calculate basic metrics
            returns = hist_data['close'].pct_change()
            metrics = {
                'symbol': symbol,
                'sharpe': (returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252)),
                'volatility': returns.std() * np.sqrt(252),
                'momentum': hist_data['close'].iloc[-1] / hist_data['close'].iloc[-60] - 1,  # 3-month momentum
                'drawdown': (hist_data['close'] / hist_data['close'].cummax() - 1).min()
            }
            
            # Filter based on basic criteria
            if (metrics['sharpe'] > 0.5 and  # Decent Sharpe ratio
                metrics['volatility'] < 0.4 and  # Not too volatile
                metrics['momentum'] > 0 and  # Positive momentum
                metrics['drawdown'] > -0.2):  # Not in deep drawdown
                initial_candidates.append(metrics)
        
        if not initial_candidates:
            return []
            
        # Sort by Sharpe ratio and take top N
        selected = sorted(initial_candidates, key=lambda x: x['sharpe'], reverse=True)[:top_n]
        return [s['symbol'] for s in selected]
        
    except Exception as e:
        logger.error(f"Error selecting initial portfolio: {str(e)}")
        return []
    
def check_market_regime(date, lookback_days=60):
    """Check overall market regime before trading"""
    try:
        market_data = yf.download('SPY', 
                                start=(date - pd.Timedelta(days=lookback_days)),
                                end=date)['Adj Close']
        
        # Calculate market metrics
        market_return = (market_data.iloc[-1] / market_data.iloc[0] - 1)
        market_vol = market_data.pct_change().std() * np.sqrt(252)
        sma_50 = market_data.rolling(50).mean()
        
        regime = {
            'bullish': market_return > 0 and market_data.iloc[-1] > sma_50.iloc[-1],
            'volatility': market_vol,
            'trend_strength': abs(market_return) / market_vol if market_vol > 0 else 0
        }
        
        return regime
        
    except Exception as e:
        logger.error(f"Error checking market regime: {str(e)}")
        return {'bullish': True, 'volatility': 0, 'trend_strength': 0}




class BacktestEngine:
    """Main backtesting engine with organized workflow"""
    
    def __init__(self, initial_capital: float, data_pipeline, strategy_type: str = 'etf'):
        self.initial_capital = initial_capital
        #from data_preparation import ETFDataPipeline
        self.data_pipeline = data_pipeline
        self.portfolio_optimizer = EnhancedPortfolioOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.risk_manager = RiskManager(STOP_LOSS, TRAILING_STOP, RE_ENTRY_THRESHOLD)
        self.options_strategy = EnhancedOptionsStrategy(min_mispricing=0.02) 
        self.pairs_strategy = SectorPairsTrading()
         # Calculate required history start date
        self.required_history_start = pd.to_datetime(BACKTEST_START_DATE) - pd.DateOffset(years=MINIMUM_HISTORY_YEARS)
        self.strategy_type = strategy_type

              # ETF-specific constraints
        self.etf_constraints = {
            'max_etf_allocation': 0.30,  # Maximum allocation to ETFs
            'min_etf_volume': 100000,    # Minimum ETF daily volume
            'max_tracking_error': 0.02,  # Maximum tracking error for ETFs
            'min_holdings_overlap': 0.30  # Minimum holdings overlap for related ETFs
        }
        
        # Sector constraints with ETF consideration
        self.sector_limits = {
            'Information Technology': 0.25,
            'Financials': 0.20,
            'Health Care': 0.20,
            'Consumer Discretionary': 0.15,
            'Industrials': 0.15,
            'Consumer Staples': 0.15,
            'Energy': 0.12,
            'Materials': 0.12,
            'Utilities': 0.10,
            'Real Estate': 0.10,
            'Communication Services': 0.15
        }
        
        # Enhanced tracking
        self.etf_holdings_history = {}
        self.etf_tracking_errors = defaultdict(list)
        self.sector_exposures = defaultdict(list)

        # Initialize components based on strategy type
        self.risk_manager = ETFRiskManager() if strategy_type == 'etf' else RiskManager()
        self.rebalance_tracker = ETFRebalanceTracker() if strategy_type == 'etf' else RebalanceTracker()
        
        

    
    def initialize_portfolio_state(self,
                                price_data: Dict[str, pd.DataFrame],
                                options_data: Dict[str, pd.DataFrame],
                                start_date: str,
                                end_date: str,
                                initial_capital: float,
                                option_pricing_model: Optional[object] = None,
                                treasury_data: Optional[Dict] = None,
                                sector_map: Optional[Dict[str, str]] = None) -> Dict:
        """Initialize portfolio state with data"""
        try:
            # Convert dates
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Prepare data - using the instance method
            combined_prices, date_range = self._prepare_price_data(
                price_data=price_data, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if len(date_range) == 0:
                raise ValueError("No valid dates in range")
                
            # Initialize portfolio state
            state = PortfolioState(initial_capital, combined_prices)
            
            # Store additional state information
            state._state.update({
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'date_range': date_range,
                'price_data': price_data,
                'options_data': options_data,
                'treasury_data': treasury_data,
                'sector_map': sector_map,
                'last_rebalance': start_date - pd.Timedelta(days=1),
                'options_enabled': False,
                'option_pricing_model': option_pricing_model,
                'in_recovery': False
            })
            
            # Validate options if available
            if option_pricing_model and option_pricing_model.is_trained:
                state._state['options_enabled'] = True
                logger.info("Options trading enabled")
                
            return state
                
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Failed to initialize portfolio state")
            return None
        
    def process_date(self, date, data):
        try:
            # Ensure model is not None before attempting prediction
            if not hasattr(self, 'model') or self.model is None:
                logger.error(f"Model is not initialized for date: {date}")
                return None

            # Proceed with prediction only if model is available
            prediction = self.process_date(date, data)
            return prediction
        except AttributeError as e:
            logger.error(f"Attribute error processing {date}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing {date}: {str(e)}")
            return None

    def detect_regime(self, series):
        try:
            # Ensure the series is properly checked
            if series is None or series.empty:
                logger.warning("Series is empty or None. Cannot detect regime.")
                return "neutral"

            # Use explicit numerical checks to determine the regime
            mean_value = series.mean()
            if mean_value > 0.05:  # Just an example threshold
                return "bullish"
            elif mean_value < -0.05:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return "neutral"

    
    def _log_backtest_summary(self, results):
        """Log summary of backtest results"""
        try:
            logger.info("\nBacktest Summary:")
            logger.info(f"Total Trading Days: {len(results['portfolio_value'])}")
            logger.info(f"Total Transactions: {len(results['transactions'])}")
            
            initial_value = results['portfolio_value'].iloc[0]
            final_value = results['portfolio_value'].iloc[-1]
            total_return = (final_value / initial_value - 1) * 100
            
            logger.info(f"Initial Portfolio Value: ${initial_value:,.2f}")
            logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
            logger.info(f"Total Return: {total_return:.2f}%")
            
            # Add metrics summary if available
            if results.get('metrics'):
                logger.info("\nPerformance Metrics:")
                for metric_name, value in results['metrics'].items():
                    logger.info(f"{metric_name}: {value:.2f}")
                    
        except Exception as e:
            logger.error(f"Error logging backtest summary: {str(e)}")

    def _verify_features(self, price_data, features, dates):
        """Verify feature calculation for debugging"""
        try:
            logger.info("\nVerifying feature calculation:")
            logger.info(f"Required features: {features}")
            
            
            symbol = list(price_data.keys())[0]  # Test with first symbol
            
            df = price_data[symbol].copy()
            df_features = self.data_pipeline._calculate_enhanced_features(df)
            
            logger.info(f"\nAvailable features for {symbol}:")
            logger.info(f"Columns: {df_features.columns.tolist()}")
            
            missing_features = [f for f in features if f not in df_features.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
            else:
                logger.info("All required features available")
                
            # Check for NaN values
            nan_features = df_features.columns[df_features.isna().any()].tolist()
            if nan_features:
                logger.warning(f"Features with NaN values: {nan_features}")
                
            return len(missing_features) == 0
            
        except Exception as e:
            logger.error(f"Error verifying features: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def _select_stocks(self, model, date, price_data, features, sequence_length, sector_map=None):
        """Select stocks using model predictions"""
        try:
            from predictor import EnhancedPredictor
            return EnhancedPredictor.select_stocks_with_combined_model(
                model=model,
                date=date,
                price_data=price_data,
                features=features,
                sequence_length=sequence_length,
                sector_map=sector_map
            )
        except Exception as e:
            logger.error(f"Error selecting stocks: {str(e)}")
            return []

    def _calculate_performance_metrics(self, portfolio_value: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict:
        """Calculate performance metrics with proper error handling"""
        try:
            metrics = {}
            
            if len(portfolio_value) < 2:
                logger.warning("Insufficient data for performance calculation")
                return self._create_empty_metrics()
            
            # Calculate returns
            returns = portfolio_value.pct_change(fill_method=None).dropna()
            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
            trading_days = len(portfolio_value)
            annual_return = ((1 + total_return/100) ** (252/trading_days) - 1) * 100
            
            # Safe volatility calculation
            returns_std = returns.std()
            if returns_std > 0:
                volatility = returns_std * np.sqrt(252) * 100
                sharpe = (returns.mean() * 252 - RISK_FREE_RATE) / (returns_std * np.sqrt(252))
            else:
                volatility = 0
                sharpe = 0
            
            # Calculate benchmark comparison if available
            if benchmark is not None and len(benchmark) > 1:
                # Align benchmark to portfolio dates
                benchmark = benchmark.reindex(portfolio_value.index, method='ffill')
                bench_returns = benchmark.pct_change(fill_method=None).dropna()
                
                # Only proceed if we have aligned data
                if len(returns) == len(bench_returns):
                    # Calculate beta and alpha
                    bench_std = bench_returns.std()
                    if bench_std > 0:
                        covariance = np.cov(returns, bench_returns)[0][1]
                        beta = covariance / (bench_std * bench_std)
                    else:
                        beta = 1
                    
                    bench_total_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
                    alpha = annual_return - (RISK_FREE_RATE + beta * (bench_total_return - RISK_FREE_RATE))
                    
                    # Add benchmark metrics
                    metrics.update({
                        'Alpha': alpha,
                        'Beta': beta,
                        'Benchmark Return (%)': bench_total_return,
                        'Active Return (%)': total_return - bench_total_return
                    })
            
            # Add base metrics with safe win rate calculation
            metrics.update({
                'Total Return (%)': total_return,
                'Annual Return (%)': annual_return,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': self._calculate_drawdown_series(portfolio_value).min() * 100,
                'Win Rate (%)': (returns > 0).mean() * 100 if len(returns) > 0 else 0,
                'Number of Trades': trading_days
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._create_empty_metrics()
            
    def _calculate_total_sector_exposures(self, 
                                    allocations: Dict[str, float],
                                    price_data: Dict[str, pd.DataFrame],
                                    date: datetime,
                                    sector_map: Dict[str, str],
                                    etf_holdings: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate total exposure by sector including ETF holdings"""
        try:
            sector_exposures = defaultdict(float)
            
            # Direct exposures from stock allocations
            for symbol, shares in allocations.items():
                if symbol in sector_map:
                    if symbol in price_data:
                        df = price_data[symbol]
                        mask = df['date'] <= date
                        if mask.any():
                            price = df.loc[mask, 'close'].iloc[-1]
                            value = shares * price
                            sector = sector_map[symbol]
                            sector_exposures[sector] += value

            # Normalize by total value
            total_value = sum(sector_exposures.values())
            if total_value > 0:
                for sector in sector_exposures:
                    sector_exposures[sector] /= total_value

            return dict(sector_exposures)
                
        except Exception as e:
            logger.error(f"Error calculating sector exposures: {str(e)}")
            return {}
    
    def _check_risk_limits(self,
                          date: datetime,
                          state: Dict,
                          trackers: Dict) -> bool:
        """Check ETF-specific risk limits"""
        try:
            # Calculate basic risk metrics
            current_value = state['total_value']
            initial_value = state['initial_capital']
            drawdown = (self.peak_value - current_value) / self.peak_value
            
            # Check ETF-specific limits
            holdings = state['holdings']['equities']
            if holdings:
                # Check concentration
                max_position = max(holdings.values()) / current_value
                if max_position > self.etf_constraints['max_position']:
                    logger.warning(f"Position size limit exceeded: {max_position:.1%}")
                    return False
                    
                # Check minimum positions
                if len(holdings) < self.etf_constraints['min_positions']:
                    logger.warning(f"Insufficient positions: {len(holdings)}")
                    return False
                    
                # Calculate volatility if enough history
                if len(state['portfolio_value']) >= 20:
                    returns = state['portfolio_value'].pct_change()
                    volatility = returns.tail(20).std() * np.sqrt(252)
                    if volatility > MAX_VOLATILITY:
                        logger.warning(f"Volatility limit exceeded: {volatility:.1%}")
                        return False
                        
            # Regular drawdown check
            if drawdown > MAX_DRAWDOWN:
                logger.warning(f"Maximum drawdown exceeded: {drawdown:.1%}")
                self.handle_risk_breach(date, state, trackers)
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False

    def _calculate_volatility(self, portfolio_value: pd.Series) -> float:
        """Calculate rolling volatility"""
        if len(portfolio_value) >= 20:
            returns = portfolio_value.pct_change()
            return returns.tail(20).std() * np.sqrt(252)
        return 0.0

    def run_backtest(self, 
                    equity_data: dict, 
                    options_data: dict,
                    model,
                    features: list,
                    rebalance_frequency: str,
                    transaction_cost: float,
                    allocation_ratios: dict,
                    start_date: str,
                    end_date: str,
                    sequence_length: int,
                    sector_map: Optional[Dict] = None,
                    sector_constraints: Optional[Dict] = None,
                    option_pricing_model: Optional[object] = None,  
                    treasury_data: Optional[Dict] = None 
                    ) -> dict:
        """
        Run ETF backtest with benchmark comparison
        """
        try:
            logger.info("\nStarting ETF backtest...")
            
            # 1. Validate dates
            dates = self.parse_and_validate_dates(start_date, end_date)
            if not dates:
                logger.error("Date validation failed")
                return self._create_empty_results(start_date)
            
            # 2. Prepare and validate data
            valid_data = self._prepare_data_with_etfs(
                price_data=equity_data,
                options_data=options_data,
                dates=dates
            )
            
            if not valid_data:
                logger.error("Data preparation failed")
                return self._create_empty_results(start_date)
            
            # 3. Initialize portfolio state
            state = self.initialize_portfolio_state(
                price_data=equity_data,
                options_data=options_data,
                treasury_data=valid_data['treasury_data'],
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                option_pricing_model=option_pricing_model,
                sector_map=sector_map  
            )
            
            if state is None:
                logger.error("Failed to initialize portfolio state")
                return self._create_empty_results(start_date)
                
            # 4. Initialize tracking systems - using class-level instances
            trackers = {
                'trade': TradeTracker(),
                'rebalance': self.rebalance_tracker,
                'event': self.performance_monitor,
                'risk': self.risk_manager,
                'transactions': []
            }
            
            # 5. Create SPY benchmark
            spy_benchmark = self._create_benchmark_series(
                start_date=dates['equity_start'],
                end_date=dates['equity_end']
            )
            logger.info(f"Valid data keys: {valid_data.keys() if valid_data else None}")

            spy_data = self._get_spy_data(start_date, end_date)
            if spy_data is None:
                logger.error("Failed to get SPY data needed for ETF model training")
                return self._create_empty_results(start_date)
            
            logger.info(f"SPY data shape: {spy_data.shape}")
            logger.info(f"SPY date range: {spy_data['date'].min()} to {spy_data['date'].max()}")

            
            etf_model = train_etf_model(
                df=valid_data['price_data'],
                spy_data=spy_data,
                features=features
                
                
            )

            if etf_model is None:
                logger.error("Failed to train ETF model")
                return self._create_empty_results(start_date)


            # 6. Main backtest loop
            rebalance_attempts = 0
            successful_trades = 0
            last_rebalance = pd.to_datetime(start_date) - pd.Timedelta(days=1)
            
            for date in valid_data['date_range']:
                try:
                    # Update portfolio state
                    previous_value = state.total_value
                    state.update(date, valid_data['combined_prices'])
                    
                    # Risk checks using class-level risk manager
                    if not self.risk_manager.check_stops(
                        current_value=state.total_value,
                        date=date,
                        holdings=state.holdings
                    ):
                        self.risk_manager.handle_risk_breach(date, trackers)
                        continue
                    
                    # Check for rebalancing
                    needs_rebalance = self._need_rebalance(
                        date, last_rebalance, rebalance_frequency
                    )
                        
                    if needs_rebalance:

                        rebalance_attempts += 1
                        logger.debug(f"\nAttempting rebalance #{rebalance_attempts}")
                        
                        # Get ML data and predictions
                        df_ml_data = self.data_pipeline.prepare_features(
                            state.price_data,
                            date=date
                        )
                        
                        if df_ml_data is not None:

                             # Get ETF relative performance prediction
                            etf_prediction = etf_model.predict(df_ml_data)
                            
                            # Adjust allocation ratios based on prediction
                            allocation_ratios = self._adjust_allocations(
                                base_ratios=allocation_ratios,
                                etf_prediction=etf_prediction,
                                prediction_threshold=self.config.PREDICTION_THRESHOLD 
                            )

                              # Integrate re-entry check before executing trades
                            price_series = state.price_data.get('close')  # Assuming price_series is available here
                            if price_series is not None:
                                recovery_from_bottom = state.get_recovery_from_bottom()
                                market_momentum = state.get_market_momentum()
                                days_out = state.get_days_out_of_market()
                                base_threshold = self.config.RE_ENTRY_THRESHOLD
                                
                                should_reenter = self.check_reentry(price_series, recovery_from_bottom, base_threshold, market_momentum, days_out)
                                
                                if should_reenter:
                                # Select stocks using model
                                    if model is None:
                                        logger.error("Main model is None")
                                        return self._create_empty_results(start_date)


                                    selected_stocks = EnhancedPredictor.select_stocks_with_combined_model(
                                        model=model,
                                        date=date,
                                        price_data=state.price_data,
                                        features=features,
                                        sequence_length=sequence_length,
                                        sector_map=sector_map
                                    )
                                    
                                    if selected_stocks:
                                        # Calculate optimal weights with sector constraints
                                        weights = self.portfolio_optimizer.optimize_portfolio(
                                            selected_stocks=selected_stocks,
                                            price_data=state.price_data,
                                            date=date,
                                            sector_map=sector_map,
                                            sector_constraints=sector_constraints
                                        )
                                        
                                        if weights:
                                            # Execute trades with sector tracking
                                            executed = state.execute_trades(
                                                date=date,
                                                selected_symbols=selected_stocks,
                                                weights=weights,
                                                allocation_ratios=allocation_ratios,
                                                transaction_cost=transaction_cost,
                                                combined_prices=valid_data['combined_prices'],
                                                trackers=trackers
                                            )
                                            
                                            if executed:
                                                successful_trades += 1
                                                last_rebalance = date
                                                
                                                # Record rebalancing attempt with sector impact
                                                self.rebalance_tracker.add_attempt(
                                                    date=date,
                                                    success=True,
                                                    available_symbols=len(selected_stocks),
                                                    total_symbols=len(equity_data),
                                                    sector_impact=state.sector_weights
                                                )
                                                
                                                logger.debug(
                                                    f"Rebalance successful. Value change: "
                                                    f"${state.total_value - previous_value:,.2f}"
                                                )
                    
                    # Handle options if enabled
                    if state.options_enabled and state.holdings['equities']:
                        self._handle_options_trading(
                            state=state,
                            date=date,
                            options_data=valid_data['options_data'],
                            treasury_data=valid_data['treasury_data'],
                            option_pricing_model=option_pricing_model,
                            allocation_ratios=allocation_ratios,
                            transaction_cost=transaction_cost,
                            trackers=trackers
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing {date}: {str(e)}")
                    continue
            
            # 7. Calculate final results
            results = self._prepare_results(
                state=state,
                trackers=trackers,
                spy_benchmark=spy_benchmark,
                rebalance_attempts=rebalance_attempts,
                successful_trades=successful_trades,
                sector_map=sector_map
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            logger.error(traceback.format_exc())
            return self._create_empty_results(start_date)
        
    def _adjust_allocations(self, base_ratios: Dict, etf_prediction: float, prediction_threshold: float) -> Dict:
        """Adjust allocation ratios based on ETF prediction"""
        adjustment = 1.0
        if etf_prediction > prediction_threshold:
            adjustment = 1.2  # Increase allocation
        elif etf_prediction < -prediction_threshold:
            adjustment = 0.8  # Decrease allocation
            
        return {k: v * adjustment for k, v in base_ratios.items()}
                
    def _update_sector_exposures(self, 
                               weights: Dict[str, float],
                               sector_map: Dict[str, str],
                               date: datetime):
        """Track sector exposures over time"""
        try:
            sector_weights = defaultdict(float)
            for symbol, weight in weights.items():
                if symbol in sector_map:
                    sector = sector_map[symbol]
                    sector_weights[sector] += weight
            
            for sector, weight in sector_weights.items():
                self.sector_exposures[sector].append({
                    'date': date,
                    'weight': weight
                })
                
        except Exception as e:
            logger.error(f"Error updating sector exposures: {str(e)}")

    def _prepare_results(self,
                        state: Dict,
                        trackers: Dict,
                        spy_benchmark: pd.Series,
                        rebalance_attempts: int,
                        successful_trades: int,
                        sector_map: Optional[Dict] = None) -> Dict:
        """
        Prepare comprehensive backtest results with ETF-specific metrics
        """
        try:
            # Basic validation
            if not isinstance(state._state['portfolio_value'], pd.Series):
                logger.error("Invalid portfolio value series")
                return self._create_empty_results(state._state['start_date'])

            portfolio_value = state._state['portfolio_value'].copy()
            # Ensure timezone-naive dates
            portfolio_value.index = portfolio_value.index.tz_localize(None)

             # Initialize trackers if empty
            if 'transactions' not in trackers:
                trackers['transactions'] = []

            
            # Initialize results
            results = {
                'portfolio_value': portfolio_value,
                'transactions': pd.DataFrame(trackers['transactions']),
                'metrics': {},
                'sector_analysis': {},
                'risk_events': trackers['risk'].risk_events if 'risk' in trackers else []
            }
            
            # Add benchmark comparison
            if spy_benchmark is not None:
                spy_index = spy_benchmark.index.tz_localize(None)
                aligned_benchmark = spy_benchmark.copy()
                aligned_benchmark.index = spy_index
                
                # Align and scale benchmark
                aligned_benchmark = aligned_benchmark.reindex(
                    portfolio_value.index, 
                    method='ffill'
                )
                scale_factor = portfolio_value.iloc[0] / aligned_benchmark.iloc[0]
                results['spy_benchmark'] = aligned_benchmark * scale_factor
                
                # Calculate tracking error
                tracking_diff = portfolio_value.pct_change() - aligned_benchmark.pct_change()
                results['metrics']['tracking_error'] = tracking_diff.std() * np.sqrt(252)
                
                # Calculate information ratio
                excess_returns = tracking_diff.mean() * 252
                info_ratio = excess_returns / (tracking_diff.std() * np.sqrt(252))
                results['metrics']['information_ratio'] = info_ratio
            
            # Add sector analysis if sector map available
            if sector_map:
                sector_analysis = self._analyze_sector_performance(
                    sector_exposures=self.sector_exposures,
                    portfolio_value=portfolio_value
                )
                results['sector_analysis'] = sector_analysis
            
            # Calculate ETF-specific metrics
            results['metrics'].update(self._calculate_etf_metrics(
                portfolio_value=portfolio_value,
                transactions=results['transactions'],
                spy_benchmark=results.get('spy_benchmark')
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error preparing results: {str(e)}")
            return self._create_empty_results(state['start_date'])
        
    def parse_and_validate_dates(self, start_date: str, end_date: str) -> Dict[str, pd.Timestamp]:
        """Parse and validate all date configurations"""

        
        try:
            dates = {
                'equity_start': pd.to_datetime(start_date),
                'backtest_start': pd.to_datetime(BACKTEST_START_DATE),
                'equity_end': pd.to_datetime(end_date)
            }
            
            logger.info("\nValidating dates:")
            for name, date in dates.items():
                logger.info(f"{name}: {date}")
                
            # Validate date order
            if dates['backtest_start'] > dates['equity_start']:
                logger.error("Backtest start date must be before equity start date")
                return None
                
            if dates['equity_start'] > dates['equity_end']:
                logger.error("Equity start date must be before end date")
                return None
                
            # Check required history period
            history_requirement = dates['equity_start'] - pd.DateOffset(years=MINIMUM_HISTORY_YEARS)
            if history_requirement < dates['backtest_start']:
                logger.error(f"Insufficient history period. Need data from {history_requirement}")
                return None
                
            logger.info("Date validation successful")
            return dates
            
        except Exception as e:
            logger.error(f"Error parsing dates: {str(e)}")
            return None

    def _prepare_data_with_etfs(self,
                           price_data: Dict[str, pd.DataFrame],
                           options_data: Dict[str, pd.DataFrame],
                           dates: Dict[str, pd.Timestamp],
                           etf_holdings: Optional[Dict] = None) -> Optional[Dict]:
        """
        Prepare data with enhanced ETF handling
        
        Args:
            price_data: Price data dictionary
            options_data: Options data dictionary
            dates: Dictionary of important dates
            etf_holdings: ETF holdings data
            
        Returns:
            Dictionary of prepared data
        """
        try:
            logger.info("\nPreparing and validating data with ETF analysis...")
            
            time.sleep(2) 

              # Create fallback treasury data
            treasury_data = pd.DataFrame([TREASURY_FALLBACK_RATES])

            # 1. First fetch SPY data separately if not in price_data
            spy_data = None
            try:
                # Use AlphaVantage API
                url = (
                    f'https://www.alphavantage.co/query?'
                    f'function=TIME_SERIES_DAILY_ADJUSTED'
                    f'&symbol=SPY'
                    f'&outputsize=full'
                    f'&apikey={ALPHAVANTAGE_API_KEY}'
                )
                
                logger.info("Fetching SPY data from AlphaVantage")
                response = requests.get(url)
                
                if response.status_code != 200:
                    logger.error(f"API request failed with status {response.status_code}")
                    return None
                    
                data = response.json()
                
                if 'Time Series (Daily)' not in data:
                    logger.error(f"Unexpected API response format: {data.keys()}")
                    return None
                    
                # Convert to DataFrame
                spy_data = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                spy_data.index = pd.to_datetime(spy_data.index)
                spy_data = spy_data.sort_index()
                
                # Rename columns
                column_map = {  
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low', 
                    '4. close': 'Close',
                    '5. adjusted close': 'Adj Close',  # Changed to match yfinance
                    '6. volume': 'Volume'
                }
                spy_data = spy_data.rename(columns=column_map)
                
                # Convert string values to float
                for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                    spy_data[col] = pd.to_numeric(spy_data[col], errors='coerce')
                
                # Add date column
                spy_data['date'] = spy_data.index
                
                # Filter to required date range
                mask = (spy_data['date'] >= dates['backtest_start']) & (spy_data['date'] <= dates['equity_end'])
                spy_data = spy_data.loc[mask]
                
                # Debug info
                logger.info(f"SPY data shape: {spy_data.shape}")
                logger.info(f"SPY date range: {spy_data['date'].min()} to {spy_data['date'].max()}")
                logger.info(f"SPY columns: {spy_data.columns.tolist()}")
                logger.info(f"Sample data:\n{spy_data.head()}")
                
                # Add to price data
                price_data['SPY'] = spy_data
                logger.info("Successfully added SPY data")

            except Exception as e:
                logger.error(f"Error fetching SPY data: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")

            # 2. Separate ETFs and stocks
            etf_symbols = [s for s in price_data.keys() if self._is_etf(s)]
            stock_symbols = [s for s in price_data.keys() if not self._is_etf(s)]
            
            logger.info(f"Found {len(etf_symbols)} ETFs and {len(stock_symbols)} stocks")
            
            # 3. Validate ETF data
            validation_results = self._validate_etf_data(
                etf_data={symbol: price_data[symbol] for symbol in etf_symbols},
                etf_holdings=etf_holdings,
                start_date=dates['equity_start']
            )
            
            valid_etfs = validation_results['valid_etfs']
            logger.info(f"Validated {len(valid_etfs)} ETFs")
            
            # 4. Calculate ETF metrics
            etf_metrics = self._calculate_etf_metrics(
                portfolio_value=spy_data['Adj Close'] if spy_data is not None else None,
                transactions=pd.DataFrame(),  # Empty DataFrame if no transactions yet
                spy_benchmark=None
            )
            
            # 5. Filter price data to valid symbols
            valid_symbols = stock_symbols + valid_etfs
            price_data = {symbol: price_data[symbol] for symbol in valid_symbols}
            
            # 6. Prepare market data
            combined_prices, date_range = self._prepare_price_data(
                price_data=price_data,
                start_date=dates['equity_start'],
                end_date=dates['equity_end']
            )
            
            if len(date_range) == 0:
                logger.error("No valid date range")
                return None
            
            # 7. Process options data with ETF consideration
            processed_options = None
            options_enabled = False
            
            if options_data:
                try:
                    processed_options = self.data_pipeline.preprocess_options_data(
                        options_data=options_data,
                        equity_data=price_data
                    )
                    
                    if not processed_options.empty:
                        logger.info(f"Successfully processed {len(processed_options)} options")
                        options_enabled = True
                    else:
                        logger.warning("No valid options data after preprocessing")
                except Exception as e:
                    logger.error(f"Error processing options data: {str(e)}")
            
            # 8. Add ETF-specific metrics to validation results
            validation_results.update({
                'etf_metrics': etf_metrics,
                'price_data': price_data,
                'options_data': processed_options,
                'combined_prices': combined_prices,
                'date_range': date_range,
                'options_enabled': options_enabled,
                'treasury_data': treasury_data
            })
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None

    def _get_spy_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get SPY data from already fetched AlphaVantage data"""
        try:
            logger.info("Fetching SPY data...")
            
            # First try to get from data cache
            if 'SPY' in self.data_pipeline.data_cache:
                spy_data = self.data_pipeline.data_cache['SPY']
                mask = (spy_data['date'] >= pd.to_datetime(start_date)) & (spy_data['date'] <= pd.to_datetime(end_date))
                filtered_data = spy_data[mask]
                if not filtered_data.empty:
                    logger.info("Successfully retrieved SPY data from cache")
                    return filtered_data

            # If not in cache or filtered data is empty, fetch directly
            spy_data = yf.download('SPY', 
                                start=pd.to_datetime(start_date),
                                end=pd.to_datetime(end_date),
                                progress=False)
            
            if spy_data.empty:
                logger.error("Failed to fetch SPY data")
                return None
                
            spy_data = spy_data.reset_index()
            spy_data.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            logger.info(f"Successfully fetched SPY data with {len(spy_data)} rows")
            
            return spy_data

        except Exception as e:
            logger.error(f"Error getting cached SPY data: {str(e)}")
            return None
        
    def _prepare_price_data(self, price_data: Dict[str, pd.DataFrame], 
                      start_date: Union[str, datetime, pd.Timestamp], 
                      end_date: Union[str, datetime, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        try:
            combined_prices = pd.DataFrame()
            for symbol, df in price_data.items():
                # Standardize column names
                df, col_map = standardize_dataframe_columns(df)
                if df is None:
                    continue

                # Get standardized column names
                date_col = col_map['date']
                close_col = col_map['close']

                # Reset index if date is both in columns and index
                if date_col in df.columns and df.index.name == date_col:
                    df = df.reset_index(drop=True)

                # Convert dates to datetime
                df[date_col] = pd.to_datetime(df[date_col])

                # Select and rename columns using mapped names
                df = df[[date_col, close_col]].copy()
                df = df.rename(columns={close_col: symbol})
                df = df.set_index(date_col)

                if combined_prices.empty:
                    combined_prices = df
                else:
                    combined_prices = combined_prices.join(df, how='outer')

            combined_prices.sort_index(inplace=True)
            date_range = combined_prices.index[
                (combined_prices.index >= start_date) &
                (combined_prices.index <= end_date)
            ]
            return combined_prices, date_range

        except Exception as e:
            logger.error(f"Error preparing price data: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame(), pd.DatetimeIndex([])
        
    def _is_etf(self, symbol: str) -> bool:
        """Check if symbol is an ETF"""
        return symbol.lower().endswith('etf') or symbol in self._common_etfs()

    def _common_etfs(self) -> List[str]:
        """List of common ETFs that might not end in 'ETF'"""
        return [
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO',
            'EFA', 'AGG', 'BND', 'LQD', 'TLT', 'GLD', 'SLV', 'XLF',
            'XLE', 'XLV', 'XLK', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU'
        ]

    def _validate_etf_data(self, etf_data: Dict[str, pd.DataFrame], 
                          etf_holdings: Optional[Dict] = None,
                          start_date: datetime = None) -> Dict:
        try:
            validation_results = {
                'valid_etfs': [],
                'invalid_etfs': [],
                'validation_metrics': {}
            }

            for symbol, df in etf_data.items():
                try:
                    logger.info(f"\nValidating {symbol}:")
                    df = df.copy()
                    
                    # Basic data validation
                    if df is None or df.empty:
                        logger.warning(f"{symbol}: Empty data")
                        validation_results['invalid_etfs'].append(symbol)
                        continue

                    df, col_map = standardize_dataframe_columns(df)
                    if df is None:
                        validation_results['invalid_etfs'].append(symbol)
                        continue
                    
                    # Validate using standardized columns
                    if self._validate_etf_data_quality(df, col_map, start_date):
                        validation_results['valid_etfs'].append(symbol)
                    else:
                        validation_results['invalid_etfs'].append(symbol)

                    # Handle date column consistently
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'])
                    df['date'] = df['date'].dt.tz_localize(None)

                    # Start date check
                    if start_date and pd.to_datetime(df['date'].min()) > pd.to_datetime(start_date):
                        logger.warning(f"{symbol}: Data starts after required start date")
                        validation_results['invalid_etfs'].append(symbol)
                        continue

                    # Check for missing values in key columns
                    key_cols = ['Close', 'Volume']  # Changed from lowercase
                    if df[key_cols].isna().any().any():
                        logger.warning(f"{symbol}: Contains missing values")
                        validation_results['invalid_etfs'].append(symbol)
                        continue

                    # Add to valid ETFs
                    validation_results['valid_etfs'].append(symbol)
                    validation_results['validation_metrics'][symbol] = {
                        'trading_days': len(df),
                        'date_range': f"{df['date'].min()} to {df['date'].max()}",
                        'avg_volume': df['Volume'].mean()  # Changed from lowercase
                    }

                except Exception as e:
                    logger.error(f"Error validating {symbol}: {str(e)}")
                    validation_results['invalid_etfs'].append(symbol)
                    continue

            return validation_results

        except Exception as e:
            logger.error(f"Error in ETF validation: {str(e)}")
            return {'valid_etfs': [], 'invalid_etfs': [], 'validation_metrics': {}}
        
    def _validate_etf_data_quality(self, df: pd.DataFrame, col_map: Dict[str, str], start_date: datetime) -> bool:
        """Validate ETF data quality and coverage"""
        try:
            if df is None or not col_map:
                return False
                
            # Check for required columns
            if not all(col in col_map for col in ['date', 'close', 'volume']):
                logger.warning("Missing required columns")
                return False
                
            date_col = col_map['date']
            
            # Convert dates if needed
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Check date coverage
            if df[date_col].min() > start_date:
                logger.warning(f"Data starts after required start date")
                return False
                
            # Check for sufficient data
            if len(df) < 20:  # Minimum required history
                logger.warning(f"Insufficient history: {len(df)} days")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating ETF data quality: {str(e)}")
            return False
            
    def _calculate_etf_metrics(self,
                            portfolio_value: Optional[pd.Series],
                            transactions: pd.DataFrame,
                            spy_benchmark: Optional[pd.Series] = None) -> Dict:
        """Calculate ETF-specific performance metrics"""
        try:
            if portfolio_value is None or portfolio_value.empty:
                return {}
                
            # Convert index to datetime if needed
            portfolio_value.index = pd.to_datetime(portfolio_value.index)
            if spy_benchmark is not None:
                spy_benchmark.index = pd.to_datetime(spy_benchmark.index)
                
                # Reindex spy_benchmark to match portfolio_value
                spy_benchmark = spy_benchmark.reindex(portfolio_value.index)
                
            metrics = {}
            returns = portfolio_value.pct_change(fill_method=None).dropna()
            if len(returns) < 2:
                logger.warning("Insufficient return data")
                return {}

            # 4. Calculate basic metrics with safety checks
            
            metrics['total_return'] = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1)
            
            std_dev = returns.std()
            if std_dev > 0:
                metrics['volatility'] = std_dev * np.sqrt(252)
                metrics['sharpe_ratio'] = (returns.mean() * 252 - RISK_FREE_RATE) / (std_dev * np.sqrt(252))
            else:
                metrics['volatility'] = 0
                metrics['sharpe_ratio'] = 0
            
            # 5. Calculate transaction-based metrics
            if not transactions.empty and 'value' in transactions.columns:
                daily_volume = transactions.groupby('date')['value'].sum()
                metrics['avg_daily_volume'] = daily_volume.mean() if not daily_volume.empty else 0
                
                total_cost = transactions['cost'].sum() if 'cost' in transactions.columns else 0
                metrics['rebalancing_cost'] = total_cost / portfolio_value.iloc[0]
            
            # 6. Calculate benchmark relative metrics
            if spy_benchmark is not None:
                try:
                    spy_returns = spy_benchmark.pct_change(fill_method=None).dropna()
                    
                    # Calculate beta
                    cov_matrix = np.cov(returns, spy_returns)
                    spy_var = np.var(spy_returns)
                    
                    if spy_var > 0 and len(cov_matrix) > 1:
                        beta = cov_matrix[0,1] / spy_var
                        metrics['beta'] = beta
                        
                        # Calculate alpha
                        etf_return = returns.mean() * 252
                        spy_return = spy_returns.mean() * 252
                        metrics['alpha'] = etf_return - (RISK_FREE_RATE + beta * (spy_return - RISK_FREE_RATE))
                    else:
                        logger.warning("Unable to calculate beta due to insufficient variance")
                        metrics['beta'] = 1.0
                        metrics['alpha'] = 0.0
                        
                except Exception as e:
                    logger.error(f"Error calculating benchmark metrics: {str(e)}")
                    metrics['beta'] = 1.0
                    metrics['alpha'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating ETF metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return {}


    def _analyze_recovery_periods(self, drawdown: pd.Series) -> List[Dict]:
        """Analyze drawdown recovery periods"""
        try:
            if drawdown.empty:
                return []
                
            recovery_periods = []
            in_drawdown = False
            start_date = None
            peak_drawdown = 0
            
            for date, value in drawdown.items():
                if not in_drawdown and value < 0:
                    # Start of drawdown period
                    in_drawdown = True
                    start_date = date
                    peak_drawdown = value
                elif in_drawdown:
                    if value < peak_drawdown:
                        peak_drawdown = value  # New lowest point
                    elif value >= 0:
                        # Recovery
                        recovery_periods.append({
                            'start': start_date,
                            'end': date,
                            'duration': (date - start_date).days,
                            'drawdown': abs(peak_drawdown)
                        })
                        in_drawdown = False
            
            # Handle ongoing drawdown
            if in_drawdown:
                recovery_periods.append({
                    'start': start_date,
                    'end': drawdown.index[-1],
                    'duration': (drawdown.index[-1] - start_date).days,
                    'drawdown': abs(peak_drawdown)
                })
            
            return recovery_periods
            
        except Exception as e:
            logger.error(f"Error analyzing recovery periods: {str(e)}")
            return []

    def _calculate_sector_exposures(self, holdings: Dict[str, float], 
                              price_data: Dict[str, pd.DataFrame],
                              date: datetime,
                              sector_map: Dict[str, str]) -> Dict[str, float]:
        """Calculate current exposure to each sector"""
        try:
            total_value = 0
            sector_values = defaultdict(float)
            
            # Calculate total portfolio value and sector values
            for symbol, shares in holdings.items():
                if symbol in price_data and symbol in sector_map:
                    df = price_data[symbol]
                    mask = df['date'] <= date
                    if mask.any():
                        price = df.loc[mask, 'close'].iloc[-1]
                        position_value = shares * price
                        total_value += position_value
                        sector = sector_map[symbol]
                        sector_values[sector] += position_value
            
            # Calculate sector exposures as percentages
            if total_value > 0:
                sector_exposures = {
                    sector: value/total_value
                    for sector, value in sector_values.items()
                }
            else:
                sector_exposures = {sector: 0.0 for sector in sector_values.keys()}
                
            # Fill in missing sectors with zero exposure
            for sector in self.sector_limits.keys():
                if sector not in sector_exposures:
                    sector_exposures[sector] = 0.0
                        
            return sector_exposures
            
        except Exception as e:
            logger.error(f"Error calculating sector exposures: {str(e)}")
            return {sector: 0.0 for sector in self.sector_limits.keys()}
                
    def _validate_dates(self, start_date, end_date):
        """Parse and validate all date configurations"""
        try:
            backtest_start = pd.to_datetime(BACKTEST_START_DATE)
            equity_start = pd.to_datetime(start_date)
            equity_end = pd.to_datetime(end_date)
            required_history_start = equity_start - pd.DateOffset(years=MINIMUM_HISTORY_YEARS)
            
            if not (backtest_start <= equity_start <= equity_end):
                logger.error("Invalid date range")
                return None
                
            return {
                'backtest_start': backtest_start,
                'equity_start': equity_start, 
                'equity_end': equity_end,
                'required_history_start': required_history_start
            }
                
        except Exception as e:
            logger.error(f"Error validating dates: {str(e)}")
            return None
            
    
    def _prepare_data(self, price_data, options_data, dates, option_pricing_model=None):
        """Prepare and validate all data"""
        try:
            logger.info("\nPreparing and validating data...")
            
            # Process price data to ensure consistent columns
            processed_price_data = {}
            for symbol, df in price_data.items():
                df = df.copy()
                if 'close' not in df.columns:
                    logger.warning(f"Missing close column for {symbol}")
                    continue
                    
                # Create basic OHLC if not available
                if 'open' not in df.columns:
                    df['open'] = df['close']
                if 'high' not in df.columns:
                    df['high'] = df['close']
                if 'low' not in df.columns:
                    df['low'] = df['close']
                
                processed_price_data[symbol] = df
                
            if not processed_price_data:
                logger.error("No valid price data after processing")
                return None
                
            combined_prices, date_range = prepare_price_data(
                processed_price_data, 
                dates['equity_start'], 
                dates['equity_end']
            )
                
            if len(date_range) == 0:
                logger.error("No valid date range")
                return None

            return {
                'price_data': processed_price_data,
                'options_data': options_data,
                'treasury_data': None,  # Add if needed
                'combined_prices': combined_prices,
                'date_range': date_range,
                'valid_symbols': list(processed_price_data.keys()),
                'options_enabled': False  # Update based on options availability
            }
                    
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
    def _create_benchmark_series(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        try:
            if isinstance(start_date, tuple):
                start_date = start_date[0]
            if isinstance(end_date, tuple):
                end_date = end_date[0]
                
            start_date = pd.to_datetime(start_date).tz_localize(None)
            end_date = pd.to_datetime(end_date).tz_localize(None)
            
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
            if spy_data.empty:
                return None
                
            spy_series = spy_data['Adj Close']
            spy_series = spy_series * (self.initial_capital / spy_series.iloc[0])
            
            return spy_series
                
        except Exception as e:
            logger.error(f"Error creating benchmark series: {str(e)}")
            return None

    def _run_backtest_loop(self, state, data, model, dates, features, sequence_length,
                        rebalance_frequency, allocation_ratios, transaction_cost, trackers):
        """Execute main backtest loop with enhanced debugging and logging"""
        try:

            if model is None:
                logger.error("Model is None - cannot proceed with backtest")
                return

            # Basic model validation
            if not hasattr(model, 'predict'):
                logger.error("Invalid model - missing predict method")
                return
            # Initialize rebalance timing
            if rebalance_frequency == 'D':
                last_rebalance = dates['equity_start'] - pd.Timedelta(days=1)
            elif rebalance_frequency == 'W':
                last_rebalance = dates['equity_start'] - pd.Timedelta(weeks=1)
            elif rebalance_frequency == 'M':
                last_rebalance = dates['equity_start'] - pd.DateOffset(months=1)
            elif rebalance_frequency == 'Q':
                last_rebalance = dates['equity_start'] - pd.DateOffset(months=3)
            else:
                last_rebalance = dates['equity_start'] - pd.Timedelta(days=1)

            # Initialize pairs trading
            last_pairs_update = dates['equity_start']
            pairs_rebalance_days = 20  # Update pairs analysis every 20 trading days
            active_pairs = []

            logger.info("\nStarting backtest loop...")
            logger.info(f"Start date: {dates['equity_start']}")
            logger.info(f"End date: {dates['equity_end']}")
            logger.info(f"Rebalance frequency: {rebalance_frequency}")
            logger.info(f"Initial last_rebalance: {last_rebalance}")
            logger.info(f"Initial portfolio value: ${state['total_value']:,.2f}")
            
            metrics = {
                'last_value': state['total_value'],
                'max_value': state['total_value']
            }
            
            trades_count = 0
            for date in data['date_range']:
                try:
                    # Update portfolio state
                    previous_value = state['total_value']
                    state.update(date, data['combined_prices'])

            
                    
                    current_value = state['total_value']
                    daily_return = (current_value - metrics['last_value']) / metrics['last_value'] if metrics['last_value'] != 0 else 0
                    
                    # Pairs Trading Update
                    days_since_pairs = (date - last_pairs_update).days
                    if days_since_pairs >= pairs_rebalance_days:
                        logger.info(f"\nUpdating pairs analysis on {date}")
                        _, sector_map = fetch_sp500_constituents()
                        active_pairs = self.pairs_strategy.find_pairs(data['price_data'], sector_map)
                        last_pairs_update = date
                        logger.info(f"Found {len(active_pairs)} cointegrated pairs")

                    # Execute Pairs Trades
                    if active_pairs:
                        pairs_signals = self.pairs_strategy.generate_trading_signals(
                            active_pairs, data['price_data'], date
                        )
                        if pairs_signals:
                            logger.info(f"\nExecuting {len(pairs_signals)} pairs trades on {date}")
                            pairs_trades = self.pairs_strategy.execute_pairs_trades(
                                pairs_signals, state['holdings'], state['cash'],
                                data['combined_prices'], date, transaction_cost
                            )
                            for trade in pairs_trades:
                                symbol = trade['symbol']
                                if symbol not in state['holdings']['equities']:
                                    state['holdings']['equities'][symbol] = 0
                                state['holdings']['equities'][symbol] += trade['shares']
                                trade_value = trade['shares'] * trade['price']
                                trade_cost = abs(trade_value) * transaction_cost
                                state['cash'] -= (trade_value + trade_cost)
                                trackers['trade'].record_trade(
                                    date=date,
                                    action=trade['action'],
                                    symbol=symbol,
                                    shares=trade['shares'],
                                    price=trade['price'],
                                    value=abs(trade_value),
                                    cost=trade_cost,
                                    pair_trade=True,
                                    pair_symbol=trade.get('pair_symbol')
                                )
                    
                    # Track significant moves and drawdowns
                    if abs(daily_return) > 0.02:
                        logger.info(f"\nSignificant Move on {date}:")
                        logger.info(f"Daily Return: {daily_return:.2%}")
                        logger.info(f"Current Value: ${current_value:,.2f}")
                        
                    if current_value > metrics['max_value']:
                        metrics['max_value'] = current_value
                    elif current_value < metrics['max_value'] * 0.95:
                        drawdown = (metrics['max_value'] - current_value) / metrics['max_value']
                        logger.info(f"\nDrawdown Alert on {date}:")
                        logger.info(f"Drawdown: {drawdown:.2%}")
                    
                    # Regular rebalancing check
                    needs_rebalance = self._need_rebalance(date, last_rebalance, rebalance_frequency)
                    if needs_rebalance:
                        logger.info(f"\nRebalancing check for {date}:")
                        if model is None:
                            logger.error("Model is None during rebalancing")
                            continue
                        
                        if 'options_data' in data and data['options_data']:
                            sample_symbol = list(data['options_data'].keys())[0]
                            if isinstance(data['options_data'][sample_symbol], pd.DataFrame):
                                df = data['options_data'][sample_symbol]
                                logger.info("\nOptions Data Structure:")
                                logger.info(f"Columns available: {df.columns.tolist()}")
                                
                                date_columns = ['date', 'expirationDate', 'tradingDate']
                                for col in date_columns:
                                    if col in df.columns:
                                        dates_in_data = pd.to_datetime(df[col]).unique()
                                        logger.info(f"Found dates in column '{col}': {min(dates_in_data)} to {max(dates_in_data)}")
                                
                                valid_symbols = 0
                                for symbol, options_df in data['options_data'].items():
                                    if isinstance(options_df, pd.DataFrame):
                                        if 'expirationDate' in options_df.columns:
                                            if date <= pd.to_datetime(options_df['expirationDate']).max():
                                                valid_symbols += 1
                                
                                logger.info(f"Symbols with valid options for {date}: {valid_symbols}/{len(data['options_data'])}")
    
                        trades_executed = self._handle_rebalancing_cycle(
                            state=state,
                            date=date,
                            data=data,
                            model=model,
                            features=features,
                            sequence_length=sequence_length,
                            allocation_ratios=allocation_ratios,
                            transaction_cost=transaction_cost,
                            trackers=trackers
                        )
                        
                        if trades_executed:
                            trades_count += 1
                            last_rebalance = date
                            logger.info(f"\nRebalancing completed:")
                            logger.info(f"Portfolio value change: ${state['total_value'] - previous_value:,.2f}")
                            logger.info(f"Total trades executed: {trades_count}")
                    
                    metrics['last_value'] = current_value
                    
                except Exception as e:
                    logger.error(f"Error processing {date}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                    
            logger.info("\nBacktest Summary:")
            logger.info(f"Total trading days: {len(data['date_range'])}")
            logger.info(f"Total trades executed: {trades_count}")
            logger.info(f"Final portfolio value: ${state['total_value']:,.2f}")
            logger.info(f"Total return: {(state['total_value'] / state._state['initial_capital'] - 1) * 100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error in backtest loop: {str(e)}")
            logger.error(traceback.format_exc())

    def _handle_rebalancing_cycle(self, state: Dict, date: datetime, data: Dict,
                            model, features: List[str], sequence_length: int,
                            allocation_ratios: Dict, transaction_cost: float,
                            trackers: Dict, sector_map: Optional[Dict] = None,
                            etf_holdings: Optional[Dict] = None) -> bool:
        try:
            logger.info(f"\nProcessing rebalancing cycle for {date}")
            
            # Select stocks
            selected_stocks = self._select_stocks(
                model=model,
                date=date,
                price_data=data['price_data'],
                features=features,
                sequence_length=sequence_length,
                sector_map=sector_map
            )
            
            # Calculate optimal allocations
            allocations = self._calculate_optimal_allocations(
                selected_stocks=selected_stocks,
                state=state,
                date=date,
                sector_map=sector_map,
                price_data=data['price_data'],
                etf_holdings=etf_holdings,
                allocation_ratios=allocation_ratios  
            )
            
            if not allocations:
                logger.warning("Failed to calculate allocations")
                return False
                
            # Execute trades through portfolio state
            return state.execute_rebalancing_trades(
                date=date,
                allocations=allocations,
                allocation_ratios=allocation_ratios,  
                transaction_cost=transaction_cost,
                trackers=trackers
            )
                
        except Exception as e:
            logger.error(f"Error in rebalancing cycle: {str(e)}")
            return False
    
    def _select_etfs(self,
                    date: datetime,
                    etf_data: Dict[str, pd.DataFrame],
                    etf_holdings: Dict,
                    current_holdings: Dict,
                    sector_map: Optional[Dict] = None) -> List[Dict]:
        """
        Select ETFs based on multiple criteria
        """
        try:
            selected_etfs = []
            
            for symbol, df in etf_data.items():
                try:
                    # Skip if insufficient data
                    if df is None or df.empty:
                        continue
                    
                    df = df[df['date'] <= date].copy()
                    if len(df) < 60:  # Minimum history requirement
                        continue
                    
                    # Calculate ETF metrics
                    metrics = self._calculate_etf_selection_metrics(
                        df=df,
                        holdings=etf_holdings.get(symbol, {}).get('holdings', None),
                        sector_map=sector_map
                    )
                    
                    # Apply selection criteria
                    if self._etf_meets_criteria(metrics, current_holdings):
                        selected_etfs.append({
                            'symbol': symbol,
                            'metrics': metrics,
                            'score': self._calculate_etf_score(metrics)
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing ETF {symbol}: {str(e)}")
                    continue
            
            # Sort by score and limit selection
            selected_etfs.sort(key=lambda x: x['score'], reverse=True)
            max_etfs = min(5, len(selected_etfs))  # Maximum number of ETFs to hold
            
            return selected_etfs[:max_etfs]
            
        except Exception as e:
            logger.error(f"Error selecting ETFs: {str(e)}")
            return []

    def _calculate_etf_selection_metrics(self,
                                       df: pd.DataFrame,
                                       holdings: Optional[pd.DataFrame] = None,
                                       sector_map: Optional[Dict] = None) -> Dict:
        """
        Calculate comprehensive ETF selection metrics
        """
        try:
            metrics = {}
            
            # Price-based metrics
            returns = df['close'].pct_change()
            metrics['volatility'] = returns.std() * np.sqrt(252)
            metrics['sharpe'] = (returns.mean() * 252 - RISK_FREE_RATE) / metrics['volatility']
            metrics['max_drawdown'] = self._calculate_max_drawdown(df['close'])
            
            # Momentum metrics
            metrics['momentum_1m'] = df['close'].iloc[-1] / df['close'].iloc[-20] - 1
            metrics['momentum_3m'] = df['close'].iloc[-1] / df['close'].iloc[-60] - 1
            
            # Volume metrics
            metrics['avg_volume'] = df['volume'].tail(20).mean()
            metrics['volume_trend'] = (
                df['volume'].tail(20).mean() / 
                df['volume'].tail(40).head(20).mean()
            )
            
            # Holdings-based metrics
            if holdings is not None:
                holdings_metrics = self._calculate_holdings_metrics(
                    holdings=holdings,
                    sector_map=sector_map
                )
                metrics.update(holdings_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating ETF metrics: {str(e)}")
            return {}
        
    def _calculate_holdings_metrics(self,
                                  holdings: pd.DataFrame,
                                  sector_map: Optional[Dict] = None) -> Dict:
        """
        Calculate metrics based on ETF holdings
        """
        try:
            metrics = {}
            
            # Basic holdings metrics
            metrics['num_holdings'] = len(holdings)
            
            # Concentration metrics
            weights = holdings['weight'].values
            metrics['top_10_concentration'] = np.sum(np.sort(weights)[-10:])
            metrics['herfindahl_index'] = np.sum(weights * weights)
            
            # Sector metrics if mapping available
            if sector_map:
                sector_weights = defaultdict(float)
                for _, row in holdings.iterrows():
                    if row['symbol'] in sector_map:
                        sector = sector_map[row['symbol']]
                        sector_weights[sector] += row['weight']
                
                metrics['sector_concentration'] = max(sector_weights.values())
                metrics['num_sectors'] = len(sector_weights)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating holdings metrics: {str(e)}")
            return {}

    def _etf_meets_criteria(self,
                          metrics: Dict,
                          current_holdings: Dict) -> bool:
        """
        Check if ETF meets selection criteria
        """
        try:
            # Minimum liquidity
            if metrics['avg_volume'] < self.etf_constraints['min_etf_volume']:
                return False
            
            # Volatility constraints
            if metrics['volatility'] > 0.40:  # 40% annualized volatility cap
                return False
            
            # Performance filters
            if metrics['sharpe'] < 0.5:  # Minimum Sharpe ratio
                return False
            
            if metrics['max_drawdown'] < -0.30:  # Maximum drawdown constraint
                return False
            
            # Concentration limits
            if 'top_10_concentration' in metrics:
                if metrics['top_10_concentration'] > 0.60:  # Maximum concentration in top 10
                    return False
            
            # Holdings constraints
            if 'num_holdings' in metrics:
                if metrics['num_holdings'] < 20:  # Minimum holdings
                    return False
            
            # Sector constraints
            if 'sector_concentration' in metrics:
                if metrics['sector_concentration'] > 0.40:  # Maximum sector concentration
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking ETF criteria: {str(e)}")
            return False

    def _calculate_etf_score(self, metrics: Dict) -> float:
        """
        Calculate comprehensive ETF score
        """
        try:
            # Component scores
            performance_score = (
                metrics.get('sharpe', 0) * 0.4 +
                metrics.get('momentum_1m', 0) * 0.3 +
                metrics.get('momentum_3m', 0) * 0.3
            )
            
            risk_score = (
                (1 - abs(metrics.get('max_drawdown', -1))) * 0.4 +
                (1 - metrics.get('volatility', 1)) * 0.3 +
                metrics.get('volume_trend', 1) * 0.3
            )
            
            # Holdings score if available
            if 'num_holdings' in metrics:
                holdings_score = (
                    (1 - metrics.get('top_10_concentration', 1)) * 0.4 +
                    (metrics.get('num_sectors', 1) / 11) * 0.3 +  # Normalize by max sectors
                    (1 - metrics.get('herfindahl_index', 1)) * 0.3
                )
            else:
                holdings_score = 0.5  # Neutral score if no holdings data
            
            # Combined score with weights
            total_score = (
                performance_score * 0.4 +
                risk_score * 0.3 +
                holdings_score * 0.3
            )
            
            return float(total_score)
            
        except Exception as e:
            logger.error(f"Error calculating ETF score: {str(e)}")
            return 0.0
        
    def _calculate_optimal_allocations(self,
                                selected_stocks: List[str],
                                selected_etfs: List[Dict],
                                state: Dict,
                                date: datetime,
                                sector_map: Optional[Dict] = None,
                                current_exposures: Optional[Dict] = None,
                                etf_holdings: Optional[Dict] = None,
                                allocation_ratios: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate optimal portfolio allocations considering both stocks and ETFs
        while respecting both allocation ratios and constraints
        """
        try:
            logger.info("\nCalculating optimal allocations...")
            
            # Initialize allocation containers
            stock_allocations = {}
            etf_allocations = {}
            total_portfolio_value = state._state['total_value']
            
            # 1. Calculate budgets respecting both constraints and allocation ratios
            max_etf_by_constraint = total_portfolio_value * self.etf_constraints['max_etf_allocation']
            max_etf_by_ratio = total_portfolio_value * allocation_ratios.get('etfs', 0.30)
            
            # Take the more conservative limit
            etf_budget = min(max_etf_by_constraint, max_etf_by_ratio)
            stock_budget = total_portfolio_value - etf_budget
            
            logger.info(f"ETF budget: ${etf_budget:,.2f} (constraint: ${max_etf_by_constraint:,.2f}, ratio: ${max_etf_by_ratio:,.2f})")
            logger.info(f"Stock budget: ${stock_budget:,.2f}")
            
            # 2. Calculate ETF allocations
            if selected_etfs:
                etf_allocations = self._allocate_etf_budget(
                    selected_etfs=selected_etfs,
                    total_budget=etf_budget,
                    sector_map=sector_map,
                    current_exposures=current_exposures
                )
            
            # 3. Calculate stock allocations
            if selected_stocks:
                stock_allocations = self._allocate_stock_budget(
                    selected_stocks=selected_stocks,
                    total_budget=stock_budget,
                    sector_map=sector_map,
                    current_exposures=current_exposures,
                    etf_holdings=etf_holdings
                )
            
            # 4. Combine and normalize allocations
            combined_allocations = {**stock_allocations, **etf_allocations}
            
            # 5. Apply sector constraints
            if sector_map:
                combined_allocations = self._apply_sector_constraints(
                    allocations=combined_allocations,
                    sector_map=sector_map,
                    etf_holdings=etf_holdings
                )
            
            # 6. Final position size constraints
            combined_allocations = self._apply_position_constraints(combined_allocations)
            
            return combined_allocations
            
        except Exception as e:
            logger.error(f"Error calculating allocations: {str(e)}")
            return {}
        
    def _allocate_etf_budget(self,
                            selected_etfs: List[Dict],
                            total_budget: float,
                            sector_map: Optional[Dict] = None,
                            current_exposures: Optional[Dict] = None) -> Dict[str, float]:
        """
        Allocate budget among selected ETFs
        """
        try:
            allocations = {}
            total_score = sum(etf['score'] for etf in selected_etfs)
            
            if total_score == 0:
                return allocations
                
            # Initial allocations based on scores
            for etf in selected_etfs:
                symbol = etf['symbol']
                score = etf['score']
                allocation = (score / total_score) * total_budget
                
                # Apply ETF-specific constraints
                allocation = min(allocation, total_budget * 0.40)  # Max 40% per ETF
                allocation = max(allocation, total_budget * 0.10)  # Min 10% per ETF
                
                allocations[symbol] = allocation
            
            # Normalize to respect total budget
            total_allocated = sum(allocations.values())
            if total_allocated > 0:
                for symbol in allocations:
                    allocations[symbol] = (allocations[symbol] / total_allocated) * total_budget
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error allocating ETF budget: {str(e)}")
            return {}

    def _allocate_stock_budget(self,
                             selected_stocks: List[str],
                             total_budget: float,
                             sector_map: Optional[Dict] = None,
                             current_exposures: Optional[Dict] = None,
                             etf_holdings: Optional[Dict] = None) -> Dict[str, float]:
        """
        Allocate budget among selected stocks considering ETF holdings
        """
        try:
            allocations = {}
            num_stocks = len(selected_stocks)
            
            if num_stocks == 0:
                return allocations
                
            # Calculate stock overlap with ETF holdings
            stock_overlap = self._calculate_stock_etf_overlap(
                stocks=selected_stocks,
                etf_holdings=etf_holdings
            )
            
            # Adjust allocations based on overlap
            base_allocation = total_budget / num_stocks
            for symbol in selected_stocks:
                # Reduce allocation for stocks with high ETF overlap
                overlap_factor = 1.0 - (stock_overlap.get(symbol, 0) * 0.5)  # Max 50% reduction
                allocation = base_allocation * overlap_factor
                
                # Apply stock-specific constraints
                allocation = min(allocation, total_budget * 0.05)  # Max 5% per stock
                allocation = max(allocation, total_budget * 0.01)  # Min 1% per stock
                
                allocations[symbol] = allocation
            
            # Normalize to respect total budget
            total_allocated = sum(allocations.values())
            if total_allocated > 0:
                for symbol in allocations:
                    allocations[symbol] = (allocations[symbol] / total_allocated) * total_budget
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error allocating stock budget: {str(e)}")
            return {}

    def _calculate_stock_etf_overlap(self,
                                   stocks: List[str],
                                   etf_holdings: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate overlap between stocks and ETF holdings
        """
        try:
            overlap = defaultdict(float)
            
            if not etf_holdings:
                return dict(overlap)
                
            # Calculate overlap for each stock
            for symbol in stocks:
                total_weight = 0
                count = 0
                
                for etf_data in etf_holdings.values():
                    holdings = etf_data.get('holdings', pd.DataFrame())
                    if not holdings.empty and symbol in holdings['symbol'].values:
                        weight = holdings.loc[holdings['symbol'] == symbol, 'weight'].iloc[0]
                        total_weight += weight
                        count += 1
                
                if count > 0:
                    overlap[symbol] = total_weight / count
                    
            return dict(overlap)
            
        except Exception as e:
            logger.error(f"Error calculating stock-ETF overlap: {str(e)}")
            return {}
        
    def _apply_sector_constraints(self, allocations: Dict[str, float],
                                sector_map: Dict[str, str],
                                etf_holdings: Optional[Dict] = None) -> Dict[str, float]:
        """Apply sector constraints to allocations"""
        try:
            # Calculate current sector exposures
            sector_exposures = self._calculate_total_sector_exposures(
                allocations=allocations,
                sector_map=sector_map,
                etf_holdings=etf_holdings
            )
            
            # Adjust allocations to meet sector constraints
            adjusted_allocations = allocations.copy()
            iterations = 0
            max_iterations = 10
            
            while iterations < max_iterations:
                violations = False
                
                for sector, exposure in sector_exposures.items():
                    if exposure > self.sector_limits.get(sector, 0.25):
                        # Reduce allocations in over-exposed sector
                        reduction_needed = exposure - self.sector_limits[sector]
                        sector_symbols = [s for s in allocations if sector_map.get(s) == sector]
                        
                        if sector_symbols:
                            reduction_per_symbol = reduction_needed / len(sector_symbols)
                            for symbol in sector_symbols:
                                adjusted_allocations[symbol] = max(
                                    0,
                                    adjusted_allocations[symbol] - reduction_per_symbol
                                )
                            violations = True
                
                if not violations:
                    break
                    
                iterations += 1
                
                # Recalculate exposures
                sector_exposures = self._calculate_total_sector_exposures(
                    allocations=adjusted_allocations,
                    sector_map=sector_map,
                    etf_holdings=etf_holdings
                )
            
            # Normalize final allocations
            total_allocated = sum(adjusted_allocations.values())
            if total_allocated > 0:
                adjusted_allocations = {
                    symbol: alloc / total_allocated 
                    for symbol, alloc in adjusted_allocations.items()
                }
                
            return adjusted_allocations
            
        except Exception as e:
            logger.error(f"Error applying sector constraints: {str(e)}")
            return allocations


    def _apply_position_constraints(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Apply final position size constraints
        """
        try:
            adjusted_allocations = {}
            
            for symbol, allocation in allocations.items():
                # Apply min/max constraints
                adjusted = min(max(allocation, MIN_POSITION_SIZE), MAX_POSITION_SIZE)
                adjusted_allocations[symbol] = adjusted
            
            # Normalize to sum to 1.0
            total = sum(adjusted_allocations.values())
            if total > 0:
                return {
                    symbol: alloc / total 
                    for symbol, alloc in adjusted_allocations.items()
                }
            
            return adjusted_allocations
            
        except Exception as e:
            logger.error(f"Error applying position constraints: {str(e)}")
            return allocations
        
    def _handle_options_trading(self, state, date, data, allocation_ratios, 
                               transaction_cost, trackers):
        """Handle options trading with simplified parameters"""

        logger.info(f"\nHandling options trading for {date}")
        logger.info(f"Current cash: ${state._state['cash']:,.2f}")
        logger.info(f"Options enabled: {state._state.get('options_enabled', False)}")
        
        if not state._state.get('options_enabled'):
            logger.info("Options trading disabled")
            return
        try:
            # Execute options trading strategy
            new_holdings, new_cash, options_trades = state.options_strategy.trade_options(
                holdings=state._state['holdings'],
                cash=state._state['cash'],
                options_data=data['options_data'],
                price_data=data['price_data'],
                date=date,
                allocation_ratios=allocation_ratios,
                transaction_cost=transaction_cost,
                trackers=trackers,  
                treasury_data=data.get('treasury_data'),
                option_model=state._state.get('option_pricing_model')
            )
            
            # Process trades if any executed
            if options_trades:
                logger.info(f"\nExecuted {len(options_trades)} options trades")
                state._state['holdings'] = new_holdings
                state._state['cash'] = new_cash
                
                for trade in options_trades:
                    trackers['trade'].record_trade(
                        date=date,
                        action=f"{trade['type']}_option",
                        symbol=trade['symbol'],
                        quantity=trade['contracts'],
                        price=trade['price'],
                        value=trade['total_cost'],
                        cost=trade['transaction_cost']
                    )
                    trackers['transactions'].append(trade)
                    
        except Exception as e:
            logger.error(f"Error in options trading: {str(e)}")
            logger.error(traceback.format_exc())


    def _manage_existing_options(self, state, date, data, transaction_cost, trackers):
        """Manage existing options positions"""
        try:
            current_prices = {
                symbol: data['combined_prices'].loc[date, symbol]
                for symbol in data['price_data'].keys()
            }
            
            actions = state.options_strategy.manage_option_positions(
                date=date,
                holdings=state._state['holdings'],
                current_prices=current_prices
            )
            
            if not actions:
                return
                
            logger.info(f"\nProcessing {len(actions)} options position actions")
            
            for action in actions:
                position = action['position']
                option_id = action['option_id']
                
                if action['action'] == 'expire':
                    self._handle_option_expiration(
                        state, date, position, current_prices, 
                        option_id, trackers
                    )
                elif action['action'] == 'close':
                    self._handle_option_closure(
                        state, date, position, current_prices,
                        option_id, transaction_cost, trackers
                    )
                    
        except Exception as e:
            logger.error(f"Error managing options positions: {str(e)}")
            logger.error(traceback.format_exc())

    def _handle_option_expiration(self, state, date, position, current_prices, 
                                option_id, trackers):
        """Handle option expiration"""
        try:
            value = state.options_strategy._calculate_option_value(
                position,
                current_prices[position['symbol']]
            )
            
            # Record expiration
            trackers['trade'].record_trade(
                date=date,
                action='expire_option',
                symbol=position['symbol'],
                quantity=position['contracts'],
                price=0,
                value=value,
                cost=0
            )
            
            # Add to transactions
            trackers['transactions'].append({
                'date': date,
                'type': 'expire',
                'symbol': position['symbol'],
                'contracts': position['contracts'],
                'strike': position['strike'],
                'expiration': position['expiration'],
                'value': value,
                'total_cost': 0,
                'transaction_cost': 0
            })
            
            # Remove from holdings
            if option_id in state._state['holdings']['options']:
                del state._state['holdings']['options'][option_id]
                
        except Exception as e:
            logger.error(f"Error handling option expiration: {str(e)}")

    def _handle_option_closure(self, state, date, position, current_prices,
                            option_id, transaction_cost, trackers):
        """Handle option closure"""
        try:
            current_price = current_prices[position['symbol']]
            value = state.options_strategy._calculate_option_value(
                position, 
                current_price
            )
            trade_cost = value * transaction_cost
            
            # Update cash
            state._state['cash'] += (value - trade_cost)
            
            # Record closing trade
            trackers['trade'].record_trade(
                date=date,
                action='close_option',
                symbol=position['symbol'],
                quantity=position['contracts'],
                price=current_price,
                value=value,
                cost=trade_cost
            )
            
            # Add to transactions
            trackers['transactions'].append({
                'date': date,
                'type': 'close',
                'symbol': position['symbol'],
                'contracts': position['contracts'],
                'strike': position['strike'],
                'expiration': position['expiration'],
                'price': current_price,
                'value': value,
                'total_cost': value,
                'transaction_cost': trade_cost
            })
            
            # Remove from holdings
            if option_id in state._state['holdings']['options']:
                del state._state['holdings']['options'][option_id]
                
        except Exception as e:
            logger.error(f"Error handling option closure: {str(e)}")

    def _need_rebalance(self, current_date, last_rebalance, frequency):
        """Determines if rebalancing is needed based on frequency"""
        try:
            logger.info(f"\nChecking rebalance need:")
            logger.info(f"Current date: {current_date}")
            logger.info(f"Last rebalance: {last_rebalance}")
            logger.info(f"Frequency: {frequency}")

            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            if isinstance(last_rebalance, str):
                last_rebalance = pd.to_datetime(last_rebalance)

            days_passed = (current_date - last_rebalance).days
            logger.info(f"Days passed: {days_passed}")
            
            needs_rebalance = False
            
            if frequency == 'D':  # Daily
                needs_rebalance = True
            elif frequency == 'W':  # Weekly
                current_week = current_date.isocalendar()[1]
                last_week = last_rebalance.isocalendar()[1]
                needs_rebalance = (current_week != last_week)
            elif frequency == 'M':  # Monthly
                needs_rebalance = (
                    current_date.year != last_rebalance.year or 
                    current_date.month != last_rebalance.month
                )
            elif frequency == 'Q':  # Quarterly
                current_quarter = (current_date.year, (current_date.month - 1) // 3)
                last_quarter = (last_rebalance.year, (last_rebalance.month - 1) // 3)
                needs_rebalance = (current_quarter != last_quarter)
                    
            logger.info(f"Needs rebalance: {needs_rebalance}")
            if frequency == 'Q':
                logger.info(f"Current quarter: {current_quarter}")
                logger.info(f"Last quarter: {last_quarter}")
                    
            return needs_rebalance
                
        except Exception as e:
            logger.error(f"Error checking rebalance need: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def _process_option_action(self, state, action, date, current_prices, trackers):
        """Process option position actions"""
        try:
            position = action['position']
            symbol = position['symbol']
            current_price = current_prices[symbol]
            
            if action['action'] == 'exercise':
                # Handle exercise
                exercise_value = abs(current_price - position['strike']) * position['contracts'] * 100
                state._state['cash'] += exercise_value
                
                trackers['trade'].record_trade(
                    date=date,
                    action='exercise_option',
                    symbol=symbol,
                    quantity=position['contracts'],
                    price=position['strike'],
                    value=exercise_value,
                    cost=0
                )
                
            elif action['action'] == 'expire':
                # Record expired worthless
                trackers['trade'].record_trade(
                    date=date,
                    action='expire_option',
                    symbol=symbol,
                    quantity=position['contracts'],
                    price=0,
                    value=0,
                    cost=0
                )
                
            elif action['action'] == 'close':
                # Calculate closing value
                mid_price = position['entry_price']  # Simplified
                close_value = mid_price * position['contracts'] * 100
                
                state._state['cash'] += close_value
                
                trackers['trade'].record_trade(
                    date=date,
                    action='close_option',
                    symbol=symbol,
                    quantity=position['contracts'],
                    price=mid_price,
                    value=close_value,
                    cost=close_value * 0.0015  # Typical option closing cost
                )
                
        except Exception as e:
            logger.error(f"Error processing option action: {str(e)}")
        

        
    def check_risk_metrics(self, date, current_metrics, state, trackers):
        """Check risk metrics and take appropriate actions"""
        try:
            # Check volatility
            if current_metrics['volatility'] > MAX_VOLATILITY:
                self._handle_high_volatility(date, current_metrics, state, trackers)
                return False
                
            # Check drawdown
            if current_metrics['drawdown'] > MAX_DRAWDOWN:
                self._handle_excessive_drawdown(date, current_metrics, state, trackers)
                return False
                
            # Check concentration
            if current_metrics['position_concentration'] > MAX_POSITION_SIZE:
                self._handle_high_concentration(date, current_metrics)
                return True  # Continue with forced rebalancing
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk metrics: {str(e)}")
            return False
            
    def _handle_high_volatility(self, date, current_metrics, state, trackers):
        """Handle high volatility situation"""
        try:
            logger.warning(f"High volatility detected: {current_metrics['volatility']:.2%}")
            
            self.performance_monitor.record_risk_event(
                date=date,
                event_type='HIGH_VOLATILITY',
                details={
                    'volatility': current_metrics['volatility'],
                    'threshold': MAX_VOLATILITY
                }
            )
            
            if state['holdings']['equities']:
                self._reduce_position_sizes(date, state, trackers)
                
            if state['cash'] / state['total_value'] < 0.20:
                self._increase_cash_buffer(date, state, trackers)
                
        except Exception as e:
            logger.error(f"Error handling high volatility: {str(e)}")

    def perform_rebalancing(self, state, date, model, price_data, features, sequence_length, trackers):
        """Enhanced rebalancing with better risk control"""
        
        portfolio=EnhancedPortfolioOptimizer()
        predictor=EnhancedPredictor
        try:
            # First check if we can trade
            if not state.can_resume_trading(date):
                logger.info("Cannot trade - in recovery mode")
                return {'success': False, 'reason': 'In recovery mode'}
            
            features_for_date = self.data_pipeline.prepare_features(price_data, date)
            if not features_for_date:
                logger.warning(f"No features available for {date}")
                return {'success': False, 'reason': 'No ML data available'}

            # Create DataFrame from features
            df_ml_data = pd.DataFrame.from_dict(features_for_date, orient='index')
            df_ml_data.index.name = 'symbol'
            df_ml_data.reset_index(inplace=True)

            
            # Get model predictions
            selected_symbols = predictor.select_stocks_with_combined_model(
                model=model,
                date=date,
                price_data=price_data,
                features=features,
                sequence_length=sequence_length,
                top_n=TOP_N
            )
            
            if not selected_symbols or len(selected_symbols) < MIN_POSITIONS:
                return {'success': False, 'reason': f'Insufficient symbols selected: {len(selected_symbols) if selected_symbols else 0}'}
            
            # Calculate portfolio parameters with risk checks
            expected_returns, cov_matrix = portfolio.calculate_portfolio_parameters(
                selected_symbols, price_data, date
            )
            
            if expected_returns is None or cov_matrix is None:
                return {'success': False, 'reason': 'Failed to calculate portfolio parameters'}
                
            # Additional risk checks on selected stocks
            for symbol in selected_symbols:
                df = price_data[symbol]
                recent_data = df[df['date'] <= date].tail(20)
                returns = recent_data['close'].pct_change()
                vol = returns.std() * np.sqrt(252)
                
                # Skip high volatility stocks
                if vol > 0.5:  # 50% annualized volatility
                    logger.warning(f"Skipping {symbol} due to high volatility: {vol:.1%}")
                    selected_symbols.remove(symbol)
                    
            if len(selected_symbols) < MIN_POSITIONS:
                return {'success': False, 'reason': 'Insufficient symbols after volatility filter'}
                
            # Optimize portfolio
            weights = portfolio.optimize_portfolio(selected_symbols, expected_returns, cov_matrix)
            
            if not weights:
                return {'success': False, 'reason': 'Portfolio optimization failed'}
            
            return {
                'success': True,
                'selected_symbols': selected_symbols,
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"Error in rebalancing: {str(e)}")
            logger.error(traceback.format_exc())


            return {'success': False, 'reason': str(e)}
    def execute_trading_decisions(state, date, rebalancing_result, allocation_ratios,
                                transaction_cost, trackers):
        """Execute trading decisions with enhanced logging"""
        try:
            logger.info(f"\nExecuting trades for {date}")
            logger.info(f"Initial cash: ${state['cash']:,.2f}")
            logger.info(f"Initial equity positions: {state['holdings']['equities']}")
            
            # Execute trades
            state.execute_trades(
                date=date,
                selected_symbols=rebalancing_result['selected_symbols'],
                weights=rebalancing_result['weights'],
                allocation_ratios=allocation_ratios,
                transaction_cost=transaction_cost,
                combined_prices=state['combined_prices'],
                trackers=trackers
            )
            
            logger.info(f"After trades - Cash: ${state['cash']:,.2f}")
            logger.info(f"After trades - Equity positions: {state['holdings']['equities']}")
            
            # Handle options if enabled
            if state['options_enabled']:
                state.handle_options_trading(
                    date=date,
                    options_data=state['options_data'],
                    price_data=state['price_data'],
                    treasury_data=state['treasury_data'],
                    option_pricing_model=state['option_pricing_model'],
                    allocation_ratios=allocation_ratios,
                    transaction_cost=transaction_cost,
                    trackers=trackers
                )
                
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            logger.error(traceback.format_exc())

    
    def _prepare_backtest_results(self,
                                state: Dict,
                                trackers: Dict,
                                spy_benchmark: Optional[pd.Series] = None,
                                etf_holdings: Optional[Dict] = None) -> Dict:
        """
        Prepare comprehensive backtest results with ETF analysis
        
        Args:
            state: Portfolio state
            trackers: Performance trackers
            spy_benchmark: SPY benchmark data
            etf_holdings: ETF holdings data
            
        Returns:
            Dictionary of backtest results
        """
        try:
            # Basic validation
            if not isinstance(state['portfolio_value'], pd.Series) or len(state['portfolio_value']) == 0:
                logger.error("Invalid portfolio value series")
                return self._create_empty_results(state['start_date'])
            
            portfolio_value = state['portfolio_value'].copy()
            portfolio_value.index = portfolio_value.index.tz_localize(None)
            
            results = {
                'portfolio_value': portfolio_value,
                'transactions': pd.DataFrame(trackers['transactions']),
                'metrics': {},
                'etf_analysis': {},
                'sector_analysis': {},
                'risk_events': trackers['risk'].risk_events
            }

            
            if spy_benchmark is not None:
                # Ensure timezone-naive dates
                spy_index = spy_benchmark.index.tz_localize(None)
                aligned_benchmark = spy_benchmark.copy()
                aligned_benchmark.index = spy_index
                
                # Align and scale benchmark
                aligned_benchmark = aligned_benchmark.reindex(portfolio_value.index, method='ffill')
                scale_factor = portfolio_value.iloc[0] / aligned_benchmark.iloc[0]
                results['spy_benchmark'] = aligned_benchmark * scale_factor
                
            # Calculate metrics with aligned benchmark
            results['metrics'] = self._calculate_performance_metrics(
                portfolio_value=portfolio_value,
                benchmark=results.get('spy_benchmark')
            )
            
            # Add ETF-specific analysis
            if etf_holdings:
                etf_analysis = self._analyze_etf_performance(
                    portfolio_value=state['portfolio_value'],
                    transactions=results['transactions'],
                    etf_holdings=etf_holdings
                )
                results['etf_analysis'] = etf_analysis
            
            # Add sector analysis
            sector_analysis = self._analyze_sector_performance(
                transactions=results['transactions'],
                portfolio_value=state['portfolio_value'],
                state=state
            )
            results['sector_analysis'] = sector_analysis
            
            # Add risk analysis
            risk_analysis = self._analyze_risk_metrics(
                portfolio_value=state['portfolio_value'],
                transactions=results['transactions']
            
            )
            results['risk_analysis'] = risk_analysis
            
            # Log results summary
            self._log_results_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error preparing results: {str(e)}")
            return self._create_empty_results(state['start_date'])
        
    def _calculate_sector_turnover(self, transactions: pd.DataFrame) -> Dict[str, float]:
        """Calculate sector turnover metrics"""
        try:
            sector_turnover = defaultdict(float)
            if 'sector' not in transactions.columns:
                return {}
                
            # Group by date and sector
            grouped = transactions.groupby(['date', 'sector'])
            
            for (date, sector), group in grouped:
                buys = group[group['action'] == 'buy']['value'].sum()
                sells = abs(group[group['action'] == 'sell']['value'].sum())
                turnover = (buys + sells) / 2
                sector_turnover[sector] += turnover
                
            return dict(sector_turnover)
            
        except Exception as e:
            logger.error(f"Error calculating sector turnover: {str(e)}")
            return {}
        
    def _calculate_drawdown_series(self, portfolio_value: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        try:
            peak = portfolio_value.expanding(min_periods=1).max()
            drawdown = (portfolio_value - peak) / peak
            return drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown series: {str(e)}")
            return pd.Series(index=portfolio_value.index)
        
    def _analyze_etf_performance(self,
                               portfolio_value: pd.Series,
                               transactions: pd.DataFrame,
                               etf_holdings: Dict) -> Dict:
        """
        Analyze ETF-specific performance metrics
        """
        try:
            etf_analysis = {
                'etf_metrics': {},
                'holdings_overlap': {},
                'tracking_analysis': {}
            }
            
            # Filter ETF transactions
            etf_transactions = transactions[
                transactions['symbol'].apply(lambda x: self._is_etf(x))
            ]
            
            for symbol in etf_transactions['symbol'].unique():
                try:
                    # Get ETF-specific transactions
                    symbol_txns = etf_transactions[
                        etf_transactions['symbol'] == symbol
                    ]
                    
                    # Calculate ETF metrics
                    metrics = {
                        'total_trades': len(symbol_txns),
                        'total_volume': symbol_txns['value'].sum(),
                        'avg_position_size': symbol_txns['value'].mean(),
                        'holding_periods': self._calculate_holding_periods(symbol_txns)
                    }
                    
                    # Calculate holdings overlap if available
                    if symbol in etf_holdings:
                        overlap_analysis = self._analyze_holdings_overlap(
                            transactions=transactions,
                            holdings=etf_holdings[symbol]['holdings']
                        )
                        etf_analysis['holdings_overlap'][symbol] = overlap_analysis
                    
                    # Calculate tracking metrics
                    tracking_analysis = self._analyze_tracking_metrics(
                        etf_txns=symbol_txns,
                        holdings=etf_holdings.get(symbol, {}).get('holdings')
                    )
                    
                    etf_analysis['etf_metrics'][symbol] = metrics
                    etf_analysis['tracking_analysis'][symbol] = tracking_analysis
                    
                except Exception as e:
                    logger.error(f"Error analyzing ETF {symbol}: {str(e)}")
                    continue
            
            return etf_analysis
            
        except Exception as e:
            logger.error(f"Error in ETF performance analysis: {str(e)}")
            return {}

    def _analyze_sector_performance(self,
                                  sector_exposures: Dict[str, List],
                                  portfolio_value: pd.Series) -> Dict:
        """Analyze sector-level performance"""
        try:
            sector_analysis = {}
            
            for sector, history in sector_exposures.items():
                # Convert history to DataFrame
                df = pd.DataFrame(history)
                df.set_index('date', inplace=True)
                
                # Calculate sector metrics
                sector_analysis[sector] = {
                    'avg_weight': df['weight'].mean(),
                    'max_weight': df['weight'].max(),
                    'min_weight': df['weight'].min(),
                    'weight_volatility': df['weight'].std(),
                    'num_days_overweight': sum(df['weight'] > self.etf_constraints['max_sector'])
                }
                
                # Calculate sector contribution if we have portfolio value
                if len(df) == len(portfolio_value):
                    portfolio_returns = portfolio_value.pct_change()
                    sector_contribution = (portfolio_returns * df['weight']).sum()
                    sector_analysis[sector]['return_contribution'] = sector_contribution
                    
            return sector_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sector performance: {str(e)}")
            return {}

    def _analyze_risk_metrics(self, portfolio_value: pd.Series,
                              transactions: pd.DataFrame) -> Dict:
                
        """Calculate comprehensive risk metrics with proper error handling"""
        try:
            risk_analysis = {
                'drawdown_analysis': {},
                'volatility_analysis': {},
                'concentration_risk': {},
                'tail_risk': {}
            }

            if len(portfolio_value) < 2:
                logger.warning("Insufficient data for risk analysis")
                return risk_analysis

            # Drawdown analysis
            drawdown = self._calculate_drawdown_series(portfolio_value)
            max_dd = drawdown.min() if not drawdown.empty else 0
            risk_analysis['drawdown_analysis'] = {
                'max_drawdown': max_dd,
                'avg_drawdown': drawdown.mean() if not drawdown.empty else 0,
                'drawdown_days': len(drawdown[drawdown < 0]) if not drawdown.empty else 0,
                'recovery_periods': self._analyze_recovery_periods(drawdown)
            }

            # Volatility analysis
            returns = portfolio_value.pct_change().dropna()
            if len(returns) > 0:
                risk_analysis['volatility_analysis'] = {
                    'annual_vol': returns.std() * np.sqrt(252),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'var_95': np.percentile(returns, 5) if len(returns) > 20 else 0,
                    'cvar_95': returns[returns <= np.percentile(returns, 5)].mean() if len(returns) > 20 else 0
                }

            # Concentration risk if transactions exist
            if not transactions.empty and 'value' in transactions.columns:
                position_sizes = transactions['value'] / portfolio_value[transactions['date']]
                risk_analysis['concentration_risk'] = {
                    'max_position': position_sizes.max() if not position_sizes.empty else 0,
                    'avg_position': position_sizes.mean() if not position_sizes.empty else 0,
                    'position_stdev': position_sizes.std() if not position_sizes.empty else 0
                }

            # Tail risk metrics
            if len(returns) > 0:
                risk_analysis['tail_risk'] = {
                    'worst_day': returns.min(),
                    'best_day': returns.max(),
                    'downside_vol': returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
                    'sortino': (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)) 
                            if len(returns[returns < 0]) > 0 and returns[returns < 0].std() > 0 else 0
                }

            return risk_analysis

        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            return {
                'drawdown_analysis': {},
                'volatility_analysis': {},
                'concentration_risk': {},
                'tail_risk': {}
            }

    def _log_results_summary(self, results: Dict):
        """Print comprehensive results summary"""
        try:
            logger.info("\n=== Backtest Results Summary ===")
            
            # Performance metrics
            logger.info("\nPerformance Metrics:")
            for metric, value in results['metrics'].items():
                logger.info(f"{metric}: {value:.2%}")
            
            # ETF analysis
            if results.get('etf_analysis'):
                logger.info("\nETF Analysis:")
                for symbol, metrics in results['etf_analysis']['etf_metrics'].items():
                    logger.info(f"\n{symbol}:")
                    logger.info(f"Total trades: {metrics['total_trades']}")
                    logger.info(f"Average position size: ${metrics['avg_position_size']:,.2f}")
                    logger.info(f"Average holding period: {np.mean(metrics['holding_periods']):.1f} days")
            
            # Sector analysis
            if results.get('sector_analysis'):
                logger.info("\nSector Performance:")
                for sector, metrics in results['sector_analysis']['sector_returns'].items():
                    logger.info(f"\n{sector}:")
                    logger.info(f"Total return: {metrics['total_return']:.2%}")
                    logger.info(f"Average exposure: {metrics['avg_exposure']:.2%}")
                    logger.info(f"Volatility: {metrics['volatility']:.2%}")
            
            # Risk metrics
            if results.get('risk_analysis'):
                logger.info("\nRisk Metrics:")
                risk = results['risk_analysis']
                logger.info(f"Maximum drawdown: {risk['drawdown_analysis']['max_drawdown']:.2%}")
                logger.info(f"Annual volatility: {risk['volatility_analysis']['annual_vol']:.2%}")
                logger.info(f"Value at Risk (95%): {risk['volatility_analysis']['var_95']:.2%}")
                logger.info(f"Sortino ratio: {risk['tail_risk']['sortino']:.2f}")
            
        except Exception as e:
            logger.error(f"Error logging results summary: {str(e)}")
        
    def log_final_results(state, trackers, metrics):
        """Log final backtest results and summaries"""
        performance_monitor = PerformanceMonitor()
        try:
            logger.info("\nFinal Portfolio State:")
            logger.info(f"Series length: {len(state['portfolio_value'])}")
            logger.info(f"Date range: {state['portfolio_value'].index[0]} to {state['portfolio_value'].index[-1]}")
            logger.info(f"Final value: ${state['portfolio_value'].iloc[-1]:,.2f}")
            
            # Print tracker summaries
            trackers['trade'].print_summary()
            trackers['event'].print_summary()
            trackers['rebalance'].print_summary()
            
            logger.info("\n=== Final Performance Metrics ===")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.2f}")

            performance_monitor.print_summary()
            performance_monitor.save_results()
                
        except Exception as e:
            logger.error(f"Error logging final results: {str(e)}")



    def _create_empty_results(self, start_date):
        """Create empty results structure for error cases"""
        return {
            'portfolio_value': pd.Series(self.initial_capital, index=[pd.to_datetime(start_date)]),
            'transactions': pd.DataFrame(),
            'metrics': {
                'Total Return (%)': 0.0,
                'Annual Return (%)': 0.0,
                'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Win Rate (%)': 0.0,
                'Number of Trades': 0,
                'Success Rate (%)': 0.0,
                'Average Trade Return (%)': 0.0,
                'Average Holding Period': 0.0
            }
        }

    def initialize_trackers(self):
        """Initialize all tracking objects for backtest"""
        trackers = {
            'rebalance': RebalanceTracker(),
            'trade': TradeTracker(),
            'event': self.performance_monitor,  # Use instance from class
            'risk': self.risk_manager,  # Use instance from class
            'transactions': []
        }
        logger.info("Initialized tracking systems")
        return trackers

    def initialize_state(self, price_data, options_data, start_date, end_date, initial_capital, 
                        option_pricing_model=None, treasury_data=None):
        """
        Initialize backtest state including data preparation
        Args:
            price_data: Dictionary of price data for each symbol
            options_data: Dictionary of options data for each symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital amount
            option_pricing_model: Trained options pricing model (optional)
            treasury_data: Treasury rates data (optional)
        """
        try:
            # Convert dates
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Prepare data
            treasury_data = prepare_treasury_data(treasury_data)
            combined_prices, date_range = prepare_price_data(price_data, start_date, end_date)
            
            if len(date_range) == 0:
                raise ValueError("No valid dates in range")
                
            # Initialize portfolio state
            state = PortfolioState(initial_capital, combined_prices)
            
            # Store additional state information
            state._state.update({
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'date_range': date_range,
                'price_data': price_data,
                'options_data': options_data,
                'treasury_data': treasury_data,
                'last_rebalance': start_date - pd.Timedelta(days=1),
                'options_enabled': False,  # Will be updated if options are validated
                'option_pricing_model': option_pricing_model,  # Store the model in state
                'options_strategy': self.options_strategy
            })
            
            # Validate options if available
            if option_pricing_model and option_pricing_model.is_trained:
                if self.options_strategy.validate_options_data(options_data):
                    state._state['options_enabled'] = True
                    logger.info("Options trading enabled")
                else:
                    logger.warning("Options data validation failed - options trading disabled")
                    
            return state
            
        except Exception as e:
            logger.error(f"Error initializing state: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def is_risk_check_day(date):
        """Determine if risk should be checked on this date"""
        if RISK_CHECK_FREQUENCY == 'D':
            return True
        elif RISK_CHECK_FREQUENCY == 'W':
            return date.weekday() == 0  # Monday
        return False

    def count_consecutive_losses(returns):
        """Count maximum consecutive losing days"""
        try:
            # Convert returns to boolean array of losses
            losses = returns < 0
            if not any(losses):  # If no losses at all
                return 0
                
            # Group consecutive losses and count
            loss_runs = [sum(1 for _ in group) 
                        for key, group in itertools.groupby(losses) 
                        if key]
            
            return max(loss_runs) if loss_runs else 0
            
        except Exception as e:
            logger.error(f"Error counting consecutive losses: {str(e)}")
            return 0


    def handle_risk_checks(self, date, trackers):
        """Modified risk management with position-based checks"""
        try:
            # Only do full risk checks if we have positions
            has_positions = bool(self._state['holdings']['equities'])
            
            current_value = self._state['total_value']
            initial_value = self._state['initial_capital']
            
            # Basic metrics
            if len(self._state['portfolio_value']) > 1:
                daily_return = self._state['portfolio_value'].pct_change().iloc[-1]
                drawdown = (initial_value - current_value) / initial_value
            else:
                daily_return = 0
                drawdown = 0
            
            risk_breach = False
            reasons = []
            
            if has_positions:
                # 1. Emergency stop (always check)
                if daily_return < -0.07:  # 7% daily loss
                    risk_breach = True
                    reasons.append(f"Emergency stop - large loss: {daily_return:.1%}")
                
                # 2. Regular checks
                elif len(self._state['portfolio_value']) >= 20:
                    # Volatility check
                    returns = self._state['portfolio_value'].pct_change()
                    vol = returns.tail(20).std() * np.sqrt(252)
                    
                    if vol > MAX_VOLATILITY:
                        risk_breach = True
                        reasons.append(f"High volatility: {vol:.1%}")
                    
                    # Drawdown check relative to market
                    if drawdown > MAX_DRAWDOWN:
                        # Get market drawdown
                        try:
                            market_data = yf.download('SPY', 
                                                    start=(date - pd.Timedelta(days=30)),
                                                    end=date,
                                                    progress=False)['Adj Close']
                            market_dd = (market_data.max() - market_data.iloc[-1]) / market_data.max()
                            
                            if drawdown > market_dd * 1.5:
                                risk_breach = True
                                reasons.append(f"Excessive drawdown vs market: {drawdown:.1%} vs {market_dd:.1%}")
                        except:
                            pass  # Skip market comparison if data fetch fails
            
            if risk_breach:
                self.handle_risk_breach(date, trackers)
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Risk check error: {str(e)}")
            return False



    def _calculate_empty_metrics(self):
        """Return empty metrics dictionary for error cases"""
        return {
            'Total Return (%)': 0.0,
            'Annual Return (%)': 0.0,
            'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown (%)': 0.0,
            'Win Rate (%)': 0.0,
            'Number of Trades': 0,
            'Success Rate (%)': 0.0,
            'Average Trade Return (%)': 0.0,
            'Average Holding Period': 0.0
        }




   

    def _handle_high_volatility(self, date, current_metrics, state, trackers, performance_monitor):
        """Handle high volatility situation"""
        try:
            logger.warning(f"High volatility detected: {current_metrics['volatility']:.2%}")
            
            # Record risk event
            performance_monitor.record_risk_event(
                date=date,
                event_type='HIGH_VOLATILITY',
                details={
                    'volatility': current_metrics['volatility'],
                    'threshold': MAX_VOLATILITY
                }
            )
            
            # Reduce position sizes
            if state['holdings']['equities']:
                self._reduce_position_sizes(
                    date=date,
                    state=state,
                    trackers=trackers,
                    reduction_factor=0.5  # Reduce to half size
                )
                
            # Increase cash buffer if needed
            if state['cash'] / state['total_value'] < 0.20:
                self._increase_cash_buffer(
                    date=date,
                    state=state,
                    trackers=trackers
                )
                
        except Exception as e:
            logger.error(f"Error handling high volatility: {str(e)}")
            logger.error(traceback.format_exc())

    def _handle_excessive_drawdown(self, date, current_metrics, state, trackers, performance_monitor):
        """Handle excessive drawdown situation"""
        try:
            logger.warning(f"Maximum drawdown exceeded: {current_metrics['drawdown']:.2%}")
            
            performance_monitor.record_risk_event(
                date=date,
                event_type='EXCESSIVE_DRAWDOWN',
                details={
                    'drawdown': current_metrics['drawdown'],
                    'threshold': MAX_DRAWDOWN
                }
            )
            
            # Handle risk breach
            state.handle_risk_breach(date, trackers)
            
        except Exception as e:
            logger.error(f"Error handling excessive drawdown: {str(e)}")
            logger.error(traceback.format_exc())

    def _handle_high_concentration(self, date, current_metrics, performance_monitor):
        """Handle high position concentration"""
        try:
            logger.warning("Position concentration too high")
            
            performance_monitor.record_risk_event(
                date=date,
                event_type='HIGH_CONCENTRATION',
                details={
                    'concentration': current_metrics['position_concentration'],
                    'threshold': MAX_POSITION_SIZE
                }
            )
            
            # Return True to force rebalancing
            return True
            
        except Exception as e:
            logger.error(f"Error handling high concentration: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _reduce_position_sizes(self, date, state, trackers, reduction_factor):
        """Reduce position sizes by given factor"""
        try:
            logger.info(f"Reducing position sizes by factor: {reduction_factor}")
            
            for symbol in list(state['holdings']['equities'].keys()):
                current_size = (state['holdings']['equities'][symbol] * 
                            state['combined_prices'].loc[date, symbol] / 
                            state['total_value'])
                
                max_reduced_size = MAX_POSITION_SIZE * reduction_factor
                
                if current_size > max_reduced_size:
                    # Calculate new position
                    new_shares = int(
                        (max_reduced_size * state['total_value']) / 
                        state['combined_prices'].loc[date, symbol]
                    )
                    
                    # Execute reduction
                    self._execute_position_reduction(
                        date=date,
                        symbol=symbol,
                        current_shares=state['holdings']['equities'][symbol],
                        new_shares=new_shares,
                        state=state,
                        trackers=trackers,
                        reason='volatility_reduction'
                    )
                    
        except Exception as e:
            logger.error(f"Error reducing position sizes: {str(e)}")
            logger.error(traceback.format_exc())

    def _increase_cash_buffer(self, date, state, trackers):
        """Increase cash buffer by selling portion of positions"""
        try:
            logger.info("Increasing cash buffer")
            
            for symbol in list(state['holdings']['equities'].keys()):
                current_shares = state['holdings']['equities'][symbol]
                shares_to_sell = int(current_shares * 0.10)  # Sell 10% of position
                
                if shares_to_sell > 0:
                    self._execute_position_reduction(
                        date=date,
                        symbol=symbol,
                        current_shares=current_shares,
                        new_shares=current_shares - shares_to_sell,
                        state=state,
                        trackers=trackers,
                        reason='increase_cash_buffer'
                    )
                    
        except Exception as e:
            logger.error(f"Error increasing cash buffer: {str(e)}")
            logger.error(traceback.format_exc())

    def _execute_position_reduction(self, date, symbol, current_shares, new_shares, 
                                state, trackers, reason):
        """Execute position reduction and record transaction"""
        try:
            shares_to_sell = current_shares - new_shares
            
            if shares_to_sell > 0:
                price = state['combined_prices'].loc[date, symbol]
                sale_value = shares_to_sell * price * (1 - TRANSACTION_COST)
                trade_cost = shares_to_sell * price * TRANSACTION_COST
                
                trackers['transactions'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': -shares_to_sell,
                    'price': price,
                    'value': sale_value,
                    'cost': trade_cost,
                    'reason': reason
                })
                
                # Update holdings and cash
                state['holdings']['equities'][symbol] = new_shares
                state['cash'] += sale_value
                
        except Exception as e:
            logger.error(f"Error executing position reduction: {str(e)}")
            logger.error(traceback.format_exc())
                
       
    def _calculate_portfolio_metrics(self, portfolio_value_series):
        """Calculate portfolio metrics with sanity checks"""
        try:
            returns = portfolio_value_series.pct_change().dropna()
            
            # Add sanity checks for returns
            max_daily_return = 0.5  # 50% maximum daily return
            min_daily_return = -0.5  # -50% minimum daily return
            returns = returns.clip(min_daily_return, max_daily_return)
            
            total_return = (portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0] - 1) * 100
            
            # Cap maximum total return at a reasonable level
            max_annual_return = 200  # 200% maximum annual return
            years = len(portfolio_value_series) / 252
            max_total_return = max_annual_return * years
            total_return = min(total_return, max_total_return)
            
            # Calculate other metrics
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe = (returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252))
            
            return {
                'Total Return (%)': total_return,
                'Annual Return (%)': ((1 + total_return/100) ** (1/years) - 1) * 100,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe,
                'Max Drawdown (%)': self._calculate_max_drawdown(portfolio_value_series),
                'Win Rate (%)': (returns > 0).mean() * 100,
                'Number of Trades': len(portfolio_value_series)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return self._create_empty_metrics()
                
            
        
    def _plot_results(self, results, start_date, end_date):
        """Plot and save backtest results"""
        try:
            plot_portfolio_vs_market(
                portfolio_value=results['portfolio_value'],
                start_date=start_date,
                end_date=end_date,
                save_path='backtest_results_plot.png'
            )
            
            # Save results to file
            save_results(results)
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")


    def _calculate_max_drawdown(self, portfolio_value_series):
        """Calculate maximum drawdown"""
        try:
            peak = portfolio_value_series.expanding(min_periods=1).max()
            drawdown = ((portfolio_value_series - peak) / peak) * 100
            return drawdown.min()
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _create_empty_metrics(self):
        """Create empty metrics dictionary"""
        return {
            'Total Return (%)': 0.0,
            'Annual Return (%)': 0.0,
            'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown (%)': 0.0,
            'Win Rate (%)': 0.0,
            'Number of Trades': 0
        }