
from collections import defaultdict
import logging
import traceback
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)

class OptionsReport:
    """Generate detailed options trading analysis and reports"""
    
    def __init__(self):
        self.trades = []
        self.positions = []
        self.expirations = []
        self.performance_metrics = defaultdict(float)

    def _safe_get_value(self, transaction, key, default=0):
        """Safely get value from transaction, whether it's a dict or string"""
        if isinstance(transaction, dict):
            return float(transaction.get(key, default))
        return default
        
    def generate_report(self, transactions, portfolio_value_series):
        """Generate comprehensive options trading report"""
        try:
            logger.info("\n=== Options Trading Report ===")
            
            # Filter for options transactions
            if isinstance(transactions, pd.DataFrame):
                options_df = transactions[transactions['trade_type'] == 'option'].copy()
            else:
                options_df = pd.DataFrame([
                    t for t in transactions 
                    if isinstance(t, dict) and t.get('trade_type') == 'option'
                ])
            
            if options_df.empty:
                logger.info("No options transactions to report")
                return
                
            # Basic trading metrics
            total_premium = options_df['value'].sum() if 'value' in options_df.columns else 0
            total_profit = options_df['expected_profit'].sum() if 'expected_profit' in options_df.columns else 0
            
            logger.info("\nTrading Activity:")
            logger.info(f"Total options trades: {len(options_df)}")
            logger.info(f"Total premium collected: ${total_premium:,.2f}")
            logger.info(f"Expected total profit: ${total_profit:,.2f}")
            if len(options_df) > 0:
                avg_profit = total_profit / len(options_df)
                logger.info(f"Average profit per trade: ${avg_profit:.2f}")

            # Strategy breakdown
            if 'action' in options_df.columns:
                logger.info("\nStrategy Breakdown:")
                by_strategy = options_df.groupby('action').agg({
                    'value': ['count', 'sum'],
                    'expected_profit': 'sum'
                }).round(2)
                
                for strategy in by_strategy.index:
                    counts = by_strategy.loc[strategy]
                    logger.info(f"\n{strategy}:")
                    logger.info(f"Number of trades: {counts[('value', 'count')]}")
                    logger.info(f"Total value: ${counts[('value', 'sum')]:,.2f}")
                    logger.info(f"Total profit: ${counts[('expected_profit', 'sum')]:,.2f}")

            # Monthly performance
            if 'date' in options_df.columns:
                logger.info("\nMonthly Performance:")
                options_df['month'] = pd.to_datetime(options_df['date']).dt.to_period('M')
                monthly_stats = options_df.groupby('month').agg({
                    'value': ['count', 'sum'],
                    'expected_profit': 'sum'
                }).round(2)
                
                for month in monthly_stats.index:
                    stats = monthly_stats.loc[month]
                    logger.info(f"\n{month}:")
                    logger.info(f"Number of trades: {stats[('value', 'count')]}")
                    logger.info(f"Total value: ${stats[('value', 'sum')]:,.2f}")
                    logger.info(f"Total profit: ${stats[('expected_profit', 'sum')]:,.2f}")

            # Add risk metrics if portfolio value provided
            if isinstance(portfolio_value_series, pd.Series) and len(portfolio_value_series) > 0:
                self._print_risk_metrics(options_df, portfolio_value_series)
            
        except Exception as e:
            logger.error(f"Error generating options report: {str(e)}")
            logger.error(traceback.format_exc())

    def _calculate_total_pnl(self, transactions):
        """Calculate total P&L from options trades"""
        pnl = 0
        open_positions = {}
        
        for t in transactions:
            if 'buy' in t['action']:
                # Opening trade
                key = f"{t['symbol']}_{t['price']}_{t.get('expiration', '')}"
                open_positions[key] = {
                    'cost': t['value'] + t['cost'],
                    'contracts': t['quantity']
                }
            elif any(x in t['action'] for x in ['sell', 'exercise', 'expire']):
                # Closing trade
                key = f"{t['symbol']}_{t['price']}_{t.get('expiration', '')}"
                if key in open_positions:
                    entry = open_positions[key]
                    pnl += (t['value'] - entry['cost'])
                    del open_positions[key]
                    
        return pnl
        
    def _print_strategy_breakdown(self, transactions: pd.DataFrame):
        """Breakdown by strategy type"""
        try:
            if 'type' not in transactions.columns:
                return
                
            logger.info("\nStrategy Breakdown:")
            strategy_stats = transactions.groupby('type').agg({
                'value': ['count', 'sum'],
                'expected_profit': 'sum'
            }).round(2)
            
            for strategy in strategy_stats.index:
                stats = strategy_stats.loc[strategy]
                trades = stats[('value', 'count')]
                premium = stats[('value', 'sum')]
                profit = stats[('expected_profit', 'sum')]
                win_rate = len(transactions[
                    (transactions['type'] == strategy) & 
                    (transactions['expected_profit'] > 0)
                ]) / trades * 100 if trades > 0 else 0
                
                logger.info(f"\n{strategy}:")
                logger.info(f"Number of trades: {trades}")
                logger.info(f"Total premium collected: ${premium:,.2f}")
                logger.info(f"Expected profit: ${profit:,.2f}")
                logger.info(f"Win rate: {win_rate:.1f}%")
                
        except Exception as e:
            logger.error(f"Error in strategy breakdown: {str(e)}")


    def _print_outcome_analysis(self, transactions):
        """Analyze and print options trade outcomes"""
        try:
            logger.info("\nTrade Outcomes:")
            
            outcomes = {
                'exercised': 0,
                'expired': 0,
                'closed': 0
            }
            
            for t in transactions:
                if 'exercise' in t['action']:
                    outcomes['exercised'] += 1
                elif 'expire' in t['action']:
                    outcomes['expired'] += 1
                elif 'close' in t['action']:
                    outcomes['closed'] += 1
                    
            total = sum(outcomes.values())
            if total > 0:
                for outcome, count in outcomes.items():
                    percentage = (count / total) * 100
                    logger.info(f"{outcome.capitalize()}: {count} ({percentage:.1f}%)")
                    
        except Exception as e:
            logger.error(f"Error in outcome analysis: {str(e)}")
            
    def _print_monthly_performance(self, transactions: pd.DataFrame):
        """Monthly performance analysis"""
        try:
            if 'date' not in transactions.columns:
                return
                
            logger.info("\nMonthly Performance:")
            transactions['month'] = pd.to_datetime(transactions['date']).dt.to_period('M')
            monthly_stats = transactions.groupby('month').agg({
                'value': ['count', 'sum'],
                'expected_profit': 'sum'
            }).round(2)
            
            for month in monthly_stats.index:
                stats = monthly_stats.loc[month]
                logger.info(f"\n{month}:")
                logger.info(f"Number of trades: {stats[('value', 'count')]}")
                logger.info(f"Premium collected: ${stats[('value', 'sum')]:,.2f}")
                logger.info(f"Expected profit: ${stats[('expected_profit', 'sum')]:,.2f}")
                
        except Exception as e:
            logger.error(f"Error in monthly performance: {str(e)}")
            
    def _print_position_management(self, transactions):
        """Analyze and print position management metrics"""
        try:
            logger.info("\nPosition Management:")
            
            # Average holding period
            holding_periods = []
            open_positions = {}
            
            for t in transactions:
                if 'buy' in t['action']:
                    key = f"{t['symbol']}_{t['price']}_{t.get('expiration', '')}"
                    open_positions[key] = pd.to_datetime(t['date'])
                elif any(x in t['action'] for x in ['sell', 'exercise', 'expire']):
                    key = f"{t['symbol']}_{t['price']}_{t.get('expiration', '')}"
                    if key in open_positions:
                        start_date = open_positions[key]
                        end_date = pd.to_datetime(t['date'])
                        holding_period = (end_date - start_date).days
                        holding_periods.append(holding_period)
                        
            if holding_periods:
                avg_holding = np.mean(holding_periods)
                logger.info(f"Average holding period: {avg_holding:.1f} days")
                logger.info(f"Max holding period: {max(holding_periods)} days")
                logger.info(f"Min holding period: {min(holding_periods)} days")
                
        except Exception as e:
            logger.error(f"Error in position management: {str(e)}")
            
    def _print_risk_metrics(self, options_df: pd.DataFrame, portfolio_value_series: pd.Series):
        """Calculate and print risk metrics"""
        try:
            logger.info("\nRisk Metrics:")
            
            # Calculate maximum drawdown from options
            options_value = pd.Series(0, index=portfolio_value_series.index)
            for idx, row in options_df.iterrows():
                date = pd.to_datetime(row['date'])
                if date in options_value.index:
                    options_value[date:] += row['value']
                
            max_drawdown = ((options_value.cummax() - options_value) / portfolio_value_series).max()
            logger.info(f"Maximum drawdown from options: {max_drawdown:.1%}")
            
            # Calculate other risk metrics
            total_value = options_df['value'].sum()
            max_position = options_df['value'].max()
            
            logger.info(f"Total options exposure: ${total_value:,.2f}")
            logger.info(f"Largest single position: ${max_position:,.2f}")
                
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")