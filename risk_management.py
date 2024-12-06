# risk_management.py

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from config import (
    MAX_DRAWDOWN, MAX_VOLATILITY, MAX_POSITION_SIZE,
    RISK_CHECK_FREQUENCY, MIN_RECOVERY, MAX_RECOVERY_DAYS
)

logger = logging.getLogger(__name__)

class ETFRiskManager:
    """Enhanced risk management specialized for ETF portfolios"""
    
    def __init__(self,
                 max_tracking_error: float = 0.02,  # 2% max tracking error
                 max_sector_deviation: float = 0.05,  # 5% max sector deviation from SPY
                 min_liquidity_score: float = 0.8,  # Minimum liquidity requirement
                 max_position_size: float = 0.05,  # 5% max single position
                 max_drawdown: float = MAX_DRAWDOWN,
                 max_volatility: float = MAX_VOLATILITY):
        
        self.max_tracking_error = max_tracking_error
        self.max_sector_deviation = max_sector_deviation
        self.min_liquidity_score = min_liquidity_score
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_volatility = max_volatility
        
        # Initialize tracking
        self.tracking_error_history = []
        self.sector_exposures = defaultdict(list)
        self.risk_events = []
        self.liquidity_scores = {}
        self.portfolio_stats = defaultdict(list)
        
    def calculate_etf_risk_metrics(self,
                                 holdings: Dict[str, float],
                                 prices: Dict[str, float],
                                 date: datetime,
                                 sector_map: Optional[Dict] = None,
                                 spy_data: Optional[pd.Series] = None) -> Dict:
        """Calculate comprehensive ETF risk metrics"""
        try:
            metrics = {}
            portfolio_value = sum(shares * prices.get(symbol, 0) 
                                for symbol, shares in holdings.items())
            
            # 1. Tracking Error vs SPY
            if spy_data is not None:
                tracking_error = self._calculate_tracking_error(
                    holdings=holdings,
                    prices=prices,
                    spy_data=spy_data,
                    lookback_days=60
                )
                metrics['tracking_error'] = tracking_error
                self.tracking_error_history.append(tracking_error)
            
            # 2. Sector Analysis
            if sector_map:
                sector_weights = self._calculate_sector_weights(
                    holdings=holdings,
                    prices=prices,
                    sector_map=sector_map
                )
                metrics['sector_weights'] = sector_weights
                
                # Store sector history
                for sector, weight in sector_weights.items():
                    self.sector_exposures[sector].append({
                        'date': date,
                        'weight': weight
                    })
            
            # 3. Liquidity Analysis
            liquidity_scores = self._calculate_liquidity_scores(
                holdings=holdings,
                prices=prices,
                date=date
            )
            metrics['liquidity_scores'] = liquidity_scores
            
            # 4. Portfolio Concentration
            position_weights = {
                symbol: (shares * prices.get(symbol, 0)) / portfolio_value
                for symbol, shares in holdings.items()
            }
            metrics['max_position'] = max(position_weights.values())
            metrics['concentration'] = sum(w * w for w in position_weights.values())
            
            # Store metrics history
            self.portfolio_stats['concentration'].append({
                'date': date,
                'value': metrics['concentration']
            })
            
            # 5. Risk Flags
            risk_flags = {
                'high_tracking_error': metrics.get('tracking_error', 0) > self.max_tracking_error,
                'sector_deviation': self._check_sector_deviations(sector_weights),
                'liquidity_warning': min(liquidity_scores.values()) < self.min_liquidity_score,
                'concentration_warning': metrics['concentration'] > 0.15
            }
            metrics['risk_flags'] = risk_flags
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating ETF risk metrics: {str(e)}")
            return {}
            
    def _calculate_tracking_error(self,
                                holdings: Dict[str, float],
                                prices: Dict[str, float],
                                spy_data: pd.Series,
                                lookback_days: int = 60) -> float:
        """Calculate tracking error versus SPY"""
        try:
            # Get historical prices for holdings
            end_date = spy_data.index[-1]
            start_date = end_date - pd.Timedelta(days=lookback_days)
            
            # Calculate portfolio returns
            portfolio_value = pd.Series(0.0, index=spy_data.index)
            for symbol, shares in holdings.items():
                if symbol in prices:
                    portfolio_value += shares * prices[symbol]
                    
            portfolio_returns = portfolio_value.pct_change().dropna()
            spy_returns = spy_data.pct_change().dropna()
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            if len(common_dates) < 20:  # Need minimum history
                return 0.0
                
            portfolio_returns = portfolio_returns[common_dates]
            spy_returns = spy_returns[common_dates]
            
            # Calculate tracking error
            return_diff = portfolio_returns - spy_returns
            tracking_error = return_diff.std() * np.sqrt(252)
            
            return tracking_error
            
        except Exception as e:
            logger.error(f"Error calculating tracking error: {str(e)}")
            return 0.0
            
    def _calculate_sector_weights(self,
                                holdings: Dict[str, float],
                                prices: Dict[str, float],
                                sector_map: Dict[str, str]) -> Dict[str, float]:
        """Calculate sector weights of portfolio"""
        try:
            sector_values = defaultdict(float)
            total_value = 0
            
            for symbol, shares in holdings.items():
                if symbol in prices and symbol in sector_map:
                    value = shares * prices[symbol]
                    sector = sector_map[symbol]
                    sector_values[sector] += value
                    total_value += value
                    
            # Calculate weights
            return {
                sector: value/total_value
                for sector, value in sector_values.items()
            } if total_value > 0 else {}
            
        except Exception as e:
            logger.error(f"Error calculating sector weights: {str(e)}")
            return {}
            
    def _calculate_liquidity_scores(self,
                                  holdings: Dict[str, float],
                                  prices: Dict[str, float],
                                  date: datetime) -> Dict[str, float]:
        """Calculate liquidity scores for holdings"""
        try:
            liquidity_scores = {}
            lookback_days = 20
            
            for symbol in holdings:
                try:
                    # Get historical volume data
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=date - timedelta(days=lookback_days),
                        end=date
                    )
                    
                    if hist.empty:
                        continue
                        
                    # Calculate liquidity metrics
                    avg_volume = hist['Volume'].mean()
                    avg_price = hist['Close'].mean()
                    avg_dollar_volume = avg_volume * avg_price
                    
                    # Calculate liquidity score (0-1)
                    # Higher is better
                    liquidity_score = min(1.0, avg_dollar_volume / 10000000)  # $10M daily volume benchmark
                    liquidity_scores[symbol] = liquidity_score
                    
                except Exception as e:
                    logger.warning(f"Error calculating liquidity for {symbol}: {str(e)}")
                    continue
                    
            return liquidity_scores
            
        except Exception as e:
            logger.error(f"Error calculating liquidity scores: {str(e)}")
            return {}
            
    def _check_sector_deviations(self, sector_weights: Dict[str, float]) -> bool:
        """Check if sector weights deviate too much from benchmark"""
        try:
            # Get SPY sector weights (could be cached/updated periodically)
            spy = yf.Ticker('SPY')
            spy_holdings = spy.get_holdings()
            
            if spy_holdings is None:
                return False
                
            spy_sector_weights = defaultdict(float)
            for holding in spy_holdings:
                sector = holding.get('sector', 'Unknown')
                weight = holding.get('weight', 0)
                spy_sector_weights[sector] += weight
                
            # Check deviations
            max_deviation = 0.0
            for sector, weight in sector_weights.items():
                deviation = abs(weight - spy_sector_weights.get(sector, 0))
                max_deviation = max(max_deviation, deviation)
                
            return max_deviation > self.max_sector_deviation
            
        except Exception as e:
            logger.error(f"Error checking sector deviations: {str(e)}")
            return False
            
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        try:
            report = {
                'tracking_error': {
                    'current': self.tracking_error_history[-1] if self.tracking_error_history else 0.0,
                    'avg_30d': np.mean(self.tracking_error_history[-30:]) if len(self.tracking_error_history) >= 30 else 0.0,
                    'max': max(self.tracking_error_history) if self.tracking_error_history else 0.0
                },
                'sector_exposure': {},
                'liquidity_analysis': {
                    'min_score': min(self.liquidity_scores.values()) if self.liquidity_scores else 0.0,
                    'avg_score': np.mean(list(self.liquidity_scores.values())) if self.liquidity_scores else 0.0
                },
                'concentration': {
                    'current': self.portfolio_stats['concentration'][-1]['value'] if self.portfolio_stats['concentration'] else 0.0,
                    'trend': self._calculate_metric_trend('concentration', lookback=20)
                },
                'risk_events': self._summarize_risk_events()
            }
            
            # Add sector analysis
            for sector, history in self.sector_exposures.items():
                weights = [h['weight'] for h in history]
                report['sector_exposure'][sector] = {
                    'current_weight': weights[-1] if weights else 0.0,
                    'avg_weight': np.mean(weights) if weights else 0.0,
                    'max_weight': max(weights) if weights else 0.0,
                    'volatility': np.std(weights) if len(weights) > 1 else 0.0
                }
                
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return {}
            
    def _calculate_metric_trend(self, metric: str, lookback: int = 20) -> str:
        """Calculate trend direction for a metric"""
        try:
            history = self.portfolio_stats[metric]
            if len(history) < lookback:
                return "insufficient_data"
                
            recent_values = [h['value'] for h in history[-lookback:]]
            slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            if slope > 0.001:
                return "increasing"
            elif slope < -0.001:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating metric trend: {str(e)}")
            return "error"
            
    def _summarize_risk_events(self) -> List[Dict]:
        """Summarize recent risk events"""
        try:
            recent_events = self.risk_events[-10:]  # Last 10 events
            summary = []
            
            for event in recent_events:
                summary.append({
                    'date': event['date'],
                    'type': event['type'],
                    'severity': event.get('severity', 'medium'),
                    'metrics': event.get('metrics', {})
                })
                
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing risk events: {str(e)}")
            return []
            
    def print_risk_summary(self):
        """Print comprehensive risk management summary"""
        try:
            report = self.generate_risk_report()
            
            logger.info("\n=== ETF Risk Management Summary ===")
            
            # Tracking Error
            logger.info("\nTracking Error Analysis:")
            logger.info(f"Current: {report['tracking_error']['current']:.2%}")
            logger.info(f"30-day Average: {report['tracking_error']['avg_30d']:.2%}")
            logger.info(f"Maximum: {report['tracking_error']['max']:.2%}")
            
            # Sector Exposure
            logger.info("\nSector Exposure:")
            for sector, metrics in report['sector_exposure'].items():
                logger.info(f"\n{sector}:")
                logger.info(f"Current Weight: {metrics['current_weight']:.2%}")
                logger.info(f"Average Weight: {metrics['avg_weight']:.2%}")
                logger.info(f"Weight Volatility: {metrics['volatility']:.2%}")
                
            # Liquidity
            logger.info("\nLiquidity Analysis:")
            logger.info(f"Minimum Score: {report['liquidity_analysis']['min_score']:.2f}")
            logger.info(f"Average Score: {report['liquidity_analysis']['avg_score']:.2f}")
            
            # Concentration
            logger.info("\nConcentration Risk:")
            logger.info(f"Current Level: {report['concentration']['current']:.2f}")
            logger.info(f"Trend: {report['concentration']['trend']}")
            
            # Risk Events
            if report['risk_events']:
                logger.info("\nRecent Risk Events:")
                for event in report['risk_events']:
                    logger.info(f"\n{event['date']}: {event['type']} (Severity: {event['severity']})")
                    
        except Exception as e:
            logger.error(f"Error printing risk summary: {str(e)}")