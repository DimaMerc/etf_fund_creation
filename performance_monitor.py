
# performance_monitor.py 

# performance_monitor.py - Part 1: Core Functionality

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from config import RISK_FREE_RATE

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Enhanced performance monitoring system for ETF tracking"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.alerts = []
        self.risk_events = []
        self.benchmark_data = None
        self.daily_returns = []
        self.position_history = []
        self.tracking_error_history = []
        
        # Performance tracking
        self.peak_value = None
        self.drawdown_periods = []
        self.current_drawdown = None
        
        # Sector tracking
        self.sector_performance = defaultdict(list)
        self.sector_allocations = defaultdict(list)

    def update_metrics(self, 
                      date: datetime,
                      portfolio_value: float,
                      holdings: Dict,
                      prices: Dict[str, float],
                      benchmark_value: Optional[float] = None,
                      sector_map: Optional[Dict] = None) -> Dict:
        """
        Update all performance metrics
        
        Args:
            date: Current date
            portfolio_value: Current portfolio value
            holdings: Current holdings
            prices: Current prices
            benchmark_value: Optional benchmark value
            sector_map: Optional sector mapping
            
        Returns:
            Dictionary of current metrics
        """
        try:
            # Track peak value
            if self.peak_value is None or portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
                
            # Calculate basic metrics
            metrics = self._calculate_basic_metrics(
                date=date,
                portfolio_value=portfolio_value,
                holdings=holdings,
                prices=prices
            )
            
            # Add benchmark comparison if available
            if benchmark_value is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(
                    portfolio_value=portfolio_value,
                    benchmark_value=benchmark_value
                )
                metrics.update(benchmark_metrics)
                
            # Add sector analysis if mapping available
            if sector_map is not None:
                sector_metrics = self._calculate_sector_metrics(
                    holdings=holdings,
                    prices=prices,
                    sector_map=sector_map
                )
                metrics.update(sector_metrics)
                
            # Store metrics
            for key, value in metrics.items():
                self.metrics_history[key].append({
                    'date': date,
                    'value': value
                })
                
            # Check for alerts
            self._check_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            return {}
            
    def _calculate_basic_metrics(self,
                               date: datetime,
                               portfolio_value: float,
                               holdings: Dict,
                               prices: Dict[str, float]) -> Dict:
        """Calculate basic performance metrics"""
        try:
            metrics = {}
            
            # Returns
            if self.daily_returns:
                prev_value = self.daily_returns[-1]['value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append({
                    'date': date,
                    'value': portfolio_value,
                    'return': daily_return
                })
                
                # Calculate return metrics
                returns = [r['return'] for r in self.daily_returns]
                metrics['daily_return'] = daily_return
                metrics['cumulative_return'] = (portfolio_value / self.daily_returns[0]['value']) - 1
                metrics['volatility'] = np.std(returns) * np.sqrt(252)
                metrics['sharpe_ratio'] = (np.mean(returns) * 252 - RISK_FREE_RATE) / (np.std(returns) * np.sqrt(252))
            else:
                self.daily_returns.append({
                    'date': date,
                    'value': portfolio_value,
                    'return': 0.0
                })
                
            # Drawdown metrics
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
            metrics['current_drawdown'] = drawdown
            
            if self.current_drawdown is None and drawdown > 0:
                self.current_drawdown = {
                    'start_date': date,
                    'peak_value': self.peak_value,
                    'max_drawdown': drawdown
                }
            elif self.current_drawdown is not None:
                if drawdown > self.current_drawdown['max_drawdown']:
                    self.current_drawdown['max_drawdown'] = drawdown
                elif drawdown == 0:
                    self.current_drawdown['end_date'] = date
                    self.drawdown_periods.append(self.current_drawdown)
                    self.current_drawdown = None
                    
            # Position metrics
            position_values = {
                symbol: shares * prices.get(symbol, 0)
                for symbol, shares in holdings.items()
            }
            
            metrics['num_positions'] = len(position_values)
            if position_values:
                metrics['max_position'] = max(position_values.values()) / portfolio_value
                metrics['min_position'] = min(position_values.values()) / portfolio_value
                metrics['avg_position'] = np.mean(list(position_values.values())) / portfolio_value
                
            # Portfolio concentration (Herfindahl Index)
            if position_values:
                weights = [v/portfolio_value for v in position_values.values()]
                metrics['concentration'] = sum(w*w for w in weights)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            return {}

    def _calculate_benchmark_metrics(self,
                                  portfolio_value: float,
                                  benchmark_value: float) -> Dict:
        """Calculate benchmark relative metrics"""
        try:
            metrics = {}
            
            # Calculate tracking error if enough history
            if len(self.daily_returns) > 1:
                portfolio_returns = [r['return'] for r in self.daily_returns[1:]]
                benchmark_returns = self.tracking_error_history
                
                if len(benchmark_returns) == len(portfolio_returns):
                    tracking_diff = np.array(portfolio_returns) - np.array(benchmark_returns)
                    metrics['tracking_error'] = np.std(tracking_diff) * np.sqrt(252)
                    metrics['information_ratio'] = np.mean(tracking_diff) / np.std(tracking_diff) * np.sqrt(252)
            
            # Active return
            metrics['active_return'] = (portfolio_value / self.daily_returns[0]['value']) - (benchmark_value / benchmark_value)
            
            self.tracking_error_history.append(
                (benchmark_value - self.tracking_error_history[-1]['value']) / self.tracking_error_history[-1]['value']
                if self.tracking_error_history else 0.0
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {str(e)}")
            return {}
    
    # performance_monitor.py - Part 2: Sector and Risk Analysis

    def _calculate_sector_metrics(self,
                                holdings: Dict,
                                prices: Dict[str, float],
                                sector_map: Dict[str, str]) -> Dict:
        """Calculate sector-level performance metrics"""
        try:
            metrics = {}
            sector_values = defaultdict(float)
            total_value = 0
            
            # Calculate sector values
            for symbol, shares in holdings.items():
                if symbol in sector_map:
                    sector = sector_map[symbol]
                    value = shares * prices.get(symbol, 0)
                    sector_values[sector] += value
                    total_value += value
                    
            # Calculate sector metrics
            if total_value > 0:
                sector_weights = {
                    sector: value/total_value 
                    for sector, value in sector_values.items()
                }
                
                metrics['max_sector'] = max(sector_weights.values())
                metrics['min_sector'] = min(sector_weights.values())
                metrics['num_sectors'] = len(sector_weights)
                
                # Store sector performance
                for sector, weight in sector_weights.items():
                    self.sector_allocations[sector].append(weight)
                    
                    # Calculate sector return if history exists
                    if sector in self.sector_performance and self.sector_performance[sector]:
                        prev_value = self.sector_performance[sector][-1]['value']
                        sector_return = (sector_values[sector] - prev_value) / prev_value
                    else:
                        sector_return = 0.0
                        
                    self.sector_performance[sector].append({
                        'value': sector_values[sector],
                        'weight': weight,
                        'return': sector_return
                    })
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating sector metrics: {str(e)}")
            return {}

    def _check_alerts(self, metrics: Dict):
        """Check for alert conditions"""
        try:
            # Drawdown alert
            if metrics.get('current_drawdown', 0) > 0.10:
                self.alerts.append({
                    'date': datetime.now(),
                    'type': 'DRAWDOWN',
                    'value': metrics['current_drawdown'],
                    'threshold': 0.10
                })
                
            # Volatility alert
            if metrics.get('volatility', 0) > 0.25:
                self.alerts.append({
                    'date': datetime.now(),
                    'type': 'VOLATILITY',
                    'value': metrics['volatility'],
                    'threshold': 0.25
                })
                
            # Tracking error alert
            if metrics.get('tracking_error', 0) > 0.10:
                self.alerts.append({
                    'date': datetime.now(),
                    'type': 'TRACKING_ERROR',
                    'value': metrics['tracking_error'],
                    'threshold': 0.10
                })
                
            # Position concentration alert
            if metrics.get('max_position', 0) > 0.10:
                self.alerts.append({
                    'date': datetime.now(),
                    'type': 'POSITION_SIZE',
                    'value': metrics['max_position'],
                    'threshold': 0.10
                })
                
            # Sector concentration alert
            if metrics.get('max_sector', 0) > 0.30:
                self.alerts.append({
                    'date': datetime.now(),
                    'type': 'SECTOR_CONCENTRATION',
                    'value': metrics['max_sector'],
                    'threshold': 0.30
                })
                
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")

    def record_risk_event(self,
                         date: datetime,
                         event_type: str,
                         details: Dict):
        """Record risk management event"""
        try:
            self.risk_events.append({
                'date': date,
                'type': event_type,
                'details': details
            })
            
        except Exception as e:
            logger.error(f"Error recording risk event: {str(e)}")

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get all metrics history as a DataFrame"""
        try:
            data = []
            for metric, history in self.metrics_history.items():
                for entry in history:
                    data.append({
                        'date': entry['date'],
                        'metric': metric,
                        'value': entry['value']
                    })
            
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            return df.pivot(index='date', columns='metric', values='value')
            
        except Exception as e:
            logger.error(f"Error creating metrics DataFrame: {str(e)}")
            return pd.DataFrame()

    def get_sector_performance(self) -> pd.DataFrame:
        """Get sector performance history as a DataFrame"""
        try:
            data = []
            for sector, history in self.sector_performance.items():
                for entry in history:
                    data.append({
                        'sector': sector,
                        'value': entry['value'],
                        'weight': entry['weight'],
                        'return': entry['return']
                    })
                    
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error creating sector performance DataFrame: {str(e)}")
            return pd.DataFrame()

    def get_risk_events_summary(self) -> pd.DataFrame:
        """Get summary of risk events"""
        try:
            if not self.risk_events:
                return pd.DataFrame()
                
            data = []
            for event in self.risk_events:
                event_data = {
                    'date': event['date'],
                    'type': event['type']
                }
                event_data.update(event['details'])
                data.append(event_data)
                
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error creating risk events summary: {str(e)}")
            return pd.DataFrame()

    def get_alerts_summary(self) -> pd.DataFrame:
        """Get summary of alerts"""
        try:
            if not self.alerts:
                return pd.DataFrame()
                
            return pd.DataFrame(self.alerts)
            
        except Exception as e:
            logger.error(f"Error creating alerts summary: {str(e)}")
            return pd.DataFrame()
        
    # performance_monitor.py - Part 3: Reporting and Analysis

    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        try:
            logger.info("\n=== Performance Summary ===")
            
            # Return metrics
            if self.daily_returns:
                total_return = (self.daily_returns[-1]['value'] / self.daily_returns[0]['value']) - 1
                returns = [r['return'] for r in self.daily_returns[1:]]
                volatility = np.std(returns) * np.sqrt(252)
                sharpe = (np.mean(returns) * 252 - RISK_FREE_RATE) / (volatility)
                
                logger.info("\nReturn Metrics:")
                logger.info(f"Total Return: {total_return:.2%}")
                logger.info(f"Annualized Volatility: {volatility:.2%}")
                logger.info(f"Sharpe Ratio: {sharpe:.2f}")
                
                # Rolling metrics
                if len(returns) >= 20:
                    rolling_vol = np.std(returns[-20:]) * np.sqrt(252)
                    rolling_sharpe = (np.mean(returns[-20:]) * 252 - RISK_FREE_RATE) / rolling_vol
                    logger.info(f"\n20-day Rolling Volatility: {rolling_vol:.2%}")
                    logger.info(f"20-day Rolling Sharpe: {rolling_sharpe:.2f}")
            
            # Drawdown analysis
            if self.drawdown_periods:
                max_dd = max(period['max_drawdown'] for period in self.drawdown_periods)
                avg_dd = np.mean([period['max_drawdown'] for period in self.drawdown_periods])
                
                logger.info("\nDrawdown Analysis:")
                logger.info(f"Maximum Drawdown: {max_dd:.2%}")
                logger.info(f"Average Drawdown: {avg_dd:.2%}")
                logger.info(f"Number of Drawdown Periods: {len(self.drawdown_periods)}")
                
                # Worst drawdown details
                worst_dd = max(self.drawdown_periods, key=lambda x: x['max_drawdown'])
                logger.info("\nWorst Drawdown Period:")
                logger.info(f"Start Date: {worst_dd['start_date']}")
                logger.info(f"End Date: {worst_dd['end_date']}")
                logger.info(f"Duration: {(worst_dd['end_date'] - worst_dd['start_date']).days} days")
                logger.info(f"Magnitude: {worst_dd['max_drawdown']:.2%}")
            
            # Benchmark comparison
            if self.tracking_error_history:
                tracking_error = np.std(self.tracking_error_history) * np.sqrt(252)
                info_ratio = (np.mean(self.tracking_error_history) / 
                            np.std(self.tracking_error_history) * np.sqrt(252))
                
                logger.info("\nBenchmark Comparison:")
                logger.info(f"Tracking Error: {tracking_error:.2%}")
                logger.info(f"Information Ratio: {info_ratio:.2f}")
            
            # Sector analysis
            if self.sector_performance:
                logger.info("\nSector Performance:")
                for sector, history in self.sector_performance.items():
                    sector_return = ((history[-1]['value'] / history[0]['value']) - 1 
                                  if len(history) > 1 else 0)
                    avg_weight = np.mean([h['weight'] for h in history])
                    
                    logger.info(f"\n{sector}:")
                    logger.info(f"Return: {sector_return:.2%}")
                    logger.info(f"Average Weight: {avg_weight:.2%}")
                    logger.info(f"Current Weight: {history[-1]['weight']:.2%}")
            
            # Risk events summary
            if self.risk_events:
                logger.info("\nRisk Events Summary:")
                event_types = pd.Series([e['type'] for e in self.risk_events])
                type_counts = event_types.value_counts()
                
                for event_type, count in type_counts.items():
                    logger.info(f"{event_type}: {count} occurrences")
            
            # Alert summary
            if self.alerts:
                logger.info("\nAlert Summary:")
                alert_types = pd.Series([a['type'] for a in self.alerts])
                alert_counts = alert_types.value_counts()
                
                for alert_type, count in alert_counts.items():
                    logger.info(f"{alert_type}: {count} alerts")
                    
        except Exception as e:
            logger.error(f"Error printing performance summary: {str(e)}")

    def analyze_drawdowns(self) -> pd.DataFrame:
        """Analyze drawdown characteristics"""
        try:
            if not self.drawdown_periods:
                return pd.DataFrame()
                
            data = []
            for period in self.drawdown_periods:
                duration = (period['end_date'] - period['start_date']).days
                recovery_time = None
                
                # Find recovery if available
                for i, ret in enumerate(self.daily_returns):
                    if ret['date'] > period['end_date']:
                        if ret['value'] >= period['peak_value']:
                            recovery_time = (ret['date'] - period['end_date']).days
                            break
                
                data.append({
                    'start_date': period['start_date'],
                    'end_date': period['end_date'],
                    'duration': duration,
                    'max_drawdown': period['max_drawdown'],
                    'recovery_time': recovery_time
                })
                
            df = pd.DataFrame(data)
            
            # Add summary statistics
            if not df.empty:
                summary = pd.DataFrame({
                    'avg_duration': df['duration'].mean(),
                    'avg_drawdown': df['max_drawdown'].mean(),
                    'avg_recovery': df['recovery_time'].mean(),
                    'max_drawdown': df['max_drawdown'].max(),
                    'max_duration': df['duration'].max()
                }, index=[0])
                
                return pd.concat([df, summary], keys=['periods', 'summary'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing drawdowns: {str(e)}")
            return pd.DataFrame()

    def analyze_sector_rotation(self) -> pd.DataFrame:
        """Analyze sector rotation patterns"""
        try:
            if not self.sector_allocations:
                return pd.DataFrame()
                
            data = []
            for sector, weights in self.sector_allocations.items():
                avg_weight = np.mean(weights)
                min_weight = min(weights)
                max_weight = max(weights)
                weight_volatility = np.std(weights)
                
                data.append({
                    'sector': sector,
                    'avg_weight': avg_weight,
                    'min_weight': min_weight,
                    'max_weight': max_weight,
                    'weight_range': max_weight - min_weight,
                    'weight_volatility': weight_volatility
                })
                
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {str(e)}")
            return pd.DataFrame()

    def calculate_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics"""
        try:
            metrics = {}
            
            if len(self.daily_returns) > 1:
                returns = [r['return'] for r in self.daily_returns[1:]]
                
                # Basic risk metrics
                metrics['volatility'] = np.std(returns) * np.sqrt(252)
                metrics['downside_vol'] = np.std([r for r in returns if r < 0]) * np.sqrt(252)
                metrics['skewness'] = pd.Series(returns).skew()
                metrics['kurtosis'] = pd.Series(returns).kurtosis()
                
                # VaR and CVaR
                returns_sorted = sorted(returns)
                var_95 = np.percentile(returns_sorted, 5)
                cvar_95 = np.mean([r for r in returns_sorted if r <= var_95])
                
                metrics['var_95'] = var_95
                metrics['cvar_95'] = cvar_95
                
                # Maximum drawdown
                if self.drawdown_periods:
                    metrics['max_drawdown'] = max(p['max_drawdown'] for p in self.drawdown_periods)
                
                # Rolling metrics (if enough history)
                if len(returns) >= 20:
                    rolling_returns = returns[-20:]
                    metrics['rolling_vol'] = np.std(rolling_returns) * np.sqrt(252)
                    metrics['rolling_var_95'] = np.percentile(rolling_returns, 5)
            
            # Position and sector risk
            recent_metrics = self.metrics_history.get('max_position', [])
            if recent_metrics:
                metrics['max_position'] = recent_metrics[-1]['value']
                
            recent_sector = self.metrics_history.get('max_sector', [])
            if recent_sector:
                metrics['max_sector'] = recent_sector[-1]['value']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}