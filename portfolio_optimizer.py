# portfolio_optimizer.py

# portfolio_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from config import (
    RISK_FREE_RATE, MAX_POSITION_SIZE, MIN_POSITION_SIZE,
    MAX_SECTOR_ALLOCATION, TARGET_PORTFOLIO_VOLATILITY
)

logger = logging.getLogger(__name__)

class EnhancedPortfolioOptimizer:
    """Enhanced portfolio optimizer with risk-aware optimization and sector rotation"""
    
    def __init__(self,
                 risk_free_rate: float = RISK_FREE_RATE,
                 target_volatility: float = TARGET_PORTFOLIO_VOLATILITY,
                 max_position_size: float = MAX_POSITION_SIZE,
                 min_position_size: float = MIN_POSITION_SIZE,
                 max_sector_allocation: float = MAX_SECTOR_ALLOCATION):
        """
        Initialize portfolio optimizer with enhanced parameters
        """
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_sector_allocation = max_sector_allocation
        self.optimization_history = []
        self.sector_allocations = {}

    def optimize_portfolio(self,
                        selected_symbols: List[str],
                        price_data: Dict[str, pd.DataFrame],
                        date: datetime,
                        sector_map: Optional[Dict] = None,
                        sector_constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Optimize ETF composition with sector constraints
        Args:
            sector_constraints: Dict of sector weights from SPY for benchmarking
        """
        try:
            logger.info(f"\nOptimizing ETF portfolio for {date}")
            
            # Calculate returns and covariance
            returns_data = self._prepare_returns_data(price_data, selected_symbols, date)
            if returns_data.empty:
                logger.error("No valid returns data for optimization")
                return {}
                
            # Calculate expected returns and covariance
            expected_returns = self._calculate_expected_returns(returns_data)
            covariance_matrix = self._calculate_robust_covariance(returns_data)
            
            # Setup optimization problem
            num_assets = len(selected_symbols)
            initial_weights = np.array([1.0/num_assets] * num_assets)
            
            # Create constraints including sector limits
            constraints = self._create_optimization_constraints(
                num_assets=num_assets,
                sector_map=sector_map,
                selected_symbols=selected_symbols,
                sector_constraints=sector_constraints  # Pass SPY sector weights
            )
            
            # Position size bounds
            bounds = [(self.min_position_size, self.max_position_size) 
                    for _ in range(num_assets)]
            
            try:
                # Optimize portfolio
                result = minimize(
                    fun=self._portfolio_objective,
                    x0=initial_weights,
                    args=(expected_returns, covariance_matrix),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    # Create weights dictionary
                    weights = dict(zip(selected_symbols, result.x))
                    
                    # Store optimization results
                    self._store_optimization_results(
                        weights=weights,
                        expected_returns=expected_returns,
                        covariance_matrix=covariance_matrix,
                        date=date
                    )
                    
                    # Log sector exposures with benchmark comparison
                    if sector_map and sector_constraints:
                        self._log_sector_exposures_with_benchmark(
                            weights, 
                            sector_map, 
                            sector_constraints
                        )
                    else:
                        self._log_sector_exposures(weights, sector_map)
                        
                    return weights
                else:
                    logger.error(f"Optimization failed: {result.message}")
                    return {}
                    
            except Exception as e:
                logger.error(f"Optimization error: {str(e)}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {}
    
    def _create_optimization_constraints(self,
                                    num_assets: int,
                                    sector_map: Optional[Dict] = None,
                                    selected_symbols: Optional[List[str]] = None,
                                    sector_constraints: Optional[Dict] = None) -> List[Dict]:
        """Create optimization constraints with sector limits based on SPY weights"""
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        })
        
        if sector_map and selected_symbols and sector_constraints:
            # Group symbols by sector
            sector_indices = defaultdict(list)
            for i, symbol in enumerate(selected_symbols):
                sector = sector_map.get(symbol, 'Unknown')
                sector_indices[sector].append(i)
                
            # Add sector constraints based on SPY weights
            for sector, spy_weight in sector_constraints.items():
                if sector in sector_indices:
                    indices = sector_indices[sector]
                    if indices:
                        # Allow ±5% deviation from SPY weights
                        min_weight = max(0, spy_weight - 0.05)
                        max_weight = min(1, spy_weight + 0.05)
                        
                        # Min weight constraint
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda x, idx=indices, w=min_weight: 
                                np.sum(x[idx]) - w
                        })
                        
                        # Max weight constraint
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda x, idx=indices, w=max_weight: 
                                w - np.sum(x[idx])
                        })
        
        return constraints

    def _log_sector_exposures_with_benchmark(self,
                                        weights: Dict[str, float],
                                        sector_map: Dict[str, str],
                                        sector_constraints: Dict[str, float]):
        """Log sector exposures compared to SPY benchmark"""
        sector_weights = defaultdict(float)
        for symbol, weight in weights.items():
            sector = sector_map.get(symbol, 'Unknown')
            sector_weights[sector] += weight
            
        logger.info("\nSector Exposures vs SPY:")
        for sector in sorted(set(sector_weights) | set(sector_constraints)):
            portfolio_weight = sector_weights.get(sector, 0)
            spy_weight = sector_constraints.get(sector, 0)
            difference = portfolio_weight - spy_weight
            logger.info(f"{sector:15} Portfolio: {portfolio_weight:6.2%}  "
                    f"SPY: {spy_weight:6.2%}  Diff: {difference:6.2%}")

    def _prepare_returns_data(self, 
                            price_data: Dict[str, pd.DataFrame],
                            selected_symbols: List[str],
                            date: datetime,
                            lookback_days: int = 252) -> pd.DataFrame:
        """Prepare returns data for optimization"""
        try:
            all_returns = pd.DataFrame()
            date = pd.to_datetime(date)
            
            for symbol in selected_symbols:
                if symbol not in price_data:
                    continue
                    
                df = price_data[symbol].copy()
                
                # Convert date if needed
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                    
                # Handle timezones
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                
                # Filter data and calculate returns
                mask = (df['date'] <= date) & (df['date'] > date - pd.Timedelta(days=lookback_days))
                if mask.sum() < lookback_days // 2:  # Require at least half the history
                    continue
                    
                returns = df.loc[mask, 'close'].pct_change().dropna()
                if not returns.empty:
                    all_returns[symbol] = returns
                    
            return all_returns
            
        except Exception as e:
            logger.error(f"Error preparing returns data: {str(e)}")
            return pd.DataFrame()

    def _calculate_expected_returns(self, 
                                  returns_data: pd.DataFrame,
                                  lookback_days: int = 252) -> pd.Series:
        """Calculate expected returns using multiple factors"""
        try:
            expected_returns = pd.Series(index=returns_data.columns)
            
            # Historical mean return (annualized)
            historical_return = returns_data.mean() * 252
            
            # Momentum factor (3-month and 6-month)
            returns_3m = (1 + returns_data).tail(63).prod() - 1
            returns_6m = (1 + returns_data).tail(126).prod() - 1
            
            # Volatility adjustment
            vols = returns_data.std() * np.sqrt(252)
            vol_factor = 1 - (vols - self.target_volatility) * 2
            
            # Combine factors with weights
            expected_returns = (
                historical_return * 0.3 +  # Historical return
                returns_3m * 0.3 +         # 3-month momentum
                returns_6m * 0.2           # 6-month momentum
            ) * vol_factor                 # Volatility adjustment
            
            return expected_returns
            
        except Exception as e:
            logger.error(f"Error calculating expected returns: {str(e)}")
            return pd.Series()
        
    def _calculate_robust_covariance(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate robust covariance matrix with shrinkage"""
        try:
            # Sample covariance (annualized)
            sample_cov = returns_data.cov() * 252
            
            # Create shrinkage target (diagonal matrix)
            diagonal = np.diag(np.diag(sample_cov))
            
            # Apply shrinkage
            shrunk_cov = (
                self.covariance_shrinkage * diagonal + 
                (1 - self.covariance_shrinkage) * sample_cov
            )
            
            # Ensure positive definite
            min_eigenval = np.linalg.eigvals(shrunk_cov).min()
            if min_eigenval < 0:
                shrunk_cov += (-min_eigenval + 1e-8) * np.eye(len(returns_data.columns))
                
            return shrunk_cov
            
        except Exception as e:
            logger.error(f"Error calculating covariance: {str(e)}")
            return pd.DataFrame()

    def _create_optimization_constraints(self,
                                      num_assets: int,
                                      sector_map: Optional[Dict],
                                      selected_symbols: List[str]) -> List[Dict]:
        """Create optimization constraints including sector limits"""
        try:
            constraints = []
            
            # Full investment constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
            
            # Sector constraints if sector mapping available
            if sector_map:
                sectors = defaultdict(list)
                for i, symbol in enumerate(selected_symbols):
                    if symbol in sector_map:
                        sector = sector_map[symbol]
                        sectors[sector].append(i)
                
                # Add constraint for each sector
                for sector, indices in sectors.items():
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=indices: 
                            self.max_sector_allocation - np.sum(x[idx])
                    })
            
            return constraints
            
        except Exception as e:
            logger.error(f"Error creating constraints: {str(e)}")
            return [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            

    def _portfolio_objective(self,
                           weights: np.ndarray,
                           expected_returns: pd.Series,
                           covariance_matrix: pd.DataFrame) -> float:
        """Portfolio objective function (maximize Sharpe with penalties)"""
        try:
            # Calculate basic portfolio metrics
            portfolio_return = np.sum(expected_returns * weights) * 252
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(covariance_matrix, weights))
            ) * np.sqrt(252)
            
            if portfolio_vol == 0:
                return 1e6  # Penalty for zero volatility
                
            # Calculate Sharpe ratio
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Concentration penalty
            concentration = np.sum(weights * weights)
            concentration_penalty = 0.5 * concentration
            
            # Volatility deviation penalty
            vol_deviation = abs(portfolio_vol - self.target_volatility)
            vol_penalty = 0.5 * vol_deviation
            
            # Return negative objective (for minimization)
            return -(sharpe - concentration_penalty - vol_penalty)
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return 1e6
        
    def _get_equal_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Get equal weight portfolio as fallback"""
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}

    def _log_optimization_results(self,
                                weights: Dict[str, float],
                                expected_returns: pd.Series,
                                covariance_matrix: pd.DataFrame,
                                sector_map: Optional[Dict] = None):
        """Log optimization results and statistics"""
        try:
            logger.info("\nOptimization Results:")
            
            # Portfolio statistics
            weights_array = np.array(list(weights.values()))
            symbols = list(weights.keys())
            
            port_return = np.sum(expected_returns * weights_array) * 252
            port_vol = np.sqrt(
                np.dot(weights_array.T, np.dot(covariance_matrix, weights_array))
            ) * np.sqrt(252)
            
            logger.info(f"Expected Return: {port_return:.2%}")
            logger.info(f"Expected Volatility: {port_vol:.2%}")
            logger.info(f"Sharpe Ratio: {(port_return - self.risk_free_rate) / port_vol:.2f}")
            
            # Position sizes
            logger.info("\nPosition Sizes:")
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{symbol}: {weight:.2%}")
            
            # Sector exposures
            if sector_map:
                sector_weights = defaultdict(float)
                for symbol, weight in weights.items():
                    sector = sector_map.get(symbol, 'Unknown')
                    sector_weights[sector] += weight
                    
                logger.info("\nSector Allocations:")
                for sector, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"{sector}: {weight:.2%}")
            
            # Store results in history
            self.optimization_history.append({
                'date': datetime.now(),
                'num_assets': len(weights),
                'expected_return': port_return,
                'volatility': port_vol,
                'max_position': max(weights.values()),
                'min_position': min(weights.values())
            })
            
        except Exception as e:
            logger.error(f"Error logging optimization results: {str(e)}")

    def _store_optimization_results(self,
                                  weights: Dict[str, float],
                                  expected_returns: pd.Series,
                                  covariance_matrix: pd.DataFrame,
                                  date: datetime):
        """Store optimization results for analysis"""
        try:
            # Calculate portfolio metrics
            weights_array = np.array(list(weights.values()))
            portfolio_return = np.sum(expected_returns * weights_array) * 252
            portfolio_vol = np.sqrt(
                np.dot(weights_array.T, np.dot(covariance_matrix, weights_array))
            ) * np.sqrt(252)
            
            # Store results
            self.optimization_history.append({
                'date': date,
                'num_positions': len(weights),
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_vol,
                'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_vol,
                'max_position': max(weights.values()),
                'min_position': min(weights.values()),
                'concentration': sum(w * w for w in weights.values())
            })
            
        except Exception as e:
            logger.error(f"Error storing optimization results: {str(e)}")
    def print_optimization_summary(self):
        """Print comprehensive optimization summary"""
        try:
            if not self.optimization_history:
                logger.info("No optimization history available")
                return
                
            logger.info("\n=== ETF Optimization History ===")
            
            # Convert history to DataFrame for analysis
            history_df = pd.DataFrame(self.optimization_history)
            
            # Basic statistics
            logger.info("\nOptimization Statistics:")
            logger.info(f"Total optimizations: {len(history_df)}")
            logger.info(f"Average positions: {history_df['num_positions'].mean():.1f}")
            logger.info(f"Average expected return: {history_df['expected_return'].mean():.2%}")
            logger.info(f"Average volatility: {history_df['expected_volatility'].mean():.2%}")
            logger.info(f"Average Sharpe ratio: {history_df['sharpe_ratio'].mean():.2f}")
            
            # Position size analysis
            logger.info("\nPosition Size Analysis:")
            logger.info(f"Average maximum position: {history_df['max_position'].mean():.2%}")
            logger.info(f"Average minimum position: {history_df['min_position'].mean():.2%}")
            logger.info(f"Average concentration (HHI): {history_df['concentration'].mean():.3f}")
            
            # Recent trend
            logger.info("\nRecent Optimizations:")
            recent = history_df.tail(5)
            for _, row in recent.iterrows():
                logger.info(f"\nDate: {row['date']}")
                logger.info(f"Expected Return: {row['expected_return']:.2%}")
                logger.info(f"Expected Volatility: {row['expected_volatility']:.2%}")
                logger.info(f"Sharpe Ratio: {row['sharpe_ratio']:.2f}")
                logger.info(f"Number of Positions: {row['num_positions']}")
                
        except Exception as e:
            logger.error(f"Error printing optimization summary: {str(e)}")

    def get_optimization_metrics(self) -> Dict:
        """Get optimization metrics for analysis"""
        try:
            if not self.optimization_history:
                return {}
                
            history_df = pd.DataFrame(self.optimization_history)
            
            metrics = {
                'avg_return': history_df['expected_return'].mean(),
                'avg_volatility': history_df['expected_volatility'].mean(),
                'avg_sharpe': history_df['sharpe_ratio'].mean(),
                'avg_positions': history_df['num_positions'].mean(),
                'avg_max_position': history_df['max_position'].mean(),
                'avg_concentration': history_df['concentration'].mean(),
                'optimization_count': len(history_df)
            }
            
            # Calculate stability metrics
            metrics.update({
                'return_stability': history_df['expected_return'].std(),
                'volatility_stability': history_df['expected_volatility'].std(),
                'position_count_stability': history_df['num_positions'].std()
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting optimization metrics: {str(e)}")
            return {}

    def _log_sector_exposures(self, weights: Dict[str, float], sector_map: Dict[str, str]):
        """Log sector exposures for analysis"""
        try:
            sector_exposures = defaultdict(float)
            for symbol, weight in weights.items():
                if symbol in sector_map:
                    sector = sector_map[symbol]
                    sector_exposures[sector] += weight
            
            logger.info("\nSector Exposures:")
            for sector, exposure in sorted(sector_exposures.items()):
                logger.info(f"{sector}: {exposure:.1%}")
                if exposure > self.max_sector_allocation:
                    logger.warning(f"WARNING: {sector} exposure ({exposure:.1%}) "
                                 f"exceeds limit ({self.max_sector_allocation:.1%})")
                    
        except Exception as e:
            logger.error(f"Error logging sector exposures: {str(e)}")

    def print_optimization_summary(self):
        """Print summary of optimization history"""
        try:
            if not self.optimization_history:
                logger.info("No optimization history available")
                return
                
            logger.info("\n=== Optimization History Summary ===")
            
            # Convert history to DataFrame
            history_df = pd.DataFrame(self.optimization_history)
            
            # Basic statistics
            logger.info("\nOptimization Statistics:")
            logger.info(f"Total optimizations: {len(history_df)}")
            logger.info(f"Average assets: {history_df['num_assets'].mean():.1f}")
            logger.info(f"Average expected return: {history_df['expected_return'].mean():.2%}")
            logger.info(f"Average volatility: {history_df['volatility'].mean():.2%}")
            
            # Position size analysis
            logger.info("\nPosition Size Analysis:")
            logger.info(f"Average maximum position: {history_df['max_position'].mean():.2%}")
            logger.info(f"Average minimum position: {history_df['min_position'].mean():.2%}")
            
            # Recent trend
            recent = history_df.tail(5)
            logger.info("\nRecent Optimizations:")
            for _, row in recent.iterrows():
                logger.info(f"Date: {row['date']}")
                logger.info(f"Expected Return: {row['expected_return']:.2%}")
                logger.info(f"Volatility: {row['volatility']:.2%}")
                logger.info("---")
                
        except Exception as e:
            logger.error(f"Error printing optimization summary: {str(e)}")

    def validate_portfolio(self, 
                         weights: Dict[str, float],
                         sector_map: Optional[Dict[str, str]] = None) -> bool:
        """Validate portfolio against ETF constraints"""
        try:
            # Check basic constraints
            if not weights:
                return False
                
            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 1.0, rtol=1e-5):
                logger.error(f"Total weight {total_weight:.4f} != 1.0")
                return False
                
            # Check position limits
            for symbol, weight in weights.items():
                if weight > self.max_position_size:
                    logger.error(f"Position {symbol} ({weight:.2%}) exceeds maximum {self.max_position_size:.2%}")
                    return False
                if weight < self.min_position_size:
                    logger.error(f"Position {symbol} ({weight:.2%}) below minimum {self.min_position_size:.2%}")
                    return False
                    
            # Check sector constraints if mapping available
            if sector_map:
                sector_weights = defaultdict(float)
                for symbol, weight in weights.items():
                    if symbol in sector_map:
                        sector = sector_map[symbol]
                        sector_weights[sector] += weight
                        
                for sector, weight in sector_weights.items():
                    if weight > self.max_sector_allocation:
                        logger.error(f"Sector {sector} ({weight:.2%}) exceeds maximum {self.max_sector_allocation:.2%}")
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error validating portfolio: {str(e)}")
            return False