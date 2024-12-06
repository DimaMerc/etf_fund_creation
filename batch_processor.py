# batch_processor.py

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from db_manager import DatabaseManager
from data_fetcher import fetch_equity_data_for_symbols, fetch_options_data
from data_preparation import EnhancedDataPipeline, ETFDataPipeline
from options_model import OptionsPricingMLP
import logging
import traceback
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, 
                 base_batch_size: int = 150, 
                 max_parallel_jobs: int = 4,
                 db_path: str = 'symbol_data.db'):
        """
        Enhanced batch processor with parallel processing capabilities
        
        Args:
            base_batch_size: Base number of symbols to process in each batch
            max_parallel_jobs: Maximum number of parallel processing jobs
            db_path: Path to SQLite database
        """
        self.base_batch_size = base_batch_size
        self.max_parallel_jobs = max_parallel_jobs
        self.db = DatabaseManager(db_path)
        self.data_pipeline = ETFDataPipeline()
        self.current_batch_id = None
        self.market_metrics = {}

    def dynamic_batch_size(self, market_volatility: float) -> int:
        """
        Dynamically adjust batch size based on market conditions
        
        Args:
            market_volatility: Current market volatility (annualized)
            
        Returns:
            Adjusted batch size
        """
        try:
            if market_volatility > 0.25:  # High volatility regime
                return max(50, self.base_batch_size // 2)
            elif market_volatility < 0.15:  # Low volatility regime
                return min(300, self.base_batch_size * 2)
            return self.base_batch_size
            
        except Exception as e:
            logger.error(f"Error in dynamic_batch_size: {str(e)}")
            return self.base_batch_size

    def process_batches(self, 
                       symbols: List[str], 
                       start_date: str, 
                       end_date: str,
                       model) -> Dict:
        """
        Process all symbols in optimized batches with parallel execution
        
        Args:
            symbols: List of symbols to process
            start_date: Start date for analysis
            end_date: End date for analysis
            model: Machine learning model for predictions
            
        Returns:
            Combined results from all batches
        """
        try:
            logger.info(f"\nProcessing {len(symbols)} symbols in batches")
            
            # Get market conditions
            market_vol = self._get_market_volatility(start_date, end_date)
            batch_size = self.dynamic_batch_size(market_vol)
            
            logger.info(f"Market volatility: {market_vol:.1%}")
            logger.info(f"Using batch size: {batch_size}")
            
            # Split symbols into batches
            symbol_batches = [
                symbols[i:i + batch_size] 
                for i in range(0, len(symbols), batch_size)
            ]
            
            all_results = []
            failed_symbols = []
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
                future_to_batch = {
                    executor.submit(
                        self.process_batch, 
                        batch_symbols,
                        start_date,
                        end_date,
                        model
                    ): batch_symbols 
                    for batch_symbols in symbol_batches
                }
                
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        result = future.result()
                        if result:
                            all_results.extend(result.get('selected_symbols', []))
                        else:
                            failed_symbols.extend(batch)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {str(e)}")
                        failed_symbols.extend(batch)
            
            # Log processing summary
            logger.info("\nBatch Processing Summary:")
            logger.info(f"Successfully processed: {len(all_results)} symbols")
            logger.info(f"Failed to process: {len(failed_symbols)} symbols")
            
            if failed_symbols:
                logger.info("Failed symbols:")
                for symbol in failed_symbols[:10]:  # Show first 10
                    logger.info(f"- {symbol}")
                if len(failed_symbols) > 10:
                    logger.info("... and more")
            
            return self._combine_batch_results(all_results)
            
        except Exception as e:
            logger.error(f"Error in process_batches: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def process_batch(self, 
                     batch_symbols: List[str], 
                     start_date: str, 
                     end_date: str,
                     model) -> Optional[Dict]:
        """
        Process a single batch of symbols
        
        Args:
            batch_symbols: List of symbols in this batch
            start_date: Start date for analysis
            end_date: End date for analysis
            model: Machine learning model for predictions
            
        Returns:
            Dictionary containing batch results
        """
        try:
            logger.info(f"\nProcessing batch of {len(batch_symbols)} symbols")
            self.current_batch_id = self.db.start_batch(start_date, end_date)
            
            # Fetch equity data
            equity_data = fetch_equity_data_for_symbols(
                batch_symbols, 
                start_date, 
                end_date
            )
            
            if not equity_data:
                logger.warning("No equity data fetched for batch")
                return None
                
            # Store price data
            self.db.store_price_data(equity_data, self.current_batch_id)
            
            # Get batch data and prepare features
            batch_data = self.db.get_batch_data(batch_symbols, start_date, end_date)
            features_df = self.data_pipeline.prepare_features(equity_data)
            
            if features_df.empty:
                logger.warning("No features generated for batch")
                return None
            
            # Make predictions
            predictions = model.predict(features_df)
            
            # Select symbols considering risk limits
            selected_symbols = self._select_symbols_with_risk_limits(
                predictions,
                batch_data,
                equity_data
            )
            
            if not selected_symbols:
                logger.warning("No symbols selected after risk filtering")
                return None
            
            # Fetch options data for selected symbols
            options_data = fetch_options_data(selected_symbols, start_date, end_date)
            if options_data:
                self.db.store_options_data(options_data)
                
            # Calculate batch score
            batch_score = self._calculate_batch_score(predictions, selected_symbols)
            
            # Update results
            self.db.update_batch_results(
                self.current_batch_id,
                selected_symbols,
                batch_score
            )
            
            return {
                'selected_symbols': selected_symbols,
                'predictions': predictions,
                'batch_score': batch_score
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _select_symbols_with_risk_limits(self, 
                                       predictions: Dict, 
                                       batch_data: pd.DataFrame,
                                       equity_data: Dict) -> List[str]:
        """
        Select symbols with enhanced risk controls
        
        Args:
            predictions: Model predictions
            batch_data: Batch historical data
            equity_data: Current equity data
            
        Returns:
            List of selected symbols
        """
        try:
            selected = []
            portfolio_vol = 0
            
            # Sort by prediction score
            sorted_symbols = sorted(
                predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for symbol, score in sorted_symbols:
                # Calculate individual stock metrics
                symbol_data = batch_data[batch_data['symbol'] == symbol]
                returns = symbol_data['return'].values
                
                # Volatility check
                symbol_vol = np.std(returns) * np.sqrt(252)
                if symbol_vol > 0.40:  # Skip highly volatile stocks
                    continue
                    
                # Momentum check
                momentum = self._calculate_momentum(equity_data[symbol])
                if momentum < -0.10:  # Skip stocks in strong downtrend
                    continue
                    
                # Liquidity check
                avg_volume = equity_data[symbol]['volume'].mean()
                if avg_volume < 100000:  # Skip illiquid stocks
                    continue
                    
                # Portfolio risk check
                new_portfolio_vol = self._calculate_portfolio_vol(
                    selected + [symbol],
                    batch_data
                )
                
                if new_portfolio_vol <= 0.25:  # Target portfolio volatility
                    selected.append(symbol)
                    portfolio_vol = new_portfolio_vol
                    
                if len(selected) >= 50:  # Maximum positions
                    break
                    
            return selected
            
        except Exception as e:
            logger.error(f"Error selecting symbols: {str(e)}")
            return []

    def _calculate_batch_score(self, 
                             predictions: Dict,
                             selected_symbols: List[str]) -> float:
        """Calculate overall batch quality score"""
        try:
            if not selected_symbols:
                return 0.0
                
            scores = [predictions[s] for s in selected_symbols]
            
            # Weighted score components
            avg_score = np.mean(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            score_std = np.std(scores)
            
            # Combined score with penalties for high dispersion
            batch_score = (
                avg_score * 0.4 +
                min_score * 0.3 +
                max_score * 0.2 -
                score_std * 0.1  # Penalty for high dispersion
            )
            
            return float(batch_score)
            
        except Exception as e:
            logger.error(f"Error calculating batch score: {str(e)}")
            return 0.0

    def _combine_batch_results(self, results: List[Dict]) -> Dict:
        """Combine results from multiple batches"""
        try:
            combined = {
                'selected_symbols': [],
                'predictions': {},
                'batch_scores': []
            }
            
            for result in results:
                if isinstance(result, dict):
                    combined['selected_symbols'].extend(
                        result.get('selected_symbols', [])
                    )
                    combined['predictions'].update(
                        result.get('predictions', {})
                    )
                    if 'batch_score' in result:
                        combined['batch_scores'].append(result['batch_score'])
            
            # Remove duplicates while preserving order
            combined['selected_symbols'] = list(dict.fromkeys(
                combined['selected_symbols']
            ))
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining batch results: {str(e)}")
            return {}
        

class ETFBatchProcessor(BatchProcessor):
    """Enhanced batch processor specialized for ETF construction"""
    
    def __init__(self, 
                 base_batch_size: int = 50,  # Smaller batches for more stable processing
                 max_parallel_jobs: int = 4,
                 db_path: str = 'etf_data.db'):
        """Initialize ETF batch processor"""
        self.base_batch_size = base_batch_size
        self.max_parallel_jobs = max_parallel_jobs
        self.db = DatabaseManager(db_path)
        self.data_pipeline = ETFDataPipeline()
        self.current_batch_id = None
        self.market_metrics = {}
        self.sector_weights = defaultdict(float)
        
        # ETF-specific constraints
        self.min_market_cap = 1e9  # $1B minimum market cap
        self.min_daily_volume = 1e6  # $1M minimum daily volume
        self.max_sector_weight = 0.25  # 25% maximum sector weight
        self.min_symbols_per_sector = 3  # Minimum symbols per sector

    def process_sp500_batches(self, 
                            symbols: List[str], 
                            start_date: str, 
                            end_date: str,
                            model,
                            sector_map: Optional[Dict] = None) -> Dict:
        """
        Process S&P 500 constituents in optimized batches
        
        Args:
            symbols: List of S&P 500 symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            model: Machine learning model for predictions
            sector_map: Optional sector mapping
        """
        try:
            logger.info(f"\nProcessing {len(symbols)} S&P 500 constituents")
            
            # Get market conditions for batch size adjustment
            market_vol = self._get_market_volatility(start_date, end_date)
            batch_size = self._get_dynamic_batch_size(market_vol)
            
            logger.info(f"Market volatility: {market_vol:.1%}")
            logger.info(f"Using batch size: {batch_size}")
            
            # Split symbols by sector if sector map available
            if sector_map:
                symbol_batches = self._create_sector_balanced_batches(
                    symbols=symbols,
                    sector_map=sector_map,
                    batch_size=batch_size
                )
            else:
                # Simple batch splitting
                symbol_batches = [
                    symbols[i:i + batch_size] 
                    for i in range(0, len(symbols), batch_size)
                ]
            
            # Process batches in parallel
            all_results = []
            failed_symbols = []
            sector_metrics = defaultdict(list)
            
            with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
                future_to_batch = {
                    executor.submit(
                        self._process_batch,
                        batch_symbols=batch,
                        start_date=start_date,
                        end_date=end_date,
                        model=model,
                        sector_map=sector_map
                    ): batch
                    for batch in symbol_batches
                }
                
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        result = future.result()
                        if result:
                            all_results.extend(result.get('selected_symbols', []))
                            
                            # Track sector metrics
                            if sector_map:
                                for symbol in result.get('selected_symbols', []):
                                    sector = sector_map.get(symbol)
                                    if sector:
                                        sector_metrics[sector].append(result['metrics'][symbol])
                        else:
                            failed_symbols.extend(batch)
                            
                    except Exception as e:
                        logger.error(f"Batch processing failed: {str(e)}")
                        failed_symbols.extend(batch)
            
            # Analyze and log results
            self._log_processing_summary(
                all_results=all_results,
                failed_symbols=failed_symbols,
                sector_metrics=sector_metrics
            )
            
            return self._combine_batch_results(
                all_results=all_results,
                sector_metrics=sector_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _create_sector_balanced_batches(self,
                                      symbols: List[str],
                                      sector_map: Dict[str, str],
                                      batch_size: int) -> List[List[str]]:
        """Create balanced batches maintaining sector representation"""
        try:
            # Group symbols by sector
            sector_symbols = defaultdict(list)
            for symbol in symbols:
                sector = sector_map.get(symbol, 'Unknown')
                sector_symbols[sector].append(symbol)
            
            # Calculate target symbols per sector per batch
            num_sectors = len(sector_symbols)
            target_per_sector = max(
                self.min_symbols_per_sector,
                batch_size // num_sectors
            )
            
            # Create balanced batches
            batches = []
            current_batch = []
            
            while any(sector_symbols.values()):
                # Take symbols from each sector
                for sector in list(sector_symbols.keys()):
                    sector_list = sector_symbols[sector]
                    symbols_to_take = min(
                        target_per_sector,
                        len(sector_list)
                    )
                    
                    if symbols_to_take > 0:
                        batch_symbols = sector_list[:symbols_to_take]
                        current_batch.extend(batch_symbols)
                        sector_symbols[sector] = sector_list[symbols_to_take:]
                    
                    # Check if batch is full
                    if len(current_batch) >= batch_size:
                        batches.append(current_batch)
                        current_batch = []
                
                # Clean up empty sectors
                sector_symbols = {k: v for k, v in sector_symbols.items() if v}
            
            # Add remaining symbols
            if current_batch:
                batches.append(current_batch)
                
            return batches
            
        except Exception as e:
            logger.error(f"Error creating balanced batches: {str(e)}")
            # Fallback to simple batching
            return [symbols[i:i + batch_size] 
                   for i in range(0, len(symbols), batch_size)]

    def _process_batch(self,
                      batch_symbols: List[str],
                      start_date: str,
                      end_date: str,
                      model,
                      sector_map: Optional[Dict] = None) -> Optional[Dict]:
        """Process a single batch of symbols"""
        try:
            logger.info(f"\nProcessing batch of {len(batch_symbols)} symbols")
            self.current_batch_id = self.db.start_batch(start_date, end_date)
            
            # Fetch equity data
            equity_data = fetch_equity_data_for_symbols(
                symbols=batch_symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            if not equity_data:
                logger.error("No equity data fetched for batch")
                return None
            
            # Apply initial filters
            filtered_symbols = self._apply_etf_filters(
                symbols=batch_symbols,
                equity_data=equity_data
            )
            
            if not filtered_symbols:
                logger.warning("No symbols passed ETF filters")
                return None
            
            # Prepare features
            features_df = self.data_pipeline.prepare_etf_features(equity_data)
            
            if features_df.empty:
                logger.error("Failed to prepare features")
                return None
            
            # Make predictions
            predictions = model.predict(features_df)
            selected_symbols = []
            metrics = {}
            
            # Select symbols ensuring sector constraints
            if sector_map:
                selected_symbols = self._select_symbols_with_sector_constraints(
                    predictions=predictions,
                    symbols=filtered_symbols,
                    sector_map=sector_map
                )
            else:
                # Simple selection based on predictions
                selected_symbols = [
                    symbol for symbol, score in 
                    sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    if score > 0.5  # Minimum score threshold
                ]
            
            # Calculate metrics for selected symbols
            for symbol in selected_symbols:
                metrics[symbol] = self._calculate_symbol_metrics(
                    symbol=symbol,
                    equity_data=equity_data[symbol],
                    prediction_score=predictions.get(symbol, 0)
                )
            
            # Store results
            self.db.store_batch_results(
                batch_id=self.current_batch_id,
                selected_symbols=selected_symbols,
                metrics=metrics
            )
            
            return {
                'selected_symbols': selected_symbols,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return None

    def _apply_etf_filters(self,
                          symbols: List[str],
                          equity_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Apply ETF-specific filters to symbols"""
        try:
            filtered_symbols = []
            
            for symbol in symbols:
                if symbol not in equity_data:
                    continue
                    
                df = equity_data[symbol]
                
                # Calculate average daily volume
                avg_volume = df['volume'].mean()
                avg_price = df['close'].mean()
                daily_volume = avg_volume * avg_price
                
                if daily_volume < self.min_daily_volume:
                    continue
                
                # Check price stability
                returns = df['close'].pct_change()
                volatility = returns.std() * np.sqrt(252)
                
                if volatility > 0.5:  # Skip highly volatile stocks
                    continue
                
                filtered_symbols.append(symbol)
            
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Error applying ETF filters: {str(e)}")
            return []

    def _select_symbols_with_sector_constraints(self,
                                              predictions: Dict[str, float],
                                              symbols: List[str],
                                              sector_map: Dict[str, str]) -> List[str]:
        """Select symbols while maintaining sector constraints"""
        try:
            selected_symbols = []
            sector_counts = defaultdict(int)
            sector_weights = defaultdict(float)
            
            # Sort symbols by prediction score
            sorted_symbols = sorted(
                [(s, predictions.get(s, 0)) for s in symbols],
                key=lambda x: x[1],
                reverse=True
            )
            
            for symbol, score in sorted_symbols:
                sector = sector_map.get(symbol, 'Unknown')
                
                # Check sector constraints
                if (sector_weights[sector] + score <= self.max_sector_weight and
                    sector_counts[sector] < 10):  # Max 10 stocks per sector
                    
                    selected_symbols.append(symbol)
                    sector_counts[sector] += 1
                    sector_weights[sector] += score
                    
            return selected_symbols
            
        except Exception as e:
            logger.error(f"Error selecting symbols with sector constraints: {str(e)}")
            return []

    def _calculate_symbol_metrics(self,
                                symbol: str,
                                equity_data: pd.DataFrame,
                                prediction_score: float) -> Dict:
        """Calculate comprehensive metrics for selected symbol"""
        try:
            returns = equity_data['close'].pct_change()
            
            metrics = {
                'prediction_score': prediction_score,
                'volatility': returns.std() * np.sqrt(252),
                'avg_volume': equity_data['volume'].mean(),
                'price': equity_data['close'].iloc[-1],
                'momentum': (equity_data['close'].iloc[-1] / 
                           equity_data['close'].iloc[-60] - 1)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating symbol metrics: {str(e)}")
            return {}

    def _combine_batch_results(self,
                             all_results: List[Dict],
                             sector_metrics: Dict[str, List]) -> Dict:
        """Combine results from all batches"""
        try:
            combined = {
                'selected_symbols': [],
                'metrics': defaultdict(dict),
                'sector_analysis': {}
            }
            
            # Combine selected symbols and metrics
            for result in all_results:
                if isinstance(result, dict):
                    combined['selected_symbols'].extend(
                        result.get('selected_symbols', [])
                    )
                    combined['metrics'].update(
                        result.get('metrics', {})
                    )
            
            # Remove duplicates while preserving order
            combined['selected_symbols'] = list(dict.fromkeys(
                combined['selected_symbols']
            ))
            
            # Add sector analysis
            if sector_metrics:
                for sector, metrics in sector_metrics.items():
                    combined['sector_analysis'][sector] = {
                        'count': len(metrics),
                        'avg_score': np.mean([m.get('prediction_score', 0) 
                                            for m in metrics]),
                        'avg_volume': np.mean([m.get('avg_volume', 0) 
                                             for m in metrics]),
                        'total_weight': sum([m.get('prediction_score', 0) 
                                           for m in metrics])
                    }
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining batch results: {str(e)}")
            return {}