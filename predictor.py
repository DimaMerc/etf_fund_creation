

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from data_preparation import ETFDataPipeline
from config import FEATURES, TOP_N, MIN_POSITIONS, MAX_POSITIONS
import traceback
from collections import defaultdict
from column_utils import standardize_dataframe_columns, get_column_name
from data_utils import concatenate_metrics_arrays
import tensorflow as tf
import os

logger = logging.getLogger(__name__)

class EnhancedPredictor:
    """Enhanced prediction system for ETF constituent selection"""
    
    def __init__(self, data_pipeline: Optional[ETFDataPipeline] = None):
        self.data_pipeline = data_pipeline or ETFDataPipeline()
        self.prediction_history = []
        self.model_metrics = {}
        self.sector_predictions = {}
        
    def prepare_prediction_data(self,
                              date: datetime,
                              price_data: Dict[str, pd.DataFrame],
                              features: List[str],
                              sequence_length: int) -> Tuple[Optional[np.ndarray], 
                                                           Optional[np.ndarray],
                                                           Optional[np.ndarray],
                                                           List[str]]:
        """
        Prepare data for prediction with enhanced validation
        
        Args:
            date: Current date
            price_data: Historical price data
            features: List of features to use
            sequence_length: Length of sequence for LSTM
            
        Returns:
            Tuple of (X_lstm, node_indices, graph_info, symbols)
        """
        try:
            logger.info(f"\nPreparing prediction data for {date}")
            all_sequences = []
            node_indices_list = []
            symbols = []
            
            symbol_list = list(price_data.keys())
            symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbol_list)}
            num_nodes = len(symbol_list)
            
            logger.info(f"Processing {num_nodes} symbols")
            
            # Build node features and adjacency matrix
            node_feature_dim = 64
            node_features = np.random.normal(size=(num_nodes, node_feature_dim))
            
            # Calculate required history
            required_days = sequence_length * 3  # Add buffer for feature calculation
            history_start = pd.to_datetime(date).tz_localize(None) - pd.Timedelta(days=required_days)
            
            for symbol in symbol_list:
                try:
                    equity_df = price_data[symbol]

                     # Convert dates to timezone-naive if needed
                    if 'date' in equity_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(equity_df['date']):
                            equity_df['date'] = equity_df['date'].dt.tz_localize(None)
                        else:
                            equity_df['date'] = pd.to_datetime(equity_df['date']).tz_localize(None)
                
                    
                    df, col_map = standardize_dataframe_columns(df)
                    if df is None:
                        continue
                        
                    date_col = col_map['date']
                    mask = (df[date_col] >= history_start) & (df[date_col] <= date)
                    df_filtered = df.loc[mask]
                    
                    if len(df_filtered) < sequence_length:
                        continue
                        
                    feature_data = self._calculate_features(df_filtered, col_map)
                    if feature_data is not None:
                        all_sequences.append(feature_data)
                    
                    # Verify features
                    if not all(f in df.columns for f in features):
                        missing = [f for f in features if f not in df.columns]
                        logger.debug(f"Missing features for {symbol}: {missing}")
                        continue
                    
                    # Get sequence
                    feature_data = df[features].iloc[-sequence_length:].values
                    if len(feature_data) < sequence_length:
                        continue
                    
                    # Add to sequences
                    X_seq = np.expand_dims(feature_data, axis=0)
                    all_sequences.append(X_seq)
                    symbols.append(symbol)
                    node_indices_list.append(symbol_to_index[symbol])
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            if not all_sequences:
                logger.warning("No valid sequences generated")
                return None, None, None, []
            
            # Create final data
            X_lstm = concatenate_metrics_arrays(all_sequences)
            if X_lstm is None:
                logger.error("Failed to concatenate LSTM sequences.")

            node_indices = np.array(node_indices_list)
            edges = self._build_graph_edges(len(symbols))
            graph_info = (node_features, edges)
            
            logger.info(f"Successfully prepared data:")
            logger.info(f"X_lstm shape: {X_lstm.shape}")
            logger.info(f"Symbols processed: {len(symbols)}")
            
            return X_lstm, node_indices, graph_info, symbols
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None, None, []

    def _build_graph_edges(self, num_nodes: int) -> np.ndarray:
        """Build graph edges for fully connected graph"""
        try:
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append([i, j])
            return np.array(edges).T
            
        except Exception as e:
            logger.error(f"Error building graph edges: {str(e)}")
            return np.array([])

    def predict_with_confidence(self,
                              model,
                              X_lstm: np.ndarray,
                              node_indices: np.ndarray,
                              symbols: List[str]) -> Dict[str, Dict]:
        """
        Make predictions with confidence scores
        
        Args:
            model: Trained model
            X_lstm: LSTM input data
            node_indices: Node indices for graph
            symbols: List of symbols
            
        Returns:
            Dictionary of predictions with confidence scores
        """
        try:
               # Ensure proper shapes
            if len(X_lstm.shape) != 3:
                logger.error(f"Invalid X_lstm shape: {X_lstm.shape}")
                return {}
                
            # Prepare input in correct format
            inputs = {
                'lstm_input': tf.convert_to_tensor(X_lstm, dtype=tf.float32),
                'node_indices': tf.convert_to_tensor(node_indices, dtype=tf.int32)
            }
            
            # Make predictions
            raw_predictions = model.predict(inputs, batch_size=32)
            
            # Calculate prediction statistics
            mean_pred = np.mean(raw_predictions)
            std_pred = np.std(raw_predictions)
            
            # Calculate confidence scores
            results = {}
            for i, symbol in enumerate(symbols):
                pred_value = raw_predictions[i]
                
                # Z-score based confidence
                z_score = abs((pred_value - mean_pred) / std_pred)
                confidence = 1 / (1 + np.exp(-z_score))  # Sigmoid transformation
                
                results[symbol] = {
                    'prediction': float(pred_value),
                    'confidence': float(confidence),
                    'z_score': float(z_score)
                }
            
            # Store prediction metrics
            self.model_metrics[datetime.now()] = {
                'mean_prediction': mean_pred,
                'std_prediction': std_pred,
                'num_predictions': len(symbols)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {}
    
    @classmethod
    def select_stocks_with_combined_model(cls,
                                        model,
                                        date: datetime,
                                        price_data: Dict[str, pd.DataFrame],
                                        features: List[str],
                                        sequence_length: int,
                                        sector_map: Optional[Dict] = None) -> List[str]:
        """
        Select stocks using combined model with sector awareness
        
        Args:
            model: Trained model
            date: Current date
            price_data: Historical price data
            features: List of features to use
            sequence_length: Sequence length for LSTM
            sector_map: Optional mapping of symbols to sectors
            
        Returns:
            List of selected symbols
        """

        predictor = cls()
        try:
            logger.info(f"\nSelecting stocks for {date}")
            
            # Prepare prediction data
            X_lstm, node_indices, graph_info, symbols = predictor.prepare_prediction_data(
                date=date,
                price_data=price_data,
                features=features,
                sequence_length=sequence_length
            )
            
            if X_lstm is None or len(symbols) == 0:
                logger.error("Failed to prepare prediction data")
                return []
                
            logger.info(f"Prepared data for {len(symbols)} symbols")
            
            # Make predictions with confidence scores
            predictions = predictor.predict_with_confidence(
                model=model,
                X_lstm=X_lstm,
                node_indices=node_indices,
                symbols=symbols
            )
            
            if not predictions:
                logger.error("No predictions generated")
                return []
            
            # Apply filters
            filtered_symbols = predictor._apply_stock_filters(
                predictions=predictions,
                price_data=price_data,
                date=date,
                sector_map=sector_map
            )
            
            # Select top stocks
            selected = predictor._select_top_stocks(
                filtered_symbols=filtered_symbols,
                predictions=predictions,
                sector_map=sector_map
            )
            
            # Log selection summary
            predictor._log_selection_summary(
                selected=selected,
                predictions=predictions,
                sector_map=sector_map
            )
            
            return selected
            
        except Exception as e:
            logger.error(f"Error in stock selection: {str(e)}")
            return []

    def _apply_stock_filters(self,
                            predictions: Dict[str, Dict],
                            price_data: Dict[str, pd.DataFrame],
                            date: datetime,
                            sector_map: Optional[Dict] = None) -> List[str]:
        """
        Apply filtering criteria to predictions
        """
        try:
            filtered_symbols = []
            lookback_days = 60  # For historical metrics
            
            for symbol, pred_info in predictions.items():
                try:
                    df = price_data[symbol]
                    df = df[df['date'] <= date].tail(lookback_days)
                    
                    if len(df) < lookback_days:
                        continue
                    
                    # Calculate filtering metrics
                    returns = df['close'].pct_change()
                    volume = df['volume']
                    
                    # Volatility filter
                    volatility = returns.std() * np.sqrt(252)
                    if volatility > 0.40:  # Skip highly volatile stocks
                        continue
                    
                    # Liquidity filter
                    avg_volume = volume.mean()
                    if avg_volume < 100000:  # Skip illiquid stocks
                        continue
                    
                    # Momentum filter
                    momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
                    if momentum < -0.15:  # Skip strong downtrends
                        continue
                    
                    # Confidence filter
                    if pred_info['confidence'] < 0.60:  # Skip low confidence predictions
                        continue
                    
                    filtered_symbols.append(symbol)
                    
                except Exception as e:
                    logger.error(f"Error filtering {symbol}: {str(e)}")
                    continue
            
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return []

    def _select_top_stocks(self,
                        filtered_symbols: List[str],
                        predictions: Dict[str, Dict],
                        sector_map: Optional[Dict] = None) -> List[str]:
        """
        Select top stocks with sector diversification
        """
        try:
            if not filtered_symbols:
                return []
                
            # Sort by prediction value
            sorted_symbols = sorted(
                filtered_symbols,
                key=lambda x: predictions[x]['prediction'],
                reverse=True
            )
            
            selected = []
            sector_counts = defaultdict(int)
            max_sector_stocks = min(5, TOP_N // 4)  # Maximum stocks per sector
            
            for symbol in sorted_symbols:
                # Check sector limits if sector mapping available
                if sector_map and symbol in sector_map:
                    sector = sector_map[symbol]
                    if sector_counts[sector] >= max_sector_stocks:
                        continue
                    sector_counts[sector] += 1
                
                selected.append(symbol)
                
                # Check overall limits
                if len(selected) >= TOP_N:
                    break
            
            # Ensure minimum positions
            if len(selected) < MIN_POSITIONS:
                logger.warning(f"Selected only {len(selected)} stocks, minimum is {MIN_POSITIONS}")
                return []
                
            return selected
            
        except Exception as e:
            logger.error(f"Error selecting top stocks: {str(e)}")
            return []

    def _log_selection_summary(self,
                            selected: List[str],
                            predictions: Dict[str, Dict],
                            sector_map: Optional[Dict] = None):
        """Log summary of stock selection"""
        try:
            logger.info("\nStock Selection Summary:")
            logger.info(f"Selected {len(selected)} stocks")
            
            # Prediction statistics
            pred_values = [predictions[s]['prediction'] for s in selected]
            conf_values = [predictions[s]['confidence'] for s in selected]
            
            logger.info("\nPrediction Statistics:")
            logger.info(f"Mean prediction: {np.mean(pred_values):.4f}")
            logger.info(f"Mean confidence: {np.mean(conf_values):.4f}")
            
            # Sector distribution
            if sector_map:
                sector_counts = defaultdict(int)
                for symbol in selected:
                    if symbol in sector_map:
                        sector_counts[sector_map[symbol]] += 1
                        
                logger.info("\nSector Distribution:")
                for sector, count in sorted(sector_counts.items()):
                    logger.info(f"{sector}: {count} stocks")
            
            # Top selections
            logger.info("\nTop 5 Selections:")
            for symbol in selected[:5]:
                pred = predictions[symbol]
                sector = sector_map.get(symbol, 'Unknown') if sector_map else 'Unknown'
                logger.info(
                    f"{symbol} ({sector}): "
                    f"Prediction={pred['prediction']:.4f}, "
                    f"Confidence={pred['confidence']:.2f}"
                )
                
        except Exception as e:
            logger.error(f"Error logging selection summary: {str(e)}")



    def analyze_prediction_performance(self,
                                    holdings: Dict[str, float],
                                    current_prices: Dict[str, float],
                                    predictions: Dict[str, Dict]) -> Dict:
        """
        Analyze prediction performance against actual outcomes
        
        Args:
            holdings: Current portfolio holdings
            current_prices: Current prices for holdings
            predictions: Previous predictions with confidence
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            analysis = {
                'accuracy': 0.0,
                'confidence_correlation': 0.0,
                'sector_performance': {},
                'prediction_stats': {}
            }
            
            if not holdings or not predictions:
                return analysis
                
            # Calculate actual returns
            actual_returns = {}
            for symbol, shares in holdings.items():
                if symbol in predictions and symbol in current_prices:
                    entry_price = predictions[symbol].get('entry_price', 0)
                    if entry_price > 0:
                        actual_return = (current_prices[symbol] - entry_price) / entry_price
                        actual_returns[symbol] = actual_return
            
            if not actual_returns:
                return analysis
            
            # Calculate prediction accuracy
            correct_predictions = 0
            for symbol, actual_return in actual_returns.items():
                pred_return = predictions[symbol]['prediction']
                if (pred_return > 0 and actual_return > 0) or (pred_return < 0 and actual_return < 0):
                    correct_predictions += 1
                    
            analysis['accuracy'] = correct_predictions / len(actual_returns)
            
            # Analyze confidence correlation
            if len(actual_returns) > 1:
                confidences = [predictions[s]['confidence'] for s in actual_returns.keys()]
                abs_returns = [abs(r) for r in actual_returns.values()]
                analysis['confidence_correlation'] = np.corrcoef(confidences, abs_returns)[0, 1]
            
            # Prediction statistics
            pred_values = [p['prediction'] for p in predictions.values()]
            analysis['prediction_stats'] = {
                'mean': np.mean(pred_values),
                'std': np.std(pred_values),
                'min': np.min(pred_values),
                'max': np.max(pred_values)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing prediction performance: {str(e)}")
            return {}

    def monitor_prediction_drift(self,
                            current_predictions: Dict[str, Dict],
                            lookback_days: int = 20) -> Dict:
        """
        Monitor for prediction drift over time
        
        Args:
            current_predictions: Current prediction values
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary of drift metrics
        """
        try:
            drift_metrics = {}
            
            if len(self.prediction_history) < lookback_days:
                return drift_metrics
                
            recent_history = self.prediction_history[-lookback_days:]
            current_mean = np.mean([p['prediction'] for p in current_predictions.values()])
            
            # Calculate historical means
            historical_means = [
                np.mean([p['prediction'] for p in hist_pred.values()])
                for hist_pred in recent_history
            ]
            
            # Calculate drift metrics
            drift_metrics['mean_shift'] = current_mean - np.mean(historical_means)
            drift_metrics['volatility'] = np.std(historical_means)
            drift_metrics['z_score'] = (
                (current_mean - np.mean(historical_means)) / 
                (np.std(historical_means) if np.std(historical_means) > 0 else 1)
            )
            
            # Track prediction stability
            common_symbols = set(current_predictions.keys())
            for hist_pred in recent_history:
                common_symbols &= set(hist_pred.keys())
                
            if common_symbols:
                prediction_changes = []
                for symbol in common_symbols:
                    current_pred = current_predictions[symbol]['prediction']
                    hist_preds = [h[symbol]['prediction'] for h in recent_history]
                    avg_hist_pred = np.mean(hist_preds)
                    prediction_changes.append(abs(current_pred - avg_hist_pred))
                    
                drift_metrics['avg_symbol_change'] = np.mean(prediction_changes)
                drift_metrics['max_symbol_change'] = np.max(prediction_changes)
                
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error monitoring prediction drift: {str(e)}")
            return {}

    def analyze_sector_predictions(self,
                                predictions: Dict[str, Dict],
                                sector_map: Dict[str, str]) -> Dict:
        """
        Analyze predictions by sector
        
        Args:
            predictions: Current predictions
            sector_map: Mapping of symbols to sectors
            
        Returns:
            Dictionary of sector-level metrics
        """
        try:
            sector_analysis = defaultdict(lambda: {
                'count': 0,
                'avg_prediction': 0.0,
                'avg_confidence': 0.0,
                'symbols': []
            })
            
            for symbol, pred_info in predictions.items():
                if symbol in sector_map:
                    sector = sector_map[symbol]
                    sector_analysis[sector]['count'] += 1
                    sector_analysis[sector]['avg_prediction'] += pred_info['prediction']
                    sector_analysis[sector]['avg_confidence'] += pred_info['confidence']
                    sector_analysis[sector]['symbols'].append(symbol)
                    
            # Calculate averages
            for sector in sector_analysis:
                count = sector_analysis[sector]['count']
                if count > 0:
                    sector_analysis[sector]['avg_prediction'] /= count
                    sector_analysis[sector]['avg_confidence'] /= count
                    
            return dict(sector_analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing sector predictions: {str(e)}")
            return {}

    def print_prediction_summary(self,
                            predictions: Dict[str, Dict],
                            sector_map: Optional[Dict] = None):
        """Print comprehensive prediction summary"""
        try:
            logger.info("\n=== Prediction Summary ===")
            
            # Basic statistics
            pred_values = [p['prediction'] for p in predictions.values()]
            conf_values = [p['confidence'] for p in predictions.values()]
            
            logger.info("\nOverall Statistics:")
            logger.info(f"Total predictions: {len(predictions)}")
            logger.info(f"Average prediction: {np.mean(pred_values):.4f}")
            logger.info(f"Average confidence: {np.mean(conf_values):.2f}")
            
            # Prediction distribution
            logger.info("\nPrediction Distribution:")
            positive_preds = sum(1 for p in pred_values if p > 0)
            logger.info(f"Positive predictions: {positive_preds} ({positive_preds/len(pred_values):.1%})")
            
            # Confidence distribution
            high_conf = sum(1 for c in conf_values if c > 0.8)
            logger.info(f"High confidence predictions: {high_conf} ({high_conf/len(conf_values):.1%})")
            
            # Sector analysis if available
            if sector_map:
                sector_analysis = self.analyze_sector_predictions(predictions, sector_map)
                
                logger.info("\nSector Analysis:")
                for sector, metrics in sorted(
                    sector_analysis.items(),
                    key=lambda x: x[1]['avg_prediction'],
                    reverse=True
                ):
                    logger.info(f"\n{sector}:")
                    logger.info(f"Count: {metrics['count']}")
                    logger.info(f"Average prediction: {metrics['avg_prediction']:.4f}")
                    logger.info(f"Average confidence: {metrics['avg_confidence']:.2f}")
                    
            # Drift analysis
            drift_metrics = self.monitor_prediction_drift(predictions)
            if drift_metrics:
                logger.info("\nDrift Analysis:")
                logger.info(f"Mean shift: {drift_metrics['mean_shift']:.4f}")
                logger.info(f"Z-score: {drift_metrics['z_score']:.2f}")
                if 'avg_symbol_change' in drift_metrics:
                    logger.info(f"Average symbol change: {drift_metrics['avg_symbol_change']:.4f}")
                    
        except Exception as e:
            logger.error(f"Error printing prediction summary: {str(e)}")

    def save_prediction_metrics(self) -> bool:
        """Save prediction metrics for analysis"""
        try:
            metrics_df = pd.DataFrame(self.model_metrics).T
            metrics_df.to_csv('prediction_metrics.csv')
            
            # Save recent predictions
            if self.prediction_history:
                recent_preds = pd.DataFrame([
                    {
                        'date': date,
                        'symbol': symbol,
                        'prediction': pred_info['prediction'],
                        'confidence': pred_info['confidence']
                    }
                    for date, preds in self.prediction_history[-20:].items()
                    for symbol, pred_info in preds.items()
                ])
                recent_preds.to_csv('recent_predictions.csv')
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction metrics: {str(e)}")
            return False

    def load_prediction_history(self) -> bool:
        """Load saved prediction history"""
        try:
            if os.path.exists('prediction_metrics.csv'):
                metrics_df = pd.read_csv('prediction_metrics.csv', index_col=0)
                self.model_metrics = metrics_df.to_dict('index')
                
            if os.path.exists('recent_predictions.csv'):
                recent_preds = pd.read_csv('recent_predictions.csv')
                
                # Reconstruct prediction history
                self.prediction_history = defaultdict(dict)
                for _, row in recent_preds.iterrows():
                    self.prediction_history[row['date']][row['symbol']] = {
                        'prediction': row['prediction'],
                        'confidence': row['confidence']
                    }
                    
            return True
            
        except Exception as e:
            logger.error(f"Error loading prediction history: {str(e)}")
            return False