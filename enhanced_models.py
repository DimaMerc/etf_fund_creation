# enhanced_models.py


import keras
from keras import layers
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import logging
import traceback
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

import os

from graph_layers import GraphConvLayer  
from config import BATCH_SIZE
from models_utils import prepare_training_data, create_datasets, build_graph_data

logger = logging.getLogger(__name__)

class TemporalAttentionBlock(layers.Layer):
    """Enhanced Temporal Attention with ETF support"""
    
    def __init__(self, d_model: int, num_heads: int, etf_aware: bool = True):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=0.1
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        
        # Enhanced feed-forward network
        self.ffn = self._build_ffn(d_model, etf_aware)
        
        # Optional ETF-specific attention
        self.etf_aware = etf_aware
        if etf_aware:
            self.etf_attention = layers.MultiHeadAttention(
                num_heads=2,  # Reduced heads for ETF attention
                key_dim=d_model,
                dropout=0.1
            )
            self.etf_layernorm = layers.LayerNormalization()
    
    def _build_ffn(self, d_model: int, etf_aware: bool) -> keras.Sequential:
        """Build enhanced feed-forward network"""
        ffn_dim = d_model * 4
        
        if etf_aware:
            return keras.Sequential([
                layers.Dense(ffn_dim, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(d_model),
                layers.Dropout(0.1)
            ])
        else:
            return keras.Sequential([
                layers.Dense(ffn_dim, activation="relu"),
                layers.Dense(d_model)
            ])
    
    def call(self, x: tf.Tensor, is_etf: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass with optional ETF awareness"""
        try:
            # Regular self-attention
            attention_output = self.attention(x, x, x)
            x1 = self.layernorm1(x + attention_output)
            
            # ETF-specific attention if enabled and ETF indicator provided
            if self.etf_aware and is_etf is not None:
                etf_mask = tf.expand_dims(is_etf, -1)
                etf_attention = self.etf_attention(x1, x1, x1, attention_mask=etf_mask)
                x1 = self.etf_layernorm(x1 + etf_attention)
            
            # Feed-forward network
            ffn_output = self.ffn(x1)
            return self.layernorm2(x1 + ffn_output)
            
        except Exception as e:
            logger.error(f"Error in TemporalAttentionBlock: {str(e)}")
            return x




class EnhancedCombinedModel(keras.Model):
    def __init__(self,
                 sequence_length: int,
                 feature_dim: int,
                 node_feature_dim: int,
                 hidden_units: List[int],
                 graph_info: Tuple,
                 etf_aware: bool = True):
        
        super().__init__()
        # Store parameters
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.node_feature_dim = node_feature_dim
        self.lstm_units = hidden_units[0]
        self.dense_units = hidden_units[1]
        self.graph_info = graph_info
        self.etf_aware = etf_aware
        
        # Input processing
        self.input_dense = layers.Dense(self.dense_units, activation='relu')
        
        # LSTM Branch
        self.lstm_1 = self._build_lstm_branch()
        
        # Technical Branch
        self.technical_dense = self._build_technical_branch()
        
        # Graph Branch
        self.graph_layers = self._build_graph_branch()
        
        # ETF-specific components
        if etf_aware:
            self.etf_attention = TemporalAttentionBlock(
                d_model=self.dense_units,
                num_heads=4,
                etf_aware=True
            )
            self.etf_processor = self._build_etf_processor()
        
        # Output layers
        self.final_dense = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(1)
        
        # Regularization
        self.dropout = layers.Dropout(0.2)
        self.batch_norm = layers.BatchNormalization()

    def _build_lstm_branch(self) -> List[layers.Layer]:
        """Build LSTM branch with enhanced sequence handling"""
        return [
            layers.LSTM(self.lstm_units, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.dense_units, return_sequences=False)
        ]
    
    def _build_technical_branch(self) -> List[layers.Layer]:
        """Build technical analysis branch"""
        return [
            layers.Dense(self.dense_units, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.dense_units // 2, activation='relu')
        ]
    
    def _build_graph_branch(self) -> List[GraphConvLayer]:
        """Build graph convolutional layers"""
        return [
            GraphConvLayer(output_dim=self.dense_units),
            GraphConvLayer(output_dim=self.dense_units // 2)
        ]
    
    def _build_etf_processor(self) -> keras.Sequential:
        """Build ETF-specific processing layers"""
        return keras.Sequential([
            layers.Dense(self.dense_units, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.dense_units // 2, activation='relu')
        ])

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Forward pass with ETF awareness
        
        Args:
            inputs: Dictionary containing:
                - lstm_input: LSTM input sequences
                - node_indices: Graph node indices
                - is_etf: Optional tensor indicating ETF status
            training: Whether in training mode
        """
        try:
            # Unpack inputs
            lstm_input = inputs['lstm_input']
            node_indices = inputs['node_indices']
            is_etf = inputs.get('is_etf', None)
            
            # Input validation
            tf.debugging.assert_rank(lstm_input, 3, "lstm_input must be rank 3")
            tf.debugging.assert_rank(node_indices, 1, "node_indices must be rank 1")
            
            # Process LSTM branch
            x_lstm = self.input_dense(lstm_input)
            for layer in self.lstm_1:
                x_lstm = layer(x_lstm)
            
            # Process technical branch
            x_tech = lstm_input[:, -1, :]
            for layer in self.technical_dense:
                x_tech = layer(x_tech)
            
            # Process graph branch
            node_features, edges, edge_weights = self.graph_info
            x_graph = node_features
            for layer in self.graph_layers:
                x_graph = layer((x_graph, edges, edge_weights))
            node_embeddings = tf.gather(x_graph, node_indices)
            
            # ETF-specific processing
            if self.etf_aware and is_etf is not None:
                # Apply ETF attention
                x_etf = self.etf_attention(
                    tf.concat([x_lstm, x_tech, node_embeddings], axis=-1),
                    is_etf
                )
                
                # Process ETF features
                x_etf = self.etf_processor(x_etf)
                
                # Combine with other features
                combined = tf.concat([x_lstm, x_tech, node_embeddings, x_etf], axis=-1)
            else:
                combined = tf.concat([x_lstm, x_tech, node_embeddings], axis=-1)
            
            # Final processing
            x = self.dropout(combined, training=training)
            x = self.batch_norm(x, training=training)
            x = self.final_dense(x)
            output = self.output_layer(x)
            
            return tf.squeeze(output, axis=-1)
            
        except Exception as e:
            logger.error(f"Error in model forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise



class ETFAttentionBlock(layers.Layer):
    """Enhanced attention mechanism for ETF constituent selection"""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout
        )
        self.layernorm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)
        
        # ETF-specific dense transformation
        self.etf_transform = layers.Dense(d_model, activation='relu')
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with ETF-specific attention"""
        attention_output = self.attention(inputs, inputs, inputs)
        attention_output = self.dropout(attention_output, training=training)
        normalized = self.layernorm(inputs + attention_output)
        return self.etf_transform(normalized)

class ETFConstituentModel(EnhancedCombinedModel):
    """Enhanced model for ETF constituent selection"""
    
    def __init__(self,
                 sequence_length: int,
                 feature_dim: int,
                 hidden_units: List[int],
                 sector_embedding_dim: int = 32,
                 num_attention_heads: int = 4):
        super().__init__()
        
        # Model parameters
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_units = hidden_units
        self.sector_embedding_dim = sector_embedding_dim
        
        # Input layers
        self.temporal_embedding = layers.Dense(hidden_units[0])
        self.sector_embedding = layers.Embedding(
            input_dim=20,  # Assuming max 20 sectors
            output_dim=sector_embedding_dim
        )
        
        # LSTM branch for temporal features
        self.lstm_layers = [
            layers.LSTM(units, return_sequences=True)
            for units in hidden_units[:-1]
        ]
        self.lstm_layers.append(layers.LSTM(hidden_units[-1]))
        
        # Attention mechanism
        self.attention = ETFAttentionBlock(
            d_model=hidden_units[0],
            num_heads=num_attention_heads
        )
        
        # Final layers
        self.dropout = layers.Dropout(0.2)
        self.final_dense = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Forward pass with ETF-aware processing"""
        try:
            # Unpack inputs
            temporal_data = inputs['temporal_data']
            sector_data = inputs.get('sector_data')
            
            # Process temporal data
            x = self.temporal_embedding(temporal_data)
            x = self.attention(x, training=training)
            
            # LSTM processing
            for lstm_layer in self.lstm_layers:
                x = lstm_layer(x)
                
            # Add sector information if available
            if sector_data is not None:
                sector_embed = self.sector_embedding(sector_data)
                x = tf.concat([x, sector_embed], axis=-1)
                
            # Final processing
            x = self.dropout(x, training=training)
            x = self.final_dense(x)
            return self.output_layer(x)
            
        except Exception as e:
            logger.error(f"Error in model forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class ETFPortfolioLoss(keras.losses.Loss):
    """Custom loss function for ETF portfolio construction"""
    
    def __init__(self,
                 tracking_weight: float = 0.4,
                 sector_weight: float = 0.3,
                 liquidity_weight: float = 0.3,
                 name: str = 'etf_portfolio_loss'):
        super().__init__(name=name)
        self.tracking_weight = tracking_weight
        self.sector_weight = sector_weight
        self.liquidity_weight = liquidity_weight
        self.mse = keras.losses.MeanSquaredError()
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor,
            sample_weight: Optional[Dict] = None) -> tf.Tensor:
        """Calculate loss with ETF-specific components"""
        try:
            # Base prediction loss
            prediction_loss = self.mse(y_true, y_pred)
            
            if sample_weight is None:
                return prediction_loss
                
            # Tracking error component
            tracking_error = sample_weight.get('tracking_error', 0.0)
            tracking_loss = self.tracking_weight * tf.reduce_mean(tracking_error)
            
            # Sector deviation component
            sector_deviation = sample_weight.get('sector_deviation', 0.0)
            sector_loss = self.sector_weight * tf.reduce_mean(sector_deviation)
            
            # Liquidity component
            liquidity_score = sample_weight.get('liquidity_score', 1.0)
            liquidity_loss = self.liquidity_weight * (1.0 - tf.reduce_mean(liquidity_score))
            
            # Combine losses
            total_loss = prediction_loss + tracking_loss + sector_loss + liquidity_loss
            return total_loss
            
        except Exception as e:
            logger.error(f"Error calculating loss: {str(e)}")
            return self.mse(y_true, y_pred)

def train_enhanced_model(df: pd.DataFrame,
                        features: List[str],
                        target: str,
                        sequence_length: int,
                        etf_holdings: Optional[Dict] = None) -> Optional[keras.Model]:
    """
    Train enhanced model with ETF awareness
    
    Args:
        df: Input DataFrame
        features: Feature columns
        target: Target column
        sequence_length: Sequence length for LSTM
        etf_holdings: Optional ETF holdings data
    """
    try:
        logger.info("Starting enhanced model training...")
        
        # Prepare data with ETF information
        X_lstm, node_indices, y, graph_info, sample_weights = prepare_training_data_with_etf(
            df=df,
            features=features,
            target=target,
            sequence_length=sequence_length,
            etf_holdings=etf_holdings
        )
        
        if X_lstm is None or graph_info is None:
            logger.error("Failed to prepare training data")
            return None
        
        # Convert data types
        X_lstm = tf.cast(X_lstm, tf.float32)
        node_indices = tf.cast(node_indices, tf.int32)
        y = tf.cast(y, tf.float32)
        
        # Sample if dataset is too large
        if len(X_lstm) > 10000:
            indices = np.random.choice(len(X_lstm), 10000, replace=False)
            X_lstm = tf.gather(X_lstm, indices)
            node_indices = tf.gather(node_indices, indices)
            y = tf.gather(y, indices)
            sample_weights = {
                k: tf.gather(v, indices) for k, v in sample_weights.items()
            }
        
        # Build model
        model = EnhancedCombinedModel(
            sequence_length=sequence_length,
            feature_dim=len(features),
            node_feature_dim=graph_info[0].shape[1],
            hidden_units=[64, 32],
            graph_info=graph_info,
            etf_aware=True
        )
        
        # Custom training setup
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = ETFPortfolioLoss()
        
        # Training loop with ETF awareness
        @tf.function
        def train_step(x, y_true, weights):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_fn(y_true, y_pred, sample_weight=weights)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        
        # Training loop
        batch_size = 128
        train_dataset = create_etf_aware_dataset(
            X_lstm, node_indices, y, sample_weights, batch_size
        )
        
        num_epochs = 5
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_loss = tf.keras.metrics.Mean()
            
            for step, (x_batch, y_batch, weights_batch) in enumerate(train_dataset):
                loss = train_step(x_batch, y_batch, weights_batch)
                epoch_loss.update_state(loss)
                
                if step % 100 == 0:
                    logger.info(f"Step {step}, Loss: {float(loss):.4f}")
            
            logger.info(f"Epoch {epoch + 1} Loss: {epoch_loss.result():.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_etf_aware_dataset(X_lstm: tf.Tensor,
                           node_indices: tf.Tensor,
                           y: tf.Tensor,
                           sample_weights: Dict[str, tf.Tensor],
                           batch_size: int) -> tf.data.Dataset:
    """
    Create dataset with ETF awareness
    
    Args:
        X_lstm: LSTM input data
        node_indices: Graph node indices
        y: Target values
        sample_weights: Dictionary of sample weights
        batch_size: Batch size
    """
    try:
        # Create dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((
            {'lstm_input': X_lstm, 'node_indices': node_indices},
            y,
            sample_weights
        ))
        
        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error creating ETF-aware dataset: {str(e)}")
        return None

def prepare_training_data_with_etf(df: pd.DataFrame,
                                 features: List[str],
                                 target: str,
                                 sequence_length: int,
                                 etf_holdings: Optional[Dict] = None) -> Tuple:
    """
    Prepare training data with ETF-specific information
    
    Args:
        df: Input DataFrame
        features: Feature columns
        target: Target column
        sequence_length: Sequence length for LSTM
        etf_holdings: Optional ETF holdings data
    """
    try:
        # Identify ETFs
        is_etf = df['symbol'].apply(lambda x: x.endswith('ETF')).values
        
        # Calculate standard sequence data
        X_lstm, node_indices, y, graph_info = prepare_training_data(
            df, features, target, sequence_length
        )
        
        if X_lstm is None:
            return None, None, None, None, None
        
        # Prepare sample weights
        sample_weights = {
            'is_etf': tf.cast(is_etf, tf.float32)
        }
        
        # Add tracking error if ETF holdings available
        if etf_holdings is not None:
            tracking_error = calculate_tracking_errors(
                df, etf_holdings
            )
            if tracking_error is not None:
                sample_weights['tracking_error'] = tf.cast(
                    tracking_error, tf.float32
                )
        
        # Add volume-based weights
        volume_ratios = calculate_volume_ratios(df)
        if volume_ratios is not None:
            sample_weights['volume_ratio'] = tf.cast(
                volume_ratios, tf.float32
            )
        
        return X_lstm, node_indices, y, graph_info, sample_weights
        
    except Exception as e:
        logger.error(f"Error preparing ETF training data: {str(e)}")
        return None, None, None, None, None
    
# enhanced_models.py - Additional Utility Functions

def calculate_tracking_errors(df: pd.DataFrame,
                            etf_holdings: Dict) -> Optional[np.ndarray]:
    """
    Calculate tracking errors for ETFs relative to their holdings
    
    Args:
        df: DataFrame with price data
        etf_holdings: Dictionary of ETF holdings data
    """
    try:
        tracking_errors = np.zeros(len(df))
        
        # Group data by symbol
        grouped = df.groupby('symbol')
        
        for symbol, group in grouped:
            if symbol in etf_holdings:
                # Get holdings data
                holdings = etf_holdings[symbol]['holdings']
                
                # Calculate ETF returns
                etf_returns = group['close'].pct_change()
                
                # Calculate theoretical returns based on holdings
                theoretical_returns = np.zeros_like(etf_returns)
                for _, holding in holdings.iterrows():
                    if holding['symbol'] in grouped.groups:
                        holding_data = grouped.get_group(holding['symbol'])
                        holding_returns = holding_data['close'].pct_change()
                        theoretical_returns += holding['weight'] * holding_returns
                
                # Calculate tracking error
                tracking_error = np.std(etf_returns - theoretical_returns) * np.sqrt(252)
                tracking_errors[group.index] = tracking_error
        
        return tracking_errors
        
    except Exception as e:
        logger.error(f"Error calculating tracking errors: {str(e)}")
        return None

def calculate_volume_ratios(df: pd.DataFrame,
                          window: int = 20) -> Optional[np.ndarray]:
    """
    Calculate volume ratios for weighting predictions
    
    Args:
        df: DataFrame with volume data
        window: Rolling window size for volume calculations
    """
    try:
        volume_ratios = np.zeros(len(df))
        
        # Group data by symbol
        for symbol, group in df.groupby('symbol'):
            # Calculate average daily volume
            avg_volume = group['volume'].rolling(window=window).mean()
            
            # Calculate volume ratio relative to recent average
            current_volume = group['volume']
            ratio = current_volume / avg_volume
            
            # Store ratios
            volume_ratios[group.index] = ratio
        
        # Normalize ratios
        volume_ratios = np.nan_to_num(volume_ratios, nan=1.0)
        volume_ratios = np.clip(volume_ratios, 0.5, 2.0)
        
        return volume_ratios
        
    except Exception as e:
        logger.error(f"Error calculating volume ratios: {str(e)}")
        return None

def calculate_etf_metrics(df: pd.DataFrame,
                         etf_holdings: Optional[Dict] = None,
                         window: int = 60) -> Dict[str, np.ndarray]:
    """
    Calculate additional ETF-specific metrics for model inputs
    
    Args:
        df: DataFrame with price/volume data
        etf_holdings: Optional ETF holdings data
        window: Rolling window for calculations
    """
    try:
        metrics = {}
        
        # Group data by symbol
        grouped = df.groupby('symbol')
        
        # Initialize arrays for each metric
        length = len(df)
        metrics['volume_stability'] = np.zeros(length)
        metrics['price_stability'] = np.zeros(length)
        metrics['tracking_deviation'] = np.zeros(length)
        metrics['holdings_correlation'] = np.zeros(length)
        
        for symbol, group in grouped:
            if symbol.endswith('ETF') or (etf_holdings and symbol in etf_holdings):
                # Calculate volume stability
                vol_std = group['volume'].rolling(window).std()
                vol_mean = group['volume'].rolling(window).mean()
                vol_stability = 1 - (vol_std / vol_mean)
                metrics['volume_stability'][group.index] = vol_stability
                
                # Calculate price stability
                returns = group['close'].pct_change()
                price_stability = 1 - returns.rolling(window).std()
                metrics['price_stability'][group.index] = price_stability
                
                # Calculate tracking metrics if holdings available
                if etf_holdings and symbol in etf_holdings:
                    holdings = etf_holdings[symbol]['holdings']
                    
                    # Calculate correlation with holdings
                    holdings_returns = np.zeros_like(returns)
                    for _, holding in holdings.iterrows():
                        if holding['symbol'] in grouped.groups:
                            holding_data = grouped.get_group(holding['symbol'])
                            holding_returns = holding_data['close'].pct_change()
                            holdings_returns += holding['weight'] * holding_returns
                    
                    # Calculate rolling correlation
                    correlation = returns.rolling(window).corr(holdings_returns)
                    metrics['holdings_correlation'][group.index] = correlation
                    
                    # Calculate tracking deviation
                    tracking_dev = np.abs(returns - holdings_returns).rolling(window).mean()
                    metrics['tracking_deviation'][group.index] = tracking_dev
        
        # Clean up metrics
        for key in metrics:
            metrics[key] = np.nan_to_num(metrics[key], nan=0.0)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating ETF metrics: {str(e)}")
        return {}
    

