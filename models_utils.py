import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Tuple, List
import logging
import traceback
from predictor import EnhancedPredictor

logger = logging.getLogger(__name__)

def create_datasets(X_lstm: np.ndarray, node_indices: np.ndarray, y: np.ndarray, 
                   batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create training, validation, and test datasets"""
    try:
        # Split data
        total_size = len(y)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)

        # Training data
        X_lstm_train = X_lstm[:train_size]
        node_indices_train = node_indices[:train_size]
        y_train = y[:train_size]

        # Validation data
        X_lstm_val = X_lstm[train_size:train_size + val_size]
        node_indices_val = node_indices[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        # Test data
        X_lstm_test = X_lstm[train_size + val_size:]
        node_indices_test = node_indices[train_size + val_size:]
        y_test = y[train_size + val_size:]

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {'lstm_input': X_lstm_train, 'node_indices': node_indices_train},
            y_train
        )).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            {'lstm_input': X_lstm_val, 'node_indices': node_indices_val},
            y_val
        )).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((
            {'lstm_input': X_lstm_test, 'node_indices': node_indices_test},
            y_test
        )).batch(batch_size)

        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        return None, None, None

def build_graph_data(symbols_list: List[str]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Build graph data for the LSTM/GCN model"""
    try:
        num_nodes = len(symbols_list)
        node_feature_dim = 64
        
        # Create node features
        node_features = tf.random.uniform((num_nodes, node_feature_dim), dtype=tf.float32)
        
        # Create edges for fully connected graph
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        edges = np.array(edges).T
        edges = tf.convert_to_tensor(edges, dtype=tf.int64)
        
        # Create edge weights
        edge_weights = tf.ones(edges.shape[1], dtype=tf.float32)
        
        return node_features, edges, edge_weights
        
    except Exception as e:
        logger.error(f"Error building graph data: {str(e)}")
        return None, None, None

def create_lstm_sequences(df: pd.DataFrame, features: List[str], target: str, 
                        sequence_length: int) -> Tuple[np.ndarray, np.ndarray, List]:
    """Create sequences for LSTM training"""
    try:
        if len(df) < sequence_length + 1:
            return np.array([]), np.array([]), []
            
        X_sequences = []
        y_sequences = []
        dates = []
        
        for i in range(len(df) - sequence_length):
            X_seq = df[features].iloc[i:i+sequence_length].values
            y_seq = df[target].iloc[i+sequence_length]
            date = df['date'].iloc[i+sequence_length]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
            dates.append(date)
        
        return np.array(X_sequences), np.array(y_sequences), dates
        
    except Exception as e:
        logger.error(f"Error creating LSTM sequences: {str(e)}")
        return np.array([]), np.array([]), []
    

def prepare_training_data(df: pd.DataFrame,
                         features: List[str],
                         target: str,
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple]:
    try:
        logger.info("\nPreparing training data...")
        
        # 1. Input validation
        if df is None or df.empty:
            logger.error("Empty DataFrame provided")
            return None, None, None, None
            
        if not all(f in df.columns for f in features):
            missing = [f for f in features if f not in df.columns]
            logger.error(f"Missing features: {missing}")
            return None, None, None, None
            
        # 2. Group by symbol and create sequences
        symbol_list = list(df.index.get_level_values(0).unique())
        symbol_to_index = {symbol: idx for idx, symbol in enumerate(symbol_list)}
        
        logger.info(f"Found {len(symbol_list)} unique symbols")
        
        all_sequences = []
        node_indices_list = []
        y_values = []
        
        for symbol in symbol_list:
            try:
                group = df.loc[symbol]
                feature_data = group[features].values
                
                if len(feature_data) < sequence_length:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                    
                # Create sequences
                for i in range(len(feature_data) - sequence_length):
                    X_seq = feature_data[i:i+sequence_length]
                    y_val = group[target].iloc[i+sequence_length]
                    
                    # Validate sequence
                    if np.any(np.isnan(X_seq)) or np.isnan(y_val):
                        continue
                        
                    all_sequences.append(X_seq)
                    node_indices_list.append(symbol_to_index[symbol])
                    y_values.append(y_val)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
                
        if not all_sequences:
            logger.error("No valid sequences generated")
            return None, None, None, None
            
        # 3. Convert to arrays
        X_lstm = np.array(all_sequences)
        node_indices = np.array(node_indices_list)
        y = np.array(y_values)
        
        # 4. Create graph info
        node_features = np.random.normal(size=(len(symbol_list), 64))
        edges = np.array([[i, j] for i in range(len(symbol_list)) 
                         for j in range(len(symbol_list)) if i != j]).T
        edge_weights = np.ones(edges.shape[1])
        
        graph_info = (node_features, edges, edge_weights)
        
        logger.info(f"Created {len(all_sequences)} sequences")
        return X_lstm, node_indices, y, graph_info
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None