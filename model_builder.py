# model_builder.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as keras
import keras
import ta
import matplotlib.pyplot as plt
import os
import traceback
import logging
from graph_layers import GraphConvLayer
from enhanced_models import EnhancedCombinedModel, train_enhanced_model
from data_utils import concatenate_metrics_arrays

import time
from typing import List, Dict, Optional, Tuple
from data_preparation import ETFDataPipeline, EnhancedDataPipeline
from models_utils import prepare_training_data
from sklearn.preprocessing import StandardScaler
from column_utils import standardize_dataframe_columns, get_column_name


from config import EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH, FEATURES, TARGET, ETF_FEATURES

logger = logging.getLogger(__name__)



class CombinedModel(keras.Model):
    def __init__(self, sequence_length, feature_dim, node_feature_dim, num_nodes, hidden_units, graph_info):
        super().__init__()
        
        self.graph_info = graph_info
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.node_feature_dim = node_feature_dim
        self.hidden_units = hidden_units
        self.num_nodes = num_nodes
        
        # LSTM branch
        self.input_projection = keras.layers.Dense(32)
        self.lstm_1 = keras.layers.LSTM(32, return_sequences=True)  # Removed input_shape
        self.lstm_2 = keras.layers.LSTM(32, return_sequences=True)
        self.lstm_3 = keras.layers.LSTM(32)
        
        # GCN branch
        self.node_projection = keras.layers.Dense(32)
        self.gcn_dense = keras.layers.Dense(32, activation='relu')
        self.conv1 = GraphConvLayer(output_dim=32)
        self.conv2 = GraphConvLayer(output_dim=32)
        self.gcn_output = keras.layers.Dense(32, activation='relu')
        
        # Final layers
        self.concat_layer = keras.layers.Concatenate()
        self.fc1 = keras.layers.Dense(32, activation='relu')
        self.fc2 = keras.layers.Dense(16, activation='relu')
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # LSTM branch
        lstm_input = inputs['lstm_input']
        x_lstm = self.input_projection(lstm_input)
        x_lstm = self.lstm_1(x_lstm)
        x_lstm = self.lstm_2(x_lstm)
        x_lstm = self.lstm_3(x_lstm)
        
        # GCN branch
        node_features, edges, edge_weights = self.graph_info
        node_features = tf.cast(node_features, tf.float32)
        edges = tf.cast(edges, tf.int32)
        edge_weights = tf.cast(edge_weights, tf.float32)
        
        # Project node features to correct dimension
        node_features_projected = self.node_projection(node_features)
        
        x_gcn = self.gcn_dense(node_features)
        x_gcn = self.conv1((x_gcn, edges, edge_weights))
        x_gcn = x_gcn + node_features_projected
        x_gcn = self.conv2((x_gcn, edges, edge_weights))
        x_gcn = x_gcn + node_features_projected
        x_gcn = self.gcn_output(x_gcn)
        
        # Get relevant node embeddings
        node_embeddings = tf.gather(x_gcn, inputs['node_indices'])
        
        # Combine branches
        x = self.concat_layer([x_lstm, node_embeddings])
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.output_layer(x)
        
        return tf.squeeze(output, axis=-1)
    
    def get_config(self):
        """Add get_config for serialization"""
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'node_feature_dim': self.node_feature_dim,
            'hidden_units': self.hidden_units,
            'graph_info': self.graph_info
        })
        return config
    

class ETFBenchmarkModel(keras.Model):
    def __init__(self, feature_dim):
        super().__init__()
        
        # Store parameter
        self.feature_dim = feature_dim
        # Feature processing layers
         # Feature processing layers - now using feature_dim
        self.feature_dense = keras.layers.Dense(feature_dim * 2)  # Double the input dimension
        self.dropout = keras.layers.Dropout(0.2)
        self.lstm = keras.layers.LSTM(feature_dim)  # Match LSTM units to feature dimension
        
        # Attention mechanism
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=feature_dim,
            value_dim=feature_dim
        )
        self.attention_norm = keras.layers.LayerNormalization()
        
        # Output layers - scaled based on feature_dim
        self.dense1 = keras.layers.Dense(feature_dim, activation='relu')
        self.dense2 = keras.layers.Dense(feature_dim // 2, activation='relu')  # Half the dimension
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.feature_dense(inputs)
        x = self.dropout(x)
        
        # LSTM processing
        lstm_out = self.lstm(x)
        
        # Self-attention
        attention_out = self.attention(x, x)
        attention_out = self.attention_norm(x + attention_out)
        
        # Combine features
        x = keras.layers.concatenate([lstm_out, keras.layers.GlobalAveragePooling1D()(attention_out)])
        
        # Final processing
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
    
    def get_config(self):
        """Add get_config for serialization"""
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim
        })
        return config
    

def should_use_enhanced_model(df, market_data=None):
    """
    Determines whether to use enhanced model with improved error handling
    """
    try:
        # First try to get market data with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if market_data is None:
                    market_data = yf.download('SPY', 
                                            start=df['date'].min(),
                                            end=df['date'].max(),
                                            progress=False)['Adj Close']
                    if not market_data.empty:
                        break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to fetch market data failed: {str(e)}")
                time.sleep(2)  # Wait before retry
                
        # If market data fetch failed, use conservative approach
        if market_data is None or market_data.empty:
            logger.warning("Unable to fetch market data, defaulting to enhanced model")
            return True
            
        # Calculate market metrics safely
        returns = market_data.pct_change().dropna()
        
        if len(returns) < 20:  # Not enough data
            logger.warning("Insufficient market data for volatility calculation")
            return True
            
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        if len(volatility) == 0:  # No volatility data
            logger.warning("No volatility data available")
            return True
            
        current_volatility = volatility.iloc[-1]
        
        try:
            ma_50 = market_data.rolling(50).mean()
            ma_200 = market_data.rolling(200).mean()
            
            # Market conditions analysis
            is_high_vol = current_volatility > volatility.quantile(0.7)
            trend_change = abs(ma_50.iloc[-1] / ma_200.iloc[-1] - 1) if len(ma_200) > 0 else 0
            is_trend_change = trend_change > 0.02
            
            # Data characteristics
            data_points = len(df)
            num_features = len([col for col in df.columns if col not in ['date', 'symbol']])
            
            # Market regime strength
            market_deviation = abs(market_data.iloc[-1] / ma_200.iloc[-1] - 1) if len(ma_200) > 0 else 0
            strong_market_move = market_deviation > 0.1
            
            # Decision criteria
            use_enhanced = any([
                is_high_vol,
                is_trend_change,
                data_points > 1000 and num_features > 10,
                strong_market_move
            ])
            
            logger.info("\nModel Selection Analysis:")
            logger.info(f"Market Volatility: {current_volatility:.2%}")
            logger.info(f"High Volatility Regime: {is_high_vol}")
            logger.info(f"Trend Change Magnitude: {trend_change:.2%}")
            logger.info(f"Market Deviation: {market_deviation:.2%}")
            logger.info(f"Data Points: {data_points}")
            logger.info(f"Number of Features: {num_features}")
            logger.info(f"Selected Model: {'Enhanced' if use_enhanced else 'Standard'}")
            
            return use_enhanced
            
        except Exception as e:
            logger.warning(f"Error in market analysis: {str(e)}")
            return True  # Default to enhanced model on error
            
    except Exception as e:
        logger.error(f"Error in model selection: {str(e)}")
        logger.error(traceback.format_exc())
        return True  # Default to enhanced model on error

def train_option_pricing_model(df, features, target):
    """
    Trains a Random Forest Regressor to predict option prices.
    """
    X = df[features]
    y = df[target]

    # Handle missing values
    X = X.dropna()
    y = y.loc[X.index]

    # Encode 'optionType' if present
    if 'optionType' in X.columns:
        le = LabelEncoder()
        X['optionType'] = le.fit_transform(X['optionType'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Option Pricing Model MAE: {mae:.6f}")

    return model

def create_lstm_sequences(df, features, target, sequence_length):
    """
    Creates sequences of data for LSTM models.
    """
    X_sequences = []
    y_sequences = []
    dates = []

    if len(df) < sequence_length + 1:
        # Not enough data to create a sequence
        return np.array([]), np.array([]), []

    for i in range(len(df) - sequence_length):
        X_seq = df[features].iloc[i:i+sequence_length].values
        y_seq = df[target].iloc[i+sequence_length]
        date = df['date'].iloc[i+sequence_length]

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
        dates.append(date)

    return np.array(X_sequences), np.array(y_sequences), dates

def build_graph_data(symbols: List[str],
                    market_relations: Optional[Dict] = None) -> Tuple[np.ndarray, 
                                                                    np.ndarray]:
    """
    Build graph data with enhanced market relationships
    
    Args:
        symbols: List of symbols
        market_relations: Optional dictionary of market relationships
        
    Returns:
        Tuple of (node_features, edges)
    """
    try:
        num_nodes = len(symbols)
        node_feature_dim = 64  # Keep original dimension
        
        # Initialize node features
        node_features = tf.random.normal((num_nodes, node_feature_dim))
        
        # Build edges with market relationships if available
        if market_relations:
            edges = []
            edge_weights = []
            
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i != j:
                        # Check if symbols have a market relationship
                        relation_strength = market_relations.get(
                            (sym1, sym2),
                            market_relations.get((sym2, sym1), 0.5)
                        )
                        
                        edges.append([i, j])
                        edge_weights.append(relation_strength)
                        
            edges = tf.convert_to_tensor(edges, dtype=tf.int32)
            edge_weights = tf.convert_to_tensor(edge_weights, dtype=tf.float32)
            
        else:
            # Default to fully connected graph
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append([i, j])
            edges = tf.convert_to_tensor(edges, dtype=tf.int32)
            edge_weights = tf.ones(shape=(len(edges),), dtype=tf.float32)
        
        return node_features, edges, edge_weights
        
    except Exception as e:
        logger.error(f"Error building graph data: {str(e)}")
        return None, None, None

    
def train_combined_model(df, features, target, sequence_length, sector_map=None):
    try:
        logger.info("\nStarting model training...")
        
        # 1. Initial data inspection
        logger.info("\nInitial Data Analysis:")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Features: {features}")
        logger.info("\nMissing values before cleaning:")
        for col in df.columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                logger.info(f"{col}: {na_count} NaN values ({na_count/len(df):.2%})")

        # 2. Data cleaning
        df_cleaned = df.copy()
        
        # Handle inf values first
        df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill for time series data
        df_cleaned = df_cleaned.groupby(level=0).ffill()
        df_cleaned = df_cleaned.groupby(level=0).bfill()
        
        # If any NaNs remain, fill with 0 (last resort)
        df_cleaned = df_cleaned.fillna(0)
        
        # 3. Verify cleaning results
        logger.info("\nMissing values after cleaning:")
        remaining_nas = df_cleaned[features].isna().sum()
        if remaining_nas.any():
            logger.error("Still have NaN values after cleaning!")
            logger.error(remaining_nas[remaining_nas > 0])
            return None
            
        # 4. Normalization
        scaler = StandardScaler()
        df_cleaned[features] = scaler.fit_transform(df_cleaned[features])
        
        # 5. Remove outliers
        df_cleaned[features] = np.clip(df_cleaned[features], -10, 10)
        
        # 6. Prepare training data
        X_lstm, node_indices, y, graph_info = prepare_training_data(
            df=df_cleaned,
            features=features,
            target=target,
            sequence_length=sequence_length,
        )

        # 7. Validate prepared data
        if X_lstm is None:
            logger.error("Failed to prepare training data")
            return None
            
        logger.info("\nPrepared Data Stats:")
        logger.info(f"X_lstm shape: {X_lstm.shape}")
        logger.info(f"y shape: {y.shape if y is not None else 'None'}")
        
        # Check for NaN/Inf in prepared data
        if np.any(np.isnan(X_lstm)):
            logger.error("NaN values in X_lstm")
            nan_indices = np.where(np.isnan(X_lstm))
            logger.error(f"NaN locations: {nan_indices}")
            return None
            
        if np.any(np.isnan(y)):
            logger.error("NaN values in y")
            nan_indices = np.where(np.isnan(y))
            logger.error(f"NaN locations: {nan_indices}")
            return None

        # 8. Build and compile model
        model = CombinedModel(
            sequence_length=sequence_length,
            feature_dim=len(features),
            node_feature_dim=graph_info[0].shape[1],
            num_nodes=len(np.unique(node_indices)),
            hidden_units=[64, 32],
            graph_info=graph_info
        )
        
        # Use Huber loss for robustness
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.Huber(),
            metrics=[keras.metrics.RootMeanSquaredError()]
        )

        # 9. Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'lstm_input': X_lstm,
                'node_indices': node_indices
            },
            y
        )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # 10. Train with callbacks
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ModelCheckpoint(
                    'best_model.h5.keras',
                    save_best_only=True,
                    monitor='loss'
                )
            ],
            verbose=1
        )

        logger.info("\nTraining completed successfully")
        return model

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    

def add_sector_features(node_features, symbols, sector_map):
    unique_sectors = list(set(sector_map.values()))
    sector_to_idx = {sector: idx for idx, sector in enumerate(unique_sectors)}
    
    # Create one-hot encoded sector features
    sector_features = np.zeros((len(symbols), len(unique_sectors)))
    for i, symbol in enumerate(symbols):
        if symbol in sector_map:
            sector_features[i, sector_to_idx[sector_map[symbol]]] = 1
            
    # Concatenate with existing node features
    arrays_list = [node_features, sector_features]
    enhanced_features = concatenate_metrics_arrays(arrays_list)
    if enhanced_features is None:
        logger.error("Failed to concatenate node and sector features.")
        return enhanced_features, unique_sectors

def train_batch_model(batch_data, features, target, sequence_length):
    try:
        # Prepare data for this batch
        X_lstm, node_indices, y, graph_info = prepare_training_data(
            batch_data, features, target, sequence_length
        )

        # Initialize model for batch
        model = EnhancedCombinedModel(
            sequence_length=sequence_length,
            feature_dim=len(features),
            node_feature_dim=graph_info[0].shape[1],
            hidden_units=[64, 32],
            graph_info=graph_info
        )

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')

        # Train on batch
        inputs = {
            'lstm_input': X_lstm,
            'node_indices': node_indices
        }
        model.fit(inputs, y, epochs=5, batch_size=32)

        return model

    except Exception as e:
        logger.error(f"Error training batch model: {str(e)}")
        return None
    


def create_datasets(X_lstm: np.ndarray,
                   node_indices: np.ndarray,
                   y: np.ndarray,
                   batch_size: int = BATCH_SIZE) -> Tuple[tf.data.Dataset, 
                                                        tf.data.Dataset,
                                                        tf.data.Dataset]:
    """Create training datasets with enhanced batching"""
    try:
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_lstm, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Split node indices accordingly
        train_size = len(X_train)
        val_size = len(X_val)
        
        node_train = node_indices[:train_size]
        node_val = node_indices[train_size:train_size + val_size]
        node_test = node_indices[train_size + val_size:]
        
        # Create TensorFlow datasets with prefetch
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {'lstm_input': X_train, 'node_indices': node_train},
            y_train
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {'lstm_input': X_val, 'node_indices': node_val},
            y_val
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            {'lstm_input': X_test, 'node_indices': node_test},
            y_test
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        return None, None, None

def train_etf_model(df, spy_data, features):
    try:
        if isinstance(df, dict):
            dataframes = []
            for symbol, data in df.items():
                df_copy = data.copy()
                df_copy['symbol'] = symbol
                dataframes.append(df_copy)
            df = pd.concat(dataframes, ignore_index=True)

        df, col_map = standardize_dataframe_columns(df)
        logger.info(f"Initial df shape: {df.shape}")
        
        # Convert to timezone-naive
        df['date'] = pd.to_datetime(df['date'])
        spy_data['date'] = pd.to_datetime(spy_data['date']).dt.tz_localize(None)

        df.set_index('date', inplace=True)
        spy_data.set_index('date', inplace=True)
        
        logger.info(f"Date ranges after timezone fix - df: {df.index.min()} to {df.index.max()}")
        logger.info(f"Date ranges after timezone fix - spy: {spy_data.index.min()} to {spy_data.index.max()}")

        # Align on common dates
        common_dates = df.index.intersection(spy_data.index)
        if len(common_dates) == 0:
            logger.error("No overlapping dates found between df and spy_data.")
            return None

        df = df.loc[common_dates]
        spy_data = spy_data.loc[common_dates]

        logger.info(f"Shape after date alignment: {df.shape}")
        
        # Calculate returns
        df['equity_return'] = df[col_map['close']].pct_change()
        spy_returns = spy_data['Close'].pct_change()
        df['relative_return'] = df['equity_return'] - spy_returns

        # Calculate features
        df['volume'] = df[col_map['volume']]
        df['equity_ma_5'] = df[col_map['close']].rolling(window=5).mean()
        df['equity_ma_10'] = df[col_map['close']].rolling(window=10).mean()
        df['volume_ma_5'] = df[col_map['volume']].rolling(window=5).mean()
        df['volume_ma_10'] = df[col_map['volume']].rolling(window=10).mean()

        # Technical Indicators
        df['rsi_14'] = ta.momentum.RSIIndicator(close=df[col_map['close']]).rsi()
        macd = ta.trend.MACD(close=df[col_map['close']])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(close=df[col_map['close']])
        df['bollinger_mavg'] = bb.bollinger_mavg()
        df['bollinger_hband'] = bb.bollinger_hband()
        df['bollinger_lband'] = bb.bollinger_lband()

        logger.info(f"Shape before cleaning: {df.shape}")
        nan_counts = df[features + ['relative_return']].isnull().sum()
        logger.info(f"NaN counts:\n{nan_counts}")

        # Clean data
        df_clean = df.dropna(subset=features + ['relative_return'])

        if df_clean.empty:
            logger.error("No valid samples after cleaning. Cannot train model.")
            return None

        X = df_clean[features]
        y = df_clean['relative_return']
        logger.info(f"Shape after cleaning: {df_clean.shape}")

        # Ensure there are samples to train on
        if X.shape[0] == 0:
            logger.error("No samples available to train the model after cleaning.")
            return None

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    except Exception as e:
        logger.error(f"Error in train_etf_model: {str(e)}")
        return None

    
def calculate_relative_performance(etf_data: pd.DataFrame, spy_data: pd.DataFrame) -> pd.Series:
   """Calculate ETF vs SPY performance"""
   etf_returns = etf_data['close'].pct_change()
   spy_returns = spy_data['close'].pct_change()
   return etf_returns - spy_returns

def _build_market_relations(symbols: List[str],
                          sector_map: Dict[str, str]) -> Dict[Tuple[str, str], float]:
    """
    Build market relationship strengths between symbols
    """
    try:
        relations = {}
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:  # Avoid duplicates
                    # Check if either symbol is an ETF
                    sym1_is_etf = sym1.endswith('ETF') or sym1 in ['SPY', 'QQQ', 'IWM']
                    sym2_is_etf = sym2.endswith('ETF') or sym2 in ['SPY', 'QQQ', 'IWM']
                    
                    if sym1_is_etf and sym2_is_etf:
                        # ETF-ETF relationship
                        relations[(sym1, sym2)] = 0.7
                    elif sym1_is_etf or sym2_is_etf:
                        # ETF-Stock relationship
                        relations[(sym1, sym2)] = 0.5
                    else:
                        # Stock-Stock relationship
                        sector1 = sector_map.get(sym1, 'Unknown')
                        sector2 = sector_map.get(sym2, 'Unknown')
                        
                        if sector1 == sector2:
                            relations[(sym1, sym2)] = 0.8
                        else:
                            relations[(sym1, sym2)] = 0.3
        
        return relations
        
    except Exception as e:
        logger.error(f"Error building market relations: {str(e)}")
        return {}