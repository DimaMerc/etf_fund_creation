
# options_model.py
import tensorflow as keras
import numpy as np
import pandas as pd
import logging
import keras
import tensorflow as tf
import traceback
logger = logging.getLogger(__name__)

class OptionsPricingMLP:
    def __init__(self):
        self.model = self._build_model()
        self.is_trained = False  # Track training status

    @staticmethod
    def get_feature_columns():
        """Define all required and optional features for the model"""
        return [
            # Required features
            'moneyness',           # S/K
            'timeToExpiration',    # T
            'impliedVolatility',   # σ
            'optionType',          # Call/Put indicator
            'bidAskSpread',        # Current spread
            'volume',              # Trading volume
            'openInterest',        # Open interest
            'historicalVolatility' # Historical σ
        ]
        
    @staticmethod
    def get_required_features():
        """For backwards compatibility - returns same as get_feature_columns"""
        return OptionsPricingMLP.get_feature_columns()
        
    

    def _build_model(self):
        """Build model with numerical stability improvements"""
        input_layer = keras.layers.Input(shape=(len(self.get_required_features()),))
        
        # Add batch normalization at input
        x = keras.layers.BatchNormalization()(input_layer)
        
        # First dense block
        x = keras.layers.Dense(400, kernel_initializer='he_normal')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Second dense block
        x = keras.layers.Dense(200, kernel_initializer='he_normal')(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Output layers with constraints
        bid_output = keras.layers.Dense(1, activation='sigmoid', name='bid',
                                    kernel_constraint=tf.keras.constraints.NonNeg())(x)
        ask_output = keras.layers.Dense(1, activation='sigmoid', name='ask',
                                    kernel_constraint=tf.keras.constraints.NonNeg())(x)
        
        model = keras.Model(inputs=input_layer, outputs=[bid_output, ask_output])
        return model
    
    def _prepare_features(self, X):
        """Prepare features for training"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.asarray(X, dtype=np.float32)
    
    def _prepare_model_features(self, option, current_price, volatility_window=20):
        """Prepare features for options model prediction with enhanced feature calculation"""
        try:
            # Convert option data to proper format
            expiry = pd.to_datetime(option['expirationDate'])
            days_to_expiry = (expiry - pd.Timestamp.now().tz_localize(None)).days
            
            # Calculate bid-ask spread
            bid = float(option.get('bid', 0))
            ask = float(option.get('ask', 0))
            mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            spread = (ask - bid) / mid_price if mid_price > 0 else 0
            
            # Calculate historical volatility if price history available
            hist_vol = 0.3  # Default value
            if hasattr(option, 'price_history') and len(option.price_history) >= volatility_window:
                returns = option.price_history.pct_change().dropna()
                hist_vol = returns.std() * np.sqrt(252)

            features = pd.DataFrame([{
                'moneyness': current_price / float(option['strike']),
                'timeToExpiration': days_to_expiry / 365.0,
                'impliedVolatility': float(option.get('impliedVolatility', hist_vol)),
                'optionType': 1 if str(option['optionType']).lower() == 'call' else 0,
                'bidAskSpread': spread,
                'volume': float(option.get('volume', 0)),
                'openInterest': float(option.get('openInterest', 0)),
                'historicalVolatility': hist_vol
            }])
            
            logger.debug(f"Prepared features for option: {option['strike']} {option['optionType']}")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing model features: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def predict(self, X):
        """Make predictions with improved numerical stability"""
        try:
            if not self.is_trained or not hasattr(self, 'scaling_params'):
                logger.warning("Model not ready for predictions")
                return np.array([]), np.array([])
                
            # Verify we have all required features
            required_features = self.get_required_features()
            if not all(feat in X.columns for feat in required_features):
                missing = [feat for feat in required_features if feat not in X.columns]
                logger.error(f"Missing prediction features: {missing}")
                return np.array([]), np.array([])
            
            # Prepare features
            X = X[required_features]
            X = np.asarray(X, dtype=np.float32)
            
            # Clip and scale features
            X = np.clip(X, -1e6, 1e6)
            X_scaled = np.clip(
                (X - self.scaling_params['X_mean']) / self.scaling_params['X_std'],
                -10, 10
            )
            
            # Make predictions
            predictions = self.model.predict(X_scaled, verbose=0)
            
            # Unscale predictions
            if isinstance(predictions, (list, tuple)):
                bid_pred = predictions[0] * self.scaling_params['y_bid_max']
                ask_pred = predictions[1] * self.scaling_params['y_ask_max']
            else:
                bid_pred = predictions * self.scaling_params['y_bid_max']
                ask_pred = bid_pred * 1.01  # Small spread if single output
            
            # Ensure ask >= bid
            ask_pred = np.maximum(ask_pred, bid_pred * 1.001)
            
            # Log prediction ranges
            logger.debug(f"Prediction ranges - Bid: [{bid_pred.min():.2f}, {bid_pred.max():.2f}], " +
                        f"Ask: [{ask_pred.min():.2f}, {ask_pred.max():.2f}]")
            
            return bid_pred, ask_pred
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return np.array([]), np.array([])
        
   
    
    
    def train(self, X_train, y_train):
        """Train the model with enhanced numerical stability"""
        try:
            if X_train is None or y_train is None:
                logger.error("Training data is None")
                return []
                
            logger.info("\nStarting model training...")
            logger.info(f"Training data shape: {X_train.shape}")
            
            # 1. Data validation and cleaning
            valid_rows = np.all(np.isfinite(X_train), axis=1)
            X_train = X_train[valid_rows]
            y_train['bid'] = y_train['bid'][valid_rows]
            y_train['ask'] = y_train['ask'][valid_rows]
            
            logger.info(f"Shape after cleaning: {X_train.shape}")
            
            # 2. Prepare and scale features
            X = X_train[self.get_required_features()].values
            X = np.clip(X, -1e6, 1e6)  # Clip extreme values
            
            eps = 1e-8
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + eps
            X_scaled = np.clip((X - X_mean) / X_std, -10, 10)
            
            # Scale targets individually
            y_bid = np.array(y_train['bid'])
            y_ask = np.array(y_train['ask'])
            
            # Ensure positive values for bid/ask
            y_bid = np.maximum(y_bid, 0.0)
            y_ask = np.maximum(y_ask, 0.0)
            
            # Log value ranges
            logger.info("\nValue ranges:")
            logger.info(f"Bid range: {y_bid.min():.2f} to {y_bid.max():.2f}")
            logger.info(f"Ask range: {y_ask.min():.2f} to {y_ask.max():.2f}")
            
            # Scale targets to [0, 1] range for better stability
            y_bid_max = np.maximum(y_bid.max(), eps)
            y_ask_max = np.maximum(y_ask.max(), eps)
            
            y_scaled = {
                'bid': y_bid / y_bid_max,
                'ask': y_ask / y_ask_max
            }
            
            # 3. Model compilation with updated loss
            class BoundedMSE(tf.keras.losses.Loss):
                def __init__(self, name='bounded_mse'):
                    super().__init__(name=name)
                    self.mse = tf.keras.losses.MeanSquaredError()
                    
                def call(self, y_true, y_pred):
                    mse = self.mse(y_true, y_pred)
                    return tf.clip_by_value(mse, 0, 1000)
                    
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss={'bid': BoundedMSE(), 'ask': BoundedMSE()}
            )
            
            # 4. Training setup
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='best_model.weights.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )
            ]
            
            # 5. Training phases
            best_val_loss = float('inf')
            best_weights = None
            
            lr_schedule = [
                {'lr': 1e-3, 'epochs': 20},
                {'lr': 1e-4, 'epochs': 15},
                {'lr': 1e-5, 'epochs': 10}
            ]
            
            for i, phase in enumerate(lr_schedule):
                logger.info(f"\nTraining phase {i+1}")
                logger.info(f"Learning rate: {phase['lr']}")
                
                self.model.optimizer.learning_rate.assign(phase['lr'])
                
                history = self.model.fit(
                    X_scaled,
                    {'bid': y_scaled['bid'], 'ask': y_scaled['ask']},
                    validation_split=0.2,
                    batch_size=32,
                    epochs=phase['epochs'],
                    callbacks=callbacks,
                    verbose=1
                )
                
                val_loss = history.history['val_loss'][-1]
                if not np.isnan(val_loss):
                    logger.info(f"Phase {i+1} val_loss: {val_loss:.6f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self.model.get_weights()
                        # Store scaling parameters
                        self.scaling_params = {
                            'X_mean': X_mean,
                            'X_std': X_std,
                            'y_bid_max': y_bid_max,
                            'y_ask_max': y_ask_max
                        }
                        logger.info(f"New best val_loss: {val_loss:.6f}")
                else:
                    logger.warning(f"Phase {i+1} produced NaN loss")
                    
            if best_weights is not None:
                logger.info("\nTraining completed successfully")
                logger.info(f"Best validation loss: {best_val_loss:.6f}")
                self.model.set_weights(best_weights)
                self.is_trained = True
                return True
            else:
                logger.error("Training failed - no valid weights found")
                return False
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        
    def _validate_training_data(self, X, y):
        """Validate training data format and dimensions"""
        try:
            if not isinstance(y, dict) or 'bid' not in y or 'ask' not in y:
                logger.error("y_train must be a dictionary with 'bid' and 'ask' keys")
                return False
                
            if len(X) != len(y['bid']) or len(X) != len(y['ask']):
                logger.error("X and y dimensions do not match")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating training data: {str(e)}")
            return False
    
    
    
    def _prepare_targets(self, y):
        """Prepare targets for training"""
        return {
            'bid': np.asarray(y['bid'], dtype=np.float32),
            'ask': np.asarray(y['ask'], dtype=np.float32)
        }
    
    def _get_callbacks(self, phase_name):
        """Get callbacks for training phase"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2
            )
        ]
    
    def _log_phase_results(self, history, phase_name):
        """Log results for training phase"""
        logger.info(f"\n{phase_name} phase results:")
        logger.info(f"Final loss: {history['loss'][-1]:.6f}")
        if 'val_loss' in history:
            logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")
        
        for metric in history:
            if metric not in ['loss', 'val_loss']:
                logger.info(f"Final {metric}: {history[metric][-1]:.6f}")


    