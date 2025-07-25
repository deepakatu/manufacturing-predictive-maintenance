
"""
Predictive Models for Manufacturing Equipment Health and Failure Prediction.
Implements various ML models for health scoring, RUL estimation, and failure classification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import os
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for predictive models."""
    model_type: str = 'random_forest'
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    sequence_length: int = 50
    hidden_units: int = 64
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32

@dataclass
class PredictionResult:
    """Result of model prediction."""
    equipment_id: str
    timestamp: datetime
    prediction: Union[float, int, str]
    confidence: float
    model_type: str
    features_used: List[str]
    metadata: Dict[str, Any]

class HealthScoringModel:
    """Model for predicting equipment health scores (0-1 scale)."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Initialize model based on type
        if config.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=config.random_state,
                n_jobs=-1
            )
        elif config.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=config.random_state
            )
        elif config.model_type == 'svm':
            self.model = SVR(kernel='rbf')
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=config.random_state
            )
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for health scoring."""
        features = data.copy()
        
        # Select sensor columns
        sensor_cols = [col for col in features.columns if col.startswith('sensor_')]
        
        # Add statistical features
        for col in sensor_cols:
            if col in features.columns:
                # Rolling statistics
                features[f'{col}_rolling_mean'] = features[col].rolling(window=10, min_periods=1).mean()
                features[f'{col}_rolling_std'] = features[col].rolling(window=10, min_periods=1).std()
                features[f'{col}_rolling_max'] = features[col].rolling(window=10, min_periods=1).max()
                features[f'{col}_rolling_min'] = features[col].rolling(window=10, min_periods=1).min()
                
                # Trend features
                features[f'{col}_trend'] = features[col].diff()
                features[f'{col}_trend_ma'] = features[f'{col}_trend'].rolling(window=5, min_periods=1).mean()
        
        # Add operating condition features
        operating_cols = [col for col in features.columns if col.startswith('operating_')]
        
        # Add time-based features if timestamp is available
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['month'] = features['timestamp'].dt.month
        
        # Fill NaN values
        features = features.fillna(method='forward').fillna(method='backward').fillna(0)
        
        return features
    
    def train(self, data: pd.DataFrame, target_column: str = 'equipment_health'):
        """Train the health scoring model."""
        logger.info(f"Training health scoring model ({self.config.model_type})")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Select feature columns (exclude non-feature columns)
        exclude_cols = ['timestamp', 'equipment_id', target_column, 'active_failures']
        self.feature_columns = [col for col in features.columns if col not in exclude_cols]
        
        if len(self.feature_columns) == 0:
            raise ValueError("No valid feature columns found")
        
        # Prepare training data
        X = features[self.feature_columns]
        y = features[target_column] if target_column in features.columns else np.ones(len(features))
        
        # Handle missing target values
        valid_indices = ~pd.isna(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) == 0:
            raise ValueError("No valid training samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=self.config.cv_folds, scoring='r2'
        )
        
        logger.info(f"Training completed:")
        logger.info(f"  Train R²: {train_score:.4f}")
        logger.info(f"  Test R²: {test_score:.4f}")
        logger.info(f"  CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'cv_scores': cv_scores,
            'feature_count': len(self.feature_columns)
        }
    
    def predict(self, data: pd.DataFrame) -> List[PredictionResult]:
        """Predict health scores for new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Extract features
        X = features[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Calculate confidence (for tree-based models)
        if hasattr(self.model, 'predict_proba'):
            # For classification models
            probabilities = self.model.predict_proba(X_scaled)
            confidences = np.max(probabilities, axis=1)
        elif hasattr(self.model, 'estimators_'):
            # For ensemble models, use prediction variance
            individual_predictions = np.array([
                estimator.predict(X_scaled) for estimator in self.model.estimators_
            ])
            prediction_std = np.std(individual_predictions, axis=0)
            confidences = 1.0 / (1.0 + prediction_std)  # Higher std = lower confidence
        else:
            # Default confidence
            confidences = np.ones(len(predictions)) * 0.8
        
        # Create results
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            result = PredictionResult(
                equipment_id=data.iloc[i].get('equipment_id', 'unknown'),
                timestamp=datetime.now(),
                prediction=float(np.clip(pred, 0, 1)),  # Ensure health score is between 0 and 1
                confidence=float(conf),
                model_type=f'health_scoring_{self.config.model_type}',
                features_used=self.feature_columns,
                metadata={
                    'feature_count': len(self.feature_columns),
                    'raw_prediction': float(pred)
                }
            )
            results.append(result)
        
        return results

class FailureClassificationModel:
    """Model for predicting equipment failure types."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
        # Initialize model
        if config.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=config.random_state,
                n_jobs=-1
            )
        elif config.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=config.random_state,
                max_iter=1000
            )
        elif config.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=config.random_state
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=config.random_state
            )
    
    def prepare_failure_labels(self, data: pd.DataFrame) -> pd.Series:
        """Prepare failure labels from active_failures column."""
        labels = []
        
        for _, row in data.iterrows():
            active_failures = row.get('active_failures', '')
            
            if pd.isna(active_failures) or active_failures == '':
                labels.append('normal')
            else:
                # Take the first failure if multiple
                failure_list = str(active_failures).split(',')
                labels.append(failure_list[0].strip())
        
        return pd.Series(labels)
    
    def train(self, data: pd.DataFrame):
        """Train the failure classification model."""
        logger.info(f"Training failure classification model ({self.config.model_type})")
        
        # Prepare features
        health_model = HealthScoringModel(self.config)
        features = health_model.prepare_features(data)
        
        # Select feature columns
        exclude_cols = ['timestamp', 'equipment_id', 'equipment_health', 'active_failures']
        self.feature_columns = [col for col in features.columns if col not in exclude_cols]
        
        # Prepare labels
        y = self.prepare_failure_labels(data)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Prepare training data
        X = features[self.feature_columns]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        class_names = self.label_encoder.classes_
        
        logger.info(f"Training completed:")
        logger.info(f"  Train Accuracy: {train_score:.4f}")
        logger.info(f"  Test Accuracy: {test_score:.4f}")
        logger.info(f"  Classes: {list(class_names)}")
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'classes': list(class_names),
            'feature_count': len(self.feature_columns)
        }
    
    def predict(self, data: pd.DataFrame) -> List[PredictionResult]:
        """Predict failure types for new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        health_model = HealthScoringModel(self.config)
        features = health_model.prepare_features(data)
        
        # Extract features
        X = features[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode predictions
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        confidences = np.max(probabilities, axis=1)
        
        # Create results
        results = []
        for i, (pred_class, conf) in enumerate(zip(predicted_classes, confidences)):
            result = PredictionResult(
                equipment_id=data.iloc[i].get('equipment_id', 'unknown'),
                timestamp=datetime.now(),
                prediction=pred_class,
                confidence=float(conf),
                model_type=f'failure_classification_{self.config.model_type}',
                features_used=self.feature_columns,
                metadata={
                    'all_probabilities': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.label_encoder.classes_, probabilities[i])
                    }
                }
            )
            results.append(result)
        
        return results

class RULEstimationModel:
    """Model for Remaining Useful Life (RUL) estimation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        if config.model_type == 'lstm':
            self._build_lstm_model()
        else:
            # Traditional ML model
            if config.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=config.random_state
                )
            elif config.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=config.random_state
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=config.random_state
                )
    
    def _build_lstm_model(self):
        """Build LSTM model for time series RUL prediction."""
        self.model = Sequential([
            LSTM(self.config.hidden_units, return_sequences=True, 
                 input_shape=(self.config.sequence_length, 1)),
            Dropout(self.config.dropout_rate),
            LSTM(self.config.hidden_units // 2, return_sequences=False),
            Dropout(self.config.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def prepare_rul_targets(self, data: pd.DataFrame) -> pd.Series:
        """Prepare RUL targets based on equipment health degradation."""
        # Simple RUL estimation based on health score
        # In practice, this would be based on actual failure times
        
        rul_values = []
        for _, row in data.iterrows():
            health = row.get('equipment_health', 1.0)
            
            # Estimate RUL based on health degradation rate
            # Assume linear degradation for simplicity
            if health > 0.8:
                rul = 1000  # High health = long RUL
            elif health > 0.6:
                rul = 500
            elif health > 0.4:
                rul = 200
            elif health > 0.2:
                rul = 50
            else:
                rul = 10  # Low health = short RUL
            
            # Add some noise
            rul += np.random.normal(0, rul * 0.1)
            rul_values.append(max(0, rul))
        
        return pd.Series(rul_values)
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(len(data) - self.config.sequence_length):
            X.append(data[i:(i + self.config.sequence_length)])
            y.append(targets[i + self.config.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame):
        """Train the RUL estimation model."""
        logger.info(f"Training RUL estimation model ({self.config.model_type})")
        
        # Prepare features
        health_model = HealthScoringModel(self.config)
        features = health_model.prepare_features(data)
        
        # Prepare RUL targets
        rul_targets = self.prepare_rul_targets(data)
        
        if self.config.model_type == 'lstm':
            # LSTM training
            # Use equipment health as the main feature for sequence prediction
            health_values = features['equipment_health'].fillna(1.0).values
            rul_values = rul_targets.values
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(health_values.reshape(-1, 1), rul_values)
            
            if len(X_seq) == 0:
                raise ValueError("Insufficient data for sequence creation")
            
            # Split data
            split_idx = int(len(X_seq) * (1 - self.config.test_size))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # Train LSTM
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)[0]
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)[0]
            
            logger.info(f"LSTM Training completed:")
            logger.info(f"  Train Loss (MSE): {train_loss:.4f}")
            logger.info(f"  Test Loss (MSE): {test_loss:.4f}")
            
            self.is_trained = True
            
            return {
                'train_loss': train_loss,
                'test_loss': test_loss,
                'epochs_trained': len(history.history['loss'])
            }
        
        else:
            # Traditional ML training
            exclude_cols = ['timestamp', 'equipment_id', 'equipment_health', 'active_failures']
            self.feature_columns = [col for col in features.columns if col not in exclude_cols]
            
            X = features[self.feature_columns].fillna(0)
            y = rul_targets
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"Training completed:")
            logger.info(f"  Train R²: {train_score:.4f}")
            logger.info(f"  Test R²: {test_score:.4f}")
            logger.info(f"  Test MSE: {mse:.4f}")
            logger.info(f"  Test MAE: {mae:.4f}")
            
            self.is_trained = True
            
            return {
                'train_score': train_score,
                'test_score': test_score,
                'mse': mse,
                'mae': mae
            }
    
    def predict(self, data: pd.DataFrame) -> List[PredictionResult]:
        """Predict RUL for new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.config.model_type == 'lstm':
            # LSTM prediction
            health_values = data['equipment_health'].fillna(1.0).values
            
            if len(health_values) < self.config.sequence_length:
                # Pad with last known value
                padding = [health_values[-1]] * (self.config.sequence_length - len(health_values))
                health_values = np.concatenate([padding, health_values])
            
            # Take last sequence
            X_seq = health_values[-self.config.sequence_length:].reshape(1, self.config.sequence_length, 1)
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
            
            results = [PredictionResult(
                equipment_id=data.iloc[-1].get('equipment_id', 'unknown'),
                timestamp=datetime.now(),
                prediction=float(max(0, prediction)),
                confidence=0.8,  # Default confidence for LSTM
                model_type=f'rul_estimation_lstm',
                features_used=['equipment_health_sequence'],
                metadata={'sequence_length': self.config.sequence_length}
            )]
            
        else:
            # Traditional ML prediction
            health_model = HealthScoringModel(self.config)
            features = health_model.prepare_features(data)
            
            X = features[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            predictions = self.model.predict(X_scaled)
            
            # Calculate confidence
            if hasattr(self.model, 'estimators_'):
                individual_predictions = np.array([
                    estimator.predict(X_scaled) for estimator in self.model.estimators_
                ])
                prediction_std = np.std(individual_predictions, axis=0)
                confidences = 1.0 / (1.0 + prediction_std / np.mean(individual_predictions, axis=0))
            else:
                confidences = np.ones(len(predictions)) * 0.8
            
            results = []
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                result = PredictionResult(
                    equipment_id=data.iloc[i].get('equipment_id', 'unknown'),
                    timestamp=datetime.now(),
                    prediction=float(max(0, pred)),
                    confidence=float(conf),
                    model_type=f'rul_estimation_{self.config.model_type}',
                    features_used=self.feature_columns,
                    metadata={'feature_count': len(self.feature_columns)}
                )
                results.append(result)
        
        return results

class PredictiveModelService:
    """Service for managing multiple predictive models."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.models = {}
        self.model_configs = {}
        
        # Initialize models
        for model_name, model_config in self.config.get('models', {}).items():
            config_obj = ModelConfig(**model_config)
            self.model_configs[model_name] = config_obj
            
            if model_name == 'health_scoring':
                self.models[model_name] = HealthScoringModel(config_obj)
            elif model_name == 'failure_classification':
                self.models[model_name] = FailureClassificationModel(config_obj)
            elif model_name == 'rul_estimation':
                self.models[model_name] = RULEstimationModel(config_obj)
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file."""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        return {
            'models': {
                'health_scoring': {
                    'model_type': 'random_forest',
                    'test_size': 0.2,
                    'random_state': 42
                },
                'failure_classification': {
                    'model_type': 'random_forest',
                    'test_size': 0.2,
                    'random_state': 42
                },
                'rul_estimation': {
                    'model_type': 'random_forest',
                    'test_size': 0.2,
                    'random_state': 42
                }
            }
        }
    
    def train_all_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train all models on the provided data."""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name} model...")
                
                if model_name == 'health_scoring':
                    result = model.train(training_data, 'equipment_health')
                else:
                    result = model.train(training_data)
                
                results[model_name] = result
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict_all(self, data: pd.DataFrame) -> Dict[str, List[PredictionResult]]:
        """Get predictions from all trained models."""
        predictions = {}
        
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    predictions[model_name] = model.predict(data)
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {str(e)}")
                    predictions[model_name] = []
            else:
                logger.warning(f"Model {model_name} is not trained")
                predictions[model_name] = []
        
        return predictions
    
    def save_models(self, directory: str):
        """Save all trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model.is_trained:
                model_path = os.path.join(directory, f'{model_name}_model.joblib')
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} model to {model_path}")
    
    def load_models(self, directory: str):
        """Load trained models from disk."""
        for model_name in self.models.keys():
            model_path = os.path.join(directory, f'{model_name}_model.joblib')
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {str(e)}")
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models."""
        status = {}
        
        for model_name, model in self.models.items():
            status[model_name] = {
                'is_trained': model.is_trained,
                'model_type': self.model_configs[model_name].model_type,
                'feature_count': len(getattr(model, 'feature_columns', [])),
                'config': self.model_configs[model_name].__dict__
            }
        
        return status

def main():
    """Example usage of the predictive models."""
    from sensor_simulator import SensorSimulator
    
    # Generate training data
    simulator = SensorSimulator("PUMP_001")
    
    # Generate normal operation data
    normal_data = simulator.generate_batch_data(hours=48, interval_seconds=300)
    
    # Generate data with failures
    simulator.inject_failure("bearing_wear", severity=0.3)
    failure_data = simulator.generate_batch_data(hours=24, interval_seconds=300)
    
    # Combine data
    training_data = pd.concat([normal_data, failure_data], ignore_index=True)
    
    print(f"Generated {len(training_data)} training samples")
    
    # Create and train models
    service = PredictiveModelService()
    training_results = service.train_all_models(training_data)
    
    print("\nTraining Results:")
    for model_name, result in training_results.items():
        print(f"{model_name}: {result}")
    
    # Test predictions
    test_data = simulator.generate_batch_data(hours=2, interval_seconds=300)
    predictions = service.predict_all(test_data)
    
    print("\nPrediction Results:")
    for model_name, preds in predictions.items():
        if preds:
            print(f"\n{model_name}:")
            for pred in preds[:3]:  # Show first 3 predictions
                print(f"  Prediction: {pred.prediction}, Confidence: {pred.confidence:.3f}")
    
    # Model status
    status = service.get_model_status()
    print("\nModel Status:")
    for model_name, info in status.items():
        print(f"{model_name}: Trained={info['is_trained']}, Features={info['feature_count']}")

if __name__ == "__main__":
    main()
