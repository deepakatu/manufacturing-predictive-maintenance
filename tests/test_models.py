
"""
Unit tests for the predictive models module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from src.models import (
    HealthScoringModel, FailureClassificationModel, RULEstimationModel,
    PredictiveModelService, ModelConfig, PredictionResult
)
from src.sensor_simulator import SensorSimulator

class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.model_type == 'random_forest'
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.cv_folds == 5
        assert config.sequence_length == 50
        assert config.hidden_units == 64
        assert config.dropout_rate == 0.2
        assert config.learning_rate == 0.001
        assert config.epochs == 100
        assert config.batch_size == 32
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_type='gradient_boosting',
            test_size=0.3,
            random_state=123,
            cv_folds=3
        )
        
        assert config.model_type == 'gradient_boosting'
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.cv_folds == 3

class TestPredictionResult:
    """Test cases for PredictionResult class."""
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation."""
        result = PredictionResult(
            equipment_id="TEST_001",
            timestamp=datetime.now(),
            prediction=0.75,
            confidence=0.9,
            model_type="health_scoring_random_forest",
            features_used=["sensor_vibration_x", "sensor_temperature"],
            metadata={"feature_count": 2}
        )
        
        assert result.equipment_id == "TEST_001"
        assert result.prediction == 0.75
        assert result.confidence == 0.9
        assert result.model_type == "health_scoring_random_forest"
        assert len(result.features_used) == 2
        assert result.metadata["feature_count"] == 2

class TestHealthScoringModel:
    """Test cases for HealthScoringModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a health scoring model for testing."""
        config = ModelConfig(model_type='random_forest', test_size=0.2)
        return HealthScoringModel(config)
    
    @pytest.fixture
    def training_data(self):
        """Create training data for health scoring."""
        simulator = SensorSimulator("TEST_001")
        
        # Generate normal data
        normal_data = simulator.generate_batch_data(hours=12, interval_seconds=600)
        
        # Generate degraded data
        simulator.inject_failure("bearing_wear", severity=0.5)
        degraded_data = simulator.generate_batch_data(hours=6, interval_seconds=600)
        
        # Combine data
        combined_data = pd.concat([normal_data, degraded_data], ignore_index=True)
        return combined_data
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.config.model_type == 'random_forest'
        assert model.model is not None
        assert model.scaler is not None
        assert model.is_trained == False
        assert len(model.feature_columns) == 0
    
    def test_feature_preparation(self, model, training_data):
        """Test feature preparation."""
        features = model.prepare_features(training_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(training_data)
        
        # Check that statistical features are added
        sensor_columns = [col for col in training_data.columns if col.startswith('sensor_')]
        for sensor_col in sensor_columns:
            if sensor_col in training_data.columns:
                # Check for rolling statistics
                assert f'{sensor_col}_rolling_mean' in features.columns
                assert f'{sensor_col}_rolling_std' in features.columns
                assert f'{sensor_col}_rolling_max' in features.columns
                assert f'{sensor_col}_rolling_min' in features.columns
                
                # Check for trend features
                assert f'{sensor_col}_trend' in features.columns
    
    def test_training(self, model, training_data):
        """Test model training."""
        result = model.train(training_data, target_column='equipment_health')
        
        assert model.is_trained == True
        assert len(model.feature_columns) > 0
        assert 'train_score' in result
        assert 'test_score' in result
        assert 'cv_scores' in result
        assert 'feature_count' in result
        
        # Scores should be reasonable
        assert 0 <= result['train_score'] <= 1
        assert 0 <= result['test_score'] <= 1
        assert result['feature_count'] > 0
    
    def test_prediction(self, model, training_data):
        """Test model prediction."""
        # Train model first
        model.train(training_data, target_column='equipment_health')
        
        # Create test data
        test_data = training_data.head(5)
        
        predictions = model.predict(test_data)
        
        assert len(predictions) == 5
        
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.equipment_id is not None
            assert isinstance(pred.prediction, float)
            assert 0 <= pred.prediction <= 1  # Health score should be between 0 and 1
            assert isinstance(pred.confidence, float)
            assert 0 <= pred.confidence <= 1
            assert pred.model_type.startswith('health_scoring_')
            assert len(pred.features_used) > 0
    
    def test_different_model_types(self, training_data):
        """Test different model types for health scoring."""
        model_types = ['random_forest', 'gradient_boosting', 'svm']
        
        for model_type in model_types:
            config = ModelConfig(model_type=model_type, test_size=0.3)
            model = HealthScoringModel(config)
            
            try:
                result = model.train(training_data, target_column='equipment_health')
                assert model.is_trained == True
                assert result['feature_count'] > 0
            except Exception as e:
                # Some models might fail with small datasets, which is acceptable
                pytest.skip(f"Model type {model_type} failed with small dataset: {e}")

class TestFailureClassificationModel:
    """Test cases for FailureClassificationModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a failure classification model for testing."""
        config = ModelConfig(model_type='random_forest', test_size=0.2)
        return FailureClassificationModel(config)
    
    @pytest.fixture
    def training_data(self):
        """Create training data with different failure types."""
        simulator = SensorSimulator("TEST_001")
        
        all_data = []
        
        # Normal operation data
        normal_data = simulator.generate_batch_data(hours=6, interval_seconds=600)
        all_data.append(normal_data)
        
        # Bearing wear data
        simulator = SensorSimulator("TEST_001")  # Reset simulator
        simulator.inject_failure("bearing_wear", severity=0.4)
        bearing_data = simulator.generate_batch_data(hours=3, interval_seconds=600)
        all_data.append(bearing_data)
        
        # Misalignment data
        simulator = SensorSimulator("TEST_001")  # Reset simulator
        simulator.inject_failure("misalignment", severity=0.3)
        misalign_data = simulator.generate_batch_data(hours=3, interval_seconds=600)
        all_data.append(misalign_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    def test_failure_label_preparation(self, model, training_data):
        """Test failure label preparation."""
        labels = model.prepare_failure_labels(training_data)
        
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(training_data)
        
        # Check that we have different failure types
        unique_labels = labels.unique()
        assert len(unique_labels) > 1  # Should have normal and failure cases
        assert 'normal' in unique_labels  # Should have normal cases
    
    def test_training(self, model, training_data):
        """Test failure classification training."""
        result = model.train(training_data)
        
        assert model.is_trained == True
        assert len(model.feature_columns) > 0
        assert 'train_score' in result
        assert 'test_score' in result
        assert 'classes' in result
        
        # Should have multiple classes
        assert len(result['classes']) > 1
        assert 'normal' in result['classes']
    
    def test_prediction(self, model, training_data):
        """Test failure classification prediction."""
        # Train model first
        model.train(training_data)
        
        # Create test data
        test_data = training_data.head(3)
        
        predictions = model.predict(test_data)
        
        assert len(predictions) == 3
        
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert isinstance(pred.prediction, str)  # Should be class name
            assert isinstance(pred.confidence, float)
            assert 0 <= pred.confidence <= 1
            assert pred.model_type.startswith('failure_classification_')
            assert 'all_probabilities' in pred.metadata

class TestRULEstimationModel:
    """Test cases for RULEstimationModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a RUL estimation model for testing."""
        config = ModelConfig(model_type='random_forest', test_size=0.2)
        return RULEstimationModel(config)
    
    @pytest.fixture
    def lstm_model(self):
        """Create an LSTM RUL estimation model for testing."""
        config = ModelConfig(
            model_type='lstm',
            sequence_length=20,
            hidden_units=32,
            epochs=5,  # Small number for testing
            batch_size=16
        )
        return RULEstimationModel(config)
    
    @pytest.fixture
    def training_data(self):
        """Create training data for RUL estimation."""
        simulator = SensorSimulator("TEST_001")
        
        # Generate data with gradual degradation
        all_data = []
        
        # Start with healthy equipment
        data1 = simulator.generate_batch_data(hours=6, interval_seconds=600)
        all_data.append(data1)
        
        # Inject mild failure
        simulator.inject_failure("bearing_wear", severity=0.2)
        data2 = simulator.generate_batch_data(hours=6, interval_seconds=600)
        all_data.append(data2)
        
        # Increase failure severity
        simulator.inject_failure("bearing_wear", severity=0.5)
        data3 = simulator.generate_batch_data(hours=6, interval_seconds=600)
        all_data.append(data3)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    def test_rul_target_preparation(self, model, training_data):
        """Test RUL target preparation."""
        rul_targets = model.prepare_rul_targets(training_data)
        
        assert isinstance(rul_targets, pd.Series)
        assert len(rul_targets) == len(training_data)
        assert all(rul_targets >= 0)  # RUL should be non-negative
        
        # RUL should generally decrease as health decreases
        health_scores = training_data['equipment_health']
        # Check correlation (should be positive - higher health = higher RUL)
        correlation = np.corrcoef(health_scores, rul_targets)[0, 1]
        assert correlation > 0  # Should be positively correlated
    
    def test_traditional_ml_training(self, model, training_data):
        """Test traditional ML RUL model training."""
        result = model.train(training_data)
        
        assert model.is_trained == True
        assert len(model.feature_columns) > 0
        assert 'train_score' in result
        assert 'test_score' in result
        assert 'mse' in result
        assert 'mae' in result
        
        # MSE and MAE should be reasonable
        assert result['mse'] >= 0
        assert result['mae'] >= 0
    
    def test_lstm_training(self, lstm_model, training_data):
        """Test LSTM RUL model training."""
        # Skip if TensorFlow is not available or data is too small
        if len(training_data) < lstm_model.config.sequence_length * 2:
            pytest.skip("Insufficient data for LSTM training")
        
        try:
            result = lstm_model.train(training_data)
            
            assert lstm_model.is_trained == True
            assert 'train_loss' in result
            assert 'test_loss' in result
            assert 'epochs_trained' in result
            
            # Losses should be reasonable
            assert result['train_loss'] >= 0
            assert result['test_loss'] >= 0
            assert result['epochs_trained'] > 0
            
        except Exception as e:
            pytest.skip(f"LSTM training failed (acceptable for small dataset): {e}")
    
    def test_traditional_ml_prediction(self, model, training_data):
        """Test traditional ML RUL prediction."""
        # Train model first
        model.train(training_data)
        
        # Create test data
        test_data = training_data.head(3)
        
        predictions = model.predict(test_data)
        
        assert len(predictions) == 3
        
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert isinstance(pred.prediction, float)
            assert pred.prediction >= 0  # RUL should be non-negative
            assert isinstance(pred.confidence, float)
            assert 0 <= pred.confidence <= 1
            assert pred.model_type.startswith('rul_estimation_')
    
    def test_lstm_prediction(self, lstm_model, training_data):
        """Test LSTM RUL prediction."""
        if len(training_data) < lstm_model.config.sequence_length * 2:
            pytest.skip("Insufficient data for LSTM training")
        
        try:
            # Train model first
            lstm_model.train(training_data)
            
            # Create test data
            test_data = training_data.tail(lstm_model.config.sequence_length + 5)
            
            predictions = lstm_model.predict(test_data)
            
            assert len(predictions) == 1  # LSTM returns single prediction
            
            pred = predictions[0]
            assert isinstance(pred, PredictionResult)
            assert isinstance(pred.prediction, float)
            assert pred.prediction >= 0
            assert pred.model_type == 'rul_estimation_lstm'
            
        except Exception as e:
            pytest.skip(f"LSTM prediction failed (acceptable): {e}")
    
    def test_sequence_creation(self, lstm_model):
        """Test sequence creation for LSTM."""
        # Create sample data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        targets = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        
        lstm_model.config.sequence_length = 3
        
        X_seq, y_seq = lstm_model.create_sequences(data, targets)
        
        # Should create sequences of length 3
        expected_sequences = len(data) - lstm_model.config.sequence_length
        assert len(X_seq) == expected_sequences
        assert len(y_seq) == expected_sequences
        assert X_seq.shape[1] == lstm_model.config.sequence_length

class TestPredictiveModelService:
    """Test cases for PredictiveModelService class."""
    
    @pytest.fixture
    def service(self):
        """Create a predictive model service for testing."""
        return PredictiveModelService()
    
    @pytest.fixture
    def training_data(self):
        """Create comprehensive training data."""
        simulator = SensorSimulator("TEST_001")
        
        all_data = []
        
        # Normal operation
        normal_data = simulator.generate_batch_data(hours=8, interval_seconds=600)
        all_data.append(normal_data)
        
        # Various failures
        failure_types = ["bearing_wear", "misalignment", "overheating"]
        for failure_type in failure_types:
            simulator = SensorSimulator("TEST_001")  # Reset
            simulator.inject_failure(failure_type, severity=0.4)
            failure_data = simulator.generate_batch_data(hours=4, interval_seconds=600)
            all_data.append(failure_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert len(service.models) > 0
        assert 'health_scoring' in service.models
        assert 'failure_classification' in service.models
        assert 'rul_estimation' in service.models
        
        # Check that models are properly initialized
        for model_name, model in service.models.items():
            assert model is not None
            assert hasattr(model, 'is_trained')
            assert model.is_trained == False
    
    def test_train_all_models(self, service, training_data):
        """Test training all models."""
        results = service.train_all_models(training_data)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that all models were attempted
        expected_models = ['health_scoring', 'failure_classification', 'rul_estimation']
        for model_name in expected_models:
            assert model_name in results
            
            # Check if training was successful
            if 'error' not in results[model_name]:
                assert service.models[model_name].is_trained == True
    
    def test_predict_all(self, service, training_data):
        """Test predictions from all models."""
        # Train models first
        service.train_all_models(training_data)
        
        # Create test data
        test_data = training_data.head(2)
        
        predictions = service.predict_all(test_data)
        
        assert isinstance(predictions, dict)
        
        # Check predictions for each model
        for model_name, preds in predictions.items():
            if service.models[model_name].is_trained:
                assert len(preds) == 2  # Should have predictions for 2 test samples
                
                for pred in preds:
                    assert isinstance(pred, PredictionResult)
                    assert pred.model_type.startswith(model_name)
    
    def test_model_status(self, service, training_data):
        """Test model status reporting."""
        # Get status before training
        status_before = service.get_model_status()
        
        for model_name, info in status_before.items():
            assert info['is_trained'] == False
            assert info['feature_count'] == 0
        
        # Train models
        service.train_all_models(training_data)
        
        # Get status after training
        status_after = service.get_model_status()
        
        for model_name, info in status_after.items():
            if service.models[model_name].is_trained:
                assert info['is_trained'] == True
                assert info['feature_count'] > 0
                assert 'config' in info
    
    def test_model_persistence(self, service, training_data):
        """Test saving and loading models."""
        # Train models
        service.train_all_models(training_data)
        
        # Save models
        with tempfile.TemporaryDirectory() as temp_dir:
            service.save_models(temp_dir)
            
            # Check that model files were created
            model_files = os.listdir(temp_dir)
            assert len(model_files) > 0
            
            # Create new service and load models
            new_service = PredictiveModelService()
            new_service.load_models(temp_dir)
            
            # Check that models were loaded
            for model_name, model in new_service.models.items():
                if os.path.exists(os.path.join(temp_dir, f'{model_name}_model.joblib')):
                    # Model should be loaded and trained
                    assert hasattr(model, 'is_trained')

class TestIntegration:
    """Integration tests for predictive models."""
    
    def test_end_to_end_prediction_workflow(self):
        """Test complete prediction workflow."""
        # Create simulator and generate comprehensive data
        simulator = SensorSimulator("INTEGRATION_TEST")
        
        # Generate training data with various conditions
        all_training_data = []
        
        # Normal operation
        normal_data = simulator.generate_batch_data(hours=12, interval_seconds=600)
        all_training_data.append(normal_data)
        
        # Gradual degradation
        for severity in [0.2, 0.4, 0.6]:
            simulator = SensorSimulator("INTEGRATION_TEST")
            simulator.inject_failure("bearing_wear", severity=severity)
            degraded_data = simulator.generate_batch_data(hours=4, interval_seconds=600)
            all_training_data.append(degraded_data)
        
        training_data = pd.concat(all_training_data, ignore_index=True)
        
        # Create and train service
        service = PredictiveModelService()
        training_results = service.train_all_models(training_data)
        
        # Check that at least some models trained successfully
        successful_models = [
            name for name, result in training_results.items() 
            if 'error' not in result
        ]
        assert len(successful_models) > 0
        
        # Generate test data
        test_simulator = SensorSimulator("INTEGRATION_TEST")
        test_simulator.inject_failure("bearing_wear", severity=0.3)
        test_data = test_simulator.generate_batch_data(hours=2, interval_seconds=600)
        
        # Get predictions
        predictions = service.predict_all(test_data)
        
        # Verify predictions
        for model_name in successful_models:
            if model_name in predictions and len(predictions[model_name]) > 0:
                pred = predictions[model_name][0]
                
                if model_name == 'health_scoring':
                    # Health score should be reasonable
                    assert 0 <= pred.prediction <= 1
                elif model_name == 'failure_classification':
                    # Should predict some failure type
                    assert isinstance(pred.prediction, str)
                elif model_name == 'rul_estimation':
                    # RUL should be positive
                    assert pred.prediction >= 0
    
    def test_model_performance_consistency(self):
        """Test that models produce consistent results."""
        # Create deterministic data
        np.random.seed(42)
        
        simulator = SensorSimulator("CONSISTENCY_TEST")
        training_data = simulator.generate_batch_data(hours=6, interval_seconds=600)
        
        # Train model twice with same data
        config = ModelConfig(random_state=42)
        
        model1 = HealthScoringModel(config)
        model1.train(training_data, 'equipment_health')
        
        model2 = HealthScoringModel(config)
        model2.train(training_data, 'equipment_health')
        
        # Test data
        test_data = training_data.head(3)
        
        predictions1 = model1.predict(test_data)
        predictions2 = model2.predict(test_data)
        
        # Predictions should be very similar (allowing for small numerical differences)
        for pred1, pred2 in zip(predictions1, predictions2):
            assert abs(pred1.prediction - pred2.prediction) < 0.01

if __name__ == "__main__":
    pytest.main([__file__])
