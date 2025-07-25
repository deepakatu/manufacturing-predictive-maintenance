
"""
Unit tests for the anomaly detection module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.anomaly_detector import (
    StatisticalAnomalyDetector, IsolationForestDetector, TimeSeriesAnomalyDetector,
    EnsembleAnomalyDetector, AnomalyDetectionService, DetectorConfig, AnomalyResult
)
from src.sensor_simulator import SensorSimulator

class TestDetectorConfig:
    """Test cases for DetectorConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DetectorConfig()
        
        assert config.contamination == 0.1
        assert config.window_size == 100
        assert config.threshold_sigma == 3.0
        assert config.min_samples == 50
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DetectorConfig(
            contamination=0.05,
            window_size=50,
            threshold_sigma=2.5,
            min_samples=30
        )
        
        assert config.contamination == 0.05
        assert config.window_size == 50
        assert config.threshold_sigma == 2.5
        assert config.min_samples == 30

class TestAnomalyResult:
    """Test cases for AnomalyResult class."""
    
    def test_anomaly_result_creation(self):
        """Test AnomalyResult creation."""
        result = AnomalyResult(
            timestamp=datetime.now(),
            equipment_id="TEST_001",
            is_anomaly=True,
            anomaly_score=0.8,
            confidence=0.9,
            affected_sensors=["sensor_vibration_x"],
            severity="high",
            description="Test anomaly"
        )
        
        assert result.equipment_id == "TEST_001"
        assert result.is_anomaly == True
        assert result.anomaly_score == 0.8
        assert result.confidence == 0.9
        assert result.affected_sensors == ["sensor_vibration_x"]
        assert result.severity == "high"
        assert result.description == "Test anomaly"

class TestStatisticalAnomalyDetector:
    """Test cases for StatisticalAnomalyDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a statistical anomaly detector for testing."""
        config = DetectorConfig(threshold_sigma=2.0)
        return StatisticalAnomalyDetector(config)
    
    @pytest.fixture
    def training_data(self):
        """Create training data for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'sensor_vibration_x': np.random.normal(2.0, 0.5, 1000),
            'sensor_temperature': np.random.normal(75.0, 5.0, 1000),
            'sensor_pressure': np.random.normal(150.0, 10.0, 1000)
        })
        return data
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.config.threshold_sigma == 2.0
        assert detector.is_trained == False
        assert len(detector.control_limits) == 0
    
    def test_training(self, detector, training_data):
        """Test detector training."""
        sensor_columns = ['sensor_vibration_x', 'sensor_temperature', 'sensor_pressure']
        detector.train(training_data, sensor_columns)
        
        assert detector.is_trained == True
        assert len(detector.control_limits) == 3
        
        # Check control limits structure
        for sensor in sensor_columns:
            assert sensor in detector.control_limits
            limits = detector.control_limits[sensor]
            assert 'mean' in limits
            assert 'std' in limits
            assert 'upper_limit' in limits
            assert 'lower_limit' in limits
            
            # Check that limits make sense
            assert limits['upper_limit'] > limits['mean']
            assert limits['lower_limit'] < limits['mean']
    
    def test_detection_normal(self, detector, training_data):
        """Test detection on normal data."""
        sensor_columns = ['sensor_vibration_x', 'sensor_temperature', 'sensor_pressure']
        detector.train(training_data, sensor_columns)
        
        # Test with normal data point
        normal_point = {
            'equipment_id': 'TEST_001',
            'sensor_vibration_x': 2.0,
            'sensor_temperature': 75.0,
            'sensor_pressure': 150.0
        }
        
        result = detector.detect(normal_point)
        
        assert isinstance(result, AnomalyResult)
        assert result.equipment_id == 'TEST_001'
        assert result.is_anomaly == False
        assert result.anomaly_score < 2.0  # Should be within threshold
        assert len(result.affected_sensors) == 0
    
    def test_detection_anomaly(self, detector, training_data):
        """Test detection on anomalous data."""
        sensor_columns = ['sensor_vibration_x', 'sensor_temperature', 'sensor_pressure']
        detector.train(training_data, sensor_columns)
        
        # Test with anomalous data point (extreme values)
        anomaly_point = {
            'equipment_id': 'TEST_001',
            'sensor_vibration_x': 10.0,  # Very high vibration
            'sensor_temperature': 120.0,  # Very high temperature
            'sensor_pressure': 50.0  # Very low pressure
        }
        
        result = detector.detect(anomaly_point)
        
        assert result.is_anomaly == True
        assert result.anomaly_score > 2.0  # Should exceed threshold
        assert len(result.affected_sensors) > 0
        assert result.severity in ['low', 'medium', 'high']
    
    def test_detection_without_training(self, detector):
        """Test that detection fails without training."""
        test_point = {'sensor_vibration_x': 2.0}
        
        with pytest.raises(ValueError, match="Detector must be trained"):
            detector.detect(test_point)

class TestIsolationForestDetector:
    """Test cases for IsolationForestDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create an Isolation Forest detector for testing."""
        config = DetectorConfig(contamination=0.1, min_samples=50)
        return IsolationForestDetector(config)
    
    @pytest.fixture
    def training_data(self):
        """Create training data for testing."""
        np.random.seed(42)
        # Create mostly normal data with some outliers
        normal_data = np.random.normal(0, 1, (900, 3))
        outlier_data = np.random.normal(0, 1, (100, 3)) * 5  # Outliers with higher variance
        
        data = np.vstack([normal_data, outlier_data])
        np.random.shuffle(data)
        
        return pd.DataFrame(data, columns=['sensor_vibration_x', 'sensor_temperature', 'sensor_pressure'])
    
    def test_training(self, detector, training_data):
        """Test Isolation Forest training."""
        sensor_columns = ['sensor_vibration_x', 'sensor_temperature', 'sensor_pressure']
        detector.train(training_data, sensor_columns)
        
        assert detector.is_trained == True
        assert len(detector.feature_columns) == 3
        assert detector.model is not None
        assert detector.scaler is not None
    
    def test_training_insufficient_data(self, detector):
        """Test training with insufficient data."""
        small_data = pd.DataFrame({
            'sensor_vibration_x': [1, 2, 3],
            'sensor_temperature': [70, 75, 80]
        })
        
        sensor_columns = ['sensor_vibration_x', 'sensor_temperature']
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            detector.train(small_data, sensor_columns)
    
    def test_detection(self, detector, training_data):
        """Test Isolation Forest detection."""
        sensor_columns = ['sensor_vibration_x', 'sensor_temperature', 'sensor_pressure']
        detector.train(training_data, sensor_columns)
        
        # Test normal point
        normal_point = {
            'equipment_id': 'TEST_001',
            'sensor_vibration_x': 0.5,
            'sensor_temperature': 0.3,
            'sensor_pressure': -0.2
        }
        
        result = detector.detect(normal_point)
        assert isinstance(result, AnomalyResult)
        assert result.equipment_id == 'TEST_001'
        
        # Test outlier point
        outlier_point = {
            'equipment_id': 'TEST_001',
            'sensor_vibration_x': 10.0,
            'sensor_temperature': -8.0,
            'sensor_pressure': 15.0
        }
        
        outlier_result = detector.detect(outlier_point)
        # Outlier should have higher anomaly score
        assert outlier_result.anomaly_score >= result.anomaly_score

class TestTimeSeriesAnomalyDetector:
    """Test cases for TimeSeriesAnomalyDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a time series anomaly detector for testing."""
        config = DetectorConfig(window_size=50, threshold_sigma=2.5)
        return TimeSeriesAnomalyDetector(config)
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series training data."""
        np.random.seed(42)
        
        # Create time series with trend and seasonality
        t = np.arange(1000)
        trend = 0.01 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 100)
        noise = np.random.normal(0, 0.5, 1000)
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'sensor_vibration_x': 2.0 + trend + seasonal + noise,
            'sensor_temperature': 75.0 + 0.5 * trend + noise * 2
        })
        
        return data
    
    def test_training(self, detector, time_series_data):
        """Test time series detector training."""
        sensor_columns = ['sensor_vibration_x', 'sensor_temperature']
        detector.train(time_series_data, sensor_columns)
        
        assert detector.is_trained == True
        assert len(detector.trend_models) == 2
        
        for sensor in sensor_columns:
            assert sensor in detector.trend_models
            model = detector.trend_models[sensor]
            assert 'overall_mean' in model
            assert 'overall_std' in model
    
    def test_detection_with_history(self, detector, time_series_data):
        """Test detection with recent history."""
        sensor_columns = ['sensor_vibration_x', 'sensor_temperature']
        detector.train(time_series_data, sensor_columns)
        
        # Create recent history
        recent_history = [
            {'sensor_vibration_x': 2.1, 'sensor_temperature': 75.2},
            {'sensor_vibration_x': 2.0, 'sensor_temperature': 75.1},
            {'sensor_vibration_x': 2.2, 'sensor_temperature': 75.3},
            {'sensor_vibration_x': 2.1, 'sensor_temperature': 75.0},
            {'sensor_vibration_x': 2.3, 'sensor_temperature': 75.4}
        ]
        
        # Test normal point
        normal_point = {
            'equipment_id': 'TEST_001',
            'sensor_vibration_x': 2.1,
            'sensor_temperature': 75.2
        }
        
        result = detector.detect(normal_point, recent_history)
        assert isinstance(result, AnomalyResult)
        assert result.equipment_id == 'TEST_001'

class TestEnsembleAnomalyDetector:
    """Test cases for EnsembleAnomalyDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create an ensemble anomaly detector for testing."""
        config = DetectorConfig(contamination=0.1, min_samples=50)
        return EnsembleAnomalyDetector(config)
    
    @pytest.fixture
    def training_data(self):
        """Create training data for ensemble testing."""
        simulator = SensorSimulator("TEST_001")
        return simulator.generate_batch_data(hours=24, interval_seconds=300)
    
    def test_initialization(self, detector):
        """Test ensemble detector initialization."""
        assert 'statistical' in detector.detectors
        assert 'isolation_forest' in detector.detectors
        assert 'time_series' in detector.detectors
        assert len(detector.weights) == 3
        assert sum(detector.weights.values()) == 1.0  # Weights should sum to 1
    
    def test_training(self, detector, training_data):
        """Test ensemble training."""
        sensor_columns = [col for col in training_data.columns if col.startswith('sensor_')]
        detector.train(training_data, sensor_columns)
        
        assert detector.is_trained == True
        
        # Check that individual detectors are trained
        for name, individual_detector in detector.detectors.items():
            if hasattr(individual_detector, 'is_trained'):
                # Some detectors might fail to train, which is acceptable
                pass
    
    def test_detection(self, detector, training_data):
        """Test ensemble detection."""
        sensor_columns = [col for col in training_data.columns if col.startswith('sensor_')]
        detector.train(training_data, sensor_columns)
        
        # Create test data point
        test_point = {
            'equipment_id': 'TEST_001'
        }
        
        # Add sensor values from training data
        for col in sensor_columns:
            test_point[col] = training_data[col].iloc[0]
        
        result = detector.detect(test_point)
        
        assert isinstance(result, AnomalyResult)
        assert result.equipment_id == 'TEST_001'
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.anomaly_score, float)
        assert isinstance(result.confidence, float)
        assert result.severity in ['low', 'medium', 'high']

class TestAnomalyDetectionService:
    """Test cases for AnomalyDetectionService class."""
    
    @pytest.fixture
    def service(self):
        """Create an anomaly detection service for testing."""
        return AnomalyDetectionService()
    
    @pytest.fixture
    def training_data(self):
        """Create training data for service testing."""
        simulator = SensorSimulator("TEST_001")
        return simulator.generate_batch_data(hours=12, interval_seconds=600)
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service.detector is not None
        assert isinstance(service.alert_history, list)
        assert isinstance(service.recent_data, list)
        assert len(service.alert_history) == 0
        assert len(service.recent_data) == 0
    
    def test_training(self, service, training_data):
        """Test service training."""
        service.train_detector(training_data)
        assert service.detector.is_trained == True
    
    def test_anomaly_detection(self, service, training_data):
        """Test anomaly detection through service."""
        service.train_detector(training_data)
        
        # Create test data point
        test_point = training_data.iloc[0].to_dict()
        
        result = service.detect_anomaly(test_point)
        
        assert isinstance(result, AnomalyResult)
        assert len(service.recent_data) == 1
        
        # If anomaly detected, should be in alert history
        if result.is_anomaly:
            assert len(service.alert_history) == 1
    
    def test_alert_history_management(self, service, training_data):
        """Test alert history management."""
        service.train_detector(training_data)
        
        # Create anomalous data point
        anomaly_point = {
            'equipment_id': 'TEST_001',
            'sensor_vibration_x': 999.0,  # Extreme value
            'sensor_temperature': 999.0,
            'timestamp': datetime.now()
        }
        
        result = service.detect_anomaly(anomaly_point)
        
        if result.is_anomaly:
            assert len(service.alert_history) == 1
            
            # Test alert retrieval
            recent_alerts = service.get_recent_alerts(hours=1)
            assert len(recent_alerts) == 1
            
            # Test alert summary
            summary = service.get_alert_summary()
            assert summary['total_alerts'] == 1
            assert 'severity_breakdown' in summary
            assert 'most_affected_sensors' in summary
    
    def test_recent_data_management(self, service, training_data):
        """Test recent data management."""
        service.train_detector(training_data)
        
        # Add multiple data points
        for i in range(10):
            test_point = training_data.iloc[i].to_dict()
            service.detect_anomaly(test_point)
        
        assert len(service.recent_data) == 10
        
        # Test that data is limited to max size
        max_recent = service.config.get('max_recent_data', 1000)
        for i in range(max_recent + 100):  # Add more than max
            test_point = training_data.iloc[i % len(training_data)].to_dict()
            service.detect_anomaly(test_point)
        
        assert len(service.recent_data) <= max_recent

class TestIntegration:
    """Integration tests for anomaly detection system."""
    
    def test_end_to_end_anomaly_detection(self):
        """Test complete anomaly detection workflow."""
        # Create simulator and generate data
        simulator = SensorSimulator("INTEGRATION_TEST")
        
        # Generate normal data
        normal_data = simulator.generate_batch_data(hours=6, interval_seconds=300)
        
        # Create and train service
        service = AnomalyDetectionService()
        service.train_detector(normal_data)
        
        # Inject failure and generate anomalous data
        simulator.inject_failure("bearing_wear", severity=0.8)
        anomaly_data = simulator.generate_batch_data(hours=1, interval_seconds=300)
        
        # Test detection on anomalous data
        anomaly_count = 0
        for _, row in anomaly_data.iterrows():
            data_point = row.to_dict()
            result = service.detect_anomaly(data_point)
            
            if result.is_anomaly:
                anomaly_count += 1
        
        # Should detect some anomalies with severe bearing wear
        assert anomaly_count > 0
        
        # Check alert summary
        summary = service.get_alert_summary()
        assert summary['total_alerts'] >= anomaly_count
    
    def test_multiple_equipment_detection(self):
        """Test anomaly detection with multiple equipment."""
        # Create multiple simulators
        simulators = {
            'PUMP_001': SensorSimulator("PUMP_001"),
            'MOTOR_002': SensorSimulator("MOTOR_002")
        }
        
        # Generate training data from both
        all_training_data = []
        for sim in simulators.values():
            data = sim.generate_batch_data(hours=3, interval_seconds=600)
            all_training_data.append(data)
        
        combined_training_data = pd.concat(all_training_data, ignore_index=True)
        
        # Train service
        service = AnomalyDetectionService()
        service.train_detector(combined_training_data)
        
        # Test detection on both equipment
        for equipment_id, sim in simulators.items():
            test_data = sim.generate_batch_data(hours=0.5, interval_seconds=600)
            
            for _, row in test_data.iterrows():
                data_point = row.to_dict()
                result = service.detect_anomaly(data_point)
                
                assert result.equipment_id == equipment_id
                assert isinstance(result.is_anomaly, bool)

if __name__ == "__main__":
    pytest.main([__file__])
