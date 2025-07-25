
"""
Unit tests for the sensor simulator module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
import os
from src.sensor_simulator import SensorSimulator, SensorConfig, FailureMode

class TestSensorSimulator:
    """Test cases for SensorSimulator class."""
    
    @pytest.fixture
    def simulator(self):
        """Create a sensor simulator for testing."""
        return SensorSimulator("TEST_PUMP_001")
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            "equipment": [
                {
                    "id": "TEST_PUMP_001",
                    "type": "test_pump",
                    "sensors": {
                        "vibration_x": {
                            "type": "accelerometer",
                            "range": [0, 10],
                            "units": "mm/s",
                            "sampling_rate": 1000
                        },
                        "temperature": {
                            "type": "thermocouple",
                            "range": [0, 150],
                            "units": "°C",
                            "sampling_rate": 1
                        }
                    }
                }
            ],
            "failure_modes": {
                "test_failure": {
                    "affected_sensors": ["vibration_x"],
                    "progression_rate": "medium",
                    "warning_indicators": {
                        "vibration_increase": 2.0
                    }
                }
            }
        }
    
    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.equipment_id == "TEST_PUMP_001"
        assert simulator.equipment_health == 1.0
        assert len(simulator.active_failures) == 0
        assert simulator.is_running == False
        assert len(simulator.sensors) > 0
        assert len(simulator.failure_modes) > 0
    
    def test_config_loading(self, sample_config):
        """Test configuration loading."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config, f)
            config_path = f.name
        
        try:
            simulator = SensorSimulator("TEST_PUMP_001", config_file=config_path)
            assert "vibration_x" in simulator.sensors
            assert "temperature" in simulator.sensors
            assert "test_failure" in simulator.failure_modes
        finally:
            os.unlink(config_path)
    
    def test_sensor_config_creation(self, simulator):
        """Test sensor configuration objects."""
        for sensor_name, sensor_config in simulator.sensors.items():
            assert isinstance(sensor_config, SensorConfig)
            assert sensor_config.sensor_type is not None
            assert len(sensor_config.normal_range) == 2
            assert sensor_config.normal_range[0] < sensor_config.normal_range[1]
            assert sensor_config.units is not None
            assert sensor_config.sampling_rate > 0
    
    def test_failure_mode_creation(self, simulator):
        """Test failure mode configuration objects."""
        for failure_name, failure_mode in simulator.failure_modes.items():
            assert isinstance(failure_mode, FailureMode)
            assert failure_mode.name == failure_name
            assert len(failure_mode.affected_sensors) > 0
            assert failure_mode.progression_rate in ['slow', 'medium', 'fast']
            assert isinstance(failure_mode.warning_indicators, dict)
    
    def test_baseline_value_generation(self, simulator):
        """Test baseline sensor value generation."""
        values = simulator._generate_baseline_values()
        
        assert isinstance(values, dict)
        assert len(values) > 0
        
        for sensor_name, value in values.items():
            assert isinstance(value, (int, float))
            assert value >= 0  # All sensor values should be non-negative
            
            # Check if value is within reasonable range for sensor
            if sensor_name in simulator.sensors:
                sensor_config = simulator.sensors[sensor_name]
                min_val, max_val = sensor_config.normal_range
                # Allow some tolerance for noise
                assert value <= max_val * 2  # Values can exceed normal range due to load factors
    
    def test_operating_conditions_effect(self, simulator):
        """Test that operating conditions affect sensor values."""
        # Test with low load
        simulator.set_operating_conditions(load=0.2, speed=1000)
        low_load_values = simulator._generate_baseline_values()
        
        # Test with high load
        simulator.set_operating_conditions(load=0.9, speed=2000)
        high_load_values = simulator._generate_baseline_values()
        
        # Vibration and temperature should generally be higher with higher load
        if 'vibration_x' in low_load_values and 'vibration_x' in high_load_values:
            # Allow for some randomness in the comparison
            assert high_load_values['vibration_x'] >= low_load_values['vibration_x'] * 0.8
        
        if 'temperature' in low_load_values and 'temperature' in high_load_values:
            assert high_load_values['temperature'] >= low_load_values['temperature']
    
    def test_failure_injection(self, simulator):
        """Test manual failure injection."""
        initial_health = simulator.equipment_health
        initial_failures = len(simulator.active_failures)
        
        # Inject a failure
        simulator.inject_failure("bearing_wear", severity=0.3)
        
        assert len(simulator.active_failures) == initial_failures + 1
        assert "bearing_wear" in simulator.degradation_factors
        assert simulator.degradation_factors["bearing_wear"] == 0.3
        
        # Equipment health should be affected
        assert simulator.equipment_health <= initial_health
    
    def test_failure_effects_on_sensors(self, simulator):
        """Test that failures affect sensor readings."""
        # Get baseline values
        baseline_values = simulator._generate_baseline_values()
        
        # Inject bearing wear failure
        simulator.inject_failure("bearing_wear", severity=0.5)
        
        # Get values with failure
        failure_values = simulator._apply_failure_effects(baseline_values)
        
        # Vibration should increase with bearing wear
        if 'vibration_x' in baseline_values and 'vibration_x' in failure_values:
            assert failure_values['vibration_x'] > baseline_values['vibration_x']
        
        # Temperature should increase with bearing wear
        if 'temperature' in baseline_values and 'temperature' in failure_values:
            assert failure_values['temperature'] >= baseline_values['temperature']
    
    def test_data_point_generation(self, simulator):
        """Test single data point generation."""
        data_point = simulator.generate_data_point()
        
        # Check required fields
        assert 'timestamp' in data_point
        assert 'equipment_id' in data_point
        assert 'sensors' in data_point
        assert 'operating_conditions' in data_point
        assert 'equipment_health' in data_point
        assert 'active_failures' in data_point
        
        # Check data types
        assert isinstance(data_point['sensors'], dict)
        assert isinstance(data_point['operating_conditions'], dict)
        assert isinstance(data_point['equipment_health'], float)
        assert isinstance(data_point['active_failures'], list)
        
        # Check sensor values
        for sensor_name, value in data_point['sensors'].items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_batch_data_generation(self, simulator):
        """Test batch data generation."""
        hours = 2
        interval_seconds = 300  # 5 minutes
        
        data = simulator.generate_batch_data(hours=hours, interval_seconds=interval_seconds)
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        
        # Check expected number of data points
        expected_points = (hours * 3600) // interval_seconds
        assert len(data) == expected_points
        
        # Check required columns
        required_columns = ['timestamp', 'equipment_id', 'equipment_health']
        for col in required_columns:
            assert col in data.columns
        
        # Check sensor columns
        sensor_columns = [col for col in data.columns if col.startswith('sensor_')]
        assert len(sensor_columns) > 0
        
        # Check timestamp ordering
        timestamps = pd.to_datetime(data['timestamp'])
        assert timestamps.is_monotonic_increasing
        
        # Check data types
        assert data['equipment_health'].dtype in [np.float64, np.float32]
        for col in sensor_columns:
            assert data[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    
    def test_failure_progression(self, simulator):
        """Test failure progression over time."""
        # Inject a failure
        simulator.inject_failure("bearing_wear", severity=0.1)
        
        initial_degradation = simulator.degradation_factors["bearing_wear"]
        initial_health = simulator.equipment_health
        
        # Simulate progression
        for _ in range(10):
            simulator._update_failure_progression()
        
        # Degradation should increase
        final_degradation = simulator.degradation_factors["bearing_wear"]
        assert final_degradation > initial_degradation
        
        # Health should decrease
        assert simulator.equipment_health <= initial_health
    
    def test_real_time_simulation(self, simulator):
        """Test real-time simulation functionality."""
        # Start simulation
        simulator.start_real_time_simulation(interval_seconds=0.1)  # Fast interval for testing
        
        assert simulator.is_running == True
        assert simulator.simulation_thread is not None
        
        # Wait a bit for data generation
        import time
        time.sleep(0.5)
        
        # Check that data is being generated
        latest_data = simulator.get_latest_data(count=5)
        assert len(latest_data) > 0
        
        # Stop simulation
        simulator.stop_real_time_simulation()
        assert simulator.is_running == False
    
    def test_data_buffer_management(self, simulator):
        """Test data buffer management in real-time mode."""
        # Start simulation
        simulator.start_real_time_simulation(interval_seconds=0.05)
        
        import time
        time.sleep(0.3)  # Let it generate some data
        
        # Get data
        data_batch1 = simulator.get_latest_data(count=3)
        data_batch2 = simulator.get_latest_data(count=2)
        
        # Stop simulation
        simulator.stop_real_time_simulation()
        
        # Check that data was retrieved
        assert len(data_batch1) <= 3
        assert len(data_batch2) <= 2
        
        # Check data structure
        for data_point in data_batch1:
            assert 'timestamp' in data_point
            assert 'equipment_id' in data_point
            assert 'sensors' in data_point
    
    def test_equipment_status(self, simulator):
        """Test equipment status reporting."""
        status = simulator.get_equipment_status()
        
        # Check required fields
        assert 'equipment_id' in status
        assert 'timestamp' in status
        assert 'equipment_health' in status
        assert 'active_failures' in status
        assert 'operating_conditions' in status
        assert 'sensor_count' in status
        assert 'is_running' in status
        
        # Check data types
        assert isinstance(status['equipment_health'], float)
        assert isinstance(status['active_failures'], list)
        assert isinstance(status['operating_conditions'], dict)
        assert isinstance(status['sensor_count'], int)
        assert isinstance(status['is_running'], bool)
    
    def test_data_export(self, simulator):
        """Test data export functionality."""
        # Generate some data
        data = simulator.generate_batch_data(hours=1, interval_seconds=600)
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            simulator.export_data(data, csv_path, format='csv')
            assert os.path.exists(csv_path)
            
            # Verify exported data
            exported_data = pd.read_csv(csv_path)
            assert len(exported_data) == len(data)
            assert list(exported_data.columns) == list(data.columns)
        finally:
            os.unlink(csv_path)
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            simulator.export_data(data, json_path, format='json')
            assert os.path.exists(json_path)
            
            # Verify exported data
            with open(json_path, 'r') as f:
                exported_data = json.load(f)
            assert len(exported_data) == len(data)
        finally:
            os.unlink(json_path)
    
    def test_multiple_failure_modes(self, simulator):
        """Test handling of multiple simultaneous failures."""
        # Inject multiple failures
        simulator.inject_failure("bearing_wear", severity=0.2)
        simulator.inject_failure("misalignment", severity=0.3)
        
        assert len(simulator.active_failures) == 2
        assert "bearing_wear" in simulator.degradation_factors
        assert "misalignment" in simulator.degradation_factors
        
        # Generate data with multiple failures
        data_point = simulator.generate_data_point()
        
        # Should have both failures listed
        assert len(data_point['active_failures']) == 2
        assert "bearing_wear" in data_point['active_failures']
        assert "misalignment" in data_point['active_failures']
    
    def test_sensor_value_bounds(self, simulator):
        """Test that sensor values stay within reasonable bounds."""
        # Inject severe failure
        simulator.inject_failure("overheating", severity=0.9)
        
        # Generate multiple data points
        for _ in range(10):
            data_point = simulator.generate_data_point()
            
            for sensor_name, value in data_point['sensors'].items():
                # Values should be positive and finite
                assert value >= 0
                assert np.isfinite(value)
                
                # Values shouldn't be extremely large (sanity check)
                assert value < 10000  # Reasonable upper bound for any sensor
    
    def test_time_progression(self, simulator):
        """Test that time progresses correctly in batch generation."""
        start_time = datetime.now()
        simulator.current_time = start_time
        
        hours = 1
        interval_seconds = 300
        
        data = simulator.generate_batch_data(hours=hours, interval_seconds=interval_seconds)
        
        # Check time progression
        timestamps = pd.to_datetime(data['timestamp'])
        time_diffs = timestamps.diff().dropna()
        
        # All time differences should be approximately equal to interval
        expected_diff = pd.Timedelta(seconds=interval_seconds)
        for diff in time_diffs:
            assert abs((diff - expected_diff).total_seconds()) < 1  # Allow 1 second tolerance
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test with invalid configuration
        invalid_config = {
            "equipment": [],  # Empty equipment list
            "failure_modes": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name
        
        try:
            # Should still work with default configuration
            simulator = SensorSimulator("TEST_EQUIPMENT", config_file=config_path)
            assert len(simulator.sensors) > 0  # Should fall back to defaults
        finally:
            os.unlink(config_path)
    
    def test_edge_cases(self, simulator):
        """Test edge cases and error conditions."""
        # Test with zero interval (should handle gracefully)
        try:
            data = simulator.generate_batch_data(hours=0, interval_seconds=60)
            assert len(data) == 0
        except:
            pass  # It's acceptable to raise an error for zero hours
        
        # Test with very small interval
        data = simulator.generate_batch_data(hours=0.01, interval_seconds=1)  # 36 seconds, 1-second intervals
        assert len(data) > 0
        
        # Test failure injection with invalid failure name
        initial_failures = len(simulator.active_failures)
        simulator.inject_failure("nonexistent_failure", severity=0.5)
        # Should not add invalid failure
        assert len(simulator.active_failures) == initial_failures

if __name__ == "__main__":
    pytest.main([__file__])
