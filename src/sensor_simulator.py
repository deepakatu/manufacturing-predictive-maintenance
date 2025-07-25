
"""
IoT Sensor Data Simulator for Manufacturing Equipment.
Generates realistic sensor data with configurable failure modes and degradation patterns.
"""

import numpy as np
import pandas as pd
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensorConfig:
    """Configuration for individual sensors."""
    sensor_type: str
    normal_range: Tuple[float, float]
    warning_threshold: float
    critical_threshold: float
    units: str
    sampling_rate: int
    noise_level: float = 0.05

@dataclass
class FailureMode:
    """Configuration for equipment failure modes."""
    name: str
    affected_sensors: List[str]
    progression_rate: str  # 'slow', 'medium', 'fast'
    warning_indicators: Dict[str, float]
    probability: float = 0.01

class SensorSimulator:
    """
    Simulates IoT sensor data for manufacturing equipment with realistic
    failure patterns and degradation modes.
    """
    
    def __init__(self, equipment_id: str, config_file: Optional[str] = None):
        """
        Initialize the sensor simulator.
        
        Args:
            equipment_id: Unique identifier for the equipment
            config_file: Path to sensor configuration file
        """
        self.equipment_id = equipment_id
        self.config = self._load_config(config_file)
        self.sensors = self._initialize_sensors()
        self.failure_modes = self._initialize_failure_modes()
        
        # Simulation state
        self.current_time = datetime.now()
        self.equipment_health = 1.0  # 1.0 = perfect health, 0.0 = failed
        self.active_failures = []
        self.degradation_factors = {}
        self.operating_conditions = {
            'load': 0.75,
            'speed': 1750,
            'ambient_temperature': 25.0
        }
        
        # Data storage
        self.data_buffer = queue.Queue(maxsize=10000)
        self.is_running = False
        self.simulation_thread = None
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load sensor configuration from file or use defaults."""
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                logger.warning(f"Config file {config_file} not found, using defaults")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default sensor configuration."""
        return {
            "equipment": [
                {
                    "id": self.equipment_id,
                    "type": "generic_equipment",
                    "sensors": {
                        "vibration_x": {
                            "type": "accelerometer",
                            "range": [0, 10],
                            "units": "mm/s",
                            "sampling_rate": 1000
                        },
                        "vibration_y": {
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
                        },
                        "pressure": {
                            "type": "pressure_transducer",
                            "range": [0, 300],
                            "units": "PSI",
                            "sampling_rate": 10
                        },
                        "current": {
                            "type": "current_transformer",
                            "range": [0, 30],
                            "units": "A",
                            "sampling_rate": 10
                        }
                    }
                }
            ],
            "failure_modes": {
                "bearing_wear": {
                    "affected_sensors": ["vibration_x", "vibration_y", "temperature"],
                    "progression_rate": "slow",
                    "warning_indicators": {
                        "vibration_increase": 1.5,
                        "temperature_increase": 10
                    }
                },
                "misalignment": {
                    "affected_sensors": ["vibration_x", "vibration_y"],
                    "progression_rate": "medium",
                    "warning_indicators": {
                        "vibration_increase": 2.0
                    }
                },
                "overheating": {
                    "affected_sensors": ["temperature", "current"],
                    "progression_rate": "fast",
                    "warning_indicators": {
                        "temperature_increase": 20,
                        "current_increase": 1.2
                    }
                }
            }
        }
    
    def _initialize_sensors(self) -> Dict[str, SensorConfig]:
        """Initialize sensor configurations."""
        sensors = {}
        
        # Find equipment config
        equipment_config = None
        for eq in self.config["equipment"]:
            if eq["id"] == self.equipment_id:
                equipment_config = eq
                break
        
        if not equipment_config:
            equipment_config = self.config["equipment"][0]
        
        for sensor_name, sensor_data in equipment_config["sensors"].items():
            sensors[sensor_name] = SensorConfig(
                sensor_type=sensor_data["type"],
                normal_range=tuple(sensor_data["range"]),
                warning_threshold=sensor_data["range"][1] * 0.8,
                critical_threshold=sensor_data["range"][1] * 0.95,
                units=sensor_data["units"],
                sampling_rate=sensor_data["sampling_rate"],
                noise_level=0.05
            )
        
        return sensors
    
    def _initialize_failure_modes(self) -> Dict[str, FailureMode]:
        """Initialize failure mode configurations."""
        failure_modes = {}
        
        for mode_name, mode_data in self.config["failure_modes"].items():
            failure_modes[mode_name] = FailureMode(
                name=mode_name,
                affected_sensors=mode_data["affected_sensors"],
                progression_rate=mode_data["progression_rate"],
                warning_indicators=mode_data["warning_indicators"],
                probability=0.001  # Low probability per time step
            )
        
        return failure_modes
    
    def _generate_baseline_values(self) -> Dict[str, float]:
        """Generate baseline sensor values for healthy equipment."""
        values = {}
        
        # Base values depend on operating conditions
        load_factor = self.operating_conditions['load']
        speed_factor = self.operating_conditions['speed'] / 1750  # Normalized to rated speed
        
        for sensor_name, config in self.sensors.items():
            min_val, max_val = config.normal_range
            
            if 'vibration' in sensor_name:
                # Vibration increases with load and speed
                base_value = min_val + (max_val - min_val) * 0.3 * load_factor * speed_factor
                
            elif sensor_name == 'temperature':
                # Temperature increases with load and ambient conditions
                ambient_temp = self.operating_conditions['ambient_temperature']
                base_value = ambient_temp + 30 + (20 * load_factor)
                
            elif sensor_name == 'pressure':
                # Pressure related to load
                base_value = min_val + (max_val - min_val) * 0.5 * load_factor
                
            elif sensor_name == 'current':
                # Current proportional to load
                base_value = min_val + (max_val - min_val) * 0.4 * load_factor
                
            else:
                # Generic sensor
                base_value = min_val + (max_val - min_val) * 0.5
            
            # Add some random variation
            noise = np.random.normal(0, base_value * config.noise_level)
            values[sensor_name] = max(0, base_value + noise)
        
        return values
    
    def _apply_failure_effects(self, baseline_values: Dict[str, float]) -> Dict[str, float]:
        """Apply effects of active failures to sensor values."""
        modified_values = baseline_values.copy()
        
        for failure_mode in self.active_failures:
            degradation = self.degradation_factors.get(failure_mode.name, 0.0)
            
            for sensor_name in failure_mode.affected_sensors:
                if sensor_name in modified_values:
                    
                    if failure_mode.name == 'bearing_wear':
                        if 'vibration' in sensor_name:
                            # Increase vibration with bearing wear
                            increase_factor = 1 + (degradation * 2.0)
                            modified_values[sensor_name] *= increase_factor
                        elif sensor_name == 'temperature':
                            # Increase temperature with bearing wear
                            modified_values[sensor_name] += degradation * 15
                    
                    elif failure_mode.name == 'misalignment':
                        if 'vibration' in sensor_name:
                            # Significant vibration increase with misalignment
                            increase_factor = 1 + (degradation * 3.0)
                            modified_values[sensor_name] *= increase_factor
                    
                    elif failure_mode.name == 'overheating':
                        if sensor_name == 'temperature':
                            # Rapid temperature increase
                            modified_values[sensor_name] += degradation * 25
                        elif sensor_name == 'current':
                            # Current may increase with overheating
                            increase_factor = 1 + (degradation * 0.5)
                            modified_values[sensor_name] *= increase_factor
        
        return modified_values
    
    def _update_failure_progression(self):
        """Update the progression of active failures."""
        for failure_mode in self.active_failures:
            current_degradation = self.degradation_factors.get(failure_mode.name, 0.0)
            
            # Progression rates
            if failure_mode.progression_rate == 'slow':
                progression_step = 0.001
            elif failure_mode.progression_rate == 'medium':
                progression_step = 0.005
            else:  # fast
                progression_step = 0.01
            
            # Update degradation
            new_degradation = min(1.0, current_degradation + progression_step)
            self.degradation_factors[failure_mode.name] = new_degradation
            
            # Update equipment health
            self.equipment_health = min(self.equipment_health, 1.0 - new_degradation * 0.5)
    
    def _check_new_failures(self):
        """Check for new failure initiation."""
        for failure_name, failure_mode in self.failure_modes.items():
            if failure_mode not in self.active_failures:
                # Check if failure should start
                if random.random() < failure_mode.probability:
                    self.active_failures.append(failure_mode)
                    self.degradation_factors[failure_name] = 0.01  # Start with small degradation
                    logger.info(f"New failure initiated: {failure_name} on {self.equipment_id}")
    
    def generate_data_point(self) -> Dict[str, Any]:
        """Generate a single data point with current sensor readings."""
        # Generate baseline values
        baseline_values = self._generate_baseline_values()
        
        # Apply failure effects
        sensor_values = self._apply_failure_effects(baseline_values)
        
        # Create data point
        data_point = {
            'timestamp': self.current_time.isoformat(),
            'equipment_id': self.equipment_id,
            'sensors': sensor_values,
            'operating_conditions': self.operating_conditions.copy(),
            'equipment_health': self.equipment_health,
            'active_failures': [f.name for f in self.active_failures],
            'degradation_factors': self.degradation_factors.copy()
        }
        
        return data_point
    
    def generate_batch_data(self, hours: int = 24, interval_seconds: int = 60) -> pd.DataFrame:
        """
        Generate batch data for a specified time period.
        
        Args:
            hours: Number of hours to simulate
            interval_seconds: Interval between data points in seconds
            
        Returns:
            DataFrame with simulated sensor data
        """
        data_points = []
        start_time = self.current_time
        end_time = start_time + timedelta(hours=hours)
        
        current_time = start_time
        
        logger.info(f"Generating {hours} hours of data for {self.equipment_id}")
        
        while current_time < end_time:
            self.current_time = current_time
            
            # Update failure progression
            self._update_failure_progression()
            
            # Check for new failures
            self._check_new_failures()
            
            # Generate data point
            data_point = self.generate_data_point()
            data_points.append(data_point)
            
            # Advance time
            current_time += timedelta(seconds=interval_seconds)
        
        # Convert to DataFrame
        df_data = []
        for point in data_points:
            row = {
                'timestamp': point['timestamp'],
                'equipment_id': point['equipment_id'],
                'equipment_health': point['equipment_health']
            }
            
            # Add sensor values
            for sensor_name, value in point['sensors'].items():
                row[f'sensor_{sensor_name}'] = value
            
            # Add operating conditions
            for condition, value in point['operating_conditions'].items():
                row[f'operating_{condition}'] = value
            
            # Add failure information
            row['active_failures'] = ','.join(point['active_failures'])
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Generated {len(df)} data points")
        return df
    
    def start_real_time_simulation(self, interval_seconds: int = 1):
        """Start real-time data generation in a separate thread."""
        if self.is_running:
            logger.warning("Simulation is already running")
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._real_time_simulation_loop,
            args=(interval_seconds,)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info(f"Started real-time simulation for {self.equipment_id}")
    
    def stop_real_time_simulation(self):
        """Stop real-time data generation."""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        
        logger.info(f"Stopped real-time simulation for {self.equipment_id}")
    
    def _real_time_simulation_loop(self, interval_seconds: int):
        """Real-time simulation loop running in separate thread."""
        while self.is_running:
            self.current_time = datetime.now()
            
            # Update failure progression
            self._update_failure_progression()
            
            # Check for new failures
            self._check_new_failures()
            
            # Generate data point
            data_point = self.generate_data_point()
            
            # Add to buffer
            try:
                self.data_buffer.put_nowait(data_point)
            except queue.Full:
                # Remove oldest item and add new one
                try:
                    self.data_buffer.get_nowait()
                    self.data_buffer.put_nowait(data_point)
                except queue.Empty:
                    pass
            
            time.sleep(interval_seconds)
    
    def get_latest_data(self, count: int = 1) -> List[Dict[str, Any]]:
        """Get the latest data points from the buffer."""
        data_points = []
        
        for _ in range(min(count, self.data_buffer.qsize())):
            try:
                data_points.append(self.data_buffer.get_nowait())
            except queue.Empty:
                break
        
        return data_points
    
    def inject_failure(self, failure_name: str, severity: float = 0.1):
        """Manually inject a failure for testing purposes."""
        if failure_name in self.failure_modes:
            failure_mode = self.failure_modes[failure_name]
            if failure_mode not in self.active_failures:
                self.active_failures.append(failure_mode)
            
            self.degradation_factors[failure_name] = min(1.0, severity)
            logger.info(f"Injected failure: {failure_name} with severity {severity}")
        else:
            logger.error(f"Unknown failure mode: {failure_name}")
    
    def set_operating_conditions(self, **conditions):
        """Update operating conditions."""
        for key, value in conditions.items():
            if key in self.operating_conditions:
                self.operating_conditions[key] = value
                logger.info(f"Updated {key} to {value}")
    
    def get_equipment_status(self) -> Dict[str, Any]:
        """Get current equipment status summary."""
        return {
            'equipment_id': self.equipment_id,
            'timestamp': self.current_time.isoformat(),
            'equipment_health': self.equipment_health,
            'active_failures': [f.name for f in self.active_failures],
            'degradation_factors': self.degradation_factors.copy(),
            'operating_conditions': self.operating_conditions.copy(),
            'sensor_count': len(self.sensors),
            'is_running': self.is_running
        }
    
    def export_data(self, data: pd.DataFrame, file_path: str, format: str = 'csv'):
        """Export generated data to file."""
        if format.lower() == 'csv':
            data.to_csv(file_path, index=False)
        elif format.lower() == 'json':
            data.to_json(file_path, orient='records', date_format='iso')
        elif format.lower() == 'parquet':
            data.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(data)} records to {file_path}")

def main():
    """Example usage of the sensor simulator."""
    # Create simulator for a pump
    simulator = SensorSimulator("PUMP_001")
    
    # Generate 24 hours of data
    data = simulator.generate_batch_data(hours=24, interval_seconds=60)
    
    print(f"Generated {len(data)} data points")
    print("\nSample data:")
    print(data.head())
    
    print("\nData summary:")
    print(data.describe())
    
    # Inject a failure and generate more data
    simulator.inject_failure("bearing_wear", severity=0.3)
    failure_data = simulator.generate_batch_data(hours=2, interval_seconds=60)
    
    print(f"\nGenerated {len(failure_data)} data points with bearing wear")
    print("Equipment status:", simulator.get_equipment_status())

if __name__ == "__main__":
    main()
