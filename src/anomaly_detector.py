
"""
Anomaly Detection System for Manufacturing Equipment.
Implements multiple anomaly detection algorithms for real-time monitoring.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    timestamp: datetime
    equipment_id: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    affected_sensors: List[str]
    severity: str  # 'low', 'medium', 'high'
    description: str

@dataclass
class DetectorConfig:
    """Configuration for anomaly detectors."""
    contamination: float = 0.1
    window_size: int = 100
    threshold_sigma: float = 3.0
    min_samples: int = 50

class StatisticalAnomalyDetector:
    """Statistical process control based anomaly detection."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.control_limits = {}
        self.rolling_stats = {}
        self.is_trained = False
    
    def train(self, data: pd.DataFrame, sensor_columns: List[str]):
        """Train the statistical detector on normal data."""
        logger.info("Training statistical anomaly detector")
        
        for sensor in sensor_columns:
            if sensor in data.columns:
                values = data[sensor].dropna()
                
                mean = values.mean()
                std = values.std()
                
                self.control_limits[sensor] = {
                    'mean': mean,
                    'std': std,
                    'upper_limit': mean + self.config.threshold_sigma * std,
                    'lower_limit': mean - self.config.threshold_sigma * std
                }
        
        self.is_trained = True
        logger.info(f"Trained on {len(sensor_columns)} sensors")
    
    def detect(self, data_point: Dict[str, float]) -> AnomalyResult:
        """Detect anomalies in a single data point."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection")
        
        anomalies = []
        anomaly_scores = []
        
        for sensor, value in data_point.items():
            if sensor.startswith('sensor_') and sensor in self.control_limits:
                limits = self.control_limits[sensor]
                
                # Calculate z-score
                z_score = abs(value - limits['mean']) / limits['std']
                anomaly_scores.append(z_score)
                
                # Check if outside control limits
                if value > limits['upper_limit'] or value < limits['lower_limit']:
                    anomalies.append(sensor)
        
        # Overall anomaly decision
        max_score = max(anomaly_scores) if anomaly_scores else 0
        is_anomaly = len(anomalies) > 0
        
        # Determine severity
        if max_score > 4:
            severity = 'high'
        elif max_score > 3:
            severity = 'medium'
        else:
            severity = 'low'
        
        return AnomalyResult(
            timestamp=datetime.now(),
            equipment_id=data_point.get('equipment_id', 'unknown'),
            is_anomaly=is_anomaly,
            anomaly_score=max_score,
            confidence=min(1.0, max_score / 5.0),
            affected_sensors=anomalies,
            severity=severity,
            description=f"Statistical anomaly detected in {len(anomalies)} sensors"
        )

class IsolationForestDetector:
    """Isolation Forest based anomaly detection."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = IsolationForest(
            contamination=config.contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
    
    def train(self, data: pd.DataFrame, sensor_columns: List[str]):
        """Train the Isolation Forest on normal data."""
        logger.info("Training Isolation Forest anomaly detector")
        
        self.feature_columns = [col for col in sensor_columns if col in data.columns]
        
        if len(self.feature_columns) == 0:
            raise ValueError("No valid sensor columns found in data")
        
        # Prepare training data
        X = data[self.feature_columns].dropna()
        
        if len(X) < self.config.min_samples:
            raise ValueError(f"Insufficient training data: {len(X)} < {self.config.min_samples}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        self.is_trained = True
        logger.info(f"Trained Isolation Forest on {len(X)} samples with {len(self.feature_columns)} features")
    
    def detect(self, data_point: Dict[str, float]) -> AnomalyResult:
        """Detect anomalies using Isolation Forest."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection")
        
        # Extract features
        features = []
        for col in self.feature_columns:
            if col in data_point:
                features.append(data_point[col])
            else:
                features.append(0.0)  # Default value for missing features
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        anomaly_score = self.model.decision_function(X_scaled)[0]
        
        # Convert to positive score (higher = more anomalous)
        normalized_score = max(0, -anomaly_score)
        
        is_anomaly = prediction == -1
        
        # Determine severity based on score
        if normalized_score > 0.6:
            severity = 'high'
        elif normalized_score > 0.3:
            severity = 'medium'
        else:
            severity = 'low'
        
        return AnomalyResult(
            timestamp=datetime.now(),
            equipment_id=data_point.get('equipment_id', 'unknown'),
            is_anomaly=is_anomaly,
            anomaly_score=normalized_score,
            confidence=min(1.0, normalized_score),
            affected_sensors=self.feature_columns if is_anomaly else [],
            severity=severity,
            description="Isolation Forest anomaly detection"
        )

class TimeSeriesAnomalyDetector:
    """Time series based anomaly detection using sliding windows."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.historical_data = {}
        self.trend_models = {}
        self.is_trained = False
    
    def train(self, data: pd.DataFrame, sensor_columns: List[str]):
        """Train time series models on historical data."""
        logger.info("Training time series anomaly detector")
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
        
        for sensor in sensor_columns:
            if sensor in data.columns:
                values = data[sensor].dropna()
                
                if len(values) >= self.config.window_size:
                    # Store historical data for trend analysis
                    self.historical_data[sensor] = values.tolist()
                    
                    # Calculate trend statistics
                    rolling_mean = values.rolling(window=min(20, len(values)//4)).mean()
                    rolling_std = values.rolling(window=min(20, len(values)//4)).std()
                    
                    self.trend_models[sensor] = {
                        'mean_trend': rolling_mean.dropna().tolist(),
                        'std_trend': rolling_std.dropna().tolist(),
                        'overall_mean': values.mean(),
                        'overall_std': values.std()
                    }
        
        self.is_trained = True
        logger.info(f"Trained time series models for {len(self.trend_models)} sensors")
    
    def detect(self, data_point: Dict[str, float], recent_history: Optional[List[Dict]] = None) -> AnomalyResult:
        """Detect time series anomalies."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection")
        
        anomalies = []
        anomaly_scores = []
        
        for sensor, value in data_point.items():
            if sensor.startswith('sensor_') and sensor in self.trend_models:
                model = self.trend_models[sensor]
                
                # Calculate deviation from expected trend
                expected_mean = model['overall_mean']
                expected_std = model['overall_std']
                
                # Z-score based anomaly detection
                z_score = abs(value - expected_mean) / expected_std if expected_std > 0 else 0
                anomaly_scores.append(z_score)
                
                # Check for trend anomalies if recent history is available
                if recent_history and len(recent_history) >= 5:
                    recent_values = [h.get(sensor, 0) for h in recent_history[-5:]]
                    recent_trend = np.mean(np.diff(recent_values))
                    
                    # Check for sudden trend changes
                    if abs(recent_trend) > 2 * expected_std:
                        anomalies.append(sensor)
                        anomaly_scores.append(z_score * 1.5)  # Boost score for trend anomalies
                
                # Standard threshold check
                if z_score > self.config.threshold_sigma:
                    anomalies.append(sensor)
        
        # Overall anomaly decision
        max_score = max(anomaly_scores) if anomaly_scores else 0
        is_anomaly = len(anomalies) > 0
        
        # Determine severity
        if max_score > 4:
            severity = 'high'
        elif max_score > 3:
            severity = 'medium'
        else:
            severity = 'low'
        
        return AnomalyResult(
            timestamp=datetime.now(),
            equipment_id=data_point.get('equipment_id', 'unknown'),
            is_anomaly=is_anomaly,
            anomaly_score=max_score,
            confidence=min(1.0, max_score / 5.0),
            affected_sensors=list(set(anomalies)),
            severity=severity,
            description=f"Time series anomaly detected in {len(set(anomalies))} sensors"
        )

class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detection methods."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detectors = {
            'statistical': StatisticalAnomalyDetector(config),
            'isolation_forest': IsolationForestDetector(config),
            'time_series': TimeSeriesAnomalyDetector(config)
        }
        self.weights = {
            'statistical': 0.3,
            'isolation_forest': 0.4,
            'time_series': 0.3
        }
        self.is_trained = False
    
    def train(self, data: pd.DataFrame, sensor_columns: List[str]):
        """Train all detectors in the ensemble."""
        logger.info("Training ensemble anomaly detector")
        
        for name, detector in self.detectors.items():
            try:
                detector.train(data, sensor_columns)
                logger.info(f"Successfully trained {name} detector")
            except Exception as e:
                logger.error(f"Failed to train {name} detector: {str(e)}")
        
        self.is_trained = True
    
    def detect(self, data_point: Dict[str, float], recent_history: Optional[List[Dict]] = None) -> AnomalyResult:
        """Detect anomalies using ensemble voting."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before detection")
        
        results = {}
        
        # Get predictions from all detectors
        for name, detector in self.detectors.items():
            try:
                if name == 'time_series':
                    result = detector.detect(data_point, recent_history)
                else:
                    result = detector.detect(data_point)
                results[name] = result
            except Exception as e:
                logger.error(f"Error in {name} detector: {str(e)}")
                continue
        
        if not results:
            # Fallback result if all detectors fail
            return AnomalyResult(
                timestamp=datetime.now(),
                equipment_id=data_point.get('equipment_id', 'unknown'),
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                affected_sensors=[],
                severity='low',
                description="All detectors failed"
            )
        
        # Ensemble voting
        weighted_score = 0.0
        anomaly_votes = 0
        all_affected_sensors = set()
        
        for name, result in results.items():
            weight = self.weights.get(name, 1.0)
            weighted_score += result.anomaly_score * weight
            
            if result.is_anomaly:
                anomaly_votes += 1
            
            all_affected_sensors.update(result.affected_sensors)
        
        # Normalize weighted score
        total_weight = sum(self.weights[name] for name in results.keys())
        normalized_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Ensemble decision (majority vote)
        is_anomaly = anomaly_votes >= len(results) / 2
        
        # Determine severity
        if normalized_score > 3:
            severity = 'high'
        elif normalized_score > 2:
            severity = 'medium'
        else:
            severity = 'low'
        
        return AnomalyResult(
            timestamp=datetime.now(),
            equipment_id=data_point.get('equipment_id', 'unknown'),
            is_anomaly=is_anomaly,
            anomaly_score=normalized_score,
            confidence=min(1.0, normalized_score / 4.0),
            affected_sensors=list(all_affected_sensors),
            severity=severity,
            description=f"Ensemble detection: {anomaly_votes}/{len(results)} detectors flagged anomaly"
        )
    
    def save_models(self, directory: str):
        """Save trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        for name, detector in self.detectors.items():
            if hasattr(detector, 'model') and detector.is_trained:
                model_path = os.path.join(directory, f'{name}_model.joblib')
                joblib.dump(detector, model_path)
                logger.info(f"Saved {name} model to {model_path}")
    
    def load_models(self, directory: str):
        """Load trained models from disk."""
        for name in self.detectors.keys():
            model_path = os.path.join(directory, f'{name}_model.joblib')
            if os.path.exists(model_path):
                try:
                    self.detectors[name] = joblib.load(model_path)
                    logger.info(f"Loaded {name} model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {name} model: {str(e)}")
        
        self.is_trained = True

class AnomalyDetectionService:
    """Service for real-time anomaly detection."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.detector = EnsembleAnomalyDetector(
            DetectorConfig(
                contamination=self.config.get('contamination', 0.1),
                window_size=self.config.get('window_size', 100),
                threshold_sigma=self.config.get('threshold_sigma', 3.0)
            )
        )
        self.alert_history = []
        self.recent_data = []
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file."""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        return {
            'contamination': 0.1,
            'window_size': 100,
            'threshold_sigma': 3.0,
            'alert_retention_hours': 24,
            'max_recent_data': 1000
        }
    
    def train_detector(self, training_data: pd.DataFrame):
        """Train the anomaly detector on historical normal data."""
        sensor_columns = [col for col in training_data.columns if col.startswith('sensor_')]
        self.detector.train(training_data, sensor_columns)
        logger.info("Anomaly detection service trained successfully")
    
    def detect_anomaly(self, data_point: Dict[str, Any]) -> AnomalyResult:
        """Detect anomalies in real-time data."""
        # Add to recent data history
        self.recent_data.append(data_point)
        
        # Maintain recent data window
        max_recent = self.config.get('max_recent_data', 1000)
        if len(self.recent_data) > max_recent:
            self.recent_data = self.recent_data[-max_recent:]
        
        # Detect anomaly
        result = self.detector.detect(data_point, self.recent_data[-50:])
        
        # Store alert if anomaly detected
        if result.is_anomaly:
            self.alert_history.append(result)
            self._cleanup_old_alerts()
        
        return result
    
    def _cleanup_old_alerts(self):
        """Remove old alerts from history."""
        retention_hours = self.config.get('alert_retention_hours', 24)
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
    
    def get_recent_alerts(self, hours: int = 24) -> List[AnomalyResult]:
        """Get recent alerts within specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        recent_alerts = self.get_recent_alerts(24)
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        sensor_counts = {}
        
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1
            
            for sensor in alert.affected_sensors:
                sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'most_affected_sensors': sorted(sensor_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'last_alert_time': recent_alerts[-1].timestamp.isoformat() if recent_alerts else None
        }

def main():
    """Example usage of the anomaly detection system."""
    from sensor_simulator import SensorSimulator
    
    # Create sensor simulator and generate training data
    simulator = SensorSimulator("PUMP_001")
    training_data = simulator.generate_batch_data(hours=48, interval_seconds=60)
    
    print(f"Generated {len(training_data)} training samples")
    
    # Create and train anomaly detector
    service = AnomalyDetectionService()
    service.train_detector(training_data)
    
    # Inject failure and test detection
    simulator.inject_failure("bearing_wear", severity=0.5)
    test_data = simulator.generate_batch_data(hours=2, interval_seconds=60)
    
    print("\nTesting anomaly detection on data with bearing wear:")
    
    anomaly_count = 0
    for _, row in test_data.iterrows():
        data_point = row.to_dict()
        result = service.detect_anomaly(data_point)
        
        if result.is_anomaly:
            anomaly_count += 1
            print(f"ANOMALY DETECTED: {result.severity} severity, score: {result.anomaly_score:.3f}")
            print(f"  Affected sensors: {result.affected_sensors}")
            print(f"  Description: {result.description}")
    
    print(f"\nDetected {anomaly_count} anomalies out of {len(test_data)} data points")
    
    # Print alert summary
    summary = service.get_alert_summary()
    print("\nAlert Summary:")
    print(f"Total alerts: {summary['total_alerts']}")
    print(f"Severity breakdown: {summary['severity_breakdown']}")
    print(f"Most affected sensors: {summary['most_affected_sensors']}")

if __name__ == "__main__":
    main()
