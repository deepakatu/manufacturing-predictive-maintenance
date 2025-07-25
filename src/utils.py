
"""
Utility functions for the Manufacturing Predictive Maintenance system.
"""

import pandas as pd
import numpy as np
import json
import yaml
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import pickle
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for the predictive maintenance system."""
    
    def __init__(self, db_path: str = "data/maintenance.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sensor data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    equipment_id TEXT,
                    sensor_name TEXT,
                    sensor_value REAL,
                    equipment_health REAL,
                    active_failures TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Anomalies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    equipment_id TEXT,
                    anomaly_score REAL,
                    confidence REAL,
                    severity TEXT,
                    affected_sensors TEXT,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    equipment_id TEXT,
                    model_type TEXT,
                    prediction_value TEXT,
                    confidence REAL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Maintenance tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS maintenance_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    equipment_id TEXT,
                    task_type TEXT,
                    priority TEXT,
                    status TEXT,
                    scheduled_start DATETIME,
                    scheduled_end DATETIME,
                    actual_start DATETIME,
                    actual_end DATETIME,
                    description TEXT,
                    cost_estimate REAL,
                    actual_cost REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def insert_sensor_data(self, data_points: List[Dict[str, Any]]):
        """Insert sensor data into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for point in data_points:
                timestamp = point.get('timestamp', datetime.now())
                equipment_id = point.get('equipment_id', 'unknown')
                equipment_health = point.get('equipment_health', 1.0)
                active_failures = point.get('active_failures', '')
                
                # Insert each sensor reading
                for sensor_name, sensor_value in point.get('sensors', {}).items():
                    cursor.execute('''
                        INSERT INTO sensor_data 
                        (timestamp, equipment_id, sensor_name, sensor_value, equipment_health, active_failures)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, equipment_id, sensor_name, sensor_value, equipment_health, active_failures))
            
            conn.commit()
    
    def insert_anomaly(self, anomaly_result):
        """Insert anomaly detection result into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO anomalies 
                (timestamp, equipment_id, anomaly_score, confidence, severity, affected_sensors, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                anomaly_result.timestamp,
                anomaly_result.equipment_id,
                anomaly_result.anomaly_score,
                anomaly_result.confidence,
                anomaly_result.severity,
                ','.join(anomaly_result.affected_sensors),
                anomaly_result.description
            ))
            
            conn.commit()
    
    def insert_prediction(self, prediction_result):
        """Insert prediction result into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, equipment_id, model_type, prediction_value, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                prediction_result.timestamp,
                prediction_result.equipment_id,
                prediction_result.model_type,
                str(prediction_result.prediction),
                prediction_result.confidence,
                json.dumps(prediction_result.metadata)
            ))
            
            conn.commit()
    
    def get_sensor_data(self, equipment_id: Optional[str] = None, 
                       hours_back: int = 24) -> pd.DataFrame:
        """Retrieve sensor data from database."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT * FROM sensor_data 
                WHERE timestamp > datetime('now', '-{} hours')
            '''.format(hours_back)
            
            if equipment_id:
                query += f" AND equipment_id = '{equipment_id}'"
            
            query += " ORDER BY timestamp DESC"
            
            return pd.read_sql_query(query, conn)
    
    def get_anomalies(self, equipment_id: Optional[str] = None, 
                     hours_back: int = 24) -> pd.DataFrame:
        """Retrieve anomalies from database."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT * FROM anomalies 
                WHERE timestamp > datetime('now', '-{} hours')
            '''.format(hours_back)
            
            if equipment_id:
                query += f" AND equipment_id = '{equipment_id}'"
            
            query += " ORDER BY timestamp DESC"
            
            return pd.read_sql_query(query, conn)
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove old data from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up sensor data
            cursor.execute('''
                DELETE FROM sensor_data 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            # Clean up anomalies
            cursor.execute('''
                DELETE FROM anomalies 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            # Clean up predictions
            cursor.execute('''
                DELETE FROM predictions 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            conn.commit()
            logger.info(f"Cleaned up data older than {days_to_keep} days")

class ConfigManager:
    """Manages configuration for the predictive maintenance system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Return default configuration
        return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'sensors': {
                'vibration': {
                    'normal_range': [0.5, 3.0],
                    'warning_threshold': 4.0,
                    'critical_threshold': 6.0,
                    'units': 'mm/s'
                },
                'temperature': {
                    'normal_range': [60, 85],
                    'warning_threshold': 90,
                    'critical_threshold': 100,
                    'units': '°C'
                }
            },
            'models': {
                'health_scoring': {
                    'model_type': 'random_forest',
                    'retrain_interval_hours': 168  # Weekly
                },
                'anomaly_detection': {
                    'contamination': 0.1,
                    'window_size': 100
                }
            },
            'alerts': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587
                },
                'thresholds': {
                    'health_score_warning': 0.7,
                    'health_score_critical': 0.5
                }
            },
            'data': {
                'retention_days': 90,
                'backup_enabled': True,
                'backup_interval_hours': 24
            }
        }
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value

class AlertManager:
    """Manages alerts and notifications for the predictive maintenance system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = []
    
    def send_email_alert(self, subject: str, message: str, recipients: List[str]):
        """Send email alert."""
        if not self.config.get('alerts', {}).get('email', {}).get('enabled', False):
            logger.info("Email alerts disabled")
            return
        
        try:
            smtp_server = self.config['alerts']['email']['smtp_server']
            smtp_port = self.config['alerts']['email']['smtp_port']
            username = self.config['alerts']['email'].get('username')
            password = self.config['alerts']['email'].get('password')
            
            if not username or not password:
                logger.error("Email credentials not configured")
                return
            
            msg = MimeMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent to {recipients}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def check_health_score_alert(self, equipment_id: str, health_score: float):
        """Check if health score warrants an alert."""
        warning_threshold = self.config.get('alerts', {}).get('thresholds', {}).get('health_score_warning', 0.7)
        critical_threshold = self.config.get('alerts', {}).get('thresholds', {}).get('health_score_critical', 0.5)
        
        if health_score < critical_threshold:
            self.send_critical_alert(equipment_id, health_score)
        elif health_score < warning_threshold:
            self.send_warning_alert(equipment_id, health_score)
    
    def send_critical_alert(self, equipment_id: str, health_score: float):
        """Send critical health score alert."""
        subject = f"CRITICAL: Equipment {equipment_id} Health Alert"
        message = f"""
        CRITICAL ALERT: Equipment {equipment_id} has reached a critical health score.
        
        Current Health Score: {health_score:.2f}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Immediate maintenance action is required to prevent equipment failure.
        
        Please contact the maintenance team immediately.
        """
        
        recipients = self.config.get('alerts', {}).get('email', {}).get('critical_recipients', [])
        if recipients:
            self.send_email_alert(subject, message, recipients)
        
        # Log alert
        self.alert_history.append({
            'timestamp': datetime.now(),
            'type': 'critical',
            'equipment_id': equipment_id,
            'health_score': health_score,
            'message': message
        })
    
    def send_warning_alert(self, equipment_id: str, health_score: float):
        """Send warning health score alert."""
        subject = f"WARNING: Equipment {equipment_id} Health Alert"
        message = f"""
        WARNING: Equipment {equipment_id} health score is below normal levels.
        
        Current Health Score: {health_score:.2f}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Preventive maintenance should be scheduled soon to avoid potential issues.
        
        Please review the equipment status and plan maintenance accordingly.
        """
        
        recipients = self.config.get('alerts', {}).get('email', {}).get('warning_recipients', [])
        if recipients:
            self.send_email_alert(subject, message, recipients)
        
        # Log alert
        self.alert_history.append({
            'timestamp': datetime.now(),
            'type': 'warning',
            'equipment_id': equipment_id,
            'health_score': health_score,
            'message': message
        })

class DataProcessor:
    """Processes and transforms data for the predictive maintenance system."""
    
    @staticmethod
    def calculate_statistical_features(data: pd.Series, window_size: int = 10) -> Dict[str, float]:
        """Calculate statistical features for sensor data."""
        if len(data) < window_size:
            window_size = len(data)
        
        if window_size == 0:
            return {}
        
        recent_data = data.tail(window_size)
        
        features = {
            'mean': recent_data.mean(),
            'std': recent_data.std(),
            'min': recent_data.min(),
            'max': recent_data.max(),
            'median': recent_data.median(),
            'skewness': recent_data.skew(),
            'kurtosis': recent_data.kurtosis(),
            'range': recent_data.max() - recent_data.min(),
            'iqr': recent_data.quantile(0.75) - recent_data.quantile(0.25)
        }
        
        # Handle NaN values
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0.0
        
        return features
    
    @staticmethod
    def calculate_trend_features(data: pd.Series, window_size: int = 10) -> Dict[str, float]:
        """Calculate trend-based features for sensor data."""
        if len(data) < 2:
            return {'trend': 0.0, 'trend_strength': 0.0}
        
        # Calculate first difference (trend)
        diff = data.diff().dropna()
        
        if len(diff) == 0:
            return {'trend': 0.0, 'trend_strength': 0.0}
        
        # Recent trend
        recent_diff = diff.tail(min(window_size, len(diff)))
        trend = recent_diff.mean()
        trend_strength = abs(trend) / (data.std() + 1e-8)  # Normalize by data variability
        
        return {
            'trend': trend if not pd.isna(trend) else 0.0,
            'trend_strength': trend_strength if not pd.isna(trend_strength) else 0.0
        }
    
    @staticmethod
    def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> List[int]:
        """Detect outliers in sensor data."""
        if len(data) < 4:
            return []
        
        outlier_indices = []
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outlier_indices = data[z_scores > threshold].index.tolist()
        
        return outlier_indices
    
    @staticmethod
    def normalize_sensor_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize sensor data."""
        normalized_data = data.copy()
        
        sensor_columns = [col for col in data.columns if col.startswith('sensor_')]
        
        for col in sensor_columns:
            if method == 'minmax':
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val > 0:
                    normalized_data[col] = (data[col] - mean_val) / std_val
        
        return normalized_data

class ModelManager:
    """Manages machine learning models for the predictive maintenance system."""
    
    def __init__(self, models_directory: str = "models/trained_models"):
        self.models_directory = models_directory
        os.makedirs(models_directory, exist_ok=True)
    
    def save_model(self, model, model_name: str, metadata: Optional[Dict] = None):
        """Save a trained model to disk."""
        model_path = os.path.join(self.models_directory, f"{model_name}.pkl")
        metadata_path = os.path.join(self.models_directory, f"{model_name}_metadata.json")
        
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            if metadata:
                metadata['saved_at'] = datetime.now().isoformat()
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
    
    def load_model(self, model_name: str) -> Tuple[Any, Optional[Dict]]:
        """Load a trained model from disk."""
        model_path = os.path.join(self.models_directory, f"{model_name}.pkl")
        metadata_path = os.path.join(self.models_directory, f"{model_name}_metadata.json")
        
        model = None
        metadata = None
        
        try:
            # Load model
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            if model is not None:
                logger.info(f"Model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
        
        return model, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        
        for file in os.listdir(self.models_directory):
            if file.endswith('.pkl'):
                model_name = file[:-4]  # Remove .pkl extension
                metadata_path = os.path.join(self.models_directory, f"{model_name}_metadata.json")
                
                model_info = {
                    'name': model_name,
                    'path': os.path.join(self.models_directory, file),
                    'size_mb': os.path.getsize(os.path.join(self.models_directory, file)) / (1024 * 1024),
                    'modified': datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(self.models_directory, file))
                    ).isoformat()
                }
                
                # Add metadata if available
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        model_info['metadata'] = metadata
                    except:
                        pass
                
                models.append(model_info)
        
        return models

def create_sample_data(equipment_ids: List[str], hours: int = 24) -> pd.DataFrame:
    """Create sample sensor data for testing."""
    from sensor_simulator import SensorSimulator
    
    all_data = []
    
    for equipment_id in equipment_ids:
        simulator = SensorSimulator(equipment_id)
        data = simulator.generate_batch_data(hours=hours, interval_seconds=300)
        all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def export_data_to_csv(data: pd.DataFrame, filename: str):
    """Export data to CSV file."""
    try:
        data.to_csv(filename, index=False)
        logger.info(f"Data exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export data: {e}")

def import_data_from_csv(filename: str) -> pd.DataFrame:
    """Import data from CSV file."""
    try:
        data = pd.read_csv(filename)
        logger.info(f"Data imported from {filename}")
        return data
    except Exception as e:
        logger.error(f"Failed to import data: {e}")
        return pd.DataFrame()

def main():
    """Example usage of utility functions."""
    # Initialize database
    db = DatabaseManager()
    
    # Create sample data
    sample_data = create_sample_data(["PUMP_001", "MOTOR_002"], hours=2)
    print(f"Created {len(sample_data)} sample records")
    
    # Insert into database
    data_points = sample_data.to_dict('records')
    db.insert_sensor_data(data_points)
    
    # Retrieve data
    retrieved_data = db.get_sensor_data(hours_back=2)
    print(f"Retrieved {len(retrieved_data)} records from database")
    
    # Test configuration manager
    config_manager = ConfigManager()
    print(f"Health warning threshold: {config_manager.get('alerts.thresholds.health_score_warning')}")
    
    # Test data processing
    processor = DataProcessor()
    if len(sample_data) > 0:
        vibration_data = sample_data['sensor_vibration_x']
        features = processor.calculate_statistical_features(vibration_data)
        print(f"Statistical features: {features}")

if __name__ == "__main__":
    main()
