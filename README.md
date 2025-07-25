
# Manufacturing Predictive Maintenance System

A comprehensive IoT-based predictive maintenance system for manufacturing equipment. Features real-time sensor data simulation, anomaly detection, predictive modeling, and interactive monitoring dashboards.

## Features

- **IoT Sensor Simulation**:
  - Multi-sensor data generation (vibration, temperature, pressure, current)
  - Realistic failure pattern simulation
  - Configurable sensor parameters and failure modes
  - MQTT and HTTP data streaming

- **Anomaly Detection**:
  - Isolation Forest for outlier detection
  - Statistical process control (SPC) methods
  - Time-series anomaly detection
  - Real-time alerting system

- **Predictive Modeling**:
  - Machine learning models for failure prediction
  - Remaining useful life (RUL) estimation
  - Equipment health scoring
  - Maintenance scheduling optimization

- **Monitoring Dashboard**:
  - Real-time sensor data visualization
  - Equipment health status monitoring
  - Maintenance alerts and notifications
  - Historical trend analysis

## Installation

### Using pip
```bash
pip install -r requirements.txt
```

### Using Docker
```bash
docker build -t predictive-maintenance .
docker run -p 8050:8050 predictive-maintenance
```

## Quick Start

### Start the System
```bash
# Start sensor data simulation
python src/sensor_simulator.py

# Start the monitoring dashboard (in another terminal)
python src/dashboard.py

# Run anomaly detection service
python src/anomaly_detector.py
```

### Python API
```python
from src.models import PredictiveModel
from src.sensor_simulator import SensorSimulator

# Initialize sensor simulator
simulator = SensorSimulator(equipment_id="PUMP_001")
data = simulator.generate_batch_data(hours=24)

# Train predictive model
model = PredictiveModel()
model.train(data)

# Predict equipment health
health_score = model.predict_health(current_data)
```

## Web Dashboard

Launch the Dash web application:
```bash
python src/dashboard.py
```

Access at `http://localhost:8050`

## Project Structure

```
manufacturing-predictive-maintenance/
├── src/
│   ├── __init__.py
│   ├── sensor_simulator.py     # IoT sensor data simulation
│   ├── anomaly_detector.py     # Anomaly detection algorithms
│   ├── models.py              # Predictive ML models
│   ├── dashboard.py           # Web monitoring dashboard
│   ├── scheduler.py           # Maintenance scheduling
│   ├── data_processor.py      # Data preprocessing utilities
│   └── utils.py               # Common utilities
├── tests/
│   ├── test_simulator.py
│   ├── test_models.py
│   └── test_anomaly_detector.py
├── data/
│   ├── sensor_config.json     # Sensor configuration
│   ├── equipment_specs.json   # Equipment specifications
│   └── sample_data.csv        # Sample sensor data
├── models/
│   └── trained_models/        # Saved ML models
├── docker/
│   └── Dockerfile
└── docs/
    └── system_architecture.md
```

## System Architecture

### Data Flow
1. **Sensor Simulation** → Generates realistic IoT sensor data
2. **Data Processing** → Cleans and preprocesses sensor readings
3. **Anomaly Detection** → Identifies unusual patterns in real-time
4. **Predictive Modeling** → Forecasts equipment failures
5. **Maintenance Scheduling** → Optimizes maintenance activities
6. **Dashboard** → Visualizes system status and alerts

### Key Components

#### Sensor Simulator
- Simulates vibration, temperature, pressure, and current sensors
- Configurable failure modes and degradation patterns
- Supports multiple equipment types

#### Anomaly Detection
- **Isolation Forest**: Unsupervised outlier detection
- **Statistical Methods**: Control charts and threshold-based detection
- **Time Series**: LSTM-based sequence anomaly detection

#### Predictive Models
- **Classification**: Failure/no-failure prediction
- **Regression**: Remaining useful life estimation
- **Health Scoring**: Overall equipment condition assessment

## Configuration

Edit `config.yaml` to customize:
- Sensor parameters and failure modes
- Model hyperparameters
- Alert thresholds
- Dashboard settings

## Testing

```bash
pytest tests/ -v --cov=src
```

## API Documentation

### Sensor Data Format
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "equipment_id": "PUMP_001",
  "sensors": {
    "vibration_x": 2.5,
    "vibration_y": 1.8,
    "temperature": 75.2,
    "pressure": 145.6,
    "current": 12.3
  },
  "operating_conditions": {
    "load": 0.85,
    "speed": 1750
  }
}
```

### Health Score Response
```json
{
  "equipment_id": "PUMP_001",
  "health_score": 0.78,
  "risk_level": "medium",
  "predicted_failure_date": "2024-02-15",
  "recommended_actions": [
    "Schedule vibration analysis",
    "Check bearing condition"
  ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
