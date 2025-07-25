
"""
Real-time Monitoring Dashboard for Manufacturing Predictive Maintenance.
Built with Dash for interactive web-based monitoring.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
import logging
from typing import Dict, List, Any, Optional
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensor_simulator import SensorSimulator
from anomaly_detector import AnomalyDetectionService
from models import PredictiveModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for data storage
data_store = {
    'sensor_data': [],
    'anomalies': [],
    'predictions': [],
    'equipment_status': {}
}

# Initialize services
simulator = SensorSimulator("PUMP_001")
anomaly_service = AnomalyDetectionService()
model_service = PredictiveModelService()

# Dashboard configuration
REFRESH_INTERVAL = 5000  # milliseconds
MAX_DATA_POINTS = 1000
EQUIPMENT_IDS = ["PUMP_001", "MOTOR_002", "COMPRESSOR_003"]

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Predictive Maintenance Dashboard"

# Custom CSS
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define colors
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'primary': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'card': '#ffffff'
}

def create_header():
    """Create dashboard header."""
    return html.Div([
        html.H1(
            [
                html.I(className="fas fa-cogs", style={'margin-right': '10px'}),
                "Manufacturing Predictive Maintenance Dashboard"
            ],
            style={
                'text-align': 'center',
                'color': colors['text'],
                'margin-bottom': '30px',
                'font-family': 'Arial, sans-serif'
            }
        ),
        html.Hr(style={'margin-bottom': '30px'})
    ])

def create_equipment_status_cards():
    """Create equipment status cards."""
    return html.Div([
        html.H3("Equipment Status", style={'color': colors['text'], 'margin-bottom': '20px'}),
        html.Div(id='equipment-status-cards', children=[
            create_status_card(eq_id) for eq_id in EQUIPMENT_IDS
        ], style={'display': 'flex', 'justify-content': 'space-around', 'margin-bottom': '30px'})
    ])

def create_status_card(equipment_id: str):
    """Create individual equipment status card."""
    return html.Div([
        html.Div([
            html.H4(equipment_id, style={'margin': '0', 'color': colors['text']}),
            html.P(id=f'health-{equipment_id}', children="Health: --", 
                   style={'margin': '5px 0', 'font-size': '14px'}),
            html.P(id=f'status-{equipment_id}', children="Status: --", 
                   style={'margin': '5px 0', 'font-size': '14px'}),
            html.P(id=f'alerts-{equipment_id}', children="Alerts: --", 
                   style={'margin': '5px 0', 'font-size': '14px'})
        ], style={
            'background-color': colors['card'],
            'padding': '20px',
            'border-radius': '8px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
            'text-align': 'center',
            'min-width': '200px'
        })
    ])

def create_control_panel():
    """Create control panel for simulation."""
    return html.Div([
        html.H3("Simulation Control", style={'color': colors['text'], 'margin-bottom': '20px'}),
        html.Div([
            html.Div([
                html.Label("Equipment:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='equipment-selector',
                    options=[{'label': eq_id, 'value': eq_id} for eq_id in EQUIPMENT_IDS],
                    value=EQUIPMENT_IDS[0],
                    style={'margin-bottom': '10px'}
                ),
                html.Label("Inject Failure:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='failure-selector',
                    options=[
                        {'label': 'Bearing Wear', 'value': 'bearing_wear'},
                        {'label': 'Misalignment', 'value': 'misalignment'},
                        {'label': 'Overheating', 'value': 'overheating'}
                    ],
                    placeholder="Select failure type",
                    style={'margin-bottom': '10px'}
                ),
                html.Label("Severity:", style={'font-weight': 'bold'}),
                dcc.Slider(
                    id='severity-slider',
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    value=0.3,
                    marks={i/10: f'{i/10:.1f}' for i in range(1, 11)},
                    style={'margin-bottom': '20px'}
                ),
                html.Button(
                    'Inject Failure',
                    id='inject-failure-btn',
                    n_clicks=0,
                    style={
                        'background-color': colors['warning'],
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px',
                        'border-radius': '4px',
                        'cursor': 'pointer',
                        'margin-right': '10px'
                    }
                ),
                html.Button(
                    'Reset Equipment',
                    id='reset-equipment-btn',
                    n_clicks=0,
                    style={
                        'background-color': colors['success'],
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }
                )
            ], style={
                'background-color': colors['card'],
                'padding': '20px',
                'border-radius': '8px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ])
    ], style={'margin-bottom': '30px'})

def create_sensor_charts():
    """Create sensor data visualization charts."""
    return html.Div([
        html.H3("Real-time Sensor Data", style={'color': colors['text'], 'margin-bottom': '20px'}),
        html.Div([
            html.Div([
                dcc.Graph(id='vibration-chart')
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='temperature-chart')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([
                dcc.Graph(id='pressure-chart')
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='current-chart')
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ], style={'margin-bottom': '30px'})

def create_anomaly_alerts():
    """Create anomaly alerts section."""
    return html.Div([
        html.H3("Anomaly Alerts", style={'color': colors['text'], 'margin-bottom': '20px'}),
        html.Div(id='anomaly-alerts', style={
            'background-color': colors['card'],
            'padding': '20px',
            'border-radius': '8px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
            'max-height': '300px',
            'overflow-y': 'auto'
        })
    ], style={'margin-bottom': '30px'})

def create_predictions_panel():
    """Create predictions panel."""
    return html.Div([
        html.H3("Predictive Analytics", style={'color': colors['text'], 'margin-bottom': '20px'}),
        html.Div([
            html.Div([
                html.H4("Health Score", style={'text-align': 'center', 'color': colors['text']}),
                dcc.Graph(id='health-score-gauge')
            ], style={'width': '33%', 'display': 'inline-block'}),
            html.Div([
                html.H4("Failure Prediction", style={'text-align': 'center', 'color': colors['text']}),
                html.Div(id='failure-prediction', style={'text-align': 'center', 'padding': '20px'})
            ], style={'width': '33%', 'display': 'inline-block'}),
            html.Div([
                html.H4("Remaining Useful Life", style={'text-align': 'center', 'color': colors['text']}),
                html.Div(id='rul-prediction', style={'text-align': 'center', 'padding': '20px'})
            ], style={'width': '33%', 'display': 'inline-block'})
        ])
    ])

# App layout
app.layout = html.Div([
    create_header(),
    create_equipment_status_cards(),
    create_control_panel(),
    create_sensor_charts(),
    create_anomaly_alerts(),
    create_predictions_panel(),
    
    # Hidden div to store data
    html.Div(id='data-store', style={'display': 'none'}),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL,
        n_intervals=0
    )
], style={
    'background-color': colors['background'],
    'padding': '20px',
    'font-family': 'Arial, sans-serif'
})

def generate_sample_data():
    """Generate sample sensor data for demonstration."""
    current_time = datetime.now()
    
    # Generate data point
    data_point = simulator.generate_data_point()
    data_point['timestamp'] = current_time
    
    # Add to data store
    data_store['sensor_data'].append(data_point)
    
    # Keep only recent data
    if len(data_store['sensor_data']) > MAX_DATA_POINTS:
        data_store['sensor_data'] = data_store['sensor_data'][-MAX_DATA_POINTS:]
    
    # Detect anomalies (if service is trained)
    try:
        anomaly_result = anomaly_service.detect_anomaly(data_point)
        if anomaly_result.is_anomaly:
            data_store['anomalies'].append(anomaly_result)
            
            # Keep only recent anomalies
            cutoff_time = current_time - timedelta(hours=24)
            data_store['anomalies'] = [
                a for a in data_store['anomalies'] 
                if a.timestamp > cutoff_time
            ]
    except:
        pass  # Service not trained yet
    
    # Get predictions (if models are trained)
    try:
        df = pd.DataFrame([data_point])
        predictions = model_service.predict_all(df)
        data_store['predictions'] = predictions
    except:
        pass  # Models not trained yet

@app.callback(
    Output('data-store', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_data_store(n):
    """Update data store with new sensor data."""
    generate_sample_data()
    return json.dumps({'updated': datetime.now().isoformat()})

@app.callback(
    [Output('vibration-chart', 'figure'),
     Output('temperature-chart', 'figure'),
     Output('pressure-chart', 'figure'),
     Output('current-chart', 'figure')],
    Input('data-store', 'children')
)
def update_sensor_charts(data_store_json):
    """Update sensor data charts."""
    if not data_store['sensor_data']:
        # Return empty charts
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No Data Available")
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # Convert to DataFrame
    df = pd.DataFrame(data_store['sensor_data'])
    
    # Vibration chart
    vibration_fig = go.Figure()
    vibration_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=[sensors.get('vibration_x', 0) for sensors in df['sensors']],
        mode='lines',
        name='Vibration X',
        line=dict(color=colors['primary'])
    ))
    vibration_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=[sensors.get('vibration_y', 0) for sensors in df['sensors']],
        mode='lines',
        name='Vibration Y',
        line=dict(color=colors['warning'])
    ))
    vibration_fig.update_layout(
        title='Vibration (mm/s)',
        xaxis_title='Time',
        yaxis_title='Vibration (mm/s)',
        template='plotly_white'
    )
    
    # Temperature chart
    temperature_fig = go.Figure()
    temperature_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=[sensors.get('temperature', 0) for sensors in df['sensors']],
        mode='lines',
        name='Temperature',
        line=dict(color=colors['danger'])
    ))
    temperature_fig.update_layout(
        title='Temperature (°C)',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        template='plotly_white'
    )
    
    # Pressure chart
    pressure_fig = go.Figure()
    pressure_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=[sensors.get('pressure', 0) for sensors in df['sensors']],
        mode='lines',
        name='Pressure',
        line=dict(color=colors['success'])
    ))
    pressure_fig.update_layout(
        title='Pressure (PSI)',
        xaxis_title='Time',
        yaxis_title='Pressure (PSI)',
        template='plotly_white'
    )
    
    # Current chart
    current_fig = go.Figure()
    current_fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=[sensors.get('current', 0) for sensors in df['sensors']],
        mode='lines',
        name='Current',
        line=dict(color='purple')
    ))
    current_fig.update_layout(
        title='Current (A)',
        xaxis_title='Time',
        yaxis_title='Current (A)',
        template='plotly_white'
    )
    
    return vibration_fig, temperature_fig, pressure_fig, current_fig

@app.callback(
    Output('anomaly-alerts', 'children'),
    Input('data-store', 'children')
)
def update_anomaly_alerts(data_store_json):
    """Update anomaly alerts display."""
    if not data_store['anomalies']:
        return html.P("No anomalies detected", style={'color': colors['success']})
    
    alerts = []
    for anomaly in data_store['anomalies'][-10:]:  # Show last 10 anomalies
        severity_color = {
            'low': colors['warning'],
            'medium': 'orange',
            'high': colors['danger']
        }.get(anomaly.severity, colors['text'])
        
        alert_div = html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", 
                       style={'margin-right': '10px', 'color': severity_color}),
                html.Strong(f"{anomaly.severity.upper()} ANOMALY", 
                           style={'color': severity_color}),
                html.Span(f" - {anomaly.timestamp.strftime('%H:%M:%S')}", 
                         style={'float': 'right', 'color': colors['text']})
            ]),
            html.P(f"Equipment: {anomaly.equipment_id}", 
                   style={'margin': '5px 0', 'font-size': '12px'}),
            html.P(f"Score: {anomaly.anomaly_score:.3f}, Confidence: {anomaly.confidence:.3f}", 
                   style={'margin': '5px 0', 'font-size': '12px'}),
            html.P(f"Affected sensors: {', '.join(anomaly.affected_sensors)}", 
                   style={'margin': '5px 0', 'font-size': '12px'})
        ], style={
            'border-left': f'4px solid {severity_color}',
            'padding': '10px',
            'margin-bottom': '10px',
            'background-color': '#f8f9fa'
        })
        alerts.append(alert_div)
    
    return alerts

@app.callback(
    Output('health-score-gauge', 'figure'),
    Input('data-store', 'children')
)
def update_health_gauge(data_store_json):
    """Update health score gauge."""
    if not data_store['sensor_data']:
        health_score = 1.0
    else:
        health_score = data_store['sensor_data'][-1].get('equipment_health', 1.0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score (%)"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': colors['danger']},
                {'range': [50, 80], 'color': colors['warning']},
                {'range': [80, 100], 'color': colors['success']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

@app.callback(
    [Output('failure-prediction', 'children'),
     Output('rul-prediction', 'children')],
    Input('data-store', 'children')
)
def update_predictions(data_store_json):
    """Update prediction displays."""
    failure_pred = html.Div([
        html.P("No prediction available", style={'color': colors['text']})
    ])
    
    rul_pred = html.Div([
        html.P("No prediction available", style={'color': colors['text']})
    ])
    
    if data_store['predictions']:
        # Failure prediction
        if 'failure_classification' in data_store['predictions']:
            failure_results = data_store['predictions']['failure_classification']
            if failure_results:
                result = failure_results[0]
                failure_pred = html.Div([
                    html.H5(result.prediction, style={'color': colors['text']}),
                    html.P(f"Confidence: {result.confidence:.1%}", 
                           style={'color': colors['text'], 'font-size': '12px'})
                ])
        
        # RUL prediction
        if 'rul_estimation' in data_store['predictions']:
            rul_results = data_store['predictions']['rul_estimation']
            if rul_results:
                result = rul_results[0]
                rul_pred = html.Div([
                    html.H5(f"{result.prediction:.0f} hours", style={'color': colors['text']}),
                    html.P(f"Confidence: {result.confidence:.1%}", 
                           style={'color': colors['text'], 'font-size': '12px'})
                ])
    
    return failure_pred, rul_pred

@app.callback(
    Output('inject-failure-btn', 'children'),
    [Input('inject-failure-btn', 'n_clicks')],
    [State('equipment-selector', 'value'),
     State('failure-selector', 'value'),
     State('severity-slider', 'value')]
)
def inject_failure(n_clicks, equipment_id, failure_type, severity):
    """Handle failure injection."""
    if n_clicks > 0 and failure_type:
        simulator.inject_failure(failure_type, severity)
        return f"Injected {failure_type} (severity: {severity})"
    return "Inject Failure"

@app.callback(
    Output('reset-equipment-btn', 'children'),
    Input('reset-equipment-btn', 'n_clicks')
)
def reset_equipment(n_clicks):
    """Handle equipment reset."""
    if n_clicks > 0:
        # Reset simulator
        global simulator
        simulator = SensorSimulator("PUMP_001")
        return "Equipment Reset"
    return "Reset Equipment"

def initialize_services():
    """Initialize anomaly detection and predictive models with sample data."""
    logger.info("Initializing services with sample data...")
    
    # Generate training data
    training_data = simulator.generate_batch_data(hours=24, interval_seconds=300)
    
    # Train anomaly detector
    try:
        anomaly_service.train_detector(training_data)
        logger.info("Anomaly detection service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize anomaly service: {e}")
    
    # Train predictive models
    try:
        model_service.train_all_models(training_data)
        logger.info("Predictive models initialized")
    except Exception as e:
        logger.error(f"Failed to initialize predictive models: {e}")

def run_dashboard():
    """Run the dashboard application."""
    # Initialize services in background
    init_thread = threading.Thread(target=initialize_services)
    init_thread.daemon = True
    init_thread.start()
    
    # Run the app
    app.run_server(
        debug=False,
        host='0.0.0.0',
        port=8050,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )

if __name__ == '__main__':
    run_dashboard()
