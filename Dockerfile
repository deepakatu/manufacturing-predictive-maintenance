
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p models/trained_models data/logs

# Expose port for dashboard
EXPOSE 8050

# Set environment variables
ENV PYTHONPATH=/app
ENV DASH_HOST=0.0.0.0
ENV DASH_PORT=8050

# Run the dashboard application
CMD ["python", "src/dashboard.py"]
