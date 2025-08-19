#!/bin/bash
# mlflow_setup.sh - Setup and launch MLflow UI

echo "🚀 Setting up MLflow for model tracking..."

# Install MLflow if not already installed
pip install -q mlflow

# Create MLflow directory if it doesn't exist
mkdir -p mlruns

# Launch MLflow UI
echo "📊 Launching MLflow UI..."
echo "   Access at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000