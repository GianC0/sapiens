#!/bin/bash
# mlflow_setup.sh - Setup and launch MLflow UI

# Launch MLflow UI
echo "📊 Launching MLflow UI..."
echo "   Access at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

mlflow ui --backend-store-uri file:./logs/mlflow --host 0.0.0.0 --port 5000


## Update nautilus_trader
pip install --upgrade nautilus_trader==1.222.0.dev20251106+11800 --index-url=https://packages.nautechsystems.io/simple