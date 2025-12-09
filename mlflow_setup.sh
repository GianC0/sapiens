#!/bin/bash
# mlflow_setup.sh - Setup and launch MLflow UI

# Launch MLflow UI
echo "📊 Launching MLflow UI..."
echo "   Access at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

mlflow ui --backend-store-uri sqlite:///.runs/mlflow/mlflow.db --host 0.0.0.0 --port 5000


## Update nautilus_trader
pip install --upgrade nautilus_trader==1.222.0.dev20251207+12377 --index-url=https://packages.nautechsystems.io/simple