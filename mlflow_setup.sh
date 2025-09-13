#!/bin/bash
# mlflow_setup.sh - Setup and launch MLflow UI

# Launch MLflow UI
echo "📊 Launching MLflow UI..."
echo "   Access at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

mlflow ui --backend-store-uri file:./logs/mlruns --host 0.0.0.0 --port 5000