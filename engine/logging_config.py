import logging
from pathlib import Path
from datetime import datetime
import sys
import mlflow

# Logs
logs_dir = None
logs_subdir_sapiens = None
logs_subdir_nautilus = None

# Runs
runs_dir = None
runs_mlflow_subdir = None
runs_models_subdir = None
runs_strategies_subdir = None 
runs_backtests_subdir = None

_configured = set()
_project_prefixes = []

def setup_logging_and_mlflow(log_dir=".logs/", run_dir=".runs/", level= logging.INFO, project_modules=None, optimization_id=None):
    global logs_dir, _project_prefixes, logs_subdir_sapiens, logs_subdir_nautilus
    global runs_dir, runs_mlflow_subdir, runs_models_subdir, runs_strategies_subdir, runs_backtests_subdir
    
    # Define which modules get individual log files
    if project_modules is None:
        project_modules = ['engine', 'strategies', 'models', 'tests', 'run_backtest', 'research', 'make_requirements']
    
    _project_prefixes = project_modules
    
    # Create timestamped directory
    folder_name = optimization_id if optimization_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create sapiens and nautilus logs directories
    logs_dir = Path(log_dir) / folder_name
    logs_subdir_sapiens = logs_dir / "sapiens"
    logs_subdir_nautilus = logs_dir / "nautilus"
    logs_subdir_nautilus.mkdir(parents=True, exist_ok=True)
    logs_subdir_sapiens.mkdir(parents=True, exist_ok=True)

    # Create mlflow, models and strategies runs directories
    runs_dir = Path(run_dir)
    runs_mlflow_subdir = runs_dir / "mlflow"
    runs_models_subdir = runs_dir / "Models"
    runs_strategies_subdir = runs_dir / "Strategies" 
    runs_backtests_subdir = runs_dir / "Backtests"
    for d in [runs_mlflow_subdir,runs_models_subdir,runs_strategies_subdir,runs_backtests_subdir]:
        d.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{runs_mlflow_subdir / 'mlflow.db'}")

    
    # Root logger -> FULL.log + console
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_subdir_sapiens / "ALL.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    # Monkey-patch getLogger to auto-create module logs
    original_getLogger = logging.getLogger
    
    def enhanced_getLogger(name=None):
        logger = original_getLogger(name)
        
        # Create individual logs for sapiens modules
        if name and name not in _configured:
            is_project_module = any(name.startswith(prefix) for prefix in _project_prefixes)
            
            if is_project_module:
                _configured.add(name)
                handler = logging.FileHandler(logs_subdir_sapiens / f"{name.replace('.', '_')}.log")
                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(handler)
        
        return logger
    
    logging.getLogger = enhanced_getLogger
    return logs_dir, runs_dir

def get_nautilus_logs_dir():
    return logs_subdir_nautilus

def get_sapiens_logs_dir():
    return logs_subdir_sapiens


def get_mlflow_uri():
    return f"sqlite:///{runs_mlflow_subdir / 'mlflow.db'}"

def get_runs_dirs():
    return {
        "mlflow": runs_mlflow_subdir,
        "models": runs_models_subdir,
        "strategies": runs_strategies_subdir,
        "backtests": runs_backtests_subdir,
    }