
# Core imports
from pathlib import Path
from sklearn import base
import yaml
import logging
import pandas as pd
import mlflow
from mlflow import MlflowClient
import tqdm as notebook_tqdm
from datetime import datetime


# Nautilus Trader
from nautilus_trader.model.objects import Currency
from nautilus_trader.core.nautilus_pyo3 import CurrencyType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import TradeTick

# Project modules
from engine.databento_loader import DatabentoTickLoader
from engine.hparam_tuner import OptunaHparamsTuner
from engine.ModelGenerator import ModelGenerator
from engine.logging_config import setup_logging_and_mlflow



# Load Sapiens config
sapiens_config_path = Path("configs/sapiens_config.yaml")
sapiens_config = yaml.safe_load(sapiens_config_path.read_text(encoding="utf-8"))


optid = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logs_dir, runs_dir = setup_logging_and_mlflow(log_dir=sapiens_config["logs_dir"], run_dir=sapiens_config["runs_dir"], level=sapiens_config["logs_level"], optimization_id=optid)
logger = logging.getLogger('run_backtest')

# Load strategy and model config for initial setup
strategy_name = sapiens_config["SAPIENS_STRATEGY"]['strategy_name']
strategy_config_path = Path(f"strategies/{strategy_name}/strategy_config.yaml")
strategy_config = yaml.safe_load(strategy_config_path.read_text(encoding="utf-8"))["STRATEGY"]



# Model generation or Setup
model_name = sapiens_config["SAPIENS_MODEL"]['model_name']
# Generate new model via DeepCode if necessary
if sapiens_config["SAPIENS_MODEL"]['generate_model']:
    logger.info("="*70)
    logger.info("MODEL GENERATION VIA DEEPCODE")
    logger.info("="*70)
    

    
    gen_cfg = sapiens_config["SAPIENS_MODEL"]['generation']
    generator = ModelGenerator(gen_cfg)
    
    
    model_dir = generator.generate_model(
        source_type=gen_cfg['source_type'],
        source_path=gen_cfg['source_path'],
        model_name=model_name,
    )
    
    print(f"Model generated: {model_dir}")
    print("Review the generated code before continuing!")

# Model Config
model_config_path = Path(f"models/{model_name}/model_config.yaml")
model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))["MODEL"]

print("Configuration loaded successfully")
print(f"Model: {model_config['PARAMS']['model_name']}")
print(f"Strategy: {strategy_config['PARAMS']['strategy_name']}")
print(f"Backtest period: {sapiens_config['backtest_start']} to {sapiens_config['backtest_end']}")




# Catalog Configuration
FORCE_RELOAD_CATALOG = False  # Set to True to rebuild catalog
CATALOG_PATH = None  # Set custom path or None for default

# Initialize loader
logger.info("Initializing Databento loader...")
loader = DatabentoTickLoader(
    cfg=strategy_config["PARAMS"],
    venue_name=strategy_config["PARAMS"]["venue_name"],
    data_dir=sapiens_config["data_dir"],
    #universe_subset = ["TSLA","NVDA"],
)

# Determine catalog path
catalog_path = Path(CATALOG_PATH) if CATALOG_PATH else loader.catalog_path

# Load or create catalog
if not FORCE_RELOAD_CATALOG and loader.catalog_exists(catalog_path):
    logger.info(f"Reusing existing catalog at: {catalog_path}")
    catalog = ParquetDataCatalog(path=str(catalog_path))
else:
    logger.info(f"Loading Databento ticks to catalog at: {catalog_path}")
    if FORCE_RELOAD_CATALOG:
        logger.info("Force reload enabled - rebuilding catalog")
    
    # Load with progress bar and memory management
    catalog = loader.load_to_catalog(
        catalog_path=catalog_path,
    )

# Add catalog path to config
strategy_config["PARAMS"]["catalog_path"] = str(catalog_path)

# Verify catalog
#instruments = catalog.instruments(instrument_type=TradeTick)  # takes too long on laptop. Use loader class instruments property instead
instruments = set(inst.id.value for inst in catalog.instruments())
print(f"\nCatalog ready: {catalog.list_data_types()} data loaded")
print(f"Universe: {[str(symbol) for symbol in instruments]}")

# Initialize hyperparameter tuner
tuner = OptunaHparamsTuner(
    sapiens_config=sapiens_config,
    catalog=catalog,
    model_config=model_config,
    strategy_config=strategy_config,
    run_dir=runs_dir,
    log_dir=logs_dir,
    optimization_id=optid,
)

print("Hyperparameter tuner initialized")
print(f"Model trials: {sapiens_config["SAPIENS_MODEL"]["optimization"]['n_trials']}")
print(f"Strategy trials: {sapiens_config["SAPIENS_STRATEGY"]["optimization"]['n_trials']}")


# Run model hyperparameter optimization
logger.info("\n" + "="*70)
logger.info("STAGE 2: MODEL HYPERPARAMETER OPTIMIZATION")
logger.info("="*70 + "\n")

model_results = tuner.optimize_model()
#model_results['param_importance_fig'].show()

print("\nModel optimization complete!")
print(f"Best model path: {model_results['model_path']}")
print(f"MLflow run ID: {model_results['mlflow_run_id']}")


# Run strategy hyperparameter optimization
logger.info("\n" + "="*70)
logger.info("STAGE 3: STRATEGY HYPERPARAMETER OPTIMIZATION")
logger.info("="*70 + "\n")

strategy_results = tuner.optimize_strategy(model_name=model_name)
#strategy_results['param_importance_fig'].show()

print("\nStrategy optimization complete!")
print(f"Best hyperparameters: {strategy_results['hparams']}")
print(f"\nBest metrics:")
for metric, value in strategy_results['metrics'].items():
    print(f"  {metric}: {value:.4f}")
print(f"\nMLflow run ID: {strategy_results['mlflow_run_id']}")
print(f"\nInspect results at: {strategy_results["results_path"]}")


"""

# Get optimization context
run = client.get_run(strategy_results["mlflow_run_id"])
optimization_id = run.data.tags.get("optimization_id", "")

# Define backtest period
backtest_start = sapiens_config["backtest_start"]
backtest_end = sapiens_config["backtest_end"]

print(f"Running final backtest: {backtest_start} to {backtest_end}")


# Execute final backtest
logger.info("\n" + "="*70)
logger.info("STAGE 4: FINAL BACKTEST")
logger.info("="*70 + "\n")

final_metrics, final_time_series = tuner.run_final_backtest(
    backtest_start=backtest_start,
    backtest_end=backtest_end,
    strategy_hpo_run_id=strategy_results["mlflow_run_id"],
    optimization_id=optimization_id
)

print("\nFinal backtest complete!")
print("\nFinal Performance Metrics:")
print("="*50)
for metric, value in sorted(final_metrics.items()):
    print(f"{metric:.<40} {value:>10.4f}")

"""