
# Core imports
from pathlib import Path
import yaml
import logging
import pandas as pd
import mlflow
from mlflow import MlflowClient
import tqdm as notebook_tqdm


# Nautilus Trader
from nautilus_trader.model.objects import Currency
from nautilus_trader.core.nautilus_pyo3 import CurrencyType
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import TradeTick

# Project modules
from algos.engine.databento_loader import DatabentoTickLoader
from algos.engine.hparam_tuner import OptunaHparamsTuner
from algos.engine.performance_plots import (
    get_frequency_params, align_series,
    plot_balance_breakdown, plot_cumulative_returns,
    plot_rolling_sharpe, plot_underwater,
    plot_active_returns, plot_portfolio_allocation
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Load configuration
cfg_path = Path("configs/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

# Setup currency
currency_code = cfg["STRATEGY"]["PARAMS"]["currency"]
if currency_code == "USD":
    cfg["STRATEGY"]["PARAMS"]["currency"] = Currency(
        code='USD', precision=3, iso4217=840,
        name='United States dollar', currency_type=CurrencyType.FIAT
    )
elif currency_code == "EUR":
    cfg["STRATEGY"]["PARAMS"]["currency"] = Currency(
        code='EUR', precision=3, iso4217=978,
        name='Euro', currency_type=CurrencyType.FIAT
    )

# Setup directories
logs_dir = Path("logs/")
logs_dir.mkdir(parents=True, exist_ok=True)

print("Configuration loaded successfully")
print(f"Model: {cfg['MODEL']['PARAMS']['model_name']}")
print(f"Strategy: {cfg['STRATEGY']['PARAMS']['strategy_name']}")
print(f"Backtest period: {cfg['STRATEGY']['PARAMS']['backtest_start']} to {cfg['STRATEGY']['PARAMS']['backtest_end']}")


# Configuration
FORCE_RELOAD_CATALOG = False  # Set to True to rebuild catalog
CATALOG_PATH = None  # Set custom path or None for default

# Initialize loader
logger.info("Initializing Databento loader...")
loader = DatabentoTickLoader(
    cfg=cfg["STRATEGY"]["PARAMS"],
    venue_name=cfg["STRATEGY"]["PARAMS"]["venue_name"]
)

# Determine catalog path
catalog_path = Path(CATALOG_PATH) if CATALOG_PATH else loader.catalog_path




# Load or create catalog
if not FORCE_RELOAD_CATALOG and loader.catalog_exists(catalog_path):
    logger.info(f"📂 Reusing existing catalog at: {catalog_path}")
    catalog = ParquetDataCatalog(path=str(catalog_path))
else:
    logger.info(f"🔄 Loading Databento ticks to catalog at: {catalog_path}")
    if FORCE_RELOAD_CATALOG:
        logger.info("Force reload enabled - rebuilding catalog")
    
    # Load with progress bar and memory management
    catalog = loader.load_to_catalog(
        catalog_path=catalog_path,
    )

# Add catalog path to config
cfg["STRATEGY"]["PARAMS"]["catalog_path"] = str(catalog_path)

# Verify catalog
#instruments = catalog.instruments(instrument_type=TradeTick)  # takes too long on laptop. Use loader class instruments property instead
instruments = set(inst.id.value for inst in catalog.instruments())
print(f"\n✅ Catalog ready: {catalog.list_data_types()} data loaded")
print(f"Universe: {[str(symbol) for symbol in instruments]}")


# Initialize hyperparameter tuner
tuner = OptunaHparamsTuner(
    cfg=cfg,
    catalog=catalog,
    run_dir=logs_dir,
    seed=2025
)

print("Hyperparameter tuner initialized")
print(f"Model trials: {cfg['MODEL']['PARAMS']['n_trials']}")
print(f"Strategy trials: {cfg['STRATEGY']['PARAMS']['n_trials']}")


# Run model hyperparameter optimization
logger.info("\n" + "="*70)
logger.info("🔬 STAGE 2: MODEL HYPERPARAMETER OPTIMIZATION")
logger.info("="*70 + "\n")

model_results = tuner.optimize_model()

print("\n✅ Model optimization complete!")
print(f"Best model path: {model_results['model_path']}")
print(f"MLflow run ID: {model_results['mlflow_run_id']}")




# Run strategy hyperparameter optimization
logger.info("\n" + "="*70)
logger.info("📊 STAGE 3: STRATEGY HYPERPARAMETER OPTIMIZATION")
logger.info("="*70 + "\n")

model_name = cfg['MODEL']['PARAMS']['model_name']
strategy_results = tuner.optimize_strategy(model_name=model_name)

print("\n✅ Strategy optimization complete!")
print(f"Best hyperparameters: {strategy_results['hparams']}")
print(f"\nBest metrics:")
for metric, value in strategy_results['metrics'].items():
    print(f"  {metric}: {value:.4f}")
print(f"\nMLflow run ID: {strategy_results['mlflow_run_id']}")

