#!/usr/bin/env python3
"""
Nautilus Trader BacktestEngine and runs Strategies and Models hyper-parameter tuning.
"""
from __future__ import annotations
import argparse
import code
from pathlib import Path
from sklearn.metrics import precision_recall_curve
import yaml
import pandas as pd
from pandas.tseries.frequencies import to_offset
import logging
from logging import Logger
from datetime import datetime
import mlflow


from nautilus_trader.model.currencies import USD,EUR
from nautilus_trader.model.objects import Currency
from nautilus_trader.core.nautilus_pyo3 import CurrencyType

from models.utils import  yaml_safe
from algos.engine.hparam_tuner import OptunaHparamsTuner
from algos.engine.databento_loader import DatabentoTickLoader

def main():
    # loads configuration file
    cfg_path = Path("configs/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Setup some params type
    currency = cfg["STRATEGY"]["PARAMS"]["currency"]
    if currency == "USD":
        cfg["STRATEGY"]["PARAMS"]["currency"] = Currency(code='USD', precision=3, iso4217=840, name='United States dollar', currency_type = CurrencyType.FIAT ) #
    elif currency == "EUR":
        cfg["STRATEGY"]["PARAMS"]["currency"] = Currency(code='EUR', precision=3, iso4217=978, name='Euro', currency_type=CurrencyType.FIAT)

    # Setup directories
    logs_dir = Path("logs/")

    # Logging
    logging.basicConfig(    
            level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(logs_dir / "logging.log"), encoding='utf-8'),
            #logging.StreamHandler()  # This adds console output
        ]
    )
    logger = logging.getLogger(__name__)

    
    # Load DBN tick data to catalog
    logger.info("Loading Databento trade ticks...")
    
    loader = DatabentoTickLoader(cfg=cfg["STRATEGY"]["PARAMS"],venue_name=cfg["STRATEGY"]["PARAMS"]["venue_name"])
    catalog = loader.load_to_catalog()

    # Error handling
    if len(catalog.instruments()) == 0:
        logger.warning("No instruments in catalog")
        return {
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0,
        }, {}



    logger.info("\n" + "="*70)
    logger.info('STARTING HYPERPARAMETER OPTIMIZATION')
    logger.info("="*70 + "\n")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1 & 2: HYPERPARAMETER OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    
    # Initialize tuner
    tuner = OptunaHparamsTuner(
        cfg=cfg,
        loader=loader,
        run_dir=logs_dir,
        seed=2025
    )
    
    # Phase 1: Optimize model hyperparameters
    logger.info("Starting model hyperparameter optimization...")
    model_results = tuner.optimize_model()
    logger.info(f"Model optimization complete. Best model saved to: {model_results['model_path']}")
    
    # Phase 2: Optimize strategy hyperparameters
    logger.info("Starting strategy hyperparameter optimization...")
    strategy_results = tuner.optimize_strategy()
    logger.info(f"Strategy optimization complete. Best Metrics: {strategy_results['metrics']}")
    
    """
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: FINAL BACKTEST
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "="*60)
    logger.info("FINAL BACKTEST")
    logger.info("="*60 + "\n")

    # Get optimized configuration
    run = mlflow.get_run(strategy_results["mlflow_run_id"])
    optimization_id = run.data.tags.get("optimization_id")
    strategy_name = run.data.params.get("strategy_name")
    model_name = run.data.params.get("model_name")
    optimized_config_flat = tuner.get_best_full_params_flat(strategy_name=strategy_name, model_name=model_name)

    # Set backtest dates
    backtest_start = optimized_config_flat["STRATEGY"]["backtest_start"], tz="UTC"
    backtest_end = optimized_config_flat["STRATEGY"]["backtest_end"], tz="UTC"

    # Run final backtest with optional linking
    if not optimization_id:
        optimization_id = ''
    final_metrics, final_time_series = tuner.run_final_backtest(backtest_start, backtest_end, strategy_hpo_run_id=run.info.run_id, optimization_id= optimization_id)
    
    logger.info(f"Final backtest complete. Metrics logged to MLflow.")

    logger.info("\n" + "="*70)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*70)
    """

if __name__ == "__main__":
    main()