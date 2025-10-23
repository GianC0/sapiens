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
    
    # Get optimized configuration
    optimized_config_flat = tuner.get_best_config_flat()
    
    # Save optimized config
    optimized_config_flat_path = run_dir / f"{timestamp}_optimized_config_flat.yaml"
    with open(optimized_config_flat_path, 'w') as f:
        yaml.dump(optimized_config_flat, f)
    logger.info(f"Optimized configuration saved to: {optimized_config_flat_path}")
    
    
""" 
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: FINAL BACKTEST
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "="*60)
    logger.info("FINAL BACKTEST")
    logger.info("="*60 + "\n")
    
    # Set backtest dates
    backtest_start = pd.Timestamp(optimized_config_flat["STRATEGY"]["backtest_start"], tz="UTC")
    backtest_end = pd.Timestamp(optimized_config_flat["STRATEGY"]["backtest_end"], tz="UTC")

    # Test directory
    test_dir = run_dir / "test" 

    logger.info(f"Running final backtest from {backtest_start} to {backtest_end}")
    
    with mlflow.start_run(
        run_name="Backtest",
        nested=True,
        parent_run_id=tuner.parent_run.info.run_id
    ) as final_run:
        # Log period and configuration
        mlflow.log_params({
            "period": "out_of_sample",
            "start": str(backtest_start),
            "end": str(backtest_end),
        })
        mlflow.log_artifact(str(optimized_config_flat_path))
        mlflow.log_param("test_directory", str(test_dir))
        
        # Run backtest
        final_metrics, final_time_series = tuner._backtest(
            model_params_flat = optimized_config_flat["MODEL"],
            strategy_params_flat = optimized_config_flat["STRATEGY"],
            start = backtest_start,
            end = backtest_end,
        )


        tuner._generate_performance_charts(
            time_series=final_time_series,
            strategy_params_flat=optimized_config_flat["STRATEGY"],
            trial_number=-1,  # -1 indicates final backtest
            output_dir=test_dir
        )
        
        
        # Log all metrics
        for metric_name, metric_value in final_metrics.items():
            mlflow.log_metric(f"{metric_name}", metric_value)
        
        # Save detailed report
        report = {
            "final_metrics": final_metrics,
            "optimized_model_params": yaml_safe(model_results),
            "optimized_strategy_params": yaml_safe(strategy_results),
            "backtest_period": {
                "start": str(backtest_start),
                "end": str(backtest_end)
            }
        }
        
        report_path = run_dir / f"{timestamp}_backtest_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f)
        mlflow.log_artifact(str(report_path))
        
        logger.info(f"Final backtest complete:")

    # Close MLflow parent run
    mlflow.end_run()
    
    logger.info("\n" + "="*70)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*70)
"""

if __name__ == "__main__":
    main()