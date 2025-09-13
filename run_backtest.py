#!/usr/bin/env python3
"""
Nautilus Trader BacktestEngine and runs Strategies and Models hyper-parameter tuning.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd
from pandas.tseries.frequencies import to_offset
import logging
from logging import Logger
from datetime import datetime
import mlflow



from nautilus_trader.model.currencies import USD,EUR
from models.utils import  yaml_safe
from algos.engine.hparam_tuner import OptunaHparamsTuner


def main():
    # loads configuration file
    cfg_path = Path("configs/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Setup some params type
    currency = cfg["STRATEGY"]["PARAMS"]["currency"]
    if currency == "USD":
        cfg["STRATEGY"]["PARAMS"]["currency"] = USD
    elif currency == "EUR":
        cfg["STRATEGY"]["PARAMS"]["currency"] = EUR

    # Setup directories
    logs_dir = Path(cfg["STRATEGY"]["PARAMS"]["logs_dir"]).parent  # log_dir is parent directory of strategy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = logs_dir / "backtests" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.basicConfig(filename=f'{run_dir}/backtest.log', encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)

    
    logger.info("\n" + "="*70)
    logger.info('STARTING HYPERPARAMETER OPTIMIZATION')
    logger.info("="*70 + "\n")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1 & 2: HYPERPARAMETER OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    
    # Initialize tuner
    tuner = OptunaHparamsTuner(
        cfg=cfg,
        run_timestamp=timestamp,
        run_dir=run_dir,
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
    
    
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: FINAL BACKTEST
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "="*60)
    logger.info("FINAL BACKTEST")
    logger.info("="*60 + "\n")
    
    # Set backtest dates
    backtest_start = pd.Timestamp(optimized_config_flat["STRATEGY"]["backtest_start"], tz="UTC")
    backtest_end = pd.Timestamp(optimized_config_flat["STRATEGY"]["backtest_end"], tz="UTC")

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
        
        # Run backtest
        final_metrics = tuner._backtest(
            model_params_flat = optimized_config_flat["MODEL"],
            strategy_params_flat = optimized_config_flat["STRATEGY"],
            start = backtest_start,
            end = backtest_end,
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


if __name__ == "__main__":
    main()