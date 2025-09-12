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

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestDataConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestRunConfig, BacktestEngineConfig, CacheConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model import Bar
from nautilus_trader.model.enums import OmsType, AccountType
from nautilus_trader.model.currencies import USD,EUR


from models.utils import freq2barspec, freq2pdoffset
from algos.LongShortStrategy import LongShortStrategy
from models.utils import freq2pdoffset
from algos.engine.hparam_tuner import OptunaHparamsTuner
from algos.engine.data_loader import CsvBarLoader
from models.UMIModel import UMIModel
from nautilus_trader.examples.algorithms.twap import TWAPExecAlgorithm


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
    logger.info(f"Strategy optimization complete. Best Sharpe: {strategy_results['sharpe_ratio']:.4f}")
    
    # Get optimized configuration
    optimized_config = tuner.get_best_config()
    
    # Save optimized config
    optimized_config_path = run_dir / f"{timestamp}_optimized_config.yaml"
    with open(optimized_config_path, 'w') as f:
        yaml.dump(optimized_config, f)
    logger.info(f"Optimized configuration saved to: {optimized_config_path}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: FINAL BACKTEST ON OUT-OF-SAMPLE PERIOD
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "="*60)
    logger.info("FINAL BACKTEST ON OUT-OF-SAMPLE PERIOD")
    logger.info("="*60 + "\n")
    
    # Use walk-forward period for final backtest
    backtest_start = pd.Timestamp(cfg["STRATEGY"]["PARAMS"]["backtest_start"], tz="UTC")
    backtest_end = pd.Timestamp(cfg["STRATEGY"]["PARAMS"]["backtest_end"], tz="UTC")
    
    # Initialize strategy with optimized parameters
    strategy_class = tuner.strategy_class
    strategy = strategy_class(config=optimized_config)
    
    # Setup venue
    venue = Venue(
        name="SIM",
        book_type="L1_MBP",
        account_type=AccountType.CASH,
        base_currency=cfg["STRATEGY"]["PARAMS"]["currency"],
        starting_balances=f"{cfg['STRATEGY']['PARAMS']['initial_cash']} {cfg['STRATEGY']['PARAMS']['currency']}",
    )
    
    # Initialize BacktestEngine for final test
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            strategies=[strategy],
            trader_id="FINAL-001",
        ),
        data_configs=[
            BacktestDataConfig(
                catalog_path=":memory:",
                data_cls=Bar,
                start_time=backtest_start,
                end_time=backtest_end,
                bar_spec=freq2barspec(cfg["STRATEGY"]["PARAMS"]["freq"])
            )
        ],
        venues=[venue],
        cache=CacheConfig(
            bar_capacity=cfg["STRATEGY"]["PARAMS"]["engine"]["cache"]["bar_capacity"]
        ),
        fill_model=FillModel(
            prob_fill_on_limit=cfg["STRATEGY"]["PARAMS"]["costs"]["prob_fill_on_limit"],
            prob_slippage=cfg["STRATEGY"]["PARAMS"]["costs"]["prob_slippage"],
            random_seed=2025,
        ),
    )
    
    # Load data
    loader = CsvBarLoader(cfg=cfg, venue_name="SIM")
    
    # Add instruments
    for symbol, instrument in loader.instruments.items():
        engine.add_instrument(instrument)
    
    # Add historical data
    logger.info("Loading historical data for final backtest...")
    bar_count = 0
    for bar_or_feature in loader.bar_iterator():
        if isinstance(bar_or_feature, Bar):
            bar_ts = pd.Timestamp(bar_or_feature.ts_event, unit='ns', tz='UTC')
            if backtest_start <= bar_ts <= backtest_end:
                engine.add_data(bar_or_feature)
                bar_count += 1
    
    logger.info(f"Loaded {bar_count} bars for final backtest")
    
    # Run final backtest
    logger.info("Running final backtest...")
    with mlflow.start_run(run_name="Final_Backtest", nested=False) as final_run:
        # Log optimized parameters
        mlflow.log_params({
            "period": "out_of_sample",
            "start": str(backtest_start),
            "end": str(backtest_end),
        })
        mlflow.log_artifact(str(optimized_config_path))
        
        # Run engine
        engine.run(start=backtest_start, end=backtest_end)
        
        # Generate and log reports
        account = engine.cache.account_for_venue(venue)
        portfolio_stats = engine.analyzer.get_portfolio_stats()
        
        # Calculate final metrics
        final_balance = float(account.balance_total(currency=cfg["STRATEGY"]["PARAMS"]["currency"]))
        initial_balance = float(cfg["STRATEGY"]["PARAMS"]["initial_cash"])
        total_return = (final_balance / initial_balance) - 1
        
        # Log metrics
        mlflow.log_metrics({
            "final_balance": final_balance,
            "total_return": total_return,
            "num_trades": len(list(engine.cache.positions_closed())),
        })
        
        # Save detailed report
        report = {
            "final_balance": final_balance,
            "total_return": total_return,
            "portfolio_stats": portfolio_stats,
            "optimized_model_params": model_results,
            "optimized_strategy_params": strategy_results,
        }
        
        report_path = run_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_final_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f)
        mlflow.log_artifact(str(report_path))
        
        logger.info(f"Final backtest complete:")
        logger.info(f"  Final Balance: {final_balance:.2f}")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()