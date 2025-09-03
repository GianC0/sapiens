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

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.data import BacktestDataConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestRunConfig, BacktestEngineConfig, CacheConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model import Bar
from nautilus_trader.model.enums import OmsType, AccountType
from nautilus_trader.model.currencies import USD,EUR


from models.utils import freq2barspec, freq2pdoffset
from algos.strategy import LongShortStrategy
from ..models.utils import freq2pdoffset
from algos.engine.hparam_tuner import OptunaHparamsTuner
from algos.engine.data_loader import CsvBarLoader
from models.UMIModel import UMIModel
from nautilus_trader.examples.algorithms.twap import TWAPExecAlgorithm


def main():
    # loads configuration file
    cfg_path = Path("configs/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Setup directories
    logs_dir = Path(cfg["STRATEGY"]["PARAMS"]["logs_dir"]) / "../"  # log_dir is parent directory of strategy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = logs_dir / "backtests" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.basicConfig(filename=f'{run_dir}/backtest.log', encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)

    
    logger.info("\n" + "="*70)
    logger.info('STARTING HYPERPARAMETER OPTIMIZATION')
    logger.info("="*70 + "\n")


    # TODO: let these be set during strategy hp optimization
    # --- Engine Bootstrap --------------------------------------------
    start = cfg["STRATEGY"]["PARAMS"]["backtest_start"]
    end = cfg["STRATEGY"]["PARAMS"]["backtest_end"]
    venue = Venue(
            name="SIM",
            book_type="L1_MBP",     # bars are inluded in L1 market-by-price
            account_type=AccountType.CASH,
            base_currency=cfg["STRATEGY"]["PARAMS"]["currency"],
            starting_balances=str(cfg["STRATEGY"]["PARAMS"]["initial_cash"])+" "+str(cfg["STRATEGY"]["PARAMS"]["currency"]),
            bar_adaptive_high_low_ordering=False,  # Enable adaptive ordering of High/Low bar prices
            )
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            strategies=[LongShortStrategy(config=cfg)],
            trader_id="TESTER-001",
            ),
        data_configs=[
            BacktestDataConfig(
                catalog_path=":memory:",
                data_cls= Bar,
                start_time=start,
                end_time=end,
                bar_spec=freq2barspec(cfg["STRATEGY"]["PARAMS"]["freq"])
                )
            ],
        venues=[venue],
        cache=CacheConfig(
        #tick_capacity=0,  # I'm only trading on L1 - market by price
        bar_capacity=cfg["STRATEGY"]["PARAMS"]["engine"]["cache"]["bar_capacity"],    # Store last 5,000 bars per bar type: 5000 days ~ 13.5 years
        ),
        fill_model=FillModel(
            prob_fill_on_limit = cfg["STRATEGY"]["PARAMS"]["prob_fill_on_limit"],    # Chance a limit order fills when price matches (applied to bars/trades/quotes + L1/L2/L3 order book)
            prob_slippage =  cfg["STRATEGY"]["PARAMS"]["prob_slippage"],             # Chance of 1-tick slippage (applied to bars/trades/quotes + L1 order book only)
            random_seed=2025,   # random seed used also for hptuner 2, usefull for reproducibility
            ),
        )
    engine.add_exec_algorithm(TWAPExecAlgorithm())

    # TODO: rewrite tuner call
    # Initialize tuner
    tuner = OptunaHparamsTuner(
        cfg=cfg,
        n_model_trials=model_trials,
        n_strategy_trials=strategy_trials
    )
    


    loader     = CsvBarLoader(cfg=cfg, venue_name="SIM")
    
    # Add instruments to engine
    print(f"[INSTRUMENTS] Adding {len(loader.instruments)} instruments...")
    for symbol, instrument in loader.instruments.items():
        engine.add_instrument(instrument)
        print(f"  Added {symbol}: {instrument}")

    # Add historical data
    print("[DATA] Loading historical bars...")
    bar_count = 0
    for bar_or_feature in loader.bar_iterator():
        # Only add Bar objects to the engine (skip FeatureBarData)
        if isinstance(bar_or_feature, Bar):
            bar_ts = pd.Timestamp(bar_or_feature.ts_event, unit='ns', tz='UTC')
            # Only include bars in our date range
            if backtest_start <= bar_ts <= backtest_end:
                engine.add_data(bar_or_feature)
                bar_count += 1
                
                # Progress indicator
                if bar_count % 10000 == 0:
                    print(f"  Loaded {bar_count} bars...")


    # --- run ----------------------------------------------------------
    print("[run_backtest] starting")
    engine.run(start=start,end=end)
    print("[run_backtest] finished")

    # --- Generate reports --------------------------------------------
    # Get account statistics
    account = engine.cache.account_for_venue(venue)
    if account:
        print(f"Final Account Balance: {account.balance_total()}")
        print(f"Final Net Liquidation: {account.calculate_net_liquidation_value()}")


if __name__ == "__main__":
    main()