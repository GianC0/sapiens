#!/usr/bin/env python3
"""
CLI helper — spins up a Nautilus Trader BacktestEngine and runs GenericLongShortStrategy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.data import BacktestDataConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestRunConfig, BacktestEngineConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model import Bar
from nautilus_trader.model.enums import OmsType, AccountType


from algos.strategy import BacktestLongShortStrategy
from algos.engine.hparam_tuner import OptunaHparamsTuner, split_hparam_cfg
from algos.engine.data_loader import CsvBarLoader
from models.umi import UMIModel
from model.utils import freq2barspec
from nautilus_trader.examples.algorithms.twap import TWAPExecAlgorithm

def main():
    # loads configuration file
    cfg_path = Path("configs/config.yaml")  # CHANGED: Fixed config filename
    cfg = yaml.safe_load(cfg_path.read_text())

    # --- engine bootstrap --------------------------------------------
    start = pd.Timestamp(cfg["backtest_start"], tz="UTC")
    end = pd.Timestamp(cfg["backtest_end"], tz="UTC")
    venue = Venue(
            name="SIM",
            book_type="L1_MBP",     # bars are inluded in L1 market-by-price
            account_type=AccountType.CASH,
            base_currency=cfg["currency"],
            starting_balances=str(cfg["initial_cash"])+" "+str(cfg["currency"]),
            bar_adaptive_high_low_ordering=False,  # Enable adaptive ordering of High/Low bar prices
            )
    
    
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            strategies=[BacktestLongShortStrategy(config=cfg)],
            trader_id="TESTER-001",
        ),
        data_configs=[
            BacktestDataConfig(
                catalog_path=":memory:",
                data_cls= Bar,
                start_time=start,
                end_time=end,
                bar_spec=freq2barspec(cfg["freq"])
            )
        ],
        venues=[venue],
        run_config=BacktestRunConfig(
            engine_id="001",
            run_id="001",
        ),
        cache=CacheConfig(
        #tick_capacity=0,  # I'm only trading on L1 - market by price
        bar_capacity=cfg["engine"]["cache"]["bar_capacity"],    # Store last 5,000 bars per bar type: 5000 days ~ 13.5 years
        ),
        fill_model=FillModel(
            prob_fill_on_limit=0.2,    # Chance a limit order fills when price matches (applied to bars/trades/quotes + L1/L2/L3 order book)
            prob_slippage=0.5,         # Chance of 1-tick slippage (applied to bars/trades/quotes + L1 order book only)
            random_seed=None,          # Optional: Set for reproducible results
        ),
    )
    engine.add_exec_algorithm(TWAPExecAlgorithm())

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
            engine.add_data(bar_or_feature)
            bar_count += 1
            
            # Progress indicator
            if bar_count % 10000 == 0:
                print(f"  Loaded {bar_count} bars...")
    
    print(f"[DATA] Loaded {bar_count} total bars")


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