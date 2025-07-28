#!/usr/bin/env python3
"""
CLI helper — spins up a Nautilus Trader BacktestEngine and runs GenericLongShortStrategy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd
from pandas.tseries.frequencies import to_offset

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.data import BacktestDataConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestRunConfig, BacktestEngineConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model import Bar
from nautilus_trader.model.enums import OmsType, AccountType
from models.utils import freq2barspec, freq2pdoffset


from algos.strategy import BacktestLongShortStrategy
from ..models.utils import freq2pdoffset
from algos.engine.hparam_tuner import OptunaHparamsTuner, split_hparam_cfg
from algos.engine.data_loader import CsvBarLoader
from models.UMIModel import UMIModel
from nautilus_trader.examples.algorithms.twap import TWAPExecAlgorithm

def main():
    # loads configuration file
    cfg_path = Path("configs/config.yaml")  # CHANGED: Fixed config filename
    cfg = yaml.safe_load(cfg_path.read_text())

    # Calculate proper date ranges
    backtest_start = pd.Timestamp(cfg["backtest_start"], tz="UTC")
    train_offset = freq2pdoffset(cfg["train_offset"])
    valid_offset = freq2pdoffset(cfg["valid_offset"]) 
    test_offset = freq2pdoffset(cfg["test_offset"])
    backtest_end = pd.Timestamp(cfg["backtest_end"], tz="UTC")
    pred_offset = freq2pdoffset(cfg["freq"])*int(cfg["pred_len"])

    # Calculate key dates
    train_end = backtest_start + train_offset
    valid_end = train_end
    if cfg["training"]["tune_hparams"]:
        valid_end += valid_offset
    test_end = valid_end + test_offset
    
    # Update config with calculated dates
    cfg["train_start"] = backtest_start
    cfg["train_end"] = train_end.strftime("%Y-%m-%d")
    cfg["valid_start"] = cfg["train_end"] + to_offset("1ns") #ensures dataset segregation 
    cfg["valid_end"] = valid_end.strftime("%Y-%m-%d")#
    cfg["test start"] = cfg["valid_end"] + to_offset("1ns")
    cfg["test_end"] = test_end.strftime("%Y-%m-%d")
    cfg["walkfwd_start"] = cfg["test_end"] + to_offset("1ns")
    cfg["pred_offset"] = pred_offset

    assert backtest_end >= cfg["walkfwd_start"] + pred_offset

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