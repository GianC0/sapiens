#!/usr/bin/env python3
"""
CLI helper — spins up a Nautilus Trader SimulationEngine and runs UMIStrategy.

Examples
--------
```bash
python run_backtest.py --cfg configs/umi.yaml --gpus 4 --cpus 32
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import yaml
import pandas as pd

from nautilus_trader.simulation.engine import SimulationEngine
from nautilus_trader.common.clock import SimulationClock
from nautilus_trader.persistence.catalog import Catalog

from algos.strategy import UMIStrategy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/umi.yaml")
    p.add_argument("--gpus", type=int, default=1, help="GPUs reserved per run")
    p.add_argument("--cpus", type=int, default=4, help="CPUs reserved per run")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.cfg)
    cfg = yaml.safe_load(cfg_path.read_text())

    # --- engine bootstrap --------------------------------------------
    start = pd.Timestamp(cfg["backtest_start"], tz="UTC")
    end = pd.Timestamp(cfg["backtest_end"],   tz="UTC")
    currency = str(cfg["currency"])
    starting_cash = float(cfg.get("initial_cash", 100_000))
    clock = SimulationClock(start, end)
    catalog = Catalog(":memory:")

    engine = SimulationEngine(clock=clock, catalog=catalog, base_currency=currency, starting_cash=starting_cash )

    strat = UMIStrategy(clock, catalog, cfg_path)
    engine.add_strategy(strat)

    # --- run ----------------------------------------------------------
    print("[run_backtest] starting")
    engine.run()
    print("[run_backtest] finished")


if __name__ == "__main__":
    main()
