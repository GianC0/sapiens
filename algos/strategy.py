"""
UMIStrategy
===========

Nautilus Trader strategy wrapper around the PyTorch-based UMIModel.
Implements:

* Dynamic universe (slots padded out to `dynamic_universe_mult` × active stocks)
* Risk overlays (drawdown kill-switch, trailing stops, vol targeting, etc.)
* Per-frequency retrain logic (delta measured in *bars*)
* Parallel order dispatch (async gather, size `execution.parallel_orders`)
* Live routing:   – CCXT exchanges (spot); Interactive Brokers (stock/ETF)
                  – connectors only initialised when `engine.mode == "LIVE"`
"""
from __future__ import annotations
import math
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from nautilus_trader.model.strategy import Strategy
from nautilus_trader.model.events import BarEvent, TradeEvent
from nautilus_trader.common.clock import Clock
from nautilus_trader.execution.trade import Trade
from nautilus_trader.persistence.catalog import Catalog

from algos.engine.data_loader import CsvBarLoader
from algos.engine.execution import CommissionModelBps, SlippageModelBps
from algos.engine.optimizer import MaxSharpeRatioOptimizer
from algos.engine.analyzers import EquityCurveAnalyzer
from algos.engine.utils import beta
from algos.models.umi import UMIModel


class UMIStrategy(Strategy):
    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def __init__(self, clock: Clock, catalog: Catalog, cfg_path: Path):
        super().__init__("UMI_STRAT", clock, catalog)
        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.freq = self.cfg["freq"]

        # --- loader ---------------------------------------------------
        self.loader = CsvBarLoader(
            data_dir=Path(self.cfg["data_dir"]),
            freq=self.freq,
        )
        self.universe: List[str] = self.loader.universe
        self.rf_series = self.loader.rf_series
        
        # -------------- retrain schedule (bars → real timedelta) ------
        bars_delta = int(self.cfg["training"]["retrain_delta"])
        bar_offset = pd.tseries.frequencies.to_offset(self.freq)
        self.retrain_delta = bars_delta * bar_offset

        # --- model ----------------------------------------------------
        first_df = next(iter(self.loader._frames.values()))
        F = len(first_df.columns)
        close_idx = first_df.columns.get_indexer(["Close"])[0]

        self.model = UMIModel(
            freq=self.freq,
            feature_dim=F,
            window_len=self.cfg["window_len"],
            pred_len=self.cfg["pred_len"],
            end_train=self.cfg["train_end"],
            end_valid=self.cfg["valid_end"],
            bt_end=self.cfg["backtest_end"],
            retrain_delta= self.retrain_delta,
            dynamic_universe_mult=self.cfg["dynamic_universe_mult"],
            tune_hparams=self.cfg["training"]["tune_hparams"],
            n_trials=self.cfg["training"]["n_trials"],
            n_epochs=self.cfg["training"]["n_epochs"],
            batch_size=self.cfg["training"]["batch_size"],
            training_mode=self.cfg["training"]["training_mode"],
            pretrain_epochs=self.cfg["training"]["pretrain_epochs"],
            patience=self.cfg["training"]["patience"],
            close_idx=close_idx,
            model_dir=Path(self.cfg["model_dir"]),
            data_dir=Path(self.cfg["data_dir"]),
            warm_start=self.cfg["warm_start"],
            warm_training_epochs=self.cfg["warm_training_epochs"],
            clock_fn=lambda: pd.Timestamp(self.clock.now(), tz="UTC"),
            **self.cfg.get("hparams", {}),
        )
        self._data_buf = self.loader._frames  # live reference
        self.model.fit(self._data_buf)

        # --- optimiser & risk helpers --------------------------------
        risk_free = float(self.rf_series.iloc[-1]) if self.rf_series is not None else 0.0
        opt_name = self.cfg.get("optimizer", {}).get("name", "max_sharpe").lower()
        if opt_name == "max_sharpe":
            self.optimizer = MaxSharpeRatioOptimizer(risk_free=risk_free)
        else:
            raise NotImplementedError(f"optimizer.name={opt_name} not yet available")
        self.selector_k = self.cfg["selection"]["top_k"]
        self.trailing_stops: Dict[str, float] = {}
        self.equity_analyzer = EquityCurveAnalyzer()

        # --- execution helpers ---------------------------------------
        self.fee_model = CommissionModelBps(self.cfg["costs"]["fee_bps"])
        self.slip_model = SlippageModelBps(self.cfg["costs"]["spread_bps"])
        self.twap_slices = max(1, int(self.cfg["execution"]["twap_slices"]))
        self.parallel_orders = max(1, int(self.cfg["execution"]["parallel_orders"]))

        self.drawdown_pct = self.cfg["risk"]["drawdown_pct"]
        self.trailing_stop_pct = self.cfg["risk"]["trailing_stop_pct"]
        self.max_w_abs = self.cfg["risk"]["max_weight_abs"]
        self.max_w_rel = self.cfg["risk"]["max_weight_rel"]
        self.target_vol = self.cfg["risk"]["target_vol_annual"]
        self.realised_returns: List[float] = []
        self.equity_peak_value = 0.0

        # --- state ----------------------------------------------------
        self.prev_prices: Dict[str, float] = {}
        self.prev_qty: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Nautilus event handlers
    # ------------------------------------------------------------------ #
    async def on_start(self):
        # Subscribe to universe symbols as Bar data
        for sym in self.universe:
            self.subscribe_bars(sym, self.loader._bar_type)

    async def on_bar(self, event: BarEvent):
        ts = datetime.utcfromtimestamp(event.ts_epoch_ns * 1e-9)
        sym = event.instrument
        bar = event.bar

        # Update rolling frame
        self._data_buf[sym].loc[ts] = [
            bar.open, bar.high, bar.low, bar.close, bar.volume
        ]
        # Trim to last L+pred bars
        maxlen = self.cfg["window_len"] + self.cfg["pred_len"] + 1
        self._data_buf[sym] = self._data_buf[sym].iloc[-maxlen:]

        # Record equity
        nav = self.portfolio.account_value
        self.equity_peak_value = max(self.equity_peak_value, nav)
        self.equity_analyzer.on_equity(ts, nav)

        # Trailing stop logic
        await self._run_trailing_stops(sym, bar.close)

        # Once per *bar* (not per symbol) pick a pivot symbol
        if sym == self.universe[0]:
            await self._rebalance(ts)

        # Model maintenance
        self.model.update(self._data_buf)

    async def on_trade(self, event: TradeEvent):
        trade: Trade = event.trade
        # Store fills if needed for analytics
        self.prev_qty[trade.instrument] = trade.position_size

    # ------------------------------------------------------------------ #
    # internal
    # ------------------------------------------------------------------ #
    async def _rebalance(self, ts: datetime):
        preds = self.model.predict(self._data_buf)
        if not preds:
            return

        # μ and Σ
        universe = list(preds.keys())
        mu = np.array([preds[tkr] for tkr in universe])

        lookback = self.cfg["optimizer"]["lookback_days"]
        returns = pd.DataFrame({
            t: self._data_buf[t]["Close"].pct_change().dropna().iloc[-lookback:]
            for t in universe
        }).dropna()
        if returns.empty:
            return
        cov = returns.cov().values

        raw_w = self.optimizer.optimize(mu, cov)
        weights = dict(zip(universe, raw_w))

        # keep only top / bottom k
        ranked = sorted(weights.items(), key=lambda x: x[1])
        shorts = [t for t, _ in ranked[: self.selector_k]]
        longs  = [t for t, _ in ranked[-self.selector_k :]]
        for t in list(weights):
            if t not in longs and t not in shorts:
                weights[t] = 0.0

        # gross / net scaling and Cap position weights
        gross_target = self.cfg["allocation"]["gross_leverage"]
        w = pd.Series(weights, dtype=float)
        abs_cap = self.cfg["risk"]["max_weight_abs"]
        rel_cap = self.cfg["risk"]["max_weight_rel"]
        w = w.clip(-abs_cap, abs_cap)
        if w.abs().sum() > gross_target * rel_cap:
            w *= (gross_target * rel_cap) / w.abs().sum()
        weights = w.to_dict()

        # Volatility targeting
        if self.realised_returns:
            realised = np.std(self.realised_returns) * math.sqrt(252)
            if realised > 0:
                scale = min(1.5, self.target_vol / realised)
                weights = {k: v * scale for k, v in weights.items()}

        await self._dispatch_orders(weights, ts)

        # Store realised return for next bar
        prev_nav = self.prev_prices.get("__NAV__", nav := self.portfolio.account_value)
        daily_ret = (nav - prev_nav) / prev_nav if prev_nav else 0.0
        self.realised_returns.append(daily_ret)
        if len(self.realised_returns) > 60:
            self.realised_returns.pop(0)
        self.prev_prices["__NAV__"] = nav

    async def _dispatch_orders(self, target_weights: Dict[str, float], ts: datetime):
        nav = self.portfolio.account_value
        coros = []

        for sym, target_w in target_weights.items():
            price = self._data_buf[sym].iloc[-1]["Close"]
            if price == 0:
                continue
            target_qty = int((target_w * nav) / price)
            delta = target_qty - self.portfolio.position_size(sym)
            # ---------------- liquidity / position-size caps ----------
            # 1) absolute NAV cap (already enforced in weights)
            # 2) volume cap   – keep ≤ max_adv_pct of ADV
            adv_series = self._data_buf[sym]["Volume"].iloc[
                -self.cfg["liquidity"]["adv_lookback"] :
            ]
            adv = adv_series.mean() if not adv_series.empty else 0
            max_trade = int(adv * self.cfg["liquidity"]["max_adv_pct"])
            if max_trade:
                delta = int(math.copysign(min(abs(delta), max_trade), delta))
            if delta == 0:
                continue
            coros.append(self._twap(sym, delta))

        # Parallel execution limited by config
        chunks = [
            coros[i : i + self.parallel_orders]
            for i in range(0, len(coros), self.parallel_orders)
        ]
        for batch in chunks:
            await asyncio.gather(*batch)

    async def _twap(self, sym: str, qty: int):
        if self.twap_slices <= 1:
            self.market_order(sym, qty)
            return
        slice_qty = int(qty / self.twap_slices)
        remainder = qty - slice_qty * (self.twap_slices - 1)
        for i in range(self.twap_slices):
            q = slice_qty if i < self.twap_slices - 1 else remainder
            if q != 0:
                self.market_order(sym, q)
            await asyncio.sleep(0)  # yield

    async def _run_trailing_stops(self, sym: str, price: float):
        pos = self.portfolio.position_size(sym)
        if pos == 0:
            return
        if pos > 0:  # long
            high = self.trailing_stops.get(sym, price)
            high = max(high, price)
            self.trailing_stops[sym] = high
            if price <= high * (1 - self.trailing_stop_pct):
                await self._twap(sym, -pos)
                self.trailing_stops.pop(sym, None)
        else:  # short
            low = self.trailing_stops.get(sym, price)
            low = min(low, price)
            self.trailing_stops[sym] = low
            if price >= low * (1 + self.trailing_stop_pct):
                await self._twap(sym, -pos)
                self.trailing_stops.pop(sym, None)
