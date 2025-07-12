"""
Generic long/short strategy for Nautilus Trader.

* Consumes ANY model that follows interfaces.MarketModel
* Data feed provided by CsvBarLoader (FeatureBarData + Bar)
* Risk controls: draw-down, trailing stops, ADV cap, fee/slippage models
"""
from __future__ import annotations

import asyncio
import importlib
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from nautilus_trader.common.component import Clock
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.execution import Trade
from nautilus_trader.model.events import BarEvent, TradeEvent


# ----- project imports -----------------------------------------------------
from ..models.interfaces import MarketModel
from engine.data_loader import CsvBarLoader
from ..models.utils.cache_adapter import cache_to_dict

#  risk / execution helpers (same modules you used before)
from algos.engine.execution import CommissionModelBps, SlippageModelBps
from algos.engine.optimizer import MaxSharpeRatioOptimizer
from algos.engine.analyzers import EquityCurveAnalyzer


# ========================================================================== #
# Strategy
# ========================================================================== #
class GenericLongShortStrategy(Strategy):
    """
    Long/short equity strategy, model-agnostic & frequency-agnostic.

    YAML keys used
    --------------
    model_name:          umi | my_gpt_model | …
    freq:                "1D" | "15m" | …
    data_dir:            root folder with stocks/  bonds/
    window_len, pred_len
    training:            {n_epochs,…}
    selection.top_k      (long+short per side)
    costs.fee_bps, costs.spread_bps
    execution.twap_slices, execution.parallel_orders
    liquidity.adv_lookback, liquidity.max_adv_pct
    risk.trailing_stop_pct, risk.drawdown_pct,
         risk.max_weight_abs, risk.max_weight_rel, risk.target_vol_annual
    optimizer.name:      max_sharpe | none
    """

    # ------------------------------------------------------------------ #
    def __init__(self, clock: Clock, catalog, cfg_path: Path):
        super().__init__("GENERIC_LS", clock, catalog)

        self.cfg = yaml.safe_load(Path(cfg_path).read_text())

        # 1) ---------------- data loader ------------------------------
        self.loader = CsvBarLoader(
            root=Path(self.cfg["data_dir"]),
            freq=self.cfg["freq"],
        )
        self.universe: List[str] = self.loader.universe
        self._bar_type = self.loader.bar_type

        # 2) ---------------- model (dynamic import) -------------------
        self.model: MarketModel = self._build_model()

        # 3) ---------------- optimiser --------------------------------
        opt_name = str(self.cfg.get("optimizer", {}).get("name", "max_sharpe")).lower()
        rf = (
            float(self.loader.rf_series.iloc[-1])
            if self.loader.rf_series is not None
            else 0.0
        )
        self.optimizer = MaxSharpeRatioOptimizer(risk_free=rf) if opt_name == "max_sharpe" else None

        self.selector_k = int(self.cfg["selection"]["top_k"])

        # 4) ---------------- risk helpers -----------------------------
        self.trailing_stop_pct = float(self.cfg["risk"]["trailing_stop_pct"])
        self.drawdown_pct = float(self.cfg["risk"]["drawdown_pct"])
        self.max_w_abs = float(self.cfg["risk"]["max_weight_abs"])
        self.max_w_rel = float(self.cfg["risk"]["max_weight_rel"])
        self.target_vol = float(self.cfg["risk"]["target_vol_annual"])
        self.trailing_stops: Dict[str, float] = {}
        self.realised_returns: List[float] = []
        self.equity_peak_value: float = 0.0
        self.equity_analyzer = EquityCurveAnalyzer()

        # 5) ---------------- execution --------------------------------
        self.fee_model = CommissionModelBps(float(self.cfg["costs"]["fee_bps"]))
        self.slip_model = SlippageModelBps(float(self.cfg["costs"]["spread_bps"]))
        self.twap_slices = max(1, int(self.cfg["execution"]["twap_slices"]))
        self.parallel_orders = max(1, int(self.cfg["execution"]["parallel_orders"]))
        self.adv_lookback = int(self.cfg["liquidity"]["adv_lookback"])
        self.max_adv_pct = float(self.cfg["liquidity"]["max_adv_pct"])

    # ================================================================= #
    # Nautilus event handlers
    # ================================================================= #
    async def on_start(self):
        """Subscribe to bars for every symbol."""
        for sym in self.universe:
            self.subscribe_bars(sym, self._bar_type)

    async def on_bar(self, event: BarEvent):
        ts = event.ts_event.to_pydatetime()
        sym = event.instrument
        bar = event.bar

        # ------------ trailing-stop maintenance ----------------------
        await self._run_trailing_stops(sym, bar.close)

        # ------------ equity / draw-down log ------------------------
        nav = self.portfolio.account_value
        self.equity_peak_value = max(self.equity_peak_value, nav)
        self.equity_analyzer.on_equity(ts, nav)
        if nav < self.equity_peak_value * (1 - self.drawdown_pct):
            await self._liquidate_all()
            return

        # ------------ run rebalance once per bar time-stamp ---------
        if sym != self.universe[0]:
            return                                                     # wait for pivot symbol

        snap = cache_to_dict(
            self.cache,
            tickers=self.universe,
            lookback=self.model.L + self.model.pred_len + 1,
        )

        preds = self.model.predict(snap)
        if not preds:
            return

        # === weights =================================================
        weights = self._compute_target_weights(preds)

        # === place orders ============================================
        await self._dispatch_orders(weights, ts)

        # === realised return bookkeeping ============================
        prev_nav = self.state.get("prev_nav", nav)
        if prev_nav:
            self.realised_returns.append((nav - prev_nav) / prev_nav)
            if len(self.realised_returns) > 60:
                self.realised_returns.pop(0)
        self.state["prev_nav"] = nav

        # === model upkeep ===========================================
        self.model.update(snap)

    async def on_trade(self, event: TradeEvent):
        trade: Trade = event.trade
        self.state[f"qty_{trade.instrument}"] = trade.position_size

    # ================================================================= #
    # internal helpers
    # ================================================================= #
    # ----- model factory ----------------------------------------------
    def _build_model(self) -> MarketModel:
        name = str(self.cfg["model_name"]).lower()
        mod = importlib.import_module(f"models.{name}")
        ModelCls = getattr(mod, "Model", None) or getattr(mod, f"{name.upper()}Model")
        if ModelCls is None:
            raise ImportError(f"models.{name} must export a Model class")
        first_df = next(iter(self.loader._frames.values()))
        model: MarketModel = ModelCls(
            freq=self.cfg["freq"],
            feature_dim=len(first_df.columns),
            window_len=self.cfg["window_len"],
            pred_len=self.cfg["pred_len"],
            bar_type=self._bar_type,
            end_train=self.cfg["train_end"],
            end_valid=self.cfg["valid_end"],
            **self.cfg.get("hparams", {}),
        )
        model.fit(self.loader._frames, n_epochs=self.cfg["training"]["n_epochs"])
        return model

    # ----- weight optimiser ------------------------------------------
    def _compute_target_weights(self, preds: Dict[str, float]) -> Dict[str, float]:
        mu = np.array([preds[s] for s in self.universe])
        if self.optimizer is None:
            w = pd.Series(mu, index=self.universe)
        else:
            lookback = int(self.cfg["optimizer"].get("lookback_days", 60))
            returns = pd.DataFrame({
                s: self.cache.bars(self._bar_type, limit=lookback, instrument=s)
                .to_pandas()["close"].pct_change()
                for s in self.universe
            }).dropna()
            cov = returns.cov().values if not returns.empty else np.diag(np.ones_like(mu))
            w_vec = self.optimizer.optimize(mu, cov)
            w = pd.Series(w_vec, index=self.universe)

        # ------- keep only top / bottom k ----------------------------
        shorts = w.nsmallest(self.selector_k)
        longs  = w.nlargest(self.selector_k)
        w = pd.concat([shorts, longs]).reindex(self.universe).fillna(0.0)

        # ------- cap position weights --------------------------------
        w = w.clip(-self.max_w_abs, self.max_w_abs)
        if w.abs().sum() > self.max_w_rel:
            w *= self.max_w_rel / w.abs().sum()

        # ------- vol targeting ---------------------------------------
        if self.realised_returns:
            realised_vol = np.std(self.realised_returns) * math.sqrt(252)
            if realised_vol > 0:
                scale = min(1.5, self.target_vol / realised_vol)
                w *= scale
        return w.to_dict()

    # ----- execution --------------------------------------------------
    async def _dispatch_orders(self, target_w: Dict[str, float], ts: datetime):
        nav = self.portfolio.account_value
        coros: List[asyncio.Task] = []

        for sym, target_w in target_w.items():
            price = self.cache.bar(sym, self._bar_type, index=0).close
            if price == 0:
                continue
            target_qty = int((target_w * nav) / price)
            delta = target_qty - self.portfolio.position_size(sym)

            # -------- ADV cap ----------------------------------------
            bars = self.cache.bars(self._bar_type, instrument=sym, limit=self.adv_lookback)
            volumes = [b.volume for b in bars] or [0.0]
            max_trade = int(np.mean(volumes) * self.max_adv_pct)
            if max_trade:
                delta = int(math.copysign(min(abs(delta), max_trade), delta))
            if delta == 0:
                continue

            # -------- TWAP slicing -----------------------------------
            slice_qty = int(delta / self.twap_slices)
            for i in range(self.twap_slices - 1):
                coros.append(self._async_mkt(sym, slice_qty))
            remainder = delta - slice_qty * (self.twap_slices - 1)
            coros.append(self._async_mkt(sym, remainder))

        # parallel_batches
        for batch in [coros[i:i + self.parallel_orders] for i in range(0, len(coros), self.parallel_orders)]:
            await asyncio.gather(*batch)

    async def _async_mkt(self, sym: str, qty: int):
        if qty:
            self.market_order(sym, qty)

    # ----- trailing stops --------------------------------------------
    async def _run_trailing_stops(self, sym: str, price: float):
        pos = self.portfolio.position_size(sym)
        if pos == 0:
            self.trailing_stops.pop(sym, None)
            return

        if pos > 0:                       # long
            high = self.trailing_stops.get(sym, price)
            if price > high:
                self.trailing_stops[sym] = price
            elif price <= high * (1 - self.trailing_stop_pct):
                await self._async_mkt(sym, -pos)
                self.trailing_stops.pop(sym, None)
        else:                             # short
            low = self.trailing_stops.get(sym, price)
            if price < low:
                self.trailing_stops[sym] = price
            elif price >= low * (1 + self.trailing_stop_pct):
                await self._async_mkt(sym, -pos)
                self.trailing_stops.pop(sym, None)

    # ----- emergency liquidate ---------------------------------------
    async def _liquidate_all(self):
        for sym in self.universe:
            qty = self.portfolio.position_size(sym)
            if qty != 0:
                await self._async_mkt(sym, -qty)
        self.trailing_stops.clear()
