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
from nautilus_trader.common.component import init_logging
from nautilus_trader.common.component import Logger
from nautilus_trader.common.component import Clock
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide

# ----- project imports -----------------------------------------------------
from .models.interfaces import MarketModel
from .engine.data_loader import CsvBarLoader, FeatureBarData
from ..models.utils import cache_to_dict, freq2pandas, freq2barspec

#  risk / execution helpers (same modules you used before)
from .engine.execution import CommissionModelBps, SlippageModelBps
from .engine.optimizer import MaxSharpeRatioOptimizer
from .engine.analyzers import EquityCurveAnalyzer


# ========================================================================== #
# Strategy
# ========================================================================== #
class BacktestLongShortStrategy(Strategy):
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
    def __init__(self, config: dict):  
        super().__init__()  

        self.cfg = config

        # ---------------- Universe Management ------------------------------

        self.bar_spec = freq2barspec(self.cfg["freq"])
        self.min_bars_required = self.cfg["batch_size"]
        self.walkfwd_start = self.cfg["walkfwd_start"]
        loader = CsvBarLoader(cfg=self.cfg)
        candidate_universe = loader.universe
        
        selected = []
        
        for ticker in candidate_universe:
            if ticker not in loader._frames:
                continue
                
            df = loader._frames[ticker]
            
            # Check 1: Has enough historical data before train_end
            train_data = df[df.index < walkfwd_start ]
            if len(train_data) < self.min_bars_required:
                #TODO: implement proper nautilus logger
                print(f"[Universe] Skipping {ticker}: insufficient training data ({len(train_data)} < {self.min_bars_required})")
                continue
                
            # Check 2: Active at test start (has data around test start)
            test_window = df[(df.index >= self.test_start - pd.DateOffset(days=5)) & 
                           (df.index <= self.test_start + pd.DateOffset(days=5))]
            if len(test_window) == 0:
                #TODO: implement proper nautilus logger
                print(f"[Universe] Skipping {ticker}: not active at test start")
                continue
                
            selected.append(ticker)
            
        print(f"[Universe] Selected {len(selected)} from {len(candidate_universe)} candidates")
        self.universe=sorted(selected)


        #TODO: to add cfg["strategy"] as first dimension

        # # 3) ---------------- optimiser --------------------------------
        # opt_name = str(self.cfg.get("optimizer", {}).get("name", "max_sharpe")).lower()
        # rf = (
        #     float(self.loader.rf_series.iloc[-1])
        #     if self.loader.rf_series is not None
        #     else 0.0
        # )
        # self.optimizer = MaxSharpeRatioOptimizer(risk_free=rf) if opt_name == "max_sharpe" else None

        # self.selector_k = int(self.cfg["selection"]["top_k"])

        # # 4) ---------------- risk helpers -----------------------------
        # self.trailing_stop_pct = float(self.cfg["risk"]["trailing_stop_pct"])
        # self.drawdown_pct = float(self.cfg["risk"]["drawdown_pct"])
        # self.max_w_abs = float(self.cfg["risk"]["max_weight_abs"])
        # self.max_w_rel = float(self.cfg["risk"]["max_weight_rel"])
        # self.target_vol = float(self.cfg["risk"]["target_vol_annual"])
        # self.trailing_stops: Dict[str, float] = {}
        # self.realised_returns: List[float] = []
        # self.equity_peak_value: float = 0.0
        # self.equity_analyzer = EquityCurveAnalyzer()

        # # 5) ---------------- execution --------------------------------
        # self.fee_model = CommissionModelBps(float(self.cfg["costs"]["fee_bps"]))
        # self.slip_model = SlippageModelBps(float(self.cfg["costs"]["spread_bps"]))
        # self.twap_slices = max(1, int(self.cfg["execution"]["twap_slices"]))
        # self.parallel_orders = max(1, int(self.cfg["execution"]["parallel_orders"]))
        # self.adv_lookback = int(self.cfg["liquidity"]["adv_lookback"])
        # self.max_adv_pct = float(self.cfg["liquidity"]["max_adv_pct"])

    # ================================================================= #
    # Nautilus event handlers
    # ================================================================= #
    def on_start(self): 
        """Initialize strategy."""

        # assert now + freq2pdoffset(self.cfg["holdout"] < now + pred_len 

        self.clock.set_timer(
            name="update_timer",  
            interval=freq2pdoffset(self.cfg["freq"]),  # Timer interval
            callback=self.on_update,  # Custom callback function invoked on timer
        )

        self.clock.set_timer(
            name="holdout_timer",  
            interval=freq2pdoffset(self.cfg["holdout"]),  # Timer interval
            callback=self.on_holdout,  # Custom callback function invoked on timer
        )

        self.clock.set_timer(
            name="retrain_timer",  
            interval=freq2pdoffset(self.cfg["retrain_delta"]),  # Timer interval
            callback=self.on_retrain,  # Custom callback function invoked on timer
        )

        

        # INITIAL MODEL TRAINING STEPS
        # setup train dataset, with start/end date
        loader     = CsvBarLoader(cfg=cfg, venue_name="SIM")
        data_dict  = loader._frames
        ### SETUP HPARAMS TUNING
        # pulls defaults and hparams search space from cfg file
        defaults, search_space = split_hparam_cfg(cfg["hparams"])
        if cfg["training"]["tune_hparams"]:
            print("[HPO] Optuna search started …")



            # CHANGED: Create clock function to pass to model
            def clock_fn():
                return pd.Timestamp.utcnow()

            fixed = dict(
                freq        = cfg["freq"],
                feature_dim = len(next(iter(data_dict.values())).columns),
                window_len  = cfg["window_len"],
                pred_len    = cfg["pred_len"],
                end_train   = cfg["train_end"],
                end_valid   = cfg["valid_end"],
                batch_size  = cfg["training"]["batch_size"],
                retrain_delta = pd.DateOffset(days=cfg["training"]["retrain_delta"]),
                dynamic_universe_mult = cfg["dynamic_universe_mult"],
                data_dir    = Path(cfg["data_dir"]),
                bar_spec    = freq2barspec(cfg["freq"]) # TODO: doublecheck if necessary
                **defaults,
            )

            tuner = OptunaHparamsTuner(   # TODO: check training logic and if can rigthly start to trade
                model_name  = cfg["model_name"],
                logs_dir    = Path(cfg.get("logs_dir", "logs")),
                model_cls   = UMIModel,
                train_dict  = data_dict,
                search_space= search_space,
                defaults    = defaults, 
                fixed_kwargs= fixed,
                fit_kwargs  = dict(n_epochs=cfg["training"]["n_epochs"]),
                study_name  = f"{model_name}_hpo",  
                n_trials    = cfg["training"]["n_trials"],
            )
            best = tuner.optimize()
            cfg["hparams"] = {**defaults, **best["params"]}
            print(f"Optuna tuning complete. Best hparams: {best["params"]}")
        else:
            cfg["hparams"] = defaults
        
        # setup universe and ensure enough history for each stock and let universe be universe_mult*n_instruments
        # model.initialize(): get universe at the beginning and universe at the end and do set_end - set_beg -> initialize only stocks live at end_t+1. handle case when stock is still not there. ensure all stocks have min last history to be counted
        # set self.now or similar

        for sym in self.universe:
            self.subscribe_bars(self._bar_type, sym)  

    def on_update(self, event: TimeEvent):
        if event.name == "update_timer":
            # the logic should be the following:
            # update prediction for the next pred_len = now + holdout - holdout_start
            # checks for stocks events (new or delisted) and retrain_delta and if nothing happens directly calls predict() method of the model (which should happen only if holdout period in bars is passed (otherwise do not do anything), and this variable is in the config.yaml); if retrain delta is passed without stocks event, it calls update() which runs a warm start fit() on the new training window and then the strategy calls predict(); if stock event happens then the strategy calls update() and then predict(). update() should manage the following: if delisting then just use the model active mask to cut off those stocks (so that future preds and training loss are not computed for these. the model is ready to predict after that) but if new stocks are added it checks for universe availability and enough history in the Cache for those new stocks and then calls an initialization. ensure that the most up to date mask is always set by update (or by initialize only at the start) so that the other functions use always the most up to date mask.

    def on_holdout(self, event: TimeEvent):
        if event.name == "holdout_timer":
            # new prediction up until re-optimize portfolio

    def on_retrain(self, event: TimeEvent):
        elif event.name == "retrain_timer":
    
            

    # ================================================================= #
    # DATA handlers
    # ================================================================= #
    def on_instrument(self, instrument: Instrument) -> None:
    def on_instrument_status(self, data: InstrumentStatus) -> None:
    def on_instrument_close(self, data: InstrumentClose) -> None:
    def on_historical_data(self, data: Data) -> None:
    def on_bar(self, bar: Bar):  
        # assuming on_timer happens before then on_bars are called first,
        ts = bar.ts_event
        sym = bar.instrument_id.symbol
        
        # ------------ trailing-stop maintenance ----------------------
        self._run_trailing_stops(sym, bar.close)  

        # ------------ equity / draw-down log ------------------------
        # CHANGED: Updated portfolio access methods
        nav = self.portfolio.net_liquidation
        self.equity_peak_value = max(self.equity_peak_value, nav)
        self.equity_analyzer.on_equity(ts, nav)
        if nav < self.equity_peak_value * (1 - self.drawdown_pct):
            self._liquidate_all()  
            return

        # ------------ run rebalance once per bar time-stamp ---------
        if sym != self.universe[0]:
            return                                                     

        # CHANGED: Pass cache directly to model predict
        preds = self.model.predict(self.cache)
        if not preds:
            return

        # === weights =================================================
        weights = self._compute_target_weights(preds)

        # === place orders ============================================
        self._dispatch_orders(weights, ts)  # CHANGED: Removed await

        # === realised return bookkeeping ============================
        # CHANGED: Fixed state access
        prev_nav = getattr(self, '_prev_nav', nav)
        if prev_nav:
            self.realised_returns.append((nav - prev_nav) / prev_nav)
            if len(self.realised_returns) > 60:
                self.realised_returns.pop(0)
        self._prev_nav = nav

        # === model upkeep ===========================================
        self.model.update(self.cache)


    # ================================================================= #
    # ORDER MANAGEMENT
    # ================================================================= #

    def on_order_filled(self, event: OrderFilled): 
        """Update position tracking on fills."""
        
        order = event.order
        position = self.portfolio.position(order.instrument_id)
        if position:
            self._position_qty[order.instrument_id.symbol.value] = position.quantity


    # ================================================================= #
    # POSITION MANAGEMENT
    # ================================================================= #

    # ================================================================= #
    # ACCOUNT & PORTFOLIO 
    # ================================================================= #


    # ================================================================= #
    # internal helpers
    # ================================================================= #
    # ----- model factory ----------------------------------------------
    def _build_model(self) -> MarketModel:
        name = str(self.cfg["model_name"]).lower()
        mod = importlib.import_module(f"models.{name}.{name}")
        ModelCls = getattr(mod, "Model", None) or getattr(mod, f"{name.upper()}Model")
        if ModelCls is None:
            raise ImportError(f"models.{name}.{name} must export a Model class")
        
        
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
            
            returns = pd.DataFrame()
            for s in self.universe:
                bars = []
                for bar in self.cache.bars(self._bar_type):
                    if bar.instrument_id.symbol == s:
                        bars.append(bar)
                if len(bars) >= 2:
                    closes = [b.close for b in bars[-lookback:]]
                    ret = pd.Series(closes).pct_change().dropna()
                    returns[s] = ret
            
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
    def _dispatch_orders(self, target_w: Dict[str, float], ts: datetime):  # CHANGED: Removed async
        nav = self.portfolio.net_liquidation
        
        for sym, target_weight in target_w.items():
            # CHANGED: Fixed bar and position access
            instrument_id = self.instruments[sym]
            bar = None
            for b in self.cache.bars(self._bar_type):
                if b.instrument_id == instrument_id:
                    bar = b
                    break
            
            if not bar or bar.close == 0:
                continue
                
            price = bar.close
            target_qty = int((target_weight * nav) / price)
            
            # CHANGED: Fixed position access
            position = self.portfolio.position(instrument_id)
            current_qty = position.quantity if position else 0
            delta = target_qty - current_qty

            # -------- ADV cap ----------------------------------------
            volumes = []
            for b in self.cache.bars(self._bar_type):
                if b.instrument_id == instrument_id:
                    volumes.append(b.volume)
            volumes = volumes[-self.adv_lookback:] if volumes else [0.0]
            
            max_trade = int(np.mean(volumes) * self.max_adv_pct)
            if max_trade:
                delta = int(math.copysign(min(abs(delta), max_trade), delta))
            if delta == 0:
                continue

            # -------- TWAP slicing -----------------------------------
            slice_qty = int(delta / self.twap_slices)
            for i in range(self.twap_slices - 1):
                self._submit_market_order(instrument_id, slice_qty)
            remainder = delta - slice_qty * (self.twap_slices - 1)
            self._submit_market_order(instrument_id, remainder)

    def _submit_market_order(self, instrument_id, qty: int):  # CHANGED: New method
        if qty == 0:
            return
        # CHANGED: Create and submit order using Nautilus API
        order_side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=abs(qty),
        )
        self.submit_order(order)

    # ----- trailing stops --------------------------------------------
    def _run_trailing_stops(self, sym: str, price: float):  # CHANGED: Removed async
        instrument_id = self.instruments[sym]
        position = self.portfolio.position(instrument_id)
        pos = position.quantity if position else 0
        
        if pos == 0:
            self.trailing_stops.pop(sym, None)
            return

        if pos > 0:                       # long
            high = self.trailing_stops.get(sym, price)
            if price > high:
                self.trailing_stops[sym] = price
            elif price <= high * (1 - self.trailing_stop_pct):
                self._submit_market_order(instrument_id, -pos)
                self.trailing_stops.pop(sym, None)
        else:                             # short
            low = self.trailing_stops.get(sym, price)
            if price < low:
                self.trailing_stops[sym] = price
            elif price >= low * (1 + self.trailing_stop_pct):
                self._submit_market_order(instrument_id, -pos)
                self.trailing_stops.pop(sym, None)

    # ----- emergency liquidate ---------------------------------------
    def _liquidate_all(self):  # CHANGED: Removed async
        for sym in self.universe:
            instrument_id = self.instruments[sym]
            position = self.portfolio.position(instrument_id)
            qty = position.quantity if position else 0
            if qty != 0:
                self._submit_market_order(instrument_id, -qty)
        self.trailing_stops.clear()