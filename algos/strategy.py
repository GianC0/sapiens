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
from pandas.tseries.frequencies import to_offset
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
from ..models.utils import freq2pandas, freq2barspec
from .engine.hparam_tuner import split_hparam_cfg, OptunaHparamsTuner

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
        
        # Core timing parameters
        self.backtest_start = pd.Timestamp(self.cfg["backtest_start"], tz="UTC")
        self.train_end = pd.Timestamp(self.cfg["train_end"], tz="UTC")
        self.valid_end = pd.Timestamp(self.cfg["valid_end"], tz="UTC")
        self.backtest_end = pd.Timestamp(self.cfg["backtest_end"], tz="UTC")
        self.walkfwd_start = pd.Timestamp(self.cfg["walkfwd_start"], tz="UTC")
        self.pred_offset = to_offset(cfg["pred_offset"])
        self.pred_len = int(cfg["pred_len"])
        self.retrain_delta = pd.DateOffset(days=cfg["training"]["retrain_delta"])

        # Model and data parameters
        self.model_name = self.cfg["model_name"]
        self.bar_type = None  # Set in on_start
        self.bar_spec = freq2barspec(self.cfg["freq"])
        self.min_bars_required = self.cfg["window_len"] + self.cfg["pred_len"] + 100  # Safety margin
        
        # State tracking
        self.universe: List[str] = []  # Active universe
        self.model: Optional[MarketModel] = None
        self._last_prediction_time: Optional[pd.Timestamp] = None
        self._position_qty: Dict[str, float] = {}  # Track position quantities
        
        # Loader for data access
        self.loader = CsvBarLoader(cfg=self.cfg)


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

        # Select universe based on stocks active at walk-forward start with enough history
        self._select_universe()

        # Subscribe to bars for selected universe
        for instrument in self.loader.instruments.values():
            if instrument.id.symbol.value in self.universe:
                self.bar_type = BarType(
                    instrument_id=instrument.id,
                    bar_spec=self.bar_spec
                )
                self.subscribe_bars(self.bar_type)

        # Build and initialize model
        self._initialize_model()

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
            interval=self.retrain_delta,  # Timer interval
            callback=self.on_retrain,  # Custom callback function invoked on timer
        )
 

    def on_update(self, event: TimeEvent):
        #TODO: does not account for insertion and delisting at the same time
        if event.name == "update_timer":

            # to pass self.clock.utc_now()
            # the logic should be the following:
            # update prediction for the next pred_len = now + holdout - holdout_start
            # checks for stocks events (new or delisted) and retrain_delta and if nothing happens directly calls predict() method of the model (which should happen only if holdout period in bars is passed (otherwise do not do anything), and this variable is in the config.yaml); if retrain delta is passed without stocks event, it calls update() which runs a warm start fit() on the new training window and then the strategy calls predict(); if stock event happens then the strategy calls update() and then predict(). update() should manage the following: if delisting then just use the model active mask to cut off those stocks (so that future preds and training loss are not computed for these. the model is ready to predict after that) but if new stocks are added it checks for universe availability and enough history in the Cache for those new stocks and then calls an initialization. ensure that the most up to date mask is always set by update (or by initialize only at the start) so that the other functions use always the most up to date mask.
            # Update training windows for walk-forward
            # self._last_fit_time = None                                                                 # last time fit() was called
            if self._last_fit_time:
                time_shift = current_time - self._last_fit_time
                self.train_end = self.train_end + time_shift
                self.valid_end = self.valid_end + time_shift
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
    # INTERNAL HELPERS
    # ================================================================= #

    def _select_universe(self):
        """Select stocks active at walk-forward start with sufficient history."""
        candidate_universe = self.loader.universe
        selected = []
        
        for ticker in candidate_universe:
            if ticker not in self.loader._frames:
                continue
                
            df = self.loader._frames[ticker]
            
            # Check 1: Has enough historical data before walk-forward start
            train_data = df[df.index < self.walkfwd_start]
            if len(train_data) < self.min_bars_required:
                self.log.info(f"Skipping {ticker}: insufficient training data ({len(train_data)} < {self.min_bars_required})")
                continue
            
            # Check 2: Active at walk-forward start (has data within 5 days)
            pred_window = df[(df.index >= self.walkfwd_start) ]
            if len(pred_window) < self.pred_len :
                self.log.info(f"Skipping {ticker}: not active at walk-forward start")
                continue
                

                
            selected.append(ticker)
            
        self.log.info(f"Selected {len(selected)} from {len(candidate_universe)} candidates")
        self.universe = sorted(selected)

    def _initialize_model(self):
        """Build and initialize the model."""
        # Import model class dynamically
        mod = importlib.import_module(f"models.{self.model_name}.{self.model_name}")
        ModelClass = getattr(mod, f"{self.model_name}", None) or getattr(mod, "Model")
        if ModelClass is None:
            raise ImportError(f"Could not find model class in models.{self.model_name}")

        model_params = {
            "freq"=self.cfg["freq"],
            "feature_dim"=self.cfg["feature_dim"],
            "window_len"=self.cfg["window_len"],
            "pred_len"=self.cfg["pred_len"],
            "train_end"=self.train_end,
            "valid_end"=self.valid_end,
            "batch_size"=self.cfg["training"]["batch_size"],
            "n_epochs"=self.cfg["training"]["n_epochs"],
            "patience"=self.cfg["training"]["patience"],
            "pretrain_epochs"=self.cfg["training"]["pretrain_epochs"],
            "training_mode"=self.cfg["training"]["training_mode"],
            "close_idx"=self.cfg["training"]["target_idx"],
            "warm_start"=self.cfg["training"]["warm_start"],
            "warm_training_epochs"=self.cfg["training"]["warm_training_epochs"],
            "save_backups"= self.cfg["training"]["save_backups"],
            "data_dir"=self.cfg["data_dir"],
            "logger"=self.log,  # Pass logger to model
        }

        
        # HPs
        defaults, search_space = split_hparam_cfg(self.cfg["hparams"])
        best_hparams = {}

        # Determine if we should run hyperparameter tuning 
        if self.cfg["training"]["tune_hparams"]:

            # Setup and Prepare initial training data (only universe stocks)
            self.log.info("[HPO] Optuna search ENABLED")
            search_space = search_space
            n_trials = self.cfg["training"]["n_trials"]
            train_data = {
                ticker: self.loader._frames[ticker][(self.loader._frames[ticker].index <= self.valid_end) & (self.loader._frames[ticker].index >= self.backtest_start) ]
                for ticker in self.universe
            }

            # model_dir is sepcified by the tuner
            tuner = OptunaHparamsTuner( 
                model_name  = self.cfg["model_name"],
                ModelClass   = ModelClass,
                start = self.backtest_start,
                end = self.valid_end,
                logs_dir    = Path(self.cfg.get("logs_dir", "logs")),
                train_dict  = train_data,
                model_params= model_params,
                defaults    = defaults,           # default values of hparams
                search_space= search_space,       # search sapce of hparams
                n_trials    = n_trials,
                logger = self.log
            )
            
            best = tuner.optimize()

            # Update defaults with best hyperparameters
            best_hparams = {**defaults, **best["hparams"]}
            self.log.info(f"Best hyperparameters found: {best_hparams}")
        else:
            self.log.info("[HPO] Hyperparameter tuning disabled, using defaults")
            best_hparams = defaults

        # Initialize model with best hyperparameters. It will have new model directory and trained on train + valid set
        final_model_params = model_params.copy()
        final_model_params["train_end"] = self.valid_end
        final_model_params["valid_end"] = self.valid_end

        # Prepare extended training data
        final_train_data = {
            ticker: self.loader._frames[ticker][
                (self.loader._frames[ticker].index >= self.backtest_start) & 
                (self.loader._frames[ticker].index <= self.valid_end)
            ] for ticker in self.universe
        }

        # model_dir is sepcified by the tuner
        final_tuner = OptunaHparamsTuner( 
            model_name  = self.cfg["model_name"],
            ModelClass   = ModelClass,
            start = self.backtest_start,
            end = self.valid_end,
            logs_dir    = Path(self.cfg.get("logs_dir", "logs")),
            train_dict  = final_train_data,
            model_params= final_model_params,
            defaults    = best_hparams,       # Use best hyperparameters
            search_space= None,               # empty search space: use defaults and 1 trial only
            n_trials    = 1,
            logger = self.log
        )
        
        _ , model_dir = tuner.optimize()
        

        # reload the model to use for walk-forward steps
        self.model = ModelClass(**final_model_params, **best_hparams)

        if (model_dir / "init.pt").exists():
                state_dict = torch.load(saved_state_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
        else:
            raise ImportError(f"Could not find init.pt in {model_dir}")
        


    # ----- model factory ----------------------------------------------
    def _build_model(self) -> MarketModel:
        name = str(self.cfg["model_name"]).lower()
        mod = importlib.import_module(f"models.{name}.{name}")
        ModelClass = getattr(mod, "Model", None) or getattr(mod, f"{name.upper()}Model")
        if ModelClass is None:
            raise ImportError(f"models.{name}.{name} must export a Model class")
        
        
        first_df = next(iter(self.loader._frames.values()))
        
        model: MarketModel = ModelClass(
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