"""
Generic long/short strategy for Nautilus Trader.

* Consumes ANY model that follows interfaces.MarketModel
* Data feed provided by CsvBarLoader (Bar, FeatureBar)
* Risk controls: draw-down, trailing stops, ADV cap, fee/slippage models

Implementation reference:
https://nautilustrader.io/docs/latest/concepts/strategies
"""
from __future__ import annotations

import asyncio
import calendar
import importlib
import math
from datetime import datetime
from pathlib import Path
from pyexpat import model
from tracemalloc import start
from typing import Dict, List, Optional
from pandas.tseries.frequencies import to_offset
import numpy as np
import pandas as pd
import yaml
import logging
import pandas_market_calendars as market_calendars
logger = logging.getLogger(__name__)

from nautilus_trader.common.component import init_logging, Clock, TimeEvent, Logger
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data import Bar, BarType, InstrumentStatus, InstrumentClose
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide
import torch
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.position import Position

# ----- project imports -----------------------------------------------------
from models.interfaces import MarketModel
from algos.engine.data_loader import CsvBarLoader, FeatureBarData
from algos.engine.OptimizerFactory import create_optimizer
from models.utils import freq2pdoffset, freq2barspec



# ========================================================================== #
# Strategy
# ========================================================================== #
class LongShortStrategy(Strategy):
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

        # these are already flattend
        self.strategy_params = config["STRATEGY"]
        self.model_params = config["MODEL"]

        

        self.calendar = market_calendars.get_calendar(config["STRATEGY"]["calendar"])
        
        # Core timing parameters
        self.train_start = pd.Timestamp(self.strategy_params["train_start"], tz="UTC")
        self.train_end = pd.Timestamp(self.strategy_params["train_end"], tz="UTC")
        self.valid_start = pd.Timestamp(self.strategy_params["valid_start"], tz="UTC")
        self.valid_end = pd.Timestamp(self.strategy_params["valid_end"], tz="UTC")

        # NOT NEEDED
        #self.backtest_start = pd.Timestamp(self.strategy_params["backtest_start"], tz="UTC")
        #self.backtest_end = pd.Timestamp(self.strategy_params["backtest_end"], tz="UTC")
    
        self.retrain_offset = to_offset(self.strategy_params["retrain_offset"])
        self.train_offset = to_offset(self.strategy_params["train_offset"])
        self.pred_len = int(self.model_params["pred_len"])


        # Model and data parameters
        self.model: Optional[MarketModel] = None
        self.model_name = self.model_params["model_name"]
        self.bar_type = None  # Set in on_start
        self.bar_spec = freq2barspec( self.strategy_params["freq"])
        self.min_bars_required = self.model_params["window_len"] + self.model_params["pred_len"]
        self.optimizer_lookback = freq2pdoffset( self.strategy_params["optimizer_lookback"])
        
        
        # Loader for data access
        # TODO: cols to load should be extended for features added by qlib/libraries. maybe it should include feat_dim
        cols_to_load = []
        if self.model_params["features_to_load"] == "candles":
            cols_to_load = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        else:
            raise Exception(f"FEATURES {self.model_params["features_to_load"]} NOT SUPPORTED")
        
        loader_cfg = {k: v for k, v in self.strategy_params.items() if k in ["freq", "calendar", "data_dir", "currency"]}
        self.loader = CsvBarLoader(cfg=loader_cfg, venue_name="SIM", columns_to_load=cols_to_load)

        # State tracking
        self.universe: List[str] = []  # Ordered list of instruments
        self.active_mask: Optional[torch.Tensor] = None  # (I,)

        self._last_prediction_time: Optional[pd.Timestamp] = None
        self._last_update_time: Optional[pd.Timestamp] = None
        self._last_retrain_time: Optional[pd.Timestamp] = None
        self._last_bar_time: Optional[pd.Timestamp] = None
        self._bars_since_prediction = 0


        # Initialize tracking variables
        self.equity_peak_value = 0.0    # probably should be evaluated locally
        self.realised_returns = []
        self.trailing_stops = {}
        self._prev_nav = None

        # Extract risk parameters from config
        self.selector_k = self.strategy_params.get("selection_top_k", 30)
        self.max_w_abs = self.strategy_params.get("risk_max_weight_abs", 0.03)
        self.max_w_rel = self.strategy_params.get("risk_max_weight_rel", 0.20)
        self.target_vol = self.strategy_params.get("risk_target_vol_annual", 0.15)
        self.trailing_stop_pct = self.strategy_params.get("risk_trailing_stop_pct", 0.05)
        self.drawdown_pct = self.strategy_params.get("risk_drawdown_pct", 0.15)
        self.adv_lookback = self.strategy_params.get("liquidity", {}).get("adv_lookback", 30)
        self.max_adv_pct = self.strategy_params.get("liquidity", {}).get("max_adv_pct", 0.05)
        self.twap_slices = self.strategy_params.get("execution", {}).get("twap", {}).get("slices", 4)

        # Portfolio optimizer
        # TODO: add risk_aversion config parameter for MaxQuadraticUtilityOptimizer
        # TODO: make sure to pass proper params to create_optimizer depending on the optimizer all __init__ needed by any optimizer
        optimizer_name = self.strategy_params.get("optimizer_name", "max_sharpe")
        self.optimizer = create_optimizer(optimizer_name)

        # TODO: consider sharpe ratio on (pred_ret - risk-free)  
        # Risk-free rate series (will be loaded with data)
        self.rf_rate = None


    # ================================================================= #
    # Nautilus event handlers
    # ================================================================= #
    def on_start(self): 
        """Initialize strategy."""

        # Select universe based on stocks active at walk-forward start with enough history
        self._select_universe()

        # Load risk-free rate from loader
        if self.loader.rf_series is not None:
            self.rf_rate = self.loader.rf_series
        else:
            logger.warning("No risk-free rate data available, using 0%")
            self.rf_rate = pd.Series(0.0)

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

        # Set initial update time to avoid immediate firing
        self._last_update_time = pd.Timestamp(self.clock.utc_now(), tz='UTC')
        self._last_retrain_time = pd.Timestamp(self.clock.utc_now(), tz='UTC')

        self.clock.set_timer(
            name="update_timer",  
            interval=freq2pdoffset(self.strategy_params["freq"]),  # Timer interval
            callback=self.on_update,  # Custom callback function invoked on timer
        )

        self.clock.set_timer(
            name="retrain_timer",  
            interval=self.retrain_offset,  # Timer interval
            callback=self.on_retrain,  # Custom callback function invoked on timer
        )

    # Not used ATM
    def on_resume(self) -> None:
        return
    def on_reset(self) -> None:
        return
    def on_dispose(self) -> None:
        return
    def on_degrade(self) -> None:
        return
    def on_fault(self) -> None:
        return
    def on_save(self) -> dict[str, bytes]:  # Returns user-defined dictionary of state to be saved
        return {}
    def on_load(self, state: dict[str, bytes]) -> None:
        return
    
    # Used
    def on_update(self, event: TimeEvent):
        if event.name != "update_timer":
            return

        assert event.ts_event > self._last_update_time 

        now = pd.Timestamp(self.clock.utc_now(), tz='UTC')

        # Skip if no update needed
        if self._last_update_time and now < (self._last_update_time + freq2pdoffset(self.strategy_params["freq"])):
            logger.warning("WIERD STRATEGY UPDATE() CALL")
            return
        
        logger.info(f"Update timer fired at {now}")
        
        # Skip if no update needed
        if self._last_update_time and now < (self._last_update_time + freq2pdoffset(self.strategy_params["freq"])):
            logger.warning("WIERD STRATEGY UPDATE() CALL")

        if not self.model.is_initialized:
            logger.warning("MODEL NOT INITIALIZED WITHIN STRATEGY UPDATE() CALL")
            return
        
        assert self.model.is_initialized()

        data_dict = self._cache_to_dict(window=(self.model.L + 1)) # TODO: make it more generic for all models.

        if not data_dict:
            logger.warning("DATA DICTIONARY EMPTY WITHIN STRATEGY UPDATE() CALL")
            return
        
        # TODO: superflous. consider removing compute active mask
        assert torch.equal(self.active_mask, self._compute_active_mask(data_dict) ) , "Active mask mismatch between strategy and data engine"

        assert self.universe == self.model._universe, "Universe mismatch between strategy and model"

        preds = self.model.predict(data=data_dict, current_time=now, active_mask=self.active_mask) #  preds: Dict[str, float]

        assert preds is not None, "Model predictions are empty"

        # Logic to handle orders and portfolio updates
        weights = self._compute_target_weights(preds)
        self._rebalance_portfolio(weights)

        # update last updated time
        self._last_update_time = now
        return 
        # the logic should be the following:
        # new prediction up until re-optimize portfolio
        # update prediction for the next pred_len = now + holdout - holdout_start
        # checks for stocks events (new or delisted) and retrain_offset and if nothing happens directly calls predict() method of the model (which should happen only if holdout period in bars is passed (otherwise do not do anything), and this variable is in the config.yaml); if retrain delta is passed without stocks event, it calls update() which runs a warm start fit() on the new training window and then the strategy calls predict(); if stock event happens then the strategy calls update() and then predict(). update() should manage the following: if delisting then just use the model active mask to cut off those stocks (so that future preds and training loss are not computed for these. the model is ready to predict after that) but if new stocks are added it checks for universe availability and enough history in the Cache for those new stocks and then calls an initialization. ensure that the most up to date mask is always set by update (or by initialize only at the start) so that the other functions use always the most up to date mask.
        # Update training windows for walk-forward
    def on_retrain(self, event: TimeEvent):
        """Periodic model retraining."""
        if event.name != "retrain_timer":
            return
        
        now = pd.Timestamp(self.clock.utc_now(), tz='UTC')
        
        # Skip if too soon since last retrain
        if self._last_retrain_time and (now - self._last_retrain_time) < self.retrain_offset:
            return
        
        logger.info(f"Starting model retrain at {now}")
        
        # Get updated data window
        data_dict = self._cache_to_dict(window=0)  # Get all available data
        if not data_dict:
            return
        
        # Update active mask
        self.active_mask = self._compute_active_mask(data_dict)
        
        # Retrain model with warm start
        self.model.update(
            data=data_dict,
            current_time=now,
            active_mask=self.active_mask
            warm_start=self.strategy_params.get("warm_start", False),
            warm_training_epochs=self.strategy_params.get("warm_training_epochs", 1)
        )
        
        self._last_retrain_time = now
        logger.info(f"Model retrain completed at {now}")
            
            

    # ================================================================= #
    # DATA handlers
    # ================================================================= #
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

    def on_instrument(self, instrument: Instrument) -> None:
        """Handle new instrument events."""
        #TODO: should account for insertion and delisting at the same time. insertion needs portfolio selection
        # verify if delisted or new:
        # update mask / trigger retrain + reset on_retrain timer
        pass
    def on_instrument_status(self, data: InstrumentStatus) -> None:
        pass
    def on_instrument_close(self, data: InstrumentClose) -> None:
        # update the model mask and ensure the loader still provides same input shape to the model for prediction
        # remove from cache ??
        pass
    def on_historical_data(self, data) -> None:
        """Process historical data for model training."""
        # Historical data is loaded at startup via loader
        pass


    # Unused ATM
    def on_data(self, data) -> None:  # Custom data passed to this handler
        return
    def on_signal(self, signal) -> None:  # Custom signals passed to this handler
        return
    
    # ================================================================= #
    # ORDER MANAGEMENT
    # ================================================================= #

    # TODO: use the order_management.py file

    # ================================================================= #
    # POSITION MANAGEMENT
    # ================================================================= #

    # TODO: use the position_management.py file

    # ================================================================= #
    # ACCOUNT & PORTFOLIO 
    # ================================================================= #

    # TODO: implement 

    # ================================================================= #
    # INTERNAL HELPERS
    # ================================================================= #

    def _rebalance_portfolio(self, target_weights: Dict[str, float]):
        """Rebalance portfolio to target weights."""
        nav = self.portfolio.net_liquidation
        
        for symbol, target_weight in target_weights.items():
            instrument_id = InstrumentId(Symbol(symbol), self.venue)
            
            # Get current position
            position = self.portfolio.position(instrument_id)
            current_qty = position.quantity if position else 0
            
            # Get current price from cache
            bar_type = BarType(instrument_id=instrument_id, bar_spec=self.bar_spec)
            bars = self.cache.bars(bar_type)
            if not bars:
                continue
            
            current_price = float(bars[-1].close)
            
            # Calculate target quantity
            target_value = target_weight * nav
            target_qty = int(target_value / current_price)
            
            # Calculate order quantity
            order_qty = target_qty - current_qty
            
            if order_qty != 0:
                # CHANGED: Create and submit order using Nautilus API
                order_side = OrderSide.BUY if order_qty > 0 else OrderSide.SELL
                order = self.order_factory.market(
                    instrument_id=instrument_id,
                    order_side=order_side,
                    quantity=Quantity.from_int(abs(order_qty)),
                )
                self.submit_order(order)
        return
            

    def _select_universe(self):
        """Select stocks active at walk-forward start with sufficient history."""
        candidate_universe = self.loader.universe
        selected = []
        
        for ticker in candidate_universe:
            if ticker not in self.loader._frames:
                continue
                
            df = self.loader._frames[ticker]
            
            # Check 1: Has enough historical data before walk-forward start
            train_data = df[df.index <= self.valid_end]
            if len(train_data) < self.min_bars_required:
                logger.info(f"Skipping {ticker}: insufficient training data ({len(train_data)} < {self.min_bars_required})")
                continue
            
            # Check 2: Active at walk-forward start
            pred_window = df[(df.index > self.valid_end) ]
            if len(pred_window) < self.pred_len :
                logger.info(f"Skipping {ticker}: not active at walk-forward start")
                continue
                

                
            selected.append(ticker)
            
        logger.info(f"Selected {len(selected)} from {len(candidate_universe)} candidates")
        self.universe = sorted(selected)

    def _initialize_model(self):
        """Build and initialize the model."""
        # Import model class dynamically
        mod = importlib.import_module(f"models.{self.model_name}.{self.model_name}")
        ModelClass = getattr(mod, f"{self.model_name}", None) or getattr(mod, "Model")
        if ModelClass is None:
            raise ImportError(f"Could not find model class in models.{self.model_name}")

        # Check if model hparam was trained already and stored so no init needed
        if (self.model_params["model_dir"] / "init.pt").exists():

            logger.info(f"Model {self.model_name} not found in {self.model_params["model_dir"]} . Loading in process...")
            model = ModelClass(**self.model_params)
            state_dict = torch.load(self.model_params["model_dir"] / "init.pt", map_location=model._device)
            self.model = model.load_state_dict(state_dict)
            logger.info(f"Model {self.model_name} stored in {self.model_params["model_dir"]} loaded successfully")

        
        else:
            logger.info(f"Model {self.model_name} not found in {self.model_params["model_dir"]} . Starting initialization on train data ...")
            
            model = ModelClass(**self.model_params)
            train_data = self._get_data(start = self.train_start , end = self.train_end)
            model.initialize(train_data)
            self.model = model

        return

        
    def _cache_to_dict(self, window: int) -> Dict[str, pd.DataFrame]:
        """
        Convert cache data within the specified window to dictionary format expected by model.
        Efficient implementation using nautilus trader cache's native methods.
        """
        data_dict = {}
        
        for iid in self.cache.instrument_ids():
            # ensure symbol is in universe
            if iid.symbol.value not in self.universe:
                continue
            bar_type = BarType(instrument_id=iid, bar_spec= self.bar_spec)
            
            # Get bars from cache
            if window == 0: # load all cache
                window = self.strategy_params["engine"]["cache"]["bar_capacity"]
            bars = self.cache.bars(bar_type)[:window]  # Get last 'window' bars (nautilus cache notation is opposite to pandas)
            if not bars or len(bars) < self.min_bars_required:
                logger.warning(f"Insufficient bars for {iid.symbol}: {len(bars)} < {self.min_bars_required}")
                continue
            
            # ensure there is at least one new bar after the last predition
            assert bars[0].ts_event > self._last_update_time
            
            # Convert bars to DataFrame
            for b in bars:
                bar_data = {
                    "Date": [b.ts_event],
                    "Open": [b.open ],
                    "High": [b.high],
                    "Low": [b.low ],
                    "Close": [b.close],    #TODO: verify if Close or CLose Adj
                    "Volume": [b.volume],
                }
                df = pd.DataFrame(bar_data)
                df.set_index("Date", inplace=True)
                df.sort_index(inplace=True)  # Ensure chronological order
                data_dict[iid.symbol.value] = df

        return data_dict 
    
    # TODO: superflous. consider removing
    def _compute_active_mask(self, data_dict: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """Compute mask for active instruments."""
        mask = ~torch.ones(len(self.universe), dtype=torch.bool)
        
        for i, symbol in enumerate(self.universe):
            df = data_dict[symbol]
            # Check if data is recent enough
            if len(df) > 0:
                last_date = df.index[-1]
                now = pd.Timestamp(self.clock.utc_now(), tz='UTC')
                if now < last_date:
                    mask[i] = True
        
        return mask
    
    # ----- weight optimiser ------------------------------------------
    def _get_risk_free_rate(self, timestamp: pd.Timestamp) -> float:
        """Get risk-free rate for given timestamp with average fallback logic."""
        if self.loader._benchmark_data is None:
            return 0.0
        
        df = self.loader._risk_free_df
        
        # Try exact date first
        if timestamp.normalize() in df.index:
            rf_now = df.loc[timestamp.normalize(), 'risk_free']
            if not pd.isna(rf_now):
                return float(rf_now)
        
        # Fallback: nearest available data
        nearest_idx = df.index.get_indexer([timestamp], method='nearest')[0]
        if nearest_idx >= 0:
            return float(df.iloc[nearest_idx]['risk_free'])
        
        # Last Fallback: average over window [now - freq, now + freq]
        freq_offset = freq2pdoffset(self.strategy_params["freq"])
        start = timestamp - freq_offset
        end = timestamp + freq_offset
        
        window_data = df.loc[(df.index >= start) & (df.index <= end), 'risk_free']
        
        if len(window_data) > 0:
            return float(window_data.mean() )
        
        return 0.0
    
    def _get_benchmark_volatility(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Calculate benchmark volatility using same lookback as optimizer."""
        if self.loader._benchmark_data is None:
            return None
        
        df = self.loader._benchmark_data
        
        # Calculate start date for lookback window
        lookback_start = timestamp - self.optimizer_lookback

        # Get data in lookback window
        volatility = df.loc[(df.index >= lookback_start) & (df.index <= timestamp), "Benchmark"].std()

        return volatility

    def _compute_target_weights(self, preds: Dict[str, float]) -> Dict[str, float]:
        """Compute target portfolio weights from predictions using optimizer."""
        
        # Convert predictions to array aligned with universe
        mu = np.array([preds.get(s, 0.0) for s in self.universe])
        now = pd.Timestamp(self.clock.utc_now(), tz='UTC')

        # Calculate current risk-free rate
        current_rf = self._get_risk_free_rate(now)

        # Calculate benchmark volatility
        benchmark_vol = self._get_benchmark_volatility(now)
        
        
        # Calculate covariance matrix
        # Get historical returns for covariance
        returns_data = []
        valid_symbols = []
        
        for symbol in self.universe:
            if symbol not in preds:
                continue
                
            instrument_id = InstrumentId(Symbol(symbol), self.venue)
            bar_type = BarType(instrument_id=instrument_id, bar_spec=self.bar_spec)
            bars = self.cache.bars(bar_type)
            
            if len(bars) > 2:
                # Calculate returns from bars
                closes = [float(b.close) for b in bars[-self.optimizer_lookback.n:]]
                returns = pd.Series(closes).pct_change().dropna()
                returns_data.append(returns.values)
                valid_symbols.append(symbol)
        
        if len(returns_data) > 1:
            # Build covariance matrix
            returns_df = pd.DataFrame(returns_data).T
            cov = returns_df.cov().values
            
            # Get current risk-free rate
            current_rf = 0.0
            if self.rf_rate is not None and len(self.rf_rate) > 0:
                # Get most recent risk-free rate (annualized)
                current_rf = self.rf_rate.iloc[-1] if not pd.isna(self.rf_rate.iloc[-1]) else 0.0
            
            # Optimize with valid symbols only
            valid_mu = np.array([preds[s] for s in valid_symbols])
            w_opt = self.optimizer.optimize(valid_mu, cov, current_rf, benchmark_vol)
            
            # Map back to full universe
            w = pd.Series(0.0, index=self.universe)
            for i, symbol in enumerate(valid_symbols):
                w[symbol] = w_opt[i]
        else:
            # TODO: to ensure no trades are execute in case of these failures
            # TODO: it should also be notified
            return {}
            
        
        # Select top-k long and short
        shorts = w.nsmallest(self.selector_k)
        longs = w.nlargest(self.selector_k)
        w = pd.concat([shorts, longs]).reindex(self.universe).fillna(0.0)
        
        # Apply position limits
        w = w.clip(-self.max_w_abs, self.max_w_abs)
        if w.abs().sum() > self.max_w_rel:
            w *= self.max_w_rel / w.abs().sum()
        
        # Volatility targeting
        if len(self.realised_returns) > 20:
            realised_vol = np.std(self.realised_returns) * np.sqrt(252)
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

    def _submit_market_order(self, instrument_id, qty: int):
        if qty == 0:
            return
        # CHANGED: Create and submit order using Nautilus API
        order_side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=Quantity.from_int(abs(qty)),
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
    
    def _get_data(self, start, end) -> Dict[str, pd.DataFrame]:
        # Create data dictionary for selected stocks
        calendar = market_calendars.get_calendar(self.model_params["calendar"])
        days_range = calendar.schedule(start_date=start, end_date=end)
        timestamps = market_calendars.date_range(days_range, frequency=self.model_params["freq"]).normalize()

        # init train+valid data
        data = {}
        for ticker in self.loader.universe:
            if ticker in self.loader._frames:
                df = self.loader._frames[ticker]
                # Get data up to validation end for initial training
                # TODO: this has to change in order to manage data from different timezones/market hours
                df.index = df.index.normalize()
                data[ticker] = df.reindex(timestamps).dropna()
                #logger.info(f"  {ticker}: {len(data[ticker])} bars")
        return data