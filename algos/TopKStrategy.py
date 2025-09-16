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
from typing import Any
from datetime import datetime
from pathlib import Path
from pyexpat import model
from tracemalloc import start
from typing import Dict, List, Optional
from pandas.tseries.frequencies import to_offset
import numpy as np
import pandas as pd
from sklearn import frozen
from sympy import total_degree
import yaml
import pandas_market_calendars as market_calendars
import torch
import logging
logger = logging.getLogger(__name__)

from nautilus_trader.common.component import init_logging, Clock, TimeEvent, Logger
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data import Bar, BarType, InstrumentStatus, InstrumentClose
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.position import Position
from nautilus_trader.model.events import (
    OrderAccepted, OrderCanceled, OrderCancelRejected,
    OrderDenied, OrderEmulated, OrderEvent, OrderExpired,
    OrderFilled, OrderInitialized, OrderModifyRejected,
    OrderPendingCancel, OrderPendingUpdate, OrderRejected,
    OrderReleased, OrderSubmitted, OrderTriggered, OrderUpdated
)

# ----- project imports -----------------------------------------------------
from models.interfaces import MarketModel
from algos.engine.data_loader import CsvBarLoader, FeatureBarData
from algos.engine.OptimizerFactory import create_optimizer
from models.utils import freq2pdoffset, freq2barspec
from algos.order_management import OrderManager


# ========================================================================== #
# Strategy Config
# ========================================================================== #
class TopKStrategyConfig(StrategyConfig, frozen=True):
    config = Dict[str, Any]


# ========================================================================== #
# Strategy
# ========================================================================== #
class TopKStrategy(Strategy):
    """
    Long/short TopK equity strategy, model-agnostic & frequency-agnostic.
    """


    # ------------------------------------------------------------------ #
    def __init__(self, nautilus_cfg: TopKStrategyConfig):  
        super().__init__(config = nautilus_cfg)  

        # these are already flattend
        config = nautilus_cfg.config
        self.strategy_params = config["STRATEGY"]
        self.model_params = config["MODEL"]

        

        self.calendar = market_calendars.get_calendar(config["STRATEGY"]["calendar"])
        
        # Core timing parameters
        self.train_start = pd.Timestamp(self.strategy_params["train_start"])
        self.train_end = pd.Timestamp(self.strategy_params["train_end"])
        self.valid_start = pd.Timestamp(self.strategy_params["valid_start"])
        self.valid_end = pd.Timestamp(self.strategy_params["valid_end"])

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
        loader_cfg = {k: v for k, v in self.strategy_params.items() if k in ["freq", "calendar", "data_dir", "currency"]}
        self.loader = CsvBarLoader(cfg=loader_cfg, venue_name="SIM", columns_to_load=self.model_params["features_to_load"], adjust=self.model_params["adjust"])

        # State tracking
        self.universe: List[str] = []  # Ordered list of instruments
        self.active_mask: Optional[torch.Tensor] = None  # (I,)

        self._last_prediction_time: Optional[pd.Timestamp] = None
        self._last_update_time: Optional[pd.Timestamp] = None
        self._last_retrain_time: Optional[pd.Timestamp] = None
        self._last_bar_time: Optional[pd.Timestamp] = None
        self._bars_since_prediction = 0


        # Initialize tracking variables
        self.max_registered_portfolio_nav = 0.0
        self.realised_returns = []
        self.trailing_stops = {}

        # Extract risk parameters from config
        self.selector_k = self.strategy_params.get("top_k", 30)
        self.max_w_abs = self.strategy_params.get("risk_max_weight_abs", 0.03)
        self.max_w_rel = self.strategy_params.get("risk_max_weight_rel", 0.20)
        self.target_volatility = self.strategy_params.get("risk_target_volatility_annual", 0.05)
        self.trailing_stop_pct = self.strategy_params.get("risk_trailing_stop_pct", 0.05)
        self.drawdown_pct = self.strategy_params.get("risk_drawdown_pct", 0.15)
        adv_lookback = self.strategy_params.get("liquidity", {}).get("adv_lookback", 30)
        max_adv_pct = self.strategy_params.get("liquidity", {}).get("max_adv_pct", 0.05)
        self.twap_slices = self.strategy_params.get("execution", {}).get("twap", {}).get("slices", 4)

        # Portfolio optimizer
        # TODO: add risk_aversion config parameter for MaxQuadraticUtilityOptimizer
        # TODO: make sure to pass proper params to create_optimizer depending on the optimizer all __init__ needed by any optimizer
        optimizer_name = self.strategy_params.get("optimizer_name", "max_sharpe")
        weight_bounds = (-1, 1)   # (+) = long position , (-) = short position
        if self.strategy_params["account_type"] == "CASH":
            weight_bounds = (0, 1)  # long-only positions
        self.optimizer = create_optimizer(name = optimizer_name, adv_lookback = adv_lookback, max_adv_pct = max_adv_pct, weight_bounds = weight_bounds)
        self.rebalance_only = self.strategy_params.get("rebalance_only", False)  # Rebalance mode
        self.top_k = self.strategy_params.get("top_k", 50)  # Portfolio size

        # Risk free rate dataframe to use for Sharpe Ratio
        self.rf_rate = self.loader.risk_free_df

        # Order Manager for any strategy
        self.order_manager = OrderManager(self, self.strategy_params)
        logger.info("OrderManager initialized")


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
        
        # Check if drowdown stop is needed
        self._check_drowdown_stop

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

        # Compute new portfolio target weights for all universe stocks.
        weights = self._compute_target_weights(preds)

        # Portolio Managerment through order manager
        self.order_manager.rebalance_portfolio(weights, self.universe)

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
            active_mask=self.active_mask,
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
        
        # ------------ trailing-stop  ----------------------
        self._check_trailing_stop(sym, bar.close)  

        # ------------ drawdown-stop ------------------------
        self._check_drowdown_stop()
        
        return


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
    # ORDER MANAGEMENT  -> OrderManager from order_management.py
    # ================================================================= #

    def on_order(self, order: OrderEvent) -> None:
        """Handle any order event."""
        pass  # Base handler
    
    def on_order_initialized(self, event: OrderInitialized) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_initialized(event)
    
    def on_order_denied(self, event: OrderDenied) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_denied(event)
    
    def on_order_emulated(self, event: OrderEmulated) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_emulated(event)

    def on_order_released(self, event: OrderReleased) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_released(event)
    
    def on_order_submitted(self, event: OrderSubmitted) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_submitted(event)

    def on_order_accepted(self, event: OrderAccepted) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_accepted(event)

    def on_order_rejected(self, event: OrderRejected) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_rejected(event)
    
    def on_order_canceled(self, event: OrderCanceled) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_canceled(event)
     
    def on_order_expired(self, event: OrderExpired) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.on_order_expired(event)
    
    def on_order_triggered(self, event: OrderTriggered) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.handle_order_event(event)

    def on_order_pending_update(self, event: OrderPendingUpdate) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.handle_order_event(event)
    
    def on_order_pending_cancel(self, event: OrderPendingCancel) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.handle_order_event(event)
    
    def on_order_modify_rejected(self, event: OrderModifyRejected) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.handle_order_event(event)
    
    def on_order_cancel_rejected(self, event: OrderCancelRejected) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.handle_order_event(event)

    def on_order_updated(self, event: OrderUpdated) -> None:
        """Delegate to OrderManager."""
        if self.order_manager:
            self.order_manager.handle_order_event(event)
    
    def on_order_filled(self, event: OrderFilled) -> None:
        """Delegate to OrderManager and update trailing stops."""
        if self.order_manager:
            self.order_manager.on_order_filled(event)
        
        # Update trailing stops for position management
        instrument_id = event.order.instrument_id
        position = self.portfolio.position(instrument_id)
        if position and position.quantity != 0:
            symbol = instrument_id.symbol.value
            if symbol not in self.trailing_stops:
                self.trailing_stops[symbol] = float(event.last_px)
   
    def on_order_event(self, event: OrderEvent) -> None:
        """Catch-all for any unhandled order events."""
        if self.order_manager:
            self.order_manager.handle_order_event(event)

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
        
        df = self.rf_rate
        
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
        now = pd.Timestamp(self.clock.utc_now(), tz='UTC')
        # Current risk-free rate
        current_rf = self._get_risk_free_rate(now)
        # Benchmark volatility
        benchmark_vol = self._get_benchmark_volatility(now)
        # Current portfolio net asset value
        nav = self.portfolio.net_liquidation

        # Max % of average daily traded volume to use for weights
        max_adv_pct = self.strategy_params["liquidity"]["max_adv_pct"]
        #max_weight_changes = np.ones(len(valid_symbols))  # Default: no constraint 

        # Check if we can short (margin account) or only long (cash account)
        can_short = self.strategy_params.get["account_type"] == "MARGIN"


        # ASSET SELECTION: select from whole universe or only from portfolio
        if self.strategy_params["rebalance_only"]:
            # Only rebalance current positions
            selected_symbols = [pos.instrument_id.symbol.value for pos in self.cache.positions_open()]
        else:
            # Select top-k based on predictions
            pred_series = pd.Series(preds)
            
            if can_short:
                l_n = self.selector_k // 2
                longs = pred_series.nlargest(l_n).index.tolist()
                shorts = pred_series.nsmallest(self.selector_k - l_n).index.tolist()
                selected_symbols = longs + shorts
            else:
                selected_symbols = pred_series[pred_series > 0].nlargest(self.selector_k).index.tolist()
        
        
        # MIN/MAX WEIGHT COMPUTATION + USEFULL VARS
        # Get historical returns for covariance
        returns_data = []
        valid_symbols = [] # error handling
        prices = {}      # last close per symbol
        adv_volumes = {} # ADV shares per symbol
        allowed_weight_ranges = [] # min-max portfolio-relative weight for instrument computed from ADV and relative portolio contraints

        # Single loop over selected symbols only
        for symbol in selected_symbols:
            if symbol not in preds:
                continue
                
            instrument_id = InstrumentId(Symbol(symbol), self.venue)
            bar_type = BarType(instrument_id=instrument_id, bar_spec=self.bar_spec)
            bars = self.cache.bars(bar_type)
            
            if len(bars) <= 2:
                continue
                
            # Get returns
            closes = [float(b.close) for b in bars[-self.optimizer_lookback.n:]]
            returns = pd.Series(closes).pct_change().dropna()
            if len(returns) == 0:
                continue
                
            returns_data.append(returns.values)
            valid_symbols.append(symbol)
            prices[symbol] = float(bars[-1].close)
            
            # Compute ADV and constraints
            position = self.portfolio.position(instrument_id)
            current_w = (position.quantity * prices[symbol] / nav) if position else 0.0
            
            volumes = [float(b.volume) for b in bars[-self.adv_lookback:]]
            adv = float(np.mean(volumes)) if volumes else 0.0
            max_w_relative = min((adv * max_adv_pct * prices[symbol]) / nav, self.max_w_abs) if nav > 0 else self.max_w_abs
            
            # Set allowed range
            if can_short:
                w_min = max(current_w - max_w_relative, -self.max_w_abs, self.weight_bounds[0])
                w_max = min(current_w + max_w_relative, self.max_w_abs, self.weight_bounds[1])
            else:
                w_min = 0.0
                w_max = min(current_w + max_w_relative, self.max_w_abs) if current_w > 0 else min(max_w_relative, self.max_w_abs)
            
            allowed_weight_ranges.append([w_min, w_max])
        
        if len(returns_data) <= 1:
            return {}
        
        if len(returns_data) <= 1:
            return {}  # Not enough data
        
        # Build covariance matrix
        returns_df = pd.DataFrame(returns_data).T
        cov = returns_df.cov().values
        
        # Optimize with valid symbols only
        valid_mu = np.array([preds[s] for s in valid_symbols])
         
        

        
        # Call optimizer
        w_opt = self.optimizer.optimize(
            mu=valid_mu,
            benchmark_vol = benchmark_vol,
            cov=cov,
            rf=current_rf,
            max_weight_change=np.array(allowed_weight_ranges)
        )

        # Map back to full universe
        w = pd.Series(0.0, index=self.universe)
        for i, symbol in enumerate(valid_symbols):
            w[symbol] = w_opt[i]
        
        return w.to_dict()


    # ----- trailing and drawdown stops for risk --------------------------------------------
    def _check_trailing_stop(self, sym: str, price: float):  # Removed async
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
                self.order_manager.close_position(instrument_id, reason="trailing_stop")
                self.trailing_stops.pop(sym, None)
        else:                             # short
            low = self.trailing_stops.get(sym, price)
            if price < low:
                self.trailing_stops[sym] = price
            elif price >= low * (1 + self.trailing_stop_pct):
                self.order_manager.close_position(instrument_id, reason="trailing_stop")
                self.trailing_stops.pop(sym, None)

    def _check_drowdown_stop(self):
        nav = self.portfolio.net_liquidation
        self.max_registered_portfolio_nav = max(self.max_registered_portfolio_nav, nav)
        #self.equity_analyzer.on_equity(ts, nav)
        
        if nav < self.max_registered_portfolio_nav * (1 - self.drawdown_pct):
            self.order_manager.liquidate_all(self.universe) 

        return
    
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