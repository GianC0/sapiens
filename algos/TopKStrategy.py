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
from dataclasses import dataclass, field
import logging
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.currencies import USD,EUR
from nautilus_trader.common.component import init_logging, Clock, TimeEvent, Logger
from nautilus_trader.core.data import Data
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data import Bar, BarType, InstrumentStatus, InstrumentClose
from nautilus_trader.data.messages import RequestBars
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.position import Position
from nautilus_trader.model.events import (
    OrderAccepted, OrderCanceled, OrderCancelRejected,
    OrderDenied, OrderEmulated, OrderEvent, OrderExpired,
    OrderFilled, OrderInitialized, OrderModifyRejected,
    OrderPendingCancel, OrderPendingUpdate, OrderRejected,
    OrderReleased, OrderSubmitted, OrderTriggered, OrderUpdated
)

logger = logging.getLogger(__name__)
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
    config : Dict[str, Any] = field(default_factory=dict) 


# ========================================================================== #
# Strategy
# ========================================================================== #
class TopKStrategy(Strategy):
    """
    Long/short TopK equity strategy, model-agnostic & frequency-agnostic.
    """


    # ------------------------------------------------------------------ #
    def __init__(self, config: TopKStrategyConfig):  
        super().__init__()  

        # quick & dirty: allow passing in either a plain dict or a StrategyConfig
            #if isinstance(nautilus_cfg, dict):
                # wrap dict into the expected StrategyConfig dataclass
            #    nautilus_cfg = TopKStrategyConfig(config=nautilus_cfg)

            # use the inner dict consistently
        if hasattr(config, 'config') and config.config is not None:
            # It's a TopKStrategyConfig with a config attribute
            cfg = config.config
        elif isinstance(config, dict):
            # It's already a dictionary
            cfg = config
        elif hasattr(config, '__dataclass_fields__'):
            # It's a dataclass but config field is not initialized
            # This happens when Nautilus creates the config incorrectly
            # Fall back to empty dict - will be populated by Nautilus
            cfg = {}
        else:
            # Field descriptor or other unexpected type
            cfg = {}

        self.strategy_params = cfg["STRATEGY"]
        self.model_params = cfg["MODEL"]
        self.data_params = cfg["DATA"]

        # safe handling of variable
        if self.strategy_params["currency"]  == "USD":
            self.strategy_params["currency"]  = USD 
        elif self.strategy_params["currency"]  == "EUR":
            self.strategy_params["currency"]  = EUR
        else: # currency is already in nautilus format
            pass 
        self.model_params["model_dir"] = Path(self.model_params["model_dir"])

        self.calendar = market_calendars.get_calendar(cfg["STRATEGY"]["calendar"])
        
        # Core timing parameters
        self.train_start = pd.Timestamp(self.strategy_params["train_start"])
        self.train_end = pd.Timestamp(self.strategy_params["train_end"])
        self.valid_start = pd.Timestamp(self.strategy_params["valid_start"])
        self.valid_end = pd.Timestamp(self.strategy_params["valid_end"])

        # data load start should be conservative: 
        # if bars within retrain_date - data_load_start < min_bars_required, then on_historical fails to load necessary data.  

        #self.data_load_start = pd.Timestamp(self.strategy_params["data_load_start"])     # NON-CONSERVATIVE
        self.data_load_start = self.train_start                                           # CONSERVATIVE

        # NOT NEEDED
        #self.backtest_start = pd.Timestamp(self.strategy_params["backtest_start"], tz="UTC")
        #self.backtest_end = pd.Timestamp(self.strategy_params["backtest_end"], tz="UTC")
    
        self.retrain_offset = to_offset(self.strategy_params["retrain_offset"])
        self.train_offset = to_offset(self.strategy_params["train_offset"])
        self.pred_len = int(self.model_params["pred_len"])


        # Model and data parameters
        self.model: Optional[MarketModel] = None
        self.model_name = self.model_params["model_name"]
        self.bar_spec = freq2barspec( self.strategy_params["freq"])
        self.min_bars_required = self.model_params["window_len"]
        self.optimizer_lookback = freq2pdoffset( self.strategy_params["optimizer_lookback"])
        
        
        # Loader for data access
        venue_name = self.strategy_params["venue_name"]
        self.venue = Venue(venue_name)
        self.loader = CsvBarLoader(cfg=self.strategy_params, venue_name=self.venue.value, columns_to_load=self.data_params["features_to_load"], adjust=self.data_params["adjust"])
        #self.catalog = ParquetDataCatalog( path = self.strategy_params["catalog_path"], fs_protocol="file")
        self.universe: List[str] = []  # Ordered list of instruments
        self.active_mask: Optional[torch.Tensor] = None  # (I,)

        self._last_prediction_time: Optional[pd.Timestamp] = None
        self._last_update_time: Optional[pd.Timestamp] = None
        self._last_retrain_time: Optional[pd.Timestamp] = None
        self._last_bar_time: Optional[pd.Timestamp] = None
        self._bars_since_prediction = 0


        # Initialize tracking variables
        self.max_registered_portfolio_nav = 0.0
        self.nav = 0.0
        self.realised_returns = []
        self.trailing_stops = {}

        # Extract risk params
        self.max_w_abs = self.strategy_params["risk"]["max_weight_abs"]
        self.drawdown_max = self.strategy_params["risk"]["drawdown_max"]
        self.trailing_stop_max = self.strategy_params["risk"]["trailing_stop_max"]
        self.target_volatility = self.strategy_params["risk"]["target_volatility"]
        
        # Extract execution parameters from config
        self.selector_k = self.strategy_params["top_k"]

        self.adv_lookback = self.strategy_params["liquidity"]["adv_lookback"]
        self.max_adv_pct = self.strategy_params["liquidity"]["max_adv_pct"]
        self.twap_slices = self.strategy_params["execution"]["twap"]["slices"]
        self.twap_interval_secs = self.strategy_params["execution"]["twap"]["interval_secs"]

        self.commission_rate = self.strategy_params['costs']['fee_bps'] / 100
        self.can_short = self.strategy_params["oms_type"] == "HEDGING"

        # Portfolio optimizer
        # TODO: add risk_aversion config parameter for MaxQuadraticUtilityOptimizer
        # TODO: make sure to pass proper params to create_optimizer depending on the optimizer all __init__ needed by any optimizer
        optimizer_name = self.strategy_params.get("optimizer_name", "max_sharpe")
        weight_bounds = (-1, 1)   # (-) = short position ,  (+) = long position 
        if not self.can_short:
            weight_bounds = (0, 1)  # long-only positions
        self.weight_bounds = weight_bounds
        self.optimizer = create_optimizer(name = optimizer_name, adv_lookback = self.adv_lookback, max_adv_pct = self.max_adv_pct, weight_bounds = weight_bounds)
        self.rebalance_only = self.strategy_params.get("rebalance_only", False)  # Rebalance mode
        self.top_k = self.strategy_params.get("top_k", 50)  # Portfolio size
        

        # Risk free rate dataframe to use for Sharpe Ratio
        self.risk_free_df = self.loader.risk_free_df

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
        for instrument in self.cache.instruments():
            if instrument.id.symbol.value in self.universe:
                bar_type = BarType(
                    instrument_id=instrument.id,
                    bar_spec=self.bar_spec
                )

                # request historical bars
                # in live should be done through self.request_bars
                self.on_historical_data(bar_type = bar_type, start = self.data_load_start)
                
                # subscribe bars for walk forward
                self.subscribe_bars(bar_type)





        # Build and initialize model
        self.model = self._initialize_model()
        self.model._universe = self.universe

        # Set initial update time to avoid immediate firing
        self.active_mask = torch.ones(len(self.cache.instruments()), dtype=torch.bool)
        
        # initialize the chache with the latest historical window_length data
        #self._initialize_cache_with_historical_data()

        # Get current time and check if we have enough historical data
        #now = pd.Timestamp(self.clock.utc_now())
        
        
        self._last_update_time = pd.Timestamp(self.clock.utc_now())
        self._last_retrain_time = pd.Timestamp(self.clock.utc_now())
        
        # Set the regular trading timers
        self.clock.set_timer(
            name="update_timer",
            interval=pd.Timedelta(freq2pdoffset(self.strategy_params["freq"])),
            callback=self.on_update,
        )
        
        self.clock.set_timer(
            name="retrain_timer",
            interval=pd.Timedelta(self.retrain_offset),
            callback=self.on_retrain,
        )    



    # Not used ATM
    def on_resume(self) -> None:
        return
    def on_reset(self) -> None:
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
    #def on_dispose(self) -> None:
    #    self.order_manager.liquidate_all(self.universe)
    #    return

    def on_dispose(self) -> None:
        """Ensure all positions are closed at end of backtest for accurate P&L."""
        logger.info("Strategy disposal: liquidating all positions for final P&L calculation")
        
        # Force close all open positions at current market prices
        for position in self.cache.positions_open(venue=self.venue):
            instrument_id = position.instrument_id
            symbol = instrument_id.symbol.value
            
            logger.info(f"Final liquidation: closing {symbol} position of {position.signed_qty}")
            self.order_manager.close_position(position)
        
        # Cancel any pending orders
        self.order_manager.cancel_all_orders()
        
        # Log final NAV for verification
        final_nav = self._calculate_portfolio_nav()
        logger.info(f"Final NAV at disposal: {final_nav:.2f}")
        
        return
    
    def on_update(self, event: TimeEvent):
        if event.name != "update_timer":
            return

        event_time = pd.to_datetime(event.ts_event, unit="ns", utc=True)
        if event_time <= self._last_update_time:
            logger.debug(f"Event time {event_time} not after last update {self._last_update_time}")
            return

        now = pd.Timestamp(self.clock.utc_now()) #.tz_convert(self.calendar.tz)
        freq = self.strategy_params["freq"]
        #d_r = self.calendar.schedule(start_date=str(now - freq2pdoffset(freq)), end_date=str(now + freq2pdoffset(freq)))
        
        # check if "now" falls outside market trading hours
        schedule = self.calendar.schedule(start_date=str(now), end_date=str(now))
        if schedule.empty:
            logger.info(f"Non-trading day at {now}")
            return

        # Skip if no update needed
        if self._last_update_time and now < (self._last_update_time + freq2pdoffset(freq)):
            logger.warning("Update called too soon, skipping")
            return
        
        logger.info(f"Update timer fired at {now}")
        
        # Check if drowdown stop is needed
        self._check_drowdown_stop()

        if not self.model.is_initialized:
            logger.warning("MODEL NOT INITIALIZED WITHIN STRATEGY UPDATE() CALL")
            return
        
        assert self.model.is_initialized

        data_dict = self._cache_to_dict(window=(self.min_bars_required))

        if not data_dict:
            logger.warning("DATA DICTIONARY EMPTY WITHIN STRATEGY UPDATE() CALL")
            return
        
        # TODO: superflous. consider removing compute active mask
        assert torch.equal(self.active_mask, self._compute_active_mask(data_dict) ) , "Active mask mismatch between strategy and data engine"

        assert self.universe == self.model._universe, "Universe mismatch between strategy and model"

        # ensure the model has enough data for prediction
        #start_date = now - freq2pdoffset(self.strategy_params["freq"]) * ( self.min_bars_required) 
        #days_range = self.calendar.schedule(start_date=start_date, end_date=now)
        #timestamps = market_calendars.date_range(days_range, frequency=self.strategy_params["freq"]).normalize()
        #if len(timestamps) < lookback_periods:
        #    return

        preds = self.model.predict(data=data_dict, indexes = self.min_bars_required, active_mask=self.active_mask) #  preds: Dict[str, float]

        assert preds is not None, "Model predictions are empty"
        assert len(preds) > 0

        # Compute new portfolio target weights for predicted instruments.
        weights = self._compute_target_weights(preds)

        # Check if we have valid weights
        if not weights or all(abs(w) < 1e-6 for w in weights.values()):
            logger.warning("No valid weights computed, skipping rebalance")
            self._last_update_time = now
            return

        # Portolio Managerment through order manager
        self.order_manager.rebalance_portfolio(weights, self.universe)

        # update last updated time
        self._last_update_time = now
        return 

    def on_retrain(self, event: TimeEvent):
        """Periodic model retraining."""
        if event.name != "retrain_timer":
            return
        
        now = pd.Timestamp(self.clock.utc_now(),)
        
        # Skip if too soon since last retrain
        if self._last_retrain_time and (now - self._last_retrain_time) < self.retrain_offset:
            return
        
        logger.info(f"Starting model retrain at {now}")

        # comput how much data to load
        days_range = self.calendar.schedule(start_date= self._last_retrain_time, end_date=now)
        timestamps = market_calendars.date_range(days_range, frequency=self.strategy_params["freq"])
        
        
        # ═══════════════════════════════════════════════════════════════════
        # ADAPTIVE WINDOW: Ensure minimum training size
        # ═══════════════════════════════════════════════════════════════════
        # ensures case at t=0 where len(timestamps) for new bars < min_bars_required 
        rolling_window_size = max(len(timestamps), self.min_bars_required + self.pred_len)
        
        # Log when falling back to minimum (includes overlap with previous training)
        if rolling_window_size > self.min_bars_required:
            logger.info(f"[on_retrain] Rolling window ({rolling_window_size} bars) < minimum required ({self.min_bars_required} bars). Overlap: {rolling_window_size - self.min_bars_required} bars from previous training.")
        else:
            logger.info(
                f"[on_retrain] Using rolling window of {rolling_window_size} bars (sufficient for training).")
        
        # Get updated data window
        data_dict = self._cache_to_dict(window = (rolling_window_size))  # Get all latest
        if not data_dict:
            return
        # all assets have same number of bars is ensured by get_data()
        total_bars = len(next(iter(data_dict.values())))
        
        # Update active mask
        self.active_mask = self._compute_active_mask(data_dict)
        
        # Retrain model with warm start
        self.model.update(
            data=data_dict,
            current_time=now,
            retrain_start_date = self._last_retrain_time,
            active_mask=self.active_mask,
            total_bars = total_bars,
            warm_start=self.strategy_params["warm_start"],
            warm_training_epochs=self.strategy_params["warm_training_epochs"],
        )
        
        self._last_retrain_time = now
        logger.info(f"Model retrain completed at {now}")
            
            

    # ================================================================= #
    # DATA handlers
    # ================================================================= #
    def on_bar(self, bar: Bar):  
        # assuming on_timer happens before then on_bars are called first,
        #ts = bar.ts_event
        instrument = self.cache.instrument(bar.bar_type.instrument_id)
        # ------------ trailing-stop  ----------------------
        self._check_trailing_stop(instrument, bar.close)  

        # ------------ drawdown-stop ------------------------
        #dwd_s = self._check_drowdown_stop()
        #if dwd_s:
        #    self.on_dispose()
        #    return
        
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
    def on_historical_data(self, bar_type: BarType, start: pd.Timestamp ) -> None: 
        """Load historical data for model prediction at t=0."""
        # Historical data is loaded at startup via loader
        bars = []

        for bar_or_feature in self.loader.bar_iterator(start_time=start, end_time=self.train_end, symbols=[bar_type.instrument_id.symbol.value]):
            if isinstance(bar_or_feature, Bar):
                bar_ts = pd.Timestamp(bar_or_feature.ts_event, unit='ns', tz='UTC')
                assert start <= bar_ts <= self.train_end and bar_or_feature.bar_type == bar_type
                bars.append(bar_or_feature)
        
        self.cache.add_bars(bars)
        logger.info(f"Added {len(bars)}  bars for {bar_type.instrument_id.value}")
        

        


    # Unused ATM
    def on_data(self, data: Data) -> None:  # Custom data passed to this handler
        assert False
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
        instrument_id = event.instrument_id
        
        position = self.cache.positions(instrument_id=instrument_id)
        if position and position[0].quantity != 0:
            symbol = instrument_id.symbol.value
            if symbol not in self.trailing_stops:
                self.trailing_stops[symbol] = float(event.last_px)
   
#    def on_order_event(self, event: OrderEvent) -> None:
#        """Catch-all for any unhandled order events."""
#        if self.order_manager:
#            self.order_manager.handle_order_event(event)

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
            # Check if we have data extending pred_len periods beyond valid_end
            pred_window = df[(df.index > self.valid_end)]
            
            if pred_window.empty:
                logger.info(f"Skipping {ticker}: not active at walk-forward start")
                continue

            selected.append(ticker)
            
        logger.info(f"Selected {len(selected)} from {len(candidate_universe)} candidates (including risk-free) ")
        self.universe = sorted(selected)

    def _initialize_model(self) -> MarketModel:
        """Build and initialize the model."""
        # Import model class dynamically
        mod = importlib.import_module(f"models.{self.model_name}.{self.model_name}")
        ModelClass = getattr(mod, f"{self.model_name}", None) or getattr(mod, "Model")
        if ModelClass is None:
            raise ImportError(f"Could not find model class in models.{self.model_name}")

        # Check if model hparam was trained already and stored so no init needed
        if (self.model_params["model_dir"] / "init.pt").exists():

            logger.info(f"Model {self.model_name} found in {self.model_params["model_dir"]} . Loading in process...")
            model = ModelClass(**self.model_params)
            state_dict = torch.load(self.model_params["model_dir"] / "init.pt", map_location=model._device, weights_only=False)
            model.load_state_dict(state_dict)
            logger.info(f"Model {self.model_name} stored in {self.model_params["model_dir"]} loaded successfully")

        
        else:
            logger.error(f"Model {self.model_name} not found in {self.model_params["model_dir"]}.")

            
            
            model = ModelClass(**self.model_params)
            train_data = self.loader.get_data(calendar = self.model_params["calendar"] , frequency = self.model_params["freq"], start = self.train_start , end = self.train_end)
            # all assets have same number of bars is ensured by get_data()
            total_bars = len(next(iter(train_data.values())))
            model.initialize(data = train_data, total_bars = total_bars)

            raise Exception("Model Not Initialized")
            

        return model

        
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

            count = self.cache.bar_count(bar_type)
            if count < self.min_bars_required:
                logger.warning(f"Insufficient bars for {iid.symbol}: {count} < {self.min_bars_required}")
                continue
            
            # Get bars from cache
            if window == 0: # load all cache
                window = self.strategy_params["engine"]["cache"]["bar_capacity"]
            bars = self.cache.bars(bar_type)[:window]  # Get last 'window' bars (nautilus cache notation is opposite to pandas)

            # ensure there is at least one new bar after the last predition
            bar_time = pd.to_datetime(bars[0].ts_event, unit="ns", utc=True)
            if bar_time <= self._last_update_time:
                logger.warning(f"Bar time {bar_time} not after last update {self._last_update_time}")

            # Collect all bar data into lists
            bar_data = {
                "Date": [],
                "Open": [],
                "High": [],
                "Low": [],
                "Close": [],
                "Volume": [],
            }
            # Convert bars to DataFrame
            for b in reversed(bars):
                
                bar_data["Date"].append(pd.to_datetime(b.ts_event, unit="ns", utc=True)),
                bar_data["Open"].append(float(b.open))
                bar_data["High"].append(float(b.high))
                bar_data["Low"].append(float(b.low))
                bar_data["Close"].append(float(b.close))
                bar_data["Volume"].append(float(b.volume))

            # Create DataFrame from collected data
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
            #df = data_dict[symbol]
            # Check if data is recent enough
            last_time = self.loader._frames[symbol].index[-1]
            now = pd.Timestamp(self.clock.utc_now(),)
            mask[i] = last_time >= now
    
        return mask
    
    def _initialize_cache_with_historical_data(self):
        """Load historical data into strategy's internal buffer for immediate use."""
        now = pd.Timestamp(self.clock.utc_now())
        lookback_periods = self.model_params["window_len"] + 1
        
        # Store historical data in a buffer
        self._historical_buffer = {}
        
        for symbol in self.universe:
            if symbol in self.loader._frames:
                df = self.loader._frames[symbol]
                # Get data up to current time
                historical_data = df[df.index < now]
                if len(historical_data) >= lookback_periods:
                    # Keep last lookback_periods of historical data
                    self._historical_buffer[symbol] = historical_data.iloc[-lookback_periods:]

    # ----- weight optimiser ------------------------------------------
    def _get_risk_free_rate_return(self, timestamp: pd.Timestamp) -> float:
        """Get risk-free rate for given timestamp with average fallback logic."""
        if self.loader._benchmark_data is None:
            return 0.0
        
        df = self.risk_free_df
        
        # Try exact date first
        if timestamp in df.index:
            rf_now = df.loc[timestamp, 'risk_free']
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
        lookback_start = timestamp - self.optimizer_lookback
        
        # Use iloc with index searching for robust timestamp matching
        start_idx = df.index.searchsorted(lookback_start, side='left')
        end_idx = df.index.searchsorted(timestamp, side='right')
        
        volatility = df.iloc[start_idx:end_idx]["Benchmark"].std()
        return volatility

    def _compute_target_weights(self, preds: Dict[str, float]) -> Dict[str, float]:
        """Compute target portfolio weights using M2 optimizer and optional sign enforcement.

        Behavior controlled by self.enforce_pred_sign (True -> force sign to match preds,
        False -> keep optimizer sign).
        """
        now = pd.Timestamp(self.clock.utc_now())
        current_rf = self._get_risk_free_rate_return(now)

        # build candidate list excluding risk free ticker
        risk_free_ticker = self.strategy_params["risk_free_ticker"]

        # gather return series and allowed ranges for valid symbols
        returns_data = []
        valid_symbols = []
        prices = {}
        allowed_weight_ranges = []
        current_weights = [] # Calculate current weights for fallback
        total_current_exposure = 0.0

        # Current portfolio net asset value
        self.nav = self._calculate_portfolio_nav()

        for i, symbol in enumerate(self.universe):
            
            instrument_id = InstrumentId(Symbol(symbol), self.venue)
            bar_type = BarType(instrument_id=instrument_id, bar_spec=self.bar_spec)
            bars = self.cache.bars(bar_type)
            if not bars or len(bars) < 2:
                continue

            # Get current price
            current_price = float(bars[-1].close)
            prices[symbol] = current_price

            # Calculate current position weight
            net_position = 0.0
            for position in self.cache.positions_open():
                if position.instrument_id == instrument_id:
                    net_position += float(position.signed_qty)
            current_w = (net_position * current_price / self.nav) if (net_position and self.nav > 0) else 0.0
            current_weights.append(current_w)
            total_current_exposure += abs(current_w) 

            # returns per single period (same freq as preds' base period)
            lookback_n = max(2, int(self.optimizer_lookback.n) )
            closes = [float(b.close) for b in bars[-lookback_n:]]
            returns = pd.Series(closes).pct_change().dropna()
            if len(returns) == 0:
                continue

            returns_data.append(returns.values)
            valid_symbols.append(symbol)
            prices[symbol] = float(bars[-1].close)

            # compute allowed weight ranges
            # Risk-free rate is the only one who can take up to 100% of portfolio
            if symbol == risk_free_ticker:
                # Risk-free allocation should account for commissions
                #max_weight = 1 -  ( 2 * self.commission_rate )   # Reserve for round-trip
                allowed_weight_ranges.append([0, 1.0])
                continue

            # ADV constraints
            volumes = [float(b.volume) for b in bars[-self.adv_lookback:]] if len(bars) >= self.adv_lookback else [float(b.volume) for b in bars]
            adv = float(np.mean(volumes)) if volumes else 0.0
            
            # Adjust max weight for commission reserve
            commission_adjusted_nav = self.nav * (1 - self.commission_rate * 2)  # Reserve for round-trip
            max_w_relative = min((adv * self.max_adv_pct * current_price) / commission_adjusted_nav, self.max_w_abs) if commission_adjusted_nav > 0 else self.max_w_abs

            # compute min or min/max weight for instrument
            if self.can_short:
                w_min = max(current_w - max_w_relative, -self.max_w_abs, self.weight_bounds[0])
                w_max = min(current_w + max_w_relative, self.max_w_abs, self.weight_bounds[1])
            else:
                w_min = 0.0
                w_max = min(current_w + max_w_relative, self.max_w_abs) if current_w > 0 else min(max_w_relative, self.max_w_abs)

            allowed_weight_ranges.append([w_min, w_max])

        if len(returns_data) <= 1:
            return {}

        # Build covariance per single period and scale to pred_horizon
        returns_df = pd.DataFrame(returns_data).T  # rows = times, cols = assets
        cov_per_period = returns_df.cov().values
        cov_horizon = cov_per_period * float(self.pred_len)  # scale to pred_len horizon

        # prepare expected returns
        valid_expected_returns = np.array([preds.get(s, 0.0) for s in valid_symbols])

        # If all expected returns <= rf_horizon then allocate to risk-free ticker (if available)
        if np.all(valid_expected_returns <= current_rf):
            w_series = pd.Series(0.0, index=self.universe)
            if risk_free_ticker in self.universe:
                w_series[risk_free_ticker] = 1 - ( 2 * self.commission_rate) # Reserve for round-trip
                
            logger.info(f"All assets underperform risk-free rate {current_rf:.4f}, allocating 100% to {risk_free_ticker}")
            return w_series.to_dict()

        # Call optimizer with horizon-scaled cov and risk-free rf
        w_opt = self.optimizer.optimize(
            er=pd.Series(valid_expected_returns, index=valid_symbols),
            cov=pd.DataFrame(cov_horizon, index=valid_symbols, columns=valid_symbols),
            rf=current_rf,
            benchmark_vol=self._get_benchmark_volatility(now),
            allowed_weight_ranges=np.array(allowed_weight_ranges),
            current_weights = np.array(current_weights),
            selector_k = self.selector_k,
            target_volatility = self.target_volatility,
            commission_rate = 0.0,  # commission_rate is enforced afterwards. optimizer fails beacuse: DEFAULT  sum(w) = 1 AND constraint sum(w) ≤ 1 - commissions
        )



        # Clipping (if optimizer constraints failed) and log if clipping occurred
        final_arr = np.clip(w_opt, np.array(allowed_weight_ranges)[:, 0], np.array(allowed_weight_ranges)[:, 1])  # max bounds

        # Ensure total absolute exposure leaves room for commissions
        abs_final_sum = float(np.sum(np.abs(final_arr)))
        max_deployment = 1.0 - (2 * self.commission_rate)
        if abs_final_sum > max_deployment:
            # Scale down to respect commission buffer (safe fallback)
            scale = max_deployment / abs_final_sum
            final_arr = final_arr * scale
            logger.info(f"Scaled weights by {scale:.4f} to reserve cash for commissions")

        # Map to universe
        w_series = pd.Series(0.0, index=self.universe)
        w_series[valid_symbols] = final_arr  # Direct assignment using list indexing

        return w_series.to_dict()


    # ----- trailing and drawdown stops for risk --------------------------------------------
    def _check_trailing_stop(self, instrument: Instrument, price: float):
        sym = instrument.symbol.value
        if len(self.portfolio.analyzer._positions) == 0:
            return
        position = self.portfolio.analyzer.net_position(instrument)
        pos = position.quantity if position else 0
        
        if pos == 0:
            self.trailing_stops.pop(sym, None)
            return

        if pos > 0:                       # long
            high = self.trailing_stops.get(sym, price)
            if price > high:
                self.trailing_stops[sym] = price
            elif price <= high * (1 - self.trailing_stop_max):
                self.order_manager.close_all_positions(instrument, reason="trailing_stop")
                self.trailing_stops.pop(sym, None)
        else:                             # short
            low = self.trailing_stops.get(sym, price)
            if price < low:
                self.trailing_stops[sym] = price
            elif price >= low * (1 + self.trailing_stop_max):
                self.order_manager.close_all_positions(instrument, reason="trailing_stop")
                self.trailing_stops.pop(sym, None)

    def _check_drowdown_stop(self) -> bool:
        
        nav = self._calculate_portfolio_nav()
        
        # Initialize max NAV on first real calculation
        if self.max_registered_portfolio_nav <= 0:
            self.max_registered_portfolio_nav = nav
            logger.info(f"Initialized max NAV to {nav:.2f}")
            return False  # Don't trigger drawdown on initialization
        
        self.max_registered_portfolio_nav = max(self.max_registered_portfolio_nav, nav)
        #self.equity_analyzer.on_equity(ts, nav)
        
        threshold = self.max_registered_portfolio_nav * (1 - self.drawdown_max)
        if nav < threshold:
            logger.warning(f"Drawdown stop triggered: NAV={nav:.2f} < Threshold={threshold:.2f} (Max={self.max_registered_portfolio_nav:.2f})")
            self.order_manager.liquidate_all(self.universe) 
            return True

        return False
    
    def _calculate_portfolio_nav(self) -> float:
        """Calculate total portfolio value: cash + market value of all positions."""
        # Get cash balance
        account = self.portfolio.account(self.venue)
        if account:
            cash_balance = float(account.balance_total(self.strategy_params["currency"]))
        else:
            cash_balance = float(self.strategy_params["initial_cash"])
        
        # Add market value of all open positions (handles multiple positions per instrument)
        total_positions_value = 0.0
        positions_by_instrument = {}
        
        for position in self.cache.positions_open(venue = self.venue):
            instrument_id = position.instrument_id
            bar_type = BarType(instrument_id=instrument_id, bar_spec=self.bar_spec)
            market_value = self.cache.bars(bar_type)[-1].close
            
            
            # Aggregate positions for each instrument (for HEDGING accounts)
            if instrument_id not in positions_by_instrument:
                positions_by_instrument[instrument_id] = 0.0
            positions_by_instrument[instrument_id] += float(position.signed_qty)
        
        # Calculate market value for net positions
        for instrument_id, net_quantity in positions_by_instrument.items():
            bar_type = BarType(instrument_id=instrument_id, bar_spec=self.bar_spec)
            bars = self.cache.bars(bar_type)
            
            if bars:
                current_price = float(bars[-1].close)
                position_value = net_quantity * current_price
                total_positions_value += position_value
        
        nav = cash_balance + total_positions_value
        
        # Sanity check
        if nav <= 0 and cash_balance > 0:
            logger.warning(f"NAV calculation may be incorrect: cash={cash_balance:.2f}, positions={total_positions_value:.2f}")
            nav = cash_balance  # Use cash as fallback
        
        logger.info(f"NAV Calculation: Cash={cash_balance:.2f}, Positions={total_positions_value:.2f}, Total={nav:.2f}")

        return nav
