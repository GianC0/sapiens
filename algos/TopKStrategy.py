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
        self.min_bars_required = self.model_params["window_len"] + 1
        self.optimizer_lookback = freq2pdoffset( self.strategy_params["optimizer_lookback"])
        
        
        # Loader for data access
        venue_name = self.strategy_params["venue_name"]
        self.venue = Venue(venue_name)
        self.loader = CsvBarLoader(cfg=self.strategy_params, venue_name=self.venue.value, columns_to_load=self.model_params["features_to_load"], adjust=self.model_params["adjust"])
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

        # Extract risk parameters from config
        self.selector_k = self.strategy_params.get("top_k", 30)
        self.max_w_abs = self.strategy_params.get("risk_max_weight_abs", 0.03)
        self.target_volatility = self.strategy_params.get("risk_target_volatility_annual", 0.05)
        self.trailing_stop_pct = self.strategy_params.get("risk_trailing_stop_pct", 0.05)
        self.drawdown_pct = self.strategy_params.get("risk_drawdown_pct", 0.15)
        self.adv_lookback = self.strategy_params.get("liquidity", {}).get("adv_lookback", 30)
        self.max_adv_pct = self.strategy_params.get("liquidity", {}).get("max_adv_pct", 0.05)
        self.exec_algo = self.strategy_params.get("execution", {}).get("exec_algo", "twap")
        self.twap_slices = self.strategy_params.get("execution", {}).get("twap", {}).get("slices", 4)
        self.twap_interval_secs = self.strategy_params.get("execution", {}).get("twap", {}).get("interval_secs", 2.5)
        self.can_short = self.strategy_params.get("oms_type", "NETTING") == "HEDGING"

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
    def on_dispose(self) -> None:
        self.order_manager.liquidate_all(self.universe)
        return
    
    def on_update(self, event: TimeEvent):
        if event.name != "update_timer":
            return

        assert pd.to_datetime(event.ts_event, unit="ns", utc=True) > self._last_update_time 

        now = pd.Timestamp(self.clock.utc_now(),)
        
        # check if "now" falls outside market trading hours
        d_r = self.calendar.schedule(start_date=self._last_update_time, end_date= now + freq2pdoffset(self.strategy_params["freq"]) )
        if now not in  market_calendars.date_range(d_r, frequency=self.strategy_params["freq"]).normalize():
            return

        # Skip if no update needed
        if self._last_update_time and now < (self._last_update_time + freq2pdoffset(self.strategy_params["freq"])):
            logger.warning("WIERD STRATEGY UPDATE() CALL")
            return
        
        logger.info(f"Update timer fired at {now}")
        
        # Check if drowdown stop is needed
        self._check_drowdown_stop()

        if not self.model.is_initialized:
            logger.warning("MODEL NOT INITIALIZED WITHIN STRATEGY UPDATE() CALL")
            return
        
        assert self.model.is_initialized

        data_dict = self._cache_to_dict(window=(self.min_bars_required)) # TODO: make it more generic for all models.

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

        # Compute new portfolio target weights for predicted instruments.
        weights = self._compute_target_weights(preds)

        # Portolio Managerment through order manager
        if weights:
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
        
        now = pd.Timestamp(self.clock.utc_now(),)
        
        # Skip if too soon since last retrain
        if self._last_retrain_time and (now - self._last_retrain_time) < self.retrain_offset:
            return
        
        logger.info(f"Starting model retrain at {now}")

        # comput how much data to load
        days_range = self.calendar.schedule(start_date= self._last_retrain_time, end_date=now)
        timestamps = market_calendars.date_range(days_range, frequency=self.strategy_params["freq"]).normalize()
        rolling_window_size = len(timestamps)
        
        # Get updated data window
        data_dict = self._cache_to_dict(window= (rolling_window_size))  # Get all latest
        if not data_dict:
            return
        
        # Update active mask
        self.active_mask = self._compute_active_mask(data_dict)
        
        # Retrain model with warm start
        self.model.update(
            data=data_dict,
            current_time=now,
            retrain_start_date = self._last_retrain_time,
            active_mask=self.active_mask,
            warm_start=self.strategy_params.get("warm_start", False),
            warm_training_epochs=self.strategy_params.get("warm_training_epochs", 1),
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
        dwd_s = self._check_drowdown_stop()

        if dwd_s:
            self.on_dispose()
            return
        
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
            pred_window = df[(df.index > self.valid_end) ]
            if len(pred_window) < self.pred_len :
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

            raise Exception("Model Not Initialized")
            
            #model = ModelClass(**self.model_params)
            #train_data = self._get_data(start = self.train_start , end = self.train_end)
            #model.initialize(train_data)
            

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
            assert pd.to_datetime(bars[0].ts_event, unit="ns", utc=True ) > self._last_update_time
            
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
        # Calculate current weights for fallback
        current_weights = []

        # Current portfolio net asset value
        self.nav = self._calculate_portfolio_nav()

        for i, symbol in enumerate(self.universe):
            
            instrument_id = InstrumentId(Symbol(symbol), self.venue)
            bar_type = BarType(instrument_id=instrument_id, bar_spec=self.bar_spec)
            bars = self.cache.bars(bar_type)
            if not bars or len(bars) < 2:
                continue

            # returns per single period (same freq as preds' base period)
            lookback_n = max(2, int(self.optimizer_lookback.n))
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
                allowed_weight_ranges.append([0, 1])
                continue

            net_position = 0.0
            for position in self.cache.positions_open():
                if position.instrument_id == instrument_id:
                    net_position += float(position.signed_qty)
            current_w = (net_position * prices[symbol] / self.nav) if (net_position and self.nav > 0) else 0.0
            current_weights.append(current_w) 

            # ADV
            volumes = [float(b.volume) for b in bars[-self.adv_lookback:]] if len(bars) >= self.adv_lookback else [float(b.volume) for b in bars]
            adv = float(np.mean(volumes)) if volumes else 0.0
            max_w_relative = min((adv * self.max_adv_pct * prices[symbol]) / self.nav, self.max_w_abs) if self.nav > 0 else self.max_w_abs

            # compute min or min/max weight for instrument in portfolio 
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

        # prepare mu 
        valid_mu = np.array([preds.get(s, 0.0) for s in valid_symbols])

        # If all expected returns <= rf_horizon then allocate to risk-free ticker (if available)
        # TODO: this has to be ensured it's worth with commissions
        if np.all(valid_mu <= current_rf):
            w_series = pd.Series(0.0, index=self.universe)
            if risk_free_ticker in self.universe:
                w_series[risk_free_ticker] = 1.0
            return w_series.to_dict()

        # Call optimizer (M2) with horizon-scaled cov and rf_h
        w_opt = self.optimizer.optimize(
            mu=valid_mu,
            cov=cov_horizon,
            rf=current_rf,
            benchmark_vol=self._get_benchmark_volatility(now),
            allowed_weight_ranges=np.array(allowed_weight_ranges),
            current_weights = np.array(current_weights),
        )

        if len(w_opt) != len(valid_symbols):
            logger.warning("Optimizer returned unexpected vector length. Falling back to equal weights.")
            w_opt = np.ones(len(valid_symbols)) / len(valid_symbols)

        # Take top-K by absolute weight
        k = min(self.selector_k, len(valid_symbols))
        idx_sorted = np.argsort(-np.abs(w_opt))
        top_idx = idx_sorted[:k]

        final_arr = np.zeros_like(w_opt)

        for i in top_idx:
            val = w_opt[i]
            # clip to allowed range
            w_min, w_max = allowed_weight_ranges[i]
            final_arr[i] = float(np.clip(val, w_min, w_max))

        # Optional renormalization: preserve total abs exposure of the selected set from optimizer
        abs_selected_sum = float(np.sum(np.abs(w_opt[top_idx])))
        abs_final_sum = float(np.sum(np.abs(final_arr[top_idx])))
        if abs_final_sum > 0 and abs_selected_sum > 0:
            scale = abs_selected_sum / abs_final_sum
            final_arr[top_idx] = final_arr[top_idx] * scale

        # Map back to full universe
        w_series = pd.Series(0.0, index=self.universe)
        for i, symbol in enumerate(valid_symbols):
            w_series[symbol] = final_arr[i]

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
            elif price <= high * (1 - self.trailing_stop_pct):
                self.order_manager.close_all_positions(instrument, reason="trailing_stop")
                self.trailing_stops.pop(sym, None)
        else:                             # short
            low = self.trailing_stops.get(sym, price)
            if price < low:
                self.trailing_stops[sym] = price
            elif price >= low * (1 + self.trailing_stop_pct):
                self.order_manager.close_all_positions(instrument, reason="trailing_stop")
                self.trailing_stops.pop(sym, None)

    def _check_drowdown_stop(self) -> bool:
        
        nav = self._calculate_portfolio_nav()
        self.max_registered_portfolio_nav = max(self.max_registered_portfolio_nav, self.nav)
        #self.equity_analyzer.on_equity(ts, nav)
        
        if nav < self.max_registered_portfolio_nav * (1 - self.drawdown_pct):
            self.order_manager.liquidate_all(self.universe) 
            return True

        return False
    
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
        logger.info(f"NAV Calculation: Cash={cash_balance:.2f}, Positions={total_positions_value:.2f}, Total={nav:.2f}")
        return nav
