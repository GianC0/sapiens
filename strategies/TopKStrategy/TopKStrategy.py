"""
Generic long/short strategy for Nautilus Trader.

* Consumes ANY model that follows SapiensModel
* Data feed provided by CsvBarLoader (Bar, FeatureBar)
* Risk controls: draw-down, trailing stops, ADV cap, fee/slippage models

Implementation reference:
https://nautilustrader.io/docs/latest/concepts/strategies
"""
from __future__ import annotations
from csv import Error
import importlib
from typing import Any
from datetime import datetime
from pathlib import Path
from tracemalloc import start
from typing import Dict, List, Optional
from pandas.tseries.frequencies import to_offset
import numpy as np
import pandas as pd
from sympy import total_degree
from collections import defaultdict
import yaml
import pandas_market_calendars as market_calendars
import torch
from decimal import Decimal
from dataclasses import dataclass, field
import logging
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.currencies import USD,EUR
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.common.component import TimeEvent, Logger
from nautilus_trader.core.data import Data
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data import Bar, BarType, InstrumentStatus, InstrumentClose, TradeTick
from nautilus_trader.data.messages import RequestBars
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide, TriggerType, TrailingOffsetType, OrderType
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Currency
from nautilus_trader.backtest.models import FillModel, FeeModel
from nautilus_trader.core.nautilus_pyo3 import CurrencyType
from nautilus_trader.model.position import Position
from nautilus_trader.adapters.databento import DATABENTO_CLIENT_ID
from nautilus_trader.model.events import (
    OrderAccepted, OrderCanceled, OrderCancelRejected,
    OrderDenied, OrderEmulated, OrderEvent, OrderExpired,
    OrderFilled, OrderInitialized, OrderModifyRejected,
    OrderPendingCancel, OrderPendingUpdate, OrderRejected,
    OrderReleased, OrderSubmitted, OrderTriggered, OrderUpdated,
    PositionChanged, PositionClosed, PositionEvent, PositionOpened
)

from models import SapiensModel
logger = logging.getLogger(__name__)
# ----- project imports -----------------------------------------------------

from engine.OptimizerFactory import create_optimizer
from models.utils import freq2pdoffset, freq2barspec
from engine.OrderManager import OrderManager
from models.utils import freq2bartype, freq2pdoffset
from strategies.SapiensStrategy import SapiensStrategy, SapiensStrategyConfig


# ========================================================================== #
# Strategy
# ========================================================================== #
class TopKStrategy(SapiensStrategy):
    """
    Long/short TopK equity strategy, model-agnostic & frequency-agnostic.
    """


    # ------------------------------------------------------------------ #
    def __init__(self, config: SapiensStrategyConfig):  
        super().__init__(config)  

        # safe handling of variable
        currency =self.strategy_params["currency"]
        if currency == "USD":
            self.strategy_params["currency"] = Currency(code='USD', precision=3, iso4217=840, name='United States dollar', currency_type = CurrencyType.FIAT ) #
        elif currency == "EUR":
            self.strategy_params["currency"] = Currency(code='EUR', precision=3, iso4217=978, name='Euro', currency_type=CurrencyType.FIAT)
        else: # currency is already in nautilus format
            raise Error("Currency not implemented correctly") 
        
        self.model_params["model_dir"] = Path(self.model_params["model_dir"])

        self.calendar = market_calendars.get_calendar(self.strategy_params["calendar"])

        # Core timing parameters
        self.data_load_start = pd.Timestamp(self.strategy_params["data_load_start"])
        self.train_start = pd.Timestamp(self.strategy_params["train_start"])
        self.train_end = pd.Timestamp(self.strategy_params["train_end"])
        self.valid_start = pd.Timestamp(self.strategy_params["valid_start"])
        self.valid_end = pd.Timestamp(self.strategy_params["valid_end"])
        self.inference_window_offset = freq2pdoffset(self.model_params["inference_window"])
        self.retrain_offset = freq2pdoffset(self.model_params["retrain_offset"])
        self.train_offset = freq2pdoffset(self.model_params["train_offset"])
        self.pred_len = int(self.model_params["pred_len"])

        # data load start should be conservative: 
        # if bars within retrain_date - data_load_start < min_bars_required, then on_historical fails to load necessary data.  
        #self.data_load_start = pd.Timestamp(self.strategy_params["data_load_start"])     # NON-CONSERVATIVE
        #self.data_load_start = self.train_start                                           # CONSERVATIVE

        # NOT NEEDED
        #self.backtest_start = pd.Timestamp(self.strategy_params["backtest_start"], tz="UTC")
        #self.backtest_end = pd.Timestamp(self.strategy_params["backtest_end"], tz="UTC")
        


        # Model and data parameters
        self.model: Optional[SapiensModel] = None
        self.model_name = self.model_params["model_name"]
        self.bar_spec = freq2barspec( self.strategy_params["freq"])
        self.optimizer_lookback = freq2pdoffset( self.strategy_params["optimizer_lookback"])
        self.min_bars_required = self.model_params["window_len"]
        
        # Commissions Fee Model
        self.fee_model = self._import_fee_model()

        # Cash buffer to always keep for rounding errors
        self.cash_buffer = self.strategy_params["cash_buffer"]

        # Loader for data access
        venue_name = self.strategy_params["venue_name"]
        self.venue = Venue(venue_name)

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

        # Extract risk params
        self.max_w_abs = self.strategy_params["risk"]["max_weight_abs"]
        self.drawdown_max = self.strategy_params["risk"]["drawdown_max"]
        self.trailing_stop_max = self.strategy_params["risk"]["trailing_stop_max"]
        self.target_volatility = self.strategy_params["risk"]["target_volatility"]
        self.max_balance_total = None
        
        # Extract execution parameters from config
        self.adv_lookback = pd.Timedelta(self.strategy_params["liquidity"]["adv_lookback"])
        self.max_adv_pct = self.strategy_params["liquidity"]["max_adv_pct"]
        self.twap_slices = self.strategy_params["execution"]["twap"]["slices"]
        self.twap_interval_secs = self.strategy_params["execution"]["twap"]["interval_secs"]
        
        self.can_short = self.strategy_params["oms_type"] == "HEDGING"

        # Portfolio optimizer
        # TODO: add risk_aversion config parameter for MaxQuadraticUtilityOptimizer
        # TODO: make sure to pass proper params to create_optimizer depending on the optimizer all __init__ needed by any optimizer

        self.weight_bounds = (0, 1)  # long-only positions
        if self.can_short:
            self.weight_bounds = (-1, 1)   # (-) = short position ,  (+) = long position 
        optimizer_name = self.strategy_params.get("optimizer_name", "max_sharpe")
        self.optimizer = create_optimizer(name = optimizer_name, max_adv_pct = self.max_adv_pct, weight_bounds = self.weight_bounds)
        self.rebalance_only = self.strategy_params.get("rebalance_only", False)  # Rebalance mode
        self.top_k = self.strategy_params.get("top_k", 50)  # Portfolio size
        

        # Order Manager for any strategy
        self.order_manager = OrderManager(self, self.strategy_params)
        logger.info("OrderManager initialized")

        # Final liquidation flag
        self.final_liquidation_happend = False
        


    # ================================================================= #
    # Nautilus event handlers
    # ================================================================= #
    def on_start(self): 
        """Initialize strategy."""

        # Select universe based on stocks active at walk-forward start with enough history
        #self._select_universe()
        self.universe = list(set(inst.id.value for inst in self.cache.instruments()))

        self.max_balance_total = self._calculate_portfolio_nav()

        #TODO:fix
        #self.on_historical_data(data=None)

        # Subscribe to bars for selected universe
        for iid in self.universe:
            iid = InstrumentId.from_str(iid)
            bar_type = freq2bartype(instrument_id = iid, frequency=self.strategy_params["freq"])
            #bar_type = BarType(instrument_id = iid, bar_spec =freq2barspec(freq=self.strategy_params["freq"]) )

            # request historical bars
            # in live should be done through self.request_bars
            #self.on_historical_data(bar_type = bar_type, start = self.data_load_start)
            self.request_trade_ticks(instrument_id = iid, 
                                    start=self.train_start.to_pydatetime(),
                                    client_id=DATABENTO_CLIENT_ID)
            # TODO: to fix proper load of historical data during backtest
            #self.request_bars(
            #    bar_type=bar_type,
            #    start=self.train_start.to_pydatetime(),
            #    client_id=DATABENTO_CLIENT_ID,
            #    end = None,
            #    callback=self.on_historical_data,     # called with the request ID when completed
            #    params=None,                          # dict[str, Any], optional
            #)
            
            
            
            # Subscribe bars for walk forward
            self.subscribe_bars(bar_type, client_id=DATABENTO_CLIENT_ID)
            self.subscribe_trade_ticks(iid, client_id=DATABENTO_CLIENT_ID)
            
            # Subscribe to instrument events
            #self.subscribe_mark_prices(instrument.id)
            #self.subscribe_instrument_close(instrument.id)

        # Build and initialize model
        self.model = self._initialize_model()

        # Set initial update time to avoid immediate firing
        self.active_mask = torch.ones(len(self.cache.instruments()), dtype=torch.bool)
        
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
            interval=pd.Timedelta(days=self.retrain_offset.n),
            callback=self.on_retrain,
        )    

        # Final liquidation timer one bar before the end of backtest
        date_range = self.calendar.schedule(start_date=str(self.valid_start), end_date=str(self.valid_end), )
        if not date_range.empty:
            last_trading_time = market_calendars.date_range(date_range, frequency=pd.Timedelta(freq2pdoffset(self.strategy_params["freq"])))[-1]
            liquidation_time = pd.Timestamp(last_trading_time) - freq2pdoffset(self.strategy_params["freq"])
            
            self.clock.set_time_alert(
                name="final_liquidation",
                alert_time=liquidation_time,
                callback=self.on_final_liquidation
            )
            logger.info(f"Final liquidation scheduled for {liquidation_time}")

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
    

    def on_final_liquidation(self, event: TimeEvent):
        """Liquidate all positions before backtest ends."""
        if event.name != "final_liquidation":
            return
        
        logger.info("Final liquidation: closing all positions")
        
        for instrument in self.cache.instruments(venue=self.venue):
            self.close_all_positions(instrument.id)
            self.cancel_all_orders(instrument.id)
            self.unsubscribe_instrument(instrument_id = instrument.id)
        
        self.final_liquidation_happend = True

    def on_dispose(self) -> None:
        """Log final state."""
        final_nav = self._calculate_portfolio_nav()
        logger.info(f"Final NAV at disposal: {final_nav:.2f}")

    def on_update(self, event: TimeEvent):
        if event.name != "update_timer":
            return
        if self.clock.utc_now() <= self.valid_start:
            logger.debug("This update step is used to load initial data.")
            return
        if self.final_liquidation_happend:
            return

        # Overall portfolio drawdown condition
        if self._calculate_portfolio_nav() < self.max_balance_total * self.drawdown_max:
            self.on_final_liquidation(event)
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
            logger.debug(f"Non-trading day at {now}")
            return

        # Skip if no update needed
        if self._last_update_time and now < (self._last_update_time + freq2pdoffset(freq)):
            logger.warning("Update called too soon, skipping")
            return
        
        logger.debug(f"Update timer fired at {now}")
        

        if not self.model.is_initialized:
            logger.warning("MODEL NOT INITIALIZED WITHIN STRATEGY UPDATE() CALL")
            return
        
        assert self.model.is_initialized

        data_dict = self._cache_to_dict(window=(self.min_bars_required))

        if not data_dict:
            logger.warning("DATA DICTIONARY EMPTY WITHIN STRATEGY UPDATE() CALL")
            return
        
        # TODO: superflous. consider removing compute active mask
        assert torch.equal(self.active_mask, self._compute_active_mask()) , "Active mask mismatch between strategy and data engine"

        #assert self.universe == self.model._universe, "Universe mismatch between strategy and model"

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
        if not weights or all(abs(w) < 1e-6 for w in weights.values()) or len(weights) == 0:
            logger.warning(f"Time: ### {self.clock.utc_now()} ###. No valid weights computed, skipping rebalance")
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
        
        if self.clock.utc_now() <= self.valid_start:
            logger.debug("No retrain before all initial data is loaded.")
            return
        
        now = pd.Timestamp(self.clock.utc_now(),)
        
        # Skip if too soon since last retrain
        if self._last_retrain_time and now < self._last_retrain_time - self.retrain_offset:
            return
        
        logger.debug(f"Starting model retrain at {now}")

        # comput how much data to load
        days_range = self.calendar.schedule(start_date= self._last_retrain_time, end_date=now)
        timestamps = market_calendars.date_range(days_range, frequency=self.strategy_params["freq"])
        
        
        # ═══════════════════════════════════════════════════════════════════
        # ADAPTIVE WINDOW: Ensure minimum training size
        # ═══════════════════════════════════════════════════════════════════
        # ensures case at t=0 where len(timestamps) for new bars < min_bars_required 
        rolling_window_size = max(len(timestamps), self.min_bars_required + self.pred_len)
        
        # Log when falling back to minimum (includes overlap with previous training)
        if len(timestamps) > self.min_bars_required:
            logger.debug(f"[on_retrain] Rollowing window needed for retrain ({len(timestamps)} bars) > minimum required ({self.min_bars_required} bars). Overlap: {len(timestamps) - self.min_bars_required} bars from previous training.")
        else:
            logger.debug(
                f"[on_retrain] Using rolling window of {rolling_window_size} bars (sufficient for training).")
        
        # Get updated data window
        data_dict = self._cache_to_dict(window = (rolling_window_size))  # Get all latest
        if not data_dict:
            return
        
        total_bars = len(next(iter(data_dict.values())))
        
        # Update active mask
        self.active_mask = self._compute_active_mask()
        
        # Retrain model with warm start
        self.model.update(
            data=data_dict,
            current_time=now,
            retrain_start_date = self._last_retrain_time,
            active_mask=self.active_mask,
            total_bars = total_bars,
        )
        
        self._last_retrain_time = now
        logger.debug(f"Model retrain completed at {now}")
            
            

    # ================================================================= #
    # DATA handlers
    # ================================================================= #
    def on_bar(self, bar: Bar):  
        
        logger.debug(f"Received Bar: {bar}")

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
        # TODO: update the model mask and ensure the loader still provides same input shape to the model for prediction
        # remove from cache ??
        pass

    def on_historical_data(self, data: Data) -> None: 
        """
        Load historical data for model initialization at t=0.
        Aggregates trade ticks from catalog into OHLCV bars.
        Processes each instrument separately.
        """
        # Get catalog path from strategy config
        catalog_path = self.strategy_params.get("catalog_path")
        if not catalog_path:
            logger.error("catalog_path not found in strategy config")
            return
        
        try:
            from nautilus_trader.persistence.catalog import ParquetDataCatalog
            catalog = ParquetDataCatalog(path=str(catalog_path))
            
            # Query trade ticks from catalog for all instruments
            start_time = self.data_load_start
            end_time = self.train_end
            
            logger.info(f"Loading ticks from {start_time} to {end_time} for {len(self.universe)} instruments")
            
            ticks = catalog.trade_ticks(
                start=start_time,
                end=end_time
            )
            
            if not ticks or len(ticks) == 0:
                logger.warning(f"No ticks found")
                return
            
            # Convert ticks to DataFrame
            tick_data = []
            for tick in ticks:
                tick_data.append({
                    'ts_event': tick.ts_event,
                    'instrument_id': tick.instrument_id.value,
                    'price': float(tick.price),
                    'size': float(tick.size)
                })
            
            df_all = pd.DataFrame(tick_data)
            df_all['ts_event'] = pd.to_datetime(df_all['ts_event'], unit='ns')

            # Then localize to Eastern Time
            df_all['ts_event'] = df_all['ts_event'].dt.tz_localize('US/Eastern', ambiguous='infer')
            logger.info(f"Tick time range (UTC): {df_all['ts_event'].min()} to {df_all['ts_event'].max()}")
            
            # Get market schedule for filtering
            schedule = self.calendar.schedule(start_date=start_time, end_date=end_time)
            market_opens = schedule['market_open'].dt.tz_convert('UTC')
            market_closes = schedule['market_close'].dt.tz_convert('UTC')
            logger.info(f"Market hours (UTC): {market_opens.iloc[0]} to {market_closes.iloc[0]}")

            # Create unified bar index using market calendar
            freq = self.strategy_params["freq"]
            unified_index = market_calendars.date_range(schedule, frequency=freq)
            
            # Process each instrument separately
            total_bars_added = 0
            for iid in self.universe:
                try:
                    # Query trade ticks from catalog
                    ticks = catalog.trade_ticks(
                        instrument_ids=[iid],
                        start=start_time,
                        end=end_time
                    )
                    
                    if not ticks or len(ticks) == 0:
                        logger.warning(f"No ticks found for {iid}")
                        continue
                    
                    # Vectorized conversion: pre-allocate arrays
                    n_ticks = len(ticks)
                    timestamps = np.empty(n_ticks, dtype='int64')
                    prices = np.empty(n_ticks, dtype=np.float64)
                    sizes = np.empty(n_ticks, dtype=np.float64)
                    
                    # Batch extract attributes
                    for i, tick in enumerate(ticks):
                        timestamps[i] = tick.ts_event
                        prices[i] = float(tick.price)
                        sizes[i] = float(tick.size)
                    
                    # Create DataFrame with timezone-aware index
                    df = pd.DataFrame(
                        {'price': prices, 'size': sizes},
                        index=pd.to_datetime(timestamps, unit='ns', utc=True)
                    )

                    # Filter to market hours
                    mask = pd.Series(False, index=df.index)
                    for open_time, close_time in zip(market_opens, market_closes):
                        mask |= (df.index >= open_time) & (df.index <= close_time)
                    
                    df_filtered = df[mask]
                                        
                    if len(df_filtered) == 0:
                        logger.warning(f"No ticks in market hours for {iid}")
                        continue
                    
                    df_filtered = df_filtered.copy()

                    # Find which bar each tick belongs to
                    bar_indices = np.searchsorted(unified_index, df_filtered.index, side='right') - 1
                    
                    # Clip to valid range
                    bar_indices = np.clip(bar_indices, 0, len(unified_index) - 1)
                    
                    # Assign bar timestamps
                    
                    df_filtered['bar_time'] = unified_index[bar_indices]
                    
                    # Group by bar and aggregate
                    ohlcv_df = df_filtered.groupby('bar_time').agg({
                        'price': ['first', 'max', 'min', 'last'],
                        'size': 'sum'
                    })
                    
                    # Flatten columns
                    ohlcv_df.columns = ['open', 'high', 'low', 'close', 'volume']
                    
                    # Reindex to ensure all periods present (even with no trades)
                    ohlcv_df = ohlcv_df.reindex(unified_index)
                    
                    # Forward-fill OHLC, zero-fill volume
                    ohlcv_df[['open', 'high', 'low', 'close']] = ohlcv_df[['open', 'high', 'low', 'close']].ffill()
                    ohlcv_df['volume'] = ohlcv_df['volume'].fillna(0)
                    
                    if ohlcv_df['close'].isna().all():
                        logger.warning(f"No valid data for {iid} after aggregation")
                        continue
                    
                    
                    
                    # Get instrument and bar_type
                    instrument = self.cache.instrument(InstrumentId.from_str(iid))
                    bar_type = freq2bartype(instrument_id=InstrumentId.from_str(iid), frequency=freq)
                    
                    # Convert to Nautilus Bar objects
                    bars = []
                    for ts, row in ohlcv_df.iterrows():
                        if pd.isna(row['close']):
                            continue
                        
                        bar = Bar(
                            bar_type=bar_type,
                            open=Price(row['open'], precision=instrument.price_precision),
                            high=Price(row['high'], precision=instrument.price_precision),
                            low=Price(row['low'], precision=instrument.price_precision),
                            close=Price(row['close'], precision=instrument.price_precision),
                            volume=Quantity(row['volume'], precision=0),
                            ts_event=int(ts.value),  # nanoseconds
                            ts_init=int(ts.value)
                        )
                        bars.append(bar)
                    
                    # Add bars to cache for this instrument
                    if bars:
                        self.cache.add_bars(bars)
                        total_bars_added += len(bars)
                        logger.info(f"✅ Added {len(bars)} bars for {iid}")
                    
                except Exception as e:
                    logger.error(f"Error processing {iid}: {e}", exc_info=True)
                    continue
            
            logger.info(f"✅ Total: Added {total_bars_added} bars across {len(self.universe)} instruments")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}", exc_info=True)
        

    # Unused ATM
    def on_data(self, data: Data) -> None:  # Custom data passed to this handler
        assert False
        return
    def on_signal(self, signal) -> None:  # Custom signals passed to this handler
        return
    
    # ================================================================= #
    # ORDER MANAGEMENT  -> OrderManager from OrderManager.py
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
        """Register fill order stats."""
        if self.order_manager:
            self.order_manager.on_order_filled(event)
   
    def on_order_event(self, event: OrderEvent) -> None:
        """Catch-all for any unhandled order events."""
        pass

    # ================================================================= #
    # POSITION MANAGEMENT
    # ================================================================= #

    def on_position_opened(self, event: PositionOpened) -> None:

        # Submit initial trailing stop order
        trailing_order = self.order_factory.trailing_stop_market(
            instrument_id=event.instrument_id,
            order_side=OrderSide.SELL,
            quantity=event.quantity,
            trigger_type=TriggerType.LAST_PRICE,
            trigger_price = None,
            trailing_offset=Decimal(self.trailing_stop_max * 10000),
            trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
        )
        self.submit_order(trailing_order)

    def on_position_changed(self, event: PositionChanged) -> None:
        
        instrument_id = event.instrument_id

        # cancel last trailing order to update the order quantity
        for order in self.cache.orders(instrument_id = instrument_id, side = OrderSide.SELL):
            if order.order_type == OrderType.TRAILING_STOP_MARKET:
                self.cancel_order(order)
                break

        # Submit another trailing stop order with updated quantity to liquidate
        trailing_order = self.order_factory.trailing_stop_market(
            instrument_id=instrument_id,
            order_side=OrderSide.SELL,
            quantity=event.quantity,
            trigger_type=TriggerType.LAST_PRICE,
            trigger_price = None,
            trailing_offset=Decimal(self.trailing_stop_max * 10000),
            trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
        )
        self.submit_order(trailing_order)

    def on_position_closed(self, event: PositionClosed) -> None:
        
        # Cancel all risk orders when position is closed ( mainly for TRAILING STOP MARKET)
        instrument_id = event.instrument_id
        self.cancel_all_orders( instrument_id = instrument_id )

    def on_position_event(self, event: PositionEvent) -> None:  # All position event messages are eventually passed to this handler
        pass
    
    
    # ================================================================= #
    # ACCOUNT & PORTFOLIO 
    # ================================================================= #

    # TODO: implement 

    # ================================================================= #
    # INTERNAL HELPERS
    # ================================================================= #

    def _select_universe(self):
        """Select instruments with sufficient data from cache."""
        selected = []
        
        for instrument in self.cache.instruments(venue=self.venue):
            symbol = instrument.id.symbol.value
            bar_type = freq2bartype(instrument_id = symbol, frequency=self.strategy_params["freq"])
            
            bars = self.cache.bars(bar_type)
            if not bars or len(bars) < self.min_bars_required:
                continue
            
            # Check data extends into valid period
            if bars:
                last_bar_time = pd.Timestamp(bars[0].ts_event, unit='ns', tz="UTC")
                if last_bar_time < self.valid_start:
                    continue
            
            selected.append(symbol)

        logger.info(f"Selected {len(selected)} from {len(self.cache.instruments(venue=self.venue))} candidates (including benchmark and risk-free) ")
        self.universe = sorted(selected)
    def _initialize_model(self) -> SapiensMode:
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

            # Load all needed bars for the initialization
            days_range = self.calendar.schedule(start_date=self.train_start, end_date=self.train_end)
            timestamps = market_calendars.date_range(days_range, frequency=self.strategy_params["freq"])
            total_bars = len(timestamps)
            train_data = self._cache_to_dict(window=total_bars)
            
            model = ModelClass(**self.model_params)
            
            model.initialize(data = train_data, total_bars = total_bars)

            raise Exception("Model Not Initialized")
            

        return model

    def _submit_risk_orders_for_position(self, position: Position, entry_price: float):
        """Submit stop-loss and trailing stop orders."""
        instrument_id = position.instrument_id
        position_id = str(position.id)
    
        


        
    def _cache_to_dict(self, window: Optional[int]) -> Dict[str, pd.DataFrame]:
        """
        Convert cache data within the specified window to dictionary format expected by model.
        Efficient implementation using nautilus trader cache's native methods.
        """
        data_dict = {}
        
        if not window:
            window = self.strategy_params.get("engine", {}).get("cache", {}).get("bar_capacity", 4096)
        
        for iid in self.universe:
            instrument_id = InstrumentId.from_str(iid)
            bar_type = freq2bartype(instrument_id = instrument_id, frequency=self.strategy_params["freq"])
            
            bars = self.cache.bars(bar_type)
            if not bars or len(bars) < self.min_bars_required:
                continue
            
            bars_to_use =bars[:window]
            
            # Build DataFrame directly - single pass
            df = pd.DataFrame({
                'Open': [float(b.open) for b in bars_to_use],
                'High': [float(b.high) for b in bars_to_use],
                'Low': [float(b.low) for b in bars_to_use],
                'Close': [float(b.close) for b in bars_to_use],
                'Volume': [float(b.volume) for b in bars_to_use]
            }, index=[pd.Timestamp(b.ts_event, unit='ns', tz="UTC") for b in bars_to_use])
            
            df.sort_index(inplace=True)
            data_dict[iid] = df
        
        return data_dict
    
    def _get_risk_free_return(self) -> float:
        """Get current risk-free return from cache."""
        rf_ticker = self.strategy_params.get("risk_free_ticker")
        if not rf_ticker:
            return 0.0
        
        try:
            rf_instrument_id = InstrumentId.from_str(rf_ticker)
            bar_type = freq2bartype(instrument_id = rf_instrument_id, frequency=self.strategy_params["freq"])
            bars = self.cache.bars(bar_type)
            
            if bars and len(bars) >= 2:
                latest_close = float(bars[0].close)
                previous_close = float(bars[1].close)
                rf = (latest_close - previous_close) / previous_close
                if rf > 0:
                    return rf
                return 1e-6
        except Exception as e:
            logger.warning(f"Could not get risk-free return: {e}")
        
        return 1e-6


    # TODO: superflous. consider removing
    def _compute_active_mask(self) -> torch.Tensor:
        """Compute mask for active instruments."""
        mask = ~torch.ones(len(self.universe), dtype=torch.bool)
        
        for bt in self.cache.bar_types():
            idx = self.universe.index(bt.instrument_id.value)
            # Check if data is recent enough
            last_time = pd.to_datetime(self.cache.bar(bt).ts_event, unit='ns', origin='unix', utc=True)
            now = self.clock.utc_now()
            mask[idx] = last_time + freq2pdoffset(self.strategy_params["freq"]) >= now
    
        return mask

    # ----- weight optimiser ------------------------------------------
    
    def _get_benchmark_volatility(self) -> Optional[float]:
        """Calculated from cached bars."""
        benchmark_ticker = self.strategy_params.get("benchmark_ticker")
        if not benchmark_ticker:
            return None
        
        try:
            benchmark_instrument_id = InstrumentId.from_str(benchmark_ticker)
            bar_type = freq2bartype(instrument_id = benchmark_instrument_id, frequency=self.strategy_params["freq"])
            benchmark_bars = self.cache.bars(bar_type)
            
            if not benchmark_bars or len(benchmark_bars) < 2:
                return None
            
            lookback_periods = int(self.optimizer_lookback.n)
            bars_to_use = benchmark_bars[:min(lookback_periods, len(benchmark_bars))]
            
            closes = [float(b.close) for b in reversed(bars_to_use)]
            returns = pd.Series(closes).pct_change().dropna()
            
            if len(returns) < 2:
                return None
            
            return float(returns.std())
            
        except Exception as e:
            logger.warning(f"Could not calculate benchmark volatility: {e}")
            return None

    def _compute_target_weights(self, preds: Dict[str, float]) -> Dict[str, float]:
        """Compute target portfolio weights using optimizer with vectorized operations."""


        current_rf = self._get_risk_free_return()
        nav = self._calculate_portfolio_nav()
        cash_available = float(self.portfolio.account(self.venue).balance_free(self.strategy_params["currency"])) - self.cash_buffer
        now = pd.Timestamp(self.clock.utc_now())

        # Pre-fetch all data once
        all_ticks = {iid: self.cache.trade_ticks(InstrumentId.from_str(iid)) for iid in self.universe}
        all_positions = defaultdict(float)

        for p in self.cache.positions_open(venue=self.venue):
            all_positions[p.instrument_id.value] += float(p.signed_qty)
        
        # Data collection (Pre-allocate arrays with max possible size)
        valid_symbols = []
        returns_list = []
        valid_expected_returns = np.zeros(len(self.universe))
        current_weights = np.zeros(len(self.universe))
        prices = np.zeros(len(self.universe))
        allowed_weight_ranges = np.zeros((len(self.universe), 2))
        valid_symb_count = 0

        for symbol in self.universe:
            ticks = all_ticks.get(symbol)
            if not ticks or len(ticks) < 2:
                continue

            current_price = float(ticks[0].price)
            net_position = all_positions.get(symbol, 0.0)
            current_w = (net_position * current_price / nav) if (net_position and nav > 0) else 0.0
            
            # Returns calculation
            lookback_n = max(2, int(self.optimizer_lookback.n))
            returns = pd.Series([float(t.price) for t in ticks[-lookback_n:]]).pct_change().dropna()
            if len(returns) == 0:
                continue
        
            # ADV dollar volume with EMA (with half-life 5 days)
            trading_hours = self.calendar.schedule(start_date=now, end_date=now).pipe(lambda s: (s["market_close"] - s["market_open"]).dt.total_seconds() / 3600.0).iloc[0]
            n_ticks = int(pd.Timedelta(f'{trading_hours}h') * self.adv_lookback.days / pd.Timedelta(self.strategy_params["freq"]))
            recent_ticks = ticks[-n_ticks:] if n_ticks < len(ticks) else ticks
            
            timestamps = np.empty(n_ticks, dtype='datetime64[ns]')
            currency_volumes = np.empty(n_ticks, dtype=np.float64)
            for i, t in enumerate(recent_ticks):
                timestamps[i] = t.ts_event
                currency_volumes[i] = float(t.price) * float(t.size)
            df_ticks_daily = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, utc=True),
                'currency_volume': currency_volumes
            }).set_index('timestamp').resample('1D').sum()
            # NOTE: this is average daily share volume (already accounts for price)
            adv_in_currency = float(df_ticks_daily['currency_volume'].ewm(halflife=5).mean().iloc[-1])


            # Weight bounds: constrain instrument by ADV liquidity and max allocation fraction within portfolio:
            # - keep risk-free unconstrained
            # - enforce constraints on any other type of asset
            if symbol == self.strategy_params.get("risk_free_ticker"):
                w_min = self.weight_bounds[0]
                w_max = self.weight_bounds[1]
            else:
                max_w_constrained = min((adv_in_currency * self.max_adv_pct) / nav, self.max_w_abs)
                w_min = max(-max_w_constrained, self.weight_bounds[0])
                w_max = min(max_w_constrained, self.weight_bounds[1])
            
            # Store for weight optimization
            valid_symbols.append(symbol)
            returns_list.append(returns.values)
            valid_expected_returns[valid_symb_count] = preds.get(symbol, 0.0)
            current_weights[valid_symb_count] = current_w
            prices[valid_symb_count] = current_price
            allowed_weight_ranges[valid_symb_count] = [w_min, w_max]
            valid_symb_count += 1

        if valid_symb_count <= 1:
            return {}
        
        # Trim arrays to actual size
        valid_expected_returns = valid_expected_returns[:valid_symb_count]
        current_weights = current_weights[:valid_symb_count]
        prices = prices[:valid_symb_count]
        allowed_weight_ranges = allowed_weight_ranges[:valid_symb_count]

        # Covariance from returns
        returns_df = pd.DataFrame(returns_list).T
        cov_horizon = returns_df.cov().values * float(self.pred_len) # scaled to pred_len horizon

        # Just Keep cash and sell only if expected loss > commissions
        if np.all(valid_expected_returns <= current_rf):
            logger.debug("All expected returns <= risk-free rate. Minimizing losses..")
            
            # Start with current weights (default: keep everything)
            defensive_weights = current_weights.copy()
            
            # Identify positions with negative expected returns
            has_position_mask = np.abs(current_weights) > 1e-8
            negative_return_mask = valid_expected_returns < 0
            evaluate_mask = has_position_mask & negative_return_mask
            
            if not np.any(evaluate_mask):
                # No losing positions to evaluate - keep everything
                w_series = pd.Series(0.0, index=self.universe)
                w_series[valid_symbols] = defensive_weights
                logger.debug("No losing positions found. Keeping all current positions and holding cash.")
                return w_series.to_dict()
            
            # Batch calculate expected losses
            position_values = np.abs(current_weights) * nav
            expected_losses = position_values * np.abs(valid_expected_returns)
            
            # Batch commission calculation for positions to evaluate
            evaluate_indices = np.where(evaluate_mask)[0]
            commissions = np.zeros(len(valid_symbols))
            
            for idx in evaluate_indices:
                symbol = valid_symbols[idx]
                try:
                    instrument_id = InstrumentId.from_str(symbol)
                    instrument = self.cache.instrument(instrument_id)
                    current_qty = self.order_manager._get_net_position_qty(instrument_id)
                    
                    if current_qty == 0:
                        continue
                    
                    price_for_commission = instrument.make_price(prices[idx])
                    qty_for_commission = instrument.make_qty(abs(current_qty))
                    
                    commission = self.fee_model.get_commission(
                        Order_order=None,
                        Quantity_fill_qty=qty_for_commission,
                        Price_fill_px=price_for_commission,
                        Instrument_instrument=instrument
                    )
                    commissions[idx] = float(commission)
                except Exception as e:
                    # On error, set commission to infinity so we keep the position
                    commissions[idx] = np.inf
                    logger.warning(f"Error calculating commission for {symbol}: {e}. Keeping position.")
            
            # Vectorized decision: liquidate where expected_loss > commission
            should_liquidate = (expected_losses > commissions) & evaluate_mask
            defensive_weights[should_liquidate] = 0.0
            
            # Batch logging (single log message)
            if np.any(should_liquidate):
                liquidated_symbols = [valid_symbols[i] for i in np.where(should_liquidate)[0]]
                total_saved = np.sum(expected_losses[should_liquidate] - commissions[should_liquidate])
                logger.debug(
                    f"Liquidating {len(liquidated_symbols)} positions: {liquidated_symbols}. "
                    f"Total net benefit: ${total_saved:.2f}"
                )
            
            kept_mask = evaluate_mask & ~should_liquidate
            if np.any(kept_mask):
                kept_symbols = [valid_symbols[i] for i in np.where(kept_mask)[0]]
                logger.debug(
                    f"Keeping {len(kept_symbols)} losing positions (commissions exceed expected losses): {kept_symbols}"
                )
            
            # Map to full universe
            w_series = pd.Series(0.0, index=self.universe)
            w_series[valid_symbols] = defensive_weights
            
            logger.debug(
                f"Portfolio Loss Minimization: {np.sum(should_liquidate)} liquidated, "
                f"{np.sum(~should_liquidate & has_position_mask)} kept, remaining in cash"
            )
            
            return w_series.to_dict()


        # NOTE: Controls on cash flow enforced inside the optimizer _add_constraints():
        # sells + active balance < buys + commission + (buffer)

        # Call optimizer with horizon-scaled cov and risk-free rf
        w_opt = self.optimizer.optimize(
            er=pd.Series(valid_expected_returns, index=valid_symbols),
            cov=pd.DataFrame(cov_horizon, index=valid_symbols, columns=valid_symbols),
            rf=current_rf,
            benchmark_vol=self._get_benchmark_volatility(),
            allowed_weight_ranges=allowed_weight_ranges,
            current_weights=current_weights,
            prices=prices,
            nav=nav,
            cash_available=cash_available,
            selector_k = self.strategy_params["top_k"],
            target_volatility = self.target_volatility
        )

        # Clipping (if optimizer constraints failed) and log if clipping occurred
        w_clipped = np.clip(w_opt, allowed_weight_ranges[:, 0], allowed_weight_ranges[:, 1])  # max bounds

        # Account for commissions:
        w_final = self._adjust_weights_for_commissions(
            symbols = np.array(valid_symbols),
            expected_returns_array = valid_expected_returns,
            weights_array = w_clipped,
            nav = nav)

        # Map to universe
        w_series = pd.Series(0.0, index=self.universe)
        w_series[valid_symbols] = w_final  # Direct assignment using list indexing

        return w_series.to_dict()

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
            bar_type = freq2bartype(instrument_id = instrument_id, frequency=self.strategy_params["freq"])
            
            # Aggregate positions for each instrument (for HEDGING accounts)
            if instrument_id not in positions_by_instrument:
                positions_by_instrument[instrument_id] = 0.0
            positions_by_instrument[instrument_id] += float(position.signed_qty)
        
        # Calculate market value for net positions
        for instrument_id, net_quantity in positions_by_instrument.items():
            bar_type = freq2bartype(instrument_id = instrument_id, frequency=self.strategy_params["freq"])
            bars = self.cache.bars(bar_type)
            
            if bars:
                current_price = float(bars[0].close)
                position_value = net_quantity * current_price
                total_positions_value += position_value
        
        nav = cash_balance + total_positions_value - self.cash_buffer
        
        # Sanity check
        if nav <= 0 and cash_balance > 0:
            logger.warning(f"NAV calculation may be incorrect: cash={cash_balance:.2f}, positions={total_positions_value:.2f}")
            nav = cash_balance - self.cash_buffer  # Use cash as fallback
        
        logger.debug(f"NAV Calculation: Cash={cash_balance:.2f}, Positions={total_positions_value:.2f}, Total={nav:.2f}")

        return nav

    def _import_fee_model(self) -> FeeModel:

        name =  self.strategy_params["fee_model"]["name"]

        try:
            # Try to import from algos module
            fee_module = importlib.import_module(f"engine.fees.{self.strategy_params["fee_model"]["name"]}")
            FeeModelClass = getattr(fee_module, name, None)
            FeeModelConfig = getattr(fee_module, f"{name}Config", None)
            
            if FeeModelClass is None:
                FeeModelClass = getattr(fee_module, "Strategy", None)
            
            if FeeModelClass is None:
                raise ImportError(f"Could not find FeeModel class in engine.{self.strategy_params["fee_model"]["name"]}")
            
        except ImportError as e:
            logger.error(f"Failed to import fee model {self.strategy_params["fee_model"]["name"]}: {e}")
            raise

        FeeModelConfig.config = {k: v for k, v in self.strategy_params["fee_model"].items() if k != "name"}
        fee_model = FeeModelClass(FeeModelConfig)

        return fee_model
    
    def _adjust_weights_for_commissions(
        self, 
        symbols: np.ndarray,
        expected_returns_array: np.ndarray,
        weights_array: np.ndarray,
        nav: float
    ) -> np.ndarray:
        """
        Adjust portfolio weights to ensure sufficient cash for commissions.
        
        Decision logic:
        1. Calculate total expected profit from rebalancing entire portfolio
        2. Calculate total commissions for all trades
        3. If total_profit <= total_commissions: reject rebalance (return zeros)
        4. Otherwise: reserve commission cash by reducing least profitable positions
        5. Handles both long and short positions correctly
        
        Args:
            symbols: Array of stock symbols
            expected_returns_array: Expected returns for each stock (can be negative)
            weights_array: Target weights from optimizer (can be negative for shorts)
            nav: Current portfolio NAV
            
        Returns:
            Adjusted weights accounting for commissions, or zeros if unprofitable
        """
        if nav <= 0:
            logger.error("Invalid NAV for commission adjustment")
            return np.zeros(len(weights_array))
        
        adjusted_weights = weights_array.copy()
        
        total_commission_needed = 0.0
        total_expected_profit = 0.0
        net_trades_value = 0.0
        trade_info = []
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Calculate all trades, commissions, and expected profits
        # ═══════════════════════════════════════════════════════════════
        for idx, symbol in enumerate(symbols):
            target_weight = adjusted_weights[idx]
            
            instrument_id = InstrumentId.from_str(str(symbol))
            instrument = self.cache.instrument(instrument_id)
            
            # Get current position and price
            current_qty = self.order_manager._get_net_position_qty(instrument_id)
            current_price = self.order_manager._get_current_price(instrument_id)
            
            if current_price is None or current_price <= 0:
                adjusted_weights[idx] = 0.0
                continue
            
            # Calculate target quantity as float
            target_value = target_weight * nav
            target_qty_float = target_value / current_price
            
            # ─────────────────────────────────────────────────────────
            # REQUIREMENT 2: Enforce integer quantities
            # Skip if rounds to zero when non-zero weight intended
            # rounds to smallest positive int or to biggest negative int as a safe bound
            # ─────────────────────────────────────────────────────────
            target_qty = int(target_qty_float)
            
            if target_qty == 0 and abs(target_weight) > 1e-8:
                logger.debug(
                    f"Skipping {symbol}: target_qty={target_qty_float:.3f} rounds to zero "
                    f"(price={current_price:.2f}, target_weight={target_weight:.6f})"
                )
                adjusted_weights[idx] = 0.0
                continue
            
            # Recalculate weight based on actual integer quantity
            adjusted_weight = (target_qty * current_price) / nav if target_qty != 0 else 0.0
            adjusted_weights[idx] = adjusted_weight
            
            # Calculate trade quantity needed
            trade_qty = target_qty - current_qty
            
            # Calculate expected profit from holding target position
            # Works for both long and short:
            # - Long (target_qty > 0, expected_return > 0): profit > 0
            # - Short (target_qty < 0, expected_return < 0): profit > 0
            # - Unprofitable positions: profit < 0 (will be reduced first)
            position_expected_profit = target_qty * current_price * expected_returns_array[idx]
            
            if abs(trade_qty) < 1:  # No meaningful trade needed
                # Still track for overall profit calculation
                trade_info.append({
                    'idx': idx,
                    'symbol': symbol,
                    'weight': adjusted_weight,
                    'target_qty': target_qty,
                    'current_qty': current_qty,
                    'trade_qty': 0,
                    'current_price': current_price,
                    'expected_return': expected_returns_array[idx],
                    'position_profit': position_expected_profit,
                    'commission': 0.0,
                    'needs_trade': False
                })
                continue
            
            # Calculate commission for this trade
            expected_price = current_price * (1 + expected_returns_array[idx])
            # Use max of current and expected price for conservative commission estimate
            price_for_commission = instrument.make_price(max(abs(current_price), abs(expected_price)))
            qty_for_commission = instrument.make_qty(abs(trade_qty))
            
            commission = self.fee_model.get_commission(
                Order_order=None,
                Quantity_fill_qty=qty_for_commission,
                Price_fill_px=price_for_commission,
                Instrument_instrument=instrument
            )
            commission_value = float(commission)
            total_commission_needed += commission_value
            
            # Calculate expected profit from this specific trade
            # Formula: trade_qty * current_price * expected_return
            # Works for both long and short positions
            trade_expected_profit = trade_qty * current_price * expected_returns_array[idx]
            total_expected_profit += trade_expected_profit
            net_trades_value += trade_qty * current_price
            
            trade_info.append({
                'idx': idx,
                'symbol': symbol,
                'weight': adjusted_weight,
                'target_qty': target_qty,
                'current_qty': current_qty,
                'trade_qty': trade_qty,
                'current_price': current_price,
                'expected_return': expected_returns_array[idx],
                'trade_profit': trade_expected_profit,
                'position_profit': position_expected_profit,
                'commission': commission_value,
                'needs_trade': True
            })
        
        # ═══════════════════════════════════════════════════════════════
        # REQUIREMENT 1: Decide if portfolio-wide rebalancing is profitable
        # ═══════════════════════════════════════════════════════════════
        if abs(total_expected_profit) <= total_commission_needed:
            logger.warning(
                f"Rebalancing NOT profitable: expected profit ({total_expected_profit:.2f}) "
                f"<= commissions ({total_commission_needed:.2f}). "
                f"Keeping current allocation (returning zero weights)."
            )
            return np.zeros(len(weights_array))
        
        net_expected_profit = total_expected_profit - total_commission_needed
        logger.debug(
            f"Rebalancing IS convenient: expected profit ({total_expected_profit:.2f}) "
            f"> commissions ({total_commission_needed:.2f}). "
            f"Net profit: {net_expected_profit:.2f}"
        )
        
        if total_commission_needed <= 1e-8:
            return adjusted_weights
        
        if net_trades_value + total_commission_needed < nav:
            return adjusted_weights
        # ═══════════════════════════════════════════════════════════════
        # REQUIREMENT 3: Reserve cash for commissions
        # Reduce positions with lowest expected profit first
        # ═══════════════════════════════════════════════════════════════
        commission_pct = total_commission_needed / nav
        
        # Sort by position_profit ASCENDING (lowest/most negative first)
        # This correctly handles:
        # - Unprofitable shorts (negative qty × positive return = negative profit) → reduce first
        # - Unprofitable longs (positive qty × negative return = negative profit) → reduce first  
        # - Low-profit positions → reduce next
        # - High-profit positions → keep
        trade_info.sort(key=lambda x: x['position_profit'])
        
        remaining_commission_pct = commission_pct

        for trade in trade_info:
            if remaining_commission_pct <= 1e-8:
                break
            
            idx = trade['idx']
            current_weight = adjusted_weights[idx]
            
            if abs(current_weight) < 1e-8:
                continue
            
            # Calculate maximum possible reduction (full position)
            max_reduction = abs(current_weight)
            reduction = min(max_reduction, remaining_commission_pct)
            
            # Apply reduction (preserving sign direction)
            # For long (weight > 0): reduce means subtract → less buying
            # For short (weight < 0): reduce means add → less shorting
            if current_weight > 0:
                adjusted_weights[idx] = max(0.0, current_weight - reduction)
            else:
                adjusted_weights[idx] = min(0.0, current_weight + reduction)
            
            remaining_commission_pct -= reduction
            
            logger.debug(
                f"Reduced {trade['symbol']} (position_profit={trade['position_profit']:.2f}, "
                f"return={trade['expected_return']:.4f}): "
                f"weight {current_weight:.6f} → {adjusted_weights[idx]:.6f} "
                f"(freed {reduction:.6f} of NAV)"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Verify sufficient commission reserve
        # ═══════════════════════════════════════════════════════════════
        if remaining_commission_pct > 1e-6:
            shortfall = remaining_commission_pct * nav
            logger.error(
                f"FAILED to reserve full commission amount. Shortfall: ${shortfall:.2f}. "
                f"This should not happen. Rejecting rebalance."
            )
            return np.zeros(len(weights_array))
        
        logger.debug(f"Successfully reserved ${total_commission_needed:.2f} for commissions")
        
        return adjusted_weights
    