"""
Base class for all Sapiens trading strategies.
"""

from abc import ABC
from copy import Error
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import nautilus_trader
import pandas as pd
import logging
import torch
import importlib
import pandas_market_calendars as market_calendars
from decimal import Decimal
from collections import defaultdict

from models.utils import freq2bartype, freq2pdoffset
from models.SapiensModel import SapiensModel

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity, Money, Currency
from nautilus_trader.model.currencies import USD,EUR
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.position import Position
from nautilus_trader.model.enums import ContingencyType, AggregationSource, OrderSide, TimeInForce, OrderStatus, OrderType, TriggerType, TrailingOffsetType, PositionSide
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.events import (
    OrderAccepted, OrderCanceled, OrderCancelRejected,
    OrderDenied, OrderEmulated, OrderEvent, OrderExpired,
    OrderFilled, OrderInitialized, OrderModifyRejected,
    OrderPendingCancel, OrderPendingUpdate, OrderRejected,
    OrderReleased, OrderSubmitted, OrderTriggered, OrderUpdated,
    PositionChanged, PositionClosed, PositionEvent, PositionOpened
)
from nautilus_trader.core.nautilus_pyo3 import CurrencyType
from nautilus_trader.common.component import TimeEvent
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, BarType, InstrumentStatus, InstrumentClose, TradeTick
from nautilus_trader.model.orders import (
    Order, MarketOrder, StopMarketOrder , MarketToLimitOrder, MarketIfTouchedOrder, TrailingStopMarketOrder,
    LimitOrder, StopLimitOrder, LimitIfTouchedOrder , TrailingStopLimitOrder
)
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.backtest.models import FillModel, FeeModel
from nautilus_trader.model import ExecAlgorithmId, Position
import logging
logger = logging.getLogger(__name__)



class SapiensStrategyConfig(StrategyConfig, frozen=True):
    """Base config for Sapiens strategies."""
    config: Dict[str, Any] = field(default_factory=dict)


class SapiensStrategy(Strategy, ABC):
    """
    Base class for Sapiens strategies.
    Expects config with MODEL and STRATEGY sections.
    """
    
    def __init__(self, config: SapiensStrategyConfig):
        super().__init__(config)

        # Extract nested config
        cfg = config.config
        
        # Separate MODEL and STRATEGY params
        self.model_params = cfg.get("MODEL", {})
        self.strategy_params = cfg.get("STRATEGY", {})

        # Setup Paths
        self.strategy_name = self.strategy_params.get("strategy_name", self.__class__.__name__)
        strategy_dir = Path(f"strategies/{self.strategy_name}")
        strategy_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(self.model_params["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model_params["model_dir"] = Path(self.model_params["model_dir"])

        # NOTE: safe handling of variable. could be removed in future by setting precision. now used for computing 0.005 USD/share commissions
        currency =self.strategy_params["currency"]
        if currency == "USD":
            self.strategy_params["currency"] = Currency(code='USD', precision=3, iso4217=840, name='United States dollar', currency_type = CurrencyType.FIAT ) #
        elif currency == "EUR":
            self.strategy_params["currency"] = Currency(code='EUR', precision=3, iso4217=978, name='Euro', currency_type=CurrencyType.FIAT)
        else: # currency is already in nautilus format
            raise Error("Currency not implemented correctly") 


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
        self.last_retrain_time: Optional[pd.Timestamp] = None


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
        self.optimizer_lookback = pd.Timedelta(self.strategy_params["optimizer_lookback"]) // pd.Timedelta(self.strategy_params["freq"])
        self.min_bars_required = self.model_params["window_len"]
        self.active_mask: Optional[torch.Tensor] = None  # (I,)
        
        # Risk-free instrument
        self.risk_free_ticker = self.strategy_params.get("risk_free_ticker")
        
        # Commissions Fee Model
        self.fee_model = self._import_fee_model()

        # Cash buffer to always keep for rounding errors
        self.cash_buffer = self.strategy_params["cash_buffer"]

        # Risk Management
        self.max_w_abs = self.strategy_params["risk"]["max_weight_abs"]
        self.drawdown_max = self.strategy_params["risk"]["drawdown_max"]
        self.trailing_stop_max = self.strategy_params["risk"]["trailing_stop_max"]
        # Portfolio optimizer
        # TODO: add risk_aversion config parameter for MaxQuadraticUtilityOptimizer
        # TODO: make sure to pass proper params to create_optimizer depending on the optimizer all __init__ needed by any optimizer


        # Prediction Logging
        self._prediction_log: List[Dict[str, Any]] = []

        # Max balance reached to enforce drawdown risk
        self.max_portfolio_nav = 0

        # Order Management
        self.order_retries: Dict[str, int] = defaultdict(int)
        self.timing_force = TimeInForce[cfg.get('execution', {}).get('timing_force', "GTC")]
        self.max_order_retries = cfg.get('execution', {}).get('max_retries', 3)

        # Execution Parameters
        self.adv_lookback = pd.Timedelta(self.strategy_params["liquidity"]["adv_lookback"])
        assert self.adv_lookback.days >= 1, "Use more than 1 day for the ADV lookback"
        self.max_adv_pct = self.strategy_params["liquidity"]["max_adv_pct"]
        self.twap_slices = self.strategy_params["execution"]["twap"]["slices"]
        self.twap_interval = self.strategy_params["execution"]["twap"]["interval_secs"]

        # Final liquidation flag
        self.final_liquidation_happend = False

        # On-retrain is postponed flag (when retrain falls within market hours)
        self.postponed_retrain_scheduled = False

    # =========================================================================
    # On_* ORDER EVENT HANDLERS
    # =========================================================================
    
    def handle_order_event(self, event: OrderEvent) -> None:
        pass

    def on_order_initialized(self, event: OrderInitialized) -> None:
        """Handle order initialization."""
        logger.debug(f"Order initialized: {event.client_order_id}")
    
    def on_order_denied(self, event: OrderDenied) -> None:
        """Handle order denial (pre-submission rejection)."""
        logger.warning(f"Order denied: {event.client_order_id} - {event.reason}")
    
    def on_order_emulated(self, event: OrderEmulated) -> None:
        """Handle emulated order event."""
        # TODO: emulation is used to set stop loss and drawdown. see https://nautilustrader.io/docs/latest/concepts/orders/#submitting-order-for-emulation
        logger.debug(f"Order emulated: {event.client_order_id}")
    
    def on_order_released(self, event: OrderReleased) -> None:
        """Handle order release from emulation."""
        logger.debug(f"Order released: {event.client_order_id}")
    
    def on_order_submitted(self, event: OrderSubmitted) -> None:
        """Handle order submission confirmation."""
        logger.debug(f"Order submitted: {event.client_order_id}")
    
    def on_order_accepted(self, event: OrderAccepted) -> None:
        """Handle order acceptance by venue."""
        logger.debug(f"Order accepted: {event.client_order_id}")
        
    def on_order_rejected(self, event: OrderRejected) -> None:
        """Handle order rejection from venue."""
        logger.warning(f"Order rejected: {event.client_order_id} - {event.reason}")
        order_id = str(event.client_order_id)

        # Retrieve original order from cache
        original_order = self.cache.order(event.client_order_id)
        if not original_order:
            logger.error(f"Cannot retry {order_id}: order not found in cache")
            return
        
        # Check retry count
        self.order_retries[order_id] += 1
        
        if self.order_retries[order_id] < self.max_order_retries:
            
            # Resubmit same order
            if original_order.order_type == OrderType.MARKET:
                retry_order = self.order_factory.market(
                    instrument_id=original_order.instrument_id,
                    order_side=original_order.side,
                    quantity=original_order.quantity,
                    time_in_force=original_order.time_in_force,
                    exec_algorithm_id=original_order.exec_algorithm_id,
                    exec_algorithm_params=original_order.exec_algorithm_params,
                    reduce_only=original_order.is_reduce_only,
                )
            elif original_order.order_type == OrderType.LIMIT:
                retry_order = self.order_factory.limit(
                    instrument_id=original_order.instrument_id,
                    order_side=original_order.side,
                    quantity=original_order.quantity,
                    price=original_order.price,
                    time_in_force=original_order.time_in_force,
                    exec_algorithm_id=original_order.exec_algorithm_id,
                    exec_algorithm_params=original_order.exec_algorithm_params,
                    reduce_only=original_order.is_reduce_only,
                )
            else:
                logger.warning(f"Order type {original_order.order_type} not supported for retry")
                return
            
            logger.info(f"Resubmission attempt for order {order_id} as {retry_order.client_order_id} ({self.order_retries[order_id] + 1}/{self.max_order_retries})")
            self.order_retries[str(retry_order.client_order_id)] = self.order_retries[order_id]
            self.submit_order(retry_order)
            
        else:
            logger.error(f"Max retries exceeded for order {order_id}")
            
        # Clean up
        if order_id in self.order_retries:
            del self.order_retries[order_id]

    def on_order_canceled(self, event: OrderCanceled) -> None:
        """Handle order cancellation confirmation."""
        logger.warning(f"Order canceled: {event.client_order_id}")
    
    def on_order_expired(self, event: OrderExpired) -> None:
        """Handle order expiration."""
        logger.warning(f"Order expired: {event.client_order_id}")
        pass
    
    def on_order_triggered(self, event: OrderTriggered) -> None:
        """Handle stop/limit order trigger."""
        logger.debug(f"Order triggered: {event.client_order_id}")
    
    def on_order_pending_update(self, event: OrderPendingUpdate) -> None:
        """Handle pending order update."""
        logger.debug(f"Order pending update: {event.client_order_id}")
    
    def on_order_pending_cancel(self, event: OrderPendingCancel) -> None:
        """Handle pending order cancellation."""
        logger.debug(f"Order pending cancel: {event.client_order_id}")
    
    def on_order_modify_rejected(self, event: OrderModifyRejected) -> None:
        """Handle order modification rejection."""
        logger.warning(f"Order modify rejected: {event.client_order_id} - {event.reason}")
    
    def on_order_cancel_rejected(self, event: OrderCancelRejected) -> None:
        """Handle order cancellation rejection."""
        logger.warning(f"Order cancel rejected: {event.client_order_id} - {event.reason}")
    
    def on_order_updated(self, event: OrderUpdated) -> None:
        """Handle order update confirmation."""
        logger.debug(f"Order updated: {event.client_order_id}")
    
    def on_order_filled(self, event: OrderFilled) -> None:
        """
        Handle order fill event.
        Calculate slippage and update metrics.
        """
        logger.debug(
            f"Order filled: {event.client_order_id} - "
            f"Price: {event.last_px}, Qty: {event.last_qty}"
        )
        # Update max reached NAV
        current_nav = self._calculate_portfolio_nav()
        if  current_nav > self.max_portfolio_nav:
            self.max_portfolio_nav = current_nav
    
    def on_order_event(self, event: OrderEvent) -> None:
        """Handle any order event not specifically handled above."""
        #logger.debug(f"Order event WITHOUT HANDLER: {type(event).__name__} - {event.client_order_id}")
        pass


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

    # NOTE: Unused ATM
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
            catalog = ParquetDataCatalog(path=str(catalog_path))
            
            # Query trade ticks from catalog for all instruments
            start_time = self.data_load_start
            end_time = self.train_end
            
            logger.info(f"Loading ticks from {start_time} to {end_time} for {len(self.cache.instrument_ids(self.venue))} instruments")
            
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
                        logger.info(f"Added {len(bars)} bars for {iid}")
                    
                except Exception as e:
                    logger.error(f"Error processing {iid}: {e}", exc_info=True)
                    continue
            
            logger.info(f"Total: Added {total_bars_added} bars across {len(self.cache.instrument_ids(self.venue))} instruments")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}", exc_info=True)
    def on_data(self, data: Data) -> None:  # Custom data passed to this handler
        assert False
        return
    def on_signal(self, signal) -> None:  # Custom signals passed to this handler
        return
    


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

        # modify last trailing order to update the order quantity
        for order in self.cache.orders(instrument_id = instrument_id, side = OrderSide.SELL):
            if order.order_type == OrderType.TRAILING_STOP_MARKET:
                self.modify_order(order = order, quantity = event.quantity )
                break
        """
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
        """

    def on_position_closed(self, event: PositionClosed) -> None:
        
        # Cancel all risk orders when position is closed ( mainly for TRAILING STOP MARKET)
        instrument_id = event.instrument_id
        self.cancel_all_orders( instrument_id = instrument_id )

    def on_position_event(self, event: PositionEvent) -> None:  # All position event messages are eventually passed to this handler
        pass
    


    # ================================================================= #
    # On_* EVENT ACTIONS 
    # ================================================================= #

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


    # ================================================================= #
    # ACCOUNT 
    # ================================================================= #

    # TODO: implement 



    # ═══════════════════════════════════════════════════════════════════════
    # PORTFOLIO MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def rebalance_portfolio(self, target_weights: Dict[InstrumentId, float]) -> None:
        """Rebalance using sequential execution: sells complete before buys."""
        nav = self._calculate_portfolio_nav()
        if nav <= 0:
            logger.error(f"Invalid NAV: {nav}")
            return
        
        sells = []
        buys = {}
        
        for instrument_id in self.cache.instrument_ids(self.venue):
            target_weight = target_weights.get(instrument_id, 0.0)
            
            try:
                pos = self.cache.positions(instrument_id = instrument_id, venue = self.venue, strategy_id = self.id, side=PositionSide.LONG)
                if not pos:
                    net_qty = 0.0
                else: 
                    net_qty = pos[0].quantity
            except IndexError:
                net_qty = 0

            try:
                price = float(self.cache.trade_tick(instrument_id).price)
            except Error:
                bar_type = freq2bartype(instrument_id, self.strategy_params["freq"])
                price = float(self.cache.bar(bar_type).close)
            
            target_qty = int((target_weight * nav) / price)
            order_qty = target_qty - net_qty
            
            if abs(order_qty) < 1:
                continue
            
            if order_qty < 0:  # Sell
                sell_order = self.order_factory.market(
                    instrument_id=instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=Quantity.from_int(abs(int(order_qty))),
                    time_in_force=self.timing_force,
                    exec_algorithm_id=ExecAlgorithmId("TWAP"),
                    exec_algorithm_params={
                        "horizon_secs": self.twap_slices * self.twap_interval,
                        "interval_secs": self.twap_interval
                    },
                    reduce_only=True,)
                sells.append(sell_order)

                # Submit sells first (no position_id needed for NETTING)
                self.submit_order(sell_order)
            
            else:  # Create Buy but do not submit now   
                buys[instrument_id] = abs(int(order_qty))

        for bo_id, qty in buys.items():
            # Submit buys directly if no sells were needed to free cash
            if not sells:
                buy_order = self.order_factory.market(
                    instrument_id=bo_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_int(qty),
                    time_in_force=self.timing_force,
                    exec_algorithm_id=ExecAlgorithmId("TWAP"),
                    exec_algorithm_params={
                        "horizon_secs": self.twap_slices * self.twap_interval,
                        "interval_secs": self.twap_interval
                    },
                )
            
            else:
                client_order_id = self.order_factory.generate_client_order_id()
                buy_order = MarketOrder(
                    trader_id = self.order_factory.trader_id,
                    strategy_id = self.order_factory.strategy_id,
                    instrument_id=bo_id,
                    client_order_id = client_order_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_int(qty),
                    init_id=UUID4(),
                    ts_init=self.clock.timestamp_ns(),
                    time_in_force=self.timing_force,
                    reduce_only = False,
                    contingency_type = ContingencyType.OTO,
                    linked_order_ids = [so.client_order_id for so in sells],
                    exec_algorithm_id=ExecAlgorithmId("TWAP"),
                    exec_algorithm_params={
                        "horizon_secs": self.twap_slices * self.twap_interval,
                        "interval_secs": self.twap_interval
                    },
                    exec_spawn_id = client_order_id,
                )
            self.submit_order(buy_order)
            
        return

    # ═══════════════════════════════════════════════════════════════════════
    # Prediction Logging Methods
    # ═══════════════════════════════════════════════════════════════════════
    def log_predictions(
        self,
        timestamp: pd.Timestamp,
        predictions: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> None:
        """
        Log model predictions and update previous predictions with actual values.
        
        At each update step:
        1. Update the LAST prediction entry for each ticker with current price as "actual"
        2. Add NEW prediction entry with current prediction (actual=None, to be filled next time)
        
        Args:
            timestamp: Time of prediction
            predictions: Dict mapping ticker -> predicted value (model output)
            current_prices: Dict mapping ticker -> current close price
        """
        # Step 1: Update only the LATEST prediction for each ticker with current price as actual
        # Track which tickers we've already updated
        updated_tickers = set()
        for entry in reversed(self._prediction_log):
            ticker = entry['ticker']
            if ticker in updated_tickers:
                continue  # Already updated the latest for this ticker
            if entry['actual'] is None and ticker in current_prices:
                entry['actual'] = current_prices[ticker]
                updated_tickers.add(ticker)
        
        # Step 2: Add new prediction entries (actual will be filled at next update)
        for ticker, pred_value in predictions.items():
            log_entry = {
                'timestamp': timestamp,
                'ticker': ticker,
                'predicted': pred_value,
                'actual': None  # Will be filled at next update with future price
            }
            self._prediction_log.append(log_entry)
    

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════
    def _initialize_model(self) -> SapiensModel:
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
        
    def _cache_to_dict(self, window: Optional[int], instrument_ids: Optional[List[InstrumentId]] = None) -> Dict[str, pd.DataFrame]:
        """
        Convert cache data within the specified window to dictionary format expected by model.
        Efficient implementation using nautilus trader cache's native methods.
        """
        data_dict = {}

        # TODO: this has to be addressed and fixed later on
        iids = instrument_ids if instrument_ids else [iid for iid in self.cache.instrument_ids(self.venue) if iid != InstrumentId.from_str(self.risk_free_ticker)]

        if not window:
            window = self.strategy_params.get("engine", {}).get("cache", {}).get("bar_capacity", 4096)
        
        for iid in iids:
            if iid.value == self.risk_free_ticker:
                continue

            bar_type = freq2bartype(instrument_id = iid, frequency=self.strategy_params["freq"])
            
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
    
    def _get_benchmark_volatility(self) -> Optional[float]:
        """Calculate bechmark Standard Deviation from cached bars."""
        benchmark_ticker = self.strategy_params["benchmark_ticker"]
        if not benchmark_ticker:
            return None
        
        try:
            benchmark_instrument_id = InstrumentId.from_str(benchmark_ticker)
            bar_type = freq2bartype(instrument_id = benchmark_instrument_id, frequency=self.strategy_params["freq"])
            benchmark_bars = self.cache.bars(bar_type)
            
            if not benchmark_bars or len(benchmark_bars) < 2:
                return None
            
            lookback_periods = int(self.optimizer_lookback)
            bars_to_use = benchmark_bars[:min(lookback_periods, len(benchmark_bars))]
            
            closes = [float(b.close) for b in reversed(bars_to_use)]
            returns = pd.Series(closes).pct_change().dropna()
            
            if len(returns) < 2:
                return None
            
            return float(returns.std())
            
        except Exception as e:
            logger.warning(f"Could not calculate benchmark volatility: {e}")
            return None
    
    def _calculate_portfolio_nav(self) -> float:
        """Calculate total portfolio value: cash + market value of all positions."""
        # Get cash balance
        account = self.portfolio.account(self.venue)
        if account:
            cash_balance = float(account.balance_free(self.strategy_params["currency"]))
        else:
            cash_balance = float(self.strategy_params["initial_cash"])
        
        # Add market value of all open positions (handles multiple positions per instrument)
        total_notional = 0.0
        
        for position in self.cache.positions_open(venue = self.venue):
            instrument_id = position.instrument_id
            #bar_type = freq2bartype(instrument_id = instrument_id, frequency=self.strategy_params["freq"])
            last_price = self.cache.trade_tick(instrument_id).price
            total_notional += float(position.notional_value(last_price))

        nav = cash_balance + total_notional - self.cash_buffer
        
        # Sanity check
        if nav <= 0 and cash_balance > 0:
            logger.warning(f"NAV calculation may be incorrect: cash={cash_balance:.2f}, positions={total_positions_value:.2f}")
            nav = cash_balance - self.cash_buffer  # Use cash as fallback
        
        logger.debug(f"NAV Calculation: Cash={cash_balance:.2f}, Positions={total_notional:.2f}, Total={nav:.2f}")

        return nav