"""
Order Management Module for Backtest Strategies.
Handles all order-related events and execution logic.
"""
from __future__ import annotations
from re import I
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from nautilus_trader.model import ExecAlgorithmId
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.orders import Order, OrderList
from nautilus_trader.model.enums import ContingencyType, AggregationSource
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.events import (
    OrderAccepted, OrderCanceled, OrderCancelRejected,
    OrderDenied, OrderEmulated, OrderEvent, OrderExpired,
    OrderFilled, OrderInitialized, OrderModifyRejected,
    OrderPendingCancel, OrderPendingUpdate, OrderRejected,
    OrderReleased, OrderSubmitted, OrderTriggered, OrderUpdated
)
from nautilus_trader.model.orders import (
    Order,
    MarketOrder as MO, StopMarketOrder as SM, MarketToLimitOrder as MTL, MarketIfTouchedOrder as MIT, TrailingStopMarketOrder as TSM,
    LimitOrder as LO, StopLimitOrder as SL, LimitIfTouchedOrder as LIT, TrailingStopLimitOrder as TSL
)
from nautilus_trader.model.enums import OrderSide, TimeInForce, OrderStatus, OrderType
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.objects import Quantity, Price
from nautilus_trader.model import Position
from nautilus_trader.model.data import Bar, BarType
import logging

logger = logging.getLogger(__name__)

# =========================================================================
# Definition of ORDER paramters by order type
# =========================================================================
LIMIT_BASE_REQUIRED = {
    "trader_id", "strategy_id", "instrument_id", "client_order_id",
    "order_side", "quantity", "price", "init_id", "ts_init"
}
LIMIT_BASE_OPTIONAL = {
    "time_in_force", "expire_time_ns", "post_only", "reduce_only",
    "quote_quantity", "display_qty", "emulation_trigger", "trigger_instrument_id",
    "contingency_type", "order_list_id", "linked_order_ids", "parent_order_id",
    "exec_algorithm_id", "exec_algorithm_params", "exec_spawn_id", "tags",
    # activation_price is sometimes present for trailing orders
    "activation_price"
}
MARKET_BASE_REQUIRED = {
    "trader_id", "strategy_id", "instrument_id", "client_order_id",
    "order_side", "quantity", "init_id", "ts_init"
}
MARKET_BASE_OPTIONAL = {
    "time_in_force", "expire_time_ns", "reduce_only", "quote_quantity",
    "contingency_type", "order_list_id", "linked_order_ids", "parent_order_id",
    "exec_algorithm_id", "exec_algorithm_params", "exec_spawn_id", "tags",
    # display_qty can be used in MarketToLimit resulting limit
    "display_qty", "emulation_trigger", "trigger_instrument_id"
}

# --- Per-order-type incremental requirements ---
ORDER_SPECS = {
    # Market-based
    MO: {
        "required": set(MARKET_BASE_REQUIRED),
        "optional": set(MARKET_BASE_OPTIONAL),
    },
    SM: {
        "required": set(MARKET_BASE_REQUIRED) | {"trigger_price", "trigger_type"},
        "optional": set(MARKET_BASE_OPTIONAL),
    },
    MTL: {
        "required": set(MARKET_BASE_REQUIRED),
        "optional": set(MARKET_BASE_OPTIONAL) | {"expire_time_ns", "display_qty"},
    },
    MIT: {
        "required": set(MARKET_BASE_REQUIRED) | {"trigger_price", "trigger_type"},
        "optional": set(MARKET_BASE_OPTIONAL) | {"expire_time_ns", "emulation_trigger", "trigger_instrument_id"},
    },
    TSM: {
        # trigger_price may be nullable/optional per docs; trailing params are required
        "required": set(MARKET_BASE_REQUIRED) | {"trigger_type", "trailing_offset", "trailing_offset_type"},
        "optional": set(MARKET_BASE_OPTIONAL) | {"activation_price", "trigger_price"},
    },

    # Limit-based
    LO: {
        "required": set(LIMIT_BASE_REQUIRED),
        "optional": set(LIMIT_BASE_OPTIONAL),
    },
    SL: {
        "required": set(LIMIT_BASE_REQUIRED) | {"trigger_price", "trigger_type"},
        "optional": set(LIMIT_BASE_OPTIONAL),
    },
    LIT: {
        "required": set(LIMIT_BASE_REQUIRED) | {"trigger_price", "trigger_type"},
        "optional": set(LIMIT_BASE_OPTIONAL),
    },
    TSL: {
        # trailing_limit typically requires trailing offsets and limit_offset
        "required": set(LIMIT_BASE_REQUIRED) | {"trigger_type", "limit_offset", "trailing_offset", "trailing_offset_type"},
        "optional": set(LIMIT_BASE_OPTIONAL) | {"activation_price", "trigger_price", "price"},
    },
}
    



class OrderManager:
    """
    Manages order lifecycle, execution, and tracking for the strategy.
    """
    
    def __init__(self, strategy: Strategy, config: Dict[str, Any]):
        """
        Initialize the order manager.
        
        Args:
            strategy: Reference to the parent strategy
            config: Configuration dictionary
        """
        self.strategy = strategy

        # Configuration
        self.twap_slices = config.get('execution', {}).get('twap', {}).get('slices', 4)
        self.twap_interval = config.get('execution', {}).get('twap', {}).get('interval_secs', 2.5)
        self.adv_lookback = config.get('liquidity', {}).get('adv_lookback', 30)
        self.max_adv_pct = config.get('liquidity', {}).get('max_adv_pct', 0.05)
        self.timing_force = TimeInForce[config.get('execution', {}).get('timing_force', TimeInForce.GTC)]
        self.use_limit_orders = config.get('execution', {}).get('use_limit_orders', False)
        self.limit_order_offset_bps = config.get('execution', {}).get('limit_offset_bps', 5)
        self.max_order_retries = config.get('execution', {}).get('max_retries', 3)
        
        # Order tracking
        self.pending_orders: Dict[str, List[Any]] = defaultdict(list)
        self.filled_orders: Dict[str, List[OrderFilled]] = defaultdict(list)
        self.rejected_orders: Dict[str, List[Any]] = defaultdict(list)
        self.order_retries: Dict[str, int] = defaultdict(int)

        # Position tracking for order management
        self.target_positions: Dict[str, int] = {}
        self.current_positions: Dict[str, int] = {}
        
        
        # Performance metrics
        self.order_metrics = {
            'total_submitted': 0,
            'total_filled': 0,
            'total_rejected': 0,
            'total_canceled': 0,
            'avg_fill_time': 0,
            'slippage_bps': []
        }
    

    # =========================================================================
    # HIGH-LEVEL PORTFOLIO OPERATIONS (Called by Strategy)
    # =========================================================================

    def rebalance_portfolio(self, target_weights: Dict[str, float], universe: List[str]) -> None:
        """Rebalance using sequential execution: sells complete before buys."""
        nav = self.strategy._calculate_portfolio_nav()
        if nav <= 0:
            logger.error(f"Invalid NAV: {nav}")
            return
        
        sells = []
        buys = []
        
        for symbol in universe:
            target_weight = target_weights.get(symbol, 0.0)
            instrument_id = InstrumentId.from_str(symbol)
            
            net_qty = self._get_net_position_qty(instrument_id)
            price = self._get_current_price(instrument_id)
            if not price:
                continue
            
            target_qty = int((target_weight * nav) / price)
            order_qty = target_qty - net_qty
            
            if abs(order_qty) < 1:
                continue
            
            if order_qty < 0:  # Sell
                sell_order = self.strategy.order_factory.market(
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
                self.strategy.submit_order(sell_order)
            
            else:  # Buy   
                buy_order = self.strategy.order_factory.market(
                    instrument_id=instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_int(abs(int(order_qty))),
                    time_in_force=self.timing_force,
                    exec_algorithm_id=ExecAlgorithmId("TWAP"),
                    exec_algorithm_params={
                        "horizon_secs": self.twap_slices * self.twap_interval,
                        "interval_secs": self.twap_interval
                    },
                )
                buys.append(buy_order)
        
        # Submit buys as contingent order list if sells exist
        # TODO: current Nautilus implementation of buy orders linked to seels is incomplete.
        # the current implementation will: try submit buys asap and retry config.max_retries times.
        # in future consider using the following to ensure buys are executed after sells:
        # contingency_type=ContingencyType.OTO,
        # linked_order_ids=[s.client_order_id for s in sells]

        for buy_order in buys:
            self.strategy.submit_order(buy_order)

    
    def close_position(self, position: Position):
        """Close a specific position."""
        # Determine order side to close position
        if position.is_long:
            order_side = OrderSide.SELL
        else:
            order_side = OrderSide.BUY
        
        order =self.strategy.order_factory.market(
            instrument_id=position.instrument_id,
            order_side=order_side,
            quantity=Quantity.from_int(abs(int(position.quantity))),
            time_in_force=self.timing_force,
            exec_algorithm_id=ExecAlgorithmId("TWAP"),
            exec_algorithm_params={"horizon_secs": self.twap_slices * self.twap_interval , "interval_secs": self.twap_interval},
            reduce_only = True
        )
        
        # Submit order
        if order:
            self.strategy.submit_order(order)

        


    # =========================================================================
    # Order Event Handlers
    # =========================================================================
    
    def handle_order_event(self, event: OrderEvent) -> None:
        """
        Route order events to appropriate handlers.
        
        Args:
            event: The order event to handle
        """
        if isinstance(event, OrderInitialized):
            self.on_order_initialized(event)
        elif isinstance(event, OrderDenied):
            self.on_order_denied(event)
        elif isinstance(event, OrderSubmitted):
            self.on_order_submitted(event)
        elif isinstance(event, OrderAccepted):
            self.on_order_accepted(event)
        elif isinstance(event, OrderRejected):
            self.on_order_rejected(event)
        elif isinstance(event, OrderFilled):
            self.on_order_filled(event)
        elif isinstance(event, OrderCanceled):
            self.on_order_canceled(event)
        elif isinstance(event, OrderExpired):
            self.on_order_expired(event)
        else:
            logger.debug(f"Unhandled order event: {type(event).__name__}")

    def on_order_initialized(self, event: OrderInitialized) -> None:
        """Handle order initialization."""
        logger.debug(f"Order initialized: {event.client_order_id}")
        self.order_metrics['total_submitted'] += 1
    
    def on_order_denied(self, event: OrderDenied) -> None:
        """Handle order denial (pre-submission rejection)."""
        logger.warning(f"Order denied: {event.client_order_id} - {event.reason}")
        self._handle_order_failure(event)
    
    def on_order_emulated(self, event: OrderEmulated) -> None:
        """Handle emulated order event."""
        # TODO: emulation is used to set stop loss and drawdown. see https://nautilustrader.io/docs/latest/concepts/orders/#submitting-order-for-emulation
        logger.debug(f"Order emulated: {event.client_order_id}")
    
    def on_order_released(self, event: OrderReleased) -> None:
        """Handle order release from emulation."""
        logger.debug(f"Order released: {event.client_order_id}")
    
    def on_order_submitted(self, event: OrderSubmitted) -> None:
        """Handle order submission confirmation."""
        logger.info(f"Order submitted: {event.client_order_id}")
        order_id = str(event.client_order_id)
        
        # Track submission time for latency metrics
        if order_id not in self.pending_orders:
            self.pending_orders[order_id] = []
        self.pending_orders[order_id].append({
            'submit_time': event.ts_event,
            'instrument_id': event.instrument_id
        })
    
    def on_order_accepted(self, event: OrderAccepted) -> None:
        """Handle order acceptance by venue."""
        logger.debug(f"Order accepted: {event.client_order_id}")
        
    def on_order_rejected(self, event: OrderRejected) -> None:
        """Handle order rejection from venue."""
        logger.warning(f"Order rejected: {event.client_order_id} - {event.reason}")
        self.order_metrics['total_rejected'] += 1
        self._handle_order_failure(event)

    def on_order_canceled(self, event: OrderCanceled) -> None:
        """Handle order cancellation confirmation."""
        logger.info(f"Order canceled: {event.client_order_id}")
        self.order_metrics['total_canceled'] += 1
        
        order_id = str(event.client_order_id)
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
    
    def on_order_expired(self, event: OrderExpired) -> None:
        """Handle order expiration."""
        logger.info(f"Order expired: {event.client_order_id}")
        #self._handle_order_failure(event)
        pass
    
    def on_order_triggered(self, event: OrderTriggered) -> None:
        """Handle stop/limit order trigger."""
        logger.info(f"Order triggered: {event.client_order_id}")
    
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
        logger.info(
            f"Order filled: {event.client_order_id} - "
            f"Price: {event.last_px}, Qty: {event.last_qty}"
        )
        
        order_id = str(event.client_order_id)
        self.filled_orders[str(event.instrument_id)].append(event)
        self.order_metrics['total_filled'] += 1
        
        # Calculate fill time if we have submission record
        if order_id in self.pending_orders:
            submit_info = self.pending_orders[order_id][0]
            fill_time = (event.ts_event - submit_info['submit_time']) / 1e9  # Convert to seconds
            
            # Update average fill time
            n = self.order_metrics['total_filled']
            prev_avg = self.order_metrics['avg_fill_time']
            self.order_metrics['avg_fill_time'] = (prev_avg * (n-1) + fill_time) / n
            
            # Clean up pending order
            del self.pending_orders[order_id]
        
        # Calculate slippage if this was a market order
        self._calculate_slippage(event)
    
    def on_order_event(self, event: OrderEvent) -> None:
        """Handle any order event not specifically handled above."""
        logger.debug(f"Order event WITHOUT HANDLER: {type(event).__name__} - {event.client_order_id}")
    
    # =========================================================================
    # Order Execution Methods (Internal)
    # =========================================================================
    


    def _create_limit_order(
        self,
        instrument_id: InstrumentId,
        quantity: int,
        order_side: OrderSide,
        price: float
    ) -> LimitOrder:
        """Create a limit order."""
        return self.strategy.order_factory.limit(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=Quantity.from_int(quantity),
            price=Price.from_str(str(price)),
            time_in_force=self.timing_force,
            exec_algorithm_id=ExecAlgorithmId("TWAP"),
            exec_algorithm_params={"horizon_secs": self.twap_slices * self.twap_interval , "interval_secs": self.twap_interval},
        )
        
    
    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_net_position_qty(self, instrument_id: InstrumentId) -> float:
        """Get net position quantity for an instrument."""
        net_qty = 0.0
        for pos in self.strategy.cache.positions_open(instrument_id = instrument_id):
            net_qty += float(pos.signed_qty)

        # Pending orders (submitted but not filled yet)
        for order in self.strategy.cache.orders_open(instrument_id=instrument_id, venue = self.strategy.venue):
            remaining_qty = float(order.quantity - order.filled_qty)
            if order.order_type != OrderType.TRAILING_STOP_MARKET and order.order_type != OrderType.TRAILING_STOP_LIMIT:
                if order.side == OrderSide.BUY :
                    net_qty += remaining_qty
                else:  # SELL
                    net_qty -= remaining_qty
        return net_qty
    
    def _get_current_price(self, instrument_id: InstrumentId) -> Optional[float]:
        """
        Get current price from latest trade tick (real-time) or last bar (fallback).
        """
        # Try ticks first (most recent price)
        ticks = self.strategy.cache.trade_ticks(instrument_id)
        if ticks and len(ticks) > 0:
            return float(ticks[0].price)  # Index 0 is most recent
        
        # Fallback to bars if no ticks available
        bar_type = BarType(instrument_id=instrument_id, bar_spec=self.strategy.bar_spec, aggregation_source = AggregationSource.INTERNAL)
        bars = self.strategy.cache.bars(bar_type)
        if bars and len(bars) > 0:
            return float(bars[0].close)
        
        logger.warning(f"No price data available for {instrument_id}")
        return None
    
    def _handle_order_failure(self, event: Any) -> None:
        """
        Handle order failure with retry logic.
        
        Args:
            event: Order failure event
        """
        order_id = str(event.client_order_id)

        # Retrieve original order from cache
        original_order = self.strategy.cache.order(event.client_order_id)
        if not original_order:
            logger.error(f"Cannot retry {order_id}: order not found in cache")
            return
        
        # Check retry count
        self.order_retries[order_id] += 1
        
        if self.order_retries[order_id] < self.max_order_retries:
            
            # Resubmit same order
            if original_order.order_type == OrderType.MARKET:
                retry_order = self.strategy.order_factory.market(
                    instrument_id=original_order.instrument_id,
                    order_side=original_order.side,
                    quantity=original_order.quantity,
                    time_in_force=original_order.time_in_force,
                    exec_algorithm_id=original_order.exec_algorithm_id,
                    exec_algorithm_params=original_order.exec_algorithm_params,
                    reduce_only=original_order.is_reduce_only,
                )
            elif original_order.order_type == OrderType.LIMIT:
                retry_order = self.strategy.order_factory.limit(
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
            self.strategy.submit_order(retry_order)
            
        else:
            logger.error(f"Max retries exceeded for order {order_id}")
            self.rejected_orders[str(event.instrument_id)].append(event)
            
        # Clean up
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
        if order_id in self.order_retries:
            del self.order_retries[order_id]
    
    def _calculate_slippage(self, fill_event: OrderFilled) -> None:
        """
        Calculate and record slippage for filled orders.
        
        Args:
            fill_event: Order fill event
        """
        # TODO:
        # This would need reference price from when order was submitted
        # For now, we'll skip detailed slippage calculation
        # In production, you'd compare fill price to mid-price at submission
        pass
    