"""
Order Management Module for BacktestLongShortStrategy.
Handles all order-related events and execution logic.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from nautilus_trader.trading.strategy import Strategy
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
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity, Price


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
    
    def __init__(self, strategy Strategy, config: Dict[str, Any]):
        """
        Initialize the order manager.
        
        Args:
            strategy: Reference to the parent strategy
            config: Configuration dictionary
        """
        self.strategy = strategy
        self.log = strategy.log
        
        # Configuration
        self.timing_force = TimeInForce(config.get('execution', {}).get('timing_force', TimeInForce.DAY))
        self.twap_slices = config.get('execution', {}).get('twap', {}).get('slices', 4)
        self.twap_interval = config.get('execution', {}).get('twap', {}).get('interval_secs', 2.5)
        self.use_limit_orders = config.get('execution', {}).get('use_limit_orders', False)
        self.limit_order_offset_bps = config.get('execution', {}).get('limit_offset_bps', 5)
        self.max_order_retries = config.get('execution', {}).get('max_retries', 3)
        
        # Order tracking
        self.pending_orders: Dict[str, List[Any]] = defaultdict(list)
        self.filled_orders: Dict[str, List[OrderFilled]] = defaultdict(list)
        self.rejected_orders: Dict[str, List[Any]] = defaultdict(list)
        self.order_retries: Dict[str, int] = defaultdict(int)
        
        # TWAP execution tracking
        self.twap_schedules: Dict[str, Dict] = {}
        self.parent_orders: Dict[str, Dict] = {}
        
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
    # Order Event Handlers
    # =========================================================================
    
    def on_order_initialized(self, event: OrderInitialized) -> None:
        """Handle order initialization."""
        self.log.debug(f"Order initialized: {event.client_order_id}")
        self.order_metrics['total_submitted'] += 1
    
    def on_order_denied(self, event: OrderDenied) -> None:
        """Handle order denial (pre-submission rejection)."""
        self.log.warning(f"Order denied: {event.client_order_id} - {event.reason}")
        self._handle_order_failure(event)
    
    def on_order_emulated(self, event: OrderEmulated) -> None:
        """Handle emulated order event."""
        # TODO: emulation is used to set stop loss and drawdown. see https://nautilustrader.io/docs/latest/concepts/orders/#submitting-order-for-emulation
        self.log.debug(f"Order emulated: {event.client_order_id}")
    
    def on_order_released(self, event: OrderReleased) -> None:
        """Handle order release from emulation."""
        self.log.debug(f"Order released: {event.client_order_id}")
    
    def on_order_submitted(self, event: OrderSubmitted) -> None:
        """Handle order submission confirmation."""
        self.log.info(f"Order submitted: {event.client_order_id}")
        order_id = str(event.client_order_id)
        
        # Track submission time for latency metrics
        if order_id not in self.pending_orders:
            self.pending_orders[order_id] = []
        self.pending_orders[order_id].append({
            'submit_time': event.ts_event,
            'instrument_id': event.instrument_id
        })
    
    def on_order_rejected(self, event: OrderRejected) -> None:
        """Handle order rejection from venue."""
        self.log.warning(f"Order rejected: {event.client_order_id} - {event.reason}")
        self.order_metrics['total_rejected'] += 1
        self._handle_order_failure(event)
    
    def on_order_accepted(self, event: OrderAccepted) -> None:
        """Handle order acceptance by venue."""
        self.log.debug(f"Order accepted: {event.client_order_id}")
    
    def on_order_canceled(self, event: OrderCanceled) -> None:
        """Handle order cancellation confirmation."""
        self.log.info(f"Order canceled: {event.client_order_id}")
        self.order_metrics['total_canceled'] += 1
        
        order_id = str(event.client_order_id)
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
    
    def on_order_expired(self, event: OrderExpired) -> None:
        """Handle order expiration."""
        self.log.info(f"Order expired: {event.client_order_id}")
        self._handle_order_failure(event)
    
    def on_order_triggered(self, event: OrderTriggered) -> None:
        """Handle stop/limit order trigger."""
        self.log.info(f"Order triggered: {event.client_order_id}")
    
    def on_order_pending_update(self, event: OrderPendingUpdate) -> None:
        """Handle pending order update."""
        self.log.debug(f"Order pending update: {event.client_order_id}")
    
    def on_order_pending_cancel(self, event: OrderPendingCancel) -> None:
        """Handle pending order cancellation."""
        self.log.debug(f"Order pending cancel: {event.client_order_id}")
    
    def on_order_modify_rejected(self, event: OrderModifyRejected) -> None:
        """Handle order modification rejection."""
        self.log.warning(f"Order modify rejected: {event.client_order_id} - {event.reason}")
    
    def on_order_cancel_rejected(self, event: OrderCancelRejected) -> None:
        """Handle order cancellation rejection."""
        self.log.warning(f"Order cancel rejected: {event.client_order_id} - {event.reason}")
    
    def on_order_updated(self, event: OrderUpdated) -> None:
        """Handle order update confirmation."""
        self.log.debug(f"Order updated: {event.client_order_id}")
    
    def on_order_filled(self, event: OrderFilled) -> None:
        """
        Handle order fill event.
        Calculate slippage and update metrics.
        """
        self.log.info(
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
        
        # Check if this completes a TWAP execution
        self._check_twap_completion(event)
    
    def on_order_event(self, event: OrderEvent) -> None:
        """Handle any order event not specifically handled above."""
        self.log.debug(f"Order event WITHOUT HANDLER: {type(event).__name__} - {event.client_order_id}")
    
    # =========================================================================
    # Order Execution Methods
    # =========================================================================
    
    def submit_order(
        self,
        order: Order,
        order_type: OrderType,
        params: Dict(str, Any),            # order parameters
    ) -> Optional[Any]:
        """
        Submit an order to the market.
        
        Args:
            instrument_id: Instrument to trade
            quantity: Signed quantity (positive for buy, negative for sell)
            order_type: 
                        MARKET
                        LIMIT
                        STOP_MARKET
                        STOP_LIMIT
                        MARKET_TO_LIMIT
                        MARKET_IF_TOUCHED
                        LIMIT_IF_TOUCHED
                        TRAILING_STOP_MARKET
                        TRAILING_STOP_LIMIT
            price: Limit/stop price (if applicable)
            
        Returns:
            Submitted order or None if failed
        """
        
        # validate order params:
        try:
            if order_type not in ORDER_SPECS:
                raise ValueError(f"Unknown order_type: {order_type!r}. Valid: {sorted(ORDER_SPECS.keys())}")
            spec = ORDER_SPECS[order_type]
            keys = set(params.keys())
            missing = spec["required"] - keys
            allowed = spec["required"] | spec["optional"]
            extra = keys - allowed
            is_valid = (len(missing) == 0) and (len(extra) == 0) and set(params.keys()).issubset(spec["required"] | spec["optional"])
            
            # param-specific checks
            params["quantity"] = abs(params["quantity"])
            assert params["quantity"] >= 0
            assert params["order_side"] in [OrderSide.BUY, OrderSide.SELL]
            assert params["trailing_offset_type"] in [TrailingOffsetType.BASIS_POINTS, TrailingOffsetType.PRICE]
            assert params["trigger_type"] in [TriggerType.NO_TRIGGER, TriggerType.BID_ASK, TriggerType.LAST, TriggerType.LAST_OR_BID_ASK]
            

            if is_valid:            
                self.strategy.submit_order(order)
                return order
            else:
                raise Exception("Paramters are not a subset of allowed order parameters")
            
        except Exception as e:
            self.log.error(f"Failed to submit order: {e}")
            return None
    
    def execute_twap(
        self,
        instrument_id: InstrumentId,
        total_quantity: int,
        duration_seconds: Optional[float] = None
    ) -> None:
        """
        Execute a TWAP (Time-Weighted Average Price) order.
        
        Args:
            instrument_id: Instrument to trade
            total_quantity: Total signed quantity to execute
            duration_seconds: Total duration for TWAP execution
        """

        # TODO: should follow this implementation https://nautilustrader.io/docs/latest/concepts/execution/#twap-time-weighted-average-price
        # _submit_twap_slice is specifying order type. this should be passed directly
        if total_quantity == 0:
            return
        
        duration = duration_seconds or (self.twap_slices * self.twap_interval)
        slice_qty = total_quantity // self.twap_slices
        remainder = total_quantity % self.twap_slices
        
        # Create TWAP schedule
        schedule_id = f"TWAP_{instrument_id}_{self.strategy.clock.timestamp_ns()}"
        self.twap_schedules[schedule_id] = {
            'instrument_id': instrument_id,
            'remaining_slices': self.twap_slices,
            'slice_quantity': slice_qty,
            'remainder': remainder,
            'interval': self.twap_interval,
            'start_time': self.strategy.clock.timestamp_ns()
        }
        
        # Submit first slice immediately
        self._submit_twap_slice(schedule_id)
        
        # Schedule remaining slices
        for i in range(1, self.twap_slices):
            self.strategy.clock.set_timer(
                name=f"{schedule_id}_slice_{i}",
                interval=timedelta(seconds=self.twap_interval * i),
                callback=lambda: self._submit_twap_slice(schedule_id)
            )
    
    def cancel_all_orders(self, instrument_id: Optional[InstrumentId] = None) -> None:
        """
        Cancel all pending orders, optionally for a specific instrument.
        
        Args:
            instrument_id: Optional instrument filter
        """
        orders = self.strategy.cache.orders_open()
        
        for order in orders:
            if instrument_id is None or order.instrument_id == instrument_id:
                self.strategy.cancel_order(order)
                self.log.info(f"Canceling order: {order.client_order_id}")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _handle_order_failure(self, event: Any) -> None:
        """
        Handle order failure with retry logic.
        
        Args:
            event: Order failure event
        """
        order_id = str(event.client_order_id)
        
        # Check retry count
        self.order_retries[order_id] += 1
        
        if self.order_retries[order_id] < self.max_order_retries:
            self.log.info(
                f"Retrying order {order_id} "
                f"(attempt {self.order_retries[order_id] + 1}/{self.max_order_retries})"
            )
            # Re-submit logic would go here based on original order parameters
            # This would need to be stored when order is first created
        else:
            self.log.error(f"Max retries exceeded for order {order_id}")
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
    
    def _submit_twap_slice(self, schedule_id: str) -> None:
        """
        Submit a single TWAP slice.
        
        Args:
            schedule_id: TWAP schedule identifier
        """
        if schedule_id not in self.twap_schedules:
            return
        
        schedule = self.twap_schedules[schedule_id]
        
        # Determine slice quantity
        if schedule['remaining_slices'] == 1:
            # Last slice includes remainder
            quantity = schedule['slice_quantity'] + schedule['remainder']
        else:
            quantity = schedule['slice_quantity']
        
        # Submit the slice order
        self.submit_order(
            instrument_id=schedule['instrument_id'],
            quantity=quantity,
            order_type="MARKET"
        )
        
        # Update schedule
        schedule['remaining_slices'] -= 1
        
        if schedule['remaining_slices'] == 0:
            del self.twap_schedules[schedule_id]
            self.log.info(f"TWAP execution completed: {schedule_id}")
    
    def _check_twap_completion(self, fill_event: OrderFilled) -> None:
        """
        Check if an order fill completes a TWAP execution.
        
        Args:
            fill_event: Order fill event
        """
        # Check if this fill is part of a TWAP execution
        # and update parent order tracking if needed
        # TODO
        pass
    
    def get_order_metrics(self) -> Dict[str, Any]:
        """
        Get current order execution metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.order_metrics.copy()
        
        # Calculate fill rate
        if metrics['total_submitted'] > 0:
            metrics['fill_rate'] = metrics['total_filled'] / metrics['total_submitted']
            metrics['reject_rate'] = metrics['total_rejected'] / metrics['total_submitted']
        else:
            metrics['fill_rate'] = 0
            metrics['reject_rate'] = 0
        
        # Calculate average slippage
        if len(metrics['slippage_bps']) > 0:
            metrics['avg_slippage_bps'] = np.mean(metrics['slippage_bps'])
            metrics['max_slippage_bps'] = np.max(metrics['slippage_bps'])
        else:
            metrics['avg_slippage_bps'] = 0
            metrics['max_slippage_bps'] = 0
        
        return metrics