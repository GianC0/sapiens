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
from typing import Any
from datetime import datetime
from pathlib import Path
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
from nautilus_trader.common.component import TimeEvent, Logger
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.position import Position
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.adapters.databento import DATABENTO_CLIENT_ID

logger = logging.getLogger(__name__)
# ----- project imports -----------------------------------------------------

from engine.OptimizerFactory import create_optimizer
from models.utils import freq2bartype, freq2pdoffset
from strategies.SapiensStrategy import SapiensStrategy, SapiensStrategyConfig


# ========================================================================== #
# Strategy
# ========================================================================== #
class TopKStrategy(SapiensStrategy):
    """
    Long TopK equity strategy, model-agnostic & frequency-agnostic.
    """


    # ------------------------------------------------------------------ #
    def __init__(self, config: SapiensStrategyConfig):  
        super().__init__(config)  

        # Market calendar and Venue setup
        self.venue = Venue(self.strategy_params["venue_name"])
        self.calendar = market_calendars.get_calendar(self.strategy_params["calendar"])
        
        # Strategy-specific parameters
        #self.can_short = self.strategy_params["oms_type"] == "HEDGING"
        self.weight_bounds = (0, 1)  # long-only positions
        optimizer_name = self.strategy_params.get("optimizer_name", "max_sharpe")
        self.optimizer = create_optimizer(
            name = optimizer_name, 
            max_adv_pct = self.max_adv_pct, 
            weight_bounds = self.weight_bounds, 
            target_volatility = self.strategy_params["risk"].get("target_volatility", 0.05)
            )
        self.top_k = self.strategy_params["top_k"]  # Portfolio size
        # Validate top_k feasibility: top_k assets need to cover full portfolio
        max_theoretical_coverage = self.top_k * self.max_w_abs
        assert max_theoretical_coverage >= 1.0, f"Configuration error: top_k ({self.top_k}) × max_weight_abs ({self.max_w_abs}) = {max_theoretical_coverage:.2f} < 1.0 "


    # ================================================================= #
    # TOP-K START AND TIMER EVENT ACTIONS
    # ================================================================= #
    def on_start(self): 
        """Initialize strategy."""
        self.max_portfolio_nav = self._calculate_portfolio_nav()

        #TODO:fix
        #self.on_historical_data(data=None)

        # Subscribe to bars for selected universe
        for iid in self.cache.instrument_ids(self.venue):
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

        # Register to risk free ticker updates and tick data
        rf_iid = InstrumentId.from_str(self.risk_free_ticker)
        bar_type = freq2bartype(instrument_id = rf_iid, frequency=self.strategy_params["freq"])
        self.request_trade_ticks(instrument_id = rf_iid, start=self.train_start.to_pydatetime(),client_id=DATABENTO_CLIENT_ID)
        self.subscribe_bars(bar_type, client_id=DATABENTO_CLIENT_ID)
        self.subscribe_trade_ticks(rf_iid, client_id=DATABENTO_CLIENT_ID)

        # Build and initialize model
        self.model = self._initialize_model()

        # Set initial update time to avoid immediate firing
        # it does not account for risk free
        self.active_mask = torch.ones(len(self.cache.instruments()) - 1, dtype=torch.bool)
        
        self._last_update_time = pd.Timestamp(self.clock.utc_now())
        self.last_retrain_time = pd.Timestamp(self.clock.utc_now())
        
        # Set the regular trading timers
        self.clock.set_timer(
            name="update_timer",
            interval=pd.Timedelta(freq2pdoffset(self.strategy_params["freq"])),
            callback=self.on_update,
        )
        
        # TODO: improve it to generalize
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

    def on_update(self, event: TimeEvent):
        if event.name != "update_timer":
            return
        if self.clock.utc_now() <= self.valid_start:
            logger.debug("This update step is used to load initial data.")
            return
        if self.final_liquidation_happend:
            return

        # Overall portfolio drawdown condition
        if self._calculate_portfolio_nav() < self.max_portfolio_nav * self.drawdown_max:
            self.on_final_liquidation(event)
            return

        event_time = pd.to_datetime(event.ts_event, unit="ns", utc=True)
        if event_time <= self._last_update_time:
            logger.debug(f"Event time {event_time} not after last update {self._last_update_time}")
            return

        now = pd.Timestamp(self.clock.utc_now()) #.tz_convert(self.calendar.tz)
        freq = self.strategy_params["freq"]
        #d_r = self.calendar.schedule(start_date=str(now - freq2pdoffset(freq)), end_date=str(now + freq2pdoffset(freq)))
        
        # do not update if "now" falls outside market trading hours
        schedule = self.calendar.schedule(start_date=str(now), end_date=str(now))
        if schedule.empty:
            logger.debug(f"Non-trading day at {now}")
            return
        assert schedule.size == 2
        if now < pd.Timestamp(schedule.market_open.iloc[0]) or now > pd.Timestamp(schedule.market_close.iloc[0]):
            logger.debug(f"Non-trading hour at {now}")
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

        # ensure the model has enough data for prediction
        #start_date = now - freq2pdoffset(self.strategy_params["freq"]) * ( self.min_bars_required) 
        #days_range = self.calendar.schedule(start_date=start_date, end_date=now)
        #timestamps = market_calendars.date_range(days_range, frequency=self.strategy_params["freq"]).normalize()
        #if len(timestamps) < lookback_periods:
        #    return

        preds = self.model.predict(data=data_dict, indexes = self.min_bars_required, active_mask=self.active_mask) #  preds: Dict[str, float]

        assert preds is not None, "Model predictions are empty"
        assert len(preds) > 0

        # Log predictions
        try:
            current_prices = self._get_current_prices(list(preds.keys()))
            self.log_predictions(
                timestamp=now,
                predictions=preds,
                current_prices=current_prices
            )
        except Exception as e:
            logger.debug(f"Could not log predictions: {e}")

        # Compute new portfolio target weights for predicted instruments.
        weights = self._compute_target_weights(preds)

        # Check if we have valid weights
        if not weights or len(weights) == 0:
            logger.warning(f"Time: ### {self.clock.utc_now()} ###. No valid weights computed, skipping rebalance")
            self._last_update_time = now
            return

        # Portolio Managerment through order manager
        self.rebalance_portfolio(weights)

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
        
        # Update the postponed_retrain flag for next retrain
        if event.name == "postponed_retrain":
            self.postponed_retrain_scheduled = False

        now = pd.Timestamp(self.clock.utc_now(),)
        
        # Skip if too soon since last retrain
        #if self.last_retrain_time and now < self.last_retrain_time + self.retrain_offset:
        #    return
        
        # do not retrain if "now" falls within market trading hours
        schedule = self.calendar.schedule(start_date=str(now), end_date=str(now))
        if not schedule.empty:
            market_close = pd.Timestamp(schedule.market_close.iloc[0])
            if now <= market_close:
                if not self.postponed_retrain_scheduled:
                    logger.debug(f"Market still open at {now}, postponing retrain to {market_close}")
                    self.clock.set_time_alert(
                        name="postponed_retrain",
                        alert_time=market_close,
                        callback=self.on_retrain
                    )
                    self.postponed_retrain_scheduled = True
                return

        logger.debug(f"Starting model retrain at {now}")


        # ═══════════════════════════════════════════════════════════════════
        # ADAPTIVE WINDOW: Ensure minimum training size
        # ═══════════════════════════════════════════════════════════════════
        # ensures case at t=0 where len(timestamps) for new bars < min_bars_required 
        if self.model.warm_start:
            # only use last untrained data as offset
            days_range = self.calendar.schedule(start_date= self.last_retrain_time, end_date=now)
        else:
            # use full train offset for retrain
            days_range = self.calendar.schedule(start_date= self.model.train_start, end_date=self.model.train_end)

        
        timestamps = market_calendars.date_range(days_range, frequency=self.strategy_params["freq"])
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
            retrain_start_date = self.last_retrain_time,
            active_mask=self.active_mask,
            total_bars = total_bars,
        )
        
        self.last_retrain_time = now
        logger.debug(f"Model retrain completed at {now}")
            

    # ================================================================= #
    # INTERNAL HELPERS
    # ================================================================= #

    # TODO: consider removing and doublecheck position
    def _compute_active_mask(self) -> torch.Tensor:
        """Compute mask for active instruments."""
        # TODO: because risk free should not be counted. This should be fixed in future
        iids_list_without_risk_free = [el for el in self.cache.instrument_ids(venue = self.venue) if el != InstrumentId.from_str(self.risk_free_ticker)]
        mask = ~torch.ones(len(iids_list_without_risk_free), dtype=torch.bool)
        for bt in self.cache.bar_types():
            if not bt.instrument_id.value == self.risk_free_ticker:
                idx = iids_list_without_risk_free.index(bt.instrument_id)
                # Check if data is recent enough
                last_time = pd.to_datetime(self.cache.bar(bt).ts_event, unit='ns', origin='unix', utc=True)
                now = self.clock.utc_now()
                mask[idx] = last_time + freq2pdoffset(self.strategy_params["freq"]) >= now
            else: 
                continue
    
        return mask

    # ----- weight optimiser ------------------------------------------

    def _compute_target_weights(self, preds: Dict[InstrumentId, float]) -> Dict[InstrumentId, float]:
        """Compute target portfolio weights using optimizer with vectorized operations."""

        iids_list_without_risk_free = [el for el in self.cache.instrument_ids(venue = self.venue) if el != InstrumentId.from_str(self.risk_free_ticker)]

        # NOTE: 
        # risk free is not part of the tradable instruments and it is trated as FREE CASH. This avoid optimiziation problems
        # since nav cannot be allocated to assets with non-positive returns.
        try:
            rf_last = float(self.cache.trade_tick(InstrumentId.from_str(self.risk_free_ticker), index=0).price)
            rf_prev = float(self.cache.trade_tick(InstrumentId.from_str(self.risk_free_ticker), index=1).price)
            # Ensure risk free is never zero
            current_rf = max((rf_last - rf_prev) / rf_prev, 1e-6)

        except Error:
            bt = freq2bartype(InstrumentId.from_str(self.risk_free_ticker), self.strategy_params["freq"])
            close_last, close_prev = float(self.cache.bar(bt, index=0).close), float(self.cache.bar(bt, index=1).close)
            # Ensure risk free is never zero
            current_rf = max((close_last - close_prev) / close_prev, 1e-6)

        nav = self._calculate_portfolio_nav()
        cash_available = float(self.portfolio.account(self.venue).balance_free(self.strategy_params["currency"])) - self.cash_buffer
        now = pd.Timestamp(self.clock.utc_now())

        # Pre-fetch all data once
        all_ticks = {iid: self.cache.trade_ticks(iid) for iid in iids_list_without_risk_free}
        # remove risk free from tradable tickers
        #del all_ticks[self.risk_free_ticker]
        all_positions = defaultdict(float)

        for p in self.cache.positions_open(venue=self.venue):
            all_positions[p.instrument_id] += float(p.signed_qty)
        
        # Data collection (Pre-allocate arrays with max possible size)
        valid_iids = []
        returns_list = []
        valid_expected_returns = np.zeros(len(iids_list_without_risk_free))
        current_weights = np.zeros(len(iids_list_without_risk_free))
        prices = np.zeros(len(iids_list_without_risk_free))
        allowed_weight_ranges = np.zeros((len(iids_list_without_risk_free), 2))
        valid_iids_count = 0

        for iid in iids_list_without_risk_free:
            ticks = all_ticks.get(iid)
            if not ticks or len(ticks) < 2:
                continue

            current_price = float(ticks[0].price)
            net_position = all_positions.get(iid, 0.0)
            current_w = (net_position * current_price / nav) if (net_position and nav > 0) else 0.0
            
            # Returns calculation
            lookback_n = max(2, self.optimizer_lookback)
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
            max_w_constrained = min((adv_in_currency * self.max_adv_pct) / nav, self.max_w_abs)
            w_min = max(-max_w_constrained, self.weight_bounds[0])
            w_max = min(max_w_constrained, self.weight_bounds[1])

            # Store for weight optimization
            valid_iids.append(iid)
            returns_list.append(returns.values)
            valid_expected_returns[valid_iids_count] = preds.get(iid, current_price)  # TODO: ensure all models prediction is the expected returns as % change
            current_weights[valid_iids_count] = current_w
            prices[valid_iids_count] = current_price
            allowed_weight_ranges[valid_iids_count] = [w_min, w_max]
            valid_iids_count += 1

        if valid_iids_count <= 1:
            return {}
        
        # Trim arrays to actual size
        valid_expected_returns = valid_expected_returns[:valid_iids_count]
        current_weights = current_weights[:valid_iids_count]
        prices = prices[:valid_iids_count]
        allowed_weight_ranges = allowed_weight_ranges[:valid_iids_count]

        # Covariance from returns
        returns_df = pd.DataFrame(returns_list).T
        cov_horizon = returns_df.cov().values * float(self.pred_len) # scaled to pred_len horizon

        # Keep cash and do sell-only if either case:
        # - if expected loss > commissions
        # - if all the assets with positive returns have sufficient max allocation constraints to cover for the whole portfolio nav
        if bool(np.all(valid_expected_returns <= current_rf))  : 
            logger.debug("All expected returns <= risk-free rate. Minimizing losses..")
            
            # Start with current weights (default: keep everything)
            output_weights = current_weights.copy()
            
            # Identify positions with negative expected returns
            has_position_mask = np.abs(current_weights) > 1e-8
            negative_return_mask = valid_expected_returns < 0
            evaluate_mask = has_position_mask & negative_return_mask
            
            if not np.any(evaluate_mask):
                # No losing positions to evaluate - keep everything
                logger.debug("No losing positions found. Keeping all current positions and holding cash.")
                return {}
            
            # Batch calculate expected losses
            position_values = np.abs(current_weights) * nav
            expected_losses = position_values * np.abs(valid_expected_returns)
            
            # Batch commission calculation for positions to evaluate
            evaluate_indices = np.where(evaluate_mask)[0]
            commissions = np.zeros(len(valid_iids))
            
            for idx in evaluate_indices:
                iid = valid_iids[idx]
                instrument = self.cache.instrument(iid)
                
                try:
                    pos = self.cache.positions(instrument_id = iid, venue = self.venue, strategy_id = self.id, side=PositionSide.LONG)
                    if not pos:
                        current_qty = 0.0
                    else: 
                        current_qty = int(pos[0].quantity)
                except IndexError:
                    current_qty = 0
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
            
            # Vectorized decision: liquidate where expected_loss > commission
            should_liquidate = (expected_losses > commissions) & evaluate_mask
            output_weights[should_liquidate] = 0.0
            
            # Batch logging (single log message)
            if np.any(should_liquidate):
                liquidated_iids = [valid_iids[i] for i in np.where(should_liquidate)[0]]
                total_saved = np.sum(expected_losses[should_liquidate] - commissions[should_liquidate])
                logger.debug(
                    f"Liquidating {len(liquidated_iids)} positions: {liquidated_iids}. "
                    f"Total net benefit: ${total_saved:.2f}"
                )
            
            kept_mask = evaluate_mask & ~should_liquidate
            if np.any(kept_mask):
                kept_iids = [valid_iids[i] for i in np.where(kept_mask)[0]]
                logger.debug(
                    f"Keeping {len(kept_iids)} losing positions (commissions exceed expected losses): {kept_iids}"
                )
            
            # Map to full instruments universe
            w_series = pd.Series(0.0, index=iids_list_without_risk_free)
            w_series[valid_iids] = output_weights
            
            logger.debug(
                f"Portfolio Loss Minimization: {np.sum(should_liquidate)} liquidated, "
                f"{np.sum(~should_liquidate & has_position_mask)} kept, remaining in cash"
            )
            
            return w_series.to_dict()

        elif np.sum(allowed_weight_ranges[valid_expected_returns > current_rf, 1]) < 1:
            # Positive-excess assets can't cover full NAV, allocate max to profitable ones
            positive_mask = valid_expected_returns > current_rf
            output_weights = np.zeros(len(valid_iids)) 
            
            for idx in np.where(positive_mask)[0]:
                iid = valid_iids[idx]
                max_w = allowed_weight_ranges[idx, 1]
                expected_profit = max_w * nav * valid_expected_returns[idx]
                
                # Calculate commission
                instrument = self.cache.instrument(iid)
                target_qty = int((max_w * nav) / prices[idx])
                if target_qty > 0:
                    commission = float(self.fee_model.get_commission(
                        Order_order=None,
                        Quantity_fill_qty=instrument.make_qty(target_qty),
                        Price_fill_px=instrument.make_price(prices[idx]),
                        Instrument_instrument=instrument
                    ))
                    if expected_profit > commission:
                        output_weights[idx] = max_w
            
            w_series = pd.Series(0.0, index=iids_list_without_risk_free)
            w_series[valid_iids] = output_weights
            return w_series.to_dict()


        # Add risk-free asset as free cash to optimizer inputs 
        rf_ticks = self.cache.trade_ticks(InstrumentId.from_str(self.risk_free_ticker))
        rf_price = float(rf_ticks[0].price)
        lookback_n = max(2, self.optimizer_lookback)
        rf_returns = pd.Series([float(t.price) for t in rf_ticks[-lookback_n:]]).pct_change().dropna()
        returns_list.append(rf_returns.values)
        returns_df = pd.DataFrame(returns_list).T
        cov_horizon = returns_df.cov().values * float(self.pred_len)
        valid_iids.append(InstrumentId.from_str(f"CASH.{self.venue}"))
        valid_expected_returns = np.append(valid_expected_returns, current_rf)
        current_weights = np.append(current_weights, cash_available/nav )
        prices = np.append(prices, rf_price)
        allowed_weight_ranges = np.vstack([allowed_weight_ranges, [0.0, 1.0]])


        # NOTE: Controls on cash flow enforced inside the optimizer _add_constraints():
        # sells + active balance < buys + commission + (buffer)

        # Call optimizer with horizon-scaled cov and risk-free rf
        w_opt = self.optimizer.optimize(
            er=pd.Series(valid_expected_returns, index=valid_iids),
            cov=pd.DataFrame(cov_horizon, index=valid_iids, columns=valid_iids),
            rf=current_rf,
            benchmark_vol=self._get_benchmark_volatility(),
            allowed_weight_ranges=allowed_weight_ranges,
            current_weights=current_weights,
            prices=prices,
            nav=nav,
            cash_available=cash_available,
            selector_k = self.strategy_params["top_k"],
        )

        if not w_opt.any():
            # No losing positions to evaluate - keep everything
            w_series = pd.Series(0.0, index=iids_list_without_risk_free)
            logger.warning("Probably a bug. Fix this")
            return w_series.to_dict()

        # Clipping (if optimizer constraints failed) and log if clipping occurred
        w_clipped = np.clip(w_opt, allowed_weight_ranges[:, 0], allowed_weight_ranges[:, 1])  # max bounds

        # Remove risk-free asset from weights (treat allocation as cash)
        rf_weight = w_clipped[-1]
        if rf_weight > 1e-6:
            logger.debug(f"Risk-free allocation: {rf_weight:.4f} ({rf_weight*100:.2f}%) held as cash")
        w_clipped = w_clipped[:-1]
        valid_iids = valid_iids[:-1]
        valid_expected_returns = valid_expected_returns[:-1]
        allowed_weight_ranges = allowed_weight_ranges[:-1]

        # Account for commissions:
        w_final = self._adjust_weights_for_commissions(
            iids = np.array(valid_iids),
            expected_returns_array = valid_expected_returns,
            weights_array = w_clipped,
            nav = nav)

        # Map to universe
        w_series = pd.Series(0.0, index=iids_list_without_risk_free)
        w_series[valid_iids] = w_final  # Direct assignment using list indexing

        return w_series.to_dict()

    def _adjust_weights_for_commissions(
        self, 
        iids: np.ndarray,
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
            iids: Array of instrument ids
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
        for idx, iid in enumerate(iids):
            target_weight = adjusted_weights[idx]
            
            instrument = self.cache.instrument(iid)
            
            # Get current position and price
            try:
                pos = self.cache.positions(instrument_id = iid, venue = self.venue, strategy_id = self.id, side=PositionSide.LONG)
                if not pos:
                    current_qty = 0.0
                else: 
                    current_qty = int(pos[0].quantity)
            except IndexError:
                current_qty = 0
            try:
                current_price = float(self.cache.trade_tick(iid).price)
            except Error:
                bt = freq2bartype(iid, self.strategy_params["freq"])
                current_price = float(self.cache.bar(bt).close)
            
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
                    f"Skipping {iid.value}: target_qty={target_qty_float:.3f} rounds to zero "
                    f"(price={current_price:.2f}, target_weight={target_weight:.6f})"
                )
                adjusted_weights[idx] = 0.0
                continue
            
            # Recalculate weight based on actual integer quantity
            adjusted_weight = (target_qty * current_price) / nav 
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
                    'symbol': iid.value,
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
            commission = self.fee_model.get_commission(
                Order_order=None,
                Quantity_fill_qty=instrument.make_qty(abs(trade_qty)),
                Price_fill_px=instrument.make_price(current_price),
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
                'symbol': iid.value,
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
    