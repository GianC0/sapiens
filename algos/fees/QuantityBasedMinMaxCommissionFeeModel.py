from decimal import Decimal
from nautilus_trader.config import PositiveFloat
from nautilus_trader.backtest.config import FeeModelConfig
from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price, Quantity, Money
from typing import Dict, List, Optional, Any
from dataclasses import field

class QuantityBasedMinMaxCommissionFeeModelConfig(FeeModelConfig):
    """
    Configuration for ``QuantityBasedMinMaxCommissionFeeModel``.
    
    Config parameters
    ----------
    commission_per_unit : str
        The commission per share/unit (e.g., "0.005 USD").
    min_commission : str
        The minimum commission per order (e.g., "1.00 USD").
    max_commission_pct: Decimal
        
    """
    config : Dict[str, Any] = field(default_factory=dict) 

    #commission_per_unit: Money
    #min_commission: Money
    #max_commission_pct: Decimal

class QuantityBasedMinMaxCommissionFeeModel(FeeModel):
    """
    Provides a fee model which charges a commission per share/unit with a minimum commission.
    
    This model is commonly used by brokers like Interactive Brokers where commission
    is calculated as max(commission_per_unit * quantity, min_commission).

    Config parameters
    ----------
    commission_per_unit : Money
        The commission amount per share/unit.
    min_commission : Money
        The minimum commission amount per order.
    max_commission_pct : Decimal
        The maximum commission amount as % of trade value

    Raises
    ------
    ValueError
        If `commission_per_unit` or `min_commission` is negative (< 0).
    ValueError
        If currencies of `commission_per_unit` and `min_commission` do not match.
    """

    def __init__(
        self, 
        config: QuantityBasedMinMaxCommissionFeeModelConfig,
    ) -> None:
        """
        Initialize the fee model with configuration.
        
        Parameters
        ----------
        config : QuantityBasedMinMaxCommissionFeeModelConfig
            The fee model configuration.
        """
        super().__init__()
        cfg = config.config
        
        # Validate non-negative values
        if cfg["commission_per_unit"] < 0:
            raise ValueError("commission_per_unit must be non-negative")
        if cfg["min_commission"] < 0:
            raise ValueError("min_commission must be non-negative")
        if cfg["max_commission_pct"] < 0:
            raise ValueError("max_commission_pct must be non-negative")
        
        self._commission_per_unit = cfg["commission_per_unit"]
        self._min_commission = cfg["min_commission"]
        self._max_commission_pct = cfg["max_commission_pct"]

    def get_commission(
        self,
        Order_order = None,
        Quantity_fill_qty = None,
        Price_fill_px = None,
        Instrument_instrument = None,
    ) -> Money:
        currency =  Instrument_instrument.quote_currency
        calculated_commission = float(self._commission_per_unit) * float(Quantity_fill_qty)
        max_fee = float(Price_fill_px) * float(Quantity_fill_qty) * float(self._max_commission_pct)
        
        final_commission = min( max(calculated_commission, self._min_commission) , max_fee )
        
        return Money(final_commission, currency)