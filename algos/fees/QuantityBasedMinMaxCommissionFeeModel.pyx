from decimal import Decimal
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import FeeModelConfig
from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.core.correctness import Condition
from nautilus_trader.core.rust.model import LiquiditySide
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.functions import liquidity_side_to_str
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price, Quantity, Money

class QuantityBasedMinCommissionFeeModelConfig(FeeModelConfig):
    """
    Configuration for ``QuantityBasedMinCommissionFeeModel``.
    
    Parameters
    ----------
    commission_per_unit : str
        The commission per share/unit (e.g., "0.005 USD").
    min_commission : str
        The minimum commission per order (e.g., "1.00 USD").
    max_commission_pct: Decimal
        
    """
    
    commission_per_unit: Money
    min_commission: Money
    max_commission_pct: Decimal

class QuantityBasedMinCommissionFeeModel(FeeModel):
    """
    Provides a fee model which charges a commission per share/unit with a minimum commission.
    
    This model is commonly used by brokers like Interactive Brokers where commission
    is calculated as max(commission_per_unit * quantity, min_commission).

    Parameters
    ----------
    commission_per_unit : Money
        The commission amount per share/unit.
    min_commission : Money
        The minimum commission amount per order.
    max_commission_pct : Decimal
        The maximum commission amount as % of trade value
    config : QuantityBasedMinCommissionFeeModelConfig
        The configuration for the model.

    Raises
    ------
    ValueError
        If both direct parameters **and** ``config`` are provided, **or** if both are ``None``.
    ValueError
        If `commission_per_unit` or `min_commission` is negative (< 0).
    ValueError
        If currencies of `commission_per_unit` and `min_commission` do not match.
    """

    def __init__(self, commission_per_unit: Money, min_commission: Money, max_commission: Decimal) -> None:
        super().__init__()
        self._commission_per_unit = commission_per_unit
        self._min_commission = min_commission
        self._max_commission = max_commission

    def get_commission(
        self,
        Order_order,
        Quantity_fill_qty,
        Price_fill_px,
        Instrument_instrument,
    ) -> Money:
        calculated_commission = Money(self._commission_per_unit * Quantity_fill_qty, self.commission.currency)
        max_fee = Price_fill_px * Quantity_fill_qty * self._max_commission
        
        final_commission = min( max(calculated_commission, self._min_commission) , max_fee )
        
        return final_commission