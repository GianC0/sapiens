"""
Execution helpers for Nautilus Trader.

* CommissionModelBps — linear % of notional
* SlippageModelBps   — fixed spread paid on each fill
"""
from __future__ import annotations

from nautilus_trader.execution.fees import CommissionModel
from nautilus_trader.execution.slippage import SlippageModel
from nautilus_trader.model.objects import Money


class CommissionModelBps(CommissionModel):
    """Commission fee = notional × (bps / 10 000)"""

    def __init__(self, bps: float):
        super().__init__()
        self._rate = bps * 1e-4

    # NOTE: the current framework passes an *OrderFill* instance
    #       so we reference .price / .quantity instead of .size
    def calculate(self, fill) -> Money:          # <- type hint change
        notional = abs(fill.price * fill.quantity)
        return Money(notional * self._rate, fill.currency)


class SlippageModelBps(SlippageModel):
    """Bid-ask slippage = price ± (bps / 10 000)"""

    def __init__(self, bps: float):
        super().__init__()
        self._rate = bps * 1e-4

    def apply(self, side: str, price: float) -> float:
        adj = price * self._rate
        return price + adj if side.upper() == "BUY" else price - adj
