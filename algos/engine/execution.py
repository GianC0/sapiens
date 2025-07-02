"""
Execution helpers for Nautilus Trader.

* CommissionModelBps — linear % of notional.
* SlippageModelBps   — fixed spread paid on each fill.

Both are *stateless* and therefore cheap to instantiate.
"""
from __future__ import annotations

from nautilus_trader.execution.fees import CommissionModel
from nautilus_trader.execution.slippage import SlippageModel
from nautilus_trader.model.objects import Money


class CommissionModelBps(CommissionModel):
    def __init__(self, bps: float):
        super().__init__()
        self._rate = bps * 1e-4

    def calculate(self, trade):
        notional = abs(trade.price * trade.size)
        return Money(notional * self._rate, trade.currency)


class SlippageModelBps(SlippageModel):
    def __init__(self, bps: float):
        super().__init__()
        self._rate = bps * 1e-4

    def apply(self, side: str, price: float) -> float:
        adj = price * self._rate
        return price + adj if side == "BUY" else price - adj
