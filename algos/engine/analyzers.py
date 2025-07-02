"""
Minimal analyzers re-implemented for Nautilus events.
Add more as needed.
"""
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import pandas as pd


class EquityCurveAnalyzer:
    def __init__(self):
        self._rows: List[Dict] = []

    def on_equity(self, ts: datetime, value: float):
        self._rows.append({"ts": ts, "equity": value})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows).set_index("ts")
