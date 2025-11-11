"""
Base class for all Sapiens trading strategies.
"""

from abc import ABC
from pathlib import Path
from typing import Dict, Any
from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from dataclasses import dataclass, field



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
        
        # Setup paths
        self.strategy_name = self.strategy_params.get("strategy_name", self.__class__.__name__)
        self.strategy_dir = Path(f"strategies/{self.strategy_name}")
        self.strategy_dir.mkdir(parents=True, exist_ok=True)