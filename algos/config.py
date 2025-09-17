# algos/config.py
from nautilus_trader.trading.config import StrategyConfig
from pydantic import BaseModel, Field, field_validator, validator
from typing import Dict, Any, Optional, List
from enum import Enum

class AccountType(str, Enum):
    CASH = "CASH"
    MARGIN = "MARGIN"

class OptimizerType(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    EQUAL_WEIGHT = "equal_weight"
    M2 = "m2"
    MAX_QUADRATIC_UTILITY = "max_quadratic_utility"
    EFFICIENT_RISK = "efficient_risk"

class ExecutionConfig(BaseModel, frozen=True):
    """Execution configuration"""
    timing_force: str = "DAY"
    use_limit_orders: bool = False
    limit_offset_bps: int = 5
    max_retries: int = 3
    twap_horizon_secs: float = 10.0
    twap_interval_secs: float = 2.5
    twap_slices: int = Field(default=4, ge=1)

class LiquidityConfig(BaseModel, frozen=True):
    """Liquidity constraints configuration"""
    adv_lookback: int = Field(default=30, ge=1)
    max_adv_pct: float = Field(default=0.05, gt=0, le=1)

class RiskConfig(BaseModel, frozen=True):
    """Risk management configuration"""
    max_weight_abs: float = Field(default=0.03, gt=0, le=1)
    max_weight_rel: float = Field(default=0.20, gt=0, le=1)
    trailing_stop_pct: float = Field(default=0.05, gt=0, le=1)
    drawdown_pct: float = Field(default=0.15, gt=0, le=1)
    target_volatility_annual: Optional[float] = Field(default=0.05, gt=0)
    
    @field_validator('max_weight_rel')
    def validate_weight_rel(cls, v, values):
        if 'max_weight_abs' in values and v < values['max_weight_abs']:
            raise ValueError('max_weight_rel must be >= max_weight_abs')
        return v

class CostsConfig(BaseModel, frozen=True):
    """Trading costs configuration"""
    fee_bps: float = Field(default=0.5, ge=0)
    prob_slippage: float = Field(default=0.2, ge=0, le=1)
    prob_fill_on_limit: float = Field(default=0.2, ge=0, le=1)

class CacheConfig(BaseModel, frozen=True):
    """Cache configuration"""
    bar_capacity: int = Field(default=4096, ge=100)

class EngineConfig(BaseModel, frozen=True):
    """Engine configuration"""
    cache: CacheConfig = Field(default_factory=CacheConfig)

class ModelParams(BaseModel, frozen=True):
    """Model parameters"""
    model_name: str
    model_dir: Optional[str] = None
    pred_len: int = Field(default=1, ge=1)
    window_len: int = Field(default=10, ge=1)
    features_to_load: str = "candles"
    adjust: bool = True
    feature_dim: int = 5
    target_idx: int = 3
    n_epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=64, ge=1)
    
    # Training configuration
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    valid_start: Optional[str] = None
    valid_end: Optional[str] = None
    
    # Add other model-specific parameters
    lambda_ic: float = Field(default=0.5, ge=0, le=1)
    lambda_sync: float = Field(default=1.0, ge=0)
    lambda_rankic: float = Field(default=0.1, ge=0)
    temperature: float = Field(default=0.07, gt=0)
    sync_thr: float = Field(default=0.6, ge=0, le=1)
    lr_stage1: float = Field(default=0.001, gt=0)
    lr_stage2: float = Field(default=0.0001, gt=0)

class StrategyParams(BaseModel, frozen=True):
    """Strategy-specific parameters"""
    strategy_name: str = "TopKStrategy"
    freq: str = "1D"
    calendar: str = "NYSE"
    currency: str = "USD"
    initial_cash: float = Field(default=20000, gt=0)
    account_type: AccountType = AccountType.CASH
    
    # Data configuration
    data_dir: str
    train_start: str
    backtest_end: str
    valid_split: float = Field(default=0.2, gt=0, lt=1)
    
    # Portfolio configuration
    top_k: int = Field(default=30, ge=1)
    rebalance_only: bool = False
    optimizer_name: OptimizerType = OptimizerType.MAX_SHARPE
    optimizer_lookback: str = "60B"
    
    # Retraining configuration
    retrain_offset: str = "30B"
    train_offset: str = "12BME"
    warm_start: bool = True
    warm_training_epochs: int = Field(default=5, ge=1)
    
    @field_validator('top_k')
    def validate_top_k(cls, v, values):
        if values.get('account_type') == AccountType.CASH and v < 1:
            raise ValueError('top_k must be positive for CASH accounts')
        return v

class TopKStrategyConfig(StrategyConfig, frozen=True):
    """
    Complete configuration for TopK Strategy.
    Inherits from StrategyConfig for base Nautilus parameters.
    """
    # Strategy parameters
    strategy: StrategyParams = Field(..., description="Strategy-specific parameters")
    
    # Model parameters
    model: ModelParams = Field(..., description="Model configuration")
    
    # Risk management
    risk: RiskConfig = Field(default_factory=RiskConfig)
    
    # Execution settings
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    
    # Liquidity constraints
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
    
    # Trading costs
    costs: CostsConfig = Field(default_factory=CostsConfig)
    
    # Engine settings
    engine: EngineConfig = Field(default_factory=EngineConfig)
    
    class Config:
        """Pydantic configuration"""
        extra = 'forbid'  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TopKStrategyConfig':
        """
        Create config from nested dictionary (e.g., from YAML).
        
        Args:
            config_dict: Nested configuration dictionary
            
        Returns:
            TopKStrategyConfig instance
        """
        # Restructure the dict to match our config structure
        structured = {
            'strategy': config_dict.get('STRATEGY', {}).get('PARAMS', {}),
            'model': config_dict.get('MODEL', {}).get('PARAMS', {}),
            'risk': {
                'max_weight_abs': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('risk_max_weight_abs', 0.03),
                'max_weight_rel': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('risk_max_weight_rel', 0.20),
                'trailing_stop_pct': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('risk_trailing_stop_pct', 0.05),
                'drawdown_pct': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('risk_drawdown_pct', 0.15),
                'target_volatility_annual': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('risk_target_volatility_annual', 0.05),
            },
            'execution': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('execution', {}),
            'liquidity': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('liquidity', {}),
            'costs': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('costs', {}),
            'engine': config_dict.get('STRATEGY', {}).get('PARAMS', {}).get('engine', {}),
        }
        
        # Add any StrategyConfig base fields if needed
        if 'strategy_id' in config_dict:
            structured['strategy_id'] = config_dict['strategy_id']
            
        return cls(**structured)
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for backward compatibility."""
        flat = {}
        
        # Flatten strategy params
        for k, v in self.strategy.dict().items():
            flat[k] = v
            
        # Add risk params with prefix
        flat['risk_max_weight_abs'] = self.risk.max_weight_abs
        flat['risk_max_weight_rel'] = self.risk.max_weight_rel
        flat['risk_trailing_stop_pct'] = self.risk.trailing_stop_pct
        flat['risk_drawdown_pct'] = self.risk.drawdown_pct
        flat['risk_target_volatility_annual'] = self.risk.target_volatility_annual
        
        # Add nested configs
        flat['execution'] = self.execution.dict()
        flat['liquidity'] = self.liquidity.dict()
        flat['costs'] = self.costs.dict()
        flat['engine'] = self.engine.dict()
        
        return flat