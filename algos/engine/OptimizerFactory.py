import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions
from pypfopt.exceptions import OptimizationError
import logging

logger = logging.getLogger(__name__)

def create_optimizer(name: str, **kwargs):
    """Factory for portfolio optimizers using PyPortfolioOpt."""
    optimizers = {
        "max_sharpe": MaxSharpeOptimizer,
        "min_variance": MinVarianceOptimizer,
        "equal_weight": EqualWeightOptimizer,
        "m2": M2Optimizer,
        "max_quadratic_utility": MaxQuadraticUtilityOptimizer,
        "efficient_risk": EfficientRiskOptimizer,
    }
    
    if name not in optimizers:
        logger.warning(f"Unknown optimizer {name}, using equal_weight")
        return EqualWeightOptimizer()
    
    return optimizers[name](**kwargs)


class PortfolioOptimizer:
    """Base class for portfolio optimizers using PyPortfolioOpt."""
    
    def __init__(self, weight_bounds=(-1, 1), solver=None):
        """
        Args:
            weight_bounds: tuple of (min, max) weights, default allows short selling
            solver: cvxpy solver to use (None for default)
        """
        self.weight_bounds = weight_bounds
        self.solver = solver
    
    def optimize(self, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
        raise NotImplementedError
    
    def _to_array(self, weights_dict: dict, n_assets: int) -> np.ndarray:
        """Convert PyPortfolioOpt weights dict to numpy array."""
        if isinstance(weights_dict, dict):
            return np.array(list(weights_dict.values()))
        return weights_dict


class MaxSharpeOptimizer(PortfolioOptimizer):
    """Maximum Sharpe ratio optimizer using PyPortfolioOpt."""
    
    def __init__(self, weight_bounds=(-1, 1), solver=None, risk_free_rate=None):
        super().__init__(weight_bounds, solver)
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, **kwargs) -> np.ndarray:
        """
        Find weights that maximize Sharpe ratio.
        
        Args:
            mu: Expected returns (n,)
            cov: Covariance matrix (n, n)
            rf: Risk-free rate
        """
        try:
            # Use provided rf or instance default
            risk_free = rf if rf != 0.0 else (self.risk_free_rate or 0.0)
            
            # Create efficient frontier object
            ef = EfficientFrontier(
                expected_returns=pd.Series(mu),
                cov_matrix=pd.DataFrame(cov),
                weight_bounds=self.weight_bounds,
                solver=self.solver
            )
            
            # Maximize Sharpe ratio
            weights = ef.max_sharpe(risk_free_rate=risk_free)
            
            # Clean and return weights
            cleaned_weights = ef.clean_weights()
            return self._to_array(cleaned_weights, len(mu))
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Max Sharpe optimization failed: {e}. Using equal weights.")
            return np.ones(len(mu)) / len(mu)


class MinVarianceOptimizer(PortfolioOptimizer):
    """Minimum variance portfolio optimizer using PyPortfolioOpt."""
    
    def optimize(self, mu: np.ndarray, cov: np.ndarray, **kwargs) -> np.ndarray:
        """
        Find minimum variance portfolio.
        
        Args:
            mu: Expected returns (n,)
            cov: Covariance matrix (n, n)
        """
        try:
            ef = EfficientFrontier(
                expected_returns=pd.Series(mu),
                cov_matrix=pd.DataFrame(cov),
                weight_bounds=self.weight_bounds,
                solver=self.solver
            )
            
            # Minimize volatility
            weights = ef.min_volatility()
            
            # Clean and return weights
            cleaned_weights = ef.clean_weights()
            return self._to_array(cleaned_weights, len(mu))
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Min variance optimization failed: {e}. Using equal weights.")
            return np.ones(len(mu)) / len(mu)


class M2Optimizer(PortfolioOptimizer):
    """
    Modigliani M² optimizer using PyPortfolioOpt with custom objective.
    
    M² adjusts portfolio returns to match benchmark volatility for comparison.
    """
    
    def __init__(self, weight_bounds=(-1, 1), solver=None):
        """
        Args:
            benchmark_vol: Annualized benchmark volatility (default 15%)
            weight_bounds: Weight constraints
            solver: cvxpy solver
        """
        super().__init__(weight_bounds, solver)
    
    def optimize(self, mu: np.ndarray, cov: np.ndarray, rf: float, benchmark_vol: float, **kwargs) -> np.ndarray:
        """
        Find weights that maximize M² measure.
        
        M² = (r_p - r_f) * (ro_b / ro_p) + r_f
        
        This is equivalent to maximizing Sharpe ratio scaled by benchmark vol.
        """
        try:
            ef = EfficientFrontier(
                expected_returns=pd.Series(mu),
                cov_matrix=pd.DataFrame(cov),
                weight_bounds=self.weight_bounds,
                solver=self.solver
            )
            
            # Define custom M² objective
            # Since M² is monotonic with Sharpe ratio, we can maximize Sharpe
            # and then scale to match benchmark volatility
            
            # First get max Sharpe portfolio
            ef.max_sharpe(risk_free_rate=rf)
            sharpe_weights = ef.clean_weights()
            sharpe_array = self._to_array(sharpe_weights, len(mu))
            
            # Calculate portfolio volatility
            port_vol = np.sqrt(np.dot(sharpe_array, np.dot(cov, sharpe_array)))
            
            # Scale weights to match benchmark volatility
            # This preserves the Sharpe ratio while adjusting risk
            if port_vol > 0:
                scale = benchmark_vol / port_vol
                
                # Mix with risk-free asset to achieve target volatility
                if scale < 1:
                    # Reduce exposure (add cash)
                    scaled_weights = sharpe_array * scale
                else:
                    # For scale > 1, we'd need leverage
                    # Cap at 1 to avoid leverage unless bounds allow it
                    if self.weight_bounds[1] > 1:
                        scaled_weights = sharpe_array * min(scale, self.weight_bounds[1])
                    else:
                        scaled_weights = sharpe_array
                
                # Renormalize if needed
                weight_sum = np.sum(np.abs(scaled_weights))
                if weight_sum > 0:
                    return scaled_weights / weight_sum
                    
            return sharpe_array
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"M² optimization failed: {e}. Using equal weights.")
            return np.ones(len(mu)) / len(mu)


class MaxQuadraticUtilityOptimizer(PortfolioOptimizer):
    """
    Maximize quadratic utility (expected return - risk_aversion * variance).
    Good for risk-averse investors.
    """
    
    def __init__(self, risk_aversion: int = 1, weight_bounds=(-1, 1), solver=None):
        """
        Args:
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            weight_bounds: Weight constraints
            solver: cvxpy solver
        """
        super().__init__(weight_bounds, solver)
        self.risk_aversion = risk_aversion
    
    def optimize(self, mu: np.ndarray, cov: np.ndarray, **kwargs) -> np.ndarray:
        """
        Find weights that maximize quadratic utility.
        
        U = μᵀw - (γ/2) * wᵀΣw
        where γ is risk aversion parameter.
        """
        try:
            ef = EfficientFrontier(
                expected_returns=pd.Series(mu),
                cov_matrix=pd.DataFrame(cov),
                weight_bounds=self.weight_bounds,
                solver=self.solver
            )
            
            # Maximize quadratic utility
            weights = ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
            
            # Clean and return weights
            cleaned_weights = ef.clean_weights()
            return self._to_array(cleaned_weights, len(mu))
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Quadratic utility optimization failed: {e}. Using equal weights.")
            return np.ones(len(mu)) / len(mu)


class EfficientRiskOptimizer(PortfolioOptimizer):
    """
    Target a specific risk level and maximize return.
    Useful for risk-targeted strategies.
    """
    
    def __init__(self, target_volatility: float = 0.15, weight_bounds=(-1, 1), solver=None):
        """
        Args:
            target_volatility: Target portfolio volatility
            weight_bounds: Weight constraints
            solver: cvxpy solver
        """
        super().__init__(weight_bounds, solver)
        self.target_volatility = target_volatility
    
    def optimize(self, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
        """
        Find weights that maximize return for given risk level.
        """
        try:
            ef = EfficientFrontier(
                expected_returns=pd.Series(mu),
                cov_matrix=pd.DataFrame(cov),
                weight_bounds=self.weight_bounds,
                solver=self.solver
            )
            
            # Maximize return for target volatility
            weights = ef.efficient_risk(target_volatility=self.target_volatility)
            
            # Clean and return weights
            cleaned_weights = ef.clean_weights()
            return self._to_array(cleaned_weights, len(mu))
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Efficient risk optimization failed: {e}. Using equal weights.")
            return np.ones(len(mu)) / len(mu)


class EqualWeightOptimizer(PortfolioOptimizer):
    """Simple equal weight optimizer (no optimization needed)."""
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def optimize(self, mu: np.ndarray, **kwargs) -> np.ndarray:
        """Return equal weights for all assets."""
        n = len(mu)
        return np.ones(n) / n