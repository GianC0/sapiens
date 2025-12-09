from calendar import prcal
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions
from pypfopt.exceptions import OptimizationError
import logging
import cvxpy as cp

logger = logging.getLogger(__name__)

def create_optimizer(name: str, **kwargs):
    """Factory for portfolio optimizers using PyPortfolioOpt."""
    optimizers = {
        "max_sharpe": MaxSharpeOptimizer,
        "min_variance": MinVarianceOptimizer,
        "m2": M2Optimizer,
        "max_quadratic_utility": MaxQuadraticUtilityOptimizer,
        "efficient_risk": EfficientRiskOptimizer,
    }
    
    if name not in optimizers:
        logger.warning(f"Unknown optimizer {name}, using equal_weight")
        raise NotImplementedError(f"Unknown optimizer {name}")
    
    return optimizers[name](**kwargs)


class PortfolioOptimizer:
    """Base class for portfolio optimizers using PyPortfolioOpt."""
    
    def __init__(self, max_adv_pct: float, weight_bounds: tuple, solver = None, **kwargs ):
        """
        Args:
            weight_bounds: tuple of (min, max) weights, default allows short selling
            solver: cvxpy solver to use (None for default)
            max_adv_pct: maximum fraction of ADV allowed per trade (e.g. 0.01 = 1% ADV)
        """

        self.weight_bounds = weight_bounds
        self.max_adv_pct = max_adv_pct
        self._last_valid_weights = None
    
    def optimize(self, er: pd.Series, cov: pd.DataFrame, rf: float, 
                allowed_weight_ranges: np.ndarray, 
                current_weights: np.ndarray,
                prices: np.ndarray,
                nav: float,
                cash_available: float,
                **kwargs) -> np.ndarray:
        """
        Optimize portfolio weights.
        
        Args:
            er: Expected returns (n,)
            cov: Covariance matrix (n, n)
            rf: Risk-free rate
            allowed_weight_ranges: Per-asset weight bounds (n, 2) considering liquidity
            current_weights: current weights of instruments relative to nav
            prices: np.array of current prices (float) for each instrument
            nav: net asset valuatione i.e., current portfolio value ( - cash buffer )
            cash_available: cash_available i.e., portfolio balance_free
        """
        raise NotImplementedError
    
    def _to_array(self, weights_dict: dict) -> np.ndarray:
        """Convert PyPortfolioOpt weights dict to numpy array."""
        if isinstance(weights_dict, dict):
            return np.array(list(weights_dict.values()))
        return {}

    def _apply_constraints(
        self, 
        ef: EfficientFrontier, 
        allowed_weight_ranges: np.ndarray,
        current_weights: np.ndarray,
        prices:  np.ndarray,
        nav: float,
        cash_available: float,
    ) -> None:
        """
        Apply all portfolio constraints to EfficientFrontier object.
        
        Args:
            ef: EfficientFrontier object to apply constraints to
            allowed_weight_ranges: Array of shape (n_assets, 2) with [min, max] for each asset
            current_weights: current weights of instruments relative to nav
            prices: np.array of current prices (float) for each instrument
            nav: net asset valuatione i.e., current portfolio value ( - cash buffer )
            cash_available: cash_available i.e., portfolio balance_free
        """
        # Buy notional <= Sell notional + available cash - buffer
        # Only add cash constraint if we have existing positions
        """
        if current_weights is not None and np.sum(np.abs(current_weights)) > 1e-8:
            def cash_constraint(w):
                #trade_values = (w - current_weights) * nav
                #buys = cp.sum(cp.pos(trade_values))
                #sells = cp.sum(cp.neg(trade_values))
                return nav * cp.sum(w - current_weights) <= cash_available
            
            ef.add_constraint(cash_constraint)
        """
        

    def _get_safe_fallback_weights(
        self, 
        n_assets: int, 
        allowed_weight_ranges: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Production-ready fallback when optimization fails.
        Priority order:
        1. Maintain current positions if within constraints
        2. Return zero weights (hold cash or all to risk free index)
        """
        
        # Option 1: Try to maintain current positions if provided
        #if current_weights and len(current_weights) == n_assets:
        #    # Check if current weights are within new constraints
        #    if allowed_weight_ranges:
        #        valid_current = True
        #        for i in range(n_assets):
        #            w_min, w_max = allowed_weight_ranges[i]
        #            if current_weights[i] < w_min or current_weights[i] > w_max:
        #                valid_current = False
        #                break
        #        
        #        if valid_current:
        #            logger.info("Optimization failed: maintaining current positions")
        #            return current_weights
        
        # Option 2: Return zero weights (hold cash outside of portfolio)
        logger.warning("Optimization failed: no feasible allocation, holding cash or all to risk free")
        return current_weights
    
class MaxSharpeOptimizer(PortfolioOptimizer):
    """Maximum Sharpe ratio optimizer using PyPortfolioOpt."""
    
    def __init__(self, max_adv_pct: float, weight_bounds=(-1, 1), solver=None, risk_free_rate=None, **kwargs):
        super().__init__(max_adv_pct, weight_bounds, solver)
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, er: pd.Series, cov: pd.DataFrame, rf: float, 
                allowed_weight_ranges: np.ndarray, 
                current_weights: np.ndarray, 
                prices: np.ndarray,
                nav: float,
                cash_available: float,
                **kwargs) -> np.ndarray:
        """
        Find weights that maximize Sharpe ratio.
        
        Args:
            er: Expected returns (n,)
            cov: Covariance matrix (n, n)
            rf: Risk-free rate
            allowed_weight_ranges:  (w_min, w_max) tuples of max/min weight relative to nav
            current_weights: current weights of instruments relative to nav
            prices: np.array of current latest prices (float) for each instrument
            nav: net asset valuatione i.e., current portfolio value ( - cash buffer )
            cash_available: uninvested cash i.e., portfolio balance_free
        """
        try:
            # Use provided rf or instance default
            risk_free = rf if rf != 0.0 else (self.risk_free_rate or 0.0)
            
            # Create efficient frontier object
            ef = EfficientFrontier(
                expected_returns=er,
                cov_matrix=cov,
                weight_bounds=allowed_weight_ranges,
            )
            # Apply all constraints (without top-k)
            self._apply_constraints(ef, allowed_weight_ranges, current_weights, prices, nav, cash_available)
            
            # Maximize Sharpe ratio
            ef.max_sharpe(risk_free_rate=risk_free)
            
            # Clean and return weights
            cleaned_weights = ef.clean_weights()
            sharpe_array = self._to_array(cleaned_weights)

            # If selector_k is requested, run a second pass that forces non-top-k weights to zero.
            selector_k = kwargs.get("selector_k", None)
            if selector_k is not None and selector_k <= (er > rf).sum():

                # set non-topk to have 0 as max allocation factor
                idx_sorted = np.argsort(-np.abs(sharpe_array))
                top_idx = set(idx_sorted[:selector_k])
                modified_ranges = allowed_weight_ranges.copy()
                for i in range(len(sharpe_array)):
                    if i not in top_idx:
                        modified_ranges[i] = [0.0, 0.0]

                # build a fresh EfficientFrontier to enforce zeros
                ef2 = EfficientFrontier(
                    expected_returns=er,
                    cov_matrix=cov,
                    weight_bounds=modified_ranges,
                )
                
                self._apply_constraints(ef2, modified_ranges, current_weights, prices, nav, cash_available)

                # Solve constrained max-sharpe on the top-k subset
                ef2.max_sharpe(risk_free_rate=rf)
                sharpe_weights2 = ef2.clean_weights()
                sharpe_array = self._to_array(sharpe_weights2)
                
            self._last_valid_weights = sharpe_array
            return self._last_valid_weights
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Max Sharpe optimization failed: {e}. Using safe fallback.")
            
            # Try to use last valid weights first
            fallback_current = current_weights if current_weights is not None else self._last_valid_weights
            
            return self._get_safe_fallback_weights(
                n_assets=len(er),
                allowed_weight_ranges=allowed_weight_ranges,
                current_weights=fallback_current,
            )


class MinVarianceOptimizer(PortfolioOptimizer):
    """Minimum variance portfolio optimizer using PyPortfolioOpt."""
    
    def optimize(self, er: pd.Series, cov: pd.DataFrame, rf: float, 
                allowed_weight_ranges: np.ndarray, 
                current_weights: np.ndarray, 
                prices: np.ndarray,
                nav: float,
                cash_available: float,
                **kwargs) -> np.ndarray:
        """
        Find minimum variance portfolio.
        
        Args:
            er: Expected returns (n,)
            cov: Covariance matrix (n, n)
            allowed_weight_ranges:  (w_min, w_max) tuples of max/min weight relative to nav
            current_weights: current weights of instruments relative to nav
            prices: np.array of current latest prices (float) for each instrument
            nav: net asset valuatione i.e., current portfolio value ( - cash buffer )
            cash_available: uninvested cash i.e., portfolio balance_free
        """
        try:
            ef = EfficientFrontier(
                expected_returns=er,
                cov_matrix=cov,
                weight_bounds=allowed_weight_ranges,
            )

            # Add all constraints
            self._apply_constraints(ef, allowed_weight_ranges, current_weights, prices, nav, cash_available)
            
            # Minimize volatility
            ef.min_volatility()
            
            # Clean and return weights
            cleaned_weights = ef.clean_weights()
            sharpe_array = self._to_array(cleaned_weights)

            # If selector_k is requested, run a second pass that forces non-top-k weights to zero.
            selector_k = kwargs.get("selector_k", None)
            if selector_k is not None and selector_k <= (er > rf).sum():

                # set non-topk to have 0 as max allocation factor
                idx_sorted = np.argsort(-np.abs(sharpe_array))
                top_idx = set(idx_sorted[:selector_k])
                modified_ranges = allowed_weight_ranges.copy()
                for i in range(len(sharpe_array)):
                    if i not in top_idx:
                        modified_ranges[i] = [0.0, 0.0]

                # build a fresh EfficientFrontier to enforce zeros
                ef2 = EfficientFrontier(
                    expected_returns=er,
                    cov_matrix=cov,
                    weight_bounds=modified_ranges,
                )
                
                self._apply_constraints(ef2, modified_ranges, current_weights, prices, nav, cash_available)

                # Solve constrained max-sharpe on the top-k subset
                ef2.max_sharpe(risk_free_rate=rf)
                sharpe_weights2 = ef2.clean_weights()
                sharpe_array = self._to_array(sharpe_weights2)
                
            self._last_valid_weights = sharpe_array
            return self._last_valid_weights
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Max Sharpe optimization failed: {e}. Using safe fallback.")
            
            # Try to use last valid weights first
            fallback_current = current_weights if current_weights is not None else self._last_valid_weights
            
            return self._get_safe_fallback_weights(
                n_assets=len(er),
                allowed_weight_ranges=allowed_weight_ranges,
                current_weights=fallback_current,
            )


class M2Optimizer(PortfolioOptimizer):
    """
    Modigliani M² optimizer using PyPortfolioOpt with custom objective.
    
    M² adjusts portfolio returns to match benchmark volatility for comparison.
    """
    
    def __init__(self, max_adv_pct: float, weight_bounds=(-1, 1), solver=None, **kwargs):
        """
        Args:
            benchmark_vol: Annualized benchmark volatility (default 15%)
            weight_bounds: Weight constraints
            solver: cvxpy solver
        """
        super().__init__(max_adv_pct, weight_bounds, solver)
    
    def optimize(self, er: pd.Series, cov: pd.DataFrame, rf: float, 
                allowed_weight_ranges: np.ndarray,
                current_weights: np.ndarray,
                prices: np.ndarray,
                nav: float,
                cash_available: float,
                benchmark_vol: float,
                **kwargs) -> np.ndarray:
        
        """
        Find weights that maximize M² measure.
        
        M² = (r_p - r_f) * (ro_b / ro_p) + r_f
        
        This is equivalent to maximizing Sharpe ratio scaled by benchmark vol.
        """
        try:
            ef = EfficientFrontier(
                expected_returns=er,
                cov_matrix=cov,
                weight_bounds=allowed_weight_ranges,
            )
            
            # Define custom M² objective
            # Since M² is monotonic with Sharpe ratio, we can maximize Sharpe
            # and then scale to match benchmark volatility

            # Add all constraints
            self._apply_constraints(ef, allowed_weight_ranges, current_weights, prices, nav, cash_available )
            
            # First get max Sharpe portfolio
            ef.max_sharpe(risk_free_rate=rf)
            sharpe_weights = ef.clean_weights()
            sharpe_array = self._to_array(sharpe_weights)
            
            # If selector_k is requested, run a second pass that forces non-top-k weights to zero.
            selector_k = kwargs.get("selector_k", None)
            if selector_k is not None and selector_k <= (er > rf).sum():
                
                # set non-topk to have 0 as max allocation factor
                idx_sorted = np.argsort(-np.abs(sharpe_array))
                top_idx = set(idx_sorted[:selector_k])
                modified_ranges = allowed_weight_ranges.copy()
                for i in range(len(sharpe_array)):
                    if i not in top_idx:
                        modified_ranges[i] = [0.0, 0.0]

                # build a fresh EfficientFrontier to enforce zeros
                ef2 = EfficientFrontier(
                    expected_returns=er,
                    cov_matrix=cov,
                    weight_bounds=modified_ranges,
                )

                self._apply_constraints(ef2, modified_ranges, current_weights, prices, nav, cash_available)

                # Solve constrained max-sharpe on the top-k subset
                ef2.max_sharpe(risk_free_rate=rf)
                sharpe_weights2 = ef2.clean_weights()
                sharpe_array = self._to_array(sharpe_weights2)

            # Calculate portfolio volatility
            port_vol = float(np.sqrt(np.dot(sharpe_array, np.dot(cov, sharpe_array))))
            
            # Scale weights to match benchmark volatility
            # This preserves the Sharpe ratio while adjusting risk
            if port_vol > 0:
                scale = benchmark_vol / port_vol
                
                # Mix with risk-free asset to achieve target volatility
                if scale < 1.0:
                    # Reduce exposure (add cash/risk free) to hit target volatility
                    w_scaled = sharpe_array * scale
                    self._last_valid_weights = w_scaled
                    return w_scaled
                else:
                    # For scale > 1, we'd need leverage to hit higher volatility. Only allow if global weight bounds permit >1 exposures
                    if self.weight_bounds[1] > 1.0:
                        w_scaled = sharpe_array * min(scale, self.weight_bounds[1])
                        self._last_valid_weights = w_scaled
                        return w_scaled
                    else:
                        # cannot lever: return unlevered sharpe weights (best feasible)
                        self._last_valid_weights = sharpe_array
                        return sharpe_array
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Max Sharpe optimization failed: {e}. Using safe fallback.")
            
            # Try to use last valid weights first
            fallback_current = current_weights if current_weights is not None else self._last_valid_weights
            
            return self._get_safe_fallback_weights(
                n_assets=len(er),
                allowed_weight_ranges=allowed_weight_ranges,
                current_weights=fallback_current,
            )


class MaxQuadraticUtilityOptimizer(PortfolioOptimizer):
    """
    Maximize quadratic utility (expected return - risk_aversion * variance).
    Good for risk-averse investors.
    """
    
    def __init__(self, max_adv_pct: float, risk_aversion: int = 1, weight_bounds=(-1, 1), solver=None, **kwargs):
        """
        Args:
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            weight_bounds: Weight constraints
            solver: cvxpy solver
        """
        super().__init__(max_adv_pct, weight_bounds, solver)
        self.risk_aversion = risk_aversion
    
    def optimize(self, er: pd.Series, cov: pd.DataFrame,
                allowed_weight_ranges: np.ndarray,
                current_weights: np.ndarray,
                prices: np.ndarray,
                nav: float,
                cash_available: float,
                **kwargs) -> np.ndarray:
        """
        Find weights that maximize quadratic utility.
        
        U = μᵀw - (γ/2) * wᵀΣw
        where γ is risk aversion parameter.
        """
        try:
            ef = EfficientFrontier(
                expected_returns=er,
                cov_matrix=cov,
                weight_bounds=allowed_weight_ranges,
            )

            # Add all constraints
            self._apply_constraints(ef, allowed_weight_ranges, current_weights, prices, nav, cash_available)
            
            # Maximize quadratic utility
            ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
            
            # Clean and return weights
            cleaned_weights = ef.clean_weights()
            sharpe_array = self._to_array(cleaned_weights)

            # If selector_k is requested, run a second pass that forces non-top-k weights to zero.
            selector_k = kwargs.get("selector_k", None)
            if selector_k is not None and selector_k <= (er > rf).sum():

                # set non-topk to have 0 as max allocation factor
                idx_sorted = np.argsort(-np.abs(sharpe_array))
                top_idx = set(idx_sorted[:selector_k])
                modified_ranges = allowed_weight_ranges.copy()
                for i in range(len(sharpe_array)):
                    if i not in top_idx:
                        modified_ranges[i] = [0.0, 0.0]

                # build a fresh EfficientFrontier to enforce zeros
                ef2 = EfficientFrontier(
                    expected_returns=er,
                    cov_matrix=cov,
                    weight_bounds=modified_ranges,
                )
                
                self._apply_constraints(ef2, modified_ranges, current_weights, prices, nav, cash_available)

                # Solve constrained max-sharpe on the top-k subset
                ef2.max_quadratic_utility(risk_aversion=self.risk_aversion)
                sharpe_weights2 = ef2.clean_weights()
                sharpe_array = self._to_array(sharpe_weights2)
                
            self._last_valid_weights = sharpe_array
            return self._last_valid_weights
            
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Max Sharpe optimization failed: {e}. Using safe fallback.")
            
            # Try to use last valid weights first
            fallback_current = current_weights if current_weights is not None else self._last_valid_weights
            
            return self._get_safe_fallback_weights(
                n_assets=len(er),
                allowed_weight_ranges=allowed_weight_ranges,
                current_weights=fallback_current
            )


class EfficientRiskOptimizer(PortfolioOptimizer):
    """
    Target a specific risk level and maximize return.
    Useful for risk-targeted strategies.
    """
    
    def __init__(self, max_adv_pct: float, weight_bounds=(-1, 1), solver=None, target_volatility: float = 0.05, **kwargs ):
        """
        Args:
            target_volatility: Target portfolio volatility
            weight_bounds: Weight constraints
            solver: cvxpy solver
        """
        super().__init__( max_adv_pct, weight_bounds, solver)
        self.target_volatility = target_volatility
    
    def optimize(self, er: pd.Series, cov: pd.DataFrame,
                allowed_weight_ranges: np.ndarray,
                current_weights: np.ndarray, 
                prices: np.ndarray,
                nav: float,
                cash_available: float,
                **kwargs) -> np.ndarray:
        """
        Find weights that maximize return for given risk level.
        """
        try:
            ef = EfficientFrontier(
                expected_returns=er,
                cov_matrix=cov,
                weight_bounds=allowed_weight_ranges,
            )
            
            # Add all constraints
            self._apply_constraints(ef, allowed_weight_ranges, current_weights, prices, nav, cash_available)
            
            # Maximize return for target volatility
            ef.efficient_risk(target_volatility=self.target_volatility)
            
            # Clean and return weights
            cleaned_weights = ef.clean_weights()
            sharpe_array = self._to_array(cleaned_weights)

            # If selector_k is requested, run a second pass that forces non-top-k weights to zero.
            selector_k = kwargs.get("selector_k", None)
            if selector_k is not None and selector_k <= (er > rf).sum():

                # set non-topk to have 0 as max allocation factor
                idx_sorted = np.argsort(-np.abs(sharpe_array))
                top_idx = set(idx_sorted[:selector_k])
                modified_ranges = allowed_weight_ranges.copy()
                for i in range(len(sharpe_array)):
                    if i not in top_idx:
                        modified_ranges[i] = [0.0, 0.0]

                # build a fresh EfficientFrontier to enforce zeros
                ef2 = EfficientFrontier(
                    expected_returns=er,
                    cov_matrix=cov,
                    weight_bounds=modified_ranges,
                )
                
                self._apply_constraints(ef2, modified_ranges, current_weights, prices, nav, cash_available)

                # Solve constrained max-sharpe on the top-k subset
                ef2.efficient_risk(target_volatility=self.target_volatility)
                sharpe_weights2 = ef2.clean_weights()
                sharpe_array = self._to_array(sharpe_weights2)
                
            self._last_valid_weights = sharpe_array
            return self._last_valid_weights
            
        except (OptimizationError, Exception) as e:
            logger.warning(f"Max Sharpe optimization failed: {e}. Using safe fallback.")
            
            # Try to use last valid weights first
            fallback_current = current_weights if current_weights is not None else self._last_valid_weights
            
            return self._get_safe_fallback_weights(
                n_assets=len(er),
                allowed_weight_ranges=allowed_weight_ranges,
                current_weights=fallback_current,
            )

