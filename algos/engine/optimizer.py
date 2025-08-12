"""
Maximum-Sharpe-Ratio optimiser (no constraints).
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import pinv


class MaxSharpeRatioOptimizer:
    def __init__(self, risk_free: float = 0.0):
        self.rf = risk_free

    def optimize(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------  
        mu  : expected returns  (shape N)
        cov : covariance matrix (NxN)

        Returns
        -------
        w   : weights summing to 1 (long-only can be enforced later)
        """
        mu_excess = mu - self.rf
        inv = pinv(cov)                       # Moore-Penrose in case cov is singular
        numer = inv @ mu_excess
        denom = mu_excess.T @ numer           # == mu^T Σ⁻¹ mu
        return numer / denom if denom != 0 else np.zeros_like(mu)
