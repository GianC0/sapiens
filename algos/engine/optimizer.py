"""
MaximumSharpeRatioOptimizer
===========================

Identical maths as the original Lean version, but expressed without
Algorithm-Framework references.  Returns *raw* weights which the strategy then
post-processes (top-k filter, gross/net scaling, etc.).
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import pinv


class MaxSharpeRatioOptimizer:
    def __init__(self, risk_free: float = 0.0):
        self.rf = risk_free

    # ------------------------------------------------------------------ #
    def optimize(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        mu_excess = mu - self.rf
        inv = pinv(cov)
        numer = inv @ mu_excess
        denom = mu_excess.T @ inv @ mu_excess
        if denom == 0:
            return np.zeros_like(mu)
        return numer / denom
