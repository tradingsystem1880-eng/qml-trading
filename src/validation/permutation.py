"""
Permutation Test Validator
==========================
VRD 2.0 Module 3A: Statistical Edge Validation

Tests whether observed performance is statistically significant
by shuffling trade returns and recalculating metrics.

If the real Sharpe/profit is not significantly better than random
shuffles, the edge may be due to luck.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.validation.base import BaseValidator, ValidationResult, ValidationStatus


@dataclass
class PermutationConfig:
    """Configuration for permutation test."""
    n_permutations: int = 1000
    significance_level: float = 0.05  # p < 0.05 to pass
    metric: str = 'sharpe_ratio'  # Metric to test


@dataclass
class PermutationResult:
    """Result from permutation test."""
    p_value: float
    real_metric: float
    mean_permuted_metric: float
    std_permuted_metric: float
    n_permutations: int
    n_trades: int
    is_significant: bool
    permuted_metrics: List[float] = field(default_factory=list)


# Public API
__all__ = ['PermutationTest', 'PermutationConfig', 'PermutationResult', 'run_permutation_test']


class PermutationTest(BaseValidator):
    """
    Permutation test for statistical significance of trading edge.
    
    How it works:
    1. Calculate the real Sharpe ratio (or other metric) from actual trade sequence
    2. Shuffle trade returns N times (default 1000)
    3. Calculate Sharpe for each shuffled sequence
    4. p-value = proportion of shuffled Sharpes >= real Sharpe
    5. If p < 0.05, the edge is statistically significant
    
    Usage:
        validator = PermutationTest()
        result = validator.validate(backtest_result, trades=trades_list)
        
        if result.status == ValidationStatus.PASS:
            print("Edge is statistically significant!")
    """
    
    @property
    def name(self) -> str:
        return "permutation_test"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'n_permutations': 1000,
            'significance_level': 0.05,
            'metric': 'sharpe_ratio'
        }
    
    def validate(
        self,
        backtest_result: Dict[str, Any],
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationResult:
        """
        Run permutation test on backtest results.
        
        Args:
            backtest_result: Results from BacktestEngine
            trades: List of trade dictionaries with 'pnl_pct' field
        
        Returns:
            ValidationResult with p-value and interpretation
        """
        # Extract returns from trades
        if trades is None:
            trades = backtest_result.get('trades', [])
        
        if not trades:
            return ValidationResult(
                validator_name=self.name,
                status=ValidationStatus.ERROR,
                interpretation="No trades to analyze"
            )
        
        # Get returns array
        returns = []
        for trade in trades:
            if isinstance(trade, dict):
                pnl = trade.get('pnl_pct')
            else:
                pnl = getattr(trade, 'pnl_pct', None)
            
            if pnl is not None:
                returns.append(pnl)
        
        if len(returns) < 10:
            return ValidationResult(
                validator_name=self.name,
                status=ValidationStatus.WARN,
                interpretation=f"Insufficient trades ({len(returns)}) for reliable permutation test"
            )
        
        returns = np.array(returns)
        
        # Calculate real metric
        real_sharpe = self._calculate_sharpe(returns)
        
        # Run permutations
        n_perms = self.config.get('n_permutations', 1000)
        perm_sharpes = []
        
        for _ in range(n_perms):
            shuffled = np.random.permutation(returns)
            perm_sharpes.append(self._calculate_sharpe(shuffled))
        
        perm_sharpes = np.array(perm_sharpes)
        
        # Calculate p-value (proportion of permuted >= real)
        p_value = np.mean(perm_sharpes >= real_sharpe)
        
        # Determine status
        alpha = self.config.get('significance_level', 0.05)
        
        if p_value < alpha:
            status = ValidationStatus.PASS
            interpretation = f"Edge is statistically significant (p={p_value:.4f} < {alpha})"
        elif p_value < alpha * 2:
            status = ValidationStatus.WARN
            interpretation = f"Edge is marginally significant (p={p_value:.4f})"
        else:
            status = ValidationStatus.FAIL
            interpretation = f"Edge is NOT statistically significant (p={p_value:.4f} >= {alpha})"
        
        return ValidationResult(
            validator_name=self.name,
            status=status,
            metrics={
                'real_sharpe': round(real_sharpe, 4),
                'mean_permuted_sharpe': round(np.mean(perm_sharpes), 4),
                'std_permuted_sharpe': round(np.std(perm_sharpes), 4),
                'n_permutations': n_perms,
                'n_trades': len(returns)
            },
            p_value=round(p_value, 4),
            confidence=round(1 - p_value, 4),
            interpretation=interpretation,
            details={
                'percentile_rank': round(np.mean(real_sharpe > perm_sharpes) * 100, 1)
            }
        )
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns array."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns)
