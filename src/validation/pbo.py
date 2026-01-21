"""
PBO Calculator
==============
Probability of Backtest Overfitting (Lopez de Prado)

Splits data into combinatorial subsets, trains on each,
and measures how often in-sample best != out-of-sample best.

Reference: "The Probability of Backtest Overfitting" (Bailey et al., 2014)
"""

import numpy as np
from scipy import stats
from itertools import combinations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.validation.base import BaseValidator, ValidationResult, ValidationStatus


@dataclass
class PBOResult:
    """Result from PBO calculation."""
    pbo: float  # Probability of Backtest Overfitting (0-1)
    n_combinations: int
    is_oos_rank_correlation: float  # Spearman correlation between IS and OOS ranks
    interpretation: str
    is_overfit: bool


class PBOCalculator(BaseValidator):
    """
    Probability of Backtest Overfitting calculator.

    Uses Combinatorially Symmetric Cross-Validation (CSCV):
    1. Split data into S partitions
    2. For all C(S, S/2) combinations, one half is train, other is test
    3. Compare IS and OOS rankings
    4. PBO = proportion where IS-best underperforms OOS median

    A PBO > 0.5 suggests overfitting is likely.

    Usage:
        pbo = PBOCalculator(config={'n_partitions': 16})
        result = pbo.validate(backtest_result, trades=trades)
        print(f"PBO: {result.metrics['pbo']:.2%}")
    """

    @property
    def name(self) -> str:
        return "pbo"

    def get_default_config(self) -> Dict[str, Any]:
        return {
            'n_partitions': 16,  # Must be even, typically 16
            'overfit_threshold': 0.5,  # PBO > 0.5 = overfit
        }

    def validate(
        self,
        backtest_result: Dict[str, Any],
        trades: Optional[List[Dict[str, Any]]] = None,
        returns_matrix: Optional[np.ndarray] = None
    ) -> ValidationResult:
        """
        Calculate PBO from strategy returns.

        Args:
            backtest_result: Results from BacktestEngine
            trades: List of trade dictionaries with 'pnl_pct' field
            returns_matrix: Optional pre-computed returns matrix (n_periods, n_strategies)
                           If not provided, uses single strategy returns from trades

        Returns:
            ValidationResult with PBO and related metrics
        """
        # Get returns
        if returns_matrix is not None:
            # Multi-strategy mode
            returns = returns_matrix
        else:
            # Single strategy mode - extract from trades
            if trades is None:
                trades = backtest_result.get('trades', [])

            if not trades:
                return ValidationResult(
                    validator_name=self.name,
                    status=ValidationStatus.ERROR,
                    interpretation="No trades to analyze"
                )

            # Get returns array
            pnl_list = []
            for trade in trades:
                if isinstance(trade, dict):
                    pnl = trade.get('pnl_pct')
                else:
                    pnl = getattr(trade, 'pnl_pct', None)
                if pnl is not None:
                    pnl_list.append(pnl)

            if len(pnl_list) < 20:
                return ValidationResult(
                    validator_name=self.name,
                    status=ValidationStatus.WARN,
                    interpretation=f"Insufficient trades ({len(pnl_list)}) for PBO analysis"
                )

            # For single strategy, create synthetic "strategies" by varying parameters
            # This simulates what would happen if we had tried multiple parameter sets
            returns = self._create_synthetic_strategies(np.array(pnl_list))

        # Run CSCV calculation
        pbo_result = self.calculate(returns)

        # Determine status
        threshold = self.config.get('overfit_threshold', 0.5)

        if pbo_result.pbo < 0.25:
            status = ValidationStatus.PASS
            interpretation = f"Low overfitting risk (PBO: {pbo_result.pbo:.1%})"
        elif pbo_result.pbo < threshold:
            status = ValidationStatus.WARN
            interpretation = f"Moderate overfitting risk (PBO: {pbo_result.pbo:.1%})"
        else:
            status = ValidationStatus.FAIL
            interpretation = f"High overfitting risk (PBO: {pbo_result.pbo:.1%})"

        return ValidationResult(
            validator_name=self.name,
            status=status,
            metrics={
                'pbo': round(pbo_result.pbo, 4),
                'is_oos_correlation': round(pbo_result.is_oos_rank_correlation, 4),
                'n_combinations': pbo_result.n_combinations,
                'is_overfit': pbo_result.is_overfit,
            },
            confidence=round(1 - pbo_result.pbo, 4),
            interpretation=interpretation,
            details={
                'n_partitions': self.config.get('n_partitions', 16),
                'overfit_threshold': threshold
            }
        )

    def calculate(
        self,
        returns_matrix: np.ndarray,
    ) -> PBOResult:
        """
        Calculate PBO from strategy returns matrix.

        Args:
            returns_matrix: Shape (n_periods, n_strategies)
                           Each column is returns from a different strategy/parameter set

        Returns:
            PBOResult with PBO value and diagnostics
        """
        n_periods, n_strategies = returns_matrix.shape
        n_partitions = self.config.get('n_partitions', 16)

        # Ensure even number of partitions
        if n_partitions % 2 != 0:
            n_partitions += 1

        # Split into partitions
        partition_size = n_periods // n_partitions
        if partition_size < 1:
            # Not enough data for requested partitions
            n_partitions = n_periods
            partition_size = 1

        # Create partitioned returns (sum within each partition)
        partitioned = np.zeros((n_partitions, n_strategies))
        for i in range(n_partitions):
            start = i * partition_size
            end = min((i + 1) * partition_size, n_periods)
            partitioned[i] = returns_matrix[start:end].sum(axis=0)

        # Generate all C(S, S/2) combinations
        half = n_partitions // 2
        partition_indices = list(range(n_partitions))
        all_combinations = list(combinations(partition_indices, half))

        # For each combination, calculate IS and OOS performance
        overfit_count = 0
        is_ranks_all = []
        oos_ranks_all = []

        for train_idx in all_combinations:
            test_idx = tuple(i for i in partition_indices if i not in train_idx)

            # In-sample (train) performance
            is_returns = partitioned[list(train_idx)].sum(axis=0)

            # Out-of-sample (test) performance
            oos_returns = partitioned[list(test_idx)].sum(axis=0)

            # Rank strategies by IS performance (higher = better)
            is_ranks = stats.rankdata(-is_returns)  # Negative for descending
            oos_ranks = stats.rankdata(-oos_returns)

            is_ranks_all.append(is_ranks)
            oos_ranks_all.append(oos_ranks)

            # Find IS-best strategy
            is_best_idx = np.argmax(is_returns)

            # Check if IS-best underperforms OOS median
            oos_median = np.median(oos_returns)
            if oos_returns[is_best_idx] < oos_median:
                overfit_count += 1

        n_combinations = len(all_combinations)
        pbo = overfit_count / n_combinations if n_combinations > 0 else 0.0

        # Calculate rank correlation between IS and OOS (average across combinations)
        correlations = []
        for is_r, oos_r in zip(is_ranks_all, oos_ranks_all):
            corr, _ = stats.spearmanr(is_r, oos_r)
            if not np.isnan(corr):
                correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else 0.0

        threshold = self.config.get('overfit_threshold', 0.5)

        return PBOResult(
            pbo=pbo,
            n_combinations=n_combinations,
            is_oos_rank_correlation=avg_correlation,
            interpretation=f"PBO={pbo:.1%}, IS-OOS correlation={avg_correlation:.2f}",
            is_overfit=pbo > threshold
        )

    def _create_synthetic_strategies(
        self,
        returns: np.ndarray,
        n_strategies: int = 8
    ) -> np.ndarray:
        """
        Create synthetic strategy returns for single-strategy PBO analysis.

        Simulates what would happen if we had tried different parameter sets
        by adding noise and shifting the returns.

        Args:
            returns: Original strategy returns (1D array)
            n_strategies: Number of synthetic strategies to create

        Returns:
            Matrix of shape (n_periods, n_strategies)
        """
        n_periods = len(returns)
        matrix = np.zeros((n_periods, n_strategies))

        # First strategy is the original
        matrix[:, 0] = returns

        # Create variations
        for i in range(1, n_strategies):
            # Add noise proportional to returns volatility
            noise_scale = np.std(returns) * 0.3 * (i / n_strategies)
            noise = np.random.randn(n_periods) * noise_scale

            # Slight lag/lead shift
            shift = np.random.randint(-2, 3)
            shifted = np.roll(returns, shift)

            # Combine
            matrix[:, i] = shifted + noise

        return matrix
