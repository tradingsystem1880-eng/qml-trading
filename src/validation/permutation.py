"""
Permutation Testing for Strategy Validation
============================================
Tests whether strategy performance is due to skill or luck
by comparing actual results against random permutations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class PermutationResult:
    """Result of permutation testing."""
    
    # Actual strategy performance
    actual_sharpe: float
    actual_max_dd: float
    actual_profit_factor: float
    actual_total_return: float
    
    # Statistical significance
    sharpe_p_value: float      # P(random > actual)
    max_dd_p_value: float      # P(random has lower DD)
    sharpe_percentile: float   # Percentile of actual vs random
    
    # Distribution data (for visualization)
    permutation_sharpes: np.ndarray
    permutation_dds: np.ndarray
    
    # Configuration
    n_permutations: int
    
    @property
    def is_significant_95(self) -> bool:
        """Is the strategy significant at 95% level?"""
        return self.sharpe_p_value < 0.05
    
    @property
    def is_significant_99(self) -> bool:
        """Is the strategy significant at 99% level?"""
        return self.sharpe_p_value < 0.01


class PermutationTest:
    """
    Permutation Testing for Strategy Evaluation.
    
    Tests skill vs luck by randomly shuffling trade sequence 10,000+ times
    while preserving individual trade outcomes.
    
    Answers: "What's the probability that random ordering of these
    specific trades would produce results as good or better?"
    """
    
    def __init__(
        self,
        n_permutations: int = 10000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize permutation test.
        
        Args:
            n_permutations: Number of random permutations (min 10,000 recommended)
            random_seed: Optional seed for reproducibility
        """
        if n_permutations < 1000:
            logger.warning(
                f"n_permutations={n_permutations} is low. "
                "Consider using at least 10,000 for reliable p-values."
            )
        
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_seed)
        
        logger.info(f"PermutationTest initialized with {n_permutations} permutations")
    
    def run(self, trade_returns: np.ndarray) -> PermutationResult:
        """
        Run permutation test on trade returns.
        
        Args:
            trade_returns: Array of individual trade returns (percentages or ratios)
            
        Returns:
            PermutationResult with p-values and distributions
        """
        trade_returns = np.asarray(trade_returns).flatten()
        n_trades = len(trade_returns)
        
        if n_trades < 10:
            raise ValueError(
                f"Insufficient trades ({n_trades}) for permutation testing. "
                "Need at least 10 trades."
            )
        
        logger.info(f"Running permutation test on {n_trades} trades")
        
        # Calculate actual strategy metrics
        actual_sharpe = self._calculate_sharpe(trade_returns)
        actual_max_dd = self._calculate_max_drawdown(trade_returns)
        actual_pf = self._calculate_profit_factor(trade_returns)
        actual_return = float(np.sum(trade_returns))
        
        # Run permutations
        perm_sharpes = np.zeros(self.n_permutations)
        perm_dds = np.zeros(self.n_permutations)
        
        for i in range(self.n_permutations):
            shuffled = self._shuffle_trades(trade_returns)
            perm_sharpes[i] = self._calculate_sharpe(shuffled)
            perm_dds[i] = self._calculate_max_drawdown(shuffled)
        
        # Calculate p-values
        # Sharpe: What fraction of random orderings have higher Sharpe?
        sharpe_p_value = np.mean(perm_sharpes >= actual_sharpe)
        
        # Max DD: What fraction of random orderings have lower (better) drawdown?
        max_dd_p_value = np.mean(perm_dds <= actual_max_dd)
        
        # Percentile: Where does actual fall in the distribution?
        sharpe_percentile = 100 * np.mean(perm_sharpes < actual_sharpe)
        
        result = PermutationResult(
            actual_sharpe=actual_sharpe,
            actual_max_dd=actual_max_dd,
            actual_profit_factor=actual_pf,
            actual_total_return=actual_return,
            sharpe_p_value=sharpe_p_value,
            max_dd_p_value=max_dd_p_value,
            sharpe_percentile=sharpe_percentile,
            permutation_sharpes=perm_sharpes,
            permutation_dds=perm_dds,
            n_permutations=self.n_permutations,
        )
        
        logger.info(
            f"Permutation test complete: "
            f"Sharpe p-value={sharpe_p_value:.4f}, "
            f"percentile={sharpe_percentile:.1f}%"
        )
        
        return result
    
    def _shuffle_trades(self, returns: np.ndarray) -> np.ndarray:
        """Randomly shuffle trade sequence."""
        shuffled = returns.copy()
        self.rng.shuffle(shuffled)
        return shuffled
    
    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 1.0
    ) -> float:
        """
        Calculate Sharpe ratio of trade returns.
        
        Args:
            returns: Trade returns
            risk_free_rate: Risk-free rate (default 0)
            annualization_factor: Annualization factor (default 1 for trade-level)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        
        return float(mean_return / std_return * np.sqrt(annualization_factor))
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from trade sequence.
        
        Args:
            returns: Trade returns (percentages)
            
        Returns:
            Maximum drawdown as positive percentage
        """
        if len(returns) == 0:
            return 0.0
        
        # Build equity curve (cumulative returns)
        equity = np.cumprod(1 + returns / 100)  # Assuming percentage returns
        
        # Running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Drawdown at each point
        drawdowns = (running_max - equity) / running_max * 100
        
        return float(np.max(drawdowns))
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            returns: Trade returns
            
        Returns:
            Profit factor (> 1 is profitable)
        """
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return float(gross_profit / gross_loss)
    
    def generate_report(self, result: PermutationResult) -> str:
        """
        Generate text report of permutation test results.
        
        Args:
            result: PermutationResult to report
            
        Returns:
            Formatted report string
        """
        # Determine significance verdict
        if result.sharpe_p_value < 0.01:
            verdict = "HIGHLY SIGNIFICANT (p < 0.01) - Strong evidence of skill"
        elif result.sharpe_p_value < 0.05:
            verdict = "SIGNIFICANT (p < 0.05) - Evidence of skill"
        elif result.sharpe_p_value < 0.10:
            verdict = "MARGINALLY SIGNIFICANT (p < 0.10) - Weak evidence"
        else:
            verdict = "NOT SIGNIFICANT (p >= 0.10) - Cannot distinguish from luck"
        
        lines = [
            "=" * 60,
            "PERMUTATION TEST RESULTS",
            "=" * 60,
            "",
            f"Permutations: {result.n_permutations:,}",
            "",
            "Actual Strategy Performance:",
            f"  - Sharpe Ratio: {result.actual_sharpe:.4f}",
            f"  - Max Drawdown: {result.actual_max_dd:.2f}%",
            f"  - Profit Factor: {result.actual_profit_factor:.2f}",
            f"  - Total Return: {result.actual_total_return:.2f}%",
            "",
            "Statistical Significance:",
            f"  - Sharpe p-value: {result.sharpe_p_value:.4f}",
            f"  - Max DD p-value: {result.max_dd_p_value:.4f}",
            f"  - Percentile: {result.sharpe_percentile:.1f}th",
            "",
            "Distribution Summary:",
            f"  - Random Sharpe Mean: {np.mean(result.permutation_sharpes):.4f}",
            f"  - Random Sharpe Std: {np.std(result.permutation_sharpes):.4f}",
            f"  - Random Max DD Mean: {np.mean(result.permutation_dds):.2f}%",
            "",
            "=" * 60,
            f"VERDICT: {verdict}",
            "=" * 60,
            "",
            "Interpretation:",
            f"  This strategy is in the {result.sharpe_percentile:.1f}th percentile",
            f"  of {result.n_permutations:,} random permutations.",
            "",
            "  A p-value of {:.4f} means there is a {:.2f}% chance".format(
                result.sharpe_p_value, result.sharpe_p_value * 100
            ),
            "  that random trade sequencing would produce an equal",
            "  or better Sharpe ratio.",
            "=" * 60,
        ]
        
        return "\n".join(lines)


def run_permutation_test(
    trade_returns: np.ndarray,
    n_permutations: int = 10000,
    seed: Optional[int] = None
) -> PermutationResult:
    """
    Convenience function to run permutation test.
    
    Args:
        trade_returns: Array of trade returns
        n_permutations: Number of permutations
        seed: Random seed
        
    Returns:
        PermutationResult
    """
    test = PermutationTest(n_permutations=n_permutations, random_seed=seed)
    return test.run(trade_returns)
