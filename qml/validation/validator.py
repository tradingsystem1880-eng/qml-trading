"""
Validator
=========
Unified statistical validation interface.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidationResult:
    """Validation result container."""
    verdict: str  # "DEPLOY", "CAUTION", "REJECT"
    confidence_score: float
    p_value: float
    is_significant: bool
    reasons: List[str]
    sharpe_percentile: float = 0.0
    var_95: float = 0.0
    kill_switch_prob: float = 0.0


class Validator:
    """
    Unified validation interface.
    
    Wraps existing validation infrastructure:
    - Permutation tests
    - Monte Carlo simulation
    - Bootstrap confidence intervals
    
    Example:
        validator = Validator()
        result = validator.run(backtest_result)
    """
    
    def __init__(self, config=None, n_permutations: int = 1000):
        """Initialize validator."""
        self.config = config
        self.n_permutations = n_permutations
    
    def run(self, backtest_result, significance_level: float = 0.05) -> ValidationResult:
        """
        Run statistical validation.
        
        Args:
            backtest_result: Result from BacktestEngine
            significance_level: P-value threshold
            
        Returns:
            ValidationResult with verdict
        """
        if backtest_result.total_trades < 10:
            return ValidationResult(
                verdict="REJECT",
                confidence_score=0,
                p_value=1.0,
                is_significant=False,
                reasons=["Insufficient trades (< 10)"]
            )
        
        # Extract returns
        returns = backtest_result.trades["pnl_pct"].values
        
        # Run permutation test
        p_value, sharpe_pct = self._permutation_test(returns)
        is_significant = p_value < significance_level
        
        # Calculate confidence score
        score, reasons = self._calculate_confidence(
            backtest_result,
            p_value,
            is_significant
        )
        
        # Determine verdict
        if score >= 70 and is_significant:
            verdict = "DEPLOY"
        elif score >= 50:
            verdict = "CAUTION"
        else:
            verdict = "REJECT"
        
        return ValidationResult(
            verdict=verdict,
            confidence_score=score,
            p_value=p_value,
            is_significant=is_significant,
            reasons=reasons,
            sharpe_percentile=sharpe_pct
        )
    
    def _permutation_test(self, returns: np.ndarray) -> tuple:
        """Run permutation test for Sharpe ratio."""
        real_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
        
        permuted_sharpes = []
        for _ in range(self.n_permutations):
            shuffled = np.random.permutation(returns)
            perm_sharpe = np.mean(shuffled) / (np.std(shuffled) + 1e-10)
            permuted_sharpes.append(perm_sharpe)
        
        permuted_sharpes = np.array(permuted_sharpes)
        p_value = (permuted_sharpes >= real_sharpe).mean()
        percentile = (permuted_sharpes < real_sharpe).mean() * 100
        
        return p_value, percentile
    
    def _calculate_confidence(self, result, p_value: float, is_sig: bool) -> tuple:
        """Calculate confidence score and reasons."""
        score = 0
        reasons = []
        
        # Statistical significance (30%)
        if p_value < 0.01:
            score += 30
            reasons.append(f"Strong significance (p={p_value:.4f})")
        elif p_value < 0.05:
            score += 20
            reasons.append(f"Significant (p={p_value:.4f})")
        elif p_value < 0.10:
            score += 10
            reasons.append(f"Marginal (p={p_value:.4f})")
        else:
            reasons.append(f"NOT significant (p={p_value:.4f})")
        
        # Sharpe quality (25%)
        if result.sharpe_ratio > 1.5:
            score += 25
            reasons.append(f"Excellent Sharpe ({result.sharpe_ratio:.2f})")
        elif result.sharpe_ratio > 1.0:
            score += 18
            reasons.append(f"Good Sharpe ({result.sharpe_ratio:.2f})")
        elif result.sharpe_ratio > 0.5:
            score += 10
            reasons.append(f"Moderate Sharpe ({result.sharpe_ratio:.2f})")
        else:
            reasons.append(f"Poor Sharpe ({result.sharpe_ratio:.2f})")
        
        # Win rate (20%)
        if result.win_rate > 0.55:
            score += 20
            reasons.append(f"High win rate ({result.win_rate:.1%})")
        elif result.win_rate > 0.45:
            score += 12
            reasons.append(f"Moderate win rate ({result.win_rate:.1%})")
        else:
            reasons.append(f"Low win rate ({result.win_rate:.1%})")
        
        # Trade count (10%)
        if result.total_trades >= 50:
            score += 10
            reasons.append(f"Good sample size ({result.total_trades} trades)")
        elif result.total_trades >= 30:
            score += 5
            reasons.append(f"Moderate sample ({result.total_trades} trades)")
        else:
            reasons.append(f"Small sample ({result.total_trades} trades)")
        
        # Max drawdown (15%)
        if result.max_drawdown > -15:
            score += 15
            reasons.append(f"Low drawdown ({result.max_drawdown:.1f}%)")
        elif result.max_drawdown > -25:
            score += 8
            reasons.append(f"Moderate drawdown ({result.max_drawdown:.1f}%)")
        else:
            reasons.append(f"High drawdown ({result.max_drawdown:.1f}%)")
        
        return score, reasons
