"""
Statistical Validation for Trading Strategies
=============================================
Based on López de Prado (2014) - The Deflated Sharpe Ratio

Provides:
- Deflated Sharpe Ratio (corrects for multiple testing)
- Minimum Track Record Length
- Strategy validation with clear pass/fail criteria

References:
- Bailey & López de Prado (2014) - "The Deflated Sharpe Ratio"
- Bailey & López de Prado (2012) - "The Sharpe Ratio Efficient Frontier"
- López de Prado (2018) - "Advances in Financial Machine Learning"
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.stats import norm


@dataclass
class StatisticalValidation:
    """Result of statistical validation."""
    deflated_sharpe_ratio: float
    statistically_significant: bool
    minimum_trades_needed: int
    has_sufficient_trades: bool
    validation_summary: str
    p_value: float
    confidence_level: float

    # Additional metrics
    observed_sharpe: float
    expected_max_sharpe: float
    num_trials: int
    num_observations: int


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    num_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> Tuple[float, float]:
    """
    Calculate the Deflated Sharpe Ratio.

    Corrects for:
    - Multiple testing (trying many parameter combinations)
    - Non-normal returns (skewness, kurtosis)
    - Short track record

    Reference: Bailey & López de Prado (2014)

    Args:
        observed_sharpe: The Sharpe ratio from backtest
        num_trials: Number of parameter combinations tested
        num_observations: Number of trades/periods
        skewness: Return distribution skewness (0 for normal)
        kurtosis: Return distribution kurtosis (3 for normal)

    Returns:
        Tuple of (DSR probability, expected max Sharpe from random)
    """
    if num_observations < 10 or num_trials < 1:
        return 0.0, 0.0

    # Expected maximum Sharpe from random chance given N trials
    # E[max(Z_1...Z_N)] approximation using order statistics
    # For large N: E[max] ≈ sqrt(2 * ln(N))
    if num_trials > 1:
        euler_mascheroni = 0.5772156649
        expected_max_sharpe = (
            (1 - euler_mascheroni) * norm.ppf(1 - 1/num_trials) +
            euler_mascheroni * norm.ppf(1 - 1/(num_trials * np.e))
        ) * np.sqrt(1 / num_observations)
    else:
        expected_max_sharpe = 0.0

    # Standard error of Sharpe, adjusted for non-normality
    # From Lo (2002) and Opdyke (2007)
    sr_variance = (
        1 - skewness * observed_sharpe +
        ((kurtosis - 1) / 4) * observed_sharpe**2
    ) / max(num_observations - 1, 1)

    sr_std = np.sqrt(max(sr_variance, 1e-10))

    # Calculate DSR (probability that true Sharpe > 0)
    if sr_std > 0:
        z_score = (observed_sharpe - expected_max_sharpe) / sr_std
        dsr = norm.cdf(z_score)
    else:
        dsr = 0.5  # Indeterminate

    return float(dsr), float(expected_max_sharpe)


def minimum_track_record_length(
    observed_sharpe: float,
    target_sharpe: float = 0.0,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> int:
    """
    Calculate minimum number of observations needed for statistical significance.

    Reference: Bailey & López de Prado (2012) - "The Sharpe Ratio Efficient Frontier"

    Args:
        observed_sharpe: The observed Sharpe ratio
        target_sharpe: The target to beat (usually 0)
        confidence: Confidence level (0.95 = 95%)
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis

    Returns:
        Minimum number of trades/periods needed
    """
    if observed_sharpe <= target_sharpe:
        return int(1e9)  # Effectively infinite

    z = norm.ppf(confidence)

    # Variance adjustment for non-normality
    sr_diff = observed_sharpe - target_sharpe

    numerator = z**2 * (
        1 - skewness * observed_sharpe +
        ((kurtosis - 1) / 4) * observed_sharpe**2
    )

    min_length = numerator / (sr_diff**2)

    return max(int(np.ceil(min_length)), 30)  # Minimum 30 for CLT


def estimate_return_moments(returns: np.ndarray) -> Tuple[float, float]:
    """
    Estimate skewness and kurtosis from return series.

    Args:
        returns: Array of R-multiple returns

    Returns:
        Tuple of (skewness, kurtosis)
    """
    if len(returns) < 10:
        return 0.0, 3.0  # Assume normal

    # Use sample moments
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)

    if std < 1e-10:
        return 0.0, 3.0

    # Standardized moments
    z = (returns - mean) / std
    skewness = float(np.mean(z**3))
    kurtosis = float(np.mean(z**4))

    return skewness, kurtosis


def validate_strategy(
    metrics: Dict,
    num_trials: int,
    returns: Optional[np.ndarray] = None,
    confidence: float = 0.95,
) -> StatisticalValidation:
    """
    Comprehensive statistical validation of strategy results.

    Args:
        metrics: Dict with keys: sharpe, pattern_count, profit_factor, etc.
        num_trials: Number of parameter combinations tested (optimization iterations)
        returns: Optional array of R-multiple returns for moment estimation
        confidence: Confidence level for significance testing

    Returns:
        StatisticalValidation with all results and recommendations
    """
    sharpe = metrics.get('sharpe', 0)
    n_trades = metrics.get('pattern_count', metrics.get('total_trades', 0))

    # Estimate moments from returns if available
    if returns is not None and len(returns) > 10:
        skewness, kurtosis = estimate_return_moments(returns)
    else:
        skewness = metrics.get('returns_skewness', 0.0)
        kurtosis = metrics.get('returns_kurtosis', 3.0)

    # Calculate DSR
    dsr, expected_max = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        num_trials=num_trials,
        num_observations=n_trades,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # Calculate p-value (probability of observing this Sharpe by chance)
    p_value = 1.0 - dsr

    # Minimum track record
    min_trades = minimum_track_record_length(
        observed_sharpe=sharpe,
        confidence=confidence,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # Determine validation summary
    if dsr > 0.95 and n_trades >= min_trades:
        summary = "PASS - Statistically significant edge"
    elif dsr > 0.95 and n_trades < min_trades:
        summary = f"MARGINAL - Need {min_trades - n_trades} more trades"
    elif dsr > 0.80:
        summary = "WEAK - Some evidence of edge, needs more data"
    elif dsr > 0.50:
        summary = "INCONCLUSIVE - Cannot distinguish from random"
    else:
        summary = "FAIL - Likely overfitted or no edge"

    return StatisticalValidation(
        deflated_sharpe_ratio=dsr,
        statistically_significant=dsr > 0.95,
        minimum_trades_needed=min_trades,
        has_sufficient_trades=n_trades >= min_trades,
        validation_summary=summary,
        p_value=p_value,
        confidence_level=confidence,
        observed_sharpe=sharpe,
        expected_max_sharpe=expected_max,
        num_trials=num_trials,
        num_observations=n_trades,
    )


def print_validation_report(validation: StatisticalValidation) -> None:
    """Print formatted validation report."""
    print("\n" + "="*60)
    print("STATISTICAL VALIDATION REPORT")
    print("="*60)

    print(f"\nObserved Sharpe Ratio: {validation.observed_sharpe:.4f}")
    print(f"Expected Max (Random): {validation.expected_max_sharpe:.4f}")
    print(f"Num Trials Tested:     {validation.num_trials}")
    print(f"Num Observations:      {validation.num_observations}")

    print(f"\n--- Deflated Sharpe Ratio ---")
    print(f"DSR (P(true SR > 0)): {validation.deflated_sharpe_ratio:.3f}")
    print(f"p-value:              {validation.p_value:.4f}")
    print(f"Significant (>0.95):  {'Yes' if validation.statistically_significant else 'No'}")

    print(f"\n--- Track Record Requirements ---")
    print(f"Minimum Trades Needed: {validation.minimum_trades_needed}")
    print(f"Current Trades:        {validation.num_observations}")
    print(f"Sufficient:            {'Yes' if validation.has_sufficient_trades else 'No'}")

    print(f"\n{'='*60}")
    print(f"VERDICT: {validation.validation_summary}")
    print(f"{'='*60}\n")
