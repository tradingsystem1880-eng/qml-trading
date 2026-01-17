"""
Bootstrap Resampling Validator
==============================
VRD 2.0 Module 3C: Confidence Interval Estimation

Uses bootstrap resampling to estimate confidence intervals
on performance metrics without assuming normal distribution.

Answers: "What's the range of expected performance?"
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.validation.base import BaseValidator, ValidationResult, ValidationStatus


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap resampling."""
    n_resamples: int = 1000
    confidence_level: float = 0.95


class BootstrapResample(BaseValidator):
    """
    Bootstrap resampling for confidence interval estimation.
    
    How it works:
    1. Resample trades with replacement N times
    2. Calculate metric (Sharpe, profit, etc.) for each resample
    3. Use percentiles to build confidence intervals
    4. No assumption of normal distribution required
    
    This answers: "Given sampling variability, what's the realistic
    range of performance I can expect?"
    
    Usage:
        bootstrap = BootstrapResample()
        result = bootstrap.validate(backtest_result, trades=trades_list)
        
        print(f"Sharpe 95% CI: {result.metrics['sharpe_ci']}")
    """
    
    @property
    def name(self) -> str:
        return "bootstrap"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'n_resamples': 1000,
            'confidence_level': 0.95
        }
    
    def validate(
        self,
        backtest_result: Dict[str, Any],
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationResult:
        """
        Run bootstrap resampling on backtest results.
        
        Args:
            backtest_result: Results from BacktestEngine
            trades: List of trade dictionaries with 'pnl_pct' field
        
        Returns:
            ValidationResult with confidence intervals on key metrics
        """
        # Extract returns
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
                interpretation=f"Insufficient trades ({len(returns)}) for bootstrap"
            )
        
        returns = np.array(returns)
        
        # Configuration
        n_resamples = self.config.get('n_resamples', 1000)
        conf_level = self.config.get('confidence_level', 0.95)
        alpha = (1 - conf_level) / 2  # Two-tailed
        
        # Bootstrap distributions
        bootstrap_sharpes = []
        bootstrap_means = []
        bootstrap_win_rates = []
        bootstrap_profit_factors = []
        
        for _ in range(n_resamples):
            # Resample with replacement
            sample = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate metrics
            bootstrap_sharpes.append(self._calculate_sharpe(sample))
            bootstrap_means.append(np.mean(sample))
            
            wins = np.sum(sample > 0)
            bootstrap_win_rates.append(wins / len(sample) * 100)
            
            gains = np.sum(sample[sample > 0])
            losses = np.abs(np.sum(sample[sample < 0]))
            pf = gains / losses if losses > 0 else float('inf')
            bootstrap_profit_factors.append(pf if pf != float('inf') else 10.0)
        
        # Calculate confidence intervals
        def ci(data):
            lower = np.percentile(data, alpha * 100)
            upper = np.percentile(data, (1 - alpha) * 100)
            return [round(lower, 4), round(upper, 4)]
        
        sharpe_ci = ci(bootstrap_sharpes)
        mean_ci = ci(bootstrap_means)
        winrate_ci = ci(bootstrap_win_rates)
        pf_ci = ci(bootstrap_profit_factors)
        
        # Determine status based on lower bound of Sharpe CI
        if sharpe_ci[0] > 0:
            status = ValidationStatus.PASS
            interpretation = f"Strategy is profitable with {conf_level:.0%} confidence (Sharpe CI: {sharpe_ci})"
        elif sharpe_ci[1] > 0:
            status = ValidationStatus.WARN
            interpretation = f"Profitability uncertain (Sharpe CI spans zero: {sharpe_ci})"
        else:
            status = ValidationStatus.FAIL
            interpretation = f"Strategy is unprofitable with {conf_level:.0%} confidence (Sharpe CI: {sharpe_ci})"
        
        return ValidationResult(
            validator_name=self.name,
            status=status,
            metrics={
                'sharpe_point': round(self._calculate_sharpe(returns), 4),
                'sharpe_ci': sharpe_ci,
                'mean_return_ci': mean_ci,
                'win_rate_ci': winrate_ci,
                'profit_factor_ci': pf_ci,
                'n_resamples': n_resamples,
                'n_trades': len(returns)
            },
            confidence=conf_level,
            interpretation=interpretation,
            details={
                'sharpe_std': round(np.std(bootstrap_sharpes), 4),
                'mean_return_std': round(np.std(bootstrap_means), 4)
            }
        )
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns array."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns)
