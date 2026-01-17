"""
Monte Carlo Simulator
=====================
VRD 2.0 Module 3B: Risk Analysis via Simulation

Simulates thousands of equity paths by resampling trades
to understand:
- Value at Risk (VaR)
- Conditional VaR (CVaR / Expected Shortfall)
- Risk of Ruin probability
- Confidence intervals on final equity
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.validation.base import BaseValidator, ValidationResult, ValidationStatus


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 1000
    n_trades_per_sim: int = 100  # Or use actual trade count
    initial_capital: float = 10000.0
    var_confidence: float = 0.95  # VaR at 95% confidence
    risk_of_ruin_threshold: float = 0.5  # 50% drawdown = ruin


class MonteCarloSim(BaseValidator):
    """
    Monte Carlo simulation for risk analysis.
    
    How it works:
    1. Take actual trade returns from backtest
    2. Resample (with replacement) to create N simulated equity paths
    3. Calculate risk metrics across all simulations:
       - Value at Risk (VaR): Worst expected loss at X% confidence
       - CVaR: Average of losses beyond VaR
       - Risk of Ruin: Probability of hitting ruin threshold
       - Drawdown distribution
    
    Usage:
        sim = MonteCarloSim()
        result = sim.validate(backtest_result, trades=trades_list)
        
        print(f"VaR (95%): {result.metrics['var_95']:.2f}%")
        print(f"Risk of Ruin: {result.metrics['risk_of_ruin']:.2%}")
    """
    
    @property
    def name(self) -> str:
        return "monte_carlo"
    
    def get_default_config(self) -> Dict[str, Any]:
        return {
            'n_simulations': 1000,
            'n_trades_per_sim': None,  # None = use actual trade count
            'initial_capital': 10000.0,
            'var_confidence': 0.95,
            'risk_of_ruin_threshold': 0.5
        }
    
    def validate(
        self,
        backtest_result: Dict[str, Any],
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationResult:
        """
        Run Monte Carlo simulation on backtest results.
        
        Args:
            backtest_result: Results from BacktestEngine
            trades: List of trade dictionaries with 'pnl_pct' field
        
        Returns:
            ValidationResult with VaR, CVaR, and risk metrics
        """
        # Extract returns
        if trades is None:
            trades = backtest_result.get('trades', [])
        
        if not trades:
            return ValidationResult(
                validator_name=self.name,
                status=ValidationStatus.ERROR,
                interpretation="No trades to simulate"
            )
        
        # Get returns array
        returns = []
        for trade in trades:
            if isinstance(trade, dict):
                pnl = trade.get('pnl_pct')
            else:
                pnl = getattr(trade, 'pnl_pct', None)
            
            if pnl is not None:
                returns.append(pnl / 100)  # Convert % to decimal
        
        if len(returns) < 5:
            return ValidationResult(
                validator_name=self.name,
                status=ValidationStatus.WARN,
                interpretation=f"Insufficient trades ({len(returns)}) for Monte Carlo"
            )
        
        returns = np.array(returns)
        
        # Configuration
        n_sims = self.config.get('n_simulations', 1000)
        n_trades = self.config.get('n_trades_per_sim') or len(returns)
        initial_capital = self.config.get('initial_capital', 10000.0)
        var_conf = self.config.get('var_confidence', 0.95)
        ruin_threshold = self.config.get('risk_of_ruin_threshold', 0.5)
        
        # Run simulations
        final_equities = []
        max_drawdowns = []
        ruin_count = 0
        
        for _ in range(n_sims):
            # Resample returns with replacement
            sim_returns = np.random.choice(returns, size=n_trades, replace=True)
            
            # Build equity curve
            equity = [initial_capital]
            for r in sim_returns:
                equity.append(equity[-1] * (1 + r))
            
            equity = np.array(equity)
            final_equities.append(equity[-1])
            
            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max
            max_dd = np.max(drawdowns)
            max_drawdowns.append(max_dd)
            
            # Check for ruin
            if max_dd >= ruin_threshold:
                ruin_count += 1
        
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate metrics
        var_idx = int((1 - var_conf) * n_sims)
        sorted_returns = np.sort((final_equities / initial_capital - 1) * 100)
        var = sorted_returns[var_idx]
        cvar = np.mean(sorted_returns[:var_idx]) if var_idx > 0 else var
        
        risk_of_ruin = ruin_count / n_sims
        median_final = np.median(final_equities)
        mean_final = np.mean(final_equities)
        
        # Confidence intervals
        ci_lower = np.percentile(final_equities, 5)
        ci_upper = np.percentile(final_equities, 95)
        
        # Determine status based on risk of ruin
        if risk_of_ruin < 0.05:
            status = ValidationStatus.PASS
            interpretation = f"Low risk profile (Risk of Ruin: {risk_of_ruin:.1%})"
        elif risk_of_ruin < 0.20:
            status = ValidationStatus.WARN
            interpretation = f"Moderate risk (Risk of Ruin: {risk_of_ruin:.1%})"
        else:
            status = ValidationStatus.FAIL
            interpretation = f"High risk (Risk of Ruin: {risk_of_ruin:.1%})"
        
        return ValidationResult(
            validator_name=self.name,
            status=status,
            metrics={
                f'var_{int(var_conf*100)}': round(var, 2),
                f'cvar_{int(var_conf*100)}': round(cvar, 2),
                'risk_of_ruin': round(risk_of_ruin, 4),
                'median_final_equity': round(median_final, 2),
                'mean_final_equity': round(mean_final, 2),
                'equity_ci_90': [round(ci_lower, 2), round(ci_upper, 2)],
                'median_max_drawdown': round(np.median(max_drawdowns) * 100, 2),
                'worst_max_drawdown': round(np.max(max_drawdowns) * 100, 2),
                'n_simulations': n_sims,
                'n_trades_per_sim': n_trades
            },
            confidence=round(1 - risk_of_ruin, 4),
            interpretation=interpretation,
            details={
                'ruin_threshold': ruin_threshold,
                'initial_capital': initial_capital,
                'actual_trades_used': len(returns)
            }
        )
    
    def simulate_equity_paths(
        self,
        returns: np.ndarray,
        n_sims: int = 100,
        n_trades: int = 100,
        initial_capital: float = 10000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simulated equity paths for visualization.
        
        Args:
            returns: Array of trade returns (as decimals)
            n_sims: Number of simulations
            n_trades: Trades per simulation
            initial_capital: Starting capital
        
        Returns:
            Tuple of (equity_paths array, max_drawdowns array)
        """
        paths = np.zeros((n_sims, n_trades + 1))
        paths[:, 0] = initial_capital
        
        for i in range(n_sims):
            sim_returns = np.random.choice(returns, size=n_trades, replace=True)
            for j, r in enumerate(sim_returns):
                paths[i, j + 1] = paths[i, j] * (1 + r)
        
        return paths
