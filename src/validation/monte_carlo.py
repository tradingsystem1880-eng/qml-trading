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
- Prop firm challenge pass probability
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from src.validation.base import BaseValidator, ValidationResult, ValidationStatus


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 1000
    n_trades_per_sim: int = 100  # Or use actual trade count
    initial_capital: float = 10000.0
    var_confidence: float = 0.95  # VaR at 95% confidence
    risk_of_ruin_threshold: float = 0.5  # 50% drawdown = ruin


class FailReason(Enum):
    """Reasons for failing a prop firm challenge."""
    DAILY_LOSS = "daily_loss_limit"
    TOTAL_LOSS = "total_loss_limit"
    TIME_LIMIT = "time_limit"
    MIN_DAYS = "min_trading_days"


@dataclass
class PropFirmRules:
    """
    Prop firm challenge parameters.

    Default values based on common FTMO/MFF-style challenges.
    Breakout $100K uses: 4% daily, 8% total, 8% target.
    """
    profit_target_pct: float = 10.0  # Must reach 10% profit
    daily_loss_limit_pct: float = 5.0  # Max -5% daily drawdown
    total_loss_limit_pct: float = 10.0  # Max -10% total drawdown
    min_trading_days: int = 10  # Minimum trading days required
    time_limit_days: int = 30  # Days to pass challenge
    # Phase 5 additions
    account_size: float = 100000.0  # Account size in dollars
    max_position_size_pct: float = 2.0  # Max position as % of account
    consistency_rule: bool = True  # No single day > 30% of profits


@dataclass
class PropFirmResult:
    """Result from prop firm challenge simulation."""
    pass_rate: float  # Probability of passing (0-1)
    avg_days_to_pass: float  # Average days when passing
    fail_reasons: Dict[str, float]  # Distribution of failure reasons
    profit_on_pass: float  # Average profit % when passing
    details: Dict[str, Any] = field(default_factory=dict)


class PropFirmSimulator:
    """
    Simulates prop firm challenge pass probability.

    Models daily P&L sequences and checks against challenge rules:
    - Daily loss limits
    - Total drawdown limits
    - Profit targets
    - Time constraints

    Usage:
        sim = PropFirmSimulator()
        rules = PropFirmRules(profit_target_pct=10.0)
        result = sim.simulate_challenge(daily_returns, rules)
        print(f"Pass rate: {result.pass_rate:.1%}")
    """

    def simulate_challenge(
        self,
        returns: np.ndarray,
        rules: PropFirmRules,
        n_simulations: int = 1000,
        trades_per_day: float = 2.0
    ) -> PropFirmResult:
        """
        Simulate prop firm challenge outcomes.

        Args:
            returns: Array of trade returns (as decimals, e.g., 0.02 for 2%)
            rules: PropFirmRules defining challenge parameters
            n_simulations: Number of Monte Carlo simulations
            trades_per_day: Average trades per trading day

        Returns:
            PropFirmResult with pass rate and failure analysis
        """
        if len(returns) < 10:
            return PropFirmResult(
                pass_rate=0.0,
                avg_days_to_pass=0.0,
                fail_reasons={'insufficient_data': 1.0},
                profit_on_pass=0.0,
                details={'error': 'Insufficient trade data'}
            )

        pass_count = 0
        days_to_pass = []
        profits_on_pass = []
        fail_counts = {
            FailReason.DAILY_LOSS.value: 0,
            FailReason.TOTAL_LOSS.value: 0,
            FailReason.TIME_LIMIT.value: 0,
            FailReason.MIN_DAYS.value: 0,
        }

        for _ in range(n_simulations):
            result = self._run_single_challenge(returns, rules, trades_per_day)

            if result['passed']:
                pass_count += 1
                days_to_pass.append(result['days'])
                profits_on_pass.append(result['final_pnl'])
            else:
                fail_counts[result['fail_reason']] += 1

        pass_rate = pass_count / n_simulations
        avg_days = np.mean(days_to_pass) if days_to_pass else 0.0
        avg_profit = np.mean(profits_on_pass) if profits_on_pass else 0.0

        # Convert fail counts to proportions
        fail_total = n_simulations - pass_count
        fail_reasons = {}
        if fail_total > 0:
            for reason, count in fail_counts.items():
                fail_reasons[reason] = count / fail_total

        return PropFirmResult(
            pass_rate=pass_rate,
            avg_days_to_pass=avg_days,
            fail_reasons=fail_reasons,
            profit_on_pass=avg_profit * 100,  # Convert to percentage
            details={
                'n_simulations': n_simulations,
                'trades_per_day': trades_per_day,
                'rules': {
                    'profit_target': rules.profit_target_pct,
                    'daily_limit': rules.daily_loss_limit_pct,
                    'total_limit': rules.total_loss_limit_pct,
                    'min_days': rules.min_trading_days,
                    'time_limit': rules.time_limit_days,
                }
            }
        )

    def _run_single_challenge(
        self,
        returns: np.ndarray,
        rules: PropFirmRules,
        trades_per_day: float
    ) -> Dict[str, Any]:
        """Run a single challenge simulation."""
        equity = 1.0  # Start at 100%
        high_water_mark = 1.0

        daily_returns = []
        trading_days = 0
        current_day_pnl = 0.0
        trades_today = 0

        for day in range(rules.time_limit_days):
            # Simulate trades for this day
            n_trades = np.random.poisson(trades_per_day)
            if n_trades == 0:
                continue

            trading_days += 1
            trades_today = 0
            current_day_pnl = 0.0

            for _ in range(n_trades):
                # Sample a trade return
                trade_return = np.random.choice(returns)

                # Apply to equity
                equity *= (1 + trade_return)
                current_day_pnl += trade_return
                trades_today += 1

                # Check total drawdown
                total_dd = (high_water_mark - equity) / high_water_mark * 100
                if total_dd >= rules.total_loss_limit_pct:
                    return {
                        'passed': False,
                        'fail_reason': FailReason.TOTAL_LOSS.value,
                        'days': day + 1,
                        'final_pnl': equity - 1
                    }

                # Update high water mark
                if equity > high_water_mark:
                    high_water_mark = equity

            # Check daily loss limit
            if current_day_pnl * 100 <= -rules.daily_loss_limit_pct:
                return {
                    'passed': False,
                    'fail_reason': FailReason.DAILY_LOSS.value,
                    'days': day + 1,
                    'final_pnl': equity - 1
                }

            daily_returns.append(current_day_pnl)

            # Check if target reached
            profit_pct = (equity - 1) * 100
            if profit_pct >= rules.profit_target_pct and trading_days >= rules.min_trading_days:
                return {
                    'passed': True,
                    'fail_reason': None,
                    'days': day + 1,
                    'final_pnl': equity - 1
                }

        # Time limit reached
        profit_pct = (equity - 1) * 100

        # Check if met target but not min days
        if profit_pct >= rules.profit_target_pct and trading_days < rules.min_trading_days:
            return {
                'passed': False,
                'fail_reason': FailReason.MIN_DAYS.value,
                'days': rules.time_limit_days,
                'final_pnl': equity - 1
            }

        return {
            'passed': False,
            'fail_reason': FailReason.TIME_LIMIT.value,
            'days': rules.time_limit_days,
            'final_pnl': equity - 1
        }


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
