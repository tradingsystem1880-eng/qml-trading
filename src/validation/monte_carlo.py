"""
Monte Carlo Simulation for Strategy Risk Analysis
==================================================
Path-dependent analysis with 50,000+ simulated equity curves
for risk metrics like VaR, expected shortfall, and kill switch analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""
    
    # Simulated paths
    equity_paths: np.ndarray        # (n_sims, n_steps)
    max_drawdowns: np.ndarray       # Max DD for each simulation
    final_returns: np.ndarray       # Final return for each simulation
    
    # Value at Risk
    var_95: float                   # 95% VaR for max drawdown
    var_99: float                   # 99% VaR for max drawdown
    expected_shortfall_95: float    # Expected shortfall (CVaR) at 95%
    
    # Recovery analysis
    time_to_recovery_mean: float
    time_to_recovery_95: float      # 95th percentile recovery time
    
    # Kill switch analysis
    kill_switch_prob: float         # P(max DD > threshold)
    kill_switch_threshold: float
    
    # Configuration
    n_simulations: int
    initial_capital: float
    
    @property
    def median_final_return(self) -> float:
        """Median final return across simulations."""
        return float(np.median(self.final_returns))
    
    @property
    def worst_case_return(self) -> float:
        """1st percentile (worst case) final return."""
        return float(np.percentile(self.final_returns, 1))
    
    @property
    def best_case_return(self) -> float:
        """99th percentile (best case) final return."""
        return float(np.percentile(self.final_returns, 99))


class MonteCarloSimulator:
    """
    Monte Carlo Simulation for Strategy Risk Analysis.
    
    Generates 50,000+ simulated equity curves via:
    - Trade sequence randomization
    - Return distribution bootstrapping
    - Volatility-adjusted path generation
    
    Outputs critical risk metrics for deployment decisions.
    """
    
    def __init__(
        self,
        n_simulations: int = 50000,
        kill_switch_threshold: float = 0.20,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulations (min 50,000 recommended)
            kill_switch_threshold: Drawdown threshold for kill switch (e.g., 0.20 = 20%)
            random_seed: Optional seed for reproducibility
        """
        if n_simulations < 10000:
            logger.warning(
                f"n_simulations={n_simulations} is low. "
                "Consider using at least 50,000 for reliable VaR estimates."
            )
        
        self.n_simulations = n_simulations
        self.kill_switch_threshold = kill_switch_threshold
        self.rng = np.random.default_rng(random_seed)
        
        logger.info(
            f"MonteCarloSimulator initialized: {n_simulations} sims, "
            f"kill switch at {kill_switch_threshold:.0%}"
        )
    
    def run(
        self,
        trade_returns: np.ndarray,
        initial_capital: float = 100000.0,
        method: str = "sequence"
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            trade_returns: Array of individual trade returns (percentages)
            initial_capital: Starting capital
            method: Simulation method - "sequence" (shuffle), "bootstrap" (resample)
            
        Returns:
            MonteCarloResult with all risk metrics
        """
        trade_returns = np.asarray(trade_returns).flatten()
        n_trades = len(trade_returns)
        
        if n_trades < 10:
            raise ValueError(
                f"Insufficient trades ({n_trades}) for Monte Carlo. "
                "Need at least 10 trades."
            )
        
        logger.info(
            f"Running Monte Carlo ({method}): {self.n_simulations} sims, "
            f"{n_trades} trades"
        )
        
        # Generate simulated return sequences
        if method == "sequence":
            sim_returns = self._sequence_randomization(trade_returns)
        elif method == "bootstrap":
            sim_returns = self._bootstrap_with_replacement(trade_returns)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Build equity curves
        equity_paths = self._build_equity_curves(sim_returns, initial_capital)
        
        # Calculate drawdowns
        max_drawdowns = np.array([
            self._calculate_max_drawdown(path) for path in equity_paths
        ])
        
        # Calculate final returns
        final_returns = (equity_paths[:, -1] - initial_capital) / initial_capital * 100
        
        # Calculate VaR (percentiles of max DD distribution)
        var_95 = float(np.percentile(max_drawdowns, 95))
        var_99 = float(np.percentile(max_drawdowns, 99))
        
        # Expected Shortfall (average of worst 5%)
        worst_5_pct = max_drawdowns >= np.percentile(max_drawdowns, 95)
        expected_shortfall_95 = float(np.mean(max_drawdowns[worst_5_pct]))
        
        # Time to recovery analysis
        recovery_times = self._calculate_recovery_times(equity_paths, initial_capital)
        time_to_recovery_mean = float(np.mean(recovery_times[recovery_times > 0]))
        time_to_recovery_95 = float(np.percentile(recovery_times[recovery_times > 0], 95)) if np.any(recovery_times > 0) else 0
        
        # Kill switch probability
        kill_switch_prob = float(np.mean(max_drawdowns >= self.kill_switch_threshold * 100))
        
        result = MonteCarloResult(
            equity_paths=equity_paths,
            max_drawdowns=max_drawdowns,
            final_returns=final_returns,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=expected_shortfall_95,
            time_to_recovery_mean=time_to_recovery_mean,
            time_to_recovery_95=time_to_recovery_95,
            kill_switch_prob=kill_switch_prob,
            kill_switch_threshold=self.kill_switch_threshold * 100,
            n_simulations=self.n_simulations,
            initial_capital=initial_capital,
        )
        
        logger.info(
            f"Monte Carlo complete: "
            f"VaR95={var_95:.1f}%, VaR99={var_99:.1f}%, "
            f"Kill switch prob={kill_switch_prob:.1%}"
        )
        
        return result
    
    def _sequence_randomization(self, returns: np.ndarray) -> np.ndarray:
        """
        Generate simulations by randomly shuffling trade sequence.
        
        Preserves individual trade outcomes but randomizes order.
        
        Args:
            returns: Original trade returns
            
        Returns:
            Array of shape (n_simulations, n_trades)
        """
        n_trades = len(returns)
        sim_returns = np.zeros((self.n_simulations, n_trades))
        
        for i in range(self.n_simulations):
            sim_returns[i] = self.rng.permutation(returns)
        
        return sim_returns
    
    def _bootstrap_with_replacement(self, returns: np.ndarray) -> np.ndarray:
        """
        Generate simulations by resampling with replacement.
        
        Creates potentially different trade counts and return distributions.
        
        Args:
            returns: Original trade returns
            
        Returns:
            Array of shape (n_simulations, n_trades)
        """
        n_trades = len(returns)
        sim_returns = np.zeros((self.n_simulations, n_trades))
        
        for i in range(self.n_simulations):
            indices = self.rng.integers(0, n_trades, size=n_trades)
            sim_returns[i] = returns[indices]
        
        return sim_returns
    
    def _build_equity_curves(
        self,
        sim_returns: np.ndarray,
        initial_capital: float
    ) -> np.ndarray:
        """
        Build equity curves from simulated returns.
        
        Args:
            sim_returns: Simulated returns (n_sims, n_trades)
            initial_capital: Starting capital
            
        Returns:
            Equity curves (n_sims, n_trades + 1)
        """
        n_sims, n_trades = sim_returns.shape
        
        # Initialize with starting capital
        equity = np.zeros((n_sims, n_trades + 1))
        equity[:, 0] = initial_capital
        
        # Build curves (assuming returns are percentages)
        for t in range(n_trades):
            equity[:, t + 1] = equity[:, t] * (1 + sim_returns[:, t] / 100)
        
        return equity
    
    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """
        Calculate maximum drawdown for an equity curve.
        
        Args:
            equity: Equity curve array
            
        Returns:
            Maximum drawdown as percentage
        """
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max * 100
        return float(np.max(drawdowns))
    
    def _calculate_recovery_times(
        self,
        equity_paths: np.ndarray,
        initial_capital: float
    ) -> np.ndarray:
        """
        Calculate time to recovery from max drawdown for each simulation.
        
        Args:
            equity_paths: All equity curves
            initial_capital: Starting capital
            
        Returns:
            Array of recovery times (in trade steps)
        """
        n_sims, n_steps = equity_paths.shape
        recovery_times = np.zeros(n_sims)
        
        for i in range(n_sims):
            equity = equity_paths[i]
            running_max = np.maximum.accumulate(equity)
            
            # Find where max drawdown occurred
            drawdowns = (running_max - equity) / running_max
            max_dd_idx = np.argmax(drawdowns)
            max_dd_level = running_max[max_dd_idx]
            
            # Find recovery point (if any)
            recovery_idx = np.where(equity[max_dd_idx:] >= max_dd_level)[0]
            
            if len(recovery_idx) > 0:
                recovery_times[i] = recovery_idx[0]
            else:
                recovery_times[i] = n_steps - max_dd_idx  # Still in drawdown
        
        return recovery_times
    
    def generate_report(self, result: MonteCarloResult) -> str:
        """
        Generate text report of Monte Carlo results.
        
        Args:
            result: MonteCarloResult to report
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "MONTE CARLO SIMULATION RESULTS",
            "=" * 60,
            "",
            f"Simulations: {result.n_simulations:,}",
            f"Initial Capital: ${result.initial_capital:,.0f}",
            "",
            "Return Distribution:",
            f"  - Median Final Return: {result.median_final_return:+.2f}%",
            f"  - Best Case (99th): {result.best_case_return:+.2f}%",
            f"  - Worst Case (1st): {result.worst_case_return:+.2f}%",
            "",
            "Drawdown Risk (Value at Risk):",
            f"  - VaR 95%: {result.var_95:.2f}%",
            f"    (95% of paths have max DD <= {result.var_95:.2f}%)",
            f"  - VaR 99%: {result.var_99:.2f}%",
            f"    (99% of paths have max DD <= {result.var_99:.2f}%)",
            f"  - Expected Shortfall (CVaR 95%): {result.expected_shortfall_95:.2f}%",
            f"    (Average DD in worst 5% of scenarios)",
            "",
            "Recovery Analysis:",
            f"  - Mean Time to Recovery: {result.time_to_recovery_mean:.1f} trades",
            f"  - 95th Percentile Recovery: {result.time_to_recovery_95:.1f} trades",
            "",
            "Kill Switch Analysis:",
            f"  - Threshold: {result.kill_switch_threshold:.1f}%",
            f"  - Probability of Trigger: {result.kill_switch_prob:.2%}",
            f"    ({result.kill_switch_prob * result.n_simulations:,.0f} of {result.n_simulations:,} paths)",
            "",
            "=" * 60,
            "Risk Summary:",
            f"  - Max tolerable leverage: {100 / result.var_99:.1f}x",
            f"    (To keep 99% VaR under 100%)",
            "=" * 60,
        ]
        
        return "\n".join(lines)


def run_monte_carlo(
    trade_returns: np.ndarray,
    n_simulations: int = 50000,
    initial_capital: float = 100000.0,
    kill_switch_threshold: float = 0.20,
    seed: Optional[int] = None
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo simulation.
    
    Args:
        trade_returns: Array of trade returns
        n_simulations: Number of simulations
        initial_capital: Starting capital
        kill_switch_threshold: Kill switch drawdown threshold
        seed: Random seed
        
    Returns:
        MonteCarloResult
    """
    sim = MonteCarloSimulator(
        n_simulations=n_simulations,
        kill_switch_threshold=kill_switch_threshold,
        random_seed=seed
    )
    return sim.run(trade_returns, initial_capital)
