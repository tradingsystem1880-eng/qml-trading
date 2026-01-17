"""
Statistical Testing Stage
=========================
Permutation tests and Monte Carlo simulation.
"""

import numpy as np
from loguru import logger

from src.validation.permutation import PermutationTest
from src.validation.monte_carlo import MonteCarloSimulator
from .base import BaseStage, StageContext


class StatisticalTestingStage(BaseStage):
    """
    Statistical significance testing stage.
    
    Performs:
    - Permutation test (is Sharpe real?)
    - Monte Carlo simulation (risk of ruin)
    """
    
    def __init__(
        self,
        n_permutations: int = 10000,
        n_monte_carlo: int = 50000,
        kill_switch_threshold: float = 0.20,
        random_seed: int = 42
    ):
        super().__init__("Statistical Testing")
        
        self.permutation_test = PermutationTest(
            n_permutations=n_permutations,
            random_seed=random_seed
        )
        self.monte_carlo = MonteCarloSimulator(
            n_simulations=n_monte_carlo,
            kill_switch_threshold=kill_switch_threshold,
            random_seed=random_seed
        )
    
    def execute(self, context: StageContext):
        """
        Run statistical tests.
        
        Args:
            context: Pipeline context with walk_forward_result
            
        Returns:
            Dict with permutation and monte carlo results
        """
        # Collect OOS returns
        trade_returns = self._collect_oos_returns(context.walk_forward_result)
        
        if len(trade_returns) < 10:
            logger.warning(f"Insufficient trades ({len(trade_returns)}) for statistical testing")
            return None
        
        # Permutation Test
        logger.info("Running Permutation Test...")
        perm_result = self.permutation_test.run(trade_returns)
        context.permutation_result = perm_result
        
        logger.info(f"  Sharpe p-value: {perm_result.sharpe_p_value:.4f}")
        logger.info(f"  Sharpe percentile: {perm_result.sharpe_percentile:.1f}%")
        
        # Monte Carlo Simulation
        logger.info("Running Monte Carlo Simulation...")
        mc_result = self.monte_carlo.run(trade_returns)
        context.monte_carlo_result = mc_result
        
        logger.info(f"  VaR 95%: {mc_result.var_95:.2f}%")
        logger.info(f"  VaR 99%: {mc_result.var_99:.2f}%")
        logger.info(f"  Kill Switch Prob: {mc_result.kill_switch_prob:.2%}")
        
        return {
            'permutation': perm_result,
            'monte_carlo': mc_result
        }
    
    def _collect_oos_returns(self, wf_result) -> np.ndarray:
        """Collect out-of-sample returns from walk-forward."""
        if wf_result is None:
            return np.array([])
        
        all_returns = []
        for fold in wf_result.fold_results:
            oos_metrics = fold.out_of_sample_metrics
            trades = fold.out_of_sample_trades
            
            if trades > 0:
                win_rate = oos_metrics.get("win_rate", 0.5)
                avg_win = oos_metrics.get("avg_win_pct", 2.0)
                avg_loss = oos_metrics.get("avg_loss_pct", -1.0)
                
                n_wins = int(trades * win_rate)
                n_losses = trades - n_wins
                
                returns = [avg_win] * n_wins + [avg_loss] * n_losses
                all_returns.extend(returns)
        
        if not all_returns and wf_result.total_oos_trades > 0:
            avg_return = wf_result.aggregate_return / wf_result.total_oos_trades
            all_returns = [avg_return] * wf_result.total_oos_trades
        
        return np.array(all_returns)
