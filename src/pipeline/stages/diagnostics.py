"""
Diagnostics Stage
================
Bootstrap confidence intervals and final diagnostics.
"""

import pandas as pd
from loguru import logger

from src.validation.bootstrap import BlockBootstrap
from .base import BaseStage, StageContext


class DiagnosticsStage(BaseStage):
    """
    Diagnostics and confidence intervals stage.
    
    Computes bootstrap confidence intervals for:
    - Sharpe ratio
    - Win rate
    - Max drawdown
    - Other key metrics
    """
    
    def __init__(
        self,
        n_bootstrap: int = 5000,
        block_size: int = 5,
        random_seed: int = 42
    ):
        super().__init__("Diagnostics & CI")
        
        self.bootstrap = BlockBootstrap(
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            random_seed=random_seed
        )
    
    def execute(self, context: StageContext):
        """
        Compute bootstrap confidence intervals.
        
        Args:
            context: Pipeline context with walk_forward_result
            
        Returns:
            Dict of bootstrap results
        """
        # Collect OOS returns
        trade_returns = self._collect_oos_returns(context.walk_forward_result)
        
        if len(trade_returns) < 10:
            logger.warning("Insufficient trades for bootstrap CI")
            return None
        
        # Compute all CIs
        trades_df = pd.DataFrame({"pnl_pct": trade_returns})
        boot_results = self.bootstrap.all_metrics_ci(trades_df)
        
        # Log key CIs
        if "sharpe_ratio" in boot_results:
            sr = boot_results["sharpe_ratio"]
            logger.info(f"Sharpe CI: [{sr.ci_lower:.3f}, {sr.ci_upper:.3f}]")
        
        if "win_rate" in boot_results:
            wr = boot_results["win_rate"]
            logger.info(f"Win Rate CI: [{wr.ci_lower:.3f}, {wr.ci_upper:.3f}]")
        
        return boot_results
    
    def _collect_oos_returns(self, wf_result):
        """Collect out-of-sample returns."""
        import numpy as np
        
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
        
        return np.array(all_returns)
