"""
Walk-Forward Optimization Stage
===============================
Purged walk-forward validation with parameter optimization.
"""

from typing import Callable, Dict, List
import pandas as pd
from loguru import logger

from src.validation.walk_forward import PurgedWalkForwardEngine, WalkForwardConfig
from src.validation.tracker import ExperimentTracker
from .base import BaseStage, StageContext


class WalkForwardStage(BaseStage):
    """
    Purged walk-forward optimization stage.
    
    Performs time-series cross-validation with:
    - Purging (remove overlapping periods)
    - Embargo (add buffer between train/test)
    - Parameter optimization per fold
    """
    
    def __init__(
        self,
        n_folds: int = 10,
        purge_bars: int = 5,
        embargo_bars: int = 5,
        train_ratio: float = 0.7,
        tracker: ExperimentTracker = None
    ):
        super().__init__("Walk-Forward Optimization")
        
        config = WalkForwardConfig(
            n_folds=n_folds,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            train_ratio=train_ratio
        )
        self.walk_forward = PurgedWalkForwardEngine(config=config, tracker=tracker)
    
    def execute(self, context: StageContext):
        """
        Run walk-forward optimization.
        
        Args:
            context: Pipeline context with df, backtest_fn, param_grid
            
        Returns:
            WalkForwardResult with OOS metrics
        """
        wf_result = self.walk_forward.run(
            df=context.df,
            objective_fn=context.backtest_fn,
            param_grid=context.param_grid,
            optimization_metric=context.optimization_metric
        )
        
        # Store in context
        context.walk_forward_result = wf_result
        
        logger.info(f"OOS Sharpe: {wf_result.aggregate_sharpe:.3f}")
        logger.info(f"OOS Max DD: {wf_result.aggregate_max_dd:.2f}%")
        logger.info(f"Total OOS Trades: {wf_result.total_oos_trades}")
        logger.info(f"Parameter Stability: {wf_result.param_stability_score:.3f}")
        
        return wf_result
