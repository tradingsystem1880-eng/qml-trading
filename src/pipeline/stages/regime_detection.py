"""
Regime Detection Stage
=====================
Detects market regimes using clustering.
"""

import pandas as pd
from loguru import logger

from src.analysis.regimes import RegimeClassifier, RegimeConfig
from .base import BaseStage, StageContext


class RegimeDetectionStage(BaseStage):
    """
    Market regime detection stage.
    
    Identifies different market conditions (trending, ranging, volatile, calm)
    using unsupervised clustering.
    """
    
    def __init__(self, n_regimes: int = 4, method: str = "kmeans", random_state: int = 42):
        super().__init__("Regime Detection")
        
        config = RegimeConfig(
            n_regimes=n_regimes,
            method=method,
            random_state=random_state
        )
        self.regime_classifier = RegimeClassifier(config=config)
    
    def execute(self, context: StageContext):
        """
        Detect market regimes.
        
        Args:
            context: Pipeline context with df
            
        Returns:
            RegimeResult with labels and statistics
        """
        df = context.df
        
        regime_result = self.regime_classifier.fit_predict(df)
        regime_stats = self.regime_classifier.get_regime_statistics(df, regime_result)
        
        # Store in context
        context.regime_labels = regime_result
        
        logger.info(f"Detected {regime_result.n_regimes} regimes:")
        for _, row in regime_stats.iterrows():
            logger.info(f"  {row['regime']}: {row['pct_of_data']:.1f}% of data")
        
        return regime_result
