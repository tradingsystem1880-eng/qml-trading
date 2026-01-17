"""
Feature Engineering Stage
=========================
Computes 200+ features for validation pipeline.
"""

import pandas as pd
from loguru import logger

from src.features.library import FeatureLibrary
from .base import BaseStage, StageContext


class FeatureEngineeringStage(BaseStage):
    """
    Feature engineering for validation pipeline.
    
    Computes comprehensive feature set including:
    - Technical indicators (RSI, MACD, ATR, etc.)
    - Volume features
    - Market microstructure
    - Regime features
    """
    
    def __init__(self, compute_features: bool = True):
        super().__init__("Feature Engineering")
        self.compute_features = compute_features
        self.feature_library = FeatureLibrary()
    
    def execute(self, context: StageContext) -> pd.DataFrame:
        """
        Compute features for entire dataset.
        
        Args:
            context: Pipeline context with df
            
        Returns:
            DataFrame with features
        """
        if not self.compute_features:
            logger.info("Feature computation skipped (disabled)")
            return None
        
        df = context.df
        features_df = self.feature_library.compute_features_for_range(df)
        
        # Store in context
        context.features_df = features_df
        
        logger.info(f"Generated {len(features_df.columns)} features for {len(features_df)} bars")
        
        return features_df
