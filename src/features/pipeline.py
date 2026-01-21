"""
Feature Pipeline for QML Trading System - Phase 3
==================================================
Batch processing of features for patterns, experiments, and ML training.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from src.data.sqlite_manager import SQLiteManager, get_db
from src.data.schemas import PatternDetection, FeatureVector, TradeOutcome
from src.features.calculator import FeatureCalculator


class FeaturePipeline:
    """
    Batch process features for patterns and prepare data for ML.

    Usage:
        pipeline = FeaturePipeline()

        # Calculate features for a single pattern
        features = pipeline.calculate_for_pattern(pattern, ohlcv_df)

        # Get training data
        X, y = pipeline.get_training_data()
    """

    def __init__(self, db: Optional[SQLiteManager] = None):
        """
        Initialize pipeline.

        Args:
            db: Database manager. Uses singleton if not provided.
        """
        self.db = db or get_db()

    def calculate_for_pattern(
        self,
        pattern: PatternDetection,
        ohlcv: pd.DataFrame,
        outcome: Optional[str] = None,
        r_multiple: Optional[float] = None,
        save: bool = True
    ) -> FeatureVector:
        """
        Calculate and optionally store features for a single pattern.

        Args:
            pattern: PatternDetection object
            ohlcv: OHLCV DataFrame
            outcome: Trade outcome ('WIN', 'LOSS', 'BREAKEVEN')
            r_multiple: Actual R achieved
            save: Whether to save to database

        Returns:
            FeatureVector with calculated features
        """
        calculator = FeatureCalculator(ohlcv)
        features = calculator.calculate_features(pattern)

        # Add outcome if provided (for labeling)
        if outcome:
            features.outcome = outcome
        if r_multiple is not None:
            features.r_multiple = r_multiple

        if save:
            self.db.save_features(features)

        return features

    def calculate_for_patterns(
        self,
        patterns: List[PatternDetection],
        ohlcv: pd.DataFrame,
        outcomes: Optional[Dict[str, str]] = None,
        r_multiples: Optional[Dict[str, float]] = None,
        save: bool = True
    ) -> List[FeatureVector]:
        """
        Calculate features for multiple patterns.

        Args:
            patterns: List of PatternDetection objects
            ohlcv: OHLCV DataFrame
            outcomes: Dict mapping pattern_id to outcome
            r_multiples: Dict mapping pattern_id to R-multiple
            save: Whether to save to database

        Returns:
            List of FeatureVector objects
        """
        calculator = FeatureCalculator(ohlcv)
        results = []

        for pattern in patterns:
            features = calculator.calculate_features(pattern)

            # Add outcome if available
            if outcomes and pattern.id in outcomes:
                features.outcome = outcomes[pattern.id]
            if r_multiples and pattern.id in r_multiples:
                features.r_multiple = r_multiples[pattern.id]

            if save:
                self.db.save_features(features)

            results.append(features)

        return results

    def calculate_for_experiment(
        self,
        experiment_id: str,
        ohlcv: pd.DataFrame,
        save: bool = True
    ) -> List[FeatureVector]:
        """
        Calculate features for all patterns in an experiment.

        Links features to trade outcomes automatically.

        Args:
            experiment_id: Experiment run_id
            ohlcv: OHLCV DataFrame
            save: Whether to save to database

        Returns:
            List of FeatureVector objects
        """
        # Get trades for this experiment
        trades = self.db.get_trades_by_experiment(experiment_id)

        # Build outcome mapping from trades
        outcomes = {}
        r_multiples = {}
        pattern_ids = set()

        for trade in trades:
            if trade.pattern_id:
                pattern_ids.add(trade.pattern_id)
                outcomes[trade.pattern_id] = trade.status
                if trade.r_multiple is not None:
                    r_multiples[trade.pattern_id] = trade.r_multiple

        # Get patterns
        patterns = []
        for pid in pattern_ids:
            p = self.db.get_pattern(pid)
            if p:
                patterns.append(p)

        if not patterns:
            return []

        return self.calculate_for_patterns(
            patterns, ohlcv, outcomes, r_multiples, save
        )

    def get_training_dataframe(
        self,
        with_outcomes_only: bool = True,
        min_samples: int = 0
    ) -> pd.DataFrame:
        """
        Get features as DataFrame ready for ML training.

        Args:
            with_outcomes_only: Only include features with labeled outcomes
            min_samples: Minimum samples required

        Returns:
            DataFrame with features and outcome columns
        """
        features = self.db.get_features_for_training(with_outcomes_only)

        if len(features) < min_samples:
            raise ValueError(
                f"Not enough training samples: {len(features)} < {min_samples}"
            )

        return pd.DataFrame(features)

    def get_training_data(
        self,
        with_outcomes_only: bool = True
    ) -> tuple:
        """
        Get X (features) and y (labels) for ML training.

        Args:
            with_outcomes_only: Only include labeled samples

        Returns:
            (X, y) tuple where X is feature array and y is labels
        """
        df = self.get_training_dataframe(with_outcomes_only)

        if len(df) == 0:
            return None, None

        # Get feature columns (numeric only, exclude metadata)
        feature_cols = FeatureVector.feature_names()
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].values

        # Labels: 1 for WIN, 0 for LOSS/BREAKEVEN
        if 'outcome' in df.columns:
            y = (df['outcome'] == 'WIN').astype(int).values
        else:
            y = None

        return X, y

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each feature.

        Returns:
            Dict of feature_name -> {mean, std, min, max, count}
        """
        df = self.get_training_dataframe(with_outcomes_only=False)

        if len(df) == 0:
            return {}

        feature_cols = FeatureVector.feature_names()
        available_cols = [c for c in feature_cols if c in df.columns]

        stats = {}
        for col in available_cols:
            series = pd.to_numeric(df[col], errors='coerce')
            stats[col] = {
                'mean': float(series.mean()) if not series.isna().all() else 0,
                'std': float(series.std()) if not series.isna().all() else 0,
                'min': float(series.min()) if not series.isna().all() else 0,
                'max': float(series.max()) if not series.isna().all() else 0,
                'count': int(series.notna().sum()),
            }

        return stats

    def get_feature_correlations(self) -> pd.DataFrame:
        """
        Get correlation matrix between features.

        Returns:
            Correlation DataFrame
        """
        df = self.get_training_dataframe(with_outcomes_only=False)

        if len(df) == 0:
            return pd.DataFrame()

        feature_cols = FeatureVector.feature_names()
        available_cols = [c for c in feature_cols if c in df.columns]

        numeric_df = df[available_cols].apply(pd.to_numeric, errors='coerce')
        return numeric_df.corr()

    def get_feature_importance_by_outcome(self) -> Dict[str, Dict[str, float]]:
        """
        Compare feature means for winning vs losing trades.

        Returns:
            Dict of feature_name -> {win_mean, loss_mean, diff}
        """
        df = self.get_training_dataframe(with_outcomes_only=True)

        if len(df) == 0 or 'outcome' not in df.columns:
            return {}

        feature_cols = FeatureVector.feature_names()
        available_cols = [c for c in feature_cols if c in df.columns]

        wins = df[df['outcome'] == 'WIN']
        losses = df[df['outcome'] == 'LOSS']

        importance = {}
        for col in available_cols:
            win_mean = pd.to_numeric(wins[col], errors='coerce').mean()
            loss_mean = pd.to_numeric(losses[col], errors='coerce').mean()

            importance[col] = {
                'win_mean': float(win_mean) if not pd.isna(win_mean) else 0,
                'loss_mean': float(loss_mean) if not pd.isna(loss_mean) else 0,
                'diff': float(win_mean - loss_mean) if not (pd.isna(win_mean) or pd.isna(loss_mean)) else 0,
            }

        return importance


def calculate_features_from_trade(
    trade: TradeOutcome,
    pattern: PatternDetection,
    ohlcv: pd.DataFrame,
    db: Optional[SQLiteManager] = None
) -> FeatureVector:
    """
    Convenience function to calculate features from a trade.

    Args:
        trade: TradeOutcome object
        pattern: PatternDetection object
        ohlcv: OHLCV DataFrame
        db: Database manager (optional)

    Returns:
        FeatureVector with outcome labeled
    """
    pipeline = FeaturePipeline(db)
    return pipeline.calculate_for_pattern(
        pattern=pattern,
        ohlcv=ohlcv,
        outcome=trade.status,
        r_multiple=trade.r_multiple,
        save=db is not None
    )
