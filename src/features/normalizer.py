"""
Feature Normalizer and Selector - Phase 3
==========================================
sklearn-based normalization and feature selection (90% rule).

Uses:
- sklearn.preprocessing.RobustScaler for normalization (handles outliers)
- sklearn.feature_selection.SelectKBest for feature selection
- joblib for model persistence
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif


class FeatureNormalizer:
    """
    sklearn-based normalization that handles outliers.

    Uses RobustScaler which uses median and IQR, making it robust to outliers.
    This is important for financial data which often has extreme values.

    Usage:
        normalizer = FeatureNormalizer()
        X_train_scaled = normalizer.fit_transform(X_train)
        X_test_scaled = normalizer.transform(X_test)

        # Save for later use
        normalizer.save('models/scaler.joblib')

        # Load saved normalizer
        normalizer = FeatureNormalizer.load('models/scaler.joblib')
    """

    def __init__(self, use_robust: bool = True):
        """
        Initialize normalizer.

        Args:
            use_robust: If True, use RobustScaler (recommended for financial data).
                       If False, use StandardScaler (mean=0, std=1).
        """
        self.use_robust = use_robust
        self.scaler = RobustScaler() if use_robust else StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.fit_time: Optional[datetime] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FeatureNormalizer':
        """
        Fit the scaler to data.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names

        Returns:
            self
        """
        self.scaler.fit(X)
        self.feature_names = feature_names
        self.is_fitted = True
        self.fit_time = datetime.now()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names

        Returns:
            Scaled feature matrix
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling transformation.

        Args:
            X_scaled: Scaled feature matrix

        Returns:
            Original scale feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return self.scaler.inverse_transform(X_scaled)

    def get_params(self) -> Dict[str, Any]:
        """Get scaler parameters for inspection."""
        if not self.is_fitted:
            return {}

        params = {
            'type': 'RobustScaler' if self.use_robust else 'StandardScaler',
            'is_fitted': self.is_fitted,
            'fit_time': str(self.fit_time) if self.fit_time else None,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
        }

        if self.use_robust:
            params['center'] = self.scaler.center_.tolist() if hasattr(self.scaler, 'center_') else None
            params['scale'] = self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        else:
            params['mean'] = self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None
            params['std'] = self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None

        return params

    def save(self, path: str) -> None:
        """
        Save normalizer to file using joblib.

        Args:
            path: File path (e.g., 'models/scaler.joblib')
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'use_robust': self.use_robust,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'fit_time': self.fit_time,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureNormalizer':
        """
        Load normalizer from file.

        Args:
            path: File path

        Returns:
            Loaded FeatureNormalizer
        """
        data = joblib.load(path)
        normalizer = cls(use_robust=data['use_robust'])
        normalizer.scaler = data['scaler']
        normalizer.feature_names = data['feature_names']
        normalizer.is_fitted = data['is_fitted']
        normalizer.fit_time = data['fit_time']
        return normalizer


class FeatureSelector:
    """
    sklearn-based feature selection using mutual information.

    Selects the most informative features for predicting trade outcomes.
    Uses mutual_info_classif which measures dependency between features and target.

    Usage:
        selector = FeatureSelector(n_features=10)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_names = selector.get_selected_features(feature_names)

        # Get feature scores
        scores = selector.get_feature_scores(feature_names)
    """

    def __init__(
        self,
        n_features: int = 10,
        method: str = 'mutual_info'
    ):
        """
        Initialize feature selector.

        Args:
            n_features: Number of features to select
            method: 'mutual_info' (recommended) or 'f_classif' (faster)
        """
        self.n_features = n_features
        self.method = method

        if method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=n_features)
        elif method == 'f_classif':
            self.selector = SelectKBest(f_classif, k=n_features)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mutual_info' or 'f_classif'")

        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None
        self.scores_: Optional[np.ndarray] = None
        self.selected_mask_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FeatureSelector':
        """
        Fit selector to find best features.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: Optional list of feature names

        Returns:
            self
        """
        # Handle NaN values
        X_clean = np.nan_to_num(X, nan=0.0)

        self.selector.fit(X_clean, y)
        self.feature_names = feature_names
        self.scores_ = self.selector.scores_
        self.selected_mask_ = self.selector.get_support()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to selected features only.

        Args:
            X: Feature matrix

        Returns:
            Reduced feature matrix with only selected features
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        X_clean = np.nan_to_num(X, nan=0.0)
        return self.selector.transform(X_clean)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names

        Returns:
            Reduced feature matrix
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_features(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Get names of selected features.

        Args:
            feature_names: Feature names (uses stored names if None)

        Returns:
            List of selected feature names
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        names = feature_names or self.feature_names
        if names is None:
            return [f"feature_{i}" for i in np.where(self.selected_mask_)[0]]

        return [names[i] for i in np.where(self.selected_mask_)[0]]

    def get_feature_scores(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get importance scores for all features.

        Args:
            feature_names: Feature names (uses stored names if None)

        Returns:
            Dict mapping feature name to score, sorted by score descending
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        names = feature_names or self.feature_names
        if names is None:
            names = [f"feature_{i}" for i in range(len(self.scores_))]

        scores_dict = {name: float(score) for name, score in zip(names, self.scores_)}
        return dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True))

    def get_feature_rankings(self, feature_names: Optional[List[str]] = None) -> List[Tuple[str, float, bool]]:
        """
        Get ranked list of features with scores and selection status.

        Args:
            feature_names: Feature names

        Returns:
            List of (name, score, is_selected) tuples, sorted by score
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        names = feature_names or self.feature_names
        if names is None:
            names = [f"feature_{i}" for i in range(len(self.scores_))]

        rankings = [
            (name, float(score), bool(selected))
            for name, score, selected in zip(names, self.scores_, self.selected_mask_)
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def save(self, path: str) -> None:
        """Save selector to file using joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'selector': self.selector,
            'n_features': self.n_features,
            'method': self.method,
            'feature_names': self.feature_names,
            'scores_': self.scores_,
            'selected_mask_': self.selected_mask_,
            'is_fitted': self.is_fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureSelector':
        """Load selector from file."""
        data = joblib.load(path)
        selector = cls(n_features=data['n_features'], method=data['method'])
        selector.selector = data['selector']
        selector.feature_names = data['feature_names']
        selector.scores_ = data['scores_']
        selector.selected_mask_ = data['selected_mask_']
        selector.is_fitted = data['is_fitted']
        return selector


class FeaturePreprocessor:
    """
    Combined preprocessing: normalization + optional feature selection.

    Convenience class that chains FeatureNormalizer and FeatureSelector.

    Usage:
        preprocessor = FeaturePreprocessor(n_select=10)
        X_processed = preprocessor.fit_transform(X, y, feature_names)

        # Save entire pipeline
        preprocessor.save('models/preprocessor.joblib')
    """

    def __init__(
        self,
        normalize: bool = True,
        use_robust: bool = True,
        select_features: bool = False,
        n_select: int = 10,
        selection_method: str = 'mutual_info'
    ):
        """
        Initialize preprocessor.

        Args:
            normalize: Whether to normalize features
            use_robust: Use RobustScaler (True) or StandardScaler (False)
            select_features: Whether to perform feature selection
            n_select: Number of features to select
            selection_method: 'mutual_info' or 'f_classif'
        """
        self.normalize = normalize
        self.select_features = select_features

        self.normalizer = FeatureNormalizer(use_robust=use_robust) if normalize else None
        self.selector = FeatureSelector(n_select, selection_method) if select_features else None

        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'FeaturePreprocessor':
        """
        Fit preprocessor.

        Args:
            X: Feature matrix
            y: Target labels (required if select_features=True)
            feature_names: Feature names

        Returns:
            self
        """
        self.feature_names = feature_names

        if self.normalizer:
            self.normalizer.fit(X, feature_names)

        if self.selector:
            if y is None:
                raise ValueError("y required for feature selection")
            X_norm = self.normalizer.transform(X) if self.normalizer else X
            self.selector.fit(X_norm, y, feature_names)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features.

        Args:
            X: Feature matrix

        Returns:
            Processed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        X_out = X
        if self.normalizer:
            X_out = self.normalizer.transform(X_out)
        if self.selector:
            X_out = self.selector.transform(X_out)

        return X_out

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_features(self) -> Optional[List[str]]:
        """Get selected feature names (if selection enabled)."""
        if self.selector and self.selector.is_fitted:
            return self.selector.get_selected_features(self.feature_names)
        return self.feature_names

    def get_feature_scores(self) -> Optional[Dict[str, float]]:
        """Get feature scores (if selection enabled)."""
        if self.selector and self.selector.is_fitted:
            return self.selector.get_feature_scores(self.feature_names)
        return None

    def save(self, path: str) -> None:
        """Save preprocessor to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'normalizer': self.normalizer,
            'selector': self.selector,
            'normalize': self.normalize,
            'select_features': self.select_features,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'FeaturePreprocessor':
        """Load preprocessor from file."""
        data = joblib.load(path)
        preprocessor = cls(
            normalize=data['normalize'],
            select_features=data['select_features']
        )
        preprocessor.normalizer = data['normalizer']
        preprocessor.selector = data['selector']
        preprocessor.feature_names = data['feature_names']
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor
