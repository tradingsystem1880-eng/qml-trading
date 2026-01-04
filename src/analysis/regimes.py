"""
Regime Detection using Unsupervised Learning
=============================================
Clusters market data into distinct regimes using K-Means or GMM.
Enables regime-stratified performance analysis.

Regimes:
- Bull/Quiet: Low volatility uptrend
- Bull/Volatile: High volatility uptrend
- Bear/Quiet: Low volatility downtrend
- Bear/Volatile: High volatility downtrend
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils.indicators import calculate_atr, calculate_adx


class RegimeType(str, Enum):
    """Market regime types."""
    BULL_QUIET = "bull_quiet"
    BULL_VOLATILE = "bull_volatile"
    BEAR_QUIET = "bear_quiet"
    BEAR_VOLATILE = "bear_volatile"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    
    # Number of regimes
    n_regimes: int = 4
    
    # Clustering method: "kmeans" or "gmm"
    method: str = "kmeans"
    
    # Feature windows
    atr_period: int = 14
    adx_period: int = 14
    return_lookback: int = 20
    volatility_lookback: int = 20
    
    # Model parameters
    random_state: int = 42
    n_init: int = 10  # For robustness


@dataclass
class RegimeResult:
    """Result of regime classification."""
    
    # Labels for each timestamp
    labels: np.ndarray
    
    # Regime centers (for interpretation)
    centers: np.ndarray
    
    # Feature names used
    feature_names: List[str]
    
    # Regime mapping (label -> interpretation)
    regime_mapping: Dict[int, str] = field(default_factory=dict)
    
    # Model metrics
    inertia: Optional[float] = None  # For K-Means
    bic: Optional[float] = None      # For GMM
    
    @property
    def n_regimes(self) -> int:
        """Number of regimes detected."""
        return len(np.unique(self.labels[~np.isnan(self.labels)]))
    
    def get_regime_at(self, idx: int) -> Optional[str]:
        """Get regime name at specific index."""
        if idx >= len(self.labels) or np.isnan(self.labels[idx]):
            return None
        label = int(self.labels[idx])
        return self.regime_mapping.get(label, f"regime_{label}")


class RegimeClassifier:
    """
    Market Regime Classifier using Unsupervised Learning.
    
    Uses clustering (K-Means or GMM) to identify 4 distinct regimes:
    - Input features: ATR (volatility), ADX (trend strength), Returns
    - Output: regime_label (0-3) for every timestamp
    
    The regimes are automatically interpreted based on cluster centers.
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize regime classifier.
        
        Args:
            config: Classification configuration
        """
        self.config = config or RegimeConfig()
        self.scaler = StandardScaler()
        self.model = None
        self._is_fitted = False
        
        logger.info(
            f"RegimeClassifier initialized: {self.config.n_regimes} regimes, "
            f"method={self.config.method}"
        )
    
    def fit(self, df: pd.DataFrame) -> "RegimeClassifier":
        """
        Fit the regime classifier on historical data.
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            
        Returns:
            self (fitted classifier)
        """
        # Compute features
        features, feature_names = self._compute_features(df)
        
        # Remove NaN rows
        valid_mask = ~np.any(np.isnan(features), axis=1)
        features_clean = features[valid_mask]
        
        if len(features_clean) < self.config.n_regimes * 10:
            raise ValueError(
                f"Insufficient data for {self.config.n_regimes} regimes. "
                f"Need at least {self.config.n_regimes * 10} valid rows."
            )
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # Fit clustering model
        if self.config.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=self.config.random_state,
                n_init=self.config.n_init,
            )
            self.model.fit(features_scaled)
            
        elif self.config.method == "gmm":
            self.model = GaussianMixture(
                n_components=self.config.n_regimes,
                random_state=self.config.random_state,
                n_init=self.config.n_init,
            )
            self.model.fit(features_scaled)
            
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        self._is_fitted = True
        self._feature_names = feature_names
        
        logger.info(f"RegimeClassifier fitted on {len(features_clean)} samples")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> RegimeResult:
        """
        Predict regime labels for data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            RegimeResult with labels and metadata
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        # Compute features
        features, feature_names = self._compute_features(df)
        
        # Initialize labels with NaN
        labels = np.full(len(df), np.nan)
        
        # Find valid rows
        valid_mask = ~np.any(np.isnan(features), axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            features_clean = features[valid_mask]
            features_scaled = self.scaler.transform(features_clean)
            
            # Predict
            if self.config.method == "kmeans":
                predicted_labels = self.model.predict(features_scaled)
            else:  # gmm
                predicted_labels = self.model.predict(features_scaled)
            
            labels[valid_indices] = predicted_labels
        
        # Get cluster centers
        if self.config.method == "kmeans":
            centers = self.scaler.inverse_transform(self.model.cluster_centers_)
            inertia = self.model.inertia_
            bic = None
        else:
            centers = self.scaler.inverse_transform(self.model.means_)
            inertia = None
            bic = self.model.bic(features_scaled) if len(valid_indices) > 0 else None
        
        # Interpret regimes based on centers
        regime_mapping = self._interpret_regimes(centers, feature_names)
        
        return RegimeResult(
            labels=labels,
            centers=centers,
            feature_names=feature_names,
            regime_mapping=regime_mapping,
            inertia=inertia,
            bic=bic,
        )
    
    def fit_predict(self, df: pd.DataFrame) -> RegimeResult:
        """
        Fit and predict in one step.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            RegimeResult
        """
        self.fit(df)
        return self.predict(df)
    
    def _compute_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute clustering features from OHLCV data.
        
        Features:
        - ATR (normalized by price) - Volatility
        - ADX - Trend strength
        - Rolling return - Direction
        """
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        
        n = len(df)
        
        # ATR (normalized by price)
        atr = calculate_atr(high, low, close, self.config.atr_period)
        atr_pct = atr / close * 100
        
        # ADX (trend strength)
        adx = calculate_adx(high, low, close, self.config.adx_period)
        
        # Rolling return
        returns = np.zeros(n)
        returns[self.config.return_lookback:] = (
            (close[self.config.return_lookback:] - close[:-self.config.return_lookback]) /
            close[:-self.config.return_lookback] * 100
        )
        returns[:self.config.return_lookback] = np.nan
        
        # Rolling volatility (std of returns)
        rolling_vol = np.zeros(n)
        for i in range(self.config.volatility_lookback, n):
            period_returns = np.diff(close[i - self.config.volatility_lookback:i + 1]) / close[i - self.config.volatility_lookback:i]
            rolling_vol[i] = np.std(period_returns) * 100
        rolling_vol[:self.config.volatility_lookback] = np.nan
        
        features = np.column_stack([
            atr_pct,
            adx,
            returns,
            rolling_vol,
        ])
        
        feature_names = [
            "atr_pct",
            "adx",
            "return_20",
            "rolling_vol_20",
        ]
        
        return features, feature_names
    
    def _interpret_regimes(
        self,
        centers: np.ndarray,
        feature_names: List[str]
    ) -> Dict[int, str]:
        """
        Interpret cluster centers to assign regime names.
        
        Uses return (direction) and volatility (atr_pct) to classify:
        - Bull vs Bear: based on return
        - Quiet vs Volatile: based on atr_pct
        """
        n_regimes = len(centers)
        
        # Find feature indices
        try:
            return_idx = feature_names.index("return_20")
            vol_idx = feature_names.index("atr_pct")
        except ValueError:
            # Fallback to generic naming
            return {i: f"regime_{i}" for i in range(n_regimes)}
        
        # Get median values for classification threshold
        median_return = np.median(centers[:, return_idx])
        median_vol = np.median(centers[:, vol_idx])
        
        regime_mapping = {}
        
        for i, center in enumerate(centers):
            is_bull = center[return_idx] > median_return
            is_volatile = center[vol_idx] > median_vol
            
            if is_bull and not is_volatile:
                regime_mapping[i] = RegimeType.BULL_QUIET.value
            elif is_bull and is_volatile:
                regime_mapping[i] = RegimeType.BULL_VOLATILE.value
            elif not is_bull and not is_volatile:
                regime_mapping[i] = RegimeType.BEAR_QUIET.value
            else:
                regime_mapping[i] = RegimeType.BEAR_VOLATILE.value
        
        logger.info(f"Regime interpretation: {regime_mapping}")
        
        return regime_mapping
    
    def get_regime_statistics(
        self,
        df: pd.DataFrame,
        result: RegimeResult
    ) -> pd.DataFrame:
        """
        Get statistics for each regime.
        
        Args:
            df: OHLCV DataFrame
            result: RegimeResult from predict()
            
        Returns:
            DataFrame with regime statistics
        """
        stats = []
        
        for label in range(self.config.n_regimes):
            mask = result.labels == label
            if not np.any(mask):
                continue
            
            regime_data = df[mask]
            returns = regime_data["close"].pct_change().dropna()
            
            stats.append({
                "regime": result.regime_mapping.get(label, f"regime_{label}"),
                "label": label,
                "count": int(np.sum(mask)),
                "pct_of_data": float(np.sum(mask) / len(df) * 100),
                "avg_return": float(returns.mean() * 100) if len(returns) > 0 else 0,
                "volatility": float(returns.std() * 100) if len(returns) > 0 else 0,
                "sharpe": float(returns.mean() / returns.std()) if len(returns) > 0 and returns.std() > 0 else 0,
            })
        
        return pd.DataFrame(stats)
    
    def add_regime_column(
        self,
        df: pd.DataFrame,
        result: RegimeResult,
        column_name: str = "regime"
    ) -> pd.DataFrame:
        """
        Add regime labels to DataFrame.
        
        Args:
            df: OHLCV DataFrame
            result: RegimeResult
            column_name: Name for regime column
            
        Returns:
            DataFrame with regime column added
        """
        df_copy = df.copy()
        df_copy[f"{column_name}_label"] = result.labels
        df_copy[column_name] = [
            result.regime_mapping.get(int(l), "unknown") if not np.isnan(l) else "unknown"
            for l in result.labels
        ]
        return df_copy
    
    def get_transition_matrix(
        self,
        result: RegimeResult
    ) -> pd.DataFrame:
        """
        Compute regime transition probability matrix.
        
        Args:
            result: RegimeResult
            
        Returns:
            DataFrame with transition probabilities
        """
        labels = result.labels
        n_regimes = self.config.n_regimes
        
        # Count transitions
        transitions = np.zeros((n_regimes, n_regimes))
        
        valid_labels = labels[~np.isnan(labels)].astype(int)
        
        for i in range(len(valid_labels) - 1):
            from_regime = valid_labels[i]
            to_regime = valid_labels[i + 1]
            transitions[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        probs = transitions / row_sums
        
        # Create DataFrame
        regime_names = [
            result.regime_mapping.get(i, f"regime_{i}")
            for i in range(n_regimes)
        ]
        
        return pd.DataFrame(probs, index=regime_names, columns=regime_names)


def detect_regimes(
    df: pd.DataFrame,
    n_regimes: int = 4,
    method: str = "kmeans"
) -> Tuple[RegimeResult, RegimeClassifier]:
    """
    Convenience function to detect regimes.
    
    Args:
        df: OHLCV DataFrame
        n_regimes: Number of regimes (default 4)
        method: Clustering method ("kmeans" or "gmm")
        
    Returns:
        Tuple of (RegimeResult, fitted RegimeClassifier)
    """
    config = RegimeConfig(n_regimes=n_regimes, method=method)
    classifier = RegimeClassifier(config=config)
    result = classifier.fit_predict(df)
    
    return result, classifier
