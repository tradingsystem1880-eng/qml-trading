"""
XGBoost Predictor - ML Signal Classifier
=========================================
Machine learning infrastructure for classifying trade signals.

Trains on historical trade outcomes to predict probability
of success for new signals.

Features extracted:
- Pattern validity score
- ATR-relative position
- Time-based features (hour, day of week)
- Recent volatility measures
- Price momentum indicators
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# sklearn for proper ML utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class XGBoostPredictor:
    """
    XGBoost-based trade outcome predictor.
    
    Trains a binary classifier to predict WIN/LOSS probability
    for trade signals based on extracted features.
    
    Usage:
        predictor = XGBoostPredictor()
        
        # Prepare training data
        X, y = predictor.prepare_data(trades_df)
        
        # Train model
        predictor.train(X, y)
        
        # Save model
        predictor.save('results/models/xgb_latest.json')
        
        # Predict on new signal
        prob = predictor.predict(signal_dict)
    """
    
    # Feature columns to extract
    FEATURE_COLUMNS = [
        'validity_score',
        'entry_price',
        'atr_at_entry',
        'sl_distance_pct',
        'tp_distance_pct',
        'risk_reward_ratio',
        'hour_of_day',
        'day_of_week',
        'volatility_percentile',
        'trend_strength',
        'bars_since_last_trade'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Optional path to load pre-trained model
        """
        if not HAS_XGBOOST:
            # Still allow class instantiation for structure
            print("Warning: XGBoost not installed. pip install xgboost")
        
        self.model = None
        self.feature_importance = {}
        self.training_metrics = {}
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def prepare_data(
        self,
        trades_df: pd.DataFrame,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Convert trades to feature matrix and labels.
        
        Args:
            trades_df: DataFrame of executed trades
            df: Optional OHLCV data for additional features
        
        Returns:
            X: Feature DataFrame
            y: Binary labels (1=WIN, 0=LOSS)
        """
        features = []
        labels = []
        
        for idx, trade in trades_df.iterrows():
            # Extract features from trade
            feature_row = self._extract_features(trade, df)
            features.append(feature_row)
            
            # Label: 1 for WIN, 0 for LOSS/BREAKEVEN
            result = trade.get('result', '')
            labels.append(1 if result == 'WIN' else 0)
        
        X = pd.DataFrame(features)
        y = np.array(labels)
        
        return X, y
    
    def _extract_features(
        self,
        trade: pd.Series,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Extract features from a single trade.
        
        Args:
            trade: Trade Series/dict
            df: Optional OHLCV data
        
        Returns:
            Dictionary of features
        """
        entry_price = trade.get('entry_price', 0)
        stop_loss = trade.get('stop_loss', 0)
        take_profit = trade.get('take_profit', 0)
        
        # Calculate SL/TP distances
        sl_distance = abs(entry_price - stop_loss) / entry_price * 100 if entry_price else 0
        tp_distance = abs(take_profit - entry_price) / entry_price * 100 if entry_price else 0
        
        # Risk/reward ratio
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        
        # Time features
        entry_time = pd.to_datetime(trade.get('entry_time'))
        hour = entry_time.hour if entry_time else 0
        day_of_week = entry_time.dayofweek if entry_time else 0
        
        return {
            'validity_score': trade.get('validity_score', 0.5),
            'entry_price': entry_price,
            'atr_at_entry': trade.get('atr_at_entry', 0),
            'sl_distance_pct': sl_distance,
            'tp_distance_pct': tp_distance,
            'risk_reward_ratio': rr_ratio,
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'volatility_percentile': trade.get('volatility_percentile', 50),
            'trend_strength': trade.get('trend_strength', 0),
            'bars_since_last_trade': trade.get('bars_since_last', 0),
            'pattern_type_bullish': 1 if 'BULLISH' in str(trade.get('pattern_type', '')) else 0,
            'side_long': 1 if trade.get('side', '') == 'LONG' else 0
        }
    
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        **xgb_params
    ) -> Dict[str, float]:
        """
        Train the XGBoost classifier.
        
        Args:
            X: Feature matrix
            y: Binary labels
            test_size: Fraction for validation
            **xgb_params: XGBoost parameters
        
        Returns:
            Dictionary of training metrics
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required: pip install xgboost")
        
        # Fill NaN values
        X = X.fillna(0)
        self._feature_names = X.columns.tolist()

        # Stratified train/test split using sklearn
        # Handles edge cases and ensures reproducibility
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # Stratification fails if a class has too few samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Default parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        params.update(xgb_params)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self._feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self._feature_names)
        
        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtest, 'test')],
            verbose_eval=False
        )
        
        # Predictions
        y_prob = self.model.predict(dtest)
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate metrics using sklearn (exact, O(n log n))
        accuracy = accuracy_score(y_test, y_pred)

        # AUC calculation using sklearn
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            # Only one class present in y_test
            auc = 0.5
        
        self.training_metrics = {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'positive_rate': float(np.mean(y))
        }
        
        # Feature importance
        importance = self.model.get_score(importance_type='weight')
        self.feature_importance = {k: float(v) for k, v in importance.items()}
        
        return self.training_metrics
    
    def predict(self, signal: Dict[str, Any]) -> float:
        """
        Predict probability of WIN for a new signal.
        
        Args:
            signal: Signal dictionary
        
        Returns:
            Probability score (0.0 to 1.0)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create mock trade from signal for feature extraction
        mock_trade = pd.Series({
            'entry_price': signal.get('price', 0),
            'stop_loss': signal.get('stop_loss', 0),
            'take_profit': signal.get('take_profit', 0),
            'validity_score': signal.get('validity_score', 0.5),
            'entry_time': signal.get('timestamp', datetime.now()),
            'pattern_type': signal.get('pattern_type', ''),
            'side': 'LONG' if signal.get('signal_type') == 'BUY' else 'SHORT'
        })
        
        features = self._extract_features(mock_trade)
        X = pd.DataFrame([features]).fillna(0)
        
        # Ensure columns match training
        if hasattr(self, '_feature_names'):
            for col in self._feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self._feature_names]
        
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        prob = self.model.predict(dmatrix)[0]
        return float(prob)
    
    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of features to return
        
        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:n]
    
    def save(self, path: str) -> None:
        """
        Save model to JSON.
        
        Args:
            path: Output path
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(path))
        
        # Save metadata
        meta_path = path.with_suffix('.meta.json')
        meta = {
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'feature_names': getattr(self, '_feature_names', []),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def load(self, path: str) -> None:
        """
        Load model from JSON.
        
        Args:
            path: Model path
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost required")
        
        self.model = xgb.Booster()
        self.model.load_model(path)
        
        # Load metadata if exists
        meta_path = Path(path).with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.training_metrics = meta.get('training_metrics', {})
                self.feature_importance = meta.get('feature_importance', {})
                self._feature_names = meta.get('feature_names', [])

