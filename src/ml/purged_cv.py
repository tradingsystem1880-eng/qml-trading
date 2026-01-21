"""
Purged K-Fold Cross-Validation for financial time series.

Standard CV causes data leakage because:
1. Future data leaks into training (lookahead bias)
2. Overlapping samples share information

Purged CV fixes this by:
1. Using only past data to predict future
2. Adding embargo periods between train/test
3. Purging overlapping samples

Based on Lopez de Prado's "Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
import warnings
from typing import Iterator, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class PurgedFold:
    """A single fold from purged CV."""
    fold_number: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation.

    Unlike standard KFold:
    - Test set is always AFTER training set (no future leakage)
    - Embargo period between train and test prevents information leakage
    - Samples overlapping with test period are purged from training

    Args:
        n_splits: Number of folds
        embargo_pct: Fraction of data to embargo after training (0.01 = 1%)
        purge_pct: Fraction of training data to purge before test (0.01 = 1%)
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        timestamps: pd.Series = None
    ) -> Iterator[PurgedFold]:
        """
        Generate purged train/test splits.

        Args:
            X: Feature DataFrame
            y: Target Series (optional, not used for splitting)
            timestamps: Timestamp series for time-based splitting

        Yields:
            PurgedFold with train and test indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Validate timestamps if provided
        if timestamps is not None:
            if not timestamps.is_monotonic_increasing:
                warnings.warn(
                    "Timestamps are not strictly increasing. "
                    "This may cause data leakage. Consider sorting by timestamp first."
                )

        # Calculate fold size
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        for fold_num in range(self.n_splits):
            # Test set: one fold at a time, moving forward
            test_start = fold_num * fold_size
            test_end = min(test_start + fold_size, n_samples)

            # Training set: everything before test (minus embargo and purge)
            train_end = max(0, test_start - embargo_size - purge_size)
            train_start = 0

            if train_end <= train_start:
                continue  # Skip folds with no training data

            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]

            # Validate no data leakage if timestamps provided
            if timestamps is not None:
                self._validate_no_leakage(timestamps, train_indices, test_indices, fold_num + 1)

            yield PurgedFold(
                fold_number=fold_num + 1,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_indices),
                n_test=len(test_indices)
            )

    def _validate_no_leakage(
        self,
        timestamps: pd.Series,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        fold_num: int
    ):
        """Verify no future data in training set."""
        max_train_time = timestamps.iloc[train_indices].max()
        min_test_time = timestamps.iloc[test_indices].min()

        if max_train_time >= min_test_time:
            raise ValueError(
                f"Data leakage detected in fold {fold_num}: "
                f"Training data ({max_train_time}) occurs at or after "
                f"test data start ({min_test_time})"
            )

    def get_n_splits(self) -> int:
        return self.n_splits


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation (Expanding Window).

    More realistic for trading:
    - Train on all data up to point T
    - Test on next period
    - Expand training window and repeat

    FIXED: Embargo is now correctly placed BEFORE test period.

    Args:
        n_splits: Number of test periods
        test_size: Size of each test period (fraction or int)
        min_train_size: Minimum training set size (fraction or int)
        embargo_pct: Embargo BEFORE test (not after train start)
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.1,
        min_train_size: float = 0.3,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        timestamps: pd.Series = None
    ) -> Iterator[PurgedFold]:
        """
        Generate walk-forward splits.

        FIXED LOGIC:
        - train_end = test_start - embargo (embargo BEFORE test)
        - Training grows with each fold (expanding window)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Validate timestamps
        if timestamps is not None and not timestamps.is_monotonic_increasing:
            warnings.warn("Timestamps are not strictly increasing.")

        # Convert fractions to counts
        test_size = int(self.test_size * n_samples) if self.test_size < 1 else int(self.test_size)
        min_train = int(self.min_train_size * n_samples) if self.min_train_size < 1 else int(self.min_train_size)
        embargo = int(self.embargo_pct * n_samples)

        # Calculate available space for test periods
        available_for_test = n_samples - min_train
        step_size = (available_for_test - test_size) // max(1, self.n_splits - 1) if self.n_splits > 1 else 0

        for fold_num in range(self.n_splits):
            # FIXED: Test starts at min_train + (fold * step), no embargo added here
            test_start = min_train + (fold_num * step_size)
            test_end = min(test_start + test_size, n_samples)

            # FIXED: Embargo is BEFORE test, so train_end = test_start - embargo
            train_end = test_start - embargo
            train_start = 0

            # Validate we have enough data
            if train_end <= train_start or test_end <= test_start:
                continue

            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]

            # Validate no data leakage
            if timestamps is not None:
                max_train_time = timestamps.iloc[train_indices].max()
                min_test_time = timestamps.iloc[test_indices].min()

                if max_train_time >= min_test_time:
                    raise ValueError(
                        f"Data leakage in fold {fold_num + 1}: "
                        f"max train time ({max_train_time}) >= min test time ({min_test_time})"
                    )

            yield PurgedFold(
                fold_number=fold_num + 1,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_indices),
                n_test=len(test_indices)
            )

    def get_n_splits(self) -> int:
        return self.n_splits
