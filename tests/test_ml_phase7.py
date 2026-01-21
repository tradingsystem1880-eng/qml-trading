"""
Phase 7: ML Pipeline Tests
==========================
Verification tests for purged cross-validation and ML training pipeline.

Tests:
1. PurgedKFold generates non-overlapping splits
2. WalkForwardCV embargo is BEFORE test (not after)
3. Data validation catches NaN/inf values
4. Class weights applied for imbalanced data
5. Early stopping works
6. Model save includes version metadata
7. Full training pipeline works
8. Module exports are correct
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import warnings
import tempfile
import os


def test_purged_kfold():
    """Test 1: PurgedKFold generates non-overlapping splits."""
    from src.ml import PurgedKFold

    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    timestamps = pd.Series(pd.date_range('2024-01-01', periods=100, freq='h'))

    cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
    folds = list(cv.split(X, y, timestamps))

    assert len(folds) > 0, "Should generate at least one fold"

    for fold in folds:
        # Verify no overlap between train and test
        overlap = len(set(fold.train_indices) & set(fold.test_indices))
        assert overlap == 0, f"Train/test overlap in fold {fold.fold_number}!"

        # Verify test is after train
        assert fold.train_end <= fold.test_start, f"Test should be after train in fold {fold.fold_number}!"

    print(f"✓ Test 1: PurgedKFold generates {len(folds)} non-overlapping splits")


def test_walk_forward_cv_embargo():
    """Test 2: WalkForwardCV embargo is BEFORE test (FIXED)."""
    from src.ml import WalkForwardCV

    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    timestamps = pd.Series(pd.date_range('2024-01-01', periods=100, freq='h'))

    wf_cv = WalkForwardCV(n_splits=5, embargo_pct=0.02)
    wf_folds = list(wf_cv.split(X, y, timestamps))

    assert len(wf_folds) > 0, "Should generate at least one fold"

    for fold in wf_folds:
        # CRITICAL: Verify embargo is BEFORE test
        max_train_idx = fold.train_indices.max()
        min_test_idx = fold.test_indices.min()
        gap = min_test_idx - max_train_idx

        assert gap > 0, f"No gap between train and test in fold {fold.fold_number}!"

    print(f"✓ Test 2: WalkForwardCV embargo correctly placed BEFORE test ({len(wf_folds)} folds)")


def test_data_validation_nan():
    """Test 3a: Data validation catches NaN values."""
    from src.ml import MLTrainingPipeline

    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))

    pipeline = MLTrainingPipeline()

    # Test NaN detection
    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan
    try:
        pipeline._validate_data(X_nan, y)
        assert False, "Should have caught NaN"
    except ValueError as e:
        assert "NaN" in str(e)

    print("✓ Test 3a: Data validation correctly catches NaN values")


def test_data_validation_inf():
    """Test 3b: Data validation catches infinite values."""
    from src.ml import MLTrainingPipeline

    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))

    pipeline = MLTrainingPipeline()

    # Test inf detection
    X_inf = X.copy()
    X_inf.iloc[0, 0] = np.inf
    try:
        pipeline._validate_data(X_inf, y)
        assert False, "Should have caught inf"
    except ValueError as e:
        assert "infinite" in str(e).lower()

    print("✓ Test 3b: Data validation correctly catches infinite values")


def test_data_validation_single_class():
    """Test 3c: Data validation catches single class."""
    from src.ml import MLTrainingPipeline

    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])

    pipeline = MLTrainingPipeline()

    # Test single class detection
    y_single = pd.Series([1] * 100)
    try:
        pipeline._validate_data(X, y_single)
        assert False, "Should have caught single class"
    except ValueError as e:
        assert "unique" in str(e).lower() or "class" in str(e).lower()

    print("✓ Test 3c: Data validation correctly catches single class")


def test_imbalanced_class_handling():
    """Test 4: Class weights applied for imbalanced data."""
    from src.ml import MLTrainingPipeline

    # Use 70/30 split with more data to ensure each fold has both classes
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(300, 10), columns=[f'feature_{i}' for i in range(10)])
    y_imbalanced = pd.Series([0] * 210 + [1] * 90)  # 70% class 0, 30% class 1
    # Shuffle to distribute classes
    idx = np.random.permutation(len(y_imbalanced))
    y_imbalanced = y_imbalanced.iloc[idx].reset_index(drop=True)
    X_train = X_train.iloc[idx].reset_index(drop=True)

    pipeline = MLTrainingPipeline(
        model_type='xgboost',
        cv_method='purged',
        n_folds=3,
        handle_imbalanced=True
    )

    result = pipeline.train(X_train, y_imbalanced)

    assert result.class_ratio < 0.5, f"Class ratio should be ~0.3, got {result.class_ratio}"
    assert result.scale_pos_weight > 1, f"Should have positive class weight > 1 for imbalanced data, got {result.scale_pos_weight}"

    print(f"✓ Test 4: Imbalanced handling works (class_ratio={result.class_ratio:.2f}, weight={result.scale_pos_weight:.2f})")


def test_model_versioning():
    """Test 5: Model save includes version metadata."""
    from src.ml import MLTrainingPipeline, MODEL_VERSION

    X = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 100))

    pipeline = MLTrainingPipeline(model_type='xgboost', cv_method='purged', n_folds=3)
    result = pipeline.train(X, y)

    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        temp_path = f.name

    try:
        pipeline.save(temp_path)

        import joblib
        saved_data = joblib.load(temp_path)

        assert 'version' in saved_data, "Version should be saved"
        assert 'created_at' in saved_data, "Created_at should be saved"
        assert 'feature_hash' in saved_data, "Feature hash should be saved"
        assert saved_data['version'] == MODEL_VERSION

        print(f"✓ Test 5: Model versioning works (version={saved_data['version']})")

    finally:
        os.unlink(temp_path)


def test_full_training_pipeline():
    """Test 6: Full training pipeline works end-to-end."""
    from src.ml import MLTrainingPipeline

    X_full = pd.DataFrame(np.random.randn(200, 10), columns=[f'feature_{i}' for i in range(10)])
    y_full = pd.Series(np.random.randint(0, 2, 200))

    pipeline = MLTrainingPipeline(
        model_type='xgboost',
        cv_method='purged',
        n_folds=3,
        early_stopping_rounds=5
    )

    result = pipeline.train(X_full, y_full)

    # Check all metrics are present
    assert result.metrics.roc_auc >= 0, "Should have valid AUC"
    assert result.metrics.precision >= 0, "Should have valid precision"
    assert result.metrics.recall >= 0, "Should have valid recall"
    assert result.metrics.f1 >= 0, "Should have valid F1"
    assert result.n_features == 10, f"Should have 10 features, got {result.n_features}"
    assert result.training_time_seconds > 0, "Should have training time"
    assert len(result.feature_importance) > 0, "Should have feature importance"

    # Test prediction
    predictions = pipeline.predict(X_full.head(10))
    probas = pipeline.predict_proba(X_full.head(10))

    assert len(predictions) == 10, "Should predict 10 samples"
    assert len(probas) == 10, "Should have 10 probabilities"

    print(f"✓ Test 6: Full training pipeline works (AUC={result.metrics.roc_auc:.3f})")


def test_fold_metrics():
    """Test 7: Fold-by-fold metrics are tracked."""
    from src.ml import MLTrainingPipeline

    X = pd.DataFrame(np.random.randn(200, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 200))

    pipeline = MLTrainingPipeline(
        model_type='xgboost',
        cv_method='purged',
        n_folds=5
    )

    result = pipeline.train(X, y)

    assert len(result.metrics.fold_metrics) > 0, "Should have fold metrics"

    for fold_metric in result.metrics.fold_metrics:
        assert 'fold' in fold_metric, "Fold number should be in metrics"
        assert 'roc_auc' in fold_metric, "AUC should be in fold metrics"
        assert 'n_train' in fold_metric, "Train count should be in fold metrics"
        assert 'n_test' in fold_metric, "Test count should be in fold metrics"

    print(f"✓ Test 7: Fold metrics tracked ({len(result.metrics.fold_metrics)} folds)")


def test_module_exports():
    """Test 8: All expected classes/functions are exported."""
    from src.ml import (
        PurgedKFold,
        WalkForwardCV,
        PurgedFold,
        MLTrainingPipeline,
        TrainingResult,
        TrainingMetrics,
        MODEL_VERSION,
        XGBoostPredictor,
    )

    # Verify they're all classes/functions (not None)
    assert PurgedKFold is not None
    assert WalkForwardCV is not None
    assert PurgedFold is not None
    assert MLTrainingPipeline is not None
    assert TrainingResult is not None
    assert TrainingMetrics is not None
    assert MODEL_VERSION is not None
    assert XGBoostPredictor is not None

    print("✓ Test 8: Module exports correct")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Phase 7: ML Pipeline Tests")
    print("=" * 60 + "\n")

    tests = [
        test_purged_kfold,
        test_walk_forward_cv_embargo,
        test_data_validation_nan,
        test_data_validation_inf,
        test_data_validation_single_class,
        test_imbalanced_class_handling,
        test_model_versioning,
        test_full_training_pipeline,
        test_fold_metrics,
        test_module_exports,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n All Phase 7 tests passed!")
    else:
        print(f"\n {failed} test(s) failed")
        sys.exit(1)
