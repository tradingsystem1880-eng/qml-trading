#!/usr/bin/env python3
"""
Phase 8.0: ML Meta-Labeling Pipeline (FIXED)
=============================================
CRITICAL FIX: Proper train/test split to prevent overfitting.

The original pipeline had a fatal flaw:
- Trained ML on ALL trades
- Tested ML on those SAME trades
- Result: Unrealistic PF 7.11 (model memorized outcomes)

This version:
- Splits data 70/30 by TIME (not random)
- Trains only on first 70%
- Tests only on held-out 30%
- Expected result: Realistic 10-30% improvement if ML adds value

Usage:
    python scripts/run_phase80_ml_fixed.py --full-pipeline
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.extended_runner import (
    ExtendedDetectionRunner,
    ExtendedRunnerConfig,
    ALL_CLUSTERED_SYMBOLS,
)
from src.optimization.trade_simulator import TradeSimulator, TradeManagementConfig, SimulationResult
from src.ml.meta_trainer import MetaTrainer, MetaTrainerConfig, create_labels_from_trades
from src.ml.kelly_sizer import KellySizer, KellyConfig
from src.ml.production_gate import ProductionGate, ProductionGateConfig


def load_phase79_params(params_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load best parameters from Phase 7.9 optimization."""
    if params_path is None:
        params_path = PROJECT_ROOT / 'results/phase77_optimization/profit_factor_penalized/final_results.json'

    if not params_path.exists():
        raise FileNotFoundError(f"Phase 7.9 results not found: {params_path}")

    with open(params_path) as f:
        data = json.load(f)

    return data['best_params'], data['best_simulation']


def run_detection_and_simulation(
    params: Dict[str, Any],
    symbols: List[str],
    timeframe: str,
    n_jobs: int = 2,
) -> Tuple[List[Dict], SimulationResult, pd.DataFrame]:
    """Run pattern detection and trade simulation."""
    print(f"\n{'='*50}")
    print(f"RUNNING DETECTION & SIMULATION")
    print(f"{'='*50}")
    print(f"Symbols: {len(symbols)}")
    print(f"Timeframe: {timeframe}")

    config = ExtendedRunnerConfig()
    runner = ExtendedDetectionRunner(
        symbols=symbols,
        timeframes=[timeframe],
        config=config,
        n_jobs=n_jobs,
    )

    print("Preloading data...")
    runner.preload_data()

    try:
        obj_result, detection_result, sim_result = runner.run_with_dict_extended(params)
    except Exception as e:
        print(f"Detection failed: {e}")
        return [], SimulationResult(trades=[]), pd.DataFrame()

    # Convert trades to dicts
    trades = []
    for trade in sim_result.trades:
        trade_dict = trade.to_dict()
        trade_dict['hit_tp_before_sl'] = (
            trade.exit_reason and
            trade.exit_reason.value == 'take_profit'
        )
        trades.append(trade_dict)

    print(f"Detected: {detection_result.total_patterns} patterns")
    print(f"Trades: {len(trades)}")
    print(f"Win Rate: {sim_result.win_rate:.1%}")
    print(f"Profit Factor: {sim_result.profit_factor:.4f}")

    feature_df = extract_trade_features(trades, sim_result)

    return trades, sim_result, feature_df


def extract_trade_features(
    trades: List[Dict],
    sim_result: SimulationResult,
) -> pd.DataFrame:
    """Extract features for ML training from trade data."""
    features = []

    for i, trade in enumerate(trades):
        feat = {
            'pattern_score': trade.get('pattern_score', 0.5),
            'risk_reward': trade.get('take_profit', 0) / max(trade.get('stop_loss', 1), 0.001) if trade.get('stop_loss') else 0,
            'atr_at_entry': trade.get('atr_at_entry', 0),
            'entry_hour': pd.Timestamp(trade.get('entry_time')).hour if trade.get('entry_time') else 12,
            'entry_dow': pd.Timestamp(trade.get('entry_time')).dayofweek if trade.get('entry_time') else 3,
            'recent_win_rate': _calc_recent_win_rate(trades, i, window=10),
            'recent_avg_r': _calc_recent_avg_r(trades, i, window=10),
            'is_long': 1 if trade.get('side') == 'LONG' else 0,
        }
        features.append(feat)

    return pd.DataFrame(features)


def _calc_recent_win_rate(trades: List[Dict], current_idx: int, window: int = 10) -> float:
    start = max(0, current_idx - window)
    recent = trades[start:current_idx]
    if not recent:
        return 0.5
    wins = sum(1 for t in recent if t.get('result') == 'WIN')
    return wins / len(recent)


def _calc_recent_avg_r(trades: List[Dict], current_idx: int, window: int = 10) -> float:
    start = max(0, current_idx - window)
    recent = trades[start:current_idx]
    if not recent:
        return 0.0
    r_mults = [t.get('pnl_r', 0) for t in recent]
    return np.mean(r_mults)


def to_native(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


def run_proper_ml_validation(
    trades: List[Dict],
    features: pd.DataFrame,
    output_dir: Path,
    train_ratio: float = 0.7,
) -> Dict:
    """
    Run ML training with PROPER train/test split.

    CRITICAL: We split by TIME, not randomly.
    Train on first 70%, test on last 30%.
    """
    print(f"\n{'='*60}")
    print(f"PROPER ML VALIDATION (TRAIN/TEST SPLIT)")
    print(f"{'='*60}")

    n_total = len(trades)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    print(f"\nTotal trades: {n_total}")
    print(f"Train set: {n_train} (first {train_ratio:.0%})")
    print(f"Test set: {n_test} (last {1-train_ratio:.0%})")

    # Split by time (trades are already sorted by entry_time)
    train_trades = trades[:n_train]
    test_trades = trades[n_train:]

    train_features = features.iloc[:n_train]
    test_features = features.iloc[n_train:]

    # Create labels
    train_labels, train_weights = create_labels_from_trades(train_trades, use_quality_labels=True)
    test_labels, test_weights = create_labels_from_trades(test_trades, use_quality_labels=True)

    print(f"\nTrain: {train_labels.sum()}/{len(train_labels)} positive ({train_labels.mean():.1%})")
    print(f"Test:  {test_labels.sum()}/{len(test_labels)} positive ({test_labels.mean():.1%})")

    # ========================================
    # TRAIN MODEL (on train set only)
    # ========================================
    print(f"\n--- Training Model (on train set only) ---")

    config = MetaTrainerConfig(
        n_folds=5,
        max_features=15,
        min_train_samples=30,
    )
    trainer = MetaTrainer(config)

    result = trainer.train(train_features, train_labels)

    if not result.success or result.model is None:
        print(f"Training FAILED: {result.failure_reason}")
        return {'success': False, 'reason': result.failure_reason}

    model = result.model
    selected_features = result.selected_features

    print(f"Training AUC: {result.mean_auc:.4f}")
    print(f"Selected features: {selected_features}")

    # ========================================
    # TEST MODEL (on held-out test set)
    # ========================================
    print(f"\n--- Testing Model (on held-out test set) ---")

    # Get predictions on TEST set only
    X_test = test_features[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_confidences = model.predict_proba(X_test)[:, 1]

    # Calculate test AUC
    from sklearn.metrics import roc_auc_score
    try:
        test_auc = roc_auc_score(test_labels, test_confidences)
    except Exception:
        test_auc = 0.5  # If all same class

    print(f"Test AUC: {test_auc:.4f}")

    # ========================================
    # COMPARE BASELINE VS ML ON TEST SET
    # ========================================
    print(f"\n--- Production Gate (Test Set Only) ---")

    # Baseline metrics from train set (what model learned from)
    train_wins = sum(1 for t in train_trades if t.get('result') == 'WIN')
    train_losses = sum(1 for t in train_trades if t.get('result') == 'LOSS')
    train_win_rate = train_wins / len(train_trades) if train_trades else 0
    train_avg_win_r = np.mean([t.get('pnl_r', 0) for t in train_trades if t.get('result') == 'WIN']) if train_wins > 0 else 0
    train_avg_loss_r = abs(np.mean([t.get('pnl_r', 0) for t in train_trades if t.get('result') == 'LOSS'])) if train_losses > 0 else 1

    # Kelly position sizing for test trades
    kelly = KellySizer(KellyConfig())
    position_sizes = kelly.calculate_batch(
        confidences=test_confidences,
        win_rate=train_win_rate,
        avg_win_r=train_avg_win_r,
        avg_loss_r=train_avg_loss_r,
        account_equity=100_000,
    )

    # Run production gate on TEST SET ONLY
    gate = ProductionGate()

    # Calculate baseline metrics from test set
    test_pnls = [t.get('pnl_r', 0) for t in test_trades]
    test_wins_pnl = sum(p for p in test_pnls if p > 0)
    test_losses_pnl = abs(sum(p for p in test_pnls if p < 0))
    baseline_pf = test_wins_pnl / test_losses_pnl if test_losses_pnl > 0 else 0

    baseline_metrics = {
        'win_rate': train_win_rate,
        'avg_win_r': train_avg_win_r,
        'avg_loss_r': train_avg_loss_r,
    }

    gate_result = gate.run_gate(
        trades=test_trades,
        ml_confidences=test_confidences,
        position_sizes=position_sizes,
        baseline_metrics=baseline_metrics,
    )

    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print(f"PROPER VALIDATION RESULTS")
    print(f"{'='*60}")

    print(f"\n| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Train AUC | {result.mean_auc:.4f} |")
    print(f"| Test AUC | {test_auc:.4f} |")
    print(f"| Test Baseline PF | {gate_result.baseline_pf:.4f} |")
    print(f"| Test ML PF | {gate_result.ml_pf:.4f} |")
    print(f"| PF Improvement | {gate_result.pf_improvement_pct:+.1f}% |")
    print(f"| Test Trades (baseline) | {gate_result.baseline_trades} |")
    print(f"| Test Trades (ML) | {gate_result.ml_trades} |")

    # Check for realistic improvement
    is_realistic = (
        gate_result.ml_pf < 3.0 and  # Not too good to be true
        abs(test_auc - result.mean_auc) < 0.1 and  # Train/test AUC similar
        gate_result.pf_improvement_pct < 100  # Not 500%+ improvement
    )

    if is_realistic and gate_result.ml_pf > gate_result.baseline_pf:
        print(f"\n✓ ML shows REALISTIC improvement")
        decision = 'DEPLOY_ML'
    elif gate_result.ml_pf < 3.0 and test_auc > 0.55:
        print(f"\n⚠️  ML shows MARGINAL improvement")
        decision = 'CAUTIOUS_DEPLOY'
    else:
        print(f"\n❌ ML improvement is NOT REALISTIC or NOT CONSISTENT")
        decision = 'USE_BASELINE'

    print(f"\n>>> DECISION: {decision} <<<")

    # Save results
    results = {
        'train_test_split': {
            'train_size': n_train,
            'test_size': n_test,
            'train_ratio': train_ratio,
        },
        'training': {
            'train_auc': result.mean_auc,
            'selected_features': selected_features,
        },
        'testing': {
            'test_auc': test_auc,
            'baseline_pf': gate_result.baseline_pf,
            'ml_pf': gate_result.ml_pf,
            'pf_improvement_pct': gate_result.pf_improvement_pct,
            'baseline_trades': gate_result.baseline_trades,
            'ml_trades': gate_result.ml_trades,
        },
        'decision': decision,
        'is_realistic': is_realistic,
    }

    with open(output_dir / 'proper_validation_result.json', 'w') as f:
        json.dump(to_native(results), f, indent=2)

    print(f"\nResults saved to: {output_dir / 'proper_validation_result.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 8.0 ML Meta-Labeling (FIXED)')
    parser.add_argument('--full-pipeline', action='store_true', help='Run complete pipeline')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated symbols (default: all 22 from Phase 7.9)')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe')
    parser.add_argument('--n-jobs', type=int, default=2, help='Parallel jobs')
    parser.add_argument('--output', type=str, default='results/phase80_ml_fixed',
                        help='Output directory')

    args = parser.parse_args()

    # Use all 22 symbols from Phase 7.9 by default
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = ALL_CLUSTERED_SYMBOLS

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PHASE 8.0: ML META-LABELING (FIXED VERSION)")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Output: {output_dir}")
    print(f"\nCRITICAL FIX: Proper 70/30 train/test split by time")

    # Load Phase 7.9 params
    try:
        params, baseline_sim = load_phase79_params()
        print(f"\nLoaded Phase 7.9 params")
        print(f"Original PF: {baseline_sim.get('profit_factor', 0):.4f}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Run detection
    trades, sim_result, features = run_detection_and_simulation(
        params=params,
        symbols=symbols,
        timeframe=args.timeframe,
        n_jobs=args.n_jobs,
    )

    if not trades:
        print("ERROR: No trades generated.")
        sys.exit(1)

    # Run proper ML validation with train/test split
    results = run_proper_ml_validation(
        trades=trades,
        features=features,
        output_dir=output_dir,
        train_ratio=0.7,
    )

    print(f"\n{'='*60}")
    print(f"PHASE 8.0 (FIXED) COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
