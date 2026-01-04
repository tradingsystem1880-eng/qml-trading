"""
Experiment Tracker
==================
Tracks experiment runs with automatic versioning, artifact storage,
and integration with VRD database.
"""

import hashlib
import json
import os
import pickle
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.validation.database import VRDDatabase, ExperimentRecord, generate_param_hash


class ExperimentTracker:
    """
    Tracks and versions experiment runs.
    
    Features:
    - Automatic git hash extraction
    - Immutable artifact storage with [TIMESTAMP]_[STRATEGY]_[PARAM_HASH] pattern
    - Fold-by-fold result logging
    - Integration with VRD database
    """
    
    def __init__(
        self,
        base_dir: str = "experiments",
        db: Optional[VRDDatabase] = None,
        db_path: Optional[str] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            base_dir: Base directory for experiment artifacts
            db: Existing VRDDatabase instance
            db_path: Path to database file (creates new if db not provided)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if db is not None:
            self.db = db
        else:
            db_file = db_path or str(self.base_dir / "vrd.db")
            self.db = VRDDatabase(db_file)
        
        # Current experiment state
        self._current_experiment: Optional[ExperimentRecord] = None
        self._current_dir: Optional[Path] = None
        self._fold_results: List[Dict] = []
        
        logger.info(f"ExperimentTracker initialized at {self.base_dir}")
    
    def start_experiment(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        data_range: Tuple[str, str],
        seed: int = 42,
        fold_count: int = 10
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            strategy_name: Name of strategy variant (e.g., "QML_BULLISH_V1")
            params: Parameter dictionary
            data_range: (start_date, end_date) tuple in ISO format
            seed: Random seed for reproducibility
            fold_count: Number of walk-forward folds
            
        Returns:
            experiment_id
        """
        # Generate identifiers
        experiment_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_hash = generate_param_hash(params)
        git_hash = self._get_git_hash()
        
        # Create experiment directory
        dir_name = f"{timestamp}_{strategy_name}_{param_hash}"
        self._current_dir = self.base_dir / dir_name
        self._current_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment record
        self._current_experiment = ExperimentRecord(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            git_hash=git_hash,
            strategy_name=strategy_name,
            param_hash=param_hash,
            params=params,
            data_start=data_range[0],
            data_end=data_range[1],
            random_seed=seed,
            fold_count=fold_count,
            artifact_path=str(self._current_dir),
            status="running",
        )
        
        # Save initial state
        self._fold_results = []
        self._save_params(params)
        self._save_metadata()
        
        # Insert into database
        self.db.insert_experiment(self._current_experiment)
        
        logger.info(
            f"Started experiment {experiment_id}: "
            f"{strategy_name} | {param_hash} | seed={seed}"
        )
        
        return experiment_id
    
    def log_fold_result(
        self,
        fold_idx: int,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        optimal_params: Dict[str, Any],
        in_sample_metrics: Dict[str, float],
        out_of_sample_metrics: Dict[str, float]
    ) -> None:
        """
        Log results from a single walk-forward fold.
        
        Args:
            fold_idx: Fold index (0-based)
            train_start: Training period start (ISO format)
            train_end: Training period end (ISO format)
            test_start: Test period start (ISO format)
            test_end: Test period end (ISO format)
            optimal_params: Parameters selected for this fold
            in_sample_metrics: In-sample performance metrics
            out_of_sample_metrics: Out-of-sample validation metrics
        """
        if self._current_experiment is None:
            raise RuntimeError("No experiment started. Call start_experiment first.")
        
        fold_result = {
            "fold_idx": fold_idx,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "optimal_params": optimal_params,
            "in_sample": in_sample_metrics,
            "out_of_sample": out_of_sample_metrics,
        }
        
        self._fold_results.append(fold_result)
        
        # Save fold result to disk
        fold_file = self._current_dir / f"fold_{fold_idx:02d}.json"
        with open(fold_file, "w") as f:
            json.dump(fold_result, f, indent=2, default=str)
        
        logger.debug(
            f"Logged fold {fold_idx}: "
            f"IS Sharpe={in_sample_metrics.get('sharpe_ratio', 'N/A'):.3f}, "
            f"OOS Sharpe={out_of_sample_metrics.get('sharpe_ratio', 'N/A'):.3f}"
        )
    
    def save_artifact(
        self,
        name: str,
        data: Any,
        format: str = "auto"
    ) -> Path:
        """
        Save an artifact to the experiment directory.
        
        Args:
            name: Artifact name (without extension)
            data: Data to save (DataFrame, dict, or any picklable object)
            format: "csv", "json", "pickle", or "auto" (infer from data type)
            
        Returns:
            Path to saved artifact
        """
        if self._current_dir is None:
            raise RuntimeError("No experiment started. Call start_experiment first.")
        
        # Infer format if auto
        if format == "auto":
            if isinstance(data, pd.DataFrame):
                format = "csv"
            elif isinstance(data, (dict, list)):
                format = "json"
            else:
                format = "pickle"
        
        # Save based on format
        if format == "csv":
            path = self._current_dir / f"{name}.csv"
            if isinstance(data, pd.DataFrame):
                data.to_csv(path, index=True)
            else:
                pd.DataFrame(data).to_csv(path, index=True)
        
        elif format == "json":
            path = self._current_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        
        else:  # pickle
            path = self._current_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(data, f)
        
        logger.debug(f"Saved artifact: {path}")
        return path
    
    def finalize(
        self,
        metrics: Dict[str, float],
        regime_metrics: Optional[Dict[str, Any]] = None,
        statistical_results: Optional[Dict[str, Any]] = None
    ) -> ExperimentRecord:
        """
        Finalize the experiment and update database.
        
        Args:
            metrics: Final aggregated performance metrics
            regime_metrics: Performance breakdown by market regime
            statistical_results: Permutation, Monte Carlo, Bootstrap results
            
        Returns:
            Final ExperimentRecord
        """
        if self._current_experiment is None:
            raise RuntimeError("No experiment started. Call start_experiment first.")
        
        # Update experiment record
        exp = self._current_experiment
        
        # Core metrics
        exp.sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        exp.sortino_ratio = metrics.get("sortino_ratio", 0.0)
        exp.calmar_ratio = metrics.get("calmar_ratio", 0.0)
        exp.total_return_pct = metrics.get("total_return_pct", 0.0)
        exp.annualized_return_pct = metrics.get("annualized_return_pct", 0.0)
        exp.max_drawdown_pct = metrics.get("max_drawdown_pct", 0.0)
        exp.win_rate = metrics.get("win_rate", 0.0)
        exp.profit_factor = metrics.get("profit_factor", 0.0)
        exp.total_trades = int(metrics.get("total_trades", 0))
        exp.avg_trade_pnl = metrics.get("avg_trade_pnl", 0.0)
        exp.avg_holding_bars = metrics.get("avg_holding_bars", 0.0)
        
        # Statistical results
        if statistical_results:
            exp.sharpe_p_value = statistical_results.get("sharpe_p_value")
            exp.sharpe_percentile = statistical_results.get("sharpe_percentile")
            exp.var_95 = statistical_results.get("var_95")
            exp.var_99 = statistical_results.get("var_99")
            exp.kill_switch_prob = statistical_results.get("kill_switch_prob")
        
        # Regime breakdown
        exp.regime_metrics = regime_metrics or {}
        
        # Fold results
        exp.fold_results = self._fold_results
        
        # Mark as completed
        exp.status = "completed"
        
        # Save final state
        self._save_summary(metrics, regime_metrics, statistical_results)
        self.db.insert_experiment(exp)
        
        logger.info(
            f"Finalized experiment {exp.experiment_id}: "
            f"Sharpe={exp.sharpe_ratio:.3f}, "
            f"MaxDD={exp.max_drawdown_pct:.1f}%, "
            f"WinRate={exp.win_rate:.1%}"
        )
        
        # Reset state
        result = self._current_experiment
        self._current_experiment = None
        self._current_dir = None
        self._fold_results = []
        
        return result
    
    def fail(self, error_message: str) -> None:
        """
        Mark experiment as failed.
        
        Args:
            error_message: Description of failure
        """
        if self._current_experiment is None:
            return
        
        self._current_experiment.status = "failed"
        self._current_experiment.error_message = error_message
        
        self.db.update_experiment(
            self._current_experiment.experiment_id,
            {"status": "failed", "error_message": error_message}
        )
        
        logger.error(f"Experiment failed: {error_message}")
        
        self._current_experiment = None
        self._current_dir = None
        self._fold_results = []
    
    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir.parent),
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception as e:
            logger.warning(f"Could not get git hash: {e}")
        
        return "unknown"
    
    def _save_params(self, params: Dict[str, Any]) -> None:
        """Save parameters to experiment directory."""
        if self._current_dir is None:
            return
        
        params_file = self._current_dir / "params.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2, default=str)
    
    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        if self._current_dir is None or self._current_experiment is None:
            return
        
        meta = {
            "experiment_id": self._current_experiment.experiment_id,
            "timestamp": self._current_experiment.timestamp,
            "git_hash": self._current_experiment.git_hash,
            "strategy_name": self._current_experiment.strategy_name,
            "param_hash": self._current_experiment.param_hash,
            "data_start": self._current_experiment.data_start,
            "data_end": self._current_experiment.data_end,
            "random_seed": self._current_experiment.random_seed,
            "fold_count": self._current_experiment.fold_count,
        }
        
        meta_file = self._current_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
    
    def _save_summary(
        self,
        metrics: Dict[str, float],
        regime_metrics: Optional[Dict],
        statistical_results: Optional[Dict]
    ) -> None:
        """Save final experiment summary."""
        if self._current_dir is None:
            return
        
        summary = {
            "metrics": metrics,
            "regime_metrics": regime_metrics or {},
            "statistical_results": statistical_results or {},
            "fold_results_summary": self._calculate_fold_summary(),
        }
        
        summary_file = self._current_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _calculate_fold_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics across folds."""
        if not self._fold_results:
            return {}
        
        # Extract OOS Sharpe ratios
        oos_sharpes = [
            fr["out_of_sample"].get("sharpe_ratio", 0)
            for fr in self._fold_results
            if fr["out_of_sample"].get("sharpe_ratio") is not None
        ]
        
        is_sharpes = [
            fr["in_sample"].get("sharpe_ratio", 0)
            for fr in self._fold_results
            if fr["in_sample"].get("sharpe_ratio") is not None
        ]
        
        import numpy as np
        
        return {
            "n_folds": len(self._fold_results),
            "oos_sharpe_mean": float(np.mean(oos_sharpes)) if oos_sharpes else 0,
            "oos_sharpe_std": float(np.std(oos_sharpes)) if oos_sharpes else 0,
            "is_sharpe_mean": float(np.mean(is_sharpes)) if is_sharpes else 0,
            "is_sharpe_std": float(np.std(is_sharpes)) if is_sharpes else 0,
            "is_oos_ratio": (
                float(np.mean(oos_sharpes) / np.mean(is_sharpes))
                if is_sharpes and oos_sharpes and np.mean(is_sharpes) != 0
                else 0
            ),
        }
    
    @property
    def current_experiment_id(self) -> Optional[str]:
        """Get current experiment ID."""
        if self._current_experiment:
            return self._current_experiment.experiment_id
        return None
    
    @property
    def current_experiment_dir(self) -> Optional[Path]:
        """Get current experiment directory."""
        return self._current_dir


def create_experiment_tracker(
    base_dir: str = "experiments",
    db_path: Optional[str] = None
) -> ExperimentTracker:
    """
    Factory function for ExperimentTracker.
    
    Args:
        base_dir: Base directory for experiments
        db_path: Optional path to database file
        
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(base_dir=base_dir, db_path=db_path)
