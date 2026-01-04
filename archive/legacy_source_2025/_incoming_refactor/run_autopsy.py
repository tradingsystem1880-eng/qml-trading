#!/usr/bin/env python3
"""
QML Strategy Autopsy - First Forensic Validation Run
=====================================================
Runs the complete validation pipeline on the QML strategy
and generates the first Deployment Dossier.

Usage:
    python run_autopsy.py
    
Output:
    results/autopsy_qml_v1/
    ├── vrd.db              # Experiment database
    ├── dossier_*.html      # HTML report
    ├── dossier_*.json      # JSON data
    └── [experiment_dir]/   # Artifacts
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.orchestrator import ValidationOrchestrator, OrchestratorConfig
from src.strategies.qml_adapter import run_qml_strategy
from src.reporting.dossier import DossierGenerator
from src.deployment.gatekeeper import DeploymentGatekeeper


def main():
    """Run the first forensic autopsy on QML strategy."""
    
    logger.info("=" * 70)
    logger.info("QML STRATEGY AUTOPSY - FORENSIC VALIDATION")
    logger.info("=" * 70)
    
    # Configuration
    OUTPUT_DIR = "results/autopsy_qml_v1"
    DATA_PATH = "data/processed/BTC/1h_master.parquet"
    STRATEGY_NAME = "QML_BULLISH_V1"
    
    # Parameter grid for walk-forward optimization
    PARAM_GRID = {
        "min_validity_score": [0.3, 0.4, 0.5],  # Lower thresholds
        "risk_reward": [2.0, 3.0],
        "stop_loss_atr": [1.5, 2.0],
        "risk_per_trade_pct": [1.0],
    }
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from: {DATA_PATH}")
    
    if not Path(DATA_PATH).exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.info("Attempting to use 4h data instead...")
        DATA_PATH = "data/processed/BTC/4h_master.parquet"
        
        if not Path(DATA_PATH).exists():
            logger.error("No data files found. Please run data download first.")
            return None
    
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df)} bars from {df['time'].min()} to {df['time'].max()}")
    
    # Normalize column names (handle capitalized OHLCV)
    column_mapping = {
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    }
    df = df.rename(columns=column_mapping)
    
    # Use last N rows for faster execution
    MAX_BARS = 5000
    if len(df) > MAX_BARS:
        logger.info(f"Sampling last {MAX_BARS} bars for validation...")
        df = df.tail(MAX_BARS).reset_index(drop=True)
    
    logger.info(f"Using {len(df)} bars for validation")
    
    # Initialize orchestrator with optimized settings for demo
    config = OrchestratorConfig(
        output_dir=OUTPUT_DIR,
        n_folds=5,              # Reduced for faster execution
        purge_bars=5,
        embargo_bars=5,
        train_ratio=0.7,
        n_permutations=1000,    # Reduced for faster execution
        n_monte_carlo=5000,     # Reduced for faster execution
        n_bootstrap=1000,       # Reduced for faster execution
        bootstrap_block_size=5,
        n_regimes=4,
        regime_method="kmeans",
        kill_switch_threshold=0.20,
        significance_level=0.05,
        random_seed=42,
        compute_features=False,  # Skip feature engineering for speed
    )
    
    logger.info("Initializing ValidationOrchestrator...")
    orchestrator = ValidationOrchestrator(config=config, output_dir=OUTPUT_DIR)
    
    # Run validation pipeline
    logger.info("Starting validation pipeline...")
    logger.info(f"Strategy: {STRATEGY_NAME}")
    logger.info(f"Parameters to test: {len(PARAM_GRID)} grids")
    
    try:
        result = orchestrator.run(
            strategy_name=STRATEGY_NAME,
            df=df,
            backtest_fn=run_qml_strategy,
            param_grid=PARAM_GRID,
            optimization_metric="sharpe_ratio",
        )
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("AUTOPSY RESULTS")
        logger.info("=" * 70)
        
        # Display key results
        logger.info(f"Experiment ID: {result.experiment_id}")
        logger.info(f"Overall Verdict: {result.overall_verdict}")
        logger.info(f"Confidence Score: {result.confidence_score}/100")
        logger.info(f"")
        logger.info(f"OOS Sharpe: {result.oos_sharpe:.3f}")
        logger.info(f"OOS Max DD: {result.oos_max_dd:.2f}%")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Sharpe p-value: {result.sharpe_p_value:.4f}")
        
        # Generate dossier
        logger.info("")
        logger.info("Generating Deployment Dossier...")
        dossier_gen = DossierGenerator()
        dossier_path = dossier_gen.generate(
            result,
            output_dir=OUTPUT_DIR,
            experiment_id=result.experiment_id
        )
        logger.info(f"Dossier saved to: {dossier_path}")
        
        # Run gatekeeper check
        logger.info("")
        logger.info("Running Deployment Gatekeeper...")
        gatekeeper = DeploymentGatekeeper()
        readiness = gatekeeper.check_readiness(result)
        
        logger.info(f"Readiness: {readiness}")
        logger.info(f"Passed: {readiness.passed_checks}")
        logger.info(f"Failed: {readiness.failed_checks}")
        
        # Print text report
        print("\n" + "=" * 70)
        print(orchestrator.generate_text_report(result))
        print("=" * 70)
        
        print(f"\n✅ Autopsy complete! Open the dossier at:")
        print(f"   {dossier_path}")
        print(f"\nOr run the Jupyter notebook:")
        print(f"   jupyter lab validation_report.ipynb")
        
        return result
        
    except Exception as e:
        logger.error(f"Autopsy failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
    
    # Exit code based on result
    if result is None:
        sys.exit(1)
    elif result.overall_verdict == "REJECT":
        sys.exit(2)
    else:
        sys.exit(0)
