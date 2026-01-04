#!/usr/bin/env python3
"""
4-Year QML Strategy Validation (BTC 4h)
========================================
Runs the full Institutional-Grade Validation Framework on 4 years of BTC 4h data.
Includes Purged Walk-Forward Analysis, Regime Detection, and Rigorous Statistical Testing.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


from src.pipeline.orchestrator import ValidationOrchestrator, OrchestratorConfig
from src.strategies.qml_adapter import QMLStrategyAdapter, QMLAdapterConfig
from src.reporting.visuals import create_visualizer
from src.reporting.dossier import create_dossier_generator

def main():
    logger.info("=" * 80)
    logger.info("STARTING 4-YEAR QML STRATEGY VALIDATION (BTC 4h)")
    logger.info("=" * 80)
    
    # 1. Configuration
    config = OrchestratorConfig(
        output_dir="results/qml_btc_4h_4y_validation",
        # Walk-Forward Settings
        n_folds=4,                  # 4 folds approx 1 year each
        purge_bars=24,              # Purge bars (not purge_overlap)
        
        # Statistical Rigor
        n_permutations=10000,       # 10k shuffles for p-value
        n_monte_carlo=50000,        # 50k paths for VaR
        n_bootstrap=5000,
        
        # Risk Management
        kill_switch_threshold=0.25, # 25% max DD allowed
        
        # System
        n_jobs=-1                   # Use all cores
    )
    
    # 2. Initialize Orchestrator
    orchestrator = ValidationOrchestrator(config)
    
    # 3. Load Data
    data_path = Path("data/processed/BTC/4h_master.parquet")
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
        
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Rename columns to standard format if needed
    # Rename columns to standard format
    column_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }
    df = df.rename(columns=column_map)
    logger.info(f"Loaded {len(df)} 4h bars ({df.index[0]} to {df.index[-1]})")
    
    # 4. Initialize Strategy Adapter
    # We use the QML adapter which wraps the detector and backtest engine
    adapter_config = QMLAdapterConfig(
        symbol="BTC/USDT",
        timeframe="4h",
        initial_capital=100_000
    )
    strategy = QMLStrategyAdapter(config=adapter_config)
    
    # 5. Run Validation Pipeline
    logger.info("Running validation pipeline...")
    
    # Define parameter grid for Walk-Forward Optimization
    param_grid = {
        "min_validity_score": [0.6, 0.7, 0.8],
        "risk_reward": [2.5, 3.0, 3.5],
        "stop_loss_atr": [1.0, 1.5, 2.0]
    }
    
    result = orchestrator.run(
        strategy_name="QML_BTC_4H_4Y",
        df=df,
        backtest_fn=strategy.run,
        param_grid=param_grid,
        optimization_metric="sharpe_ratio"
    )
    
    # 6. Generate Visual Charts
    logger.info("Generating visual analysis charts...")
    visualizer = create_visualizer(output_dir=f"{config.output_dir}/charts")
    visual_paths = visualizer.create_all_visuals(result, output_dir=f"{config.output_dir}/charts")
    
    # Add chart paths to result for dossier
    result.visual_charts = visual_paths
    
    # 7. Generate Interactive Dossier
    logger.info("Generating deployment dossier...")
    dossier_gen = create_dossier_generator()
    dossier_path = dossier_gen.generate(
        result, 
        output_dir=config.output_dir
    )
    
    logger.info("=" * 80)
    logger.info(f"VALIDATION COMPLETE")
    logger.info(f"Verdict: {result.overall_verdict} (Score: {result.confidence_score}/100)")
    logger.info(f"HTML Dossier: {dossier_path}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
