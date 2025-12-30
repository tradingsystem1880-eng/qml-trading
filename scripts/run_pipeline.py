#!/usr/bin/env python3
"""
QML System End-to-End Pipeline
===============================
Complete pipeline: Data Sync → Detection → Features → ML Scoring → Alerts

Usage:
    python scripts/run_pipeline.py --mode scan
    python scripts/run_pipeline.py --mode backtest --start 2023-01-01
    python scripts/run_pipeline.py --mode train
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import settings
from src.utils.logging import setup_logging


def run_scan_mode(symbols: list, timeframes: list, min_score: float):
    """Run pattern scanning mode."""
    from src.data.enhanced_fetcher import create_enhanced_fetcher
    from src.detection.detector import QMLDetector
    from src.features.engineer import FeatureEngineer
    from src.features.regime import RegimeClassifier
    from src.alerts.telegram import TelegramAlerts
    
    logger.info("=== QML Pattern Scanner ===")
    
    fetcher = create_enhanced_fetcher()
    detector = QMLDetector()
    feature_eng = FeatureEngineer()
    regime_clf = RegimeClassifier()
    alerts = TelegramAlerts()
    
    all_patterns = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Scanning {symbol} {timeframe}...")
            
            try:
                # Fetch data
                df = fetcher.fetch_enhanced_data(symbol, timeframe, limit=500)
                
                if df.empty or len(df) < 100:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}")
                    continue
                
                # Get regime
                regime = regime_clf.classify(df)
                logger.info(f"  Regime: {regime.combined.value} (strength: {regime.confidence:.2f})")
                
                # Detect patterns
                patterns = detector.detect(symbol, timeframe, df=df)
                
                for pattern in patterns:
                    if pattern.validity_score >= min_score:
                        all_patterns.append(pattern)
                        
                        logger.info(
                            f"  Found {pattern.pattern_type.value} pattern: "
                            f"validity={pattern.validity_score:.2f}, "
                            f"entry={pattern.trading_levels.entry:.4f}"
                        )
                        
                        # Send alert
                        if alerts.enabled:
                            alerts.send_pattern_alert_sync(pattern)
            
            except Exception as e:
                logger.error(f"Error scanning {symbol} {timeframe}: {e}")
    
    # Summary
    logger.info(f"\n=== Scan Complete ===")
    logger.info(f"Total patterns found: {len(all_patterns)}")
    
    bullish = sum(1 for p in all_patterns if p.pattern_type.value == "bullish")
    bearish = sum(1 for p in all_patterns if p.pattern_type.value == "bearish")
    logger.info(f"Bullish: {bullish}, Bearish: {bearish}")
    
    return all_patterns


def run_backtest_mode(start_date: str, end_date: str, symbols: list):
    """Run backtesting mode."""
    from src.data.fetcher import create_data_fetcher
    from src.detection.detector import QMLDetector
    from src.backtest.engine import BacktestEngine, BacktestConfig
    
    logger.info("=== QML Backtest ===")
    
    fetcher = create_data_fetcher()
    detector = QMLDetector()
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    
    # Collect patterns and price data
    all_patterns = []
    price_data = {}
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        for timeframe in settings.detection.timeframes:
            # Get data from database (already synced)
            df = fetcher.get_data(symbol, timeframe, auto_sync=False)
            
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                continue
            
            # Store per symbol (use most granular timeframe for trade simulation)
            if symbol not in price_data or len(df) > len(price_data[symbol]):
                price_data[symbol] = df
            
            # Detect patterns
            patterns = detector.detect(symbol, timeframe, df=df)
            logger.info(f"  {symbol} {timeframe}: {len(patterns)} patterns")
            all_patterns.extend(patterns)
    
    logger.info(f"Found {len(all_patterns)} patterns to backtest")
    
    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        risk_per_trade_pct=1.0,
        commission_pct=0.1,
        slippage_pct=0.05
    )
    
    engine = BacktestEngine(config)
    result = engine.run(all_patterns, price_data, start, end)
    
    # Print report
    print(engine.generate_report(result))
    
    return result


def run_train_mode(symbols: list):
    """Run ML training mode."""
    from src.data.fetcher import create_data_fetcher
    from src.detection.detector import QMLDetector
    from src.features.engineer import FeatureEngineer
    from src.ml.labeler import TripleBarrierLabeler
    from src.ml.trainer import ModelTrainer, TrainerConfig
    
    logger.info("=== QML Model Training ===")
    
    fetcher = create_data_fetcher()
    detector = QMLDetector()
    feature_eng = FeatureEngineer()
    labeler = TripleBarrierLabeler()
    
    # Collect patterns across all symbols
    all_patterns = []
    all_price_data = {}
    
    for symbol in symbols:
        logger.info(f"Collecting patterns for {symbol}...")
        
        for timeframe in settings.detection.timeframes:
            df = fetcher.fetch_ohlcv(symbol, timeframe, limit=2000)
            
            if df.empty:
                continue
            
            all_price_data[symbol] = df
            patterns = detector.detect(symbol, timeframe, df=df)
            all_patterns.extend(patterns)
    
    logger.info(f"Total patterns collected: {len(all_patterns)}")
    
    if len(all_patterns) < 50:
        logger.error("Insufficient patterns for training. Need at least 50.")
        return None
    
    # Calculate features
    features_list = feature_eng.calculate_batch_features(all_patterns, all_price_data)
    features_df = feature_eng.features_to_dataframe(features_list)
    
    # Label patterns
    labels = []
    returns = []
    
    for pattern in all_patterns:
        df = all_price_data.get(pattern.symbol)
        if df is None:
            continue
        
        result = labeler.label_pattern(pattern, df)
        if result:
            labels.append(result.label.value)
            returns.append(result.return_pct)
    
    import numpy as np
    labels = np.array(labels)
    returns = np.array(returns)
    
    # Filter to binary classification (win/loss only)
    valid_mask = labels >= 0
    features_df = features_df[valid_mask]
    labels = labels[valid_mask]
    returns = returns[valid_mask]
    
    logger.info(f"Training samples: {len(labels)} (wins: {sum(labels)}, losses: {len(labels) - sum(labels)})")
    
    # Train model
    trainer = ModelTrainer(TrainerConfig(
        n_splits=3,
        enable_hyperparam_opt=False  # Set True for production
    ))
    
    feature_cols = feature_eng.get_feature_names()
    X = features_df[feature_cols].fillna(0)
    
    result = trainer.train(X, labels, returns)
    
    # Save model
    if result.best_model:
        path = result.best_model.save()
        logger.info(f"Model saved to {path}")
    
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Mean AUC: {result.mean_auc:.3f} ± {result.std_auc:.3f}")
    logger.info(f"Mean Win Rate: {result.mean_win_rate:.2%}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="QML Trading System Pipeline")
    parser.add_argument("--mode", choices=["scan", "backtest", "train"], default="scan")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--timeframes", type=str, default="1h,4h,1d")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    
    setup_logging(log_level=args.log_level)
    
    symbols = args.symbols.split(",")
    timeframes = args.timeframes.split(",")
    
    if args.mode == "scan":
        run_scan_mode(symbols, timeframes, args.min_score)
    elif args.mode == "backtest":
        run_backtest_mode(args.start, args.end, symbols)
    elif args.mode == "train":
        run_train_mode(symbols)


if __name__ == "__main__":
    main()

