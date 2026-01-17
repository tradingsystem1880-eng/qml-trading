"""
QML Trading System - Main Entry Point
======================================
Orchestrates the complete QML pattern detection and alerting pipeline.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.utils.logging import setup_logging
from src.data.fetcher import DataFetcher
from src.data.database import get_database


def run_data_sync(args):
    """Run data synchronization."""
    logger.info("Starting data synchronization...")
    
    fetcher = DataFetcher()
    
    # Determine start date
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    
    # Get symbols and timeframes
    symbols = args.symbols.split(",") if args.symbols else settings.detection.symbols
    timeframes = args.timeframes.split(",") if args.timeframes else settings.detection.timeframes
    
    # Sync data
    results = fetcher.sync_all_symbols(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        force_full=args.force,
        show_progress=True
    )
    
    # Print summary
    total = sum(v for v in results.values() if v > 0)
    failed = sum(1 for v in results.values() if v < 0)
    
    logger.info(f"Sync complete: {total} candles stored, {failed} failures")


def run_detection(args):
    """Run pattern detection."""
    logger.info("Starting pattern detection...")
    
    from src.detection.detector import QMLDetector
    
    detector = QMLDetector()
    
    # Get symbols and timeframes
    symbols = args.symbols.split(",") if args.symbols else settings.detection.symbols
    timeframes = args.timeframes.split(",") if args.timeframes else settings.detection.timeframes
    
    # Run detection
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Scanning {symbol} {timeframe}...")
            patterns = detector.detect(symbol, timeframe)
            
            if patterns:
                logger.info(f"Found {len(patterns)} patterns for {symbol} {timeframe}")
                for p in patterns:
                    logger.info(
                        f"  {p.pattern_type.value} pattern: "
                        f"validity={p.validity_score:.2f}, "
                        f"entry={p.trading_levels.entry if p.trading_levels else 'N/A'}"
                    )


def run_backtest(args):
    """Run backtesting."""
    logger.info("Starting backtest...")
    
    from src.backtest.engine import BacktestEngine
    
    engine = BacktestEngine()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None
    
    # Run backtest
    results = engine.run(
        start_date=start_date,
        end_date=end_date,
        symbols=args.symbols.split(",") if args.symbols else None
    )
    
    # Print results
    logger.info(f"Backtest Results:")
    logger.info(f"  Total Trades: {results.get('total_trades', 0)}")
    logger.info(f"  Win Rate: {results.get('win_rate', 0):.2%}")
    logger.info(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")


def run_dashboard(args):
    """Run Streamlit dashboard."""
    import subprocess
    
    logger.info("Starting dashboard...")
    
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    
    subprocess.run([
        "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.address", "0.0.0.0"
    ])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="QML Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync data from exchanges
  python -m src.main sync --start-date 2023-01-01
  
  # Run pattern detection
  python -m src.main detect --symbols BTC/USDT,ETH/USDT
  
  # Run backtest
  python -m src.main backtest --start-date 2023-01-01 --end-date 2024-01-01
  
  # Start dashboard
  python -m src.main dashboard --port 8501
        """
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Synchronize market data")
    sync_parser.add_argument("--symbols", help="Comma-separated symbols")
    sync_parser.add_argument("--timeframes", help="Comma-separated timeframes")
    sync_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    sync_parser.add_argument("--force", action="store_true", help="Force full resync")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Run pattern detection")
    detect_parser.add_argument("--symbols", help="Comma-separated symbols")
    detect_parser.add_argument("--timeframes", help="Comma-separated timeframes")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--symbols", help="Comma-separated symbols")
    backtest_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Start dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Run command
    if args.command == "sync":
        run_data_sync(args)
    elif args.command == "detect":
        run_detection(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

