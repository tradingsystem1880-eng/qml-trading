.PHONY: help test lint format clean install dashboard backtest

help:
\t@echo "QML Trading System - Available Commands:"
\t@echo ""
\t@echo "  make install      - Install dependencies via pip"
\t@echo "  make test        - Run pytest test suite"
\t@echo "  make lint        - Run ruff linter"
\t@echo "  make format      - Format code with ruff and black"
\t@echo "  make backtest    - Run default backtest on BTC 4h"
\t@echo "  make dashboard   - Start Streamlit dashboard"
\t@echo "  make clean       - Remove Python cache files"
\t@echo ""

install:
\tpip install -r requirements.txt

test:
\tpytest tests/ -v

test-fast:
\tpytest tests/ -x --ignore=tests/integration

lint:
\t@echo "Running ruff linter..."
\t@ruff check src/ tests/ cli/ || true

format:
\t@echo "Formatting with ruff..."
\t@ruff format src/ tests/ cli/ || true
\t@echo "Formatting with black..."
\t@black src/ tests/ cli/ --line-length 100 || true

backtest:
\t@echo "Running default backtest (BTC/USDT 4h)..."
\tpython -m cli.run_backtest --symbol BTCUSDT --timeframe 4h

dashboard:
\t@echo "Starting Streamlit dashboard on http://localhost:8501"
\tstreamlit run src/dashboard/app.py

clean:
\t@echo "Cleaning Python cache files..."
\tfind . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
\tfind . -type f -name "*.pyc" -delete
\tfind . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
\tfind . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
