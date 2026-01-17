#!/usr/bin/env python3
"""
Phase 1: QML Trading System Structure Setup
============================================
This script creates the professional directory structure and safely migrates files.

Actions:
1. Create full directory skeleton
2. Safe migration of root Python files to _incoming_refactor/
3. Move PDFs and large docs to archive/
4. Generate SYSTEM_CONTEXT.md
5. Create config/default.yaml with placeholders
6. Create experiments.db in results/
"""

import os
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path("/Users/hunternovotny/Desktop/QML_SYSTEM")

# =============================================================================
# 1. DIRECTORY SKELETON
# =============================================================================

DIRECTORIES = [
    # Core source structure
    "src/core",
    "src/detection",
    "src/detection/legacy",      # For old detection versions
    "src/backtest",
    "src/validation",
    "src/reporting",
    "src/reporting/templates",
    "src/data",
    "src/deployment",
    "src/pipeline",
    "src/strategies",
    "src/features",
    "src/analysis",
    "src/trading",
    "src/utils",
    "src/dashboard",
    
    # CLI entry points
    "cli",
    
    # Data storage
    "data/raw",
    "data/raw/BTC",
    "data/processed",
    "data/processed/BTC",
    "data/samples",
    
    # Results & outputs
    "results",
    "results/charts",
    "results/reports",
    
    # Testing
    "tests/unit",
    "tests/integration",
    
    # Notebooks
    "notebooks",
    
    # Configuration
    "config",
    "config/strategies",
    
    # Documentation
    "docs",
    "docs/archive",
    
    # Archive for legacy code
    "archive",
    "archive/algo_trading_py",
    "archive/legacy_docs",
    "archive/legacy_detection",
    
    # Temporary refactor staging
    "_incoming_refactor",
]

# =============================================================================
# 2. FILES TO MIGRATE
# =============================================================================

# Root Python files -> _incoming_refactor/
ROOT_PYTHON_TO_MIGRATE = [
    "run_autopsy.py",
    "run_4y_validation.py",
    "validate_previous_backtest.py",
    "run_full_validation_on_saved_trades.py",
    "generate_professional_report.py",
    "strategy_autopsy.py",
    "qml_colab_visualizer.py",
    "debug_features.py",
    "forensic_audit.py",
]

# PDFs and large docs -> archive/legacy_docs/
DOCS_TO_ARCHIVE = [
    "QML-setup-with-results-3.pdf",
    "cursor_qml_trading_system_strategic_aud.md",
]

# Root images -> results/charts/
IMAGES_TO_MOVE = [
    "drawdown_analysis.png",
    "drawdowns.png",
    "equity_curve.png",
    "monte_carlo.png",
    "monte_carlo_cones.png",
    "permutation.png",
    "permutation_test.png",
]

# Old detection logic -> src/detection/legacy/
LEGACY_DETECTION_SOURCES = [
    ("qml_strategy_vrd/detection_logic/v1.0.0_flawed_singlepass", "v1_0_0_singlepass"),
    ("qml_strategy_vrd/detection_logic/v1.1.0_rolling_window", "v1_1_0_rolling"),
    ("qml_strategy_vrd/detection_logic/v2.0.0_atr_directional_change", "v2_0_0_atr"),
]

# =============================================================================
# 3. SYSTEM_CONTEXT.MD CONTENT
# =============================================================================

SYSTEM_CONTEXT_CONTENT = f'''# QML Trading System - System Context

> **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **Purpose**: This file serves as the "memory" for AI agents working on this codebase.

---

## 🎯 Project Overview

**QML (Quasimodo-Like) Pattern Trading System** is an algorithmic trading research platform focused on detecting and trading specific chart patterns on BTC/USDT.

### Core Objectives (VRD 2.0)
1. **Validate Reality of Edge** — Statistical proof that patterns predict price movement
2. **Understand Why It Works** — Feature analysis of winning vs losing trades
3. **Assess Durability** — Walk-forward and regime analysis
4. **Identify Failure Modes** — Drawdown and stress testing
5. **Define Deployment Rules** — Risk limits and position sizing

---

## 📁 Directory Structure

```
QML_SYSTEM/
├── src/                    # Core library (importable modules)
│   ├── core/               # Fundamental abstractions (models, config, exceptions)
│   ├── detection/          # Pattern detection algorithms
│   │   └── legacy/         # Archived detection versions (v1.0.0, v1.1.0, v2.0.0)
│   ├── backtest/           # Backtesting engine
│   ├── validation/         # Statistical validation (permutation, monte carlo, bootstrap)
│   ├── reporting/          # HTML dossier and chart generation
│   │   └── templates/      # Jinja2 HTML templates
│   ├── data/               # Data fetching and loading
│   ├── deployment/         # Production utilities (gatekeeper, paper trader)
│   ├── pipeline/           # Orchestration logic
│   ├── strategies/         # Strategy adapters
│   └── dashboard/          # Web dashboard
│
├── cli/                    # Command-line entry points
│   ├── run_backtest.py     # Primary backtest command
│   ├── run_validation.py   # VRD validation suite
│   └── run_detection.py    # Pattern detection command
│
├── data/                   # Data storage
│   ├── raw/                # Raw API downloads
│   ├── processed/          # Clean parquet files (BTC 1h, 4h)
│   └── samples/            # Sample data for tests
│
├── results/                # Output artifacts
│   ├── experiments.db      # SQLite database for dashboard
│   ├── charts/             # Generated visualizations
│   └── reports/            # HTML dossiers
│
├── config/                 # Configuration
│   ├── default.yaml        # Default parameters
│   └── strategies/         # Strategy-specific configs
│
├── tests/                  # Test suite
│   ├── unit/
│   └── integration/
│
├── notebooks/              # Research notebooks
├── docs/                   # Documentation
├── archive/                # Legacy code (reference only)
└── _incoming_refactor/     # Temporary staging for refactoring
```

---

## 🔧 Key Components

### Detection Module (`src/detection/`)
- **Primary Algorithm**: ATR Directional Change (v2.0.0)
- **Pattern Type**: QML Bullish (5-point pattern: P1→P2→P3→P4→P5)
- **Entry Signal**: P5 confirmation with ATR-based SL/TP

### Validation Module (`src/validation/`)
- **Permutation Test**: Shuffle returns to test edge significance
- **Monte Carlo**: Simulate equity paths for risk analysis
- **Bootstrap**: Confidence intervals on performance metrics
- **Walk-Forward**: Out-of-sample validation

### Reporting Module (`src/reporting/`)
- **Dossier**: HTML report generator (Strategy Autopsy Report)
- **Visuals**: Equity curves, drawdown charts, MC cones

---

## 📊 Data Contract

### OHLCV Parquet Schema
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime64 | UTC timestamp |
| open | float64 | Open price |
| high | float64 | High price |
| low | float64 | Low price |
| close | float64 | Close price |
| volume | float64 | Volume |

### Trade Record Schema
| Column | Type | Description |
|--------|------|-------------|
| entry_time | datetime64 | Entry timestamp |
| exit_time | datetime64 | Exit timestamp |
| entry_price | float64 | Entry price |
| exit_price | float64 | Exit price |
| side | str | 'LONG' or 'SHORT' |
| pnl_pct | float64 | PnL percentage |
| result | str | 'WIN', 'LOSS', 'BREAKEVEN' |

---

## 🚀 Quick Start Commands

```bash
# Run pattern detection
python -m cli.run_detection --symbol BTCUSDT --timeframe 4h

# Run backtest with validation
python -m cli.run_backtest --config config/strategies/qml_bullish.yaml

# Run full VRD validation
python -m cli.run_validation --trades results/trades.csv

# Start dashboard
python -m src.dashboard.app
```

---

## ⚠️ Important Notes

1. **Legacy Detection Code**: Old versions are preserved in `src/detection/legacy/` for reference
2. **Data Location**: Primary BTC data is in `data/processed/BTC/`
3. **Results Database**: `results/experiments.db` tracks all experiment runs
4. **Configuration**: All tunable parameters should be in YAML configs, not hardcoded

---

## 🔗 Related Files

- [Implementation Plan](/.gemini/antigravity/brain/421669f0-8aae-4a70-923e-c78c09b085fe/implementation_plan.md)
- [config/default.yaml](config/default.yaml)
- [README.md](README.md)
'''

# =============================================================================
# 4. DEFAULT.YAML CONTENT
# =============================================================================

DEFAULT_YAML_CONTENT = '''# QML Trading System - Default Configuration
# ==========================================
# This file contains all tunable parameters for the trading system.
# Override these values in strategy-specific configs under config/strategies/

# Detection Parameters
detection:
  method: atr_directional_change  # Options: atr_directional_change, rolling_window
  atr_period: 14                  # ATR calculation period
  swing_window: 5                 # Swing point detection window
  min_pattern_bars: 20            # Minimum bars for pattern formation
  max_pattern_bars: 200           # Maximum bars for pattern formation
  
  # QML Pattern Specific
  qml:
    min_depth_ratio: 0.5          # P3 must retrace at least 50% of P1-P2
    max_depth_ratio: 1.0          # P3 cannot exceed P1
    confirmation_atr_mult: 0.5    # ATR multiplier for P5 confirmation

# Backtest Parameters
backtest:
  initial_capital: 10000.0        # Starting capital in USD
  position_size: 0.1              # Position size as fraction of capital
  commission: 0.001               # Commission per trade (0.1%)
  slippage: 0.0005                # Slippage estimate (0.05%)
  
  # Risk Management
  risk:
    stop_loss_atr_mult: 1.5       # Stop loss in ATR multiples
    take_profit_atr_mult: 3.0     # Take profit in ATR multiples
    max_positions: 1              # Maximum concurrent positions
    max_daily_loss: 0.02          # Maximum daily loss (2%)

# Validation Parameters (VRD 2.0)
validation:
  # Permutation Test
  permutation:
    n_iterations: 1000            # Number of permutations
    significance_threshold: 0.05  # P-value threshold
    
  # Monte Carlo Simulation
  monte_carlo:
    n_simulations: 1000           # Number of MC paths
    confidence_levels: [0.05, 0.25, 0.50, 0.75, 0.95]
    
  # Bootstrap Analysis
  bootstrap:
    n_samples: 10000              # Number of bootstrap samples
    block_size: 5                 # Block size for stationary bootstrap
    confidence_level: 0.95        # Confidence interval level
    
  # Walk-Forward
  walk_forward:
    n_folds: 5                    # Number of WF folds
    is_oos_ratio: 0.7             # In-sample to out-of-sample ratio

# Deployment Parameters
deployment:
  # Gatekeeper Thresholds
  gatekeeper:
    min_sharpe: 1.0               # Minimum Sharpe ratio to deploy
    max_drawdown: 0.15            # Maximum acceptable drawdown (15%)
    min_trades: 30                # Minimum trades for significance
    min_win_rate: 0.4             # Minimum win rate (40%)
    p_value_threshold: 0.05       # Statistical significance threshold
    
  # Paper Trading
  paper_trading:
    check_interval_seconds: 300   # Check every 5 minutes
    log_trades: true              # Log all trades
    
  # Risk Limits
  risk_limits:
    max_position_usd: 1000        # Maximum position size in USD
    max_daily_trades: 5           # Maximum trades per day
    kill_switch_drawdown: 0.10    # Emergency stop if DD exceeds 10%

# Data Parameters
data:
  default_symbol: BTCUSDT
  default_timeframe: 4h
  lookback_days: 1460             # 4 years of data
  source: binance                 # Data source

# Reporting Parameters
reporting:
  output_format: html             # Options: html, pdf, json
  include_charts: true
  chart_style: dark               # Options: dark, light
  dpi: 150                        # Chart resolution
'''

# =============================================================================
# 5. EXPERIMENTS.DB SCHEMA
# =============================================================================

EXPERIMENTS_DB_SCHEMA = '''
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT UNIQUE NOT NULL,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    
    -- Performance Metrics
    total_return REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    calmar_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    profit_factor REAL,
    total_trades INTEGER,
    
    -- Validation Results
    permutation_pvalue REAL,
    monte_carlo_median REAL,
    bootstrap_ci_lower REAL,
    bootstrap_ci_upper REAL,
    walk_forward_score REAL,
    
    -- Deployment Status
    is_approved BOOLEAN DEFAULT 0,
    gatekeeper_passed BOOLEAN DEFAULT 0,
    
    -- Metadata
    config_path TEXT,
    results_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    entry_price REAL NOT NULL,
    exit_price REAL,
    side TEXT NOT NULL,
    quantity REAL,
    pnl_pct REAL,
    pnl_usd REAL,
    result TEXT,
    pattern_type TEXT,
    
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    validation_type TEXT NOT NULL,
    run_date TEXT NOT NULL,
    result_value REAL,
    passed BOOLEAN,
    details TEXT,
    
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_strategy ON experiments(strategy_name);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_trades_experiment ON trades(experiment_id);
CREATE INDEX IF NOT EXISTS idx_validation_experiment ON validation_runs(experiment_id);
'''

# =============================================================================
# EXECUTION
# =============================================================================

def create_directories():
    """Create the full directory skeleton."""
    print("\n" + "=" * 60)
    print("📁 CREATING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    created = 0
    for dir_path in DIRECTORIES:
        full_path = ROOT / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {dir_path}/")
            created += 1
        else:
            print(f"  ⏭️  Exists:  {dir_path}/")
    
    print(f"\n📊 Created {created} new directories")
    return created


def safe_move(src: str, dst_dir: str, description: str = ""):
    """Safely move a file, handling errors gracefully."""
    src_path = ROOT / src
    dst_path = ROOT / dst_dir / Path(src).name
    
    if not src_path.exists():
        print(f"  ⚠️  Not found: {src}")
        return False
    
    if dst_path.exists():
        print(f"  ⏭️  Already moved: {src}")
        return False
    
    try:
        shutil.move(str(src_path), str(dst_path))
        print(f"  ✅ Moved: {src} → {dst_dir}/")
        return True
    except Exception as e:
        print(f"  ❌ Error moving {src}: {e}")
        return False


def migrate_root_files():
    """Move root Python files to _incoming_refactor/."""
    print("\n" + "=" * 60)
    print("📦 MIGRATING ROOT PYTHON FILES")
    print("=" * 60)
    
    moved = 0
    for filename in ROOT_PYTHON_TO_MIGRATE:
        if safe_move(filename, "_incoming_refactor"):
            moved += 1
    
    print(f"\n📊 Moved {moved} Python files to _incoming_refactor/")
    return moved


def archive_docs():
    """Move PDFs and large docs to archive/."""
    print("\n" + "=" * 60)
    print("📚 ARCHIVING LARGE DOCUMENTS")
    print("=" * 60)
    
    moved = 0
    for filename in DOCS_TO_ARCHIVE:
        if safe_move(filename, "archive/legacy_docs"):
            moved += 1
    
    print(f"\n📊 Archived {moved} documents")
    return moved


def move_images():
    """Move root images to results/charts/."""
    print("\n" + "=" * 60)
    print("🖼️  MOVING ROOT IMAGES")
    print("=" * 60)
    
    moved = 0
    for filename in IMAGES_TO_MOVE:
        if safe_move(filename, "results/charts"):
            moved += 1
    
    print(f"\n📊 Moved {moved} images to results/charts/")
    return moved


def copy_legacy_detection():
    """Copy legacy detection logic to src/detection/legacy/."""
    print("\n" + "=" * 60)
    print("🔧 COPYING LEGACY DETECTION LOGIC")
    print("=" * 60)
    
    copied = 0
    for src_dir, target_name in LEGACY_DETECTION_SOURCES:
        src_path = ROOT / src_dir
        dst_path = ROOT / "src/detection/legacy" / target_name
        
        if not src_path.exists():
            print(f"  ⚠️  Not found: {src_dir}")
            continue
            
        if dst_path.exists():
            print(f"  ⏭️  Already exists: src/detection/legacy/{target_name}/")
            continue
        
        try:
            shutil.copytree(str(src_path), str(dst_path))
            print(f"  ✅ Copied: {src_dir} → src/detection/legacy/{target_name}/")
            copied += 1
        except Exception as e:
            print(f"  ❌ Error copying {src_dir}: {e}")
    
    print(f"\n📊 Copied {copied} legacy detection versions")
    return copied


def move_algo_trading():
    """Move ALGO TRADING PY folder to archive/."""
    print("\n" + "=" * 60)
    print("📦 ARCHIVING ALGO TRADING PY")
    print("=" * 60)
    
    src_path = ROOT / "ALGO TRADING PY"
    dst_path = ROOT / "archive/algo_trading_py"
    
    if not src_path.exists():
        print("  ⚠️  ALGO TRADING PY folder not found")
        return 0
    
    if dst_path.exists() and any(dst_path.iterdir()):
        print("  ⏭️  Already archived")
        return 0
    
    try:
        # Copy contents instead of moving (safer)
        for item in src_path.iterdir():
            if item.is_file():
                shutil.copy2(str(item), str(dst_path / item.name))
                print(f"  ✅ Copied: {item.name}")
        return 1
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0


def create_system_context():
    """Create SYSTEM_CONTEXT.md in root."""
    print("\n" + "=" * 60)
    print("📝 CREATING SYSTEM_CONTEXT.MD")
    print("=" * 60)
    
    context_path = ROOT / "SYSTEM_CONTEXT.md"
    
    with open(context_path, 'w') as f:
        f.write(SYSTEM_CONTEXT_CONTENT)
    
    print(f"  ✅ Created: SYSTEM_CONTEXT.md ({len(SYSTEM_CONTEXT_CONTENT)} bytes)")
    return True


def create_default_config():
    """Create config/default.yaml."""
    print("\n" + "=" * 60)
    print("⚙️  CREATING DEFAULT CONFIGURATION")
    print("=" * 60)
    
    config_path = ROOT / "config/default.yaml"
    
    with open(config_path, 'w') as f:
        f.write(DEFAULT_YAML_CONTENT)
    
    print(f"  ✅ Created: config/default.yaml ({len(DEFAULT_YAML_CONTENT)} bytes)")
    return True


def create_experiments_db():
    """Create SQLite experiments.db in results/."""
    print("\n" + "=" * 60)
    print("🗄️  CREATING EXPERIMENTS DATABASE")
    print("=" * 60)
    
    db_path = ROOT / "results/experiments.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.executescript(EXPERIMENTS_DB_SCHEMA)
    conn.commit()
    conn.close()
    
    print(f"  ✅ Created: results/experiments.db")
    print("  📋 Tables: experiments, trades, validation_runs")
    return True


def create_init_files():
    """Create __init__.py files in all src directories."""
    print("\n" + "=" * 60)
    print("🐍 CREATING __init__.py FILES")
    print("=" * 60)
    
    src_dirs = [
        "src", "src/core", "src/detection", "src/detection/legacy",
        "src/backtest", "src/validation", "src/reporting", "src/data",
        "src/deployment", "src/pipeline", "src/strategies", "src/features",
        "src/analysis", "src/trading", "src/utils", "src/dashboard", "cli"
    ]
    
    created = 0
    for dir_path in src_dirs:
        init_path = ROOT / dir_path / "__init__.py"
        if not init_path.exists():
            init_path.touch()
            print(f"  ✅ Created: {dir_path}/__init__.py")
            created += 1
    
    print(f"\n📊 Created {created} __init__.py files")
    return created


def main():
    """Execute all setup steps."""
    print("\n" + "=" * 80)
    print("🚀 QML TRADING SYSTEM - PHASE 1 SETUP")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Execute all steps
    dirs_created = create_directories()
    create_init_files()
    migrate_root_files()
    archive_docs()
    move_images()
    copy_legacy_detection()
    move_algo_trading()
    create_system_context()
    create_default_config()
    create_experiments_db()
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ PHASE 1 SETUP COMPLETE")
    print("=" * 80)
    print("""
    📁 Directory skeleton created
    📦 Root files moved to _incoming_refactor/
    📚 Large docs archived
    🖼️  Images moved to results/charts/
    🔧 Legacy detection copied to src/detection/legacy/
    📝 SYSTEM_CONTEXT.md created
    ⚙️  config/default.yaml created
    🗄️  results/experiments.db created
    
    Next Steps:
    1. Review _incoming_refactor/ files
    2. Split large files into focused modules
    3. Update imports across codebase
    """)


if __name__ == "__main__":
    main()
