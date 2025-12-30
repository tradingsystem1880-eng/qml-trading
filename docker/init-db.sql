-- QML Trading System Database Initialization
-- TimescaleDB schema for crypto OHLCV data and pattern storage

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- OHLCV Data Tables (Time-Series Hypertables)
-- ============================================================================

-- Raw OHLCV data from exchanges
CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL DEFAULT 'binance',
    timeframe VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    quote_volume DOUBLE PRECISION,
    trades INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, symbol, exchange, timeframe)
);

-- Convert to TimescaleDB hypertable for optimized time-series queries
SELECT create_hypertable('ohlcv', 'time', 
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe ON ohlcv (timeframe, time DESC);

-- ============================================================================
-- Swing Points Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS swing_points (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    swing_type VARCHAR(10) NOT NULL CHECK (swing_type IN ('high', 'low')),
    price DOUBLE PRECISION NOT NULL,
    significance DOUBLE PRECISION NOT NULL,  -- ATR-normalized significance
    atr_at_point DOUBLE PRECISION NOT NULL,
    confirmed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);

SELECT create_hypertable('swing_points', 'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_swing_symbol_time ON swing_points (symbol, time DESC);

-- ============================================================================
-- Market Structure Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_structure (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    structure_type VARCHAR(10) NOT NULL CHECK (structure_type IN ('HH', 'HL', 'LH', 'LL')),
    price DOUBLE PRECISION NOT NULL,
    previous_price DOUBLE PRECISION,
    trend VARCHAR(15) CHECK (trend IN ('uptrend', 'downtrend', 'consolidation')),
    trend_strength DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);

SELECT create_hypertable('market_structure', 'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- CHoCH (Change of Character) Events
-- ============================================================================

CREATE TABLE IF NOT EXISTS choch_events (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    choch_type VARCHAR(15) NOT NULL CHECK (choch_type IN ('bullish', 'bearish')),
    break_level DOUBLE PRECISION NOT NULL,
    break_strength DOUBLE PRECISION NOT NULL,  -- ATR-normalized
    volume_confirmation BOOLEAN DEFAULT FALSE,
    confirmed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);

SELECT create_hypertable('choch_events', 'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- Break of Structure (BoS) Events
-- ============================================================================

CREATE TABLE IF NOT EXISTS bos_events (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    bos_type VARCHAR(15) NOT NULL CHECK (bos_type IN ('bullish', 'bearish')),
    break_level DOUBLE PRECISION NOT NULL,
    volume_spike BOOLEAN DEFAULT FALSE,
    choch_id INTEGER,  -- Reference to triggering CHoCH
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);

SELECT create_hypertable('bos_events', 'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- ============================================================================
-- QML Patterns (Main Pattern Table)
-- ============================================================================

CREATE TABLE IF NOT EXISTS qml_patterns (
    id SERIAL,
    detection_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(15) NOT NULL CHECK (pattern_type IN ('bullish', 'bearish')),
    
    -- Pattern Components (prices)
    left_shoulder_price DOUBLE PRECISION NOT NULL,
    left_shoulder_time TIMESTAMPTZ NOT NULL,
    head_price DOUBLE PRECISION NOT NULL,
    head_time TIMESTAMPTZ NOT NULL,
    right_shoulder_price DOUBLE PRECISION,
    right_shoulder_time TIMESTAMPTZ,
    neckline_start DOUBLE PRECISION,
    neckline_end DOUBLE PRECISION,
    
    -- Trading Levels
    entry_price DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    take_profit_1 DOUBLE PRECISION,  -- 1:1 R:R
    take_profit_2 DOUBLE PRECISION,  -- 2:1 R:R
    take_profit_3 DOUBLE PRECISION,  -- 3:1 R:R
    
    -- Quality Metrics
    validity_score DOUBLE PRECISION NOT NULL CHECK (validity_score >= 0 AND validity_score <= 1),
    geometric_score DOUBLE PRECISION,
    volume_score DOUBLE PRECISION,
    context_score DOUBLE PRECISION,
    
    -- ML Prediction (filled after ML model runs)
    ml_confidence DOUBLE PRECISION,
    ml_model_version VARCHAR(50),
    
    -- Pattern State
    status VARCHAR(20) DEFAULT 'forming' CHECK (status IN ('forming', 'active', 'triggered', 'invalidated', 'completed')),
    invalidation_reason VARCHAR(100),
    
    -- Outcome Tracking
    outcome VARCHAR(20) CHECK (outcome IN ('win', 'loss', 'breakeven', 'pending', 'cancelled')),
    actual_return_pct DOUBLE PRECISION,
    bars_to_outcome INTEGER,
    
    -- Metadata
    choch_id INTEGER,
    bos_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (id, detection_time)
);

SELECT create_hypertable('qml_patterns', 'detection_time',
    chunk_time_interval => INTERVAL '90 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON qml_patterns (symbol, detection_time DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_status ON qml_patterns (status, detection_time DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_outcome ON qml_patterns (outcome) WHERE outcome IS NOT NULL;

-- ============================================================================
-- Features Table (for ML)
-- ============================================================================

CREATE TABLE IF NOT EXISTS pattern_features (
    pattern_id INTEGER NOT NULL,
    pattern_time TIMESTAMPTZ NOT NULL,
    
    -- Geometric Features
    head_depth_ratio DOUBLE PRECISION,
    shoulder_symmetry DOUBLE PRECISION,
    neckline_slope DOUBLE PRECISION,
    pattern_duration_bars INTEGER,
    
    -- Volume Features
    volume_at_head DOUBLE PRECISION,
    volume_ratio_head_shoulders DOUBLE PRECISION,
    obv_divergence DOUBLE PRECISION,
    
    -- Context Features
    atr_percentile DOUBLE PRECISION,
    distance_from_daily_high DOUBLE PRECISION,
    distance_from_daily_low DOUBLE PRECISION,
    btc_correlation DOUBLE PRECISION,
    
    -- Regime Features
    market_regime VARCHAR(20),
    trend_strength DOUBLE PRECISION,
    volatility_regime VARCHAR(20),
    
    -- Crypto-specific Features
    funding_rate DOUBLE PRECISION,
    open_interest_change DOUBLE PRECISION,
    liquidation_levels DOUBLE PRECISION,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (pattern_id, pattern_time),
    FOREIGN KEY (pattern_id, pattern_time) REFERENCES qml_patterns(id, detection_time)
);

-- ============================================================================
-- Model Registry
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL,  -- 'xgboost', 'lightgbm', etc.
    
    -- Performance Metrics
    train_start_date DATE,
    train_end_date DATE,
    test_start_date DATE,
    test_end_date DATE,
    
    -- Metrics
    accuracy DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    auc_roc DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    
    -- Model Artifacts
    model_path VARCHAR(500),
    feature_importance JSONB,
    hyperparameters JSONB,
    
    -- Status
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deprecated_at TIMESTAMPTZ
);

-- ============================================================================
-- Alerts & Notifications Log
-- ============================================================================

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    pattern_id INTEGER,
    alert_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10),
    message TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    
    -- Delivery Status
    telegram_sent BOOLEAN DEFAULT FALSE,
    telegram_sent_at TIMESTAMPTZ,
    webhook_sent BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Continuous Aggregates for Performance
-- ============================================================================

-- Daily OHLCV summary (auto-updated)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_ohlcv_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    exchange,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    count(*) AS candle_count
FROM ohlcv
WHERE timeframe = '1h'
GROUP BY time_bucket('1 day', time), symbol, exchange
WITH NO DATA;

-- Pattern detection statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS pattern_stats_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', detection_time) AS day,
    symbol,
    timeframe,
    pattern_type,
    count(*) AS pattern_count,
    avg(validity_score) AS avg_validity,
    avg(ml_confidence) AS avg_confidence
FROM qml_patterns
GROUP BY time_bucket('1 day', detection_time), symbol, timeframe, pattern_type
WITH NO DATA;

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to calculate ATR for a given symbol/timeframe
CREATE OR REPLACE FUNCTION calculate_atr(
    p_symbol VARCHAR,
    p_timeframe VARCHAR,
    p_period INTEGER DEFAULT 14,
    p_end_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_atr DOUBLE PRECISION;
BEGIN
    WITH tr_calc AS (
        SELECT
            GREATEST(
                high - low,
                ABS(high - LAG(close) OVER (ORDER BY time)),
                ABS(low - LAG(close) OVER (ORDER BY time))
            ) AS true_range
        FROM ohlcv
        WHERE symbol = p_symbol
          AND timeframe = p_timeframe
          AND time <= p_end_time
        ORDER BY time DESC
        LIMIT p_period + 1
    )
    SELECT AVG(true_range) INTO v_atr
    FROM tr_calc
    WHERE true_range IS NOT NULL;
    
    RETURN v_atr;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO qml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO qml_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO qml_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'QML Trading System database initialized successfully!';
END $$;

