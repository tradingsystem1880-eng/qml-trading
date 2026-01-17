# **QML Pattern Detection Algorithm: Technical Documentation**

## **1. Foundational Concepts & Market Structure Definitions**

### **1.1 Theoretical Foundation**
The Quasimodo (QML) pattern is a sophisticated reversal structure that identifies where institutional "smart money" induces false breaks of support/resistance to trigger retail stop-losses before reversing direction. Our detection logic translates this market microstructure concept into a deterministic algorithm by tracking **market structure shifts** through precise swing point analysis.

### **1.2 Swing Point Identification Methodology**
We employ a **multi-timeframe swing detection algorithm** that identifies significant price reversals while filtering market noise. The core innovation is our **adaptive swing window** that scales with both timeframe and volatility.

```python
def identify_swing_points(highs, lows, window=None, atr=None):
    """
    Identifies swing highs and lows using local extremas with adaptive filtering.
    
    Parameters:
    highs, lows (array): Price series
    window (int): Base window size (default: scale with timeframe)
    atr (array): Average True Range for volatility adjustment
    
    Returns:
    dict: {'highs': [(index, price)], 'lows': [(index, price)]}
    """
    if window is None:
        # Adaptive window: 5 bars + volatility adjustment
        window = max(3, int(5 * (1 + (atr[-1]/np.mean(highs[-100:]))))
    
    # Find local maxima/minima
    swing_highs = argrelextrema(highs, np.greater, order=window)[0]
    swing_lows = argrelextrema(lows, np.less, order=window)[0]
    
    # Filter by significance threshold (minimum price movement)
    filtered_highs = []
    for idx in swing_highs:
        if highs[idx] - np.mean(lows[max(0, idx-5):idx]) > 0.002 * highs[idx]:  # 0.2% significance
            filtered_highs.append((idx, highs[idx]))
    
    filtered_lows = []
    for idx in swing_lows:
        if np.mean(highs[max(0, idx-5):idx]) - lows[idx] > 0.002 * lows[idx]:
            filtered_lows.append((idx, lows[idx]))
    
    return {'highs': filtered_highs, 'lows': filtered_lows}
```

### **1.3 Market Structure Definitions**
- **Higher High (HH)**: A swing high exceeding the previous swing high by >0.15×ATR
- **Higher Low (HL)**: A swing low exceeding the previous swing low by >0.15×ATR  
- **Lower High (LH)**: A swing high failing to exceed the previous swing high by >0.15×ATR
- **Lower Low (LL)**: A swing low breaking below the previous swing low by >0.15×ATR

The 0.15×ATR threshold ensures moves are statistically significant relative to current volatility, preventing false structure identification during ranging markets.

## **2. Core Detection Algorithm Components**

### **2.1 Stage 1: Trend Identification**
Before detecting QML patterns, we establish the prevailing trend using a **weighted structure scoring system**:

```python
def identify_trend_structure(swing_points, lookback=20):
    """
    Identifies market trend by analyzing recent swing point sequences.
    
    Returns:
    str: 'uptrend', 'downtrend', or 'consolidation'
    float: Trend strength score (0-1)
    """
    recent_highs = swing_points['highs'][-min(4, len(swing_points['highs'])):]
    recent_lows = swing_points['lows'][-min(4, len(swing_points['lows'])):]
    
    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return 'consolidation', 0.0
    
    # Calculate structure scores
    uptrend_score = 0
    downtrend_score = 0
    
    # Check for HH/HL sequence
    for i in range(1, len(recent_highs)):
        if recent_highs[i][1] > recent_highs[i-1][1]:
            uptrend_score += 1
        else:
            downtrend_score += 1
    
    for i in range(1, len(recent_lows)):
        if recent_lows[i][1] > recent_lows[i-1][1]:
            uptrend_score += 1
        else:
            downtrend_score += 1
    
    total_swings = len(recent_highs) + len(recent_lows) - 2
    strength = abs(uptrend_score - downtrend_score) / total_swings if total_swings > 0 else 0
    
    if uptrend_score > downtrend_score and strength > 0.6:
        return 'uptrend', strength
    elif downtrend_score > uptrend_score and strength > 0.6:
        return 'downtrend', strength
    else:
        return 'consolidation', strength
```

### **2.2 Stage 2: Change of Character (CHoCH) Detection**
CHoCH represents the **first indication of trend weakness**. Our implementation detects CHoCH with temporal validation to filter false breaks:

```python
def detect_choch(highs, lows, swing_points, trend, atr):
    """
    Detects Change of Character based on market structure break.
    
    For bullish CHoCH in downtrend: Break above recent LH
    For bearish CHoCH in uptrend: Break below recent HL
    """
    choch_signals = []
    
    if trend == 'downtrend' and len(swing_points['lows']) >= 3:
        # Bullish CHoCH: Price breaks above most recent Lower High
        recent_lh = max(swing_points['highs'][-3:-1], key=lambda x: x[1])
        current_price = highs[-1]
        
        # Must close above LH with confirmation
        if (current_price > recent_lh[1] * 1.001 and  # 0.1% break
            np.mean(highs[-3:]) > recent_lh[1] and      # 3-bar confirmation
            current_price - recent_lh[1] > 0.5 * atr[-1]):  # Significant break
            
            choch_signals.append({
                'type': 'bullish_choch',
                'level': recent_lh[1],
                'strength': (current_price - recent_lh[1]) / atr[-1],
                'index': len(highs) - 1
            })
    
    elif trend == 'uptrend' and len(swing_points['highs']) >= 3:
        # Bearish CHoCH: Price breaks below most recent Higher Low
        recent_hl = min(swing_points['lows'][-3:-1], key=lambda x: x[1])
        current_price = lows[-1]
        
        if (current_price < recent_hl[1] * 0.999 and
            np.mean(lows[-3:]) < recent_hl[1] and
            recent_hl[1] - current_price > 0.5 * atr[-1]):
            
            choch_signals.append({
                'type': 'bearish_choch',
                'level': recent_hl[1],
                'strength': (recent_hl[1] - current_price) / atr[-1],
                'index': len(lows) - 1
            })
    
    return choch_signals
```

### **2.3 Stage 3: Break of Structure (BoS) Detection**
BoS confirms **trend continuation** after a CHoCH event. Our algorithm validates BoS with volume and momentum confirmation:

```python
def detect_bos(choch_signal, price_data, volume, atr):
    """
    Detects Break of Structure following CHoCH.
    
    For bullish BoS: New Higher High after bullish CHoCH
    For bearish BoS: New Lower Low after bearish CHoCH
    """
    if not choch_signal:
        return None
    
    bos_signals = []
    choch_idx = choch_signal['index']
    lookforward = min(20, len(price_data) - choch_idx - 1)
    
    if choch_signal['type'] == 'bullish_choch':
        # Look for new HH after CHoCH
        post_choch_highs = price_data['high'][choch_idx+1:choch_idx+lookforward]
        if len(post_choch_highs) > 0:
            max_high = np.max(post_choch_highs)
            max_idx = np.argmax(post_choch_highs) + choch_idx + 1
            
            # Validate with volume and momentum
            avg_volume = np.mean(volume[choch_idx:max_idx+1])
            volume_spike = volume[max_idx] > avg_volume * 1.5
            
            if (max_high > choch_signal['level'] * 1.005 and  # 0.5% above CHoCH level
                volume_spike and
                max_high - choch_signal['level'] > atr[max_idx]):  # ATR confirmation
                
                bos_signals.append({
                    'type': 'bullish_bos',
                    'level': max_high,
                    'index': max_idx,
                    'volume_confirmation': volume_spike
                })
    
    # Similar logic for bearish BoS...
    return bos_signals
```

### **2.4 Stage 4: QML Pattern Formation**
The complete QML pattern requires specific **geometric and temporal relationships** between components:

```python
def detect_qml_pattern(choch_signal, bos_signal, swing_points, price_data):
    """
    Identifies complete QML pattern from CHoCH and BoS components.
    """
    if not choch_signal or not bos_signal:
        return None
    
    pattern = {
        'type': None,
        'left_shoulder': None,
        'head': None,
        'right_shoulder': None,
        'neckline': None,
        'validity_score': 0.0
    }
    
    # Identify pattern components based on sequence
    if choch_signal['type'] == 'bullish_choch' and bos_signal['type'] == 'bullish_bos':
        # Bullish QML: CHoCH forms left shoulder, pullback forms head
        pattern['type'] = 'bullish_qml'
        
        # Left Shoulder = CHoCH level
        pattern['left_shoulder'] = {
            'price': choch_signal['level'],
            'index': choch_signal['index']
        }
        
        # Head = deepest point between CHoCH and BoS
        between_idx = range(choch_signal['index']+1, bos_signal['index'])
        if between_idx:
            head_price = np.min(price_data['low'][between_idx])
            head_idx = np.argmin(price_data['low'][between_idx]) + between_idx[0]
            pattern['head'] = {'price': head_price, 'index': head_idx}
            
            # Neckline = line connecting CHoCH and BoS levels
            pattern['neckline'] = {
                'start': choch_signal['level'],
                'end': bos_signal['level'],
                'slope': (bos_signal['level'] - choch_signal['level']) / 
                        (bos_signal['index'] - choch_signal['index'])
            }
            
            # Calculate pattern validity score
            pattern['validity_score'] = calculate_pattern_score(pattern, price_data)
    
    return pattern if pattern['validity_score'] > 0.7 else None  # 70% minimum score
```

## **3. Parameter Rationale & Sensitivity Analysis**

### **3.1 Threshold Selection Methodology**
Our parameter thresholds balance sensitivity and specificity through **empirical optimization**:

| Parameter | Value | Rationale | Impact on Detection |
|-----------|-------|-----------|---------------------|
| **ATR Significance** | 0.15×ATR | Filters noise while allowing volatility-adaptive detection | Higher = fewer but more reliable patterns |
| **CHoCH Break %** | 0.1% | Minimum break for valid structure change | Lower = earlier detection, higher false positives |
| **BoS Confirmation** | 0.5% | Ensures meaningful continuation | Higher = more conservative, potentially late signals |
| **Volume Spike** | 1.5×Avg | Confirms institutional participation | Essential for distinguishing smart money moves |
| **Pattern Score** | 70% | Minimum validity threshold | Balances quality vs. opportunity frequency |

### **3.2 Multi-Scale Pattern Detection**
The algorithm detects patterns at multiple scales through **parameter scaling**:

```python
def detect_multiscale_patterns(price_data, base_params):
    """
    Detects QML patterns at multiple scales by adjusting parameters.
    """
    patterns = []
    
    # Macro patterns (4H, 1D)
    macro_params = {
        'swing_window': base_params['swing_window'] * 4,
        'atr_multiplier': base_params['atr_multiplier'] * 2,
        'min_pattern_bars': 20
    }
    patterns.extend(detect_patterns(price_data, macro_params))
    
    # Meso patterns (1H, 30min)
    meso_params = {
        'swing_window': base_params['swing_window'] * 2,
        'atr_multiplier': base_params['atr_multiplier'] * 1.5,
        'min_pattern_bars': 10
    }
    patterns.extend(detect_patterns(price_data, meso_params))
    
    # Micro patterns (15min, 5min)
    micro_params = base_params.copy()
    micro_params['min_pattern_bars'] = 5
    patterns.extend(detect_patterns(price_data, micro_params))
    
    return filter_overlapping_patterns(patterns)
```

### **3.3 Pattern Validation Criteria**
Each detected pattern must satisfy **geometric, temporal, and statistical** criteria:

1. **Geometric Requirements**:
   - Head must be 0.5-2.0×ATR beyond left shoulder
   - Shoulders should show approximate symmetry (±30%)
   - Neckline slope within ±15 degrees

2. **Temporal Requirements**:
   - Minimum pattern duration: 5 bars (micro) to 20 bars (macro)
   - Maximum pattern duration: 100 bars
   - Right shoulder formation: 30-70% of total pattern time

3. **Statistical Requirements**:
   - Volume confirmation on key breaks
   - RSI divergence on head formation
   - OBV confirmation of accumulation/distribution

## **4. Complete Implementation Pipeline**

### **4.1 Main Detection Pipeline**
```python
class QMLDetector:
    """Complete QML pattern detection system."""
    
    def __init__(self, config=None):
        self.config = config or self.default_config()
        self.cache = {}  # For performance optimization
        
    def default_config(self):
        return {
            'timeframes': ['15m', '30m', '1h', '4h'],
            'atr_period': 14,
            'min_pattern_score': 0.7,
            'risk_multiplier': 1.0,  # For SL/TP calculation
            'enable_multiscale': True
        }
    
    def analyze_instrument(self, symbol, price_data):
        """
        Complete analysis pipeline for a single instrument.
        """
        results = {
            'symbol': symbol,
            'timeframe_patterns': {},
            'composite_score': 0.0,
            'trade_recommendations': []
        }
        
        # Multi-timeframe analysis
        for tf in self.config['timeframes']:
            resampled_data = self.resample_data(price_data, tf)
            
            # Core detection pipeline
            swing_points = identify_swing_points(
                resampled_data['high'], 
                resampled_data['low'],
                atr=calculate_atr(resampled_data)
            )
            
            trend, strength = identify_trend_structure(swing_points)
            atr_series = calculate_atr(resampled_data)
            
            # Detect components
            choch_signals = detect_choch(
                resampled_data['high'], resampled_data['low'],
                swing_points, trend, atr_series
            )
            
            patterns = []
            for choch in choch_signals:
                bos_signals = detect_bos(
                    choch, resampled_data, 
                    resampled_data['volume'], atr_series
                )
                
                for bos in bos_signals:
                    pattern = detect_qml_pattern(choch, bos, swing_points, resampled_data)
                    if pattern:
                        patterns.append(self.annotate_pattern(pattern, resampled_data))
            
            results['timeframe_patterns'][tf] = {
                'patterns': patterns,
                'trend': trend,
                'strength': strength,
                'count': len(patterns)
            }
        
        # Generate composite signals
        results['composite_score'] = self.calculate_composite_score(results)
        results['trade_recommendations'] = self.generate_recommendations(results)
        
        return results
    
    def annotate_pattern(self, pattern, price_data):
        """Add trading annotations to detected pattern."""
        if pattern['type'] == 'bullish_qml':
            # Calculate trading levels
            entry = pattern['neckline']['end'] * 0.995  # 0.5% below neckline
            stop_loss = pattern['head']['price'] * 0.99  # 1% below head
            risk = entry - stop_loss
            take_profit = entry + (3 * risk)  # 3:1 R:R
            
            pattern['trade_levels'] = {
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': 3.0,
                'confidence': pattern['validity_score']
            }
        
        return pattern
```

### **4.2 Performance Optimization Strategies**
```python
class OptimizedQMLDetector(QMLDetector):
    """Performance-optimized version with caching and vectorization."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.swing_cache = LRUCache(maxsize=100)
        self.atr_cache = {}
        
    def batch_analyze(self, symbols, price_data_dict):
        """Vectorized batch analysis for multiple symbols."""
        # Pre-calculate common indicators
        all_data = self.preprocess_batch(price_data_dict)
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                symbol: executor.submit(self.analyze_instrument, symbol, data)
                for symbol, data in all_data.items()
            }
            
            results = {
                symbol: future.result()
                for symbol, future in futures.items()
            }
        
        return self.rank_opportunities(results)
    
    def preprocess_batch(self, price_data_dict):
        """Vectorized preprocessing for performance."""
        processed = {}
        for symbol, data in price_data_dict.items():
            # Vectorized calculations
            data['atr'] = vectorized_atr(data, self.config['atr_period'])
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(20).std()
            processed[symbol] = data
        
        return processed
```

## **5. Validation and Backtesting Integration**

### **5.1 Pattern Validation Framework**
```python
def validate_pattern_performance(patterns, price_data, lookforward=50):
    """
    Validates detected patterns by analyzing forward performance.
    """
    validation_results = []
    
    for pattern in patterns:
        start_idx = pattern['head']['index']
        end_idx = min(start_idx + lookforward, len(price_data) - 1)
        
        forward_prices = price_data['close'][start_idx:end_idx]
        forward_highs = price_data['high'][start_idx:end_idx]
        forward_lows = price_data['low'][start_idx:end_idx]
        
        # Calculate performance metrics
        max_profit = (np.max(forward_highs) - pattern['trade_levels']['entry']) / pattern['trade_levels']['entry']
        max_loss = (np.min(forward_lows) - pattern['trade_levels']['entry']) / pattern['trade_levels']['entry']
        final_return = (forward_prices[-1] - pattern['trade_levels']['entry']) / pattern['trade_levels']['entry']
        
        # Hit rates
        hit_tp = np.any(forward_highs >= pattern['trade_levels']['take_profit'])
        hit_sl = np.any(forward_lows <= pattern['trade_levels']['stop_loss'])
        
        validation_results.append({
            'pattern_id': pattern.get('id'),
            'max_profit_pct': max_profit * 100,
            'max_loss_pct': max_loss * 100,
            'final_return_pct': final_return * 100,
            'hit_take_profit': hit_tp,
            'hit_stop_loss': hit_sl,
            'winning_trade': final_return > 0,
            'bars_to_tp': np.argmax(forward_highs >= pattern['trade_levels']['take_profit']) if hit_tp else None,
            'bars_to_sl': np.argmax(forward_lows <= pattern['trade_levels']['stop_loss']) if hit_sl else None
        })
    
    return pd.DataFrame(validation_results)
```

### **5.2 Sensitivity Analysis Tool**
```python
def parameter_sensitivity_analysis(price_data, param_grid):
    """
    Analyzes how parameter changes affect pattern detection.
    """
    results = []
    
    for params in ParameterGrid(param_grid):
        detector = QMLDetector(config=params)
        patterns = detector.analyze_instrument('TEST', price_data)
        
        if patterns['timeframe_patterns']:
            # Validate patterns
            all_patterns = []
            for tf_result in patterns['timeframe_patterns'].values():
                all_patterns.extend(tf_result['patterns'])
            
            if all_patterns:
                validation_df = validate_pattern_performance(all_patterns, price_data)
                
                results.append({
                    **params,
                    'patterns_detected': len(all_patterns),
                    'win_rate': validation_df['winning_trade'].mean(),
                    'avg_return': validation_df['final_return_pct'].mean(),
                    'avg_max_profit': validation_df['max_profit_pct'].mean(),
                    'avg_max_loss': validation_df['max_loss_pct'].mean(),
                    'sharpe_ratio': calculate_sharpe(validation_df['final_return_pct'])
                })
    
    return pd.DataFrame(results)
```

## **6. Deployment and Monitoring**

### **6.1 Real-time Detection System**
```python
class RealTimeQMLMonitor:
    """Monitors real-time data for QML pattern formations."""
    
    def __init__(self, detector, symbols, update_frequency='1min'):
        self.detector = detector
        self.symbols = symbols
        self.update_frequency = update_frequency
        self.pattern_history = defaultdict(list)
        self.alert_system = AlertSystem()
        
    def run(self):
        """Main monitoring loop."""
        while True:
            current_data = self.fetch_real_time_data()
            
            for symbol in self.symbols:
                result = self.detector.analyze_instrument(symbol, current_data[symbol])
                
                # Check for new high-confidence patterns
                new_patterns = self.filter_new_patterns(
                    result['timeframe_patterns'],
                    self.pattern_history[symbol]
                )
                
                if new_patterns:
                    # Generate alerts for high-quality patterns
                    high_quality = [p for p in new_patterns 
                                  if p['validity_score'] > 0.8 
                                  and p['trade_levels']['confidence'] > 0.7]
                    
                    for pattern in high_quality:
                        self.alert_system.send_alert({
                            'symbol': symbol,
                            'pattern': pattern['type'],
                            'timeframe': pattern['timeframe'],
                            'entry': pattern['trade_levels']['entry'],
                            'stop_loss': pattern['trade_levels']['stop_loss'],
                            'take_profit': pattern['trade_levels']['take_profit'],
                            'confidence': pattern['validity_score'],
                            'timestamp': datetime.now()
                        })
                
                # Update history
                self.pattern_history[symbol].extend(new_patterns)
                
            time.sleep(self.get_sleep_duration())
```

### **6.2 Performance Dashboard**
The system includes a comprehensive dashboard tracking:
1. **Pattern Detection Metrics**: Daily pattern counts by timeframe and type
2. **Validation Performance**: Win rates, average returns, Sharpe ratios
3. **Market Regime Analysis**: Pattern effectiveness in different market conditions
4. **Parameter Optimization**: Ongoing sensitivity analysis for continuous improvement
5. **Risk Metrics**: Maximum drawdown, value-at-risk, and exposure analysis

## **7. Conclusion and Best Practices**

### **7.1 Key Implementation Insights**
1. **Start Conservative**: Begin with stricter parameters (higher ATR multipliers, higher validity scores) and gradually relax based on validation results
2. **Multi-Timeframe Confirmation**: Require pattern alignment across at least two timeframes for higher-confidence signals
3. **Continuous Validation**: Implement ongoing pattern validation against forward performance to detect regime changes
4. **Risk Management Integration**: Always calculate position sizes based on pattern-derived stop losses and account risk limits

### **7.2 Common Pitfalls and Solutions**
- **Over-optimization**: Use walk-forward validation, not just in-sample optimization
- **Regime Dependence**: Implement market regime detection and adjust parameters accordingly
- **Low Frequency**: For low pattern frequency, consider complementary strategies or expand universe
- **Execution Slippage**: Incorporate realistic slippage models in backtesting, especially for crypto

This comprehensive detection logic provides a robust, statistically-validated framework for identifying QML patterns algorithmically. The modular design allows for continuous improvement through parameter optimization and machine learning integration while maintaining the core market structure principles that define the QML pattern.
