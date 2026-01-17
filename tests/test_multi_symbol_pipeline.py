#!/usr/bin/env python3
"""
Multi-Symbol Pipeline Test
===========================
Quick verification that the multi-symbol data pipeline works correctly.

Tests:
1. normalize_symbol() - Symbol normalization
2. get_symbol_data_dir() - Deterministic path generation (no fallback)
3. build_master_store() - Correct path construction for new symbols

Usage:
    python tests/test_multi_symbol_pipeline.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_engine import (
    normalize_symbol,
    get_symbol_data_dir,
    build_master_store,
    DATA_DIR_BASE,
)


def test_normalize_symbol():
    """Test symbol normalization to filesystem-safe format."""
    print("=" * 60)
    print("TEST 1: normalize_symbol()")
    print("=" * 60)
    
    test_cases = [
        ("BTC/USDT", "BTCUSDT"),
        ("ETH/USDT", "ETHUSDT"),
        ("ETH-USD", "ETHUSD"),
        ("SOL/USDT", "SOLUSDT"),
        ("btc/usdt", "BTCUSDT"),  # lowercase
        ("BTCUSDT", "BTCUSDT"),   # already normalized
    ]
    
    all_passed = True
    for input_symbol, expected in test_cases:
        result = normalize_symbol(input_symbol)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status} normalize_symbol(\"{input_symbol}\") -> \"{result}\" (expected: \"{expected}\")")
    
    print()
    assert all_passed


def test_get_symbol_data_dir_deterministic():
    """Test that get_symbol_data_dir returns ONLY the normalized path (no fallback)."""
    print("=" * 60)
    print("TEST 2: get_symbol_data_dir() - Deterministic (No Fallback)")
    print("=" * 60)
    
    test_cases = [
        ("BTC/USDT", "BTCUSDT"),
        ("ETH/USDT", "ETHUSDT"),
        ("SOL/USDT", "SOLUSDT"),
    ]
    
    all_passed = True
    for symbol, expected_folder in test_cases:
        result = get_symbol_data_dir(symbol)
        expected = DATA_DIR_BASE / expected_folder
        
        # Check it returns the normalized path, NOT a legacy path
        is_correct = result == expected
        is_not_legacy = "BTC" not in str(result) or "BTCUSDT" in str(result)
        
        status = "✅" if (is_correct and is_not_legacy) else "❌"
        if not (is_correct and is_not_legacy):
            all_passed = False
        
        print(f"  {status} get_symbol_data_dir(\"{symbol}\")")
        print(f"     -> {result}")
        print(f"     Expected: {expected}")
        
        # Verify no fallback to legacy 'BTC' folder
        if symbol == "BTC/USDT":
            if "BTC" in str(result) and "BTCUSDT" not in str(result):
                print(f"     ❌ FALLBACK DETECTED! Should be BTCUSDT, not BTC")
                all_passed = False
    
    print()
    assert all_passed


def test_build_master_store_path_construction():
    """Test that build_master_store constructs correct paths for new symbols."""
    print("=" * 60)
    print("TEST 3: build_master_store() - Path Construction (Mocked)")
    print("=" * 60)
    
    # Mock fetch_ohlcv to avoid actual API calls
    with patch('src.data_engine.fetch_ohlcv') as mock_fetch:
        # Set up mock to return sample DataFrame
        import pandas as pd
        mock_df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'Open': [100.0] * 100,
            'High': [101.0] * 100,
            'Low': [99.0] * 100,
            'Close': [100.5] * 100,
            'Volume': [1000.0] * 100,
        })
        mock_fetch.return_value = mock_df
        
        # Test with ETH/USDT
        symbol = "ETH/USDT"
        expected_dir = DATA_DIR_BASE / "ETHUSDT"
        
        print(f"  Testing: build_master_store(symbol=\"{symbol}\", timeframes=[\"1h\"])")
        print(f"  Expected output directory: {expected_dir}")
        print()
        
        try:
            # Run with dry_run but mock parquet writing
            with patch('pandas.DataFrame.to_parquet') as mock_parquet:
                result = build_master_store(
                    symbol=symbol,
                    timeframes=["1h"],
                    years=1,
                    dry_run=True
                )
            
            # Check result contains correct symbol
            result_symbol = result.get('symbol', '')
            symbol_correct = result_symbol == symbol
            status1 = "✅" if symbol_correct else "❌"
            print(f"  {status1} Result symbol: {result_symbol}")
            
            # Check file path in results
            file_path = result.get('files', {}).get('1h', '')
            path_contains_ethusdt = 'ETHUSDT' in file_path
            status2 = "✅" if path_contains_ethusdt else "❌"
            print(f"  {status2} Output path contains ETHUSDT: {file_path}")
            
            # Verify NOT using legacy BTC path
            not_legacy = 'BTC' not in file_path or 'BTCUSDT' in file_path
            status3 = "✅" if not_legacy else "❌"
            print(f"  {status3} Not using legacy BTC path")
            
            print()
            assert symbol_correct and path_contains_ethusdt and not_legacy
            
        except Exception as e:
            print(f"  ❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    print("\n" + "=" * 60)
    print("  MULTI-SYMBOL PIPELINE VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("normalize_symbol()", test_normalize_symbol()))
    results.append(("get_symbol_data_dir()", test_get_symbol_data_dir_deterministic()))
    results.append(("build_master_store()", test_build_master_store_path_construction()))
    
    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
