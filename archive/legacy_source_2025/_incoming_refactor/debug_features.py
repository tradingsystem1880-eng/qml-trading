
import pandas as pd
import time
from src.features.library import FeatureLibrary

def test_features():
    print("Loading data...")
    df = pd.read_parquet("data/processed/BTC/4h_master.parquet")
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    
    # Take 100 rows first for smoke test
    df_small = df.head(100).copy()
    
    lib = FeatureLibrary()
    
    print("Running features on 100 rows...")
    t0 = time.time()
    f1 = lib.compute_features_for_range(df_small)
    print(f"100 rows took {time.time()-t0:.4f}s")
    
    # Take 1000 rows
    df_med = df.head(1000).copy()
    print("Running features on 1000 rows...")
    t0 = time.time()
    f2 = lib.compute_features_for_range(df_med)
    print(f"1000 rows took {time.time()-t0:.4f}s")

if __name__ == "__main__":
    test_features()
