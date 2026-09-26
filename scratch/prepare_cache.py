"""
One-Time Normalization Cache Generator for Business Entity Resolution.
Run directly in your terminal:
    python scratch/prepare_cache.py
"""
import sys
import time
import pickle
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = PROJECT_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE_ROOT))

from src import config
from src.dataloader.loader import load_source, load_ground_truth
from src.normalization.normalizer import normalize_records

CACHE_DIR = PROJECT_ROOT / "scratch"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "norm_cache.pkl"

def main():
    print("=" * 65, flush=True)
    print("  Amazon ML Challenge 2026: Fast Normalization Cache Generator", flush=True)
    print("=" * 65, flush=True)
    t_start = time.time()

    # Step 1: Ingestion
    print("\n[Step 1/5] Loading raw source files and ground truth...", flush=True)
    t0 = time.time()
    df_s1_raw = load_source(config.TRAIN_SOURCE1_PATH, nrows=1000)
    print(f"  -> Source 1 loaded: {len(df_s1_raw):,} records (sample for fast benchmarking)")
    df_s2_raw = load_source(config.TRAIN_SOURCE2_PATH)
    print(f"  -> Source 2 loaded: {len(df_s2_raw):,} records")
    df_s3_raw = load_source(config.TRAIN_SOURCE3_PATH)
    print(f"  -> Source 3 loaded: {len(df_s3_raw):,} records")
    gt = load_ground_truth(config.TRAIN_GROUND_TRUTH_PATH)
    print(f"  -> Ground truth loaded: {len(gt):,} S1 mappings ({time.time() - t0:.2f}s)", flush=True)

    # Step 2: Normalize S1
    print("\n[Step 2/5] Normalizing Source 1 (1,000 records)...", flush=True)
    t0 = time.time()
    df_s1 = normalize_records(df_s1_raw)
    print(f"  -> Source 1 normalized in {time.time() - t0:.2f}s", flush=True)

    # Step 3: Normalize S2
    print(f"\n[Step 3/5] Normalizing Source 2 ({len(df_s2_raw):,} records)...", flush=True)
    t0 = time.time()
    df_s2 = normalize_records(df_s2_raw)
    print(f"  -> Source 2 normalized in {time.time() - t0:.2f}s", flush=True)

    # Step 4: Normalize S3
    print(f"\n[Step 4/5] Normalizing Source 3 ({len(df_s3_raw):,} records)...", flush=True)
    t0 = time.time()
    df_s3 = normalize_records(df_s3_raw)
    print(f"  -> Source 3 normalized in {time.time() - t0:.2f}s", flush=True)

    # Step 5: Merge Pool and Dump Cache
    print("\n[Step 5/5] Combining Pool (S2 + S3) and serializing to disk...", flush=True)
    t0 = time.time()
    df_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    print(f"  -> Total pool size: {len(df_pool):,} records")

    eval_s1_ids = list(df_s1["entity_id"])
    eval_gt = {k: gt.get(k, set()) for k in eval_s1_ids}

    cache_data = {
        "df_s1": df_s1,
        "df_pool": df_pool,
        "eval_gt": eval_gt,
        "eval_s1_ids": eval_s1_ids,
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = CACHE_FILE.stat().st_size / (1024 * 1024)
    print(f"  -> Cache successfully saved: {CACHE_FILE}")
    print(f"  -> File size: {size_mb:.1f} MB (saved in {time.time() - t0:.2f}s)", flush=True)
    print("\n" + "=" * 65)
    print(f"  ALL DONE in {time.time() - t_start:.2f}s! You can now run:")
    print("      python scratch/bench_fast.py")
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
