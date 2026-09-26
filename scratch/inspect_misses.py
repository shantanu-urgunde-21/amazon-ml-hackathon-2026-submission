import sys
sys.path.insert(0, 'code/business_entity_resolution')
import pandas as pd
import numpy as np
import src.config as config

print("Loading ground truth and candidate pairs...", flush=True)
gt = pd.read_csv(config.TRAIN_GROUND_TRUTH_PATH, sep='\t', dtype='string[pyarrow]', keep_default_na=False)
links = gt[gt['matched_entity_ids'] != ''].assign(m=lambda d: d['matched_entity_ids'].str.split(',')).explode('m').head(50000)
true_pairs = set(zip(links['source1_entity_id'].to_numpy(), links['m'].to_numpy()))

cands_in = pd.read_parquet('cache/candidates/train_India.25b5ef9494.parquet')
s1_in = pd.read_parquet('cache/normalized/train_source1.d6b62546f6.India.parquet', columns=['entity_id', 'name_norm', 'addr_norm', 'state'])
s1_map = s1_in['entity_id'].to_numpy()

pool_cols = ['entity_id', 'name_norm', 'addr_norm', 'state']
p2 = pd.read_parquet('cache/normalized/train_source2.d6b62546f6.India.parquet', columns=pool_cols)
p3 = pd.read_parquet('cache/normalized/train_source3.d6b62546f6.India.parquet', columns=pool_cols)
pool_in = pd.concat([p2, p3], ignore_index=True)
pool_map = pool_in['entity_id'].to_numpy()

s1_idx_df = s1_in.set_index('entity_id')
pool_idx_df = pool_in.set_index('entity_id')

cand_pairs = set(zip(s1_map[cands_in['s1_idx']], pool_map[cands_in['pool_idx']]))

print(f"Checking for misses among {len(true_pairs)} true links...", flush=True)
misses = []
for s1, p in true_pairs:
    if s1 in s1_idx_df.index and p in pool_idx_df.index:
        if (s1, p) not in cand_pairs:
            misses.append((s1, p))
            if len(misses) >= 15:
                break

print(f"\n=== SAMPLE BLOCKING MISSES ({len(misses)} found) ===", flush=True)
for s1_id, p_id in misses:
    s1_row = s1_idx_df.loc[s1_id]
    p_row = pool_idx_df.loc[p_id]
    print(f"S1:   {s1_row['name_norm']} | {s1_row['addr_norm']} | State: {s1_row['state']}")
    print(f"Pool: {p_row['name_norm']} | {p_row['addr_norm']} | State: {p_row['state']}")
    print("---")
