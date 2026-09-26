import pandas as pd
import numpy as np

print("Loading data for error analysis...")
s1_in = pd.read_parquet('cache/normalized/train_source1.d6b62546f6.India.parquet', columns=['entity_id', 'name_norm', 'addr_norm', 'state'])
s1_us = pd.read_parquet('cache/normalized/train_source1.d6b62546f6.US.parquet', columns=['entity_id', 'name_norm', 'addr_norm', 'state'])
s1_all = pd.concat([s1_in, s1_us]).set_index('entity_id')

pool_in2 = pd.read_parquet('cache/normalized/train_source2.d6b62546f6.India.parquet', columns=['entity_id', 'name_norm', 'addr_norm', 'state'])
pool_in3 = pd.read_parquet('cache/normalized/train_source3.d6b62546f6.India.parquet', columns=['entity_id', 'name_norm', 'addr_norm', 'state'])
pool_us2 = pd.read_parquet('cache/normalized/train_source2.d6b62546f6.US.parquet', columns=['entity_id', 'name_norm', 'addr_norm', 'state'])
pool_us3 = pd.read_parquet('cache/normalized/train_source3.d6b62546f6.US.parquet', columns=['entity_id', 'name_norm', 'addr_norm', 'state'])
pool_all = pd.concat([pool_in2, pool_in3, pool_us2, pool_us3]).set_index('entity_id')

pairs = pd.read_parquet('cache/holdout_report_pairs.parquet')

# False Positives
fps = pairs[(pairs['selected'] == 1) & (pairs['label'] == 0)].head(15)
print("\n=== SAMPLE FALSE POSITIVES (Model predicted Match, Truth was Non-Match) ===")
for _, r in fps.iterrows():
    s1_id, p_id, prob = r['source1_entity_id'], r['entity_id'], r['prob']
    if s1_id in s1_all.index and p_id in pool_all.index:
        s1_row = s1_all.loc[s1_id]
        p_row = pool_all.loc[p_id]
        print(f"P={prob:.3f} | S1: {s1_row['name_norm']} | {s1_row['addr_norm']} | {s1_row['state']}")
        print(f"         | Pool: {p_row['name_norm']} | {p_row['addr_norm']} | {p_row['state']}")
        print("---")

# False Negatives inside candidates
fns = pairs[(pairs['selected'] == 0) & (pairs['label'] == 1)].head(15)
print("\n=== SAMPLE FALSE NEGATIVES (True Match, but Model/Gate rejected) ===")
for _, r in fns.iterrows():
    s1_id, p_id, prob = r['source1_entity_id'], r['entity_id'], r['prob']
    if s1_id in s1_all.index and p_id in pool_all.index:
        s1_row = s1_all.loc[s1_id]
        p_row = pool_all.loc[p_id]
        print(f"P={prob:.3f} | S1: {s1_row['name_norm']} | {s1_row['addr_norm']} | {s1_row['state']}")
        print(f"         | Pool: {p_row['name_norm']} | {p_row['addr_norm']} | {p_row['state']}")
        print("---")
