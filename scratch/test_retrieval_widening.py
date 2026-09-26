import sys
sys.path.insert(0, 'code/business_entity_resolution')
import time
import pandas as pd
import numpy as np
import src.config as config
from src.retrieval import CharNgramEmbedder, name_text, addr_text
from src.retrieval.index import flat_index
from src.filters.blocking import _partition, PartitionedIndex

print("Loading data for retrieval benchmark...", flush=True)
gt = pd.read_csv(config.TRAIN_GROUND_TRUTH_PATH, sep='\t', dtype='string[pyarrow]', keep_default_na=False)
links = gt[gt['matched_entity_ids'] != ''].assign(m=lambda d: d['matched_entity_ids'].str.split(',')).explode('m')
owner = pd.Series(links['source1_entity_id'].to_numpy(), index=links['m'].astype('string[pyarrow]').to_numpy())

cols = ['entity_id', 'country', 'name_core', 'name_alias', 'addr_norm', 'state']
s1 = pd.read_parquet('cache/normalized/train_source1.d6b62546f6.India.parquet', columns=cols)
s1 = s1.reset_index(drop=True)

# Load first 50k of pool India
p2 = pd.read_parquet('cache/normalized/train_source2.d6b62546f6.India.parquet', columns=cols).iloc[:25000]
p3 = pd.read_parquet('cache/normalized/train_source3.d6b62546f6.India.parquet', columns=cols).iloc[:25000]
pool_chunk = pd.concat([p2, p3], ignore_index=True)

# Ground truth links for this chunk
pool_ids = pool_chunk['entity_id'].to_numpy()
true_s1 = owner.reindex(pool_ids).fillna('').to_numpy()
has_gt = true_s1 != ''
print(f"Pool chunk: {len(pool_chunk):,} records, {has_gt.sum():,} have true S1 links in GT", flush=True)

# Fit embedders on sample
print("Fitting embedders...", flush=True)
fit_sample = pool_chunk.sample(n=min(len(pool_chunk), 30000), random_state=42)
name_emb = CharNgramEmbedder(config.EMBED_DIM, seed=config.RANDOM_SEED).fit(name_text(fit_sample))
addr_emb = CharNgramEmbedder(config.EMBED_DIM, seed=config.RANDOM_SEED).fit(addr_text(fit_sample))

print("Indexing S1...", flush=True)
t0 = time.time()
s1_parts = _partition(s1['state'])
s1_name = name_emb.transform(name_text(s1))
name_idx = PartitionedIndex(s1_name, s1_parts)
s1_addr = addr_emb.transform(addr_text(s1))
addr_idx = PartitionedIndex(s1_addr, s1_parts)
s1_has_addr = (s1['addr_norm'] != '').to_numpy()
s1_name_f32 = s1_name.astype(np.float32)
s1_addr_f32 = s1_addr.astype(np.float32)
print(f"Indexing done in {time.time()-t0:.1f}s", flush=True)

print("Transforming pool chunk...", flush=True)
t0 = time.time()
c_name = name_emb.transform(name_text(pool_chunk))
c_addr = addr_emb.transform(addr_text(pool_chunk))
c_parts = _partition(pool_chunk['state'])
has_addr = (pool_chunk['addr_norm'] != '').to_numpy()
print(f"Transform done in {time.time()-t0:.1f}s", flush=True)

# Test different k values for name search
w = config.NAME_WEIGHT
s1_ids = s1['entity_id'].to_numpy()

for test_k, test_kk, test_margin in [(3, 5, 0.05), (5, 8, 0.08), (6, 10, 0.10), (8, 12, 0.12), (10, 15, 0.15)]:
    t0 = time.time()
    dn, i_n = name_idx.search(c_name, c_parts, test_kk)
    a_rows = np.flatnonzero(has_addr)
    da, i_a = addr_idx.search(c_addr[a_rows], c_parts[a_rows], test_kk)

    pairs = pd.DataFrame({
        'row': np.concatenate([np.repeat(np.arange(len(pool_chunk)), test_kk), np.repeat(a_rows, test_kk)]),
        's1': np.concatenate([i_n.ravel(), i_a.ravel()]),
        'top_view': np.concatenate([np.tile(np.arange(test_kk) == 0, len(pool_chunk)), np.tile(np.arange(test_kk) == 0, len(a_rows))]),
    })
    pairs = pairs[pairs['s1'] >= 0].groupby(['row', 's1'], as_index=False)['top_view'].max()
    r, s = pairs['row'].to_numpy(), pairs['s1'].to_numpy()

    cos_n = np.einsum('ij,ij->i', s1_name_f32[s], c_name[r])
    cos_a = np.einsum('ij,ij->i', s1_addr_f32[s], c_addr[r])
    both = has_addr[r] & s1_has_addr[s]
    cos = np.where(both, w * cos_n + (1 - w) * cos_a, cos_n).astype(np.float32)
    pairs['cos'] = cos
    pairs['rank'] = pairs.groupby('row')['cos'].rank(ascending=False, method='first') - 1
    best = pairs.groupby('row')['cos'].transform('max').to_numpy()
    rank = pairs['rank'].to_numpy()
    no_addr = ~has_addr[r]

    keep = (pairs['top_view'].to_numpy()
            | (rank == 0)
            | ((rank < test_k) & (best - cos <= test_margin))
            | (no_addr & (rank < test_kk) & (best - cos <= 2 * test_margin)))

    kept_r = r[keep]
    kept_s = s[keep]
    kept_pairs = pd.DataFrame({'row': kept_r, 's1': kept_s})

    # Evaluate recall on GT linked pool records
    # Check if true_s1[row] == s1_ids[s1]
    cand_s1_ids = s1_ids[kept_s]
    target_s1_ids = true_s1[kept_r]
    is_match = (cand_s1_ids == target_s1_ids) & (target_s1_ids != '')
    matched_pool_rows = set(kept_r[is_match])

    recall = len(matched_pool_rows) / max(1, has_gt.sum())
    n_pairs = len(kept_pairs)
    pairs_per_pool = n_pairs / len(pool_chunk)

    print(f"k={test_k}, kk={test_kk}, margin={test_margin:.2f} | Recall: {recall*100:.2f}% ({len(matched_pool_rows)}/{has_gt.sum()}) | Pairs: {n_pairs:,} ({pairs_per_pool:.2f}/record) | Search: {time.time()-t0:.2f}s", flush=True)
