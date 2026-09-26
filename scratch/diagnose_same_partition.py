"""Phase 1 Follow-up: Granular Decomposition of Same-Partition Misses.

Splits Same-Partition Misses into:
  1. faiss_top_5_margin_cut: S1 was in FAISS top-5 (production kk=5) for name or addr, but discarded by rev_margin/rank.
  2. faiss_top_6_to_15: S1 was in FAISS top 6-15 (recoverable by widening kk/rev_k without changing embeddings).
  3. faiss_beyond_15: S1 was outside top 15 in both name and addr (true embedding failure -> requires Phase 3).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.config as config
from src.dataloader import load_ground_truth
from src.filters.blocking import _partition, PartitionedIndex
from src.pipeline import (
    RETRIEVAL_COLUMNS,
    candidates,
    countries,
    get_split,
    load_split,
    _country_files,
)
from src.retrieval.embedder import CharNgramEmbedder, addr_text, name_text


def analyze_country_misses(country: str, max_check: int = 1500):
    print("=" * 75, flush=True)
    print(f"[{country}] SAME-PARTITION MISS DECOMPOSITION (FAISS Rank & Margin Audit)", flush=True)
    print("=" * 75, flush=True)

    split_info = get_split()
    holdout_set = set(split_info["holdout"])

    print("Loading normalized data...", flush=True)
    s1, pool = load_split("train", country, RETRIEVAL_COLUMNS)
    cands = candidates("train", country)

    s1_map = s1["entity_id"].to_numpy()
    pool_map = pool["entity_id"].to_numpy()

    s1_in_holdout = s1["entity_id"].isin(holdout_set).to_numpy()
    cands_s1_idx = cands["s1_idx"].to_numpy()
    cands_pool_idx = cands["pool_idx"].to_numpy()

    cands_holdout_mask = s1_in_holdout[cands_s1_idx]
    cand_pairs = set(zip(s1_map[cands_s1_idx[cands_holdout_mask]], pool_map[cands_pool_idx[cands_holdout_mask]]))

    print("Loading ground truth...", flush=True)
    gt_df = pd.read_csv(config.TRAIN_GROUND_TRUTH_PATH, sep="\t", dtype="string[pyarrow]", keep_default_na=False)
    gt_links = gt_df[gt_df["matched_entity_ids"] != ""].assign(m=lambda d: d["matched_entity_ids"].str.split(",")).explode("m")
    gt_links = gt_links[gt_links["source1_entity_id"].isin(holdout_set)]
    all_holdout_pairs = set(zip(gt_links["source1_entity_id"].to_numpy(), gt_links["m"].to_numpy()))

    s1_country_ids = set(s1_map)
    country_true_pairs = {p for p in all_holdout_pairs if p[0] in s1_country_ids}
    missed_links = list(country_true_pairs - cand_pairs)

    print(f"Total True Links: {len(country_true_pairs):,} | Captured: {len(country_true_pairs & cand_pairs):,} | Missed: {len(missed_links):,}", flush=True)

    miss_df = pd.DataFrame(missed_links, columns=["s1_id", "p_id"])
    miss_df = miss_df.merge(s1.rename(columns=lambda c: f"s1_{c}"), left_on="s1_id", right_on="s1_entity_id", how="inner")
    miss_df = miss_df.merge(pool.rename(columns=lambda c: f"p_{c}"), left_on="p_id", right_on="p_entity_id", how="inner")

    s1_part = _partition(miss_df["s1_state"].fillna(""))
    p_part = _partition(miss_df["p_state"].fillna(""))

    # Filter strictly to same partition
    same_part_mask = (miss_df["s1_state"] != "") & (miss_df["p_state"] != "") & (s1_part == p_part)
    same_part_misses = miss_df[same_part_mask].reset_index(drop=True)
    print(f"Same-partition misses: {len(same_part_misses):,} out of {len(miss_df):,} total misses ({len(same_part_misses)/len(miss_df):.1%})", flush=True)

    if len(same_part_misses) > max_check:
        print(f"Auditing sample of {max_check:,} same-partition misses...", flush=True)
        sample_misses = same_part_misses.sample(n=max_check, random_state=42).reset_index(drop=True)
    else:
        sample_misses = same_part_misses

    # Fit embedders exactly like candidates()
    print("Fitting reproducible embedders...", flush=True)
    s1_file, pool_files = _country_files("train", country)
    rng = np.random.default_rng(config.RANDOM_SEED)
    fit = pd.concat([pd.read_parquet(f, columns=RETRIEVAL_COLUMNS) for f in pool_files], ignore_index=True)
    fit = fit.iloc[np.sort(rng.choice(len(fit), min(len(fit), 200_000), replace=False))].reset_index(drop=True)

    name_emb = CharNgramEmbedder(config.EMBED_DIM, seed=config.RANDOM_SEED).fit(name_text(fit))
    addr_emb = CharNgramEmbedder(config.EMBED_DIM, seed=config.RANDOM_SEED).fit(addr_text(fit))
    del fit

    print("Building partitioned indexes for S1...", flush=True)
    t0 = time.time()
    s1_all_parts = _partition(s1["state"])
    s1_name_emb = name_emb.transform(name_text(s1))
    s1_addr_emb = addr_emb.transform(addr_text(s1))

    name_index = PartitionedIndex(s1_name_emb, s1_all_parts)
    addr_index = PartitionedIndex(s1_addr_emb, s1_all_parts)
    print(f"Indexes built in {time.time() - t0:.1f}s", flush=True)

    # Transform missed pool records
    pool_queries = pool.loc[sample_misses["p_id"].map(pd.Series(pool.index, index=pool["entity_id"])).to_numpy()].reset_index(drop=True)
    query_parts = _partition(pool_queries["state"])
    has_addr = (pool_queries["addr_norm"] != "").to_numpy()

    c_name = name_emb.transform(name_text(pool_queries))
    c_addr = addr_emb.transform(addr_text(pool_queries))

    K_CHECK = 30
    print(f"Searching top-{K_CHECK} in FAISS for missed pool queries...", flush=True)
    dn, in_ = name_index.search(c_name, query_parts, K_CHECK)
    da, ia_ = addr_index.search(c_addr, query_parts, K_CHECK)

    # Check rank of target S1
    s1_pos_map = pd.Series(s1.index, index=s1["entity_id"])
    target_s1_indices = sample_misses["s1_id"].map(s1_pos_map).to_numpy()

    # Bucketing
    b_top_5 = 0          # in top-5 (production kk=5) for name or address
    b_top_6_to_10 = 0    # in top 6..10
    b_top_11_to_20 = 0   # in top 11..20
    b_top_21_to_30 = 0   # in top 21..30
    b_beyond_30 = 0      # not in top 30 in either view

    name_ranks = []
    addr_ranks = []

    for idx, target_idx in enumerate(target_s1_indices):
        n_list = list(in_[idx])
        a_list = list(ia_[idx]) if has_addr[idx] else []

        r_name = n_list.index(target_idx) + 1 if target_idx in n_list else 999
        r_addr = a_list.index(target_idx) + 1 if target_idx in a_list else 999

        best_rank = min(r_name, r_addr)
        name_ranks.append(r_name)
        addr_ranks.append(r_addr)

        if best_rank <= 5:
            b_top_5 += 1
        elif best_rank <= 10:
            b_top_6_to_10 += 1
        elif best_rank <= 20:
            b_top_11_to_20 += 1
        elif best_rank <= 30:
            b_top_21_to_30 += 1
        else:
            b_beyond_30 += 1

    n_sample = len(sample_misses)
    print("\n" + "=" * 75, flush=True)
    print(f"[{country}] SAME-PARTITION MISS AUDIT RESULTS (N = {n_sample:,})", flush=True)
    print("=" * 75, flush=True)
    print(f"  1. In FAISS top-5 (production kk=5)       : {b_top_5:>5} ({b_top_5/n_sample:>6.1%}) [DISCARDED BY REV_MARGIN / RANK]")
    print(f"  2. In FAISS rank 6 - 10                   : {b_top_6_to_10:>5} ({b_top_6_to_10/n_sample:>6.1%}) [RECOVERABLE BY REV_K=8]")
    print(f"  3. In FAISS rank 11 - 20                  : {b_top_11_to_20:>5} ({b_top_11_to_20/n_sample:>6.1%}) [RECOVERABLE BY REV_K=15]")
    print(f"  4. In FAISS rank 21 - 30                  : {b_top_21_to_30:>5} ({b_top_21_to_30/n_sample:>6.1%}) [RECOVERABLE BY REV_K=25]")
    print(f"  5. Outside FAISS top-30                   : {b_beyond_30:>5} ({b_beyond_30/n_sample:>6.1%}) [TRUE EMBEDDING FAILURES]")
    print("=" * 75, flush=True)

    recoverable_by_retrieval_params = b_top_5 + b_top_6_to_10 + b_top_11_to_20
    print(f"  >>> CHEAP RETRIEVAL FIX HEADROOM (rank <= 20) : {recoverable_by_retrieval_params} / {n_sample} ({recoverable_by_retrieval_params/n_sample:.1%})", flush=True)
    print(f"  >>> EXPENSIVE EMBEDDING RE-TRAINING REQUIRED   : {b_beyond_30} / {n_sample} ({b_beyond_30/n_sample:.1%})", flush=True)
    print("=" * 75 + "\n", flush=True)


if __name__ == "__main__":
    analyze_country_misses("India", max_check=1000)
