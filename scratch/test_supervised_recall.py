"""Evaluate holdout blocking recall lift from Supervised Metric Projectors + Relaxed Margin.

Directly compares holdout candidate recall:
  Baseline: Unsupervised SVD (k=3, margin=0.05)
  Phase 3:  Supervised Residual Projector (k=6, margin=0.10)
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

import src.config as config
from src.filters.blocking import _partition, PartitionedIndex
from src.pipeline import (
    CACHE,
    RETRIEVAL_COLUMNS,
    _country_files,
    get_projectors,
    get_split,
    load_split,
)
from src.retrieval.embedder import CharNgramEmbedder, addr_text, name_text


def evaluate_recall_lift(country: str = "India", sample_queries: int = 15000):
    print("=" * 75, flush=True)
    print(f"[{country}] HOLDOUT BLOCKING RECALL EVALUATION: BASELINE VS PHASE 3", flush=True)
    print("=" * 75, flush=True)

    split_info = get_split()
    holdout_set = set(split_info["holdout"])

    print("Loading normalized data...", flush=True)
    s1, pool = load_split("train", country, RETRIEVAL_COLUMNS)

    print("Loading ground truth...", flush=True)
    gt_df = pd.read_csv(config.TRAIN_GROUND_TRUTH_PATH, sep="\t", dtype="string[pyarrow]", keep_default_na=False)
    gt_links = gt_df[gt_df["matched_entity_ids"] != ""].assign(m=lambda d: d["matched_entity_ids"].str.split(",")).explode("m")
    gt_links = gt_links[gt_links["source1_entity_id"].isin(holdout_set)]

    s1_pos_map = pd.Series(s1.index, index=s1["entity_id"])
    pool_pos_map = pd.Series(pool.index, index=pool["entity_id"])

    holdout_true = gt_links[gt_links["source1_entity_id"].isin(s1_pos_map.index) & gt_links["m"].isin(pool_pos_map.index)].copy()
    holdout_true["s1_idx"] = holdout_true["source1_entity_id"].map(s1_pos_map).to_numpy()
    holdout_true["pool_idx"] = holdout_true["m"].map(pool_pos_map).to_numpy()

    if len(holdout_true) > sample_queries:
        holdout_true = holdout_true.sample(n=sample_queries, random_state=42).reset_index(drop=True)
    print(f"Evaluating on {len(holdout_true):,} holdout true links...", flush=True)

    # 1. Base Unsupervised Embedders
    print("\nFitting base unsupervised SVD embedders...", flush=True)
    s1_file, pool_files = _country_files("train", country)
    rng = np.random.default_rng(config.RANDOM_SEED)
    fit_texts = pd.concat([pd.read_parquet(f, columns=RETRIEVAL_COLUMNS) for f in pool_files], ignore_index=True)
    fit_sample = fit_texts.iloc[np.sort(rng.choice(len(fit_texts), min(len(fit_texts), 200_000), replace=False))].reset_index(drop=True)
    del fit_texts

    name_emb = CharNgramEmbedder(config.EMBED_DIM, seed=config.RANDOM_SEED).fit(name_text(fit_sample))
    addr_emb = CharNgramEmbedder(config.EMBED_DIM, seed=config.RANDOM_SEED).fit(addr_text(fit_sample))

    # Transform S1 and queries with Base SVD
    print("Transforming S1 and queries with Base SVD...", flush=True)
    s1_parts = _partition(s1["state"])
    s1_name_base = name_emb.transform(name_text(s1))
    s1_addr_base = addr_emb.transform(addr_text(s1))

    queries_pool = pool.iloc[holdout_true["pool_idx"].to_numpy()].reset_index(drop=True)
    q_parts = _partition(queries_pool["state"])
    has_addr = (queries_pool["addr_norm"] != "").to_numpy()
    s1_has_addr = (s1["addr_norm"] != "").to_numpy()

    q_name_base = name_emb.transform(name_text(queries_pool))
    q_addr_base = addr_emb.transform(addr_text(queries_pool))

    # 2. Supervised Projectors
    print("Loading supervised metric projectors...", flush=True)
    name_proj, addr_proj = get_projectors(country, config.EMBED_DIM)

    s1_name_proj = name_proj.transform(s1_name_base)
    s1_addr_proj = addr_proj.transform(s1_addr_base) if addr_proj else s1_addr_base

    q_name_proj = name_proj.transform(q_name_base)
    q_addr_proj = addr_proj.transform(q_addr_base) if addr_proj else q_addr_base

    def run_blocking_pass(s1_name, s1_addr, q_name, q_addr, k, margin, tag):
        t0 = time.time()
        kk = k + 2
        w = config.NAME_WEIGHT

        name_idx = PartitionedIndex(s1_name, s1_parts)
        addr_idx = PartitionedIndex(s1_addr, s1_parts)

        dn, in_ = name_idx.search(q_name, q_parts, kk)
        a_rows = np.flatnonzero(has_addr)
        da, ia_ = addr_idx.search(q_addr[a_rows], q_parts[a_rows], kk)

        pairs = pd.DataFrame({
            "row": np.concatenate([np.repeat(np.arange(len(queries_pool)), kk), np.repeat(a_rows, kk)]),
            "s1": np.concatenate([in_.ravel(), ia_.ravel()]),
            "top_view": np.concatenate([np.tile(np.arange(kk) == 0, len(queries_pool)), np.tile(np.arange(kk) == 0, len(a_rows))]),
        })
        pairs = pairs[pairs["s1"] >= 0].groupby(["row", "s1"], as_index=False)["top_view"].max()
        r, s = pairs["row"].to_numpy(), pairs["s1"].to_numpy()

        cos_n = np.einsum("ij,ij->i", s1_name[s].astype(np.float32), q_name[r])
        cos_a = np.einsum("ij,ij->i", s1_addr[s].astype(np.float32), q_addr[r])
        both = has_addr[r] & s1_has_addr[s]
        cos = np.where(both, w * cos_n + (1 - w) * cos_a, cos_n).astype(np.float32)

        pairs["cos"] = cos
        pairs["rank"] = pairs.groupby("row")["cos"].rank(ascending=False, method="first") - 1
        best = pairs.groupby("row")["cos"].transform("max").to_numpy()
        rank = pairs["rank"].to_numpy()
        no_addr = ~has_addr[r]

        keep = (pairs["top_view"].to_numpy()
                | (rank == 0)
                | ((rank < k) & (best - cos <= margin))
                | (no_addr & (rank < kk) & (best - cos <= 2 * margin)))

        kept_r = r[keep]
        kept_s = s[keep]

        # Check recall against holdout_true: holdout_true.loc[row, 's1_idx'] == kept_s
        target_s1_indices = holdout_true["s1_idx"].to_numpy()
        kept_set = set(zip(kept_r, kept_s))
        captured = sum(1 for row_idx, target_idx in enumerate(target_s1_indices) if (row_idx, target_idx) in kept_set)

        recall = captured / len(holdout_true)
        n_pairs = len(kept_r)
        print(f"  [{tag:<35}] Recall: {recall:.2%} ({captured:,}/{len(holdout_true):,}) | Pairs: {n_pairs:,} ({n_pairs/len(queries_pool):.2f}/record) | Elapsed: {time.time()-t0:.1f}s", flush=True)
        return recall

    print("\n" + "=" * 75)
    print("RECALL COMPARISON ON UNTOUCHED HOLDOUT")
    print("=" * 75, flush=True)

    r_base = run_blocking_pass(s1_name_base, s1_addr_base, q_name_base, q_addr_base,
                               k=3, margin=0.05, tag="1. Baseline Unsupervised (k=3, m=0.05)")

    r_relaxed = run_blocking_pass(s1_name_base, s1_addr_base, q_name_base, q_addr_base,
                                  k=6, margin=0.10, tag="2. Relaxed Unsupervised  (k=6, m=0.10)")

    r_proj = run_blocking_pass(s1_name_proj, s1_addr_proj, q_name_proj, q_addr_proj,
                               k=6, margin=0.10, tag="3. Supervised Projector  (k=6, m=0.10)")

    print("=" * 75)
    print(f"  NET GAIN OVER BASELINE: {r_proj - r_base:+.2%} ({r_base:.2%} -> {r_proj:.2%})", flush=True)
    print("=" * 75 + "\n")


if __name__ == "__main__":
    evaluate_recall_lift("India", sample_queries=15000)
