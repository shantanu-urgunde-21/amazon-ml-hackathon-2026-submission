"""Phase 1: Granular Blocking Miss Diagnostic Script (Vectorized High-Performance).

Classifies candidate blocking misses on the holdout split into:
  1. wrong_partition: S1 and Pool have conflicting, non-empty state partitions
  2. unobserved_state_s1: S1 has empty state (unreachable by queries with known state)
  3. unobserved_state_pool: Pool has empty state (searched all via fallback, but still missed)
  4. same_partition_miss: Same partition, but never made it into the candidate table
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd

import src.config as config
from src.filters.blocking import _partition
from src.pipeline import (
    RETRIEVAL_COLUMNS,
    candidates,
    countries,
    get_split,
    load_split,
)


def run_diagnostic(sample_holdout: int = None):
    print("=" * 75, flush=True)
    print("PHASE 1: GRANULAR BLOCKING MISS DIAGNOSTIC", flush=True)
    print("=" * 75, flush=True)

    split_info = get_split()
    holdout_set = set(split_info["holdout"])
    if sample_holdout is not None and len(holdout_set) > sample_holdout:
        rng = np.random.default_rng(config.RANDOM_SEED + 99)
        holdout_list = sorted(holdout_set)
        holdout_set = set(np.array(holdout_list)[rng.choice(len(holdout_list), sample_holdout, replace=False)])
        print(f"Subsampled holdout to {len(holdout_set):,} entities for rapid diagnosis.", flush=True)
    else:
        print(f"Using full holdout set: {len(holdout_set):,} entities.", flush=True)

    print("\nLoading ground truth links...", flush=True)
    gt_df = pd.read_csv(config.TRAIN_GROUND_TRUTH_PATH, sep="\t", dtype="string[pyarrow]", keep_default_na=False)
    gt_links = gt_df[gt_df["matched_entity_ids"] != ""].assign(m=lambda d: d["matched_entity_ids"].str.split(",")).explode("m")
    gt_links = gt_links[gt_links["source1_entity_id"].isin(holdout_set)]
    all_holdout_pairs = set(zip(gt_links["source1_entity_id"].to_numpy(), gt_links["m"].to_numpy()))
    print(f"Total true links in holdout: {len(all_holdout_pairs):,}\n", flush=True)

    summary_rows = []

    for country in countries("train"):
        print(f"[{country}] Loading normalized tables and candidates...", flush=True)
        s1, pool = load_split("train", country, RETRIEVAL_COLUMNS)
        cands = candidates("train", country)

        s1_map = s1["entity_id"].to_numpy()
        pool_map = pool["entity_id"].to_numpy()

        # Ultra-fast boolean index lookup (O(1))
        s1_in_holdout = s1["entity_id"].isin(holdout_set).to_numpy()
        cands_s1_idx = cands["s1_idx"].to_numpy()
        cands_pool_idx = cands["pool_idx"].to_numpy()

        cands_holdout_mask = s1_in_holdout[cands_s1_idx]
        cand_pairs = set(zip(s1_map[cands_s1_idx[cands_holdout_mask]], pool_map[cands_pool_idx[cands_holdout_mask]]))

        # Filter true links for this country
        s1_country_ids = set(s1_map)
        country_true_pairs = {p for p in all_holdout_pairs if p[0] in s1_country_ids}
        total_links = len(country_true_pairs)
        captured_pairs = country_true_pairs & cand_pairs
        captured_links = len(captured_pairs)
        missed_links = list(country_true_pairs - cand_pairs)

        recall = captured_links / max(1, total_links)
        n_miss = len(missed_links)
        print(f"  True Links: {total_links:,} | Captured: {captured_links:,} ({recall:.2%}) | Missed: {n_miss:,}", flush=True)

        if not missed_links:
            continue

        # Fast vectorized miss analysis
        miss_df = pd.DataFrame(missed_links, columns=["s1_id", "p_id"])
        miss_df = miss_df.merge(s1.rename(columns=lambda c: f"s1_{c}"), left_on="s1_id", right_on="s1_entity_id", how="inner")
        miss_df = miss_df.merge(pool.rename(columns=lambda c: f"p_{c}"), left_on="p_id", right_on="p_entity_id", how="inner")

        s1_part = _partition(miss_df["s1_state"].fillna(""))
        p_part = _partition(miss_df["p_state"].fillna(""))

        m_unobs_s1 = (miss_df["s1_state"] == "")
        m_unobs_pool = (~m_unobs_s1) & (miss_df["p_state"] == "")
        m_wrong_part = (~m_unobs_s1) & (~m_unobs_pool) & (s1_part != p_part)
        m_same_part = (~m_unobs_s1) & (~m_unobs_pool) & (s1_part == p_part)

        cnt_unobs_s1 = int(m_unobs_s1.sum())
        cnt_unobs_pool = int(m_unobs_pool.sum())
        cnt_wrong_part = int(m_wrong_part.sum())
        cnt_same_part = int(m_same_part.sum())

        print(f"\n  --- Miss Classification Breakdown ({n_miss:,} total misses) ---", flush=True)
        print(f"    {'wrong_partition':<28}: {cnt_wrong_part:>5} ({cnt_wrong_part/max(1, n_miss):>6.1%})", flush=True)
        print(f"    {'unobserved_state_s1':<28}: {cnt_unobs_s1:>5} ({cnt_unobs_s1/max(1, n_miss):>6.1%})", flush=True)
        print(f"    {'unobserved_state_pool':<28}: {cnt_unobs_pool:>5} ({cnt_unobs_pool/max(1, n_miss):>6.1%})", flush=True)
        print(f"    {'same_partition_miss':<28}: {cnt_same_part:>5} ({cnt_same_part/max(1, n_miss):>6.1%})", flush=True)

        summary_rows.append({
            "Country": country,
            "Total Links": total_links,
            "Captured": captured_links,
            "Recall": f"{recall:.2%}",
            "Total Misses": n_miss,
            "Wrong Partition": f"{cnt_wrong_part} ({cnt_wrong_part/max(1, n_miss):.1%})",
            "Unobs S1 State": f"{cnt_unobs_s1} ({cnt_unobs_s1/max(1, n_miss):.1%})",
            "Unobs Pool State": f"{cnt_unobs_pool} ({cnt_unobs_pool/max(1, n_miss):.1%})",
            "Same Partition Miss": f"{cnt_same_part} ({cnt_same_part/max(1, n_miss):.1%})",
        })

        # Sample inspections
        print(f"\n  [Sample: wrong_partition ({cnt_wrong_part} total)]", flush=True)
        for _, row in miss_df[m_wrong_part].head(3).iterrows():
            print(f"    S1   ({row['s1_id']}): '{row['s1_name_core']} {row['s1_name_alias']}' | '{row['s1_addr_norm']}' | State: '{row['s1_state']}'", flush=True)
            print(f"    Pool ({row['p_id']}): '{row['p_name_core']} {row['p_name_alias']}' | '{row['p_addr_norm']}' | State: '{row['p_state']}'", flush=True)
            print("    " + "-" * 50, flush=True)

        print(f"\n  [Sample: same_partition_miss ({cnt_same_part} total)]", flush=True)
        for _, row in miss_df[m_same_part].head(3).iterrows():
            print(f"    S1   ({row['s1_id']}): '{row['s1_name_core']} {row['s1_name_alias']}' | '{row['s1_addr_norm']}' | State: '{row['s1_state']}'", flush=True)
            print(f"    Pool ({row['p_id']}): '{row['p_name_core']} {row['p_name_alias']}' | '{row['p_addr_norm']}' | State: '{row['p_state']}'", flush=True)
            print("    " + "-" * 50, flush=True)
        print("", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("PHASE 1 DIAGNOSTIC SUMMARY TABLE", flush=True)
    print("=" * 75, flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=25000, help="Holdout sample size")
    args = parser.parse_args()
    run_diagnostic(sample_holdout=args.sample)
