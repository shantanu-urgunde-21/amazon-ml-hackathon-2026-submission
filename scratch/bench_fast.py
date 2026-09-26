"""
Fast Head-to-Head Benchmark: Baseline (28 features) vs Triangulation (33 features).
Loads pre-normalized cache and runs in ~10 seconds.
Run directly in your terminal:
    python scratch/bench_fast.py
"""
import sys
import time
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = PROJECT_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE_ROOT))

from src.filters.blocking import generate_candidates_for_s1, evaluate_blocking_recall
from src.features.extractor import extract_pair_features, FEATURE_NAMES
from src.models.classifier import cross_validate_lgbm
from src.prediction.gate import optimize_decision_thresholds, apply_decision_gate, group_pair_predictions
from src.validation.metrics import macro_fbeta

CACHE_FILE = PROJECT_ROOT / "scratch" / "norm_cache.pkl"

def main():
    print("=" * 70, flush=True)
    print("  Amazon ML Challenge 2026: Fast Triangulation & Margin Benchmark", flush=True)
    print("=" * 70, flush=True)

    if not CACHE_FILE.exists():
        print(f"\n[ERROR] Cache file not found at: {CACHE_FILE}")
        print("Please run cache generator first:\n    python scratch/prepare_cache.py\n")
        return

    # 1. Load Cache
    print("\n[Step 1/5] Loading pre-normalized cache from disk...", flush=True)
    t0 = time.time()
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)

    df_s1 = cache["df_s1"]
    df_pool = cache["df_pool"]
    eval_gt = cache["eval_gt"]
    eval_s1_ids = cache["eval_s1_ids"]
    print(f"  -> Loaded S1 ({len(df_s1):,} rows) and Pool ({len(df_pool):,} rows) in {time.time() - t0:.2f}s!", flush=True)

    # 2. Inverted Index Blocking
    print("\n[Step 2/5] Running Inverted-Index Blocking (Tokens, Compound, Phonetics, 3-Grams)...", flush=True)
    t0 = time.time()
    candidates = generate_candidates_for_s1(df_s1, df_pool)
    recall_stats = evaluate_blocking_recall(candidates, eval_gt)
    print(f"  -> Blocking complete in {time.time() - t0:.2f}s")
    print(f"  -> Candidate Blocking Recall: {recall_stats['candidate_recall'] * 100:.2f}% ({recall_stats['captured_links']:,} / {recall_stats['total_true_links']:,} links)")
    print(f"  -> Avg candidates per S1:     {recall_stats['avg_candidates_per_s1']:.1f}", flush=True)

    # 3. Base Feature Extraction
    print("\n[Step 3/5] Extracting Rapidfuzz C++ pairwise features...", flush=True)
    t0 = time.time()
    pool_lookup = {row.entity_id: row for row in df_pool.itertuples()}
    s1_lookup = {row.entity_id: row for row in df_s1.itertuples()}

    pair_keys = []
    base_rows = []
    labels = []
    s1_to_indices = defaultdict(list)
    curr = 0

    for s1_id in eval_s1_ids:
        if s1_id not in s1_lookup:
            continue
        s1_row = s1_lookup[s1_id]
        cands = candidates.get(s1_id, [])
        true_set = eval_gt.get(s1_id, set())

        for cid in cands:
            if cid not in pool_lookup:
                continue
            cand_row = pool_lookup[cid]
            f = extract_pair_features(s1_row, cand_row)
            base_rows.append(f)
            pair_keys.append((s1_id, cid))
            labels.append(1 if cid in true_set else 0)
            s1_to_indices[s1_id].append(curr)
            curr += 1

    y = np.array(labels, dtype=np.int32)
    print(f"  -> Extracted {len(pair_keys):,} candidate pairs in {time.time() - t0:.2f}s (True matches: {y.sum():,})", flush=True)

    # 4. Model A: Baseline 28-feature model
    print("\n[Step 4/5] Training Model A: 28-feature Baseline (5-Fold GroupKFold)...", flush=True)
    t0 = time.time()
    baseline_rows = []
    for s1_id, indices in s1_to_indices.items():
        b_size = len(indices)
        sims = [base_rows[i][0] for i in indices]
        max_sim = max(sims)
        sorted_order = np.argsort(-np.array(sims))
        ranks = {indices[sorted_order[r]]: float(r) for r in range(b_size)}
        for i in indices:
            b_feat = base_rows[i]
            sim = b_feat[0]
            margin = sim - max_sim
            rank = ranks[i]
            baseline_rows.append(b_feat + [rank, margin, float(b_size)])

    X_base = pd.DataFrame(baseline_rows, columns=FEATURE_NAMES)
    oof_base, _ = cross_validate_lgbm(X_base, y, pair_keys=pair_keys, n_splits=5)
    tau_sing_b, tau_m_b, delta_p_b, _ = optimize_decision_thresholds(
        oof_base, pair_keys, eval_gt, all_s1_ids=eval_s1_ids, X=X_base
    )
    grouped_base = group_pair_predictions(oof_base, pair_keys, all_s1_ids=eval_s1_ids)
    preds_base = apply_decision_gate(grouped_base, tau_singleton=tau_sing_b, tau_match=tau_m_b, delta_prob=delta_p_b)
    m_base = macro_fbeta(eval_gt, {k: set(v) for k, v in preds_base.items()}, beta=0.5)
    print(f"  -> Model A Baseline Macro-F0.5: {m_base:.4f} (CV time: {time.time() - t0:.2f}s)", flush=True)

    # 5. Model B: Enhanced 33-feature Triangulation & Source Margin model
    print("\n[Step 5/5] Building Triangulation & Source Margins (Model B, 33 features)...", flush=True)
    t0 = time.time()
    NEW_FEATURE_NAMES = FEATURE_NAMES + [
        "cand_src_rank",          # Rank within same source (S2 or S3)
        "cand_src_margin",        # Margin from top candidate of same source
        "cross_src_max_sim",      # Max name similarity with candidate in other source
        "cross_src_has_partner",  # Strong cross-source partner (>0.80) present
        "cross_src_postal_agree", # Candidate in other source shares identical postal code
    ]

    enhanced_rows = []
    for s1_id, indices in s1_to_indices.items():
        b_size = len(indices)
        sims = [base_rows[i][0] for i in indices]
        max_sim = max(sims)
        sorted_order = np.argsort(-np.array(sims))
        ranks = {indices[sorted_order[r]]: float(r) for r in range(b_size)}

        s2_indices = [i for i in indices if pair_keys[i][1].startswith("s2_") or pool_lookup[pair_keys[i][1]].source == "s2"]
        s3_indices = [i for i in indices if pair_keys[i][1].startswith("s3_") or pool_lookup[pair_keys[i][1]].source == "s3"]

        s2_sims = {i: base_rows[i][0] for i in s2_indices}
        s2_max = max(s2_sims.values()) if s2_sims else 0.0
        s2_sorted = sorted(s2_indices, key=lambda idx: s2_sims[idx], reverse=True)
        s2_ranks = {idx: float(r) for r, idx in enumerate(s2_sorted)}

        s3_sims = {i: base_rows[i][0] for i in s3_indices}
        s3_max = max(s3_sims.values()) if s3_sims else 0.0
        s3_sorted = sorted(s3_indices, key=lambda idx: s3_sims[idx], reverse=True)
        s3_ranks = {idx: float(r) for r, idx in enumerate(s3_sorted)}

        s2_cross_max = {}
        s2_cross_post = {}
        for i2 in s2_indices:
            r2 = pool_lookup[pair_keys[i2][1]]
            max_c = 0.0
            post_c = 0.0
            for i3 in s3_indices:
                r3 = pool_lookup[pair_keys[i3][1]]
                c_sim = fuzz.token_sort_ratio(r2.norm_name, r3.norm_name) / 100.0
                if c_sim > max_c:
                    max_c = c_sim
                if r2.postal_code and r3.postal_code and r2.postal_code == r3.postal_code:
                    post_c = 1.0
            s2_cross_max[i2] = max_c
            s2_cross_post[i2] = post_c

        s3_cross_max = {}
        s3_cross_post = {}
        for i3 in s3_indices:
            r3 = pool_lookup[pair_keys[i3][1]]
            max_c = 0.0
            post_c = 0.0
            for i2 in s2_indices:
                r2 = pool_lookup[pair_keys[i2][1]]
                c_sim = fuzz.token_sort_ratio(r3.norm_name, r2.norm_name) / 100.0
                if c_sim > max_c:
                    max_c = c_sim
                if r3.postal_code and r2.postal_code and r3.postal_code == r2.postal_code:
                    post_c = 1.0
            s3_cross_max[i3] = max_c
            s3_cross_post[i3] = post_c

        for i in indices:
            b_feat = base_rows[i]
            sim = b_feat[0]
            margin = sim - max_sim
            rank = ranks[i]
            is_s2 = (pair_keys[i][1].startswith("s2_") or pool_lookup[pair_keys[i][1]].source == "s2")

            if is_s2:
                src_rank = s2_ranks.get(i, 0.0)
                src_margin = sim - s2_max
                c_max = s2_cross_max.get(i, 0.0)
                c_post = s2_cross_post.get(i, 0.0)
            else:
                src_rank = s3_ranks.get(i, 0.0)
                src_margin = sim - s3_max
                c_max = s3_cross_max.get(i, 0.0)
                c_post = s3_cross_post.get(i, 0.0)

            c_partner = 1.0 if c_max >= 0.80 else 0.0
            enhanced_rows.append(
                b_feat + [rank, margin, float(b_size), src_rank, src_margin, c_max, c_partner, c_post]
            )

    X_enh = pd.DataFrame(enhanced_rows, columns=NEW_FEATURE_NAMES)
    oof_enh, _ = cross_validate_lgbm(X_enh, y, pair_keys=pair_keys, n_splits=5)
    tau_sing_e, tau_m_e, delta_p_e, _ = optimize_decision_thresholds(
        oof_enh, pair_keys, eval_gt, all_s1_ids=eval_s1_ids, X=X_enh
    )
    grouped_enh = group_pair_predictions(oof_enh, pair_keys, all_s1_ids=eval_s1_ids)
    preds_enh = apply_decision_gate(grouped_enh, tau_singleton=tau_sing_e, tau_match=tau_m_e, delta_prob=delta_p_e)
    m_enh = macro_fbeta(eval_gt, {k: set(v) for k, v in preds_enh.items()}, beta=0.5)

    delta = m_enh - m_base
    print("\n" + "=" * 70)
    print("                      HEAD-TO-HEAD RESULTS")
    print("=" * 70)
    print(f"  Model A (Baseline, 28 features): Macro-F0.5 = {m_base:.4f}")
    print(f"  Model B (Triangulation, 33 feat): Macro-F0.5 = {m_enh:.4f}")
    print(f"  DELTA Macro-F0.5:                 {'+' if delta >= 0 else ''}{delta:.4f}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
