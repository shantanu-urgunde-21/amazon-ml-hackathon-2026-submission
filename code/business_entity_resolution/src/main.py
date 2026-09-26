"""
Main executable pipeline for Business Entity Resolution (Amazon ML Challenge 2026).
Coordinates data loading, normalization, candidate blocking, feature extraction,
LightGBM training, entity-level decision gating, and submission file generation.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add package root and src directory to sys.path so both direct scripts and package imports work
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PKG_DIR = SRC_DIR.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

# Root package imports
import src.config as config
from src.dataloader.loader import load_source, load_ground_truth
from src.normalization.normalizer import normalize_records
from src.filters.blocking import generate_candidates_for_s1, evaluate_blocking_recall
from src.features.extractor import build_feature_matrix
from src.models.classifier import cross_validate_lgbm
from src.prediction.gate import (
    apply_decision_gate,
    optimize_decision_thresholds,
    group_pair_predictions,
    apply_contradiction_penalties,
)
from src.validation.metrics import macro_fbeta



def write_submission_files(
    all_s1_ids: List[str],
    candidates_dict: Dict[str, List[str]],
    matches_dict: Dict[str, List[str]],
    output_dir: Path = config.OUTPUT_DIR,
):
    """
    Writes candidate_pairs.tsv and matching_results.tsv ensuring:
    - Exactly one row per S1 entity
    - S2/S3 IDs comma-separated with no spaces
    - Matches are strict subset of candidates
    - No header corruption, explicit tab separator
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cand_path = output_dir / "candidate_pairs.tsv"
    match_path = output_dir / "matching_results.tsv"

    print(f"Writing {cand_path} for {len(all_s1_ids)} S1 entities...")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates_dict.get(s1_id, [])
            cands_str = ",".join(cands)
            f.write(f"{s1_id}\t{cands_str}\n")

    print(f"Writing {match_path} for {len(all_s1_ids)} S1 entities...")
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = matches_dict.get(s1_id, [])
            # Filter matches to strictly be in candidate list
            cand_set = set(candidates_dict.get(s1_id, []))
            valid_matches = [m for m in matches if m in cand_set]
            matches_str = ",".join(valid_matches)
            f.write(f"{s1_id}\t{matches_str}\n")

    print("Submission files written successfully.")


def load_source_with_targets(
    path: Path,
    target_ids: Set[str],
    sample_distractors: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads source data ensuring all required target_ids are retained,
    combined with a random sample of distractors.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""

    if not target_ids and sample_distractors is None:
        return df

    is_target = df["entity_id"].isin(target_ids)
    target_df = df[is_target]
    other_df = df[~is_target]

    if sample_distractors is not None and len(other_df) > sample_distractors:
        sampled_other = other_df.sample(n=sample_distractors, random_state=random_state)
    else:
        sampled_other = other_df

    combined = pd.concat([target_df, sampled_other], ignore_index=True)
    return combined


def run_pipeline(sample_n: int = None, run_test: bool = True, seed: int = config.RANDOM_SEED):
    start_time = time.time()
    print("=" * 70)
    print(" Amazon ML Challenge 2026 - Business Entity Resolution Pipeline")
    print("=" * 70)
    if sample_n:
        print(f"*** RANDOM SAMPLE MODE: Evaluating {sample_n:,} randomly sampled entities (seed={seed}) ***")

    # -------------------------------------------------------------
    # 1. Load & Normalize Training Data
    # -------------------------------------------------------------
    print("\n[Step 1/5] Loading training data...")
    if sample_n:
        df_s1_train = load_source(config.TRAIN_SOURCE1_PATH, sample_n=sample_n, random_state=seed)
    else:
        df_s1_train = load_source(config.TRAIN_SOURCE1_PATH)
    s1_train_ids = df_s1_train["entity_id"].tolist()

    if sample_n:
        print(f"  Filtering ground truth for {len(s1_train_ids):,} sampled S1 entities...")
        gt_dict = load_ground_truth(config.TRAIN_GROUND_TRUTH_PATH, target_s1_ids=set(s1_train_ids))
        target_pool_ids = set.union(*gt_dict.values()) if gt_dict else set()
        print(f"  Identified {len(target_pool_ids):,} ground truth candidate entities across S2 & S3.")
        distractor_n = sample_n * 3
        df_s2_train = load_source_with_targets(config.TRAIN_SOURCE2_PATH, target_pool_ids, sample_distractors=distractor_n, random_state=seed)
        df_s3_train = load_source_with_targets(config.TRAIN_SOURCE3_PATH, target_pool_ids, sample_distractors=distractor_n, random_state=seed)
    else:
        gt_dict = load_ground_truth(config.TRAIN_GROUND_TRUTH_PATH)
        df_s2_train = load_source(config.TRAIN_SOURCE2_PATH)
        df_s3_train = load_source(config.TRAIN_SOURCE3_PATH)

    df_pool_train = pd.concat([df_s2_train, df_s3_train], ignore_index=True)
    print(f"  Loaded Train S1: {len(df_s1_train):,}, Pool (S2+S3): {len(df_pool_train):,}")

    print("Normalizing training records...")
    df_s1_train = normalize_records(df_s1_train)
    df_pool_train = normalize_records(df_pool_train)

    # -------------------------------------------------------------
    # 2. Candidate Blocking
    # -------------------------------------------------------------
    print("\n[Step 2/5] Candidate blocking...")
    train_candidates = generate_candidates_for_s1(
        df_s1_train,
        df_pool_train,
        max_candidates_per_s1=config.MAX_CANDIDATES_PER_S1,
    )
    recall_stats = evaluate_blocking_recall(train_candidates, gt_dict)
    print(f"  [METRIC] Candidate Recall: {recall_stats['candidate_recall']:.4f}")
    print(f"  [METRIC] Captured {recall_stats['captured_links']} / {recall_stats['total_true_links']} ground truth links")
    print(f"  [METRIC] Avg Candidates / S1: {recall_stats['avg_candidates_per_s1']:.1f}")

    # -------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------
    print("\n[Step 3/5] Extracting pairwise features...")
    X_train, y_train, train_pair_keys = build_feature_matrix(
        df_s1_train, df_pool_train, train_candidates, ground_truth=gt_dict
    )
    print(f"  Constructed feature matrix: {X_train.shape}, Positive rate: {float(np.mean(y_train)):.4f}")

    # -------------------------------------------------------------
    # 4. Model Training & Threshold Optimization
    # -------------------------------------------------------------
    print("\n[Step 4/5] Training LightGBM with GroupKFold...")
    oof_probs, fold_models = cross_validate_lgbm(
        X_train, y_train, train_pair_keys, n_splits=config.N_FOLDS
    )

    tau_sing, tau_match, delta_prob, oof_f05 = optimize_decision_thresholds(
        oof_probs, train_pair_keys, gt_dict, s1_train_ids, X=X_train
    )

    print(f"\n>>> BENCHMARK OOF MACRO F_0.5: {oof_f05:.4f} <<<")

    # -------------------------------------------------------------
    # 5. Test Inference (if requested)
    # -------------------------------------------------------------
    if run_test and config.TEST_SOURCE1_PATH.exists():
        print("\n[Step 5/5] Generating predictions for Test Set...")
        if sample_n:
            print(f"  Randomly sampling {sample_n:,} entities from Test S1 (seed={seed})...")
            df_s1_test = load_source(config.TEST_SOURCE1_PATH, sample_n=sample_n, random_state=seed)
            pool_sample = sample_n * 5
            df_s2_test = load_source(config.TEST_SOURCE2_PATH, sample_n=pool_sample, random_state=seed)
            df_s3_test = load_source(config.TEST_SOURCE3_PATH, sample_n=pool_sample, random_state=seed)
        else:
            df_s1_test = load_source(config.TEST_SOURCE1_PATH)
            df_s2_test = load_source(config.TEST_SOURCE2_PATH)
            df_s3_test = load_source(config.TEST_SOURCE3_PATH)

        test_s1_ids = df_s1_test["entity_id"].tolist()
        df_pool_test = pd.concat([df_s2_test, df_s3_test], ignore_index=True)

        print(f"  Loaded Test S1: {len(df_s1_test):,}, Test Pool (S2+S3): {len(df_pool_test):,}")

        df_s1_test = normalize_records(df_s1_test)
        df_pool_test = normalize_records(df_pool_test)

        test_candidates = generate_candidates_for_s1(
            df_s1_test, df_pool_test, max_candidates_per_s1=config.MAX_CANDIDATES_PER_S1
        )

        X_test, _, test_pair_keys = build_feature_matrix(
            df_s1_test, df_pool_test, test_candidates, ground_truth=None
        )

        print(f"  Scoring {len(X_test):,} test pairs with {len(fold_models)} ensemble models...")
        test_preds = np.zeros(len(X_test), dtype=np.float32)
        if len(X_test) > 0:
            for model in fold_models:
                test_preds += model.predict(X_test, num_iteration=model.best_iteration)
            test_preds /= len(fold_models)
            test_preds = apply_contradiction_penalties(test_preds, X_test)

        grouped_test = group_pair_predictions(test_preds, test_pair_keys, all_s1_ids=test_s1_ids)

        test_matches = apply_decision_gate(
            grouped_test,
            tau_singleton=tau_sing,
            tau_match=tau_match,
            delta_prob=delta_prob,
        )

        write_submission_files(test_s1_ids, test_candidates, test_matches)

    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size for rapid verification (e.g. 10000)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for sampling")
    parser.add_argument("--no-test", action="store_true", help="Skip test inference")
    args = parser.parse_args()

    run_pipeline(sample_n=args.sample, run_test=not args.no_test, seed=args.seed)

