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


from collections import defaultdict
from datetime import datetime
import json
from src.features.extractor import build_feature_matrix, FEATURE_NAMES
from src.validation.metrics import macro_fbeta, compute_s1_fbeta


def evaluate_and_log_experiment(
    df_s1_train: pd.DataFrame,
    df_pool_train: pd.DataFrame,
    train_candidates: Dict[str, List[str]],
    recall_stats: dict,
    oof_probs: np.ndarray,
    train_pair_keys: list,
    gt_dict: Dict[str, Set[str]],
    s1_train_ids: list,
    X_train: np.ndarray,
    fold_models: list,
    tau_sing: float,
    tau_match: float,
    delta_prob: float,
    timing_dict: dict,
    tag: str = "Phase-3-10k-Random-Sample",
    description: str = "10k Random S1 Sample with Full Pool Ground Truth Alignment & 5-Fold LGBM",
):
    print("\n[Step 4b] Computing detailed failure attribution & metrics breakdown...")
    grouped_oof = group_pair_predictions(apply_contradiction_penalties(oof_probs, X_train), train_pair_keys, all_s1_ids=s1_train_ids)
    preds = apply_decision_gate(grouped_oof, tau_singleton=tau_sing, tau_match=tau_match, delta_prob=delta_prob)
    pred_dict = {k: set(v) for k, v in preds.items()}

    singletons = {s1 for s1, true_set in gt_dict.items() if len(true_set) == 0}
    non_singletons = {s1 for s1, true_set in gt_dict.items() if len(true_set) > 0}

    sing_correct = sum(1 for s1 in singletons if len(pred_dict.get(s1, set())) == 0)
    singleton_acc = sing_correct / len(singletons) if singletons else 1.0

    non_sing_scores = []
    non_sing_precisions = []
    non_sing_recalls = []
    for s1 in non_singletons:
        t_set = gt_dict[s1]
        p_set = pred_dict.get(s1, set())
        non_sing_scores.append(compute_s1_fbeta(p_set, t_set, beta=0.5))
        tp = len(p_set & t_set)
        non_sing_precisions.append(tp / len(p_set) if len(p_set) > 0 else 0.0)
        non_sing_recalls.append(tp / len(t_set) if len(t_set) > 0 else 0.0)

    non_sing_f05 = float(np.mean(non_sing_scores)) if non_sing_scores else 0.0
    macro_p_non_sing = float(np.mean(non_sing_precisions)) if non_sing_precisions else 0.0
    macro_r_non_sing = float(np.mean(non_sing_recalls)) if non_sing_recalls else 0.0
    total_f05 = macro_fbeta(gt_dict, pred_dict, beta=0.5)

    country_map = dict(zip(df_s1_train["entity_id"], df_s1_train["country"]))
    country_scores = defaultdict(list)
    for s1, t_set in gt_dict.items():
        c = country_map.get(s1, "Unknown")
        country_scores[c].append(compute_s1_fbeta(pred_dict.get(s1, set()), t_set, beta=0.5))
    country_f05 = {c: float(np.mean(sc)) for c, sc in country_scores.items()}

    importances = np.mean([m.feature_importance(importance_type="gain") for m in fold_models], axis=0)
    top_indices = np.argsort(-importances)[:10]
    top_features = [{"feature": FEATURE_NAMES[i], "gain": float(importances[i])} for i in top_indices if i < len(FEATURE_NAMES)]

    blocking_misses = recall_stats["total_true_links"] - recall_stats["captured_links"]
    scoring_misses = 0
    false_merges = 0
    singleton_false_merges = len(singletons) - sing_correct
    for s1 in non_singletons:
        t_set = gt_dict[s1]
        p_set = pred_dict.get(s1, set())
        c_set = set(train_candidates.get(s1, []))
        for m in (t_set & c_set):
            if m not in p_set:
                scoring_misses += 1
        for p in p_set:
            if p not in t_set:
                false_merges += 1

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "description": description,
        "s1_entities": len(df_s1_train),
        "pool_records": len(df_pool_train),
        "pair_count": len(X_train),
        "feature_count": X_train.shape[1],
        "blocking_recall": float(recall_stats["candidate_recall"]),
        "captured_links": int(recall_stats["captured_links"]),
        "total_true_links": int(recall_stats["total_true_links"]),
        "avg_candidates_per_s1": float(recall_stats["avg_candidates_per_s1"]),
        "macro_f05": float(total_f05),
        "singleton_accuracy": float(singleton_acc),
        "non_singleton_f05": float(non_sing_f05),
        "macro_precision_non_sing": float(macro_p_non_sing),
        "macro_recall_non_sing": float(macro_r_non_sing),
        "total_singletons": len(singletons),
        "total_non_singletons": len(non_singletons),
        "country_f05": country_f05,
        "failure_counts": {
            "blocking_misses": int(blocking_misses),
            "scoring_misses": int(scoring_misses),
            "false_merges": int(false_merges),
            "singleton_false_merges": int(singleton_false_merges),
        },
        "optimal_thresholds": {
            "tau_singleton": float(tau_sing),
            "tau_match": float(tau_match),
            "delta_prob": float(delta_prob),
        },
        "timing_seconds": timing_dict,
        "top_features": top_features,
    }

    history_file = config.PROJECT_ROOT / "docs" / "experiment_history.json"
    data = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [LOG] Appended run record to {history_file}")

    tracker_file = config.PROJECT_ROOT / "docs" / "EXPERIMENT_TRACKER.md"
    if tracker_file.exists():
        try:
            row_id = f"{len(data):02d}"
            run_row = f"| `{row_id}` | **{tag}** | {X_train.shape[1]} | {recall_stats['candidate_recall']*100:.2f}% | **{total_f05:.4f}** | {non_sing_f05:.4f} | {singleton_acc*100:.1f}% | {country_f05.get('India', 0.0):.4f} | {country_f05.get('US', 0.0):.4f} | {timing_dict.get('total', 0.0):.2f}s | {description} |"
            with open(tracker_file, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.split("\n")
            new_lines = []
            inserted = False
            for line in lines:
                new_lines.append(line)
                if not inserted and line.strip().startswith("| `06`"):
                    new_lines.append(run_row)
                    inserted = True
            with open(tracker_file, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))
            print(f"  [LOG] Updated experiment table in {tracker_file}")
        except Exception as e:
            print(f"  [WARNING] Could not update {tracker_file}: {e}")


def run_pipeline(sample_n: int = None, run_test: bool = True, seed: int = config.RANDOM_SEED, log_exp: bool = True):
    start_time = time.time()
    timing_dict = {}
    print("=" * 70)
    print(" Amazon ML Challenge 2026 - Business Entity Resolution Pipeline")
    print("=" * 70)
    if sample_n:
        print(f"*** RANDOM SAMPLE MODE: Evaluating {sample_n:,} randomly sampled entities (seed={seed}) ***")

    # -------------------------------------------------------------
    # 1. Load & Normalize Training Data
    # -------------------------------------------------------------
    t0 = time.time()
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

    t_norm_start = time.time()
    print("Normalizing training records...")
    df_s1_train = normalize_records(df_s1_train)
    df_pool_train = normalize_records(df_pool_train)
    timing_dict["normalization"] = round(time.time() - t_norm_start, 2)

    # -------------------------------------------------------------
    # 2. Candidate Blocking
    # -------------------------------------------------------------
    t_block_start = time.time()
    print("\n[Step 2/5] Candidate blocking...")
    train_candidates = generate_candidates_for_s1(
        df_s1_train,
        df_pool_train,
        max_candidates_per_s1=config.MAX_CANDIDATES_PER_S1,
    )
    recall_stats = evaluate_blocking_recall(train_candidates, gt_dict)
    timing_dict["blocking"] = round(time.time() - t_block_start, 2)
    print(f"  [METRIC] Candidate Recall: {recall_stats['candidate_recall']:.4f}")
    print(f"  [METRIC] Captured {recall_stats['captured_links']} / {recall_stats['total_true_links']} ground truth links")
    print(f"  [METRIC] Avg Candidates / S1: {recall_stats['avg_candidates_per_s1']:.1f}")

    # -------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------
    t_feat_start = time.time()
    print("\n[Step 3/5] Extracting pairwise features...")
    X_train, y_train, train_pair_keys = build_feature_matrix(
        df_s1_train, df_pool_train, train_candidates, ground_truth=gt_dict
    )
    timing_dict["feature_extraction"] = round(time.time() - t_feat_start, 2)
    print(f"  Constructed feature matrix: {X_train.shape}, Positive rate: {float(np.mean(y_train)):.4f}")

    # -------------------------------------------------------------
    # 4. Model Training & Threshold Optimization
    # -------------------------------------------------------------
    t_train_start = time.time()
    print("\n[Step 4/5] Training LightGBM with GroupKFold...")
    oof_probs, fold_models = cross_validate_lgbm(
        X_train, y_train, train_pair_keys, n_splits=config.N_FOLDS
    )

    tau_sing, tau_match, delta_prob, oof_f05 = optimize_decision_thresholds(
        oof_probs, train_pair_keys, gt_dict, s1_train_ids, X=X_train
    )
    timing_dict["model_training"] = round(time.time() - t_train_start, 2)
    timing_dict["total"] = round(time.time() - start_time, 2)

    print(f"\n>>> BENCHMARK OOF MACRO F_0.5: {oof_f05:.4f} <<<")

    if log_exp:
        evaluate_and_log_experiment(
            df_s1_train=df_s1_train,
            df_pool_train=df_pool_train,
            train_candidates=train_candidates,
            recall_stats=recall_stats,
            oof_probs=oof_probs,
            train_pair_keys=train_pair_keys,
            gt_dict=gt_dict,
            s1_train_ids=s1_train_ids,
            X_train=X_train,
            fold_models=fold_models,
            tau_sing=tau_sing,
            tau_match=tau_match,
            delta_prob=delta_prob,
            timing_dict=timing_dict,
            tag="Phase-3-10k-Random-Sample" if sample_n else "Phase-3-Full-Run",
            description=f"Phase 3: {sample_n or 'Full'} S1 Random Evaluation with Full Pool Ground Truth Alignment",
        )

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
    parser.add_argument("--no-log", action="store_true", help="Skip logging experiment to docs")
    args = parser.parse_args()

    run_pipeline(sample_n=args.sample, run_test=not args.no_test, seed=args.seed, log_exp=not args.no_log)


