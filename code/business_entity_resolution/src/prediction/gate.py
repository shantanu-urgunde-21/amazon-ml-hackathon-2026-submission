"""
Entity-Level Decision Gate module for Business Entity Resolution.
Optimizes tau_singleton, tau_match, and delta_prob on OOF probabilities to directly
maximize the competition's macro-averaged F_0.5 metric.
"""

import pandas as pd
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
from src.validation.metrics import macro_fbeta


def apply_contradiction_penalties(
    probs: np.ndarray,
    X: Optional[pd.DataFrame] = None,
    penalty_factor: float = 1.0,
) -> np.ndarray:
    """Passes through probabilities; contradiction features are weighted natively by GBDT."""
    return probs



def group_pair_predictions(
    probs: np.ndarray,
    pair_keys: List[Tuple[str, str]],
    all_s1_ids: Iterable[str] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Groups pair probabilities by source1_entity_id:
    Returns dict: s1_id -> [(cand_id, probability), ...]
    Ensures every s1_id in all_s1_ids exists in the dictionary.
    """
    grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    if all_s1_ids:
        for s1_id in all_s1_ids:
            grouped[s1_id] = []

    for prob, (s1_id, cand_id) in zip(probs, pair_keys):
        grouped[s1_id].append((cand_id, float(prob)))

    return grouped



def apply_decision_gate(
    grouped_preds: Dict[str, List[Tuple[str, float]]],
    tau_singleton: float = 0.60,
    tau_match: float = 0.55,
    delta_prob: float = 0.15,
) -> Dict[str, List[str]]:
    """
    Applies the entity-level decision rule:
    1. If max(P) < tau_singleton: predict empty set (singleton protection).
    2. Else, include candidate j when P_j >= tau_match and (max(P) - P_j) <= delta_prob.
    """
    final_matches: Dict[str, List[str]] = {}

    for s1_id, cand_list in grouped_preds.items():
        if not cand_list:
            final_matches[s1_id] = []
            continue

        probs = [p for _, p in cand_list]
        max_p = max(probs)

        # Step 1: Singleton protection
        if max_p < tau_singleton:
            final_matches[s1_id] = []
            continue

        # Step 2: Multi-candidate inclusion
        matched = []
        for cand_id, p in cand_list:
            if p >= tau_match and (max_p - p) <= delta_prob:
                matched.append(cand_id)

        final_matches[s1_id] = matched

    return final_matches


def optimize_decision_thresholds(
    oof_probs: np.ndarray,
    pair_keys: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str],
    X: Optional[pd.DataFrame] = None,
) -> Tuple[float, float, float, float]:
    """
    Performs grid search over (tau_singleton, tau_match, delta_prob) to find
    parameters that maximize macro F_0.5 on Out-Of-Fold predictions.
    """
    if X is not None:
        oof_probs = apply_contradiction_penalties(oof_probs, X)

    print("Grouping OOF predictions by S1 entity...")
    grouped_oof = group_pair_predictions(oof_probs, pair_keys, all_s1_ids=all_s1_ids)


    # Convert ground truth subset to dict of sets for scoring
    eval_gt = {s1: ground_truth.get(s1, set()) for s1 in all_s1_ids}

    best_score = -1.0
    best_params = (0.60, 0.55, 0.15)

    print("Searching optimal decision gate thresholds for macro F_0.5...")
    for tau_sing in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for tau_m in [0.40, 0.45, 0.50, 0.55, 0.60]:
            for delta_p in [0.10, 0.15, 0.20]:
                preds = apply_decision_gate(
                    grouped_oof,
                    tau_singleton=tau_sing,
                    tau_match=tau_m,
                    delta_prob=delta_p,
                )
                score = macro_fbeta(eval_gt, {k: set(v) for k, v in preds.items()}, beta=0.5)

                if score > best_score:
                    best_score = score
                    best_params = (tau_sing, tau_m, delta_p)

    print(f"  [BEST] Thresholds found: tau_singleton={best_params[0]}, tau_match={best_params[1]}, delta_prob={best_params[2]}")
    print(f"  [BEST] Benchmark OOF Macro F_0.5: {best_score:.4f}")

    return best_params[0], best_params[1], best_params[2], best_score
