"""
Evaluation metrics module for Amazon ML Challenge 2026.
Implements exact macro-averaged F_beta (beta = 0.5) with full singleton support.
"""

from typing import Dict, Iterable, Set, Union
import numpy as np


def compute_s1_fbeta(
    pred_ids: Union[Set[str], Iterable[str]],
    true_ids: Union[Set[str], Iterable[str]],
    beta: float = 0.5,
) -> float:
    """
    Computes F_beta score for a single Source 1 entity.
    
    Singleton logic per competition rules:
    - If true_ids is empty and pred_ids is empty -> 1.0 (correct singleton identification)
    - If true_ids is empty and pred_ids is non-empty -> 0.0 (false merge penalty)
    - If true_ids is non-empty and pred_ids is empty -> 0.0 (missed all links)
    """
    if not isinstance(pred_ids, set):
        pred_ids = set(pred_ids)
    if not isinstance(true_ids, set):
        true_ids = set(true_ids)

    # Both empty: correctly identified singleton
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0

    # One empty while other is not
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)

    b2 = beta ** 2  # 0.25 for beta=0.5
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0

    return (1.0 + b2) * (precision * recall) / denom


def macro_fbeta(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5,
) -> float:
    """
    Computes macro-averaged F_beta score across all Source 1 entities in ground truth.
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(compute_s1_fbeta(pred_set, true_set, beta=beta))

    return float(np.mean(scores)) if scores else 0.0


def self_test():
    """Validates metric against competition specification test cases."""
    print("Running metrics verification...")
    pred1 = {"S2-00047", "S2-00193", "S3-00812"}
    true1 = {"S2-00047", "S3-00812"}
    score1 = compute_s1_fbeta(pred1, true1, beta=0.5)
    assert abs(score1 - 0.7142857) < 1e-4, f"Test 1 failed: expected ~0.7142857, got {score1}"
    print(f"  [PASS] PDF example passed: F_0.5 = {score1:.4f}")

    assert compute_s1_fbeta(set(), set()) == 1.0, "Test 2 failed: empty set vs empty set should be 1.0"
    print("  [PASS] Correct singleton identification: 1.0")

    assert compute_s1_fbeta({"S2-00001"}, set()) == 0.0, "Test 3 failed: false merge should be 0.0"
    print("  [PASS] False merge on singleton penalty: 0.0")

    assert compute_s1_fbeta(set(), {"S2-00001"}) == 0.0, "Test 4 failed: missed match should be 0.0"
    print("  [PASS] Missed match penalty: 0.0")

    gt = {"S1-1": true1, "S1-2": set(), "S1-3": {"S2-99"}}
    preds = {"S1-1": pred1, "S1-2": set(), "S1-3": set()}
    macro = macro_fbeta(gt, preds)
    expected_macro = (score1 + 1.0 + 0.0) / 3.0
    assert abs(macro - expected_macro) < 1e-5, f"Test 5 failed: expected {expected_macro}, got {macro}"
    print(f"  [PASS] Macro average passed: {macro:.4f}")
    print("All metric tests passed successfully!")


if __name__ == "__main__":
    self_test()
