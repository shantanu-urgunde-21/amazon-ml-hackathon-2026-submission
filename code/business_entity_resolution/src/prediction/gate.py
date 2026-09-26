"""Entity-level decision gate, tuned to maximize macro F_0.5.

For each S1 entity with candidate probabilities P:
  1. singleton protection: if max(P) < tau_singleton, predict no match
  2. otherwise include candidate j when P_j >= tau_match and max(P) - P_j <= delta_prob

Before the gate, `exclusive_assignment` enforces a fact of the data: every
S2/S3 record matches at most one S1 entity, so a pool record that is a
candidate of several S1 entities keeps only its most probable one.

Everything is vectorized over pair arrays so the threshold search can score
hundreds of thousands of entities per grid point.
"""

from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TAU_SINGLETON_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TAU_MATCH_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
DELTA_PROB_GRID = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def exclusive_assignment(s1_idx: np.ndarray, pool_idx: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Zero out every pair whose pool record has a more probable S1 entity."""
    best = pd.Series(prob).groupby(pool_idx).transform("max").to_numpy()
    order = np.lexsort((-prob, pool_idx))  # ties: keep one
    first = np.ones(len(prob), dtype=bool)
    first[order[1:]] = pool_idx[order[1:]] != pool_idx[order[:-1]]
    return np.where((prob >= best) & first, prob, 0.0).astype(np.float32)


def select_pairs(group: np.ndarray, prob: np.ndarray, tau_singleton: float, tau_match: float,
                 delta_prob: float) -> np.ndarray:
    """Boolean mask of selected pairs. `group` = dense S1 codes (0..n-1)."""
    gmax = np.full(group.max() + 1 if len(group) else 0, -1.0)
    np.maximum.at(gmax, group, prob)
    m = gmax[group]
    return (m >= tau_singleton) & (prob >= tau_match) & (m - prob <= delta_prob)


def macro_fbeta_pairs(group: np.ndarray, selected: np.ndarray, label: np.ndarray,
                      n_true: np.ndarray, beta: float = 0.5) -> float:
    """Macro F_beta over all S1 entities. n_true[g] = number of true links of
    entity g (including links blocking missed); entities without candidates
    are part of n_true too (their prediction is empty)."""
    n = len(n_true)
    tp = np.bincount(group, weights=(selected & (label == 1)), minlength=n)
    pred = np.bincount(group, weights=selected, minlength=n)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(pred > 0, tp / pred, 0.0)
        r = np.where(n_true > 0, tp / n_true, 0.0)
        f = np.where(tp > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    f = np.where((n_true == 0) & (pred == 0), 1.0, f)
    return float(f.mean())


def optimize_decision_thresholds(group, prob, label, n_true, beta: float = 0.5, log=print) -> Tuple[Dict, float]:
    best, best_score = None, -1.0
    for ts, tm, dp in product(TAU_SINGLETON_GRID, TAU_MATCH_GRID, DELTA_PROB_GRID):
        if tm > ts:
            continue
        score = macro_fbeta_pairs(group, select_pairs(group, prob, ts, tm, dp), label, n_true, beta)
        if score > best_score:
            best, best_score = {"tau_singleton": ts, "tau_match": tm, "delta_prob": dp}, score
    log(f"  gate thresholds {best} -> macro F{beta}: {best_score:.4f}")
    return best, best_score


def apply_decision_gate(s1_ids: np.ndarray, cand_ids: np.ndarray, group: np.ndarray, prob: np.ndarray,
                        thresholds: Dict) -> Dict[str, List[str]]:
    sel = select_pairs(group, prob, **thresholds)
    out: Dict[str, List[str]] = {}
    for s1, cand in zip(s1_ids[sel], cand_ids[sel]):
        out.setdefault(s1, []).append(cand)
    return out
