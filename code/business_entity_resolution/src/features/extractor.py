"""
Fast pairwise feature extraction using C++-accelerated rapidfuzz.
Extracts:
1. Name similarity (Levenshtein, Jaro-Winkler, Token-Sort, Token-Set, exact match)
2. Address similarity (Levenshtein, Token-Sort, Token-Set, exact match)
3. Numeric token overlap (count, Jaccard)
4. Cross-script indicator and phonetic skeleton Jaccard
5. Ternary contradiction states (+1 match, 0 missing, -1 contradiction) for:
   - postal_code_state
   - building_number_state
   - unit_state
6. Candidate-relative features (group margins, candidate rank, bucket size)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

FEATURE_NAMES = [
    # Basic Name Features
    "name_levenshtein",
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_exact_match",
    "name_len_diff_ratio",
    "name_prefix_match",
    # Address Features
    "addr_token_set",
    "addr_token_sort",
    "addr_levenshtein",
    "addr_exact_match",
    # Numeric Overlap
    "addr_numeric_overlap_count",
    "addr_numeric_jaccard",
    # Metadata & Cross-Script
    "country_match",
    "cand_is_s2",
    "is_cross_script",
    "phonetic_jaccard",
    # Structured Ternary Contradiction States (+1 agree, 0 missing, -1 conflict)
    "postal_code_state",
    "building_number_state",
    "unit_state",
    # Candidate-Relative Context Features
    "cand_rank_name_sim",
    "cand_margin_name_sim",
    "cand_bucket_size",
]


def slot_ternary_state(val1: str, val2: str) -> float:
    """
    Returns:
      +1.0 if both non-empty and equal (Agreement)
       0.0 if either is missing/empty (Missing / Unavailable)
      -1.0 if both non-empty but different (Explicit Contradiction)
    """
    if not val1 or not val2:
        return 0.0
    return 1.0 if val1 == val2 else -1.0


def extract_pair_features(s1_row, cand_row) -> List[float]:
    """Computes base pairwise similarity features for a single (S1, Candidate) pair."""
    n1 = s1_row.norm_name
    n2 = cand_row.norm_name

    name_lev = fuzz.ratio(n1, n2) / 100.0
    name_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_jw = distance.JaroWinkler.similarity(n1, n2)
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
    max_name_len = max(len(n1), len(n2), 1)
    name_len_diff = abs(len(n1) - len(n2)) / max_name_len
    name_prefix = 1.0 if (len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4]) else 0.0

    a1 = s1_row.norm_address
    a2 = cand_row.norm_address

    addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
    addr_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_lev = fuzz.ratio(a1, a2) / 100.0
    addr_exact = 1.0 if (a1 and a1 == a2) else 0.0

    nums1 = s1_row.numeric_tokens
    nums2 = cand_row.numeric_tokens
    num_intersection = len(nums1 & nums2)
    num_union = len(nums1 | nums2)
    num_jaccard = (num_intersection / num_union) if num_union > 0 else 0.0

    country_eq = 1.0 if (s1_row.country and s1_row.country == cand_row.country) else 0.0
    is_s2 = 1.0 if cand_row.entity_id.startswith("S2-") else 0.0

    has_nl1 = getattr(s1_row, "has_non_latin", False)
    has_nl2 = getattr(cand_row, "has_non_latin", False)
    is_cross_script = 1.0 if (has_nl1 != has_nl2) else 0.0

    ph1 = set(getattr(s1_row, "phonetic_tokens", []))
    ph2 = set(getattr(cand_row, "phonetic_tokens", []))
    ph_union = len(ph1 | ph2)
    phonetic_jaccard = (len(ph1 & ph2) / ph_union) if ph_union > 0 else 0.0

    # Structured slot states (+1 / 0 / -1)
    pin1 = getattr(s1_row, "postal_code", "")
    pin2 = getattr(cand_row, "postal_code", "")
    postal_state = slot_ternary_state(pin1, pin2)

    bldg1 = getattr(s1_row, "building_number", "")
    bldg2 = getattr(cand_row, "building_number", "")
    building_state = slot_ternary_state(bldg1, bldg2)

    unit1 = getattr(s1_row, "unit_slot", "")
    unit2 = getattr(cand_row, "unit_slot", "")
    unit_state = slot_ternary_state(unit1, unit2)

    return [
        name_lev,
        name_sort,
        name_set,
        name_jw,
        name_exact,
        name_len_diff,
        name_prefix,
        addr_set,
        addr_sort,
        addr_lev,
        addr_exact,
        float(num_intersection),
        num_jaccard,
        country_eq,
        is_s2,
        is_cross_script,
        phonetic_jaccard,
        postal_state,
        building_state,
        unit_state,
    ]


def build_feature_matrix(
    df_s1: pd.DataFrame,
    df_pool: pd.DataFrame,
    candidates_dict: Dict[str, List[str]],
    ground_truth: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]:
    """
    Builds feature matrix X including base features and candidate-relative group context
    (rank, margin from top candidate, and bucket size).
    """
    pool_lookup = {row.entity_id: row for row in df_pool.itertuples()}
    s1_lookup = {row.entity_id: row for row in df_s1.itertuples()}

    pair_keys: List[Tuple[str, str]] = []
    base_feature_rows: List[List[float]] = []
    labels: List[int] = []

    has_gt = ground_truth is not None

    # Step 1: Compute base features per S1 group
    s1_to_pair_indices: Dict[str, List[int]] = defaultdict(list)
    current_idx = 0

    for s1_id, cand_ids in candidates_dict.items():
        if s1_id not in s1_lookup:
            continue
        s1_row = s1_lookup[s1_id]
        true_set = ground_truth.get(s1_id, set()) if has_gt else set()

        for cand_id in cand_ids:
            if cand_id not in pool_lookup:
                continue
            cand_row = pool_lookup[cand_id]

            feat = extract_pair_features(s1_row, cand_row)
            base_feature_rows.append(feat)
            pair_keys.append((s1_id, cand_id))
            s1_to_pair_indices[s1_id].append(current_idx)
            current_idx += 1

            if has_gt:
                labels.append(1 if cand_id in true_set else 0)

    # Step 2: Compute Group-Relative Features (Margin from Max, Rank, Bucket Size)
    # Primary similarity index in base features: 0 (name_levenshtein)
    final_feature_rows: List[List[float]] = []
    for s1_id, indices in s1_to_pair_indices.items():
        bucket_size = len(indices)
        if bucket_size == 0:
            continue

        sims = [base_feature_rows[i][0] for i in indices]
        max_sim = max(sims)
        # Rank descending (0 = highest similarity)
        sorted_indices = np.argsort(-np.array(sims))
        ranks = {indices[sorted_indices[r]]: float(r) for r in range(bucket_size)}

        for i in indices:
            base_f = base_feature_rows[i]
            sim = base_f[0]
            margin = sim - max_sim
            rank = ranks[i]
            final_feature_rows.append(base_f + [rank, margin, float(bucket_size)])

    X = pd.DataFrame(final_feature_rows, columns=FEATURE_NAMES)
    y = np.array(labels, dtype=np.int32) if has_gt else np.array([])

    return X, y, pair_keys
