"""
Scalable Inverted Index Blocking for Business Entity Resolution.
Generates candidate pairs between Source 1 and Source 2/3 using:
1. Rare informative name tokens (IDF / frequency capped)
2. Cross-script phonetic skeletons (bridging Hindi-English transliterations)
3. Address numeric tokens (PINs, street numbers)
4. Character 3-grams (for typo tolerance)
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

COMMON_LEGAL_WORDS = {
    "ltd", "limited", "pvt", "private", "inc", "incorporated", "corp", "corporation",
    "llc", "co", "company", "and", "the", "services", "solutions", "enterprises",
    "group", "industries", "sarl", "sas", "sa", "gmbh", "international"
}


def build_inverted_indexes(
    df_pool: pd.DataFrame,
    max_token_bucket: int = 1500,
    max_num_bucket: int = 1000,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Builds inverted indexes mapping tokens/numbers/ngrams/phonetic skeletons to integer row indices.
    """
    token_to_idx: Dict[str, List[int]] = defaultdict(list)
    phonetic_to_idx: Dict[str, List[int]] = defaultdict(list)
    num_to_idx: Dict[str, List[int]] = defaultdict(list)
    ngram_to_idx: Dict[str, List[int]] = defaultdict(list)

    has_phonetic = "phonetic_tokens" in df_pool.columns

    for row in df_pool.itertuples():
        idx = row.Index
        tokens = row.name_tokens
        numbers = row.numeric_tokens
        norm_name = row.norm_name

        # 1. Rare name tokens
        for t in tokens:
            if len(t) >= 3 and t not in COMMON_LEGAL_WORDS:
                token_to_idx[t].append(idx)

        # 2. Phonetic skeletons (cross-script bridge)
        if has_phonetic:
            for ph in row.phonetic_tokens:
                if len(ph) >= 2 and ph not in COMMON_LEGAL_WORDS:
                    phonetic_to_idx[ph].append(idx)

        # 3. Numeric tokens
        for num in numbers:
            num_to_idx[num].append(idx)

        # 4. Character 3-grams of first word
        if len(norm_name) >= 3:
            first_word = tokens[0] if tokens else norm_name[:8]
            if len(first_word) >= 3 and first_word not in COMMON_LEGAL_WORDS:
                for i in range(len(first_word) - 2):
                    ng = first_word[i : i + 3]
                    ngram_to_idx[ng].append(idx)

    pruned_token_idx = {k: v for k, v in token_to_idx.items() if len(v) <= max_token_bucket}
    pruned_phonetic_idx = {k: v for k, v in phonetic_to_idx.items() if len(v) <= max_token_bucket}
    pruned_num_idx = {k: v for k, v in num_to_idx.items() if len(v) <= max_num_bucket}
    pruned_ngram_idx = {k: v for k, v in ngram_to_idx.items() if len(v) <= max_token_bucket}

    return pruned_token_idx, pruned_phonetic_idx, pruned_num_idx, pruned_ngram_idx


def generate_candidates_for_s1(
    df_s1: pd.DataFrame,
    df_pool: pd.DataFrame,
    max_candidates_per_s1: int = 50,
) -> Dict[str, List[str]]:
    """
    Generates candidate S2/S3 entity IDs for each S1 entity in df_s1.
    """
    token_idx, phonetic_idx, num_idx, ngram_idx = build_inverted_indexes(df_pool)
    pool_ids = df_pool["entity_id"].values
    pool_countries = df_pool["country"].values

    candidates_dict: Dict[str, List[str]] = {}
    has_phonetic = "phonetic_tokens" in df_s1.columns

    for row in df_s1.itertuples():
        s1_id = row.entity_id
        s1_country = row.country
        s1_tokens = row.name_tokens
        s1_numbers = row.numeric_tokens

        hit_counts: Dict[int, int] = defaultdict(int)

        # 1. Query name tokens (+3)
        for t in s1_tokens:
            if t in token_idx:
                for idx in token_idx[t]:
                    hit_counts[idx] += 3

        # 2. Query phonetic skeletons (+3)
        if has_phonetic:
            for ph in row.phonetic_tokens:
                if ph in phonetic_idx:
                    for idx in phonetic_idx[ph]:
                        hit_counts[idx] += 3

        # 3. Query numeric tokens (+2)
        for num in s1_numbers:
            if num in num_idx:
                for idx in num_idx[num]:
                    hit_counts[idx] += 2

        # 4. Query n-grams (+1)
        if len(hit_counts) < max_candidates_per_s1 and s1_tokens:
            first_word = s1_tokens[0]
            if len(first_word) >= 3 and first_word not in COMMON_LEGAL_WORDS:
                for i in range(len(first_word) - 2):
                    ng = first_word[i : i + 3]
                    if ng in ngram_idx:
                        for idx in ngram_idx[ng]:
                            hit_counts[idx] += 1

        if not hit_counts:
            candidates_dict[s1_id] = []
            continue

        valid_candidates = []
        for idx, score in hit_counts.items():
            cand_country = pool_countries[idx]
            if s1_country and cand_country and s1_country != cand_country:
                continue
            valid_candidates.append((score, pool_ids[idx]))

        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [cand_id for _, cand_id in valid_candidates[:max_candidates_per_s1]]
        candidates_dict[s1_id] = top_candidates

    return candidates_dict


def evaluate_blocking_recall(
    candidates_dict: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Evaluates candidate coverage against ground truth matches.
    """
    total_true_links = 0
    captured_links = 0
    candidate_lengths = []

    for s1_id, candidates in candidates_dict.items():
        candidate_lengths.append(len(candidates))
        true_set = ground_truth.get(s1_id, set())
        if not true_set:
            continue
        total_true_links += len(true_set)
        captured_links += len(set(candidates) & true_set)

    recall = captured_links / total_true_links if total_true_links > 0 else 1.0
    avg_cands = float(np.mean(candidate_lengths)) if candidate_lengths else 0.0

    return {
        "candidate_recall": recall,
        "total_true_links": total_true_links,
        "captured_links": captured_links,
        "avg_candidates_per_s1": avg_cands,
    }
