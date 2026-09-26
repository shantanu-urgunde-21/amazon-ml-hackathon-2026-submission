"""Pairwise features for (S1 record, candidate) pairs, computed column-wise.

All string similarities use rapidfuzz.process.cpdist, which scores aligned
pairs in parallel C++ (no Python loop per pair), so millions of pairs take
seconds. Inputs are the normalized frames (see normalization/normalizer.py)
and a candidate table with columns s1_idx, pool_idx plus the retrieval
columns produced by filters/blocking.py.

Feature groups
  name_*     name similarity (core, compact, alias, phonetic, legal form)
  addr_*     address similarity and structured slots (+1 agree / 0 missing / -1 conflict)
  ret_*      retrieval signals: dense cosines, forward/reverse ranks and margins
  grp_*      candidate-relative context within the S1 entity's candidate set
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

RETRIEVAL_FEATURES = [
    "ret_cos", "ret_cos_name", "ret_cos_addr",
    "ret_fwd_rank", "ret_rev_rank", "ret_rev_margin", "ret_rev_best",
]

FEATURE_NAMES = [
    "name_ratio", "name_token_set", "name_token_sort", "name_partial", "name_jaro_winkler",
    "name_exact", "name_compact_ratio", "name_compact_partial", "name_alias_best",
    "name_phonetic_set", "name_legal_state", "name_len_s1", "name_len_cand", "name_first_token_eq",
    "name_ocr_ratio", "name_ratio_no_addr",
    "addr_token_set", "addr_ratio", "addr_partial", "addr_cand_empty", "addr_s1_empty", "addr_both_present",
    "addr_state_state", "addr_house_state", "addr_nums_set", "addr_nums_overlap",
    "cand_is_s2", "cand_has_indic",
    *RETRIEVAL_FEATURES,
    "grp_size", "grp_rank_cos", "grp_margin_cos", "grp_rank_name", "grp_margin_name",
]


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw) / (
        1.0 if scorer is JaroWinkler.normalized_similarity else 100.0
    )


def _ternary(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both = (a != "") & (b != "")
    return np.where(both, np.where(a == b, 1.0, -1.0), 0.0).astype(np.float32)


def _first_token(s: pd.Series) -> np.ndarray:
    return s.str.split(" ", n=1).str[0].fillna("").to_numpy()


def build_features(s1: pd.DataFrame, pool: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """cands: s1_idx, pool_idx (row positions in s1 / pool) + RETRIEVAL_FEATURES."""
    A = s1.iloc[cands["s1_idx"].to_numpy()].reset_index(drop=True)
    B = pool.iloc[cands["pool_idx"].to_numpy()].reset_index(drop=True)
    f = {}

    n1, n2 = A["name_core"].tolist(), B["name_core"].tolist()
    f["name_ratio"] = _cp(n1, n2, fuzz.ratio)
    f["name_token_set"] = _cp(n1, n2, fuzz.token_set_ratio)
    f["name_token_sort"] = _cp(n1, n2, fuzz.token_sort_ratio)
    f["name_partial"] = _cp(n1, n2, fuzz.partial_ratio)
    f["name_jaro_winkler"] = _cp(n1, n2, JaroWinkler.normalized_similarity)
    f["name_exact"] = (A["name_core"].to_numpy() == B["name_core"].to_numpy()).astype(np.float32)
    c1, c2 = A["name_compact"].tolist(), B["name_compact"].tolist()
    f["name_compact_ratio"] = _cp(c1, c2, fuzz.ratio)
    f["name_compact_partial"] = _cp(c1, c2, fuzz.partial_ratio)
    alias = B["name_alias"].to_numpy()
    alias_sim = _cp(n1, alias.tolist(), fuzz.token_set_ratio)
    f["name_alias_best"] = np.where(alias != "", np.maximum(alias_sim, f["name_token_set"]), f["name_token_set"])
    f["name_phonetic_set"] = _cp(A["name_phon"].tolist(), B["name_phon"].tolist(), fuzz.token_set_ratio)
    f["name_legal_state"] = _ternary(A["legal"].to_numpy(), B["legal"].to_numpy())
    f["name_len_s1"] = A["name_core"].str.count(" ").to_numpy(np.float32) + 1
    f["name_len_cand"] = B["name_core"].str.count(" ").to_numpy(np.float32) + 1
    f["name_first_token_eq"] = (_first_token(A["name_core"]) == _first_token(B["name_core"])).astype(np.float32)
    tr_ocr = str.maketrans({"1": "l", "0": "o", "v": "u"})
    n1_ocr = [t.translate(tr_ocr).replace("rn", "m") for t in n1]
    n2_ocr = [t.translate(tr_ocr).replace("rn", "m") for t in n2]
    f["name_ocr_ratio"] = _cp(n1_ocr, n2_ocr, fuzz.ratio)

    a1, a2 = A["addr_norm"].tolist(), B["addr_norm"].tolist()
    f["addr_cand_empty"] = (B["addr_norm"].to_numpy() == "").astype(np.float32)
    f["addr_s1_empty"] = (A["addr_norm"].to_numpy() == "").astype(np.float32)
    f["addr_both_present"] = ((f["addr_cand_empty"] == 0) & (f["addr_s1_empty"] == 0)).astype(np.float32)
    f["name_ratio_no_addr"] = np.where((f["addr_cand_empty"] == 1) | (f["addr_s1_empty"] == 1), f["name_ratio"], 0.0).astype(np.float32)
    f["addr_token_set"] = _cp(a1, a2, fuzz.token_set_ratio)
    f["addr_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["addr_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    f["addr_state_state"] = _ternary(A["state"].to_numpy(), B["state"].to_numpy())
    f["addr_house_state"] = _ternary(A["house_num"].to_numpy(), B["house_num"].to_numpy())
    f["addr_nums_set"] = _cp(A["addr_nums"].tolist(), B["addr_nums"].tolist(), fuzz.token_set_ratio)

    nums1_list = A["addr_nums"].tolist()
    nums2_list = B["addr_nums"].tolist()
    nums_overlap = np.zeros(len(nums1_list), dtype=np.float32)
    for i, (x, y) in enumerate(zip(nums1_list, nums2_list)):
        if x and y:
            sx = set(x.split())
            sy = set(y.split())
            nums_overlap[i] = 1.0 if (sx & sy) else -1.0
    f["addr_nums_overlap"] = nums_overlap

    f["cand_is_s2"] = B["entity_id"].str.startswith("S2-").to_numpy(np.float32)
    f["cand_has_indic"] = B["has_indic"].to_numpy(np.float32)
    for col in RETRIEVAL_FEATURES:
        f[col] = cands[col].to_numpy(np.float32)

    X = pd.DataFrame(f)
    g = cands["s1_idx"].to_numpy()
    grp = X.assign(_g=g).groupby("_g")
    X["grp_size"] = grp["ret_cos"].transform("size").to_numpy(np.float32)
    X["grp_rank_cos"] = grp["ret_cos"].rank(ascending=False, method="min").to_numpy(np.float32)
    X["grp_margin_cos"] = (X["ret_cos"] - grp["ret_cos"].transform("max")).to_numpy(np.float32)
    X["grp_rank_name"] = grp["name_token_set"].rank(ascending=False, method="min").to_numpy(np.float32)
    X["grp_margin_name"] = (X["name_token_set"] - grp["name_token_set"].transform("max")).to_numpy(np.float32)
    return X[FEATURE_NAMES]
