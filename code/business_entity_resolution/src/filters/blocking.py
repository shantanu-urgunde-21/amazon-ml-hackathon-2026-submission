"""Candidate generation (blocking) with FAISS similarity indexes (Phase 4).

Per country (matches never cross countries):
  1. fit char n-gram embedders for names and addresses on a sample of the
     country's pool records (retrieval/embedder.py; unsupervised)
  2. index the country's S1 records in exact FAISS inner-product indexes, one
     per state partition and per view (name, address). True matches agree on
     state in > 98.8% of links; WA/DC and AP/Telangana share a partition
     because the data mixes them.
  3. stream the S2/S3 pool through the indexes in batches. Each pool record
     looks up its nearest S1 records by name and by address (reverse
     assignment, see retrieval/index.py). Those are re-scored with
        cos = w * cos(name) + (1 - w) * cos(address)     (name only if an address is empty)
     and kept when they are
        - the best S1 by name alone or by address alone, or
        - the best by combined similarity, or
        - within the top `rev_k` and within `rev_margin` of the best.
     A pool record without a state searches every partition of its country;
     one without an address searches by name only.

Each pool record proposes only a handful of S1 entities, so an S1 entity's
candidate set is small and adapts to how many records point at it. The
similarities, ranks and margins become model features.
"""

import gc
from typing import Dict, List, Set

import psutil
import numpy as np
import pandas as pd

from src.retrieval.embedder import CharNgramEmbedder, addr_text, name_text
from src.retrieval.index import flat_index

# states whose records are routinely written as the other one
PARTITION_ALIASES = {"us_dc": "us_wa", "in_ts": "in_ap"}


def _partition(state: pd.Series) -> np.ndarray:
    return state.replace(PARTITION_ALIASES).to_numpy()


class PartitionedIndex:
    """Exact FAISS flat inner-product index per partition over one vector view of S1 records."""

    def __init__(self, vectors: np.ndarray, parts: np.ndarray):
        self.ids, self.index = {}, {}
        for p in np.unique(parts):
            rows = np.flatnonzero(parts == p)
            idx = flat_index(vectors.shape[1])  # GPU when available (exact either way)
            idx.add(np.ascontiguousarray(vectors[rows]))
            self.ids[p], self.index[p] = rows, idx

    def search(self, queries: np.ndarray, parts: np.ndarray, k: int):
        """Top-k (similarity, S1 row) per query. Queries whose partition is
        unknown ("" or unseen) search every partition and keep the global top-k."""
        D = np.full((len(queries), k), -np.inf, dtype=np.float32)
        I = np.full((len(queries), k), -1, dtype=np.int64)
        known = np.isin(parts, list(self.index))
        for p in np.unique(parts[known]):
            q = np.flatnonzero(parts == p)
            d, i = self.index[p].search(np.ascontiguousarray(queries[q]), k)
            D[q], I[q] = d, np.where(i >= 0, self.ids[p][np.maximum(i, 0)], -1)
        rest = np.flatnonzero(~known)
        if len(rest):
            Q = np.ascontiguousarray(queries[rest])
            for p, idx in self.index.items():
                d, i = idx.search(Q, min(k, idx.ntotal))
                i = np.where(i >= 0, self.ids[p][np.maximum(i, 0)], -1)
                dd = np.hstack([D[rest], d])
                ii = np.hstack([I[rest], i])
                top = np.argsort(-dd, axis=1)[:, :k]
                D[rest] = np.take_along_axis(dd, top, 1)
                I[rest] = np.take_along_axis(ii, top, 1)
        return D, I


def _country_candidates(s1: pd.DataFrame, pool_batches, fit_texts: pd.DataFrame, p: dict, log=print) -> pd.DataFrame:
    """s1: one country's S1 records. pool_batches: iterator of (start_row, frame)
    over that country's pool. fit_texts: a sample of pool records to fit the
    embedders on (unsupervised, no labels)."""
    name_emb = CharNgramEmbedder(p["dim"], seed=p["seed"]).fit(name_text(fit_texts))
    addr_emb = CharNgramEmbedder(p["dim"], seed=p["seed"]).fit(addr_text(fit_texts))
    del fit_texts
    s1_parts = _partition(s1["state"])
    s1_name = name_emb.transform(name_text(s1))
    name_index = PartitionedIndex(s1_name, s1_parts)
    s1_name = s1_name.astype(np.float16)  # the index holds the float32 copy
    s1_addr = addr_emb.transform(addr_text(s1))
    addr_index = PartitionedIndex(s1_addr, s1_parts)
    s1_addr = s1_addr.astype(np.float16)
    s1_has_addr = (s1["addr_norm"] != "").to_numpy()
    gc.collect()
    log(f"    indexed {len(s1):,} S1 records: {len(name_index.index)} state partitions x (name, address)")

    w, k, kk = p["name_weight"], p["rev_k"], p["rev_k"] + 2
    out = []
    for start, chunk in pool_batches:
        c_name = name_emb.transform(name_text(chunk))
        c_addr = addr_emb.transform(addr_text(chunk))
        c_parts = _partition(chunk["state"])
        has_addr = (chunk["addr_norm"] != "").to_numpy()

        # nearest S1 records by name, and by address (records with an address only)
        dn, i_n = name_index.search(c_name, c_parts, kk)
        a_rows = np.flatnonzero(has_addr)
        da, i_a = addr_index.search(c_addr[a_rows], c_parts[a_rows], kk)
        pairs = pd.DataFrame({
            "row": np.concatenate([np.repeat(np.arange(len(chunk)), kk), np.repeat(a_rows, kk)]),
            "s1": np.concatenate([i_n.ravel(), i_a.ravel()]),
            "top_view": np.concatenate([np.tile(np.arange(kk) == 0, len(chunk)), np.tile(np.arange(kk) == 0, len(a_rows))]),
        })
        pairs = pairs[pairs["s1"] >= 0].groupby(["row", "s1"], as_index=False)["top_view"].max()
        r, s = pairs["row"].to_numpy(), pairs["s1"].to_numpy()

        cos_n = np.einsum("ij,ij->i", s1_name[s].astype(np.float32), c_name[r])
        cos_a = np.einsum("ij,ij->i", s1_addr[s].astype(np.float32), c_addr[r])
        both = has_addr[r] & s1_has_addr[s]
        cos = np.where(both, w * cos_n + (1 - w) * cos_a, cos_n).astype(np.float32)
        pairs["cos"] = cos
        pairs["rank"] = pairs.groupby("row")["cos"].rank(ascending=False, method="first") - 1
        best = pairs.groupby("row")["cos"].transform("max").to_numpy()
        rank = pairs["rank"].to_numpy()
        no_addr = ~has_addr[r]  # name-only records are ambiguous: keep a wider set
        keep = (pairs["top_view"].to_numpy()                       # best by name or by address alone
                | (rank == 0)                                       # best combined
                | ((rank < k) & (best - cos <= p["rev_margin"]))
                | (no_addr & (rank < kk) & (best - cos <= 2 * p["rev_margin"])))
        kp = keep
        out.append(pd.DataFrame({
            "s1_idx": s1.index.to_numpy()[s[kp]],
            "pool_idx": start + r[kp],
            "ret_cos": cos[kp],
            "ret_cos_name": cos_n[kp].astype(np.float32),
            "ret_cos_addr": np.where(both, cos_a, 0.0)[kp].astype(np.float32),
            "ret_rev_rank": pairs["rank"].to_numpy(np.float32)[kp],
            "ret_rev_margin": (cos - best)[kp].astype(np.float32),
            "ret_rev_best": best[kp].astype(np.float32),
        }))
        log(f"    pool {start + len(chunk):,} (rss {psutil.Process().memory_info().rss / 1e9:.1f} GB)")
        del c_name, c_addr, dn, i_n, da, i_a, pairs
    del name_index, addr_index, s1_name, s1_addr
    gc.collect()
    return pd.concat(out, ignore_index=True)


def generate_candidates(s1: pd.DataFrame, pool_batches, fit_texts: pd.DataFrame, params: dict, log=print) -> pd.DataFrame:
    """Candidate table for ONE country: s1_idx (row in s1), pool_idx (row in the
    country's pool, S2 then S3) + retrieval features. Callers loop over the
    country labels found in the data, so the country set stays open."""
    s1 = s1.reset_index(drop=True)
    cands = _country_candidates(s1, pool_batches, fit_texts, params, log)
    cands["ret_fwd_rank"] = (
        cands.groupby("s1_idx")["ret_cos"].rank(ascending=False, method="first").astype(np.float32)
    )
    return cands[cands["ret_fwd_rank"] <= params["max_candidates"]].reset_index(drop=True)


def evaluate_blocking_recall(cands: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame,
                             ground_truth: Dict[str, Set[str]], s1_ids: List[str] = None) -> Dict[str, float]:
    """Recall of true links, candidate set size and reduction ratio."""
    s1_ids = s1_ids if s1_ids is not None else s1["entity_id"].tolist()
    ids = set(s1_ids)
    pairs = set(zip(s1["entity_id"].to_numpy()[cands["s1_idx"]], pool["entity_id"].to_numpy()[cands["pool_idx"]]))
    pairs = {p for p in pairs if p[0] in ids}
    total = sum(len(ground_truth.get(s, ())) for s in s1_ids)
    captured = sum(1 for s, m in pairs if m in ground_truth.get(s, ()))
    sizes = pd.Series([s for s, _ in pairs]).value_counts().reindex(s1_ids, fill_value=0)
    return {
        "candidate_recall": captured / total if total else 1.0,
        "captured_links": captured,
        "total_true_links": total,
        "avg_candidates_per_s1": float(sizes.mean()),
        "p95_candidates_per_s1": float(sizes.quantile(0.95)),
        "reduction_ratio": 1.0 - len(pairs) / max(1, len(s1_ids) * len(pool)),
    }
