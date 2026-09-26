"""
Dataloader module for Business Entity Resolution.
Provides fast streaming TSV reading for large multi-gigabyte sources and ground truth.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from src.normalization import learn_lexicon, normalize_records
from src.normalization.normalizer import NORMALIZER_VERSION
from src.normalization.text import INDIC_RE


def load_source(
    path: Path,
    nrows: Optional[int] = None,
    sample_n: Optional[int] = None,
    random_state: int = 42,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Loads a source TSV file safely with standard columns:
    [entity_id, business_name, business_address, country]
    If sample_n is specified, randomly samples sample_n records with random_state across the entire file.
    Otherwise falls back to nrows sequential loading if nrows is specified.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Source file not found at: {path}")

    if sample_n is not None:
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
        )
        if len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            nrows=nrows,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
        )

    expected_cols = ["entity_id", "business_name", "business_address", "country"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    return df


def load_ground_truth(
    path: Path,
    nrows: Optional[int] = None,
    target_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Fast streaming parser for ground truth TSV: source1_entity_id -> set of matched entity_ids.
    Empty matched_entity_ids column produces an empty set (singleton).
    If target_s1_ids is provided, only loads ground truth for those specific S1 entities.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Ground truth file not found at: {path}")

    gt_dict: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        count = 0
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()

            if target_s1_ids is not None and s1_id not in target_s1_ids:
                continue

            matched = parts[1].strip() if len(parts) > 1 else ""
            if not matched:
                gt_dict[s1_id] = set()
            else:
                gt_dict[s1_id] = {x.strip() for x in matched.split(",") if x.strip()}
            count += 1
            if nrows is not None and count >= nrows:
                break
            if target_s1_ids is not None and len(gt_dict) >= len(target_s1_ids):
                break
    return gt_dict

def ground_truth_links(gt: Dict[str, Set[str]], s1_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Ground truth as one row per true (source1_entity_id, entity_id) link."""
    keys = gt.keys() if s1_ids is None else s1_ids
    rows = [(s1, m) for s1 in keys for m in gt.get(s1, ())]
    return pd.DataFrame(rows, columns=["source1_entity_id", "entity_id"])


def split_s1_ids(s1_ids: List[str], holdout_frac: float, seed: int):
    """Deterministic split of S1 entities into (fit, holdout). Everything learned
    from labels (lexicon, model, thresholds) uses `fit` only; `holdout` is the
    untouched benchmark."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(s1_ids))
    mask = rng.random(len(ids)) < holdout_frac
    return ids[~mask].tolist(), ids[mask].tolist()


def build_lexicon(s1: pd.DataFrame, pool: pd.DataFrame, gt: Dict[str, Set[str]], s1_ids: Iterable[str]) -> dict:
    """Learns the Indic->English lexicon from the given S1 entities' true links."""
    links = ground_truth_links(gt, s1_ids)
    s1_i = s1.set_index("entity_id")
    pool_i = pool.set_index("entity_id")
    other = pool_i.loc[links["entity_id"]]
    indic = (other["business_name"].str.contains(INDIC_RE.pattern)
             | other["business_address"].str.contains(INDIC_RE.pattern)).to_numpy()
    links = links[indic]
    pairs = pd.DataFrame({
        "name1": s1_i.loc[links["source1_entity_id"], "business_name"].to_numpy(),
        "addr1": s1_i.loc[links["source1_entity_id"], "business_address"].to_numpy(),
        "name2": pool_i.loc[links["entity_id"], "business_name"].to_numpy(),
        "addr2": pool_i.loc[links["entity_id"], "business_address"].to_numpy(),
    })
    return learn_lexicon(pairs)


def lexicon_fingerprint(lexicon: dict) -> str:
    """Cache key of normalized data: lexicon content + normalizer version."""
    key = json.dumps([NORMALIZER_VERSION, lexicon], sort_keys=True, ensure_ascii=False)
    return hashlib.md5(key.encode()).hexdigest()[:10]


def normalized_paths(path: Path, lexicon: dict, cache_dir: Path) -> Dict[str, Path]:
    """country -> parquet file of that source's normalized rows ({} if not built yet)."""
    marker = Path(cache_dir) / f"{Path(path).stem}.{lexicon_fingerprint(lexicon)}.countries.json"
    if not marker.exists():
        return {}
    return {c: marker.with_name(f"{Path(path).stem}.{lexicon_fingerprint(lexicon)}.{c}.parquet")
            for c in json.loads(marker.read_text())}


def build_normalized(path: Path, lexicon: dict, cache_dir: Path, chunk_rows: int = 1_000_000) -> Dict[str, Path]:
    """Raw TSV -> normalized parquet, one file per country (so later stages can
    load a single country without reading the others). Raw text is dropped."""
    done = normalized_paths(path, lexicon, cache_dir)
    if done:
        return done
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_rows):
        parts.append(normalize_records(chunk, lexicon).drop(columns=["business_name", "business_address"]))
    df = pd.concat(parts, ignore_index=True)
    del parts
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(path).stem}.{lexicon_fingerprint(lexicon)}"
    countries = sorted(df["country"].unique())
    for c in countries:
        df[df["country"] == c].reset_index(drop=True).to_parquet(cache_dir / f"{stem}.{c}.parquet", index=False)
    (cache_dir / f"{stem}.countries.json").write_text(json.dumps(countries))
    return normalized_paths(path, lexicon, cache_dir)
