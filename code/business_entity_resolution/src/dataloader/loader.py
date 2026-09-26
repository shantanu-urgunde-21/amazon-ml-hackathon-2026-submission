"""
Dataloader module for Business Entity Resolution.
Provides fast streaming TSV reading for large multi-gigabyte sources and ground truth.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd


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

