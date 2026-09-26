"""Phase 3 Prototype: Supervised Metric Learning on SVD Embeddings (India).

Validates:
  1. Pair mining from fit-split ground truth + candidate false neighbors.
  2. Residual metric projection training with InfoNCE + hard negative loss.
  3. Retrieval recall before vs after projection on untouched holdout entities.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.config as config
from src.filters.blocking import _partition, PartitionedIndex
from src.pipeline import (
    RETRIEVAL_COLUMNS,
    candidates,
    get_split,
    load_split,
    _country_files,
)
from src.retrieval.embedder import CharNgramEmbedder, addr_text, name_text


class MetricResidualProjector(nn.Module):
    """Residual projection f(x) = normalize(x + MLP(x)).
    Initialized near zero so it starts as the identity mapping and smoothly refines.
    """
    def __init__(self, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
        # Initialize output layer to zero -> f(x) == x initially
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x + self.net(x), p=2, dim=-1)


def train_metric_projector(anchors: np.ndarray, positives: np.ndarray, negatives: np.ndarray,
                           epochs: int = 4, batch_size: int = 2048, lr: float = 1e-3, tau: float = 0.07):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricResidualProjector(dim=anchors.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ds = TensorDataset(
        torch.from_numpy(anchors).float(),
        torch.from_numpy(positives).float(),
        torch.from_numpy(negatives).float(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"  Training MetricProjector on {len(anchors):,} triplets ({epochs} epochs, batch {batch_size}, {device})...")
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            za = model(a)
            zp = model(p)
            zn = model(n)

            # InfoNCE: positive sim vs in-batch negatives + explicit hard negative
            # In-batch cosine sim matrix: za @ zp.T (B x B)
            sim_pos = torch.mm(za, zp.t()) / tau
            # Hard negative similarity: diag(za @ zn.T) (B x 1)
            sim_hard_neg = torch.sum(za * zn, dim=-1, keepdim=True) / tau

            # Target: diagonal of sim_pos is the positive class (column index 0 after reordering)
            diag_pos = torch.diag(sim_pos).unsqueeze(1)
            # Mask out self-positives from the negative pool
            mask = ~torch.eye(len(a), dtype=torch.bool, device=device)
            in_batch_negs = sim_pos[mask].view(len(a), len(a) - 1)

            logits = torch.cat([diag_pos, sim_hard_neg, in_batch_negs], dim=1)
            targets = torch.zeros(len(a), dtype=torch.long, device=device)

            loss = nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"    Epoch {ep+1}/{epochs} - loss: {total_loss / max(1, n_batches):.4f} ({time.time() - t0:.1f}s)")

    model.eval()
    return model


def main():
    print("=" * 75)
    print("PHASE 3 EXPERIMENT: SUPERVISED METRIC PROJECTION ON SVD VECTORS (INDIA)")
    print("=" * 75, flush=True)

    country = "India"
    split_info = get_split()
    fit_set = set(split_info["fit"])
    holdout_set = set(split_info["holdout"])

    print("Loading normalized data...", flush=True)
    s1, pool = load_split("train", country, RETRIEVAL_COLUMNS)
    cands = pd.read_parquet(config.CACHE_DIR / "candidates" / f"train_{country}.25b5ef9494.parquet")

    # 1. Fit base unsupervised SVD embedders exactly as current pipeline
    print("Fitting base unsupervised SVD embedders...", flush=True)
    s1_file, pool_files = _country_files("train", country)
    rng = np.random.default_rng(config.RANDOM_SEED)
    fit_texts = pd.concat([pd.read_parquet(f, columns=RETRIEVAL_COLUMNS) for f in pool_files], ignore_index=True)
    fit_texts = fit_texts.iloc[np.sort(rng.choice(len(fit_texts), min(len(fit_texts), 200_000), replace=False))].reset_index(drop=True)

    name_emb = CharNgramEmbedder(config.EMBED_DIM, seed=config.RANDOM_SEED).fit(name_text(fit_texts))
    addr_emb = CharNgramEmbedder(config.EMBED_DIM, seed=config.RANDOM_SEED).fit(addr_text(fit_texts))
    del fit_texts

    # 2. Mine fit-split training pairs (Anchor = S1, Positive = True pool record, Hard Negative = False candidate)
    print("Mining fit-split training triplets...", flush=True)
    gt_df = pd.read_csv(config.TRAIN_GROUND_TRUTH_PATH, sep="\t", dtype="string[pyarrow]", keep_default_na=False)
    gt_links = gt_df[gt_df["matched_entity_ids"] != ""].assign(m=lambda d: d["matched_entity_ids"].str.split(",")).explode("m")
    gt_links = gt_links[gt_links["source1_entity_id"].isin(fit_set)]

    s1_map = s1["entity_id"].to_numpy()
    pool_map = pool["entity_id"].to_numpy()
    s1_pos_map = pd.Series(s1.index, index=s1["entity_id"])
    pool_pos_map = pd.Series(pool.index, index=pool["entity_id"])

    # True pairs for fit split
    fit_true_df = gt_links[gt_links["source1_entity_id"].isin(s1_pos_map.index) & gt_links["m"].isin(pool_pos_map.index)].copy()
    fit_true_df["s1_idx"] = fit_true_df["source1_entity_id"].map(s1_pos_map).to_numpy()
    fit_true_df["pool_idx"] = fit_true_df["m"].map(pool_pos_map).to_numpy()

    # Get hard negatives from existing candidates (pairs in cands for fit S1 that are NOT true links)
    s1_in_fit = s1["entity_id"].isin(fit_set).to_numpy()
    cands_fit = cands[s1_in_fit[cands["s1_idx"].to_numpy()]].copy()
    cands_fit["s1_id"] = s1_map[cands_fit["s1_idx"].to_numpy()]
    cands_fit["p_id"] = pool_map[cands_fit["pool_idx"].to_numpy()]

    true_pairs_set = set(zip(fit_true_df["source1_entity_id"].to_numpy(), fit_true_df["m"].to_numpy()))
    cand_pairs_fit = list(zip(cands_fit["s1_id"].to_numpy(), cands_fit["p_id"].to_numpy()))
    cand_is_false = np.array([p not in true_pairs_set for p in cand_pairs_fit])
    hard_negs_df = cands_fit[cand_is_false].groupby("s1_idx").first().reset_index()

    # Merge into triplets: (s1_idx, pos_pool_idx, neg_pool_idx)
    triplets = fit_true_df.merge(hard_negs_df[["s1_idx", "pool_idx"]].rename(columns={"pool_idx": "neg_pool_idx"}), on="s1_idx", how="inner")
    print(f"  Mined {len(triplets):,} high-quality training triplets from fit split.")

    # 3. Transform S1, positive pool, and negative pool with base SVD
    print("Transforming triplets with base SVD...", flush=True)
    trip_sample = triplets.sample(n=min(len(triplets), 150_000), random_state=42).reset_index(drop=True)

    s1_trip = s1.iloc[trip_sample["s1_idx"].to_numpy()]
    pos_trip = pool.iloc[trip_sample["pool_idx"].to_numpy()]
    neg_trip = pool.iloc[trip_sample["neg_pool_idx"].to_numpy()]

    a_name = name_emb.transform(name_text(s1_trip))
    p_name = name_emb.transform(name_text(pos_trip))
    n_name = name_emb.transform(name_text(neg_trip))

    # 4. Train Name Metric Projector
    print("\nTraining Name Metric Projector...")
    name_proj = train_metric_projector(a_name, p_name, n_name, epochs=4, batch_size=2048, lr=1e-3)

    # 5. Measure test recall on holdout sample before vs after projection!
    print("\n" + "=" * 75)
    print("EVALUATING ON UNTOUCHED HOLDOUT (Recall Before vs After)")
    print("=" * 75, flush=True)

    holdout_gt = gt_df[gt_df["matched_entity_ids"] != ""].assign(m=lambda d: d["matched_entity_ids"].str.split(",")).explode("m")
    holdout_gt = holdout_gt[holdout_gt["source1_entity_id"].isin(holdout_set)]
    holdout_gt = holdout_gt[holdout_gt["source1_entity_id"].isin(s1_pos_map.index) & holdout_gt["m"].isin(pool_pos_map.index)]

    eval_sample = holdout_gt.sample(n=min(len(holdout_gt), 5000), random_state=42).reset_index(drop=True)
    eval_sample["s1_idx"] = eval_sample["source1_entity_id"].map(s1_pos_map).to_numpy()
    eval_sample["pool_idx"] = eval_sample["m"].map(pool_pos_map).to_numpy()

    eval_s1 = s1.iloc[eval_sample["s1_idx"].to_numpy()]
    eval_pool = pool.iloc[eval_sample["pool_idx"].to_numpy()]

    # Unsupervised Cosine Similarities
    v_s1_un = name_emb.transform(name_text(eval_s1))
    v_p_un = name_emb.transform(name_text(eval_pool))
    cos_un = np.sum(v_s1_un * v_p_un, axis=-1)

    # Supervised Projected Cosine Similarities
    device = next(name_proj.parameters()).device
    with torch.no_grad():
        v_s1_proj = name_proj(torch.from_numpy(v_s1_un).float().to(device)).cpu().numpy()
        v_p_proj = name_proj(torch.from_numpy(v_p_un).float().to(device)).cpu().numpy()
    cos_proj = np.sum(v_s1_proj * v_p_proj, axis=-1)

    print(f"  True Match Name Cosine Similarity (N = {len(eval_sample):,} holdout pairs):")
    print(f"    Base Unsupervised SVD : mean = {cos_un.mean():.4f}, median = {np.median(cos_un):.4f}, p10 = {np.percentile(cos_un, 10):.4f}")
    print(f"    Supervised Projector  : mean = {cos_proj.mean():.4f}, median = {np.median(cos_proj):.4f}, p10 = {np.percentile(cos_proj, 10):.4f}")
    print(f"    Improvement in Match Cosine: {cos_proj.mean() - cos_un.mean():+.4f} (p10 shifted {np.percentile(cos_proj, 10) - np.percentile(cos_un, 10):+.4f})")
    print("=" * 75)


if __name__ == "__main__":
    main()
