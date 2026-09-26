"""Supervised metric projection on top of unsupervised SVD vectors.

Learns a residual projection f(x) = normalize(x + MLP(x)) that contracts
matching (S1, Pool) pairs in cosine distance while repelling hard negative
candidate distractors.

Leakage protocol:
  * Trained on the `fit` split of the training set only (never holdout).
  * Initialized near zero so it starts as the exact identity mapping f(x) == x.
  * Weights are cached under CACHE / "projectors" / {country}_{view}.pt.
"""

from pathlib import Path
from typing import Optional, Tuple
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MetricResidualProjector(nn.Module):
    """Residual projection f(x) = normalize(x + MLP(x))."""

    def __init__(self, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        # Zero-initialize the output layer so f(x) starts as the exact identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x + self.net(x), p=2, dim=-1)

    def transform(self, vectors: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        """Batch-transform numpy embeddings on GPU if available, CPU otherwise."""
        if len(vectors) == 0:
            return vectors
        device = next(self.parameters()).device
        self.eval()
        out = np.empty_like(vectors, dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(vectors), batch_size):
                chunk = torch.from_numpy(vectors[i:i + batch_size]).float().to(device)
                out[i:i + batch_size] = self(chunk).cpu().numpy()
        return out


def train_metric_projector(anchors: np.ndarray, positives: np.ndarray, negatives: np.ndarray,
                           epochs: int = 4, batch_size: int = 2048, lr: float = 1e-3, tau: float = 0.07,
                           log=print) -> MetricResidualProjector:
    """Train projector using InfoNCE loss with in-batch negatives + explicit hard negatives."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricResidualProjector(dim=anchors.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ds = TensorDataset(
        torch.from_numpy(anchors).float(),
        torch.from_numpy(positives).float(),
        torch.from_numpy(negatives).float(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    log(f"    training MetricResidualProjector on {len(anchors):,} triplets ({epochs} epochs, {device}) ...")
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

            # Cosine similarity matrix between anchors and positives (B x B)
            sim_pos = torch.mm(za, zp.t()) / tau
            # Cosine similarity with hard negative (B x 1)
            sim_hard_neg = torch.sum(za * zn, dim=-1, keepdim=True) / tau

            diag_pos = torch.diag(sim_pos).unsqueeze(1)
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

        log(f"      epoch {ep + 1}/{epochs} loss: {total_loss / max(1, n_batches):.4f} ({time.time() - t0:.1f}s)")

    model.eval()
    return model


def save_projector(model: MetricResidualProjector, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_projector(path: Path, dim: int = 128) -> MetricResidualProjector:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricResidualProjector(dim=dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
