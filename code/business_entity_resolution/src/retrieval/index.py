"""FAISS helpers: exact inner-product indexes on GPU when available, CPU otherwise.

Candidate generation uses *reverse assignment*: every S2/S3 record matches at
most one S1 entity, so instead of asking "which pool records are near this S1
entity?" (a fixed top-K per entity, mostly padding), we index the S1 records
and ask each pool record "which S1 entities are you nearest to?". An S1
entity's candidates are the pool records that put it near the top, so the
candidate set adapts per entity (see filters/blocking.py).

GPU: with the `faiss-gpu-cu12` package (requirements-gpu.txt) and
config `useGpu = true`, flat indexes live on the GPU. They are exact, so the
results are identical to the CPU `faiss-cpu` package, only faster.
"""

from typing import Any

import faiss
import numpy as np

import src.config as config

try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available() and bool(config.USE_GPU)
except Exception:
    _TORCH_CUDA = False

_GPU_RESOURCES = None


class TorchGpuIndexFlatIP:
    """Exact inner-product index implemented on PyTorch CUDA.

    Drop-in replacement for faiss.IndexFlatIP / faiss.GpuIndexFlatIP.
    Provides identical exact inner-product ranking while leveraging the GPU.
    Uses dynamic query batching to stay well within 4 GB VRAM.
    """

    def __init__(self, d: int, device: str = "cuda"):
        self.d = d
        self.device = device
        self.vectors = None
        self.ntotal = 0

    def add(self, x: np.ndarray):
        if len(x) == 0:
            return
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device, non_blocking=True)
        if self.vectors is None:
            self.vectors = t
        else:
            self.vectors = torch.cat([self.vectors, t], dim=0)
        self.ntotal = len(self.vectors)

    def search(self, queries: np.ndarray, k: int):
        n_queries = len(queries)
        if self.ntotal == 0 or n_queries == 0:
            return np.full((n_queries, k), -np.inf, dtype=np.float32), np.full((n_queries, k), -1, dtype=np.int64)
        actual_k = min(k, self.ntotal)
        chunk_q = min(8192, max(256, int(200_000_000 / max(1, self.ntotal * 4))))
        q = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to(self.device, non_blocking=True)
        all_d, all_i = [], []
        with torch.no_grad():
            for i in range(0, n_queries, chunk_q):
                qb = q[i : i + chunk_q]
                sims = torch.mm(qb, self.vectors.t())
                vals, idxs = torch.topk(sims, actual_k, dim=1)
                all_d.append(vals.cpu().numpy())
                all_i.append(idxs.cpu().numpy())
        D = np.vstack(all_d)
        I = np.vstack(all_i)
        if actual_k < k:
            pad = k - actual_k
            D = np.hstack([D, np.full((n_queries, pad), -np.inf, dtype=np.float32)])
            I = np.hstack([I, np.full((n_queries, pad), -1, dtype=np.int64)])
        return D, I


def gpu_available() -> bool:
    if not bool(config.USE_GPU):
        return False
    if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        return True
    return _TORCH_CUDA


def flat_index(d: int) -> Any:
    """Exact inner-product index (cosine for L2-normalized vectors).

    Uses native FAISS GPU if available, PyTorch CUDA if available, or CPU FAISS fallback.
    """
    global _GPU_RESOURCES
    if bool(config.USE_GPU):
        if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
            if _GPU_RESOURCES is None:
                _GPU_RESOURCES = faiss.StandardGpuResources()
                _GPU_RESOURCES.setTempMemory(256 * 1024 * 1024)
            return faiss.GpuIndexFlatIP(_GPU_RESOURCES, d)
        if _TORCH_CUDA:
            return TorchGpuIndexFlatIP(d, device="cuda")
    return faiss.IndexFlatIP(d)

