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

import math

import faiss
import numpy as np

import src.config as config

_GPU_RESOURCES = None


def gpu_available() -> bool:
    return bool(config.USE_GPU) and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0


def flat_index(d: int):
    """Exact inner-product index (cosine for L2-normalized vectors)."""
    global _GPU_RESOURCES
    if gpu_available():
        if _GPU_RESOURCES is None:
            _GPU_RESOURCES = faiss.StandardGpuResources()
            _GPU_RESOURCES.setTempMemory(256 * 1024 * 1024)  # small GPUs (4 GB) need headroom for the vectors
        return faiss.GpuIndexFlatIP(_GPU_RESOURCES, d)
    return faiss.IndexFlatIP(d)


def build_index(vectors: np.ndarray, ivf_threshold: int = 200_000, nprobe: int = 24, seed: int = 42):
    """Approximate IVF index for large CPU-only use (exact flat below `ivf_threshold`)."""
    n, d = vectors.shape
    if n < ivf_threshold or gpu_available():
        index = flat_index(d)
    else:
        nlist = int(4 * math.sqrt(n))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(seed)
        index.train(vectors[rng.choice(n, min(n, nlist * 64), replace=False)])
        index.nprobe = nprobe
    index.add(vectors)
    return index


def search(index, queries: np.ndarray, k: int, chunk: int = 200_000):
    """Top-k (similarities, ids) for every query row; -1 ids are padding."""
    D = np.empty((len(queries), k), dtype=np.float32)
    I = np.empty((len(queries), k), dtype=np.int64)
    for i in range(0, len(queries), chunk):
        D[i:i + chunk], I[i:i + chunk] = index.search(queries[i:i + chunk], k)
    return D, I
