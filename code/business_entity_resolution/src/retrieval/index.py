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

import src.config as config

_GPU_RESOURCES = None


def gpu_available() -> bool:
    return bool(config.USE_GPU) and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0


def flat_index(d: int) -> Any:
    """Exact inner-product index (cosine for L2-normalized vectors).

    Typed as Any on purpose: FAISS replaces `add(x)` / `search(x, k)` with
    numpy-friendly wrappers at import time, but its generated type stubs still
    describe the C++ signatures `add(n, x)` / `search(n, x, k, D, I)`, which
    type checkers would flag at every call site.
    """
    global _GPU_RESOURCES
    if gpu_available():
        if _GPU_RESOURCES is None:
            _GPU_RESOURCES = faiss.StandardGpuResources()
            _GPU_RESOURCES.setTempMemory(256 * 1024 * 1024)  # small GPUs (4 GB) need headroom for the vectors
        return faiss.GpuIndexFlatIP(_GPU_RESOURCES, d)
    return faiss.IndexFlatIP(d)
