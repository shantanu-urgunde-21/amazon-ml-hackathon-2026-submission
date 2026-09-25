"""
Splitters module for leak-free cross-validation.
"""

from typing import Generator, List, Tuple
import numpy as np
from sklearn.model_selection import KFold


def get_s1_kfold_splits(
    s1_ids: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields (train_idx, val_idx) arrays split strictly by S1 entity.
    Guarantees no S1 entity leakage between train and validation folds.
    """
    unique_ids = np.array(s1_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(unique_ids):
        yield train_idx, val_idx
