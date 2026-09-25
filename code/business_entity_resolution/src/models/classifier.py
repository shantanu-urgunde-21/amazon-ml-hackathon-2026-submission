"""
Model module for Business Entity Resolution.
Trains LightGBM classifier with leak-free GroupKFold cross-validation on S1 entity IDs.
"""

from typing import Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    params: Dict = None,
    num_boost_round: int = 300,
) -> lgb.Booster:
    """Trains a single LightGBM booster model."""
    lgb_params = params or DEFAULT_LGBM_PARAMS.copy()

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_data]
    valid_names = ["train"]

    callbacks = []
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

    booster = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster


def cross_validate_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    pair_keys: List[Tuple[str, str]],
    n_splits: int = 5,
    params: Dict = None,
) -> Tuple[np.ndarray, List[lgb.Booster]]:
    """
    Performs GroupKFold cross-validation grouped strictly by s1_id.
    Returns:
        oof_probs: np.ndarray of shape (len(y),) with predicted P(match)
        models: List of trained fold boosters for inference ensembling
    """
    s1_groups = np.array([k[0] for k in pair_keys])
    gkf = GroupKFold(n_splits=n_splits)

    oof_probs = np.zeros(len(y), dtype=np.float32)
    models: List[lgb.Booster] = []

    print(f"Starting {n_splits}-fold GroupKFold cross-validation across {len(set(s1_groups))} S1 entities...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=s1_groups), 1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        model = train_lightgbm(X_tr, y_tr, X_va, y_val, params=params)
        val_preds = model.predict(X_va, num_iteration=model.best_iteration)
        oof_probs[val_idx] = val_preds
        models.append(model)

        print(f"  Fold {fold}/{n_splits} complete. Val pairs: {len(val_idx)}")

    return oof_probs, models
