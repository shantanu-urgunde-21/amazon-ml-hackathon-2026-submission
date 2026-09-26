"""
Model module for Business Entity Resolution.
XGBoost binary classifier (Apache-2.0) with leak-free GroupKFold cross-validation
on S1 entity ids. Trains on the GPU (CUDA) when one is available and config
`useGpu = true`; otherwise on the CPU with the same `hist` algorithm.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import src.config as config

DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "max_bin": 256,
    "seed": 42,
}


def device() -> str:
    if not config.USE_GPU:
        return "cpu"
    try:
        xgb.train({"device": "cuda", "tree_method": "hist"}, xgb.DMatrix(np.zeros((2, 1)), label=[0, 1]), 1)
        return "cuda"
    except xgb.core.XGBoostError:
        return "cpu"


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    params: Dict = None,
    num_boost_round: int = 3000,
) -> xgb.Booster:
    """Trains one booster, early-stopped on the validation fold when given."""
    p = {**DEFAULT_XGB_PARAMS, "device": device(), **(params or {})}
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]
    kw = {}
    if X_val is not None:
        evals.append((xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain), "val"))
        kw["early_stopping_rounds"] = 50
    return xgb.train(p, dtrain, num_boost_round=num_boost_round, evals=evals, verbose_eval=False, **kw)


def cross_validate_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    params: Dict = None,
) -> Tuple[np.ndarray, List[xgb.Booster]]:
    """
    GroupKFold cross-validation grouped strictly by S1 entity (`groups`), so all
    candidates of one entity are in the same fold.
    Returns:
        oof_probs: out-of-fold P(match) for every row
        models: the fold boosters, averaged at inference time
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    models: List[xgb.Booster] = []

    print(f"Starting {n_splits}-fold GroupKFold cross-validation across {len(np.unique(groups)):,} S1 entities "
          f"(XGBoost on {device()})...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        model = train_model(X.iloc[train_idx], y[train_idx], X.iloc[val_idx], y[val_idx], params=params)
        oof_probs[val_idx] = _predict_one(model, X.iloc[val_idx])
        models.append(model)
        print(f"  Fold {fold}/{n_splits}: {len(val_idx):,} pairs, best iteration {model.best_iteration}")
    return oof_probs, models


def _predict_one(model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    # features are built on the CPU; predicting there avoids a GPU copy (same result)
    model.set_param({"device": "cpu"})
    return model.inplace_predict(X, iteration_range=(0, model.best_iteration + 1))


def predict(models: List[xgb.Booster], X: pd.DataFrame) -> np.ndarray:
    """Average probability of the fold models."""
    return np.mean([_predict_one(m, X) for m in models], axis=0).astype(np.float32)


def feature_importance(models: List[xgb.Booster], names: List[str]) -> pd.Series:
    """Mean total gain per feature over the fold models."""
    gains = [pd.Series(m.get_score(importance_type="total_gain")).reindex(names).fillna(0.0) for m in models]
    return pd.concat(gains, axis=1).mean(axis=1)
