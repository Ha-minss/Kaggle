from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class CatBoostParams:
    iterations: int = 830
    learning_rate: float = 0.05
    depth: int = 6
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"
    random_seed: int = 42
    verbose: int = 0


def _cat_feature_indices(X: pd.DataFrame, cat_cols: List[str]) -> List[int]:
    return [X.columns.get_loc(c) for c in cat_cols]


def train_oof_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: List[str],
    params: CatBoostParams,
    n_splits: int = 5,
) -> Tuple[np.ndarray, List[CatBoostClassifier]]:
    """Return OOF probabilities and fitted fold models."""
    y_np = np.asarray(y).astype(int)
    oof = np.zeros(len(X), dtype=float)
    models: List[CatBoostClassifier] = []

    cat_idx = _cat_feature_indices(X, cat_cols)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=params.random_seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_np), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        va_pool = Pool(X_va, y_va, cat_features=cat_idx)

        m = CatBoostClassifier(
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            loss_function=params.loss_function,
            eval_metric=params.eval_metric,
            random_seed=params.random_seed,
            verbose=params.verbose,
        )
        m.fit(tr_pool)
        oof[va_idx] = m.predict_proba(va_pool)[:, 1]
        models.append(m)

    return oof, models


def train_full_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: List[str],
    params: CatBoostParams,
) -> CatBoostClassifier:
    cat_idx = _cat_feature_indices(X, cat_cols)
    pool = Pool(X, y, cat_features=cat_idx)

    m = CatBoostClassifier(
        iterations=params.iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        loss_function=params.loss_function,
        eval_metric=params.eval_metric,
        random_seed=params.random_seed,
        verbose=max(params.verbose, 0),
    )
    m.fit(pool)
    return m


def predict_proba(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    cat_cols: List[str],
) -> np.ndarray:
    cat_idx = _cat_feature_indices(X, cat_cols)
    pool = Pool(X, cat_features=cat_idx)
    return model.predict_proba(pool)[:, 1]
