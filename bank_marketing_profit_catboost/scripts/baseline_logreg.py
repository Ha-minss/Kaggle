from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.utils import set_seed, ensure_dir
from src.data import load_csv, map_target_y
from src.features import DEFAULT_DROP_COLS


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


signed_log_tf = FunctionTransformer(signed_log1p, feature_names_out="one-to-one")
log1p_tf = FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build_clf(X: pd.DataFrame) -> Pipeline:
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    num_log_signed = [c for c in ["balance"] if c in X.columns]
    num_log_pos = [c for c in ["campaign", "previous"] if c in X.columns]

    num_other = [
        c for c in X.columns
        if (c not in cat_cols + num_log_signed + num_log_pos)
        and (pd.api.types.is_numeric_dtype(X[c]))
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("num_signedlog", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("log", signed_log_tf),
                ("sc", StandardScaler()),
            ]), num_log_signed),

            ("num_log", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("log", log1p_tf),
                ("sc", StandardScaler()),
            ]), num_log_pos),

            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_other),

            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop"
    )

    return Pipeline([
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def eval_auc(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> float:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = build_clf(X)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y_va, proba))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline logreg screening + permutation importance.")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--sep", type=str, default=";")
    p.add_argument("--outdir", type=str, default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    train = load_csv(args.train_csv, sep=args.sep)
    y = map_target_y(train, target_col="y")

    # A) all version: drop leakage duration only + never_contacted + (optional) pdays NA
    X_all = train.drop(columns=["y", "duration"], errors="ignore").copy()
    if "pdays" in X_all.columns:
        X_all["never_contacted"] = (X_all["pdays"] == -1).astype(int)
        X_all.loc[X_all["pdays"] == -1, "pdays"] = pd.NA
    else:
        X_all["never_contacted"] = 0

    # B) drop version: remove age/default/day/pdays but keep never_contacted
    drop_cols = ["age", "default", "day", "pdays"]
    X_drop = X_all.drop(columns=[c for c in drop_cols if c in X_all.columns], errors="ignore").copy()

    auc_all = eval_auc(X_all, y, seed=args.seed)
    auc_drop = eval_auc(X_drop, y, seed=args.seed)

    summary = {
        "auc_all": auc_all,
        "auc_drop": auc_drop,
        "diff": auc_drop - auc_all,
        "dropped": [c for c in drop_cols if c in X_all.columns],
    }

    (outdir / "baseline_logreg_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("[Baseline AUC] all:", auc_all)
    print("[Baseline AUC] drop:", auc_drop, "dropped=", summary["dropped"])
    print("[diff]", summary["diff"])

    # Permutation importance on X_drop (optional but usually helpful)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_drop, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    clf = build_clf(X_drop)
    clf.fit(X_tr, y_tr)

    r = permutation_importance(
        clf, X_va, y_va,
        n_repeats=15, random_state=args.seed, scoring="roc_auc"
    )

    imp = (pd.DataFrame({"feature": X_va.columns, "importance": r.importances_mean})
           .sort_values("importance", ascending=False))
    imp.to_csv(outdir / "baseline_logreg_perm_importance.csv", index=False)

    top = imp.head(15)[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("Permutation Importance (LogReg baseline, X_drop)")
    plt.xlabel("Importance (ROC-AUC drop)")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.savefig(outdir / "baseline_logreg_perm_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("[OK] Wrote baseline artifacts to:", outdir.resolve())


if __name__ == "__main__":
    import json
    main()
