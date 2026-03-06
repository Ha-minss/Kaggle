from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.utils import set_seed, ensure_dir
from src.data import load_csv, map_target_y
from src.features import build_features, FeatureSpec
from src.model import CatBoostParams, train_oof_catboost, train_full_catboost, predict_proba
from src.metrics import roc_pr_metrics, best_threshold_by_profit, profit_curve, profit_at_threshold, top_k_summary, profit_at_top_pct
from src.plots import plot_profit_curve, plot_pr_curve, plot_roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bank marketing: CatBoost + profit-optimized threshold.")
    p.add_argument("--train_csv", type=str, required=True, help="Path to train.csv")
    p.add_argument("--test_csv", type=str, default=None, help="Path to test.csv (optional)")
    p.add_argument("--sep", type=str, default=";", help="CSV separator (default: ';')")
    p.add_argument("--outdir", type=str, default="artifacts", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cv_splits", type=int, default=5, help="StratifiedKFold splits")
    p.add_argument("--iterations", type=int, default=830)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--revenue", type=float, default=20.0, help="Revenue per TP")
    p.add_argument("--cost", type=float, default=1.0, help="Cost per call")
    p.add_argument("--use_isotonic", action="store_true", help="Fit isotonic on OOF and calibrate probabilities")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    # -----------------------
    # 1) load data
    # -----------------------
    train_df = load_csv(args.train_csv, sep=args.sep)
    y_train = map_target_y(train_df, target_col="y")

    # features (match notebook logic)
    X_train, cat_cols = build_features(train_df, spec=FeatureSpec())
    params = CatBoostParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=args.seed,
        verbose=0,
    )

    # -----------------------
    # 2) OOF CV
    # -----------------------
    oof_proba, _ = train_oof_catboost(
        X_train, y_train, cat_cols=cat_cols, params=params, n_splits=args.cv_splits
    )

    oof_metrics = roc_pr_metrics(y_train.values, oof_proba)

# Budget-style targeting summaries (Top 5/10/20%) computed on OOF predictions
pct_list = [0.05, 0.10, 0.20]
oof_topk = top_k_summary(oof_proba, y_train.values, pct_list=pct_list)
oof_profit_topk = [profit_at_top_pct(y_train.values, oof_proba, p, args.revenue, args.cost) for p in pct_list]

    # Save OOF predictions
    oof_out = train_df.copy()
    oof_out["oof_proba"] = oof_proba
    oof_out.to_csv(outdir / "oof_predictions.csv", index=False)

    # -----------------------
    # 3) Profit threshold on OOF
    # -----------------------
    best_raw = best_threshold_by_profit(
        y_train.values, oof_proba, revenue=args.revenue, cost=args.cost, grid=1001
    )

    # profit curve plot
    thr_grid, profits, _ = profit_curve(y_train.values, oof_proba, revenue=args.revenue, cost=args.cost, grid=400)
    plot_profit_curve(thr_grid, profits, best_raw["threshold"], outdir / "profit_curve_oof.png")
    plot_pr_curve(y_train.values, oof_proba, outdir / "precision_recall_oof.png", title="Precision-Recall Curve (OOF)")

    # Optional isotonic calibration (fit on OOF preds)
    iso_report = None
    best_iso = None
    if args.use_isotonic:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_proba, y_train.values)
        oof_proba_iso = iso.transform(oof_proba)

        iso_report = {
            "brier_raw": float(brier_score_loss(y_train.values, oof_proba)),
            "brier_iso": float(brier_score_loss(y_train.values, oof_proba_iso)),
            "oof_roc_auc_raw": float(oof_metrics["roc_auc"]),
            "oof_roc_auc_iso": float(roc_pr_metrics(y_train.values, oof_proba_iso)["roc_auc"]),
            "oof_pr_auc_raw": float(oof_metrics["pr_auc_ap"]),
            "oof_pr_auc_iso": float(roc_pr_metrics(y_train.values, oof_proba_iso)["pr_auc_ap"]),
        }
        best_iso = best_threshold_by_profit(
            y_train.values, oof_proba_iso, revenue=args.revenue, cost=args.cost, grid=1001
        )

    # -----------------------
    # 4) Train full model
    # -----------------------
    full_model = train_full_catboost(X_train, y_train, cat_cols=cat_cols, params=params)

    # -----------------------
    # 5) Optional test evaluation + call list
    # -----------------------
    test_report = None
    if args.test_csv:
        test_df = load_csv(args.test_csv, sep=args.sep)

        # test features must match train columns
        X_test, _ = build_features(test_df, spec=FeatureSpec())
        X_test = X_test.reindex(columns=X_train.columns)

        proba_test = predict_proba(full_model, X_test, cat_cols=cat_cols)

        # if test has y, evaluate
        if "y" in test_df.columns:
            y_test = map_target_y(test_df, target_col="y").values
            test_metrics = roc_pr_metrics(y_test, proba_test)

            # apply best threshold (raw) learned from OOF
            test_profit = profit_at_threshold(y_test, proba_test, best_raw["threshold"], args.revenue, args.cost)
            test_topk = top_k_summary(proba_test, y_test, pct_list=[0.05, 0.10, 0.20])
            test_profit_topk = [profit_at_top_pct(y_test, proba_test, p, args.revenue, args.cost) for p in [0.05, 0.10, 0.20]]

            test_report = {
                "metrics": test_metrics,
                "profit_at_best_oof_threshold": test_profit,
                "topk": test_topk,
                "profit_topk": test_profit_topk,
            }

            plot_roc_curve(y_test, proba_test, outdir / "roc_test.png", title="ROC Curve (Test)")
            plot_pr_curve(y_test, proba_test, outdir / "pr_test.png", title="Precision-Recall Curve (Test)")

        # export call list using best_raw threshold
        call_flag = (proba_test >= best_raw["threshold"]).astype(int)
        call_list = test_df.copy()
        call_list["pred_proba"] = proba_test
        call_list["call_flag"] = call_flag
        call_list.to_csv(outdir / "call_target_list.csv", index=False)

# Also export fixed-budget call lists (top 5/10/20% by probability)
for p in [0.05, 0.10, 0.20]:
    k = int(np.ceil(len(proba_test) * p))
    top_idx = np.argsort(-proba_test)[:k]
    top_df = test_df.iloc[top_idx].copy()
    top_df["pred_proba"] = proba_test[top_idx]
    top_df.to_csv(outdir / f"call_target_list_top{int(p*100)}pct.csv", index=False)

    # -----------------------
    # 6) write metrics.json
    # -----------------------
    report = {
        "run_args": vars(args),
        "feature_spec": {
            "drop_cols": ["y", "duration", "age", "default", "day", "pdays"],
            "added_feature": "never_contacted = 1(pdays == -1)",
        },
        "oof": {
            "metrics": oof_metrics,
            "best_threshold_by_profit_raw": best_raw,
            "isotonic_report": iso_report,
            "best_threshold_by_profit_isotonic": best_iso,
        },
        "test": test_report,
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Outputs written to: {outdir.resolve()}")
    print(f"- metrics: {outdir/'metrics.json'}")
    print(f"- oof:     {outdir/'oof_predictions.csv'}")
    if args.test_csv:
        print(f"- call:    {outdir/'call_target_list.csv'}")


if __name__ == "__main__":
    main()
