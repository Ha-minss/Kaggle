from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve


def roc_pr_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc_ap": float(average_precision_score(y_true, proba)),
        "positive_rate": float(np.mean(y_true)),
    }


def top_k_summary(proba: np.ndarray, y_true: np.ndarray, pct_list: Iterable[float]) -> List[Dict[str, float]]:
    proba = np.asarray(proba)
    y_true = np.asarray(y_true).astype(int)

    n = len(proba)
    total_pos = int(y_true.sum())
    rows: List[Dict[str, float]] = []

    for pct in pct_list:
        k = int(np.ceil(n * float(pct)))
        idx = np.argsort(-proba)[:k]
        tp = int(y_true[idx].sum())
        fp = int(k - tp)

        precision = tp / k if k > 0 else float("nan")
        recall = tp / total_pos if total_pos > 0 else float("nan")

        rows.append({
            "top_pct": float(pct),
            "k": int(k),
            "tp": int(tp),
            "fp": int(fp),
            "precision": float(precision),
            "recall": float(recall),
        })
    return rows


def profit_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float, revenue: float, cost: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)

    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    profit = tp * revenue - (tp + fp) * cost
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    call_rate = float(pred.mean())

    return {
        "threshold": float(thr),
        "tp": tp,
        "fp": fp,
        "calls": int(tp + fp),
        "profit": float(profit),
        "precision": float(precision),
        "recall": float(recall),
        "call_rate": call_rate,
    }


def best_threshold_by_profit(
    y_true: np.ndarray,
    proba: np.ndarray,
    revenue: float,
    cost: float,
    grid: int = 1001
) -> Dict[str, float]:
    """Search threshold on a uniform grid in [0,1]."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)

    thresholds = np.linspace(0.0, 1.0, int(grid))
    best = None
    best_profit = -1e18

    for thr in thresholds:
        stat = profit_at_threshold(y_true, proba, float(thr), revenue=revenue, cost=cost)
        if stat["profit"] > best_profit:
            best_profit = stat["profit"]
            best = stat

    assert best is not None
    return best


def profit_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    revenue: float,
    cost: float,
    grid: int = 400
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)

    thresholds = np.linspace(0.0, 1.0, int(grid))
    profits = np.zeros_like(thresholds)
    call_rates = np.zeros_like(thresholds)

    for i, thr in enumerate(thresholds):
        stat = profit_at_threshold(y_true, proba, float(thr), revenue=revenue, cost=cost)
        profits[i] = stat["profit"]
        call_rates[i] = stat["call_rate"]

    return thresholds, profits, call_rates

def profit_at_top_pct(
    y_true: np.ndarray,
    proba: np.ndarray,
    top_pct: float,
    revenue: float,
    cost: float,
) -> Dict[str, float]:
    """Budget-style targeting: call top X% by predicted probability.

    Profit = TP * revenue - K * cost  (since you call exactly K people)
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)

    n = len(proba)
    k = int(np.ceil(n * float(top_pct)))
    if k <= 0:
        return {
            "top_pct": float(top_pct),
            "k": 0,
            "tp": 0,
            "fp": 0,
            "profit": 0.0,
            "precision": float("nan"),
            "recall": float("nan"),
        }

    idx = np.argsort(-proba)[:k]
    tp = int(y_true[idx].sum())
    fp = int(k - tp)
    total_pos = int(y_true.sum())

    profit = tp * revenue - k * cost
    precision = tp / k if k > 0 else float("nan")
    recall = tp / total_pos if total_pos > 0 else float("nan")

    return {
        "top_pct": float(top_pct),
        "k": int(k),
        "tp": int(tp),
        "fp": int(fp),
        "profit": float(profit),
        "precision": float(precision),
        "recall": float(recall),
    }
