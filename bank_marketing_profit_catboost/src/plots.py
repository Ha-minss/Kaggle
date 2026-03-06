from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

from .utils import ensure_dir


def plot_profit_curve(thresholds: np.ndarray, profits: np.ndarray, best_thr: float, outpath: str | Path) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, profits)
    plt.axvline(best_thr, linestyle="--")
    plt.scatter([best_thr], [profits[np.argmin(np.abs(thresholds - best_thr))]])
    plt.title("Profit Curve (OOF) vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Expected Profit")
    plt.grid(True)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()


def plot_pr_curve(y_true: np.ndarray, proba: np.ndarray, outpath: str | Path, title: str = "Precision-Recall Curve") -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)

    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, proba: np.ndarray, outpath: str | Path, title: str = "ROC Curve") -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)

    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
