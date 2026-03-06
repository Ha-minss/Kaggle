# Bank Marketing — Profit-Optimized Targeting (CatBoost)

This project builds a **call-targeting policy** for a bank marketing campaign using the classic bank marketing dataset (`y`: subscription yes/no).

The key idea is to move beyond “model accuracy only” and produce an **actionable decision rule**:
> **Who should we call?**  
> to maximize expected profit under call costs and conversion revenue.

---

## Problem framing (Data Scientist view)

A call campaign has two realities:

### A) Campaign / Budget view (fixed call capacity)
Sometimes you can only call a fixed number of customers (e.g., 5% of the list).
- Policy: **Call Top 5% / 10% / 20%** by predicted probability
- What we measure: **precision/recall** and **profit under fixed budget**
  - `Profit_topK = TP * revenue - K * cost`

### B) Operations view (dynamic call volume)
Sometimes you can call anyone as long as it’s profitable.
- Policy: **Call if p >= t*** (a probability threshold)
- We learn **t*** by maximizing:
  - `Profit = TP * revenue - (TP + FP) * cost`

Both views are supported and reported.

---

## Data & leakage handling

- Target: `y` is mapped to `{yes: 1, no: 0}`
- Leakage: `duration` is dropped (not available before the call)
- Final feature drop set (aligned with the exploration code):
  - `["duration", "age", "default", "day", "pdays"]`
- Added feature:
  - `never_contacted = 1(pdays == -1)`

---

## Modeling

- Model: **CatBoostClassifier** (strong baseline for mixed numeric + categorical data)
- Validation protocol: **OOF (out-of-fold) predictions with StratifiedKFold**
  - We compute OOF probabilities on train
  - We **learn the policy on OOF** (threshold selection, top-k summaries)
  - We then apply the *same policy* to the test set (if labels exist)

Optional:
- Probability calibration via **IsotonicRegression** (helps when interpreting `p` as a probability)

---

## Repository structure

- `scripts/run_experiment.py`  
  Reproducible CLI pipeline:
  - train CatBoost
  - compute OOF probabilities
  - choose profit-max threshold on OOF
  - report Top 5/10/20% summaries
  - (optional) test evaluation + call list export

- `scripts/baseline_logreg.py`  
  Baseline screening with Logistic Regression + permutation importance.

- `notebooks/kaggle_bank_marketing_profit.ipynb`  
  A single, Kaggle-friendly notebook version (no `files.upload()`; auto-finds `train.csv` / `test.csv`).

- `notebooks/original_colab/untitled40.py`  
  Original Colab export (kept for traceability).

---

## Quickstart (GitHub / local)

### 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Data
Place:
```
data/train.csv
data/test.csv   # optional
```
Default CSV separator is `;`.

### 3) Run: full profit-targeting experiment
```bash
python -m scripts.run_experiment \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --sep ";" \
  --revenue 20 \
  --cost 1 \
  --outdir artifacts
```

Optional calibration:
```bash
python -m scripts.run_experiment \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --sep ";" \
  --revenue 20 \
  --cost 1 \
  --use_isotonic \
  --outdir artifacts
```

Baseline screening:
```bash
python -m scripts.baseline_logreg \
  --train_csv data/train.csv \
  --sep ";" \
  --outdir artifacts
```

---

## Outputs

Written under `artifacts/`:

- `metrics.json`  
  Includes:
  - OOF ROC-AUC / PR-AUC (AP)
  - Profit-max threshold on OOF (raw / isotonic)
  - Top 5/10/20% summaries + profit under fixed budget
  - (if labeled test exists) test metrics and test profit at the chosen threshold

- `oof_predictions.csv`  
  Row-level OOF probabilities for auditability.

- `call_target_list.csv`  
  Test rows annotated with `call_flag` (profit-threshold policy).

- `call_target_list_top5pct.csv`, `call_target_list_top10pct.csv`, `call_target_list_top20pct.csv`  
  Fixed-budget call lists (campaign view).

- Plots:
  - `profit_curve_oof.png`
  - `precision_recall_oof.png`
  - `roc_test.png`, `pr_test.png` (only if test has labels)

---

## Kaggle notebook

Use `notebooks/kaggle_bank_marketing_profit.ipynb`:
- It searches `/kaggle/input/**/train.csv` and `/kaggle/input/**/test.csv`.
- It exports:
  - `call_target_list_profit_threshold.csv`
  - `call_target_list_top5pct.csv`, `call_target_list_top10pct.csv`, `call_target_list_top20pct.csv`

---

## Reproducibility notes

- Seed is fixed (`--seed`, default 42).
- Exact numbers may vary slightly across environments if library versions differ.
  If you need strict reproducibility, pin versions in `requirements.txt` more tightly.
