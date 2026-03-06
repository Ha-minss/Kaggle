from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


DEFAULT_DROP_COLS = ["y", "duration", "age", "default", "day", "pdays"]


@dataclass(frozen=True)
class FeatureSpec:
    drop_cols: List[str] = None
    make_never_contacted: bool = True
    pdays_col: str = "pdays"

    def resolved_drop_cols(self) -> List[str]:
        return list(DEFAULT_DROP_COLS if self.drop_cols is None else self.drop_cols)


def build_features(
    df: pd.DataFrame,
    spec: FeatureSpec = FeatureSpec(),
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (X, cat_cols).
    - Drops columns in spec
    - Adds never_contacted = 1(pdays == -1) if pdays exists
    """
    drop_cols = [c for c in spec.resolved_drop_cols() if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    # feature: never_contacted
    if spec.make_never_contacted and spec.pdays_col in df.columns:
        X["never_contacted"] = (df[spec.pdays_col] == -1).astype(int)
    elif spec.make_never_contacted and spec.pdays_col not in df.columns:
        X["never_contacted"] = 0

    # cat columns
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    return X, cat_cols
