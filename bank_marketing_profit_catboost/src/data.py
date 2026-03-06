from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path, sep: str = ";") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, sep=sep)


def map_target_y(df: pd.DataFrame, target_col: str = "y") -> pd.Series:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")
    return df[target_col].map({"yes": 1, "no": 0}).astype(int)
