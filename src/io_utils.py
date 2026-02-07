from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def find_csv_files(csv_dir: str) -> List[str]:
    """Return CSV file paths (relative) inside a directory."""
    if not os.path.isdir(csv_dir):
        return []
    files = []
    for name in os.listdir(csv_dir):
        if name.lower().endswith(".csv"):
            files.append(os.path.join(csv_dir, name))
    return sorted(files)


def _sniff_dialect(sample: str) -> Tuple[str, str]:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter, dialect.quotechar
    except csv.Error:
        return ",", '"'


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read CSV with delimiter/encoding detection.

    Tries common encodings and delimiter sniffing.
    """
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                sample = f.read(4096)
            sep, quote = _sniff_dialect(sample)
            return pd.read_csv(path, encoding=enc, sep=sep, quotechar=quote, engine="python")
        except Exception as err:
            last_err = err
    raise RuntimeError(f"Failed to read CSV: {path}") from last_err


def standardize_columns(df: pd.DataFrame, mapping: Dict[str, str | None]) -> pd.DataFrame:
    """Rename columns to standard names: x,y,z,var,domain,lithology,alteration."""
    rename_map: Dict[str, str] = {}
    for key, col in mapping.items():
        if col:
            rename_map[col] = key
    return df.rename(columns=rename_map)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, float]:
    """Convert columns to numeric; return % non-numeric per column."""
    report: Dict[str, float] = {}
    for col in cols:
        if col not in df.columns:
            continue
        before = df[col].shape[0]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        non_numeric = df[col].isna().sum()
        report[col] = (non_numeric / max(before, 1)) * 100.0
    return report
