from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


def _sniff_dialect(sample: str) -> Tuple[str, str]:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter, dialect.quotechar
    except csv.Error:
        return ",", '"'


def _read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            sample = path.read_text(encoding=enc)[:4096]
            sep, quote = _sniff_dialect(sample)
            return pd.read_csv(path, encoding=enc, sep=sep, quotechar=quote, engine="python")
        except Exception as err:  # pragma: no cover - only when all encodings fail
            last_err = err
    raise RuntimeError(f"Failed to read CSV: {path}") from last_err


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        available = ", ".join(sorted(df.columns))
        raise KeyError(
            "Missing required columns: "
            f"{missing}. Available columns: [{available}]. "
            "Revise config mapping."
        )


def load_data(config: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    data_cfg = config["data"]
    path = Path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")

    df = _read_csv(path)
    metadata = {
        "input_path": str(path),
        "input_hash": file_hash(path),
        "input_shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
    }

    required_cols = [data_cfg["x_col"], data_cfg["y_col"], data_cfg["z_col"], data_cfg["value_col"]]
    validate_columns(df, required_cols)

    domain_col = data_cfg.get("domain_col")
    if domain_col:
        validate_columns(df, [domain_col])

    from_col = data_cfg.get("from_col")
    to_col = data_cfg.get("to_col")
    if from_col or to_col:
        validate_columns(df, [from_col, to_col])

    return df, metadata


def standardize_columns(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    data_cfg = config["data"]
    mapping = {
        data_cfg["x_col"]: "x",
        data_cfg["y_col"]: "y",
        data_cfg["z_col"]: "z",
        data_cfg["value_col"]: "value",
    }
    if data_cfg.get("domain_col"):
        mapping[data_cfg["domain_col"]] = "domain"
    if data_cfg.get("from_col"):
        mapping[data_cfg["from_col"]] = "from"
    if data_cfg.get("to_col"):
        mapping[data_cfg["to_col"]] = "to"
    if data_cfg.get("hole_id_col"):
        mapping[data_cfg["hole_id_col"]] = "hole_id"
    return df.rename(columns=mapping)
