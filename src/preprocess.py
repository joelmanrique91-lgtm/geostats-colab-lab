from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .eda import apply_topcut
from .io_utils import coerce_numeric, read_csv_robust, standardize_columns
from .make_demo_data import make_demo_csv


def load_and_preprocess(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str | None]]:
    """Load CSV and apply standard preprocessing."""
    data_path = cfg["data_csv_path"]
    if not data_path:
        raise ValueError("Missing data_csv_path in config.")
    if not pd.notna(data_path):
        raise ValueError("Invalid data_csv_path in config.")

    if not os.path.exists(data_path):
        make_demo_csv(data_path)

    df_raw = read_csv_robust(data_path)

    cols_cfg = cfg.get("columns", {})
    mapping = {
        "x": cols_cfg.get("x"),
        "y": cols_cfg.get("y"),
        "z": cols_cfg.get("z"),
        "var": cols_cfg.get("variable_objetivo"),
        "domain": cols_cfg.get("domain"),
    }

    df = standardize_columns(df_raw, mapping)

    nodata_values = cfg.get("nodata_values", [])
    if nodata_values:
        df.replace(nodata_values, np.nan, inplace=True)

    coerce_numeric(df, ["x", "y", "z", "var"])
    df.dropna(subset=["x", "y", "var"], inplace=True)

    if cfg.get("topcut", {}).get("enabled") and cfg.get("topcut", {}).get("high"):
        df = apply_topcut(df, "var", float(cfg["topcut"]["high"]))

    return df, df_raw, mapping
