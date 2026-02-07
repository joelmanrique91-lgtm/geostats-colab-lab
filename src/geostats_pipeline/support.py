from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def resolve_support(df: pd.DataFrame, config: Dict[str, object]) -> Tuple[pd.DataFrame, str]:
    data_cfg = config["data"]
    compositing_enabled = bool(config.get("compositing", {}).get("enabled", False))
    support = data_cfg["support"]
    allow_pseudo = bool(data_cfg.get("allow_pseudo_support", False))
    pseudo_len = data_cfg.get("pseudo_support_length")

    has_interval = {"from", "to"}.issubset(df.columns)

    if compositing_enabled and support != "interval":
        raise ValueError("Compositing requires interval support with explicit from/to columns.")

    if support == "interval":
        if has_interval:
            return df, "interval"
        if allow_pseudo:
            if pseudo_len is None:
                raise ValueError("pseudo_support_length must be set when allow_pseudo_support is true.")
            df = df.copy()
            df["from"] = 0.0
            df["to"] = float(pseudo_len)
            logger.warning("Using pseudo-support intervals (length=%.3f).", pseudo_len)
            return df, "pseudo-interval"
        raise ValueError("Support set to 'interval' but from/to columns are missing.")

    if support == "point":
        if has_interval:
            logger.warning("Point-support requested; ignoring from/to columns.")
        return df, "point"

    raise ValueError("data.support must be 'point' or 'interval'.")
