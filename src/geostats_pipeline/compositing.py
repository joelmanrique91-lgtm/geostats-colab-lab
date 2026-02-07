from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


def _resolve_value_columns(df: pd.DataFrame, value_cols: Iterable[str] | None) -> List[str]:
    if value_cols:
        return [col for col in value_cols if col in df.columns]
    if "value" in df.columns:
        return ["value"]
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _composite_group(
    group: pd.DataFrame,
    target_length: float,
    min_length: float,
    bheid_col: str,
    from_col: str,
    to_col: str,
    value_cols: List[str],
) -> Tuple[List[Dict], List[Dict]]:
    composites: List[Dict] = []
    details: List[Dict] = []

    group_sorted = group.sort_values(by=[from_col, to_col])
    comp_id = 0
    comp_buffer: List[Dict] = []
    comp_length = 0.0
    comp_start = None

    def finalize_composite() -> None:
        nonlocal comp_id, comp_buffer, comp_length, comp_start
        if comp_start is None or comp_length <= 0:
            return
        if comp_length < min_length:
            logger.debug("Descartando composite corto: %.3f", comp_length)
            comp_buffer = []
            comp_length = 0.0
            comp_start = None
            return

        comp_id += 1
        comp_end = comp_start + comp_length
        composite_row = {
            bheid_col: group_sorted.iloc[0][bheid_col],
            "comp_id": comp_id,
            from_col: comp_start,
            to_col: comp_end,
            "length": comp_length,
        }
        for col in value_cols:
            weighted_sum = sum(item["length"] * item[col] for item in comp_buffer)
            composite_row[col] = weighted_sum / comp_length
        composites.append(composite_row)

        for item in comp_buffer:
            details.append(
                {
                    bheid_col: composite_row[bheid_col],
                    "comp_id": comp_id,
                    from_col: item[from_col],
                    to_col: item[to_col],
                    "length": item["length"],
                    "weight": item["length"] / comp_length,
                }
            )
        comp_buffer = []
        comp_length = 0.0
        comp_start = None

    for _, row in group_sorted.iterrows():
        interval_start = float(row[from_col])
        interval_end = float(row[to_col])
        interval_length = interval_end - interval_start
        if interval_length <= 0:
            continue
        remaining = interval_length
        current_pos = interval_start

        while remaining > 0:
            if comp_start is None:
                comp_start = current_pos
            available = target_length - comp_length
            take = min(available, remaining)
            record = {
                bheid_col: row[bheid_col],
                from_col: current_pos,
                to_col: current_pos + take,
                "length": take,
            }
            for col in value_cols:
                record[col] = row[col]
            comp_buffer.append(record)
            comp_length += take
            remaining -= take
            current_pos += take
            if comp_length >= target_length - 1e-9:
                finalize_composite()

    finalize_composite()
    return composites, details


def composite_by_length(df: pd.DataFrame, config: Dict, output_dir: str = "outputs") -> pd.DataFrame:
    """Compone intervalos por longitud objetivo por barreno.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas BHID/FROM/TO y variables numéricas.
    config : Dict
        Configuración (sub-sección `compositing` de configs/config.yml).
    output_dir : str
        Carpeta base para outputs.
    """

    comp_cfg = config.get("compositing", {}) if "compositing" in config else config
    bheid_col = comp_cfg.get("bheid_col", "hole_id")
    from_col = comp_cfg.get("from_col", "from")
    to_col = comp_cfg.get("to_col", "to")
    target_length = float(comp_cfg.get("target_length", 1.0))
    min_length = float(comp_cfg.get("min_length", 0.5 * target_length))
    value_cols = _resolve_value_columns(df, comp_cfg.get("value_cols"))
    value_cols = [col for col in value_cols if col not in {from_col, to_col}]

    composites: List[Dict] = []
    details: List[Dict] = []

    for _, group in df.groupby(bheid_col, dropna=False):
        comp_rows, detail_rows = _composite_group(
            group,
            target_length,
            min_length,
            bheid_col,
            from_col,
            to_col,
            value_cols,
        )
        composites.extend(comp_rows)
        details.extend(detail_rows)

    composite_df = pd.DataFrame(composites)
    detail_df = pd.DataFrame(details)

    output_cfg = comp_cfg.get("output", {})
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    composites_path = output_cfg.get("composites", os.path.join(tables_dir, "composites.csv"))
    details_path = output_cfg.get("details", os.path.join(tables_dir, "compositing_details.csv"))
    stats_path = output_cfg.get("stats", os.path.join(tables_dir, "compositing_stats.csv"))

    composite_df.to_csv(composites_path, index=False)
    detail_df.to_csv(details_path, index=False)

    stats = {
        "composites": len(composite_df),
        "mean_length": float(composite_df["length"].mean()) if not composite_df.empty else 0.0,
        "min_length": float(composite_df["length"].min()) if not composite_df.empty else 0.0,
        "max_length": float(composite_df["length"].max()) if not composite_df.empty else 0.0,
        "detail_rows": len(detail_df),
        "mean_weight": float(detail_df["weight"].mean()) if not detail_df.empty else 0.0,
        "min_weight": float(detail_df["weight"].min()) if not detail_df.empty else 0.0,
        "max_weight": float(detail_df["weight"].max()) if not detail_df.empty else 0.0,
    }
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    logger.info("Compositing guardado en %s", composites_path)
    return composite_df
