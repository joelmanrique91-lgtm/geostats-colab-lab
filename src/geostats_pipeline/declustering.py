from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


def _resolve_value_columns(df: pd.DataFrame, value_cols: Iterable[str] | None) -> List[str]:
    if value_cols:
        return [col for col in value_cols if col in df.columns]
    if "value" in df.columns:
        return ["value"]
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _weighted_stats(values: pd.Series, weights: pd.Series) -> Dict[str, float]:
    if values.empty:
        return {"weighted_mean": 0.0, "weighted_std": 0.0}
    weights = weights.to_numpy(dtype=float)
    vals = values.to_numpy(dtype=float)
    total = weights.sum()
    if total <= 0:
        return {"weighted_mean": 0.0, "weighted_std": 0.0}
    mean = float(np.sum(weights * vals) / total)
    variance = float(np.sum(weights * (vals - mean) ** 2) / total)
    return {"weighted_mean": mean, "weighted_std": float(np.sqrt(variance))}


def cell_declustering(df: pd.DataFrame, config: Dict, output_dir: str = "outputs") -> pd.DataFrame:
    """Calcula pesos de declustering por celda.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con coordenadas.
    config : Dict
        Configuración (sub-sección `declustering` de configs/config.yml).
    output_dir : str
        Carpeta base para outputs.
    """

    declust_cfg = config.get("declustering", {}) if "declustering" in config else config
    x_col = declust_cfg.get("x_col", "x")
    y_col = declust_cfg.get("y_col", "y")
    z_col = declust_cfg.get("z_col", "z")
    cell_cfg = declust_cfg.get("cell_size", {})
    dx = float(cell_cfg.get("x", 50.0))
    dy = float(cell_cfg.get("y", 50.0))
    dz = float(cell_cfg.get("z", 10.0))
    use_3d = bool(declust_cfg.get("use_3d", False))
    origin_cfg = declust_cfg.get("origin", {})

    origin_x = origin_cfg.get("x", float(df[x_col].min()))
    origin_y = origin_cfg.get("y", float(df[y_col].min()))
    origin_z = origin_cfg.get("z", float(df[z_col].min())) if use_3d and z_col in df.columns else 0.0

    value_cols = _resolve_value_columns(df, declust_cfg.get("value_cols"))
    weight_col = declust_cfg.get("weight_col", "declust_weight")

    coords = df[[x_col, y_col]].copy()
    coords["cell_x"] = np.floor((coords[x_col] - origin_x) / dx).astype(int)
    coords["cell_y"] = np.floor((coords[y_col] - origin_y) / dy).astype(int)
    if use_3d and z_col in df.columns:
        coords["cell_z"] = np.floor((df[z_col] - origin_z) / dz).astype(int)
        cell_keys = ["cell_x", "cell_y", "cell_z"]
    else:
        cell_keys = ["cell_x", "cell_y"]

    cell_counts = coords.groupby(cell_keys).size().rename("cell_count")
    df_with_cells = df.join(coords[cell_keys])
    df_with_cells = df_with_cells.join(cell_counts, on=cell_keys)
    df_with_cells[weight_col] = 1.0 / df_with_cells["cell_count"].astype(float)
    df_with_cells[weight_col] = df_with_cells[weight_col] / df_with_cells[weight_col].mean()

    output_cfg = declust_cfg.get("output", {})
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    weights_path = output_cfg.get("weights", os.path.join(tables_dir, "declustering_weights.csv"))
    stats_path = output_cfg.get("stats", os.path.join(tables_dir, "declustering_stats.csv"))

    df_with_cells.to_csv(weights_path, index=False)

    stats_rows = []
    for col in value_cols:
        stats = {
            "variable": col,
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=1)) if len(df[col]) > 1 else 0.0,
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        stats.update(_weighted_stats(df[col], df_with_cells[weight_col]))
        stats_rows.append(stats)

    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    logger.info("Declustering guardado en %s", weights_path)
    return df_with_cells
