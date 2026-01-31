"""Utilidades espaciales para EDA y variografía."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Valida que el DataFrame contenga las columnas requeridas.

    Args:
        df: DataFrame de entrada.
        columns: Columnas requeridas.

    Raises:
        ValueError: Si falta alguna columna.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def compute_pairwise_distances(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Calcula distancias euclidianas 3D entre todos los pares.

    Args:
        x, y, z: Coordenadas en metros.

    Returns:
        Matriz de distancias (n x n).
    """
    coords = np.column_stack([x, y, z])
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def lag_bins(max_range: float, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """Genera bins de lag uniformes.

    Args:
        max_range: Rango máximo en metros.
        n_lags: Número de intervalos.

    Returns:
        Tupla (bin_edges, bin_centers).
    """
    edges = np.linspace(0.0, max_range, n_lags + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers
