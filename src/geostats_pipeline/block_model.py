from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlockDiscretization:
    """Parámetros de discretización de bloques."""

    nx: int = 1
    ny: int = 1
    nz: int = 1

    def validate(self) -> None:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("nx, ny, nz deben ser positivos")


def _subcell_centers(length: float, n: int) -> np.ndarray:
    """Centros de subceldas en una dimensión."""
    if n <= 0:
        raise ValueError("n debe ser positivo")
    step = length / n
    return (np.arange(n) + 0.5) * step - 0.5 * length


def discretize_block(
    center: Iterable[float],
    size: Tuple[float, float, float],
    discretization: BlockDiscretization,
) -> np.ndarray:
    """Discretiza un bloque en subceldas y devuelve centros en 3D.

    Args:
        center: Centro del bloque (x, y, z).
        size: Dimensiones del bloque (dx, dy, dz).
        discretization: Subdivisiones por eje.

    Returns:
        Array (nsub, 3) con coordenadas de los centros subcelda.
    """
    discretization.validate()
    cx, cy, cz = (float(val) for val in center)
    dx, dy, dz = (float(val) for val in size)

    xs = _subcell_centers(dx, discretization.nx) + cx
    ys = _subcell_centers(dy, discretization.ny) + cy
    zs = _subcell_centers(dz, discretization.nz) + cz

    grid = np.array(np.meshgrid(xs, ys, zs, indexing="xy"))
    grid = grid.reshape(3, -1).T
    return grid


def discretize_blocks(
    grid_df: pd.DataFrame,
    size: Tuple[float, float, float],
    discretization: BlockDiscretization,
    id_col: str = "block_id",
) -> pd.DataFrame:
    """Discretiza múltiples bloques de un grid en subceldas.

    Args:
        grid_df: DataFrame con centros de bloques (x, y, z).
        size: Dimensiones del bloque (dx, dy, dz).
        discretization: Subdivisiones por eje.
        id_col: Nombre para el identificador del bloque.

    Returns:
        DataFrame con subceldas y columnas x, y, z, block_id.
    """
    discretization.validate()
    subcells = []
    for idx, row in grid_df.iterrows():
        centers = discretize_block((row["x"], row["y"], row["z"]), size, discretization)
        df = pd.DataFrame(centers, columns=["x", "y", "z"])
        df[id_col] = idx
        subcells.append(df)
    if not subcells:
        return pd.DataFrame(columns=["x", "y", "z", id_col])
    return pd.concat(subcells, ignore_index=True)


def block_covariance(
    block_points: np.ndarray,
    covariance_fn,
) -> float:
    """Covarianza promedio bloque-bloque usando una función de covarianza.

    Args:
        block_points: Array (nsub, 3) de subceldas.
        covariance_fn: Callable que acepta (p1, p2) y retorna covarianza.

    Returns:
        Covarianza promedio del bloque.
    """
    if block_points.size == 0:
        return float("nan")
    nsub = block_points.shape[0]
    covs = []
    for i in range(nsub):
        for j in range(nsub):
            covs.append(covariance_fn(block_points[i], block_points[j]))
    return float(np.mean(covs)) if covs else float("nan")
