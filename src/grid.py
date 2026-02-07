from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def grid_from_extents(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str,
    dx: float,
    dy: float,
    dz: float,
    pad: float = 0.0,
) -> Dict[str, float]:
    """Create a grid spec from data extents."""
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError("dx, dy, dz must be positive")
    xmin, xmax = df[xcol].min(), df[xcol].max()
    ymin, ymax = df[ycol].min(), df[ycol].max()
    zmin, zmax = df[zcol].min(), df[zcol].max() if zcol in df.columns else (0.0, 0.0)

    xmin -= pad
    ymin -= pad
    zmin -= pad
    xmax += pad
    ymax += pad
    zmax += pad

    nx = int(np.ceil((xmax - xmin) / dx))
    ny = int(np.ceil((ymax - ymin) / dy))
    nz = int(np.ceil((zmax - zmin) / dz)) if zcol in df.columns else 1

    return {
        "nx": max(nx, 1),
        "ny": max(ny, 1),
        "nz": max(nz, 1),
        "xmin": float(xmin),
        "ymin": float(ymin),
        "zmin": float(zmin),
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
    }


def make_grid_dataframe(grid_spec: Dict[str, float]) -> pd.DataFrame:
    """Return DataFrame with block centers for a 2D/3D grid."""
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    nz = int(grid_spec["nz"])
    xmin = float(grid_spec["xmin"])
    ymin = float(grid_spec["ymin"])
    zmin = float(grid_spec["zmin"])
    dx = float(grid_spec["dx"])
    dy = float(grid_spec["dy"])
    dz = float(grid_spec["dz"])

    xs = xmin + dx * (np.arange(nx) + 0.5)
    ys = ymin + dy * (np.arange(ny) + 0.5)
    zs = zmin + dz * (np.arange(nz) + 0.5)

    grid = np.array(np.meshgrid(xs, ys, zs, indexing="xy"))
    grid = grid.reshape(3, -1).T
    return pd.DataFrame(grid, columns=["x", "y", "z"])


def export_grid_to_csv(grid_df: pd.DataFrame, path: str) -> None:
    """Export grid centers to CSV."""
    grid_df.to_csv(path, index=False)
