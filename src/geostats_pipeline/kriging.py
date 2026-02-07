from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SearchParameters:
    """Parámetros de vecindad anisotrópica para kriging."""

    ranges: Tuple[float, float, float]
    angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_samples: int = 4
    max_samples: int = 12
    octants: int = 8
    max_per_drillhole: int | None = None
    drillhole_col: str | None = None
    condition_max: float = 1.0e10

    def validate(self) -> None:
        if any(r <= 0 for r in self.ranges):
            raise ValueError("ranges deben ser positivos")
        if self.min_samples <= 0 or self.max_samples <= 0:
            raise ValueError("min_samples y max_samples deben ser positivos")
        if self.min_samples > self.max_samples:
            raise ValueError("min_samples no puede ser mayor que max_samples")
        if self.octants <= 0:
            raise ValueError("octants debe ser positivo")
        if self.max_per_drillhole is not None and self.max_per_drillhole <= 0:
            raise ValueError("max_per_drillhole debe ser positivo")


def _rotation_matrix(azimuth_deg: float, dip_deg: float, rake_deg: float) -> np.ndarray:
    """Matriz de rotación 3D (azimuth, dip, rake en grados)."""
    az = np.deg2rad(azimuth_deg)
    dip = np.deg2rad(dip_deg)
    rake = np.deg2rad(rake_deg)

    rot_z = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(dip), -np.sin(dip)],
            [0.0, np.sin(dip), np.cos(dip)],
        ]
    )
    rot_z2 = np.array(
        [
            [np.cos(rake), -np.sin(rake), 0.0],
            [np.sin(rake), np.cos(rake), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rot_z2 @ rot_x @ rot_z


def _anisotropic_distance(
    coords: np.ndarray,
    target: np.ndarray,
    ranges: Tuple[float, float, float],
    angles: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula distancia anisotrópica y coords rotadas."""
    rot = _rotation_matrix(*angles)
    centered = coords - target
    rotated = centered @ rot.T
    scaled = rotated / np.array(ranges)
    dist = np.sqrt(np.sum(scaled**2, axis=1))
    return dist, rotated


def _octant_index(rotated_coords: np.ndarray) -> np.ndarray:
    """Calcula índice de octante a partir de coords rotadas."""
    signs = rotated_coords >= 0
    idx = signs[:, 0].astype(int) * 4 + signs[:, 1].astype(int) * 2 + signs[:, 2].astype(int)
    return idx


def select_neighbors(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str,
    target: Iterable[float],
    params: SearchParameters,
) -> pd.DataFrame:
    """Selecciona vecinos con vecindad anisotrópica, octantes y límites."""
    params.validate()
    coords = df[[xcol, ycol, zcol]].to_numpy(dtype=float)
    target_arr = np.array(target, dtype=float)
    dist, rotated = _anisotropic_distance(coords, target_arr, params.ranges, params.angles)
    mask = dist <= 1.0

    if not np.any(mask):
        return df.iloc[0:0].copy()

    filtered = df.loc[mask].copy()
    filtered["_dist"] = dist[mask]
    filtered["_octant"] = _octant_index(rotated[mask])

    if params.max_per_drillhole is not None and params.drillhole_col:
        filtered = (
            filtered.sort_values("_dist")
            .groupby(params.drillhole_col, as_index=False, group_keys=False)
            .head(params.max_per_drillhole)
        )

    if params.octants <= 1:
        return filtered.sort_values("_dist").head(params.max_samples)

    per_octant = params.max_samples // params.octants
    remainder = params.max_samples % params.octants

    pieces = []
    for octant in range(params.octants):
        oct_df = filtered[filtered["_octant"] == octant].sort_values("_dist")
        limit = per_octant + (1 if octant < remainder else 0)
        if limit > 0:
            pieces.append(oct_df.head(limit))

    if pieces:
        selected = pd.concat(pieces, ignore_index=True)
    else:
        selected = filtered.iloc[0:0].copy()

    if len(selected) < params.min_samples:
        extra = filtered.sort_values("_dist").drop(index=selected.index, errors="ignore")
        needed = params.min_samples - len(selected)
        if needed > 0:
            selected = pd.concat([selected, extra.head(needed)], ignore_index=True)

    return selected.sort_values("_dist").head(params.max_samples)


def _variogram(h: np.ndarray, model: dict) -> np.ndarray:
    nug = float(model.get("nug", 0.0))
    cc1 = float(model.get("cc1", 0.0))
    cc2 = float(model.get("cc2", 0.0))

    def structure(it: int, cc: float, rng: float, dist: np.ndarray) -> np.ndarray:
        if cc <= 0 or rng <= 0:
            return np.zeros_like(dist)
        hr = dist / rng
        if it == 2:
            return cc * (1.0 - np.exp(-3.0 * hr))
        if it == 3:
            return cc * (1.0 - np.exp(-3.0 * hr**2))
        hr = np.clip(hr, 0.0, 1.0)
        return cc * (1.5 * hr - 0.5 * hr**3)

    it1 = int(model.get("it1", 1))
    it2 = int(model.get("it2", 1))
    hmaj1 = float(model.get("hmaj1", 1.0))
    hmaj2 = float(model.get("hmaj2", 1.0))

    gamma = nug + structure(it1, cc1, hmaj1, h) + structure(it2, cc2, hmaj2, h)
    return gamma


def _covariance(h: np.ndarray, model: dict) -> np.ndarray:
    sill = float(model.get("nug", 0.0)) + float(model.get("cc1", 0.0)) + float(model.get("cc2", 0.0))
    return sill - _variogram(h, model)


def _pairwise_distances(
    coords: np.ndarray,
    ranges: Tuple[float, float, float],
    angles: Tuple[float, float, float],
) -> np.ndarray:
    n = coords.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        h, _ = _anisotropic_distance(coords, coords[i], ranges, angles)
        dist[i, :] = h * ranges[0]
    return dist


def _point_distances(
    coords: np.ndarray,
    target: np.ndarray,
    ranges: Tuple[float, float, float],
    angles: Tuple[float, float, float],
) -> np.ndarray:
    h, _ = _anisotropic_distance(coords, target, ranges, angles)
    return h * ranges[0]


def _validate_matrix(matrix: np.ndarray, condition_max: float) -> Tuple[bool, float]:
    cond = float(np.linalg.cond(matrix))
    return cond <= condition_max, cond


def ordinary_kriging(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str,
    vcol: str,
    target: Iterable[float],
    model: dict,
    params: SearchParameters,
    block_points: np.ndarray | None = None,
) -> dict:
    """Kriging ordinario puntual o de bloque con validaciones."""
    params.validate()
    neighbors = select_neighbors(df, xcol, ycol, zcol, target, params)
    ndata = len(neighbors)

    if ndata < params.min_samples:
        return {
            "estimate": float("nan"),
            "variance": float("nan"),
            "ndata": ndata,
            "condition": float("inf"),
            "valid": False,
            "message": "ndata insuficiente",
        }

    coords = neighbors[[xcol, ycol, zcol]].to_numpy(dtype=float)
    values = neighbors[vcol].to_numpy(dtype=float)
    n = coords.shape[0]

    dist_mat = _pairwise_distances(coords, params.ranges, params.angles)
    cov_mat = _covariance(dist_mat, model)

    matrix = np.zeros((n + 1, n + 1))
    matrix[:n, :n] = cov_mat
    matrix[:n, -1] = 1.0
    matrix[-1, :n] = 1.0

    valid, cond = _validate_matrix(matrix, params.condition_max)
    if not valid:
        return {
            "estimate": float("nan"),
            "variance": float("nan"),
            "ndata": ndata,
            "condition": cond,
            "valid": False,
            "message": "matriz mal condicionada",
        }

    if block_points is None:
        target_arr = np.array(target, dtype=float)
        dist_vec = _point_distances(coords, target_arr, params.ranges, params.angles)
        k_vec = _covariance(dist_vec, model)
        block_cov = float(_covariance(np.array([0.0]), model)[0])
    else:
        block_points = np.asarray(block_points, dtype=float)
        cov_to_block = []
        for point in block_points:
            dist_vec = _point_distances(coords, point, params.ranges, params.angles)
            cov_to_block.append(_covariance(dist_vec, model))
        k_vec = np.mean(np.vstack(cov_to_block), axis=0)

        dist_block = _pairwise_distances(block_points, params.ranges, params.angles)
        block_cov = float(np.mean(_covariance(dist_block, model)))

    rhs = np.zeros(n + 1)
    rhs[:n] = k_vec
    rhs[-1] = 1.0

    weights = np.linalg.solve(matrix, rhs)
    estimate = float(np.dot(weights[:n], values))
    variance = float(block_cov - np.dot(weights[:n], k_vec) - weights[-1])

    return {
        "estimate": estimate,
        "variance": variance,
        "ndata": ndata,
        "condition": cond,
        "valid": True,
        "message": "ok",
    }
