"""Funciones para cálculo de variogramas experimentales."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from utils_spatial import compute_pairwise_distances, lag_bins


def _directional_mask(
    dx: np.ndarray,
    dy: np.ndarray,
    azimuth_deg: float,
    tol_deg: float,
) -> np.ndarray:
    """Máscara booleana para seleccionar pares según dirección.

    Args:
        dx, dy: Diferencias en X e Y entre pares.
        azimuth_deg: Azimut objetivo en grados.
        tol_deg: Tolerancia en grados.

    Returns:
        Máscara booleana con pares dentro de la tolerancia.
    """
    angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 180.0
    azimuth = azimuth_deg % 180.0
    diff = np.abs(angles - azimuth)
    diff = np.minimum(diff, 180.0 - diff)
    return diff <= tol_deg


def experimental_variogram(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    azimuth_deg: Optional[float] = None,
    tol_deg: float = 22.5,
    n_lags: int = 20,
    max_range: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula el variograma experimental (omni o direccional).

    Args:
        x, y, z: Coordenadas en metros.
        v: Variable de interés (por ejemplo, ley).
        azimuth_deg: Azimut en grados. Si es None, usa omnidireccional.
        tol_deg: Tolerancia direccional en grados.
        n_lags: Número de intervalos.
        max_range: Rango máximo en metros (opcional).

    Returns:
        lags: Centros de lag.
        gamma: Semivarianza por lag.
        npairs: Número de pares por lag.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    v = np.asarray(v, dtype=float)

    n = len(v)
    if n < 2:
        raise ValueError("Se requieren al menos 2 puntos para variografía.")

    # Distancias 3D
    dist = compute_pairwise_distances(x, y, z)

    # Diferencias espaciales en XY para dirección
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]

    # Matriz de semivarianzas
    diff = v[:, None] - v[None, :]
    semivar = 0.5 * diff ** 2

    # Considerar solo pares i<j
    iu = np.triu_indices(n, k=1)
    dist = dist[iu]
    semivar = semivar[iu]
    dx = dx[iu]
    dy = dy[iu]

    if max_range is None:
        max_range = np.nanmax(dist)

    edges, centers = lag_bins(max_range, n_lags)

    if azimuth_deg is not None:
        dir_mask = _directional_mask(dx, dy, azimuth_deg, tol_deg)
        dist = dist[dir_mask]
        semivar = semivar[dir_mask]

    gamma = np.full(n_lags, np.nan)
    npairs = np.zeros(n_lags, dtype=int)

    for i in range(n_lags):
        mask = (dist >= edges[i]) & (dist < edges[i + 1])
        npairs[i] = int(np.sum(mask))
        if npairs[i] > 0:
            gamma[i] = np.mean(semivar[mask])

    return centers, gamma, npairs
