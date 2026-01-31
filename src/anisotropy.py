"""Herramientas para análisis de anisotropía."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from variography import experimental_variogram


def sweep_azimuths(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    azimuths_deg: Iterable[float],
    tol_deg: float = 22.5,
    n_lags: int = 20,
    max_range: float | None = None,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Calcula variogramas direccionales para múltiples azimuts.

    Args:
        x, y, z: Coordenadas en metros.
        v: Variable de interés.
        azimuths_deg: Lista de azimuts en grados.
        tol_deg: Tolerancia direccional en grados.
        n_lags: Número de intervalos.
        max_range: Rango máximo en metros (opcional).

    Returns:
        Diccionario con resultados por azimut.
    """
    results: Dict[float, Dict[str, np.ndarray]] = {}
    for azimuth in azimuths_deg:
        lags, gamma, npairs = experimental_variogram(
            x,
            y,
            z,
            v,
            azimuth_deg=float(azimuth),
            tol_deg=tol_deg,
            n_lags=n_lags,
            max_range=max_range,
        )
        results[float(azimuth)] = {
            "lags": lags,
            "gamma": gamma,
            "npairs": npairs,
        }
    return results
