"""Funciones de EDA geoestadístico."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def basic_stats(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    """Calcula estadísticas básicas para una columna numérica.

    Args:
        df: DataFrame de entrada.
        value_col: Columna numérica de interés.

    Returns:
        Diccionario con estadísticas básicas.
    """
    values = df[value_col].dropna().values
    return {
        "count": float(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def plot_grade_histogram(df: pd.DataFrame, value_col: str) -> None:
    """Grafica histograma de la variable de ley.

    Args:
        df: DataFrame de entrada.
        value_col: Columna de ley.
    """
    values = df[value_col].dropna().values
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=20, color='#4c78a8', edgecolor='white')
    plt.xlabel('Ley')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de ley')
    plt.show()


def plot_xy_scatter(df: pd.DataFrame, x_col: str, y_col: str, value_col: str) -> None:
    """Grafica dispersión XY coloreada por ley.

    Args:
        df: DataFrame de entrada.
        x_col: Columna X.
        y_col: Columna Y.
        value_col: Columna de ley.
    """
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        df[x_col],
        df[y_col],
        c=df[value_col],
        cmap='viridis',
        s=40,
        alpha=0.8,
    )
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Dispersión XY coloreada por ley')
    plt.colorbar(scatter, label='Ley')
    plt.axis('equal')
    plt.show()
