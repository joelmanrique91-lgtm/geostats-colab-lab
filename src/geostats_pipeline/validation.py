from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.kriging import ordinary_kriging_2d
from src.utils_spatial import validate_columns


@dataclass
class CVResult:
    data: pd.DataFrame
    metrics: Dict[str, float]


def _prepare_grid(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """Create a grid DataFrame with columns 'x' and 'y'."""
    return df[[xcol, ycol]].rename(columns={xcol: "x", ycol: "y"}).copy()


def spatial_kfold_indices(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    n_splits: int = 5,
    random_state: int = 13,
) -> np.ndarray:
    """Assign spatial folds using KMeans clustering on coordinates."""
    validate_columns(df, [xcol, ycol])
    coords = df[[xcol, ycol]].to_numpy()
    n_splits = max(2, min(n_splits, len(df)))
    labels = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10).fit_predict(coords)
    return labels


def kriging_cross_validation(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    vcol: str,
    vario: Dict[str, float],
    params: Dict[str, float],
    method: str = "loo",
    n_splits: int = 5,
    random_state: int = 13,
) -> CVResult:
    """Cross-validation using ordinary kriging (LOO or spatial K-fold)."""
    validate_columns(df, [xcol, ycol, vcol])
    method = method.lower()
    if method not in {"loo", "kfold"}:
        raise ValueError("method must be 'loo' or 'kfold'")

    results: List[pd.DataFrame] = []

    if method == "loo":
        for idx in df.index:
            train = df.drop(index=idx)
            test = df.loc[[idx]]
            grid_df = _prepare_grid(test, xcol, ycol)
            preds = ordinary_kriging_2d(train, xcol, ycol, vcol, grid_df, vario, params)
            out = test.copy()
            out["estimate"] = preds["estimate"].to_numpy()
            out["variance"] = preds["variance"].to_numpy()
            out["fold"] = int(idx)
            results.append(out)
    else:
        labels = spatial_kfold_indices(df, xcol, ycol, n_splits=n_splits, random_state=random_state)
        for fold_id in range(labels.max() + 1):
            train = df.loc[labels != fold_id]
            test = df.loc[labels == fold_id]
            grid_df = _prepare_grid(test, xcol, ycol)
            preds = ordinary_kriging_2d(train, xcol, ycol, vcol, grid_df, vario, params)
            out = test.copy()
            out["estimate"] = preds["estimate"].to_numpy()
            out["variance"] = preds["variance"].to_numpy()
            out["fold"] = int(fold_id)
            results.append(out)

    cv_df = pd.concat(results, axis=0).sort_index()
    cv_df["error"] = cv_df["estimate"] - cv_df[vcol]
    metrics = compute_cv_metrics(cv_df, vcol=vcol)
    return CVResult(data=cv_df, metrics=metrics)


def compute_cv_metrics(
    df: pd.DataFrame,
    vcol: str,
    pred_col: str = "estimate",
    var_col: str = "variance",
) -> Dict[str, float]:
    """Compute validation metrics (ME, RMSE, MSE, slope/intercept, MSDR)."""
    errors = df[pred_col] - df[vcol]
    mse = float(np.mean(errors**2))
    metrics = {
        "ME": float(np.mean(errors)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }

    if len(df) >= 2:
        slope, intercept = np.polyfit(df[pred_col], df[vcol], 1)
    else:
        slope, intercept = float("nan"), float("nan")
    metrics["slope"] = float(slope)
    metrics["intercept"] = float(intercept)

    if var_col in df.columns:
        variance = df[var_col].to_numpy()
        mask = np.isfinite(variance) & (variance > 0)
        if np.any(mask):
            msdr = float(np.mean((errors.to_numpy()[mask] ** 2) / variance[mask]))
            metrics["MSDR"] = msdr
    return metrics


def _swath_summary(
    df: pd.DataFrame,
    coord_col: str,
    vcol: str,
    domain_col: str,
    n_bins: int,
) -> pd.DataFrame:
    bins = pd.cut(df[coord_col], n_bins, include_lowest=True)
    grouped = (
        df.assign(_bin=bins)
        .groupby([domain_col, "_bin"], observed=True)[vcol]
        .mean()
        .reset_index()
    )
    grouped["center"] = grouped["_bin"].apply(lambda b: b.mid)
    return grouped


def plot_swath_by_domain(
    df: pd.DataFrame,
    coord_col: str,
    vcol: str,
    domain_col: str,
    n_bins: int = 10,
    title: str | None = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot swath mean values by domain along a coordinate."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    summary = _swath_summary(df, coord_col, vcol, domain_col, n_bins)
    for domain, group in summary.groupby(domain_col):
        ax.plot(group["center"], group[vcol], marker="o", label=str(domain))

    ax.set_xlabel(coord_col)
    ax.set_ylabel(vcol)
    if title:
        ax.set_title(title)
    ax.legend(title=domain_col, fontsize=8)
    return ax


def add_principal_axes(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """Add PCA-based principal axis coordinates to the DataFrame."""
    validate_columns(df, [xcol, ycol])
    coords = df[[xcol, ycol]].to_numpy()
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(coords)
    out = df.copy()
    out["axis_major"] = transformed[:, 0]
    out["axis_minor"] = transformed[:, 1]
    return out


def plot_swath_panels(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    vcol: str,
    domain_col: str,
    n_bins: int = 10,
) -> plt.Figure:
    """Create swath plots along X, Y, and PCA axes by domain."""
    df_axes = add_principal_axes(df, xcol, ycol)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_swath_by_domain(df_axes, xcol, vcol, domain_col, n_bins, title=f"Swath {xcol}", ax=axes[0, 0])
    plot_swath_by_domain(df_axes, ycol, vcol, domain_col, n_bins, title=f"Swath {ycol}", ax=axes[0, 1])
    plot_swath_by_domain(
        df_axes, "axis_major", vcol, domain_col, n_bins, title="Swath axis mayor", ax=axes[1, 0]
    )
    plot_swath_by_domain(
        df_axes, "axis_minor", vcol, domain_col, n_bins, title="Swath axis menor", ax=axes[1, 1]
    )

    fig.tight_layout()
    return fig
