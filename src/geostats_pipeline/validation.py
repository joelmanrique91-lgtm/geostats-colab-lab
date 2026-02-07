from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .kriging import SearchParameters, ordinary_kriging


@dataclass
class CVResult:
    data: pd.DataFrame
    metrics: Dict[str, float]


def spatial_kfold_indices(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    n_splits: int = 5,
    random_state: int = 13,
) -> np.ndarray:
    coords = df[[xcol, ycol]].to_numpy()
    n_splits = max(2, min(n_splits, len(df)))
    labels = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10).fit_predict(coords)
    return labels


def kriging_cross_validation(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str,
    vcol: str,
    model: Dict[str, float],
    search: SearchParameters,
    method: str = "loo",
    n_splits: int = 5,
    random_state: int = 13,
) -> CVResult:
    method = method.lower()
    if method not in {"loo", "kfold_spatial", "kfold"}:
        raise ValueError("method must be 'loo' or 'kfold_spatial'")

    results: List[pd.DataFrame] = []
    if method == "loo":
        for idx in df.index:
            train = df.drop(index=idx)
            test = df.loc[[idx]]
            pred = ordinary_kriging(
                train,
                xcol,
                ycol,
                zcol,
                vcol,
                (float(test[xcol].iloc[0]), float(test[ycol].iloc[0]), float(test[zcol].iloc[0])),
                model,
                search,
            )
            out = test.copy()
            out["estimate"] = pred["estimate"]
            out["variance"] = pred["variance"]
            out["fold"] = int(idx)
            results.append(out)
    else:
        labels = spatial_kfold_indices(df, xcol, ycol, n_splits=n_splits, random_state=random_state)
        for fold_id in range(labels.max() + 1):
            train = df.loc[labels != fold_id]
            test = df.loc[labels == fold_id]
            rows = []
            for idx, row in test.iterrows():
                pred = ordinary_kriging(
                    train,
                    xcol,
                    ycol,
                    zcol,
                    vcol,
                    (float(row[xcol]), float(row[ycol]), float(row[zcol])),
                    model,
                    search,
                )
                rows.append({**row.to_dict(), "estimate": pred["estimate"], "variance": pred["variance"], "fold": fold_id})
            results.append(pd.DataFrame(rows))

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


def plot_swath_comparison(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    actual_col: str,
    estimate_col: str,
    domain_col: str,
    n_bins: int = 10,
) -> plt.Figure:
    df_axes = add_principal_axes(df, xcol, ycol)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, coord in zip(axes.flatten(), [xcol, ycol, "axis_major", "axis_minor"]):
        actual = _swath_summary(df_axes, coord, actual_col, domain_col, n_bins)
        estimate = _swath_summary(df_axes, coord, estimate_col, domain_col, n_bins)
        for domain in sorted(df_axes[domain_col].dropna().unique().tolist()):
            act_dom = actual[actual[domain_col] == domain]
            est_dom = estimate[estimate[domain_col] == domain]
            ax.plot(act_dom["center"], act_dom[actual_col], marker="o", label=f"{domain} datos")
            ax.plot(est_dom["center"], est_dom[estimate_col], marker="x", linestyle="--", label=f"{domain} est")
        ax.set_xlabel(coord)
        ax.set_ylabel(actual_col)
        ax.set_title(f"Swath {coord}")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig
