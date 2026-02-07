from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from pykrige.ok import OrdinaryKriging


def normal_score_transform(values: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """Transform values to normal scores and return the lookup table."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Input array for normal score transform is empty.")

    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, values.size + 1)
    cdf = (ranks - 0.5) / float(values.size)
    scores = norm.ppf(cdf)

    table = pd.DataFrame({"value": values[order], "score": scores[order]})
    return scores, table


def back_transform(scores: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Back-transform normal scores to original values via interpolation."""
    scores = np.asarray(scores, dtype=float)
    table_sorted = table.sort_values("score")
    return np.interp(scores, table_sorted["score"].to_numpy(), table_sorted["value"].to_numpy())


def _variogram_model_name(vario: Dict[str, float]) -> str:
    model_map = {1: "spherical", 2: "exponential", 3: "gaussian"}
    return model_map.get(int(vario.get("it1", 1)), "spherical")


def _variogram_parameters(vario: Dict[str, float]) -> Dict[str, float]:
    nugget = float(vario.get("nug", 0.0))
    sill = float(vario.get("nug", 0.0)) + float(vario.get("cc1", 1.0))
    range_ = float(vario.get("hmaj1", 1.0))
    return {"sill": sill, "range": range_, "nugget": nugget}


def sgs_block_simulation(
    df: pd.DataFrame,
    grid_df: pd.DataFrame,
    vario: Dict[str, float],
    params: Dict[str, float],
    n_realizations: int = 10,
    seed: int | None = 12345,
    xcol: str = "x",
    ycol: str = "y",
    vcol: str = "var",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run sequential Gaussian simulation on a block grid."""
    if n_realizations < 1:
        raise ValueError("n_realizations must be >= 1")

    min_samples = int(params.get("min_samples", 4))
    max_samples = int(params.get("max_samples", 12))
    search_radius = float(params.get("search_radius", 150.0))

    scores, table = normal_score_transform(df[vcol].to_numpy())
    data_x = df[xcol].to_numpy()
    data_y = df[ycol].to_numpy()
    data_ns = scores

    grid_xy = grid_df[["x", "y"]].to_numpy()

    model_name = _variogram_model_name(vario)
    model_params = _variogram_parameters(vario)

    simulations = np.zeros((len(grid_df), n_realizations), dtype=float)

    for r in range(n_realizations):
        rng = np.random.default_rng(None if seed is None else seed + r)
        order = rng.permutation(len(grid_df))
        cond_x = data_x.copy()
        cond_y = data_y.copy()
        cond_v = data_ns.copy()
        sim_ns = np.zeros(len(grid_df), dtype=float)

        for idx in order:
            x, y = grid_xy[idx]
            if cond_x.size >= min_samples:
                dist2 = (cond_x - x) ** 2 + (cond_y - y) ** 2
                within = dist2 <= search_radius**2
                if np.sum(within) >= min_samples:
                    within_idx = np.where(within)[0]
                    sorted_idx = np.argsort(dist2[within])
                    take = within_idx[sorted_idx[:max_samples]]
                    ok = OrdinaryKriging(
                        cond_x[take],
                        cond_y[take],
                        cond_v[take],
                        variogram_model=model_name,
                        variogram_parameters=model_params,
                        enable_plotting=False,
                        coordinates_type="euclidean",
                    )
                    mean, variance = ok.execute("points", np.array([x]), np.array([y]))
                    krig_mean = float(mean[0])
                    krig_var = max(float(variance[0]), 0.0)
                else:
                    krig_mean = 0.0
                    krig_var = 1.0
            else:
                krig_mean = 0.0
                krig_var = 1.0

            sim_value = rng.normal(krig_mean, np.sqrt(krig_var))
            sim_ns[idx] = sim_value
            cond_x = np.append(cond_x, x)
            cond_y = np.append(cond_y, y)
            cond_v = np.append(cond_v, sim_value)

        simulations[:, r] = back_transform(sim_ns, table)

    realizations_df = grid_df.copy()
    for r in range(n_realizations):
        realizations_df[f"sim_{r + 1:02d}"] = simulations[:, r]

    quantiles = np.percentile(simulations, [10, 50, 90], axis=1).T
    quantiles_df = grid_df.copy()
    quantiles_df["p10"] = quantiles[:, 0]
    quantiles_df["p50"] = quantiles[:, 1]
    quantiles_df["p90"] = quantiles[:, 2]

    return realizations_df, quantiles_df
