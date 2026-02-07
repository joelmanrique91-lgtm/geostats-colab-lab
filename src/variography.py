from __future__ import annotations

import json
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _import_geostatspy():
    import sys
    import types

    if "statsmodels" not in sys.modules:
        statsmodels_mod = types.ModuleType("statsmodels")
        stats_mod = types.ModuleType("statsmodels.stats")
        weightstats_mod = types.ModuleType("statsmodels.stats.weightstats")

        class DescrStatsW:
            def __init__(self, data, weights=None):
                import numpy as _np

                data = _np.asarray(data, dtype=float)
                if weights is None:
                    weights = _np.ones_like(data)
                weights = _np.asarray(weights, dtype=float)
                wsum = _np.sum(weights)
                self.mean = float(_np.sum(weights * data) / wsum) if wsum > 0 else float(_np.nanmean(data))
                self.var = float(_np.sum(weights * (data - self.mean) ** 2) / wsum) if wsum > 0 else 0.0

        weightstats_mod.DescrStatsW = DescrStatsW
        stats_mod.weightstats = weightstats_mod
        statsmodels_mod.stats = stats_mod
        sys.modules["statsmodels"] = statsmodels_mod
        sys.modules["statsmodels.stats"] = stats_mod
        sys.modules["statsmodels.stats.weightstats"] = weightstats_mod

    if "tqdm" not in sys.modules:
        tqdm_mod = types.ModuleType("tqdm")
        tqdm_mod.tqdm = lambda x=None, **kwargs: x if x is not None else []
        sys.modules["tqdm"] = tqdm_mod
    if "numba" not in sys.modules:
        numba_mod = types.ModuleType("numba")
        numba_mod.jit = lambda *args, **kwargs: (lambda f: f)
        sys.modules["numba"] = numba_mod

    import geostatspy.geostats as gs

    return gs


def experimental_variogram_2d(
    df: pd.DataFrame, xcol: str, ycol: str, vcol: str, params: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """Compute experimental variogram using GeostatsPy gamv."""
    gs = _import_geostatspy()
    tmin = float(df[vcol].min())
    tmax = float(df[vcol].max())
    xlag = float(params["lag"])
    xltol = float(params.get("xltol", xlag / 2.0))
    nlag = int(params["nlag"])
    azm = float(params["azm"])
    atol = float(params["atol"])
    bandwh = float(params["bandwh"])

    dis, vario, npp = gs.gamv(
        df,
        xcol,
        ycol,
        vcol,
        tmin,
        tmax,
        xlag,
        xltol,
        nlag,
        azm,
        atol,
        bandwh,
        isill=1,
    )

    return {
        "lags": dis,
        "gamma": vario,
        "npp": npp,
        "params": params,
    }


def fit_variogram_model(
    exp: Dict[str, np.ndarray],
    data_variance: float,
    model_type: str = "spherical",
) -> Dict[str, float]:
    """Create a simple variogram model dictionary for GeostatsPy."""
    model_type = model_type.lower()
    it_map = {"spherical": 1, "exponential": 2, "gaussian": 3}
    it1 = it_map.get(model_type, 1)

    max_lag = float(np.nanmax(exp["lags"]))
    sill = float(data_variance) if data_variance > 0 else 1.0
    nug = 0.1 * sill
    cc1 = max(sill - nug, 0.0)
    hmaj = max_lag * 0.7 if max_lag > 0 else 1.0

    return {
        "nst": 1,
        "nug": nug,
        "cc1": cc1,
        "it1": it1,
        "azi1": float(exp["params"].get("azm", 0.0)),
        "hmaj1": hmaj,
        "hmin1": hmaj,
        "cc2": 0.0,
        "it2": 1,
        "azi2": 0.0,
        "hmaj2": hmaj,
        "hmin2": hmaj,
    }


def plot_variogram(exp: Dict[str, np.ndarray], model: Dict[str, float] | None, outpath: str) -> None:
    """Plot experimental variogram and optional model curve."""
    plt.figure(figsize=(6, 4))
    plt.plot(exp["lags"], exp["gamma"], "o-", label="experimental")

    if model:
        gs = _import_geostatspy()
        nlag = int(exp["params"]["nlag"])
        xlag = float(exp["params"]["lag"])
        azm = float(exp["params"]["azm"])
        _, h, gam, _, _ = gs.vmodel(nlag, xlag, azm, model)
        plt.plot(h, gam, "-", label="model")

    plt.xlabel("lag distance")
    plt.ylabel("semivariance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_variogram_model(model: Dict[str, float], path: str) -> None:
    """Save variogram model to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
