from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def _hash_dataframe(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha1(hashed).hexdigest()


def _direction_unit_vector(azm: float, dip: float) -> np.ndarray:
    azm_rad = np.deg2rad(azm)
    dip_rad = np.deg2rad(dip)
    return np.array(
        [
            np.cos(dip_rad) * np.cos(azm_rad),
            np.cos(dip_rad) * np.sin(azm_rad),
            np.sin(dip_rad),
        ],
        dtype=float,
    )


def experimental_variogram_3d(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str,
    vcol: str,
    params: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Compute experimental variogram for 3D coordinates with directional filters."""
    coords = df[[xcol, ycol, zcol]].to_numpy(dtype=float)
    values = df[vcol].to_numpy(dtype=float)
    n = len(df)
    nlag = int(params["nlag"])
    xlag = float(params["lag"])
    if n < 2:
        lags = np.arange(1, nlag + 1, dtype=float) * xlag
        return {"lags": lags, "gamma": np.full(nlag, np.nan), "npp": np.zeros(nlag, dtype=int), "params": params}

    azm = float(params.get("azm", 0.0))
    dip = float(params.get("dip", 0.0))
    atol = float(params.get("atol", 90.0))
    bandwh = float(params.get("bandwh", np.inf))

    diff = coords[:, None, :] - coords[None, :, :]
    iu = np.triu_indices(n, k=1)
    diff = diff[iu]
    h = np.linalg.norm(diff, axis=1)
    dv = (values[:, None] - values[None, :]) ** 2
    dv = dv[iu]

    direction = _direction_unit_vector(azm, dip)
    dot = np.abs(np.dot(diff, direction))
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.clip(dot / h, -1.0, 1.0)
    angle = np.rad2deg(np.arccos(cosang))
    perp = h * np.sin(np.deg2rad(angle))
    dir_mask = (angle <= atol) & (perp <= bandwh)

    h = h[dir_mask]
    dv = dv[dir_mask]

    max_range = xlag * nlag
    bins = np.linspace(0.0, max_range, nlag + 1)
    lags = 0.5 * (bins[:-1] + bins[1:])
    gamma = np.full(nlag, np.nan)
    npp = np.zeros(nlag, dtype=int)

    for i in range(nlag):
        mask = (h >= bins[i]) & (h < bins[i + 1])
        npp[i] = int(np.sum(mask))
        if npp[i] > 0:
            gamma[i] = 0.5 * float(np.mean(dv[mask]))

    return {"lags": lags, "gamma": gamma, "npp": npp, "params": params}


def experimental_variogram(
    df: pd.DataFrame,
    coords: Tuple[str, str, str | None],
    vcol: str,
    params: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Dispatch to 2D or 3D experimental variogram depending on z column."""
    xcol, ycol, zcol = coords
    if zcol:
        return experimental_variogram_3d(df, xcol, ycol, zcol, vcol, params)
    return experimental_variogram_2d(df, xcol, ycol, vcol, params)


def fit_variogram_model(
    exp: Dict[str, np.ndarray],
    data_variance: float,
    model_type: str = "spherical",
    min_pairs: int = 30,
    monotonic_tolerance: float = 0.02,
) -> Dict[str, float]:
    """Create a simple variogram model dictionary for GeostatsPy with basic checks."""
    model_type = model_type.lower()
    it_map = {"spherical": 1, "exponential": 2, "gaussian": 3}
    it1 = it_map.get(model_type, 1)

    npp = np.asarray(exp["npp"], dtype=float)
    gamma = np.asarray(exp["gamma"], dtype=float)
    lags = np.asarray(exp["lags"], dtype=float)
    valid = (npp >= min_pairs) & np.isfinite(gamma)
    if not np.any(valid):
        max_lag = float(np.nanmax(lags)) if np.any(np.isfinite(lags)) else 1.0
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
            "checks": {
                "min_pairs_ok": False,
                "monotonic_ok": False,
                "nugget_sill_ok": True,
                "range_ok": hmaj > 0,
            },
        }

    max_lag = float(np.nanmax(lags[valid]))
    sill = float(np.nanmax(gamma[valid]))
    if not np.isfinite(sill) or sill <= 0:
        sill = float(data_variance) if data_variance > 0 else 1.0
    nug = float(np.clip(gamma[valid][0] if np.any(valid) else 0.1 * sill, 0.0, sill))
    target = 0.95 * sill
    range_idx = np.where(gamma[valid] >= target)[0]
    if len(range_idx) > 0:
        hmaj = float(lags[valid][range_idx[0]])
    else:
        hmaj = max_lag * 0.7 if max_lag > 0 else 1.0

    cc1 = max(sill - nug, 0.0)
    monotonic_ok = True
    if np.sum(valid) > 2:
        diffs = np.diff(gamma[valid])
        monotonic_ok = bool(np.all(diffs >= -(monotonic_tolerance * sill)))

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
        "checks": {
            "min_pairs_ok": bool(np.any(valid)),
            "monotonic_ok": monotonic_ok,
            "nugget_sill_ok": nug <= sill,
            "range_ok": hmaj > 0,
        },
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


def compute_variograms_by_domain(
    df: pd.DataFrame,
    coords: Tuple[str, str, str | None],
    vcol: str,
    params: Dict[str, float],
    directions: Iterable[Dict[str, float]],
    domain_col: str | None = None,
) -> List[Dict[str, object]]:
    """Compute experimental variograms for each domain and direction."""
    results: List[Dict[str, object]] = []
    if domain_col:
        groups = df.groupby(domain_col, dropna=False)
    else:
        groups = [("ALL", df)]

    for domain, subset in groups:
        for direction in directions:
            exp_params = {**params, **direction}
            exp = experimental_variogram(subset, coords, vcol, exp_params)
            results.append(
                {
                    "domain": domain,
                    "direction": direction,
                    "exp": exp,
                }
            )
    return results


def fit_variogram_results(
    results: Iterable[Dict[str, object]],
    data_variance: float,
    model_type: str = "spherical",
    min_pairs: int = 30,
    monotonic_tolerance: float = 0.02,
) -> List[Dict[str, object]]:
    """Attach model fits with checks to variogram results."""
    fitted: List[Dict[str, object]] = []
    for item in results:
        exp = item["exp"]
        model = fit_variogram_model(
            exp,
            data_variance=data_variance,
            model_type=model_type,
            min_pairs=min_pairs,
            monotonic_tolerance=monotonic_tolerance,
        )
        fitted.append({**item, "model": model})
    return fitted


def _model_curve(model: Dict[str, float], lags: np.ndarray) -> np.ndarray:
    """Compute model curve for export without GeostatsPy."""
    nug = float(model.get("nug", 0.0))
    sill = float(model.get("nug", 0.0)) + float(model.get("cc1", 0.0))
    hmaj = float(model.get("hmaj1", 1.0))
    it1 = int(model.get("it1", 1))
    hr = np.clip(lags / max(hmaj, 1e-12), 0.0, np.inf)
    if it1 == 2:
        gamma = nug + (sill - nug) * (1.0 - np.exp(-3.0 * hr))
    elif it1 == 3:
        gamma = nug + (sill - nug) * (1.0 - np.exp(-3.0 * hr**2))
    else:
        gamma = nug + (sill - nug) * np.where(
            hr < 1.0, 1.5 * hr - 0.5 * hr**3, 1.0
        )
    return gamma


def export_variogram_results(
    results: Iterable[Dict[str, object]],
    out_dir: str,
    data_hash: str | None = None,
) -> Dict[str, str]:
    """Export variogram curves and parameters with traceability."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    curves_rows = []
    params_rows = []

    for item in results:
        exp = item["exp"]
        domain = item["domain"]
        direction = item["direction"]
        model = item.get("model")
        lags = np.asarray(exp["lags"], dtype=float)
        gamma = np.asarray(exp["gamma"], dtype=float)
        npp = np.asarray(exp["npp"], dtype=int)
        model_gamma = _model_curve(model, lags) if model else np.full_like(lags, np.nan)

        for lag, gam, pairs, mgam in zip(lags, gamma, npp, model_gamma):
            curves_rows.append(
                {
                    "run_id": run_id,
                    "domain": domain,
                    "azm": direction.get("azm", 0.0),
                    "dip": direction.get("dip", 0.0),
                    "lag": lag,
                    "gamma": gam,
                    "npp": pairs,
                    "model_gamma": mgam,
                }
            )

        params_rows.append(
            {
                "run_id": run_id,
                "domain": domain,
                "azm": direction.get("azm", 0.0),
                "dip": direction.get("dip", 0.0),
                "model": model,
            }
        )

    curves_df = pd.DataFrame(curves_rows)
    params_df = pd.DataFrame(params_rows)
    if data_hash:
        curves_df["data_hash"] = data_hash
        params_df["data_hash"] = data_hash

    curves_path = out_path / f"variogram_curves_{run_id}.csv"
    params_path = out_path / f"variogram_params_{run_id}.json"
    curves_df.to_csv(curves_path, index=False)
    params_path.write_text(params_df.to_json(orient="records", indent=2), encoding="utf-8")

    return {"curves": str(curves_path), "params": str(params_path)}
