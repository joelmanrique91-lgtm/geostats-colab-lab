from __future__ import annotations

from typing import Dict, Tuple

import contextlib
import io

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


def ordinary_kriging_2d(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    vcol: str,
    grid_df: pd.DataFrame,
    vario: Dict[str, float],
    params: Dict[str, float],
) -> pd.DataFrame:
    """Run ordinary kriging using GeostatsPy kb2d_locations (2D)."""
    gs = _import_geostatspy()

    tmin = float(df[vcol].min())
    tmax = float(df[vcol].max())

    df_loc = grid_df.copy()
    if vcol not in df_loc.columns:
        df_loc[vcol] = np.nan

    # GeostatsPy prints progress to stdout; redirect to avoid noisy output or
    # issues in environments with restricted stdout handles.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        klist, vlist = gs.kb2d_locations(
            df,
            xcol,
            ycol,
            vcol,
            tmin,
            tmax,
            df_loc,
            "x",
            "y",
            int(params["min_samples"]),
            int(params["max_samples"]),
            float(params["search_radius"]),
            ktype=1,
            skmean=0.0,
            vario=vario,
        )

    out = df_loc.copy()
    out["estimate"] = klist
    out["variance"] = vlist
    return out
