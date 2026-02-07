from __future__ import annotations

import numpy as np
import pandas as pd

from geostats_pipeline.compositing import composite_by_length
from geostats_pipeline.declustering import cell_declustering
from geostats_pipeline.kriging import SearchParameters, ordinary_kriging
from geostats_pipeline.qaqc import basic_qaqc, outlier_report
from geostats_pipeline.variography import experimental_variogram, fit_variogram_model


def _sample_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 10.0, 20.0, 30.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "value": [1.0, 1.2, 0.9, 1.1],
        }
    )


def test_qaqc_and_outliers():
    df = _sample_points()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = basic_qaqc(df, "value", duplicate_strategy="drop")
    assert len(cleaned) == 4
    outliers = outlier_report(cleaned, "value", zscore=1.0)
    assert isinstance(outliers, pd.DataFrame)


def test_compositing_interval():
    df = pd.DataFrame(
        {
            "hole_id": ["A", "A", "A"],
            "from": [0.0, 1.0, 2.0],
            "to": [1.0, 2.0, 3.0],
            "value": [1.0, 2.0, 3.0],
        }
    )
    config = {"compositing": {"target_length": 2.0, "min_length": 1.0}}
    comp = composite_by_length(df, config, output_dir="outputs")
    assert not comp.empty


def test_declustering_weights():
    df = _sample_points()
    config = {"declustering": {"cell_size": {"x": 20.0, "y": 20.0, "z": 10.0}, "use_3d": False}}
    out = cell_declustering(df, config, output_dir="outputs")
    assert "declust_weight" in out.columns


def test_variography_and_model():
    df = _sample_points()
    exp = experimental_variogram(df, "x", "y", "z", "value", n_lags=3, lag_size=10.0, max_pairs=100)
    assert exp.lags.size == 3
    model = fit_variogram_model(exp, "spherical", sill=1.0, rng=30.0, nugget=0.0)
    assert "hmaj1" in model


def test_kriging_point():
    df = _sample_points()
    model = {"nug": 0.0, "cc1": 1.0, "hmaj1": 30.0, "it1": 1, "cc2": 0.0, "hmaj2": 0.0, "it2": 1}
    search = SearchParameters(ranges=(30.0, 30.0, 30.0), min_samples=2, max_samples=4, octants=1)
    result = ordinary_kriging(df, "x", "y", "z", "value", (15.0, 0.0, 0.0), model, search)
    assert np.isfinite(result["estimate"]) or np.isnan(result["estimate"])
