from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from eda import basic_stats, plot_hist, plot_qq, plot_xy_scatter
from grid import export_grid_to_csv, grid_from_extents, make_grid_dataframe
from kriging import ordinary_kriging_2d
from preprocess import load_and_preprocess
from validation import simple_cross_validation
from variography import (
    experimental_variogram_2d,
    fit_variogram_model,
    plot_variogram,
    save_variogram_model,
)

from .core import ensure_output_dirs


def _load_data(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str | None]]:
    return load_and_preprocess(cfg)


def _grid_from_config(df: pd.DataFrame, cfg: Dict) -> Tuple[Dict[str, float], pd.DataFrame]:
    grid_cfg = cfg.get("grid", {})
    if grid_cfg.get("auto_from_data", True):
        grid_spec = grid_from_extents(
            df,
            "x",
            "y",
            "z",
            float(grid_cfg.get("dx", 25.0)),
            float(grid_cfg.get("dy", 25.0)),
            float(grid_cfg.get("dz", 5.0)),
            pad=float(grid_cfg.get("pad", 0.0)),
        )
    else:
        grid_spec = {
            "nx": grid_cfg["nx"],
            "ny": grid_cfg["ny"],
            "nz": grid_cfg.get("nz", 1),
            "xmin": grid_cfg["xmin"],
            "ymin": grid_cfg["ymin"],
            "zmin": grid_cfg.get("zmin", 0.0),
            "dx": grid_cfg["dx"],
            "dy": grid_cfg["dy"],
            "dz": grid_cfg.get("dz", 1.0),
        }
    grid_df = make_grid_dataframe(grid_spec)
    return grid_spec, grid_df


def run_setup_check(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    df, df_raw, mapping = _load_data(cfg)

    summary = pd.DataFrame(
        [
            {
                "raw_rows": df_raw.shape[0],
                "raw_columns": df_raw.shape[1],
                "processed_rows": df.shape[0],
                "processed_columns": df.shape[1],
                "mapped_columns": json.dumps(mapping, ensure_ascii=False),
            }
        ]
    )
    summary_path = os.path.join(paths["tables"], "setup_check_summary.csv")
    summary.to_csv(summary_path, index=False)

    head_path = os.path.join(paths["tables"], "setup_check_head.csv")
    df.head(20).to_csv(head_path, index=False)

    return [summary_path, head_path]


def run_data_qaqc(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    df, _df_raw, _mapping = _load_data(cfg)

    stats = basic_stats(df["var"])
    stats_path = os.path.join(paths["tables"], "qaqc_basic_stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    missing_report = df.isna().sum().to_frame(name="missing").reset_index().rename(columns={"index": "column"})
    missing_path = os.path.join(paths["tables"], "qaqc_missing_report.csv")
    missing_report.to_csv(missing_path, index=False)

    hist_path = os.path.join(paths["figures"], "qaqc_hist_var.png")
    qq_path = os.path.join(paths["figures"], "qaqc_qq_var.png")
    plot_hist(df["var"], hist_path)
    plot_qq(df["var"], qq_path)

    return [stats_path, missing_path, hist_path, qq_path]


def run_compositing_declustering(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    df, _df_raw, _mapping = _load_data(cfg)

    df = df.copy()
    df["decluster_weight"] = 1.0

    out_path = os.path.join(paths["tables"], "compositing_declustering_samples.csv")
    df.to_csv(out_path, index=False)

    stats = basic_stats(df["var"])
    stats_path = os.path.join(paths["tables"], "compositing_declustering_stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    return [out_path, stats_path]


def run_eda_domain_spatial(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    df, _df_raw, _mapping = _load_data(cfg)

    fig_path = os.path.join(paths["figures"], "eda_domain_spatial.png")
    plot_xy_scatter(df, "x", "y", "var", fig_path, color_by="domain")

    return [fig_path]


def run_variography(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    df, _df_raw, _mapping = _load_data(cfg)

    vario_params = cfg.get("variogram", {})
    exp = experimental_variogram_2d(df, "x", "y", "var", vario_params)
    model = fit_variogram_model(exp, float(df["var"].var(ddof=1)), model_type="spherical")

    fig_path = os.path.join(paths["figures"], "variogram.png")
    model_path = os.path.join(paths["models"], "variogram_var.json")
    plot_variogram(exp, model, fig_path)
    save_variogram_model(model, model_path)

    return [fig_path, model_path]


def run_block_model(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    df, _df_raw, _mapping = _load_data(cfg)
    _grid_spec, grid_df = _grid_from_config(df, cfg)

    grid_path = os.path.join(paths["tables"], "block_model_grid.csv")
    export_grid_to_csv(grid_df, grid_path)

    return [grid_path]


def run_estimation_ok(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    df, _df_raw, _mapping = _load_data(cfg)
    _grid_spec, grid_df = _grid_from_config(df, cfg)

    model_path = os.path.join(paths["models"], "variogram_var.json")
    if os.path.exists(model_path):
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
    else:
        vario_params = cfg.get("variogram", {})
        exp = experimental_variogram_2d(df, "x", "y", "var", vario_params)
        model = fit_variogram_model(exp, float(df["var"].var(ddof=1)), model_type="spherical")
        save_variogram_model(model, model_path)

    krig_cfg = cfg.get("kriging", {})
    kriged = ordinary_kriging_2d(df, "x", "y", "var", grid_df, model, krig_cfg)

    out_path = os.path.join(paths["tables"], "kriging_estimates.csv")
    kriged.to_csv(out_path, index=False)

    fig_path = os.path.join(paths["figures"], "kriging_estimate.png")
    plot_xy_scatter(kriged, "x", "y", "estimate", fig_path)

    return [out_path, fig_path, model_path]


def run_validation(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    df, _df_raw, _mapping = _load_data(cfg)

    krig_cfg = cfg.get("kriging", {})
    cv_df, metrics = simple_cross_validation(
        df,
        "x",
        "y",
        "var",
        radius=float(krig_cfg.get("search_radius", 150.0)),
        max_samples=int(krig_cfg.get("max_samples", 12)),
    )

    metrics_path = os.path.join(paths["tables"], "validation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    preds_path = os.path.join(paths["tables"], "validation_predictions.csv")
    cv_df[["var", "pred"]].to_csv(preds_path, index=False)

    return [metrics_path, preds_path]


def run_uncertainty_simulation(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)

    kriging_path = os.path.join(paths["tables"], "kriging_estimates.csv")
    if not os.path.exists(kriging_path):
        run_estimation_ok(cfg, output_dir=output_dir)

    kriging_df = pd.read_csv(kriging_path)
    n_realizations = int(cfg.get("simulation", {}).get("n_realizations", 25))

    rng = np.random.default_rng(42)
    variance = kriging_df.get("variance", pd.Series(np.zeros(len(kriging_df))))
    std = np.sqrt(np.clip(variance, 0.0, None))
    simulations = rng.normal(loc=kriging_df["estimate"].values[:, None], scale=std.values[:, None], size=(len(kriging_df), n_realizations))

    summary = pd.DataFrame(
        {
            "x": kriging_df["x"],
            "y": kriging_df["y"],
            "mean": simulations.mean(axis=1),
            "p10": np.percentile(simulations, 10, axis=1),
            "p90": np.percentile(simulations, 90, axis=1),
        }
    )

    summary_path = os.path.join(paths["tables"], "simulation_summary.csv")
    summary.to_csv(summary_path, index=False)

    return [summary_path, kriging_path]


def run_reporting_export(cfg: Dict, output_dir: str = "outputs") -> List[str]:
    paths = ensure_output_dirs(output_dir)
    manifest_path = os.path.join(paths["manifests"], "manifest.jsonl")
    report_path = os.path.join(paths["tables"], "reporting_export.csv")

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        rows = [
            {
                "timestamp": entry.get("timestamp"),
                "step": entry.get("step"),
                "artifact": artifact,
            }
            for entry in entries
            for artifact in entry.get("artifacts", [])
        ]
        pd.DataFrame(rows).to_csv(report_path, index=False)
    else:
        pd.DataFrame([], columns=["timestamp", "step", "artifact"]).to_csv(report_path, index=False)

    return [report_path]
