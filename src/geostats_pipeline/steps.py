from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .anisotropy import anisotropy_sweep
from .block_model import BlockDiscretization, build_block_grid, discretize_blocks, grid_from_extents
from .config import load_config
from .declustering import cell_declustering
from .eda import basic_stats, plot_histogram, plot_xy_scatter
from .io import load_data, standardize_columns
from .qaqc import basic_qaqc, outlier_report
from .reporting import RunPaths, create_run_dir, save_figure, save_table, write_manifest
from .support import resolve_support
from .validation import (
    CVResult,
    compute_cv_metrics,
    kriging_cross_validation,
    plot_swath_panels,
)
from .variography import experimental_variogram, fit_variogram_model, plot_variogram
from .kriging import SearchParameters, ordinary_kriging


def _prepare_data(config: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    raw_df, metadata = load_data(config)
    df = standardize_columns(raw_df, config)

    qaqc_cfg = config["qaqc"]
    df = basic_qaqc(df, "value", duplicate_strategy=qaqc_cfg["duplicate_strategy"])
    return df, metadata


def _resolve_variogram_model(df: pd.DataFrame, config: Dict[str, object], run_paths: RunPaths) -> Dict[str, float]:
    var_cfg = config["variography"]
    exp = experimental_variogram(
        df,
        "x",
        "y",
        "z",
        "value",
        n_lags=var_cfg["n_lags"],
        lag_size=float(var_cfg["lag_size"]),
        max_pairs=int(var_cfg["max_pairs"]),
    )
    sill = float(df["value"].var(ddof=1)) if var_cfg["initial_params"]["sill"] is None else float(
        var_cfg["initial_params"]["sill"]
    )
    rng = float(var_cfg["lag_size"] * var_cfg["n_lags"]) if var_cfg["initial_params"]["range"] is None else float(
        var_cfg["initial_params"]["range"]
    )
    nugget = float(var_cfg["initial_params"]["nugget"])
    model = fit_variogram_model(exp, var_cfg["model_type"], sill=sill, rng=rng, nugget=nugget)
    plot_variogram(exp, model, str(run_paths.figure_path("variogram.png")))
    model_path = run_paths.model_path("variogram_model.json")
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def _build_search_params(config: Dict[str, object]) -> SearchParameters:
    neigh = config["kriging"]["neighborhood"]
    return SearchParameters(
        ranges=(float(neigh["ranges"]["major"]), float(neigh["ranges"]["minor"]), float(neigh["ranges"]["vertical"])),
        angles=(float(neigh["angles"]["azimuth"]), float(neigh["angles"]["dip"]), float(neigh["angles"]["rake"])),
        min_samples=int(neigh["min_samples"]),
        max_samples=int(neigh["max_samples"]),
        max_per_drillhole=neigh["max_per_hole"],
        octants=int(neigh["octants"]),
        condition_max=float(neigh["condition_max"]),
        drillhole_col="hole_id" if config["data"].get("hole_id_col") else None,
    )


def run_setup_check(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, metadata = _prepare_data(config)
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "columns_list": list(df.columns),
    }
    save_table(pd.DataFrame([summary]), run_paths, "setup_check_summary.csv", index=False)
    save_table(df.head(20), run_paths, "setup_check_head.csv", index=False)
    return {"summary": summary, "metadata": metadata}


def run_data_qaqc(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    outlier_cfg = config["qaqc"]["outlier"]
    outliers = outlier_report(df, "value", zscore=float(outlier_cfg["zscore"])) if outlier_cfg["enabled"] else pd.DataFrame()
    if not outliers.empty:
        save_table(outliers, run_paths, "qaqc_outliers.csv", index=False)

    stats = basic_stats(df["value"])
    save_table(pd.DataFrame([stats]), run_paths, "qaqc_basic_stats.csv", index=False)
    save_table(df.isna().sum().to_frame("missing").reset_index(), run_paths, "qaqc_missing.csv", index=False)
    plot_histogram(df["value"], str(run_paths.figure_path("qaqc_histogram.png")), title="Histogram")
    return {"stats": stats}


def run_compositing_declustering(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    df, support_mode = resolve_support(df, config)
    result = {"support_mode": support_mode}

    if support_mode in {"interval", "pseudo-interval"} and config["compositing"]["enabled"]:
        from .compositing import composite_by_length

        comp = composite_by_length(df, config, output_dir=str(run_paths.base))
        df = comp.rename(columns={"from": "from", "to": "to"}).copy()
        result["composites"] = int(len(comp))
    else:
        result["composites"] = 0

    if config["declustering"]["enabled"]:
        declust = cell_declustering(df, config, output_dir=str(run_paths.base))
        df = declust.copy()
        result["declustering"] = True
    else:
        result["declustering"] = False

    save_table(df, run_paths, "compositing_declustering_samples.csv", index=False)
    stats_naive = basic_stats(df["value"])
    stats_weighted = (
        basic_stats(df["value"], df["declust_weight"]) if "declust_weight" in df.columns else {}
    )
    save_table(
        pd.DataFrame([{"type": "naive", **stats_naive}, {"type": "declustered", **stats_weighted}]),
        run_paths,
        "declustering_stats.csv",
        index=False,
    )
    return result


def run_eda_domain_spatial(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    plot_xy_scatter(df, "x", "y", "value", str(run_paths.figure_path("eda_spatial.png")), color_by="domain")
    return {"status": "ok"}


def run_variography(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    model = _resolve_variogram_model(df, config, run_paths)
    return {"model": model}


def run_anisotropy(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    if not config["anisotropy"]["enabled"]:
        return {"status": "disabled"}
    results = anisotropy_sweep(
        df,
        "x",
        "y",
        "value",
        config["anisotropy"]["azimuths"],
        config["variography"]["n_lags"],
        config["variography"]["lag_size"],
        config["variography"]["max_pairs"],
    )
    save_table(pd.DataFrame(results), run_paths, "anisotropy_sweep.csv", index=False)
    return {"results": results}


def run_block_model(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    grid_cfg = config["block_model"]
    if grid_cfg["auto_from_data"]:
        spec = grid_from_extents(df, float(grid_cfg["dx"]), float(grid_cfg["dy"]), float(grid_cfg["dz"]), pad=float(grid_cfg["pad"]))
    else:
        spec = {
            "xmin": float(grid_cfg["xmin"]),
            "ymin": float(grid_cfg["ymin"]),
            "zmin": float(grid_cfg["zmin"]),
            "nx": int(grid_cfg["nx"]),
            "ny": int(grid_cfg["ny"]),
            "nz": int(grid_cfg["nz"]),
            "dx": float(grid_cfg["dx"]),
            "dy": float(grid_cfg["dy"]),
            "dz": float(grid_cfg["dz"]),
        }
    grid_df = build_block_grid(spec)
    save_table(grid_df, run_paths, "block_model_grid.csv", index=False)
    return {"grid_spec": spec, "rows": int(len(grid_df))}


def run_estimation_ok(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    model = _resolve_variogram_model(df, config, run_paths)
    search = _build_search_params(config)

    grid_spec = run_block_model(config, run_paths)["grid_spec"]
    grid_df = build_block_grid(grid_spec)

    mode = config["kriging"]["mode"]
    block_cfg = config["kriging"]["block"]
    discretization = BlockDiscretization(**block_cfg["discretization"])
    block_points = None
    if mode == "block":
        if any(val is None for val in (block_cfg["dx"], block_cfg["dy"], block_cfg["dz"])):
            mode = "point"
        else:
            subcells = discretize_blocks(
                grid_df,
                size=(block_cfg["dx"], block_cfg["dy"], block_cfg["dz"]),
                discretization=discretization,
            )
            block_points = subcells.groupby("block_id")[["x", "y", "z"]].apply(lambda x: x.to_numpy()).tolist()

    rows = []
    for idx, row in grid_df.iterrows():
        if mode == "block" and block_points is not None:
            points = block_points[idx]
        else:
            points = None
        result = ordinary_kriging(
            df,
            "x",
            "y",
            "z",
            "value",
            (row["x"], row["y"], row["z"]),
            model,
            search,
            block_points=points,
        )
        rows.append({**row.to_dict(), **result})

    out = pd.DataFrame(rows)
    save_table(out, run_paths, "kriging_estimates.csv", index=False)
    plot_xy_scatter(out, "x", "y", "estimate", str(run_paths.figure_path("kriging_estimate.png")))
    return {"rows": int(len(out))}


def run_validation(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    model = _resolve_variogram_model(df, config, run_paths)
    search = _build_search_params(config)
    cv_cfg = config["validation"]
    cv_result = kriging_cross_validation(
        df,
        "x",
        "y",
        "z",
        "value",
        model,
        search,
        method=cv_cfg["cv"],
        n_splits=cv_cfg["kfold_splits"],
    )
    save_table(cv_result.data, run_paths, "validation_predictions.csv", index=False)
    save_table(pd.DataFrame([cv_result.metrics]), run_paths, "validation_metrics.csv", index=False)

    domain_col = "domain" if "domain" in df.columns else None
    if domain_col:
        fig = plot_swath_panels(df, "x", "y", "value", domain_col, n_bins=cv_cfg["swath_bins"])
        save_figure(fig, run_paths, "validation_swath.png")
    return {"metrics": cv_result.metrics}


def run_simulation(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    if not config["simulation"]["enabled"]:
        return {"status": "disabled"}
    krig_path = run_paths.table_path("kriging_estimates.csv")
    if not krig_path.exists():
        run_estimation_ok(config, run_paths)
    krig = pd.read_csv(krig_path)
    rng = np.random.default_rng(int(config["simulation"]["random_seed"]))
    nreal = int(config["simulation"]["n_realizations"])
    std = np.sqrt(np.clip(krig["variance"].to_numpy(dtype=float), 0.0, None))
    sims = rng.normal(loc=krig["estimate"].to_numpy()[:, None], scale=std[:, None], size=(len(krig), nreal))
    summary = pd.DataFrame(
        {
            "x": krig["x"],
            "y": krig["y"],
            "mean": sims.mean(axis=1),
            "p10": np.percentile(sims, 10, axis=1),
            "p90": np.percentile(sims, 90, axis=1),
        }
    )
    save_table(summary, run_paths, "simulation_summary.csv", index=False)
    return {"rows": int(len(summary))}


def run_reporting(config: Dict[str, object], run_paths: RunPaths, metadata: Dict[str, object], metrics: Dict[str, object]) -> Dict[str, object]:
    manifest_path, _ = write_manifest(
        run_paths,
        config={
            "config": config,
            "input": metadata,
            "metrics": metrics,
        },
    )
    return {"manifest": str(manifest_path)}


def run_pipeline(config_path: str, stage: str = "all") -> RunPaths:
    config = load_config(config_path)
    run_name = config["outputs"]["run_name"]
    run_paths = create_run_dir(config["outputs"]["base_dir"], prefix="run" if run_name == "auto" else run_name)

    metadata: Dict[str, object] = {}
    metrics: Dict[str, object] = {}

    if stage in {"all", "setup"}:
        result = run_setup_check(config, run_paths)
        metadata.update(result.get("metadata", {}))

    if stage in {"all", "qaqc"}:
        metrics["qaqc"] = run_data_qaqc(config, run_paths)

    if stage in {"all", "compositing"}:
        metrics["compositing"] = run_compositing_declustering(config, run_paths)

    if stage in {"all", "eda"}:
        metrics["eda"] = run_eda_domain_spatial(config, run_paths)

    if stage in {"all", "variography"}:
        metrics["variography"] = run_variography(config, run_paths)

    if stage in {"all", "anisotropy"}:
        metrics["anisotropy"] = run_anisotropy(config, run_paths)

    if stage in {"all", "blockmodel"}:
        metrics["blockmodel"] = run_block_model(config, run_paths)

    if stage in {"all", "kriging"}:
        metrics["kriging"] = run_estimation_ok(config, run_paths)

    if stage in {"all", "validation"}:
        metrics["validation"] = run_validation(config, run_paths)

    if stage in {"all", "simulation"}:
        metrics["simulation"] = run_simulation(config, run_paths)

    if stage in {"all", "report"}:
        run_reporting(config, run_paths, metadata, metrics)

    return run_paths
