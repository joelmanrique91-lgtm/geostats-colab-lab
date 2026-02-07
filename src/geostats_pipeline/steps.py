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
from .qaqc import apply_topcut, basic_qaqc
from .reporting import RunPaths, create_run_dir, save_figure, save_table, write_manifest
from .support import resolve_support
from .trending import TrendModel, fit_trend
from .transforms import indicator_transform, normal_score_transform
from .validation import compute_cv_metrics, kriging_cross_validation, plot_swath_comparison
from .variography import experimental_variogram, fit_variogram_model, plot_variogram
from .kriging import SearchParameters, ordinary_kriging


def _apply_topcut_if_needed(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    topcut_cfg = config.get("topcut", {})
    if not topcut_cfg.get("enabled"):
        return df
    high = topcut_cfg.get("high")
    if high is None:
        raise ValueError("Top-cut enabled but 'high' threshold is not set.")
    domain_col = "domain" if topcut_cfg.get("by_domain") else None
    return apply_topcut(df, "value", float(high), domain_col=domain_col)


def _apply_trend_if_needed(
    df: pd.DataFrame,
    config: Dict[str, object],
    run_paths: RunPaths,
) -> tuple[pd.DataFrame, TrendModel | None, Dict[str, object]]:
    trend_cfg = config.get("trend", {})
    enabled = trend_cfg.get("enabled", "auto")
    order = int(trend_cfg.get("order", 1))
    r2_threshold = float(trend_cfg.get("r2_threshold", 0.0))
    if enabled is False or enabled == "false":
        return df, None, {"enabled": False}

    trend_model, r2 = fit_trend(df, "value", ("x", "y", "z"), order=order)
    use_trend = enabled is True or enabled == "true" or (enabled == "auto" and r2 >= r2_threshold)

    info = {"enabled": bool(use_trend), "r2": r2, "order": order, "r2_threshold": r2_threshold}
    if not use_trend:
        save_table(pd.DataFrame([info]), run_paths, "trend_summary.csv", index=False)
        return df, None, info

    work = df.copy()
    work["trend"] = trend_model.predict(work)
    work["residual"] = work["value"] - work["trend"]
    save_table(pd.DataFrame([info]), run_paths, "trend_summary.csv", index=False)
    return work, trend_model, info


def _prepare_data(config: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    raw_df, metadata = load_data(config)
    df = standardize_columns(raw_df, config)

    qaqc_cfg = config["qaqc"]
    df = basic_qaqc(df, "value", duplicate_strategy=qaqc_cfg["duplicate_strategy"])
    df = _apply_topcut_if_needed(df, config)
    return df, metadata


def _resolve_variogram_model(
    df: pd.DataFrame,
    config: Dict[str, object],
    run_paths: RunPaths,
    tag: str | None = None,
) -> Dict[str, float]:
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
    suffix = f"_{tag}" if tag else ""
    plot_variogram(exp, model, str(run_paths.figure_path(f"variogram{suffix}.png")))
    model_path = run_paths.model_path(f"variogram_model{suffix}.json")
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
    raw_df, metadata = load_data(config)
    save_table(raw_df, run_paths, "dataset_original.csv", index=False)
    df = standardize_columns(raw_df, config)
    df = basic_qaqc(df, "value", duplicate_strategy=config["qaqc"]["duplicate_strategy"])
    df = _apply_topcut_if_needed(df, config)
    save_table(df, run_paths, "dataset_capped.csv", index=False)

    stats = basic_stats(df["value"])
    save_table(pd.DataFrame([stats]), run_paths, "qaqc_basic_stats.csv", index=False)
    save_table(df.isna().sum().to_frame("missing").reset_index(), run_paths, "qaqc_missing.csv", index=False)
    plot_histogram(df["value"], str(run_paths.figure_path("qaqc_histogram.png")), title="Histogram")

    topcut_cfg = config.get("topcut", {})
    if topcut_cfg.get("enabled"):
        cap_report = {
            "enabled": True,
            "high": float(topcut_cfg["high"]),
            "capped_count": int(df["capped_flag"].sum()) if "capped_flag" in df.columns else 0,
        }
        save_table(pd.DataFrame([cap_report]), run_paths, "topcut_report.csv", index=False)
    else:
        cap_report = {"enabled": False}

    indicators_cfg = config.get("transforms", {}).get("indicators", {})
    if indicators_cfg.get("enabled") and indicators_cfg.get("thresholds"):
        indicators, thresholds = indicator_transform(df["value"], indicators_cfg["thresholds"])
        save_table(indicators, run_paths, "indicator_variables.csv", index=False)

    return {"stats": stats, "topcut": cap_report, "input": metadata}


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
    if config["declustering"]["enabled"]:
        df = cell_declustering(df, config, output_dir=str(run_paths.base))
    plot_xy_scatter(df, "x", "y", "value", str(run_paths.figure_path("eda_spatial.png")), color_by="domain")
    stats_naive = basic_stats(df["value"])
    stats_weighted = (
        basic_stats(df["value"], df["declust_weight"]) if "declust_weight" in df.columns else {}
    )
    save_table(
        pd.DataFrame([{"type": "naive", **stats_naive}, {"type": "declustered", **stats_weighted}]),
        run_paths,
        "eda_declustering_stats.csv",
        index=False,
    )
    return {"status": "ok", "declustering_stats": True}


def run_variography(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    df, trend_model, trend_info = _apply_trend_if_needed(df, config, run_paths)
    value_col = "residual" if trend_model else "value"

    models: Dict[str, Dict[str, float]] = {}
    domain_col = "domain" if "domain" in df.columns else None
    if domain_col:
        for domain, group in df.groupby(domain_col, dropna=False):
            model = _resolve_variogram_model(
                group.rename(columns={value_col: "value"}),
                config,
                run_paths,
                tag=f"domain_{domain}",
            )
            tag = f"domain_{domain}"
            models[str(tag)] = model
    else:
        model = _resolve_variogram_model(
            df.rename(columns={value_col: "value"}),
            config,
            run_paths,
            tag="global",
        )
        models["global"] = model
    return {"models": models, "trend": trend_info}


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
    df, trend_model, trend_info = _apply_trend_if_needed(df, config, run_paths)
    value_col = "residual" if trend_model else "value"
    model = _resolve_variogram_model(df.rename(columns={value_col: "value"}), config, run_paths, tag="global")
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
            value_col,
            (row["x"], row["y"], row["z"]),
            model,
            search,
            block_points=points,
        )
        rows.append({**row.to_dict(), **result})

    out = pd.DataFrame(rows)
    if trend_model:
        trend_vals = trend_model.predict(out)
        out["trend"] = trend_vals
        out["residual_estimate"] = out["estimate"]
        out["estimate"] = out["estimate"] + out["trend"]
    save_table(out, run_paths, "kriging_estimates.csv", index=False)
    plot_xy_scatter(out, "x", "y", "estimate", str(run_paths.figure_path("kriging_estimate.png")))
    return {"rows": int(len(out)), "trend": trend_info}


def run_validation(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    df, _metadata = _prepare_data(config)
    df, trend_model, trend_info = _apply_trend_if_needed(df, config, run_paths)
    value_col = "residual" if trend_model else "value"
    model = _resolve_variogram_model(df.rename(columns={value_col: "value"}), config, run_paths, tag="global")
    search = _build_search_params(config)
    cv_cfg = config["validation"]
    cv_result = kriging_cross_validation(
        df,
        "x",
        "y",
        "z",
        value_col,
        model,
        search,
        method=cv_cfg["cv"],
        n_splits=cv_cfg["kfold_splits"],
    )
    if trend_model:
        trend_vals = trend_model.predict(cv_result.data)
        cv_result.data["trend"] = trend_vals
        cv_result.data["estimate"] = cv_result.data["estimate"] + trend_vals
    save_table(cv_result.data, run_paths, "validation_predictions.csv", index=False)
    save_table(pd.DataFrame([cv_result.metrics]), run_paths, "validation_metrics.csv", index=False)

    domain_col = "domain" if "domain" in df.columns else None
    if domain_col:
        fig = plot_swath_comparison(
            cv_result.data,
            "x",
            "y",
            "value",
            "estimate",
            domain_col,
            n_bins=cv_cfg["swath_bins"],
        )
        save_figure(fig, run_paths, "validation_swath_comparison.png")
        domain_metrics = []
        for domain, group in cv_result.data.groupby(domain_col, dropna=False):
            metrics = compute_cv_metrics(group, vcol="value")
            metrics["domain"] = domain
            domain_metrics.append(metrics)
        save_table(pd.DataFrame(domain_metrics), run_paths, "validation_metrics_by_domain.csv", index=False)
    return {"metrics": cv_result.metrics, "trend": trend_info}


def run_simulation(config: Dict[str, object], run_paths: RunPaths) -> Dict[str, object]:
    if not config["simulation"]["enabled"]:
        return {"status": "disabled"}
    krig_path = run_paths.table_path("kriging_estimates.csv")
    if not krig_path.exists():
        run_estimation_ok(config, run_paths)
    krig = pd.read_csv(krig_path)
    df, _metadata = _prepare_data(config)
    rng = np.random.default_rng(int(config["simulation"]["random_seed"]))
    nreal = int(config["simulation"]["n_realizations"])
    std = np.sqrt(np.clip(krig["variance"].to_numpy(dtype=float), 0.0, None))
    estimates = krig["estimate"].to_numpy(dtype=float)
    ns_cfg = config.get("transforms", {}).get("normal_score", {})
    if ns_cfg.get("enabled"):
        ns = normal_score_transform(df["value"])
        estimates = ns.transform(estimates)
    sims = rng.normal(loc=estimates[:, None], scale=std[:, None], size=(len(krig), nreal))
    if ns_cfg.get("enabled"):
        sims = ns.back_transform(sims.flatten()).reshape(sims.shape)
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
