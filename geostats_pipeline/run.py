from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from geostats_pipeline.reporting import Manifest, ReportingPaths
from src.eda import basic_stats, plot_hist, plot_qq, plot_xy_scatter
from src.grid import export_grid_to_csv, grid_from_extents, make_grid_dataframe
from src.kriging import ordinary_kriging_2d
from src.preprocess import load_and_preprocess
from src.validation import simple_cross_validation
from src.variography import (
    experimental_variogram_2d,
    fit_variogram_model,
    plot_variogram,
    save_variogram_model,
)

STAGES: List[str] = [
    "preprocess",
    "eda",
    "grid",
    "variography",
    "kriging",
    "validation",
]


def _load_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if config_path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    raise ValueError("Unsupported config format. Use .yml, .yaml, or .json")


def _validate_config(cfg: Dict) -> None:
    if not cfg:
        raise ValueError("Config is empty.")
    if "data_csv_path" not in cfg:
        raise KeyError("Missing data_csv_path in config.")
    cols = cfg.get("columns", {})
    required_cols = ["x", "y", "z", "variable_objetivo"]
    missing = [col for col in required_cols if col not in cols]
    if missing:
        raise KeyError(f"Missing required columns mapping: {missing}")


def _setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Log path: %s", log_path)


def _build_grid(cfg: Dict, df: pd.DataFrame) -> pd.DataFrame:
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

    return make_grid_dataframe(grid_spec)


def _build_variogram(cfg: Dict, df: pd.DataFrame) -> Dict:
    vario_params = cfg.get("variogram", {})
    exp = experimental_variogram_2d(df, "x", "y", "var", vario_params)
    model = fit_variogram_model(exp, float(df["var"].var(ddof=1)), model_type="spherical")
    return {"experimental": exp, "model": model}


def run_pipeline(config_path: str, stage: str, dry_run: bool) -> None:
    if importlib.util.find_spec("geostatspy") is None:
        msg = (
            "No se encontro geostatspy. Activá .venv antes de correr el pipeline.\n"
            "Windows PowerShell:\n"
            "  .\\.venv\\Scripts\\activate\n"
            "  python -m geostats_pipeline.run --config configs/config.yml --stage all\n"
            "macOS/Linux:\n"
            "  source .venv/bin/activate\n"
            "  python -m geostats_pipeline.run --config configs/config.yml --stage all"
        )
        raise SystemExit(msg)

    cfg = _load_config(config_path)
    _validate_config(cfg)

    if dry_run:
        print("Config validation OK.")
        return

    selected_stages = STAGES if stage == "all" else [stage]
    reporting = ReportingPaths()
    reporting.ensure_dirs()
    _setup_logging(reporting.log_path())

    manifest = Manifest(config_path=config_path, stages=selected_stages, dry_run=dry_run)

    df, _df_raw, _mapping = load_and_preprocess(cfg)
    required = ["x", "y", "var"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column after mapping: {col}")

    if "preprocess" in selected_stages:
        logging.info("Preprocess stage complete: %s rows", len(df))

    if "eda" in selected_stages:
        stats = basic_stats(df["var"])
        stats_path = reporting.tables / "basic_stats.csv"
        pd.DataFrame([stats]).to_csv(stats_path, index=False)
        manifest.add("eda", stats_path)

        hist_path = reporting.figures / "hist_var.png"
        qq_path = reporting.figures / "qq_var.png"
        scatter_path = reporting.figures / "xy_scatter.png"
        plot_hist(df["var"], str(hist_path))
        plot_qq(df["var"], str(qq_path))
        plot_xy_scatter(df, "x", "y", "var", str(scatter_path), color_by="domain")
        manifest.add("eda", hist_path)
        manifest.add("eda", qq_path)
        manifest.add("eda", scatter_path)

    grid_df = None
    if "grid" in selected_stages or "kriging" in selected_stages:
        grid_df = _build_grid(cfg, df)
        grid_path = reporting.tables / "grid.csv"
        export_grid_to_csv(grid_df, str(grid_path))
        grid_stage = "grid" if "grid" in selected_stages else "kriging"
        manifest.add(grid_stage, grid_path)

    variogram = None
    if "variography" in selected_stages or "kriging" in selected_stages:
        variogram = _build_variogram(cfg, df)
        variogram_plot = reporting.figures / "variogram.png"
        variogram_model = reporting.models / "variogram_var.json"
        plot_variogram(variogram["experimental"], variogram["model"], str(variogram_plot))
        save_variogram_model(variogram["model"], str(variogram_model))
        variography_stage = "variography" if "variography" in selected_stages else "kriging"
        manifest.add(variography_stage, variogram_plot)
        manifest.add(variography_stage, variogram_model)

    if "kriging" in selected_stages:
        if grid_df is None or variogram is None:
            raise RuntimeError("Kriging requires grid and variogram outputs.")
        krig_cfg = cfg.get("kriging", {})
        kriged = ordinary_kriging_2d(df, "x", "y", "var", grid_df, variogram["model"], krig_cfg)
        kriging_table = reporting.tables / "kriging_var.csv"
        kriged.to_csv(kriging_table, index=False)
        manifest.add("kriging", kriging_table)

        kriging_plot = reporting.figures / "kriging_estimate.png"
        plot_xy_scatter(kriged, "x", "y", "estimate", str(kriging_plot))
        manifest.add("kriging", kriging_plot)

    if "validation" in selected_stages:
        krig_cfg = cfg.get("kriging", {})
        cv_df, metrics = simple_cross_validation(
            df,
            "x",
            "y",
            "var",
            radius=float(krig_cfg.get("search_radius", 150.0)),
            max_samples=int(krig_cfg.get("max_samples", 12)),
        )
        metrics_path = reporting.tables / "validation_metrics.csv"
        predictions_path = reporting.tables / "validation_predictions.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        cv_df[["var", "pred"]].to_csv(predictions_path, index=False)
        manifest.add("validation", metrics_path)
        manifest.add("validation", predictions_path)

    manifest.write(reporting.manifest)
    logging.info("Pipeline completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the geostats pipeline")
    parser.add_argument("--config", required=True, help="Ruta del config YAML/JSON")
    parser.add_argument("--stage", default="all", choices=["all", *STAGES])
    parser.add_argument("--dry-run", action="store_true", help="Validar config y salir")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args.config, args.stage, args.dry_run)


if __name__ == "__main__":
    main()
