from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict

import pandas as pd

from .eda import basic_stats, plot_hist, plot_qq, plot_xy_scatter
from .grid import export_grid_to_csv, grid_from_extents, make_grid_dataframe
from .kriging import ordinary_kriging_2d
from .preprocess import load_and_preprocess
from .validation import simple_cross_validation
from .variography import experimental_variogram_2d, fit_variogram_model, plot_variogram, save_variogram_model


def _setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"pipeline_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_path


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(config_path: str) -> None:
    import importlib.util

    if importlib.util.find_spec("geostatspy") is None:
        msg = (
            "No se encontro geostatspy. Activá .venv antes de correr el pipeline.\n"
            "Windows PowerShell:\n"
            "  .\\.venv\\Scripts\\activate\n"
            "  python -m src.pipeline config/project.json\n"
            "macOS/Linux:\n"
            "  source .venv/bin/activate\n"
            "  python -m src.pipeline config/project.json"
        )
        raise SystemExit(msg)

    cfg = _load_config(config_path)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    log_path = _setup_logging("outputs/logs")
    logging.info("Log path: %s", log_path)

    df, _df_raw, _mapping = load_and_preprocess(cfg)
    required = ["x", "y", "var"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column after mapping: {col}")

    stats = basic_stats(df["var"])
    pd.DataFrame([stats]).to_csv("outputs/tables/basic_stats.csv", index=False)

    plot_hist(df["var"], "outputs/figures/hist_var.png")
    plot_qq(df["var"], "outputs/figures/qq_var.png")
    plot_xy_scatter(df, "x", "y", "var", "outputs/figures/xy_scatter.png", color_by="domain")

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
    export_grid_to_csv(grid_df, "outputs/tables/grid.csv")

    vario_params = cfg.get("variogram", {})
    exp = experimental_variogram_2d(df, "x", "y", "var", vario_params)
    model = fit_variogram_model(exp, float(df["var"].var(ddof=1)), model_type="spherical")
    plot_variogram(exp, model, "outputs/figures/variogram.png")
    save_variogram_model(model, "outputs/models/variogram_var.json")

    krig_cfg = cfg.get("kriging", {})
    kriged = ordinary_kriging_2d(df, "x", "y", "var", grid_df, model, krig_cfg)
    kriged.to_csv("outputs/tables/kriging_var.csv", index=False)

    plot_xy_scatter(kriged, "x", "y", "estimate", "outputs/figures/kriging_estimate.png")

    cv_df, metrics = simple_cross_validation(
        df,
        "x",
        "y",
        "var",
        radius=float(krig_cfg.get("search_radius", 150.0)),
        max_samples=int(krig_cfg.get("max_samples", 12)),
    )
    pd.DataFrame([metrics]).to_csv("outputs/tables/validation_metrics.csv", index=False)
    cv_df[["var", "pred"]].to_csv("outputs/tables/validation_predictions.csv", index=False)

    logging.info("Pipeline completed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.pipeline config/project.json")
    run_pipeline(sys.argv[1])
