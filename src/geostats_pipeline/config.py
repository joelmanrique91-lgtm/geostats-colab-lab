from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

Number = (int, float)

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "path": "data_sample/sample_drillholes.csv",
        "x_col": "X",
        "y_col": "Y",
        "z_col": "Z",
        "value_col": "grade",
        "domain_col": "domain",
        "support": "point",
        "from_col": None,
        "to_col": None,
        "hole_id_col": None,
        "allow_pseudo_support": False,
        "pseudo_support_length": None,
    },
    "qaqc": {
        "duplicate_strategy": "drop",
        "outlier": {"enabled": False, "zscore": 4.0},
    },
    "declustering": {
        "enabled": True,
        "cell_size": {"x": 50.0, "y": 50.0, "z": 10.0},
        "use_3d": False,
    },
    "compositing": {
        "enabled": False,
        "target_length": 2.0,
        "min_length": 1.0,
    },
    "variography": {
        "n_lags": 12,
        "lag_size": 50.0,
        "tolerance": 0.5,
        "directions": [0.0, 45.0, 90.0, 135.0],
        "bandwidth": 9999.0,
        "model_type": "spherical",
        "initial_params": {"nugget": 0.0, "sill": None, "range": None},
        "max_pairs": 50000,
    },
    "anisotropy": {"enabled": False, "azimuths": [0, 30, 60, 90, 120, 150]},
    "block_model": {
        "auto_from_data": True,
        "pad": 0.0,
        "dx": 25.0,
        "dy": 25.0,
        "dz": 5.0,
        "nx": None,
        "ny": None,
        "nz": 1,
        "xmin": None,
        "ymin": None,
        "zmin": None,
    },
    "kriging": {
        "mode": "block",
        "block": {"dx": 25.0, "dy": 25.0, "dz": 5.0, "discretization": {"nx": 2, "ny": 2, "nz": 1}},
        "neighborhood": {
            "ranges": {"major": 150.0, "minor": 150.0, "vertical": 50.0},
            "angles": {"azimuth": 0.0, "dip": 0.0, "rake": 0.0},
            "min_samples": 4,
            "max_samples": 16,
            "max_per_hole": None,
            "octants": 8,
            "condition_max": 1.0e10,
        },
        "variogram_model": {
            "type": "spherical",
            "nugget": 0.0,
            "sill": None,
            "range": None,
        },
    },
    "validation": {
        "cv": "loo",
        "kfold_splits": 5,
        "metrics": ["ME", "RMSE", "MSE", "slope", "intercept", "MSDR"],
        "swath_bins": 10,
    },
    "simulation": {"enabled": False, "n_realizations": 25, "random_seed": 42},
    "outputs": {"base_dir": "outputs", "run_name": "auto"},
}

SCHEMA: Dict[str, Any] = {
    "data": {
        "path": (str,),
        "x_col": (str,),
        "y_col": (str,),
        "z_col": (str,),
        "value_col": (str,),
        "domain_col": (str, type(None)),
        "support": (str,),
        "from_col": (str, type(None)),
        "to_col": (str, type(None)),
        "hole_id_col": (str, type(None)),
        "allow_pseudo_support": (bool,),
        "pseudo_support_length": Number + (type(None),),
    },
    "qaqc": {"duplicate_strategy": (str,), "outlier": {"enabled": (bool,), "zscore": Number}},
    "declustering": {
        "enabled": (bool,),
        "cell_size": {"x": Number, "y": Number, "z": Number},
        "use_3d": (bool,),
    },
    "compositing": {"enabled": (bool,), "target_length": Number, "min_length": Number},
    "variography": {
        "n_lags": (int,),
        "lag_size": Number,
        "tolerance": Number,
        "directions": [Number],
        "bandwidth": Number,
        "model_type": (str,),
        "initial_params": {"nugget": Number + (type(None),), "sill": Number + (type(None),), "range": Number + (type(None),)},
        "max_pairs": (int,),
    },
    "anisotropy": {"enabled": (bool,), "azimuths": [Number]},
    "block_model": {
        "auto_from_data": (bool,),
        "pad": Number,
        "dx": Number,
        "dy": Number,
        "dz": Number,
        "nx": (int, type(None)),
        "ny": (int, type(None)),
        "nz": (int,),
        "xmin": Number + (type(None),),
        "ymin": Number + (type(None),),
        "zmin": Number + (type(None),),
    },
    "kriging": {
        "mode": (str,),
        "block": {
            "dx": Number,
            "dy": Number,
            "dz": Number,
            "discretization": {"nx": (int,), "ny": (int,), "nz": (int,)},
        },
        "neighborhood": {
            "ranges": {"major": Number, "minor": Number, "vertical": Number},
            "angles": {"azimuth": Number, "dip": Number, "rake": Number},
            "min_samples": (int,),
            "max_samples": (int,),
            "max_per_hole": (int, type(None)),
            "octants": (int,),
            "condition_max": Number,
        },
        "variogram_model": {"type": (str,), "nugget": Number, "sill": Number + (type(None),), "range": Number + (type(None),)},
    },
    "validation": {"cv": (str,), "kfold_splits": (int,), "metrics": [str], "swath_bins": (int,)},
    "simulation": {"enabled": (bool,), "n_realizations": (int,), "random_seed": (int,)},
    "outputs": {"base_dir": (str,), "run_name": (str,)},
}


def _deep_merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_schema(cfg: Mapping[str, Any], schema: Mapping[str, Any], prefix: str = "") -> None:
    for key, expected in schema.items():
        if key not in cfg:
            continue
        value = cfg[key]
        path = f"{prefix}{key}"
        if isinstance(expected, Mapping):
            if not isinstance(value, Mapping):
                raise TypeError(f"Config key '{path}' must be a mapping, got {type(value).__name__}")
            _validate_schema(value, expected, prefix=f"{path}.")
            continue

        if isinstance(expected, list):
            if len(expected) != 1:
                raise ValueError(f"Schema for '{path}' must have a single list item type definition")
            if not isinstance(value, list):
                raise TypeError(f"Config key '{path}' must be a list, got {type(value).__name__}")
            allowed = expected[0]
            for idx, item in enumerate(value):
                if not isinstance(item, allowed):
                    raise TypeError(
                        f"Config key '{path}[{idx}]' must be {allowed}, got {type(item).__name__}"
                    )
            continue

        if not isinstance(value, expected):
            raise TypeError(f"Config key '{path}' must be {expected}, got {type(value).__name__}")


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise TypeError("Config file must be a YAML mapping (dictionary).")

    cfg = _deep_merge(DEFAULT_CONFIG, data)
    _validate_schema(cfg, SCHEMA)

    support = cfg["data"]["support"]
    if support not in {"point", "interval"}:
        raise ValueError("data.support must be 'point' or 'interval'")

    return cfg


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
