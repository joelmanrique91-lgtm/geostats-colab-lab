from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

Number = (int, float)

DEFAULT_CONFIG: Dict[str, Any] = {
    "data_csv_path": "csv/Conminution.csv",
    "columns": {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "variable_objetivo": "Bwi_kWh_t",
        "domain": "Lito",
        "lithology": None,
        "alteration": None,
    },
    "nodata_values": ["", "NA", "NaN", -999, -99, -1.0e21],
    "topcut": {"enabled": False, "high": None},
    "decluster": {"enabled": False, "cell": 50.0},
    "compositing": {"enabled": False, "length": 2.0, "alonghole_col": "BHID"},
    "grid": {
        "auto_from_data": True,
        "pad": 0.0,
        "nx": None,
        "ny": None,
        "nz": 1,
        "xmin": None,
        "ymin": None,
        "zmin": None,
        "dx": 25.0,
        "dy": 25.0,
        "dz": 5.0,
    },
    "variogram": {
        "nlag": 12,
        "lag": 50.0,
        "azm": 0.0,
        "atol": 22.5,
        "bandwh": 9999.0,
        "dip": 0.0,
        "dtol": 22.5,
        "bandwd": 9999.0,
    },
    "kriging": {"type": "ok", "search_radius": 150.0, "min_samples": 4, "max_samples": 16},
}

REQUIRED_KEYS = {
    "data_csv_path": (str,),
    "columns": (Mapping,),
}

REQUIRED_COLUMNS = {
    "x": (str,),
    "y": (str,),
    "z": (str,),
    "variable_objetivo": (str,),
}

SCHEMA: Dict[str, Any] = {
    "data_csv_path": (str,),
    "columns": {
        "x": (str, type(None)),
        "y": (str, type(None)),
        "z": (str, type(None)),
        "variable_objetivo": (str, type(None)),
        "domain": (str, type(None)),
        "lithology": (str, type(None)),
        "alteration": (str, type(None)),
    },
    "nodata_values": [(str, int, float, type(None))],
    "topcut": {"enabled": (bool,), "high": (int, float, type(None))},
    "decluster": {"enabled": (bool,), "cell": Number},
    "compositing": {"enabled": (bool,), "length": Number, "alonghole_col": (str, type(None))},
    "grid": {
        "auto_from_data": (bool,),
        "pad": Number,
        "nx": (int, type(None)),
        "ny": (int, type(None)),
        "nz": (int,),
        "xmin": (int, float, type(None)),
        "ymin": (int, float, type(None)),
        "zmin": (int, float, type(None)),
        "dx": Number,
        "dy": Number,
        "dz": Number,
    },
    "variogram": {
        "nlag": (int,),
        "lag": Number,
        "azm": Number,
        "atol": Number,
        "bandwh": Number,
        "dip": Number,
        "dtol": Number,
        "bandwd": Number,
    },
    "kriging": {
        "type": (str,),
        "search_radius": Number,
        "min_samples": (int,),
        "max_samples": (int,),
    },
}


def _deep_merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_required(cfg: Mapping[str, Any]) -> None:
    for key, types in REQUIRED_KEYS.items():
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")
        if not isinstance(cfg[key], types):
            raise TypeError(f"Config key '{key}' must be {types}, got {type(cfg[key]).__name__}")

    columns = cfg["columns"]
    for key, types in REQUIRED_COLUMNS.items():
        if key not in columns:
            raise KeyError(f"Missing required column mapping: columns.{key}")
        if not isinstance(columns[key], types):
            raise TypeError(
                f"Config key 'columns.{key}' must be {types}, got {type(columns[key]).__name__}"
            )


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
    """Load YAML config, apply defaults, and validate required keys/types."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise TypeError("Config file must be a YAML mapping (dictionary).")

    cfg = _deep_merge(DEFAULT_CONFIG, data)
    _validate_required(cfg)
    _validate_schema(cfg, SCHEMA)
    return cfg


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
