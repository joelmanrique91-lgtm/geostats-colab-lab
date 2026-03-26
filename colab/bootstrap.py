from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def run_cmd(cmd: list[str], cwd: str | Path | None = None) -> None:
    """Run shell command with clear output in notebook."""
    printable = " ".join(cmd)
    print(f"$ {printable}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def clone_or_update_repo(repo_url: str | None, repo_dir: str = "/content/geostats-colab-lab") -> Path:
    """Clone repo if missing; otherwise pull latest changes."""
    path = Path(repo_dir)
    if path.exists() and (path / ".git").exists():
        run_cmd(["git", "-C", str(path), "pull", "--ff-only"])
        return path

    if path.exists() and not (path / ".git").exists():
        raise RuntimeError(f"{path} exists but is not a git repo. Remove it or use another repo_dir.")

    if not repo_url:
        raise RuntimeError(
            "repo_url is required when repository is not already cloned in /content. "
            "Set REPO_URL in the notebook first."
        )

    run_cmd(["git", "clone", repo_url, str(path)])
    return path


def install_requirements(repo_root: Path) -> None:
    req = repo_root / "colab" / "requirements_colab.txt"
    if not req.exists():
        raise FileNotFoundError(f"Missing requirements file: {req}")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req)])


def ensure_src_import(repo_root: Path) -> Path:
    """Force import resolution to <repo>/src/geostats_pipeline."""
    src_path = (repo_root / "src").resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Missing src path: {src_path}")

    src_str = str(src_path)
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)

    if "geostats_pipeline" in sys.modules:
        del sys.modules["geostats_pipeline"]

    module = importlib.import_module("geostats_pipeline")
    module_file = Path(getattr(module, "__file__", "")).resolve()

    expected_prefix = src_path / "geostats_pipeline"
    if expected_prefix not in module_file.parents and module_file != expected_prefix:
        raise RuntimeError(
            "geostats_pipeline did not resolve from src/. "
            f"Imported from: {module_file}"
        )

    return module_file


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Config YAML must be a mapping/dictionary.")
    return data


def materialize_runtime_config(repo_root: Path) -> Path:
    """Create a runtime config with Colab-compatible absolute paths."""
    source_cfg = repo_root / "colab" / "config.colab.yml"
    cfg = load_yaml(source_cfg)

    data_path = cfg.get("data", {}).get("path", "")
    if not data_path:
        raise ValueError("config.colab.yml must define data.path")

    outputs_dir = cfg.get("outputs", {}).get("base_dir", "outputs")

    abs_data = (repo_root / data_path).resolve() if not os.path.isabs(data_path) else Path(data_path)
    abs_outputs = (repo_root / outputs_dir).resolve() if not os.path.isabs(outputs_dir) else Path(outputs_dir)

    cfg["data"]["path"] = str(abs_data)
    cfg["outputs"]["base_dir"] = str(abs_outputs)

    runtime_path = repo_root / "colab" / "config.colab.runtime.yml"
    runtime_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_path


def smoke_test_setup(runtime_config_path: Path) -> Path:
    from geostats_pipeline.steps import run_pipeline

    run_paths = run_pipeline(str(runtime_config_path), stage="setup")
    return run_paths.base
