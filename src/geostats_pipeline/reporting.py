from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping, Optional

DEFAULT_LIBRARIES = (
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "scikit-learn",
    "scikit-gstat",
    "pykrige",
)


@dataclass(frozen=True)
class RunPaths:
    base: Path
    figures: Path
    tables: Path
    models: Path
    logs: Path

    def figure_path(self, filename: str) -> Path:
        return self.figures / filename

    def table_path(self, filename: str) -> Path:
        return self.tables / filename

    def model_path(self, filename: str) -> Path:
        return self.models / filename

    def log_path(self, filename: str) -> Path:
        return self.logs / filename


def save_table(df, run_paths: RunPaths, filename: str, **kwargs) -> Path:
    path = run_paths.table_path(filename)
    df.to_csv(path, **kwargs)
    return path


def save_figure(fig, run_paths: RunPaths, filename: str, **kwargs) -> Path:
    path = run_paths.figure_path(filename)
    fig.savefig(path, **kwargs)
    return path


def create_run_dir(
    base_dir: str | Path = "outputs",
    prefix: str = "run",
    timestamp: Optional[datetime] = None,
) -> RunPaths:
    base_dir = Path(base_dir)
    ts = timestamp or datetime.now()
    run_name = f"{prefix}_{ts.strftime('%Y%m%d_%H%M')}"
    run_dir = base_dir / run_name

    if run_dir.exists():
        counter = 1
        while True:
            candidate = base_dir / f"{run_name}_{counter:02d}"
            if not candidate.exists():
                run_dir = candidate
                break
            counter += 1

    figures = run_dir / "figures"
    tables = run_dir / "tables"
    models = run_dir / "models"
    logs = run_dir / "logs"

    for path in (figures, tables, models, logs):
        path.mkdir(parents=True, exist_ok=True)

    return RunPaths(base=run_dir, figures=figures, tables=tables, models=models, logs=logs)


def _get_git_commit(repo_dir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _get_library_versions(libraries: Iterable[str]) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for lib in libraries:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def write_manifest(
    run_paths: RunPaths,
    config: Mapping[str, object],
    libraries: Iterable[str] = DEFAULT_LIBRARIES,
) -> tuple[Path, Optional[Path]]:
    now = datetime.now(timezone.utc)
    repo_dir = Path(__file__).resolve().parents[2]
    manifest = {
        "run_dir": str(run_paths.base),
        "created_at": now.isoformat(),
        "created_at_local": now.astimezone().isoformat(),
        "config": config,
        "git": {"commit": _get_git_commit(repo_dir)},
        "libraries": _get_library_versions(libraries),
    }

    json_path = run_paths.base / "manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    yaml_path: Optional[Path] = None
    try:
        import yaml

        yaml_path = run_paths.base / "manifest.yaml"
        yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        yaml_path = None

    return json_path, yaml_path
