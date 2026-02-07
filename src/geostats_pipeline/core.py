from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dirs(output_dir: str = "outputs") -> Dict[str, str]:
    paths = {
        "base": output_dir,
        "figures": os.path.join(output_dir, "figures"),
        "tables": os.path.join(output_dir, "tables"),
        "models": os.path.join(output_dir, "models"),
        "manifests": os.path.join(output_dir, "manifests"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def get_manifest_path(output_dir: str = "outputs", name: str = "manifest.jsonl") -> str:
    paths = ensure_output_dirs(output_dir)
    return os.path.join(paths["manifests"], name)


def register_manifest(
    manifest_path: str,
    step: str,
    artifacts: Iterable[str],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "step": step,
        "artifacts": list(artifacts),
        "metadata": metadata or {},
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
