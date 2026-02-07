from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class ReportingPaths:
    base_dir: Path = Path("outputs")
    figures: Path = field(init=False)
    tables: Path = field(init=False)
    models: Path = field(init=False)
    logs: Path = field(init=False)
    manifest: Path = field(init=False)

    def __post_init__(self) -> None:
        self.figures = self.base_dir / "figures"
        self.tables = self.base_dir / "tables"
        self.models = self.base_dir / "models"
        self.logs = self.base_dir / "logs"
        self.manifest = self.base_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        for path in [self.figures, self.tables, self.models, self.logs]:
            path.mkdir(parents=True, exist_ok=True)

    def log_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.logs / f"pipeline_{timestamp}.log"


@dataclass
class Manifest:
    config_path: str
    stages: List[str]
    dry_run: bool
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    outputs: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, stage: str, output_path: Path) -> None:
        self.outputs.setdefault(stage, []).append(str(output_path))

    def write(self, path: Path) -> None:
        payload = {
            "config": self.config_path,
            "stages": self.stages,
            "dry_run": self.dry_run,
            "created_at": self.created_at,
            "outputs": self.outputs,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
