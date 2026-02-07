from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_csv(path: str, n: int = 300) -> str:
    """Generate a synthetic drillhole points CSV."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 1000, n)
    z = rng.uniform(0, 100, n)

    trend = 0.5 + 0.002 * x + 0.001 * y + 0.3 * np.sin(x / 120.0)
    noise = rng.normal(0, 0.15, n)
    cu = np.maximum(trend + noise, 0.01)

    domain = np.where(x >= 500, "D2", "D1")

    df = pd.DataFrame({"X": x, "Y": y, "Z": z, "Cu": cu, "Domain": domain})

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return os.path.abspath(path)


if __name__ == "__main__":
    make_demo_csv(os.path.join("csv", "demo_points.csv"))
