from __future__ import annotations

import argparse

from .steps import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geostats pipeline stages.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "setup",
            "qaqc",
            "compositing",
            "eda",
            "variography",
            "anisotropy",
            "blockmodel",
            "kriging",
            "validation",
            "simulation",
            "report",
        ],
        help="Pipeline stage to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, stage=args.stage)


if __name__ == "__main__":
    main()
