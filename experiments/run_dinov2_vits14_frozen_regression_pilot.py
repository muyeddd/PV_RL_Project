"""Run one explicitly selected Fold 3/4 frozen-DINOv2 regression pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_dinov2_vits14_frozen_regression_date_grouped as training


ALLOWED_PILOT_FOLDS = (3, 4)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated frozen DINOv2-S/14 regression pilot (Fold 3 or Fold 4)"
    )
    parser.add_argument("--fold", type=int, choices=ALLOWED_PILOT_FOLDS, required=True)
    return parser.parse_args(argv)


def run_pilot(fold: int):
    training.validate_pilot_fold(fold)
    training.load_config()
    expected = training.expected_output_dir(fold, seed=42)
    if expected.parent.resolve() != training.PILOT_OUTPUT_ROOT.resolve():
        raise RuntimeError("Pilot output is not isolated")
    return training.run_training(fold)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_pilot(args.fold)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
