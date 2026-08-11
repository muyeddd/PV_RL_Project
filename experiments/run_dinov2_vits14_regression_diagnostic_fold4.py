"""Explicit runner for the Fold4-only DINOv2 regression diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_dinov2_vits14_regression_diagnostic_fold4 import main


if __name__ == "__main__":
    raise SystemExit(main())
