"""Explicit Fold3/Fold4 launcher for ConvNeXt-Tiny L-balanced v1."""

from __future__ import annotations

from typing import Sequence

from experiments import train_convnext_tiny_image_only_lbalanced_date_grouped as training


def parse_args(argv: Sequence[str] | None = None):
    return training.parse_args(argv)


def validate_pilot_fold(fold: int) -> None:
    training.validate_pilot_fold(fold)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_pilot_fold(args.fold)
    training.run_training(args)


if __name__ == "__main__":
    main()
