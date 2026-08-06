"""Date-grouped ResNet50+I training entry point.

This first version intentionally enables only a bounded Fold 1 smoke test.
Formal cross-validation training will be added and run in a later step.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet50_with_i import SolarResNet50WithI
from utils.date_grouped_training_data import build_fold_datasets


DEFAULT_MANIFEST = PROJECT_ROOT / "splits" / "date_grouped_v1" / "split_manifest.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "PanelImages"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "date_grouped_v1" / "smoke_test" / "fold_1"
)
EXPECTED_FOLD_1_COUNTS = (19805, 5911)
MAX_SMOKE_BATCHES = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free date-grouped ResNet50+I training entry point"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unpack_batch(batch):
    if len(batch) < 3:
        raise ValueError("Expected at least image, loss label, and irradiance")
    return batch[0], batch[1], batch[2]


def train_smoke_batches(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
) -> tuple[float, int]:
    model.train()
    losses = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= MAX_SMOKE_BATCHES:
            break
        images, labels, irradiance = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
        irradiance = irradiance.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=device.type == "cuda",
        ):
            predictions = model(images, irradiance)
            loss = criterion(predictions, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))

    if not losses:
        raise RuntimeError("Smoke training loader produced no batches")
    return float(np.mean(losses)), len(losses)


@torch.no_grad()
def validate_smoke_batches(
    model,
    loader,
    criterion,
    device,
) -> tuple[float, int]:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= MAX_SMOKE_BATCHES:
            break
        images, labels, irradiance = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
        irradiance = irradiance.to(device, non_blocking=True).float()

        with torch.amp.autocast(
            device_type="cuda",
            enabled=device.type == "cuda",
        ):
            predictions = model(images, irradiance)
            loss = criterion(predictions, labels)
        losses.append(float(loss.detach().cpu()))

    if not losses:
        raise RuntimeError("Smoke validation loader produced no batches")
    return float(np.mean(losses)), len(losses)


def run_smoke_test(args: argparse.Namespace) -> dict:
    if args.fold != 1:
        raise ValueError("Smoke-test mode is fixed to Fold 1; use --fold 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    set_reproducible_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset, validation_dataset, train_records, validation_records = (
        build_fold_datasets(
            manifest_path=args.manifest,
            image_root=args.image_root,
            fold=args.fold,
            train_transform=train_transform,
            validation_transform=validation_transform,
        )
    )
    full_train_count = len(train_records)
    full_validation_count = len(validation_records)
    if (full_train_count, full_validation_count) != EXPECTED_FOLD_1_COUNTS:
        raise RuntimeError(
            "Fold 1 count mismatch: expected "
            f"{EXPECTED_FOLD_1_COUNTS}, got "
            f"{(full_train_count, full_validation_count)}"
        )

    smoke_train_count = min(len(train_dataset), args.batch_size * MAX_SMOKE_BATCHES)
    smoke_validation_count = min(
        len(validation_dataset), args.batch_size * MAX_SMOKE_BATCHES
    )
    smoke_train_dataset = Subset(train_dataset, range(smoke_train_count))
    smoke_validation_dataset = Subset(
        validation_dataset, range(smoke_validation_count)
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        smoke_train_dataset,
        shuffle=True,
        generator=loader_generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        smoke_validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    pretrained_used = not args.no_pretrained
    model = SolarResNet50WithI(
        dropout=0.3,
        use_pretrained=pretrained_used,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    train_loss, train_batches_run = train_smoke_batches(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
    )
    validation_loss, validation_batches_run = validate_smoke_batches(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    gpu_name = None
    gpu_total_memory_bytes = 0
    peak_memory_allocated_bytes = 0
    peak_memory_reserved_bytes = 0
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
        peak_memory_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_reserved_bytes = torch.cuda.max_memory_reserved(device)

    summary = {
        "split_version": "date_grouped_v1",
        "fold": args.fold,
        "full_train_count": full_train_count,
        "full_validation_count": full_validation_count,
        "smoke_train_count": smoke_train_count,
        "smoke_validation_count": smoke_validation_count,
        "train_batches_run": train_batches_run,
        "validation_batches_run": validation_batches_run,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
        "peak_memory_reserved_bytes": peak_memory_reserved_bytes,
        "seed": args.seed,
        "pretrained_used": pretrained_used,
        "NOT_FOR_RESEARCH_METRICS": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "smoke_test_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("NOT_FOR_RESEARCH_METRICS: true")
    print(f"Fold: {args.fold}")
    print(f"Full train/validation: {full_train_count}/{full_validation_count}")
    print(f"Smoke train/validation: {smoke_train_count}/{smoke_validation_count}")
    print(f"Batches run: train={train_batches_run}, validation={validation_batches_run}")
    print(f"Loss: train={train_loss:.8f}, validation={validation_loss:.8f}")
    print(f"Device: {device}; GPU: {gpu_name}")
    print(f"Pretrained used: {pretrained_used}")
    print(f"Summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    if not args.smoke_test:
        raise SystemExit(
            "Formal date-grouped training is intentionally disabled in this step. "
            "Use --smoke-test for the bounded Fold 1 pipeline check."
        )
    run_smoke_test(args)


if __name__ == "__main__":
    main()
