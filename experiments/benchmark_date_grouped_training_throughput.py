"""Benchmark Fold 1 training throughput without producing research metrics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_resnet50_with_i_date_grouped import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MANIFEST,
    EXPECTED_FOLD_COUNTS,
    build_transforms,
    load_and_validate_dataset_fingerprint,
    make_data_loader_generator,
    seed_data_loader_worker,
    set_random_seeds,
    unpack_batch,
)
from models.resnet50_with_i import SolarResNet50WithI
from utils.dataset import SolarDataset
from utils.date_grouped_training_data import (
    FORBIDDEN_MODEL_ROLES,
    MODEL_DEVELOPMENT_ROLE,
    load_fold_records,
)


NOT_FOR_RESEARCH_METRICS = True
FOLD = 1
SEED = 42
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
AMP_REQUESTED = True
PRETRAINED_REQUESTED = True
WARMUP_BATCHES = 10
TIMED_BATCHES = 100
BATCH_SIZES = (8, 16, 32)
NUM_WORKERS = (0, 2, 4)
MEMORY_PREFERENCE_LIMIT = 0.85
THROUGHPUT_CLOSE_FRACTION = 0.03
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "date_grouped_v1" / "throughput_benchmark"
)
CSV_FIELDS = (
    "batch_size",
    "num_workers",
    "batches_completed",
    "elapsed_seconds",
    "samples_per_second",
    "seconds_per_batch",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "oom",
    "device",
    "pretrained_used",
    "NOT_FOR_RESEARCH_METRICS",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark date_grouped_v1 Fold 1 training throughput"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing benchmark CSV/JSON files in the output directory.",
    )
    return parser.parse_args(argv)


def prepare_output_paths(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir = Path(output_dir).resolve()
    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_summary.json"
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Benchmark outputs already exist ({names}); pass --overwrite to replace them"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, json_path


def build_train_dataset(
    manifest_path: Path, image_root: Path
) -> tuple[SolarDataset, int, list[str], dict[str, Any]]:
    fingerprint = load_and_validate_dataset_fingerprint(manifest_path)
    train_records, validation_records = load_fold_records(
        manifest_path=manifest_path,
        image_root=image_root,
        fold=FOLD,
    )
    expected_train, expected_validation = EXPECTED_FOLD_COUNTS[FOLD]
    if (len(train_records), len(validation_records)) != (
        expected_train,
        expected_validation,
    ):
        raise ValueError(
            "Fold 1 count mismatch: "
            f"expected {expected_train}/{expected_validation}, got "
            f"{len(train_records)}/{len(validation_records)}"
        )
    if set(train_records["top_level_role"]) != {MODEL_DEVELOPMENT_ROLE}:
        raise ValueError("Benchmark training records contain a forbidden top-level role")
    forbidden_counts = {
        role: int(train_records["top_level_role"].eq(role).sum())
        for role in FORBIDDEN_MODEL_ROLES
    }
    if any(forbidden_counts.values()):
        raise ValueError(f"Forbidden roles entered benchmark data: {forbidden_counts}")
    if train_records["cv_validation_fold"].astype(int).eq(FOLD).any():
        raise ValueError("Fold 1 validation records entered the benchmark training data")

    train_transform, _ = build_transforms()
    dataset = SolarDataset(str(Path(image_root).resolve()), transform=train_transform)
    dataset.files = train_records["filename"].tolist()
    train_dates = sorted(train_records["date"].unique().tolist())
    return dataset, len(validation_records), train_dates, fingerprint


def pretrained_cache_details() -> dict[str, Any]:
    weights = ResNet50_Weights.DEFAULT
    checkpoint_name = Path(urlparse(weights.url).path).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    return {
        "requested": PRETRAINED_REQUESTED,
        "source": f"torchvision:{weights}",
        "cache_path": str(cache_path),
        "cache_present_before_benchmark": cache_path.is_file(),
    }


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
        and "cuda" in str(error).lower()
    )


def train_one_batch(
    batch: Any,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    images, labels, irradiance = unpack_batch(batch)
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
    irradiance = irradiance.to(device, non_blocking=True).float()

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", enabled=True):
        predictions = model(images, irradiance)
        loss = criterion(predictions, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return int(labels.shape[0])


def peak_memory(device: torch.device) -> tuple[int, int]:
    try:
        return (
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.max_memory_reserved(device)),
        )
    except RuntimeError:
        return 0, 0


def shutdown_loader(iterator: Any, loader: Any) -> None:
    if iterator is not None:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
    del iterator
    del loader
    gc.collect()


def benchmark_configuration(
    dataset: SolarDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    set_random_seeds(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    loader = None
    iterator = None
    model = None
    optimizer = None
    scaler = None
    batches_completed = 0
    samples_completed = 0
    elapsed_seconds = 0.0
    timed_start: float | None = None
    pretrained_used = False
    oom = False

    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_data_loader_worker,
            generator=make_data_loader_generator(SEED),
            persistent_workers=num_workers > 0,
        )
        iterator = iter(loader)

        model = SolarResNet50WithI(dropout=0.3, use_pretrained=True).to(device)
        pretrained_used = True
        criterion = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        model.train()

        for _ in range(WARMUP_BATCHES):
            train_one_batch(
                next(iterator), model, criterion, optimizer, scaler, device
            )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        timed_start = time.perf_counter()
        for _ in range(TIMED_BATCHES):
            samples_completed += train_one_batch(
                next(iterator), model, criterion, optimizer, scaler, device
            )
            batches_completed += 1
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - timed_start
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        oom = True
        if timed_start is not None:
            try:
                torch.cuda.synchronize(device)
            except RuntimeError:
                pass
            elapsed_seconds = time.perf_counter() - timed_start

    peak_allocated, peak_reserved = peak_memory(device)
    samples_per_second = (
        samples_completed / elapsed_seconds if elapsed_seconds > 0 else None
    )
    seconds_per_batch = (
        elapsed_seconds / batches_completed if batches_completed > 0 else None
    )
    result = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "batches_completed": batches_completed,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": samples_per_second,
        "seconds_per_batch": seconds_per_batch,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "oom": oom,
        "device": str(device),
        "pretrained_used": pretrained_used,
        "NOT_FOR_RESEARCH_METRICS": NOT_FOR_RESEARCH_METRICS,
    }

    shutdown_loader(iterator, loader)
    del model
    del optimizer
    del scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def json_ready_result(result: dict[str, Any], total_memory: int) -> dict[str, Any]:
    value = dict(result)
    value["gpu_peak_allocated_fraction"] = (
        value["gpu_peak_allocated_bytes"] / total_memory if total_memory else None
    )
    value["gpu_peak_reserved_fraction"] = (
        value["gpu_peak_reserved_bytes"] / total_memory if total_memory else None
    )
    return value


def select_recommendation(
    results: list[dict[str, Any]], total_memory: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    successful = [
        row
        for row in results
        if not row["oom"] and row["batches_completed"] == TIMED_BATCHES
    ]
    if not successful:
        raise RuntimeError("No benchmark configuration completed successfully")
    fastest = max(successful, key=lambda row: row["samples_per_second"])
    within_memory_limit = [
        row
        for row in successful
        if row["gpu_peak_reserved_bytes"] <= MEMORY_PREFERENCE_LIMIT * total_memory
    ]
    candidates = within_memory_limit or successful
    candidate_fastest = max(row["samples_per_second"] for row in candidates)
    close_candidates = [
        row
        for row in candidates
        if row["samples_per_second"]
        >= candidate_fastest * (1.0 - THROUGHPUT_CLOSE_FRACTION)
    ]
    recommended = min(
        close_candidates,
        key=lambda row: (
            row["gpu_peak_reserved_bytes"],
            -row["samples_per_second"],
            row["batch_size"],
            row["num_workers"],
        ),
    )
    baseline = next(
        row
        for row in results
        if row["batch_size"] == 8 and row["num_workers"] == 0
    )
    return fastest, recommended, baseline


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RTX 3050 throughput benchmark")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    csv_path, json_path = prepare_output_paths(args.output_dir, args.overwrite)
    cache_details = pretrained_cache_details()
    if not cache_details["cache_present_before_benchmark"]:
        raise FileNotFoundError(
            "The torchvision ResNet50 DEFAULT weight cache is absent; refusing to "
            "download during this controlled benchmark"
        )

    dataset, validation_count, train_dates, fingerprint = build_train_dataset(
        Path(args.manifest).resolve(), Path(args.image_root).resolve()
    )
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    started_at = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []

    print(f"NOT_FOR_RESEARCH_METRICS = {str(NOT_FOR_RESEARCH_METRICS).lower()}")
    print(
        f"device={device} gpu={torch.cuda.get_device_name(device)} "
        f"train_samples={len(dataset)} fold={FOLD}",
        flush=True,
    )
    for batch_size in BATCH_SIZES:
        for num_workers in NUM_WORKERS:
            print(
                f"Starting batch_size={batch_size}, num_workers={num_workers}",
                flush=True,
            )
            result = benchmark_configuration(
                dataset, batch_size, num_workers, device
            )
            results.append(result)
            write_csv(csv_path, results)
            throughput = result["samples_per_second"]
            throughput_text = (
                f"{throughput:.3f}" if throughput is not None else "n/a"
            )
            print(
                f"Completed batch_size={batch_size}, num_workers={num_workers}: "
                f"oom={result['oom']} batches={result['batches_completed']} "
                f"samples/s={throughput_text}",
                flush=True,
            )

    fastest, recommended, baseline = select_recommendation(results, total_memory)
    speedup_ratio = (
        recommended["samples_per_second"] / baseline["samples_per_second"]
        if baseline["samples_per_second"]
        else math.nan
    )
    summary = {
        "NOT_FOR_RESEARCH_METRICS": NOT_FOR_RESEARCH_METRICS,
        "benchmark_kind": "training_throughput_only",
        "split_version": fingerprint["split_version"],
        "fold": FOLD,
        "data_scope": "Fold 1 model_development training records only",
        "train_count": len(dataset),
        "validation_count_not_used": validation_count,
        "train_dates": train_dates,
        "forbidden_roles_used": [],
        "warmup_batches_per_configuration": WARMUP_BATCHES,
        "timed_batches_per_configuration": TIMED_BATCHES,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "amp_requested": AMP_REQUESTED,
        "amp_enabled": True,
        "pretrained": cache_details,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": total_memory,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_batch_sizes": list(BATCH_SIZES),
        "candidate_num_workers": list(NUM_WORKERS),
        "memory_preference_limit_fraction": MEMORY_PREFERENCE_LIMIT,
        "throughput_close_fraction": THROUGHPUT_CLOSE_FRACTION,
        "results": [json_ready_result(row, total_memory) for row in results],
        "oom_configurations": [
            {
                "batch_size": row["batch_size"],
                "num_workers": row["num_workers"],
            }
            for row in results
            if row["oom"]
        ],
        "fastest_configuration": json_ready_result(fastest, total_memory),
        "recommended_configuration": json_ready_result(recommended, total_memory),
        "baseline_configuration": json_ready_result(baseline, total_memory),
        "recommended_vs_baseline_speedup_ratio": speedup_ratio,
        "recommended_vs_baseline_percent_faster": (speedup_ratio - 1.0) * 100.0,
        "recommendation_rule": (
            "Require successful completion; prefer peak reserved CUDA memory <=85% "
            "of total; among configurations within 3% of the best eligible throughput, "
            "choose the lowest peak reserved memory."
        ),
        "output_files": [csv_path.name, json_path.name],
        "checkpoints_read": [],
        "checkpoints_written": [],
        "research_metrics_computed": [],
    }
    write_json(json_path, summary)
    print(
        "Recommendation: "
        f"batch_size={recommended['batch_size']}, "
        f"num_workers={recommended['num_workers']}, "
        f"speedup={speedup_ratio:.3f}x",
        flush=True,
    )
    return summary


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
