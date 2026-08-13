"""Train ResNet50+irradiance on Paper1 Clean Random v1 development roles only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "paper1_clean_random_v1"
TRAIN_MANIFEST = SPLIT_ROOT / "train.csv"
VALIDATION_MANIFEST = SPLIT_ROOT / "model_validation.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper1_clean_random_v1" / "resnet50_with_i_v1" / "seed_42"
LEGACY_CHECKPOINT = PROJECT_ROOT / "outputs" / "models_ckpt" / "best_resnet50_with_i.pth"
ALLOWED_MANIFEST_NAMES = {"train.csv": "TRAIN", "model_validation.csv": "MODEL_VALIDATION"}
FORBIDDEN_MANIFEST_NAMES = {"random_test.csv", "cp_calibration.csv", "decision_development.csv"}
SEALED_DATES = {"2017-06-15", "2017-06-24", "2017-06-30"}
EXPECTED_N = {"TRAIN": 25830, "MODEL_VALIDATION": 3692}
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
NUM_WORKERS = 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREDICTION_FIELDS = ("sample_id", "date", "timestamp", "true_L", "pred_L", "residual", "regime")
TRUE_I_PATTERN = re.compile(r"_L_([0-9eE+.-]+)_I_([0-9eE+.-]+)$")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")


def reject_legacy_checkpoint(path: Path | None) -> None:
    if path is not None:
        raise ValueError("External/legacy checkpoints are forbidden for this training path")


def parse_label_and_irradiance(image_path: str) -> tuple[float, float]:
    match = TRUE_I_PATTERN.search(Path(image_path).stem)
    if match is None:
        raise ValueError(f"Cannot parse true_L/I from development locator: {image_path}")
    return float(match.group(1)), float(match.group(2))


def load_role_manifest(path: Path, expected_role: str) -> pd.DataFrame:
    path = Path(path)
    if path.name in FORBIDDEN_MANIFEST_NAMES:
        raise PermissionError(f"Forbidden manifest role: {path.name}")
    if ALLOWED_MANIFEST_NAMES.get(path.name) != expected_role:
        raise PermissionError(f"Manifest {path.name} is not authorized as {expected_role}")
    frame = pd.read_csv(path)
    required = {"sample_id", "image_path", "date", "timestamp", "role"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty or len(frame) != EXPECTED_N[expected_role]:
        raise ValueError(f"{expected_role} N guard failed: {len(frame)}")
    if set(frame["role"]) != {expected_role}:
        raise PermissionError(f"Role guard failed for {expected_role}")
    if set(frame["date"]) & SEALED_DATES:
        raise PermissionError("Sealed final date rejected")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be non-null and unique")
    if frame["image_path"].isna().any() or frame["image_path"].duplicated().any():
        raise ValueError("image_path must be non-null and unique")
    return frame.loc[:, ["sample_id", "image_path", "date", "timestamp", "role"]].copy()


def validate_role_isolation(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if set(train["sample_id"]) & set(validation["sample_id"]):
        raise ValueError("TRAIN/MODEL_VALIDATION sample_id overlap")
    if set(train["image_path"]) & set(validation["image_path"]):
        raise ValueError("TRAIN/MODEL_VALIDATION image_path overlap")
    if set(train["role"]) != {"TRAIN"} or set(validation["role"]) != {"MODEL_VALIDATION"}:
        raise PermissionError("Only TRAIN and MODEL_VALIDATION are authorized")


def attach_development_values(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    parsed = [parse_label_and_irradiance(value) for value in result["image_path"]]
    result["true_L"] = [value[0] for value in parsed]
    result["irradiance_raw"] = [value[1] for value in parsed]
    return result


def compute_train_irradiance_stats(train: pd.DataFrame) -> dict[str, float | int | str]:
    if set(train["role"]) != {"TRAIN"}:
        raise PermissionError("Irradiance statistics may use TRAIN only")
    values = train["irradiance_raw"].to_numpy(dtype=np.float64)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid TRAIN irradiance standard deviation")
    return {
        "N": len(values), "mean": float(values.mean()), "std_ddof0": std,
        "min": float(values.min()), "max": float(values.max()),
        "normalization": "z_score", "source_role": "TRAIN",
    }


def normalize_irradiance(frame: pd.DataFrame, stats: Mapping[str, float]) -> pd.DataFrame:
    result = frame.copy()
    result["irradiance"] = (
        result["irradiance_raw"].astype(float) - float(stats["mean"])
    ) / float(stats["std_ddof0"])
    return result


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, validation_transform


def validation_transform_is_deterministic(transform: transforms.Compose) -> bool:
    random_types = (
        transforms.RandomResizedCrop, transforms.RandomHorizontalFlip,
        transforms.RandomRotation, transforms.ColorJitter,
    )
    return not any(isinstance(item, random_types) for item in transform.transforms)


class ManifestDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: Any):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image_path = PROJECT_ROOT / Path(row["image_path"])
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        image = self.transform(image)
        return (
            image,
            torch.tensor(float(row["irradiance"]), dtype=torch.float32),
            torch.tensor(float(row["true_L"]), dtype=torch.float32),
            index,
        )


class Paper1ResNet50WithI(nn.Module):
    """Historical fusion/head geometry with explicit ImageNet-only initialization."""
    def __init__(self, weights: ResNet50_Weights, dropout: float = DROPOUT):
        super().__init__()
        if weights is not ResNet50_Weights.IMAGENET1K_V2:
            raise ValueError("This protocol requires ResNet50_Weights.IMAGENET1K_V2")
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.i_branch = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(16, 16), nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features + 16, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, irradiance: torch.Tensor) -> torch.Tensor:
        image_feature = self.backbone(image)
        irradiance_feature = self.i_branch(irradiance.unsqueeze(1))
        return torch.sigmoid(self.regressor(torch.cat([image_feature, irradiance_feature], dim=1)))


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    value = pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    residual = y_pred - y_true
    return {
        "N": len(y_true), "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(residual.mean()),
        "Pearson": safe_corr(y_true, y_pred, "pearson"),
        "Spearman": safe_corr(y_true, y_pred, "spearman"),
        "true_mean": float(y_true.mean()), "true_std": float(y_true.std(ddof=0)),
        "pred_mean": float(y_pred.mean()), "pred_std": float(y_pred.std(ddof=0)),
        "true_min": float(y_true.min()), "true_max": float(y_true.max()),
        "pred_min": float(y_pred.min()), "pred_max": float(y_pred.max()),
    }


def regime_name(value: float) -> str:
    return "LOW" if value < 0.1 else "MEDIUM" if value < 0.5 else "HIGH"


def regime_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    result = []
    regimes = {
        "LOW": y_true < 0.1,
        "MEDIUM": (y_true >= 0.1) & (y_true < 0.5),
        "HIGH": y_true >= 0.5,
    }
    for name, mask in regimes.items():
        true = y_true[mask]; pred = y_pred[mask]; residual = pred - true
        result.append({
            "regime": name, "N": len(true),
            "R2": float(r2_score(true, pred)) if len(true) >= 2 and np.std(true) > 1e-12 else None,
            "RMSE": float(mean_squared_error(true, pred) ** 0.5),
            "MAE": float(mean_absolute_error(true, pred)),
            "bias": float(residual.mean()),
            "underprediction_rate": float(np.mean(pred < true)),
        })
    return result


def autocast_context(device: torch.device):
    return torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); total_squared = 0.0; count = 0; truths = []; predictions = []; indices = []
    with torch.inference_mode():
        for images, irradiance, targets, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            irradiance = irradiance.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device):
                outputs = model(images, irradiance).squeeze(1)
            total_squared += torch.sum((outputs.float() - targets.float()) ** 2).item()
            count += len(targets)
            truths.append(targets.cpu().numpy()); predictions.append(outputs.float().cpu().numpy())
            indices.append(batch_indices.numpy())
    return math.sqrt(total_squared / count), np.concatenate(truths), np.concatenate(predictions), np.concatenate(indices)


def save_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def plot_outputs(output_dir: Path, history: pd.DataFrame, predictions: pd.DataFrame, regimes: Sequence[Mapping[str, Any]]) -> None:
    true = predictions["true_L"].to_numpy(); pred = predictions["pred_L"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(true, pred, s=9, alpha=.35)
    lo=min(true.min(),pred.min()); hi=max(true.max(),pred.max()); ax.plot([lo,hi],[lo,hi],"k--")
    ax.set(xlabel="True L", ylabel="Predicted L", title="MODEL_VALIDATION: True vs Predicted")
    fig.tight_layout(); fig.savefig(output_dir/"validation_true_vs_pred.png",dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(pred-true,bins=60); ax.axvline(0,color="k",ls="--")
    ax.set(xlabel="Residual (pred - true)",ylabel="N",title="MODEL_VALIDATION residual distribution")
    fig.tight_layout(); fig.savefig(output_dir/"validation_residual_distribution.png",dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(history["epoch"],history["train_mse"],label="TRAIN MSE")
    ax.plot(history["epoch"],history["validation_mse"],label="MODEL_VALIDATION MSE")
    ax.set(xlabel="Epoch",ylabel="MSE",title="Training and validation loss"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir/"training_validation_loss.png",dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); names=[r["regime"] for r in regimes]
    ax.bar(names,[r["MAE"] for r in regimes],label="MAE"); ax.plot(names,[r["RMSE"] for r in regimes],"o-",label="RMSE")
    ax.set(ylabel="Error",title="MODEL_VALIDATION regime error"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir/"validation_regime_error.png",dpi=300); plt.close(fig)


def run(output_dir: Path = OUTPUT_DIR, checkpoint: Path | None = None) -> dict[str, Any]:
    reject_legacy_checkpoint(checkpoint)
    ensure_output_available(output_dir)
    set_seed(SEED)
    train_manifest = load_role_manifest(TRAIN_MANIFEST, "TRAIN")
    validation_manifest = load_role_manifest(VALIDATION_MANIFEST, "MODEL_VALIDATION")
    validate_role_isolation(train_manifest, validation_manifest)
    train = attach_development_values(train_manifest)
    validation = attach_development_values(validation_manifest)
    irradiance_stats = compute_train_irradiance_stats(train)
    train = normalize_irradiance(train, irradiance_stats)
    validation = normalize_irradiance(validation, irradiance_stats)
    train_transform, validation_transform = build_transforms()
    if not validation_transform_is_deterministic(validation_transform):
        raise ValueError("MODEL_VALIDATION preprocessing is not deterministic")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this formal training run")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = Paper1ResNet50WithI(weights=weights, dropout=DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(ManifestDataset(train, train_transform), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, generator=generator)
    validation_loader = DataLoader(ManifestDataset(validation, validation_transform), batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "paper1_clean_random_v1", "architecture": "ResNet50+irradiance",
        "seed": SEED, "training_role": "TRAIN", "selection_role": "MODEL_VALIDATION",
        "selection_metric": "MODEL_VALIDATION RMSE", "train_N": len(train),
        "model_validation_N": len(validation), "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS, "early_stopping_patience": PATIENCE,
        "optimizer": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=2)",
        "loss": "MSELoss", "dropout": DROPOUT, "amp": True,
        "initialization": "ImageNet pretrained", "pretrained_source": "torchvision",
        "pretrained_weight_enum": str(weights), "legacy_checkpoint_loaded": False,
        "irradiance_normalization": "TRAIN-only z-score",
        "image_preprocessing": {"train": repr(train_transform), "validation": repr(validation_transform)},
    }
    save_json_exclusive(output_dir / "config.json", config)
    save_json_exclusive(output_dir / "train_irradiance_stats.json", irradiance_stats)

    started = time.perf_counter(); best_rmse = float("inf"); best_epoch = 0; no_improve = 0; history=[]
    best_path = output_dir / "best_model.pth"
    for epoch in range(1, EPOCHS + 1):
        model.train(); sum_squared=0.0; count=0
        for images, irradiance, targets, _ in train_loader:
            images=images.to(device,non_blocking=True); irradiance=irradiance.to(device,non_blocking=True)
            targets=targets.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                outputs=model(images,irradiance).squeeze(1); loss=criterion(outputs,targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            sum_squared += torch.sum((outputs.detach().float()-targets.float())**2).item(); count += len(targets)
        train_mse=sum_squared/count
        val_rmse,_,_,_=evaluate(model,validation_loader,device); val_mse=val_rmse**2
        scheduler.step(val_rmse); lr_now=optimizer.param_groups[0]["lr"]
        improved=val_rmse < best_rmse
        history.append({"epoch":epoch,"train_mse":train_mse,"validation_mse":val_mse,
                        "validation_rmse":val_rmse,"learning_rate":lr_now,"selected":improved})
        print(f"epoch={epoch:02d} train_mse={train_mse:.8f} val_rmse={val_rmse:.8f} lr={lr_now:.2e}",flush=True)
        if improved:
            best_rmse=val_rmse; best_epoch=epoch; no_improve=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,
                        "validation_rmse":val_rmse,"config":config},best_path)
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    duration=time.perf_counter()-started
    history_frame=pd.DataFrame(history); history_frame.to_csv(output_dir/"training_history.csv",index=False)
    checkpoint_data=torch.load(best_path,map_location=device,weights_only=False)
    if "model_state_dict" not in checkpoint_data or checkpoint_data["epoch"] != best_epoch:
        raise ValueError("Generated best checkpoint verification failed")
    model.load_state_dict(checkpoint_data["model_state_dict"])
    val_rmse,y_true,y_pred,indices=evaluate(model,validation_loader,device)
    order=np.argsort(indices); y_true=y_true[order]; y_pred=y_pred[order]
    if not np.array_equal(indices[order],np.arange(len(validation))):
        raise ValueError("Validation prediction order mismatch")
    prediction_frame=validation.loc[:,["sample_id","date","timestamp"]].copy()
    prediction_frame["true_L"]=y_true; prediction_frame["pred_L"]=y_pred
    prediction_frame["residual"]=y_pred-y_true
    prediction_frame["regime"]=[regime_name(value) for value in y_true]
    prediction_frame.to_csv(output_dir/"model_validation_predictions.csv",index=False)
    overall=metrics(y_true,y_pred); regimes=regime_metrics(y_true,y_pred)
    output_metrics={"selection_metric":"MODEL_VALIDATION RMSE","best_epoch":best_epoch,
                    "training_duration_seconds":duration,"overall":overall,"by_regime":regimes}
    save_json_exclusive(output_dir/"model_validation_metrics.json",output_metrics)
    plot_outputs(output_dir,history_frame,prediction_frame,regimes)
    provenance={
        "protocol":"paper1_clean_random_v1","training_role":"TRAIN",
        "selection_role":"MODEL_VALIDATION","random_test_accessed":False,
        "random_test_truth_accessed":False,"random_test_predictions_generated":False,
        "cp_calibration_accessed":False,"decision_development_accessed":False,
        "sealed_final_dates_accessed":False,"legacy_checkpoint_loaded":False,
        "legacy_checkpoint_path_rejected":str(LEGACY_CHECKPOINT),
        "initialization":"ImageNet pretrained","pretrained_source":"torchvision",
        "pretrained_weight_enum":str(weights),"training_performed":True,"seed":SEED,
        "hyperparameter_search_performed":False,"cp_performed":False,
        "mc_dropout_analysis_performed":False,"random_test_metric_computation":False,
        "train_manifest":str(TRAIN_MANIFEST.relative_to(PROJECT_ROOT)).replace(os.sep,"/"),
        "model_validation_manifest":str(VALIDATION_MANIFEST.relative_to(PROJECT_ROOT)).replace(os.sep,"/"),
        "training_duration_seconds":duration,"best_epoch":best_epoch,
    }
    save_json_exclusive(output_dir/"provenance.json",provenance)
    return {"metrics":output_metrics,"provenance":provenance}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    parser.add_argument("--checkpoint",type=Path,default=None,help="Forbidden; guard-test only")
    args=parser.parse_args(); result=run(args.output_dir,args.checkpoint)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
