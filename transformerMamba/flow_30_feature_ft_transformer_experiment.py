"""Independent FT-Transformer experiment on 30 flow-only features."""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from flow_30_feature_transformer import (
    FLOW_30_FEATURE_NAMES,
    PreparedSplit,
    build_scaler,
    clip_with_train_quantiles,
    compute_metrics,
    load_flow_samples,
    prepare_split,
    sanitize_feature_matrix,
    save_feature_table,
    setup_logging,
)
from flow_ft_transformer_model import FTTransformerConfig, FlowFTTransformer

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    train_path: str
    val_path: str
    test_path: str
    output_dir: str
    batch_size: int = 256
    lr: float = 1e-4
    epochs: int = 50
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 10
    seed: int = 42
    scaler_name: str = "standard"
    best_metric: str = "f1"
    device: str = "auto"
    d_token: int = 192
    n_blocks: int = 3
    attention_n_heads: int = 8
    attention_dropout: float = 0.2
    ffn_d_hidden: int = 256
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    head_dropout: float = 0.1


class FlowFeatureDataset(Dataset):
    """Simple tensor dataset for tabular flow features."""

    def __init__(self, split: PreparedSplit) -> None:
        self.features = torch.as_tensor(split.features, dtype=torch.float32)
        self.labels = torch.as_tensor(split.labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(split: PreparedSplit, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        FlowFeatureDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def find_best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = float(f1_score(labels, probabilities >= threshold, zero_division=0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def train_one_epoch(
    model: FlowFTTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    losses: list[float] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_split(
    model: FlowFTTransformer,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits)

        losses.append(float(loss.item()))
        labels_all.append(labels.cpu().numpy())
        probs_all.append(probabilities.cpu().numpy())

    labels_np = np.concatenate(labels_all, axis=0) if labels_all else np.asarray([], dtype=np.float32)
    probs_np = np.concatenate(probs_all, axis=0) if probs_all else np.asarray([], dtype=np.float32)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, labels_np.astype(np.int64), probs_np.astype(np.float32)


def build_optimizer(
    model: FlowFTTransformer,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            name.endswith("bias")
            or "norm" in name.lower()
            or "tokenizer" in name.lower()
            or "cls_token" in name.lower()
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def save_scaler(output_dir: str, scaler: Any) -> str:
    path = os.path.join(output_dir, "flow_30_ft_transformer_scaler.pkl")
    with open(path, "wb") as handle:
        pickle.dump(scaler, handle)
    return path


def save_training_log(output_dir: str, rows: list[dict[str, Any]]) -> str:
    path = os.path.join(output_dir, "train_log_ft_transformer.csv")
    if not rows:
        return path
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_checkpoint(
    path: str,
    model: FlowFTTransformer,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    scaler_path: str,
    threshold: float,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "feature_names": list(FLOW_30_FEATURE_NAMES),
        "scaler_path": scaler_path,
        "threshold": threshold,
        "metrics": metrics,
    }
    torch.save(payload, path)


def save_metrics(output_dir: str, metrics: dict[str, Any]) -> str:
    path = os.path.join(output_dir, "test_metrics_ft_transformer.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return path


def save_config(output_dir: str, config: ExperimentConfig) -> str:
    path = os.path.join(output_dir, "ft_transformer_experiment_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)
    return path


def prepare_scaled_splits(
    train_path: str,
    val_path: str,
    test_path: str,
    scaler_name: str,
    output_dir: str,
) -> tuple[PreparedSplit, PreparedSplit, PreparedSplit, str]:
    train_samples = load_flow_samples(train_path)
    val_samples = load_flow_samples(val_path)
    test_samples = load_flow_samples(test_path)

    train_split_raw = prepare_split(train_samples, "train")
    val_split_raw = prepare_split(val_samples, "val")
    test_split_raw = prepare_split(test_samples, "test")

    save_feature_table(output_dir, {"train": train_split_raw, "val": val_split_raw, "test": test_split_raw})

    train_features = sanitize_feature_matrix(train_split_raw.features)
    val_features = sanitize_feature_matrix(val_split_raw.features)
    test_features = sanitize_feature_matrix(test_split_raw.features)

    train_features = clip_with_train_quantiles(train_features, train_features)
    val_features = clip_with_train_quantiles(train_features, val_features)
    test_features = clip_with_train_quantiles(train_features, test_features)

    scaler = build_scaler(scaler_name)
    scaler.fit(train_features)
    scaler_path = save_scaler(output_dir, scaler)

    train_split = PreparedSplit(
        sample_ids=train_split_raw.sample_ids,
        labels=train_split_raw.labels,
        features=scaler.transform(train_features).astype(np.float32),
    )
    val_split = PreparedSplit(
        sample_ids=val_split_raw.sample_ids,
        labels=val_split_raw.labels,
        features=scaler.transform(val_features).astype(np.float32),
    )
    test_split = PreparedSplit(
        sample_ids=test_split_raw.sample_ids,
        labels=test_split_raw.labels,
        features=scaler.transform(test_features).astype(np.float32),
    )
    return train_split, val_split, test_split, scaler_path


def run_flow_30_feature_ft_transformer_experiment(**kwargs: Any) -> dict[str, Any]:
    config = ExperimentConfig(**kwargs)
    os.makedirs(config.output_dir, exist_ok=True)
    setup_logging(config.output_dir)
    _set_all_seeds(config.seed)
    device = _resolve_device(config.device)
    save_config(config.output_dir, config)
    logger.info("Using device=%s for 30-feature FT-Transformer experiment.", device)

    train_split, val_split, test_split, scaler_path = prepare_scaled_splits(
        train_path=config.train_path,
        val_path=config.val_path,
        test_path=config.test_path,
        scaler_name=config.scaler_name,
        output_dir=config.output_dir,
    )
    logger.info(
        "Prepared splits | train=%d val=%d test=%d features=%d",
        len(train_split.labels),
        len(val_split.labels),
        len(test_split.labels),
        train_split.features.shape[1],
    )

    train_loader = build_loader(train_split, config.batch_size, shuffle=True)
    val_loader = build_loader(val_split, config.batch_size, shuffle=False)
    test_loader = build_loader(test_split, config.batch_size, shuffle=False)

    model = FlowFTTransformer(
        FTTransformerConfig(
            n_num_features=len(FLOW_30_FEATURE_NAMES),
            d_token=config.d_token,
            n_blocks=config.n_blocks,
            attention_n_heads=config.attention_n_heads,
            attention_dropout=config.attention_dropout,
            ffn_d_hidden=config.ffn_d_hidden,
            ffn_dropout=config.ffn_dropout,
            residual_dropout=config.residual_dropout,
            head_dropout=config.head_dropout,
        )
    ).to(device)

    positive_count = float((train_split.labels == 1).sum())
    negative_count = float((train_split.labels == 0).sum())
    pos_weight = negative_count / max(positive_count, 1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = build_optimizer(model, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    history_rows: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_threshold = 0.5
    best_model_path = os.path.join(config.output_dir, "best_flow_30_feature_ft_transformer.pt")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=config.grad_clip,
        )
        val_loss, val_labels, val_probabilities = evaluate_split(model, val_loader, criterion, device)
        threshold, _ = find_best_threshold(val_labels, val_probabilities)
        val_metrics = compute_metrics(val_labels, val_probabilities, threshold=threshold)
        score = float(val_metrics["auc"] if config.best_metric.lower() == "auc" else val_metrics["f1"])

        history_rows.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "val_threshold": threshold,
            }
        )

        logger.info(
            "epoch=%d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f val_auc=%.4f threshold=%.2f",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["auc"],
            threshold,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            patience_counter = 0
            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                config=config,
                scaler_path=scaler_path,
                threshold=best_threshold,
                epoch=best_epoch,
                metrics={
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_auc": val_metrics["auc"],
                },
            )
        else:
            patience_counter += 1

        scheduler.step()

        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d.", epoch)
            break

    save_training_log(config.output_dir, history_rows)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_labels, test_probabilities = evaluate_split(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_probabilities, threshold=best_threshold)
    final_metrics = {
        "model_type": "ft_transformer",
        "feature_count": len(FLOW_30_FEATURE_NAMES),
        "best_metric": config.best_metric,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "model_path": best_model_path,
        "scaler_path": scaler_path,
        "test_loss": float(test_loss),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "auc": test_metrics["auc"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
    }
    save_metrics(config.output_dir, final_metrics)
    logger.info("Test metrics: %s", final_metrics)
    return final_metrics
