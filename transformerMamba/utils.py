"""训练、评估和持久化过程共用的工具函数模块。"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """设置 Python、NumPy 与 PyTorch 的随机种子。

    Args:
        seed: 随机种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto") -> torch.device:
    """根据配置选择运行设备。

    Args:
        device_name: 设备名称，支持 ``auto``、``cpu`` 和 ``cuda``。

    Returns:
        torch.device: 实际使用的计算设备。
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def setup_logging(log_dir: str, run_name: str) -> None:
    """配置控制台与文件双通道日志输出。

    Args:
        log_dir: 日志目录。
        run_name: 当前运行名称，用于生成日志文件名。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """将批量字典中的张量移动到指定设备。

    Args:
        batch: 原始批量字典。
        device: 目标设备。

    Returns:
        Dict[str, Any]: 迁移后的批量字典。
    """
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def extract_binary_logits(outputs: Dict[str, Any]) -> torch.Tensor:
    """从模型输出字典中提取二分类 logit。

    Args:
        outputs: 模型前向输出字典。

    Returns:
        torch.Tensor: 二分类 logit 张量。

    Raises:
        KeyError: 当输出中不存在任何兼容的二分类字段时抛出。
    """
    for key in ("binary_logit", "logits", "main_logits"):
        if key in outputs:
            return outputs[key]
    raise KeyError("Model outputs do not contain a binary logit field.")


def extract_binary_probabilities(outputs: Dict[str, Any]) -> torch.Tensor:
    """从模型输出字典中提取或计算二分类概率。

    Args:
        outputs: 模型前向输出字典。

    Returns:
        torch.Tensor: 与 batch 对齐的概率张量。
    """
    if "binary_probability" in outputs:
        return outputs["binary_probability"]
    return torch.sigmoid(extract_binary_logits(outputs))


def apply_binary_threshold(probabilities: torch.Tensor | np.ndarray, threshold: float) -> torch.Tensor | np.ndarray:
    """基于统一阈值将概率转换为离散预测。

    Args:
        probabilities: 模型输出概率，可为张量或 NumPy 数组。
        threshold: 分类阈值。

    Returns:
        torch.Tensor | np.ndarray: 二值化后的预测结果。
    """
    return (probabilities >= threshold).long() if isinstance(probabilities, torch.Tensor) else (probabilities >= threshold).astype(np.int64)


def find_optimal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """在固定阈值网格上搜索最优分类阈值。

    Args:
        labels: 真实标签数组。
        probabilities: 模型输出的概率数组。
        metric: 目标优化指标。

    Returns:
        Tuple[float, float]: 最优阈值及其对应的指标分数。

    Raises:
        ValueError: 当传入的指标名称不受支持时抛出。
    """
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(labels, predictions, zero_division=0)
        elif metric == "precision":
            score = precision_score(labels, predictions, zero_division=0)
        elif metric == "recall":
            score = recall_score(labels, predictions, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    criterion=None,
    threshold: float | None = 0.5,
    threshold_metric: str = "f1",
) -> Dict[str, Any]:
    """运行验证或测试流程并汇总评估指标。

    Args:
        model: 待评估模型。
        dataloader: 评估数据加载器。
        device: 推理设备。
        criterion: 可选损失函数，用于同时计算平均损失。
        threshold: 固定分类阈值；若为 ``None`` 则自动搜索。
        threshold_metric: 自动搜索阈值时使用的指标。

    Returns:
        Dict[str, Any]: 包含损失、分类指标、阈值和原始预测数组的字典。
    """
    model.eval()
    losses = []
    labels = []
    probabilities = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            ids_1gram=batch["ids_1gram"],
            ids_2gram=batch["ids_2gram"],
            ids_3gram=batch["ids_3gram"],
            url_mask=batch["url_mask"],
            traffic_feats=batch["traffic_feats"],
            traffic_mask=batch["traffic_mask"],
        )
        if criterion is not None:
            losses.append(criterion(outputs, batch)["total"].item())
        probabilities.extend(extract_binary_probabilities(outputs).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())

    label_array = np.asarray(labels, dtype=np.int64)
    prob_array = np.asarray(probabilities, dtype=np.float32)

    if threshold is None:
        threshold, _ = find_optimal_threshold(label_array, prob_array, threshold_metric)

    prediction_array = apply_binary_threshold(prob_array, threshold)
    cm = confusion_matrix(label_array, prediction_array, labels=[0, 1])

    try:
        auc = roc_auc_score(label_array, prob_array)
    except ValueError:
        auc = 0.0

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(label_array, prediction_array),
        "precision": precision_score(label_array, prediction_array, zero_division=0),
        "recall": recall_score(label_array, prediction_array, zero_division=0),
        "f1": f1_score(label_array, prediction_array, zero_division=0),
        "auc": auc,
        "auc_pr": average_precision_score(label_array, prob_array),
        "threshold": float(threshold),
        "cm": cm.tolist(),
        "labels": label_array,
        "probs": prob_array,
        "preds": prediction_array,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    metrics: Dict[str, Any],
    vocabs: Dict[str, Dict[str, int]],
    epoch: int,
) -> None:
    """保存模型权重、优化器状态与词表信息。

    Args:
        path: checkpoint 输出路径。
        model: 训练中的模型实例。
        optimizer: 优化器实例。
        config: 当前运行配置对象。
        metrics: 当前评估指标字典。
        vocabs: URL n-gram 词表集合。
        epoch: 当前 epoch 编号。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "metrics": {key: value for key, value in metrics.items() if key not in {"labels", "probs", "preds"}},
        # URL vocabulary is already persisted to config.vocab_path.
        # Keeping it out of the .pt checkpoint avoids multi-GB checkpoint files.
    }
    temp_path = f"{path}.tmp"
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    except Exception as exc:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to save checkpoint to '{path}'. "
            "This is often caused by the target file being locked, the disk being full, or the existing checkpoint being too large."
        ) from exc


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> Dict[str, Any]:
    """从磁盘加载 checkpoint 文件。

    Args:
        path: checkpoint 路径。
        device: 状态字典加载到的目标设备。

    Returns:
        Dict[str, Any]: checkpoint 中保存的完整字典。
    """
    return torch.load(path, map_location=device, weights_only=False)


def save_json(path: str, payload: Dict[str, Any] | Iterable[Dict[str, Any]]) -> None:
    """将字典或字典列表保存为 JSON 文件。

    Args:
        path: 输出文件路径。
        payload: 待保存的数据结构。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
