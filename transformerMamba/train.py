"""模型训练入口脚本。"""

from __future__ import annotations

import logging
import os
from typing import Dict

import torch
from tqdm import tqdm

from config import get_config
from dataset import (
    build_dataloader,
    build_url_vocabs,
    extract_ngrams,
    load_records,
    save_url_vocabs,
)
from loss import build_criterion
from models import PhishingDetector
from utils import (
    apply_binary_threshold,
    evaluate,
    extract_binary_probabilities,
    get_device,
    move_batch_to_device,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _extract_accuracy_metric(metrics: Dict[str, float], default: float = 0.0) -> float:
    """从指标字典中提取可用的准确率字段。

    Args:
        metrics: 指标字典。
        default: 未命中任何候选字段时使用的默认值。

    Returns:
        float: 提取到的准确率数值。
    """
    for key in ("accuracy", "acc", "binary_acc", "val_acc", "train_acc"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(default)


def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip: float) -> Dict[str, float]:
    """执行单个 epoch 的训练流程。

    Args:
        model: 待训练模型。
        dataloader: 训练数据加载器。
        optimizer: 优化器实例。
        criterion: 多任务损失函数。
        device: 当前训练设备。
        grad_clip: 梯度裁剪阈值。

    Returns:
        Dict[str, float]: 当前 epoch 各项损失的平均值。
    """
    model.train()
    totals = {"main": 0.0, "type": 0.0, "risk": 0.0, "total": 0.0}
    steps = 0
    correct = 0
    total_samples = 0

    progress = tqdm(
        dataloader,
        desc=f"Epoch {getattr(train_one_epoch, 'current_epoch', '?')}/{getattr(train_one_epoch, 'total_epochs', '?')}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            ids_1gram=batch["ids_1gram"],
            ids_2gram=batch["ids_2gram"],
            ids_3gram=batch["ids_3gram"],
            url_mask=batch["url_mask"],
            traffic_feats=batch["traffic_feats"],
            traffic_mask=batch["traffic_mask"],
        )
        loss_dict = criterion(outputs, batch)
        loss_dict["total"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        steps += 1
        for key in totals:
            totals[key] += loss_dict[key].item()

        probabilities = extract_binary_probabilities(outputs).detach()
        predictions = apply_binary_threshold(probabilities, 0.5)
        labels = batch["label"]
        correct += int((predictions == labels).sum().item())
        total_samples += int(labels.numel())

        avg_loss = totals["total"] / max(steps, 1)
        avg_acc = correct / max(total_samples, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    train_one_epoch.last_accuracy = correct / max(total_samples, 1)
    return {key: value / max(steps, 1) for key, value in totals.items()}


def main() -> None:
    """执行完整训练主流程。

    该流程依次完成配置加载、日志初始化、URL 词表构建、数据集装载、
    模型与优化器创建、训练循环、验证评估以及 checkpoint 保存。
    """
    config = get_config()
    setup_logging(config.log_dir, "train")
    set_seed(config.seed)
    device = get_device(config.device)

    train_records = load_records(config.train_path)
    val_records = load_records(config.val_path)
    logger.info("Traffic input mode: %s", getattr(config, "traffic_input_mode", "raw_sequence"))
    logger.info("Using TrafficMambaEncoder=%s", getattr(config, "use_traffic", True))
    # URL 词表严格基于训练集构建，避免验证集信息泄露。
    vocabs = build_url_vocabs((record.get("url", "") for record in train_records), config)
    save_url_vocabs(vocabs, config.vocab_path)
    vocab_metadata = vocabs.get("__metadata__", {})

    logger.info(
        "position-aware vocabulary enabled=%s | ngram_range=%s | include_boundary_tokens=%s | lowercase_url=%s | use_traffic=%s",
        getattr(config, "use_position_ngram_vocab", False),
        getattr(config, "ngram_range", (1, 3)),
        getattr(config, "include_boundary_tokens", True),
        config.lowercase_url,
        getattr(config, "use_traffic", True),
    )
    logger.info(
        "vocab sizes | 1gram=%d | 2gram=%d | 3gram=%d",
        len(vocabs["1gram"]),
        len(vocabs["2gram"]),
        len(vocabs["3gram"]),
    )
    if getattr(config, "use_position_ngram_vocab", False):
        logger.info("position class counts: %s", vocab_metadata.get("position_class_frequency", {}))
        logger.info("n-gram length counts: %s", vocab_metadata.get("ngram_length_frequency", {}))
        for sample in vocab_metadata.get("sample_examples", [])[:3]:
            logger.info("position-aware tokenization example | url=%s | tokens=%s", sample["url"], sample["tokens"])
    else:
        for example_url in [record["url"] for record in train_records[:3]]:
            logger.info(
                "plain tokenization example | url=%s | tokens=%s",
                example_url,
                extract_ngrams(example_url, 3, lowercase_url=config.lowercase_url)[:10],
            )

    train_loader = build_dataloader(train_records, config, vocabs, shuffle=True)
    val_loader = build_dataloader(val_records, config, vocabs, shuffle=False)

    model = PhishingDetector(config).to(device)
    criterion = build_criterion(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_auc = -1.0
    best_epoch = 0
    patience_counter = 0

    logger.info("device=%s train_samples=%d val_samples=%d", device, len(train_records), len(val_records))

    for epoch in range(1, config.epochs + 1):
        epoch_label = f"Epoch {epoch}/{config.epochs}"
        train_one_epoch.current_epoch = epoch
        train_one_epoch.total_epochs = config.epochs
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, config.grad_clip)
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            threshold=None,
            threshold_metric=config.threshold_metric,
        )
        scheduler.step()

        train_acc = float(getattr(train_one_epoch, "last_accuracy", 0.0))
        val_acc = _extract_accuracy_metric(val_metrics)
        logger.info(
            "%s | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f | val_auc=%.4f | threshold=%.2f",
            epoch_label,
            train_metrics["total"],
            train_acc,
            val_metrics["loss"],
            val_acc,
            val_metrics["f1"],
            val_metrics["auc"],
            val_metrics["threshold"],
        )

        logger.info(
            "%s validation accuracy source: %s=%.4f",
            epoch_label,
            next(
                (key for key in ("accuracy", "acc", "binary_acc", "val_acc", "train_acc") if key in val_metrics),
                "default",
            ),
            val_acc,
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                path=os.path.join(config.checkpoint_dir, "best_model.pt"),
                model=model,
                optimizer=optimizer,
                config=config,
                metrics=val_metrics,
                vocabs=vocabs,
                epoch=epoch,
            )
        else:
            patience_counter += 1

        if config.save_last_checkpoint:
            save_checkpoint(
                path=os.path.join(config.checkpoint_dir, "last_model.pt"),
                model=model,
                optimizer=optimizer,
                config=config,
                metrics=val_metrics,
                vocabs=vocabs,
                epoch=epoch,
            )

        if patience_counter >= config.patience:
            logger.info("early stopping at epoch %d", epoch)
            break

    logger.info("best_auc=%.4f best_epoch=%d", best_auc, best_epoch)


if __name__ == "__main__":
    main()
