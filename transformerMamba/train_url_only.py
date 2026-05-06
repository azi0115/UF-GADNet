"""URL-only training entrypoint."""

from __future__ import annotations

import logging
import os

import torch

from config import get_config
from dataset import build_dataloader, build_url_vocabs, extract_ngrams, load_records, save_url_vocabs
from loss import build_criterion
from models import URLOnlyDetector
from train import _extract_accuracy_metric, train_one_epoch
from utils import evaluate, get_device, save_checkpoint, set_seed, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    config = get_config()
    config.use_traffic = False
    setup_logging(config.log_dir, "train_url_only")
    set_seed(config.seed)
    device = get_device(config.device)

    train_records = load_records(config.train_path)
    val_records = load_records(config.val_path)
    logger.info("Training mode: url_only")

    vocabs = build_url_vocabs((record.get("url", "") for record in train_records), config)
    save_url_vocabs(vocabs, config.vocab_path)
    vocab_metadata = vocabs.get("__metadata__", {})

    logger.info(
        "position-aware vocabulary enabled=%s | ngram_range=%s | include_boundary_tokens=%s | lowercase_url=%s",
        getattr(config, "use_position_ngram_vocab", False),
        getattr(config, "ngram_range", (1, 3)),
        getattr(config, "include_boundary_tokens", True),
        config.lowercase_url,
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
    else:
        for example_url in [record["url"] for record in train_records[:3]]:
            logger.info(
                "plain tokenization example | url=%s | tokens=%s",
                example_url,
                extract_ngrams(example_url, 3, lowercase_url=config.lowercase_url)[:10],
            )

    train_loader = build_dataloader(train_records, config, vocabs, shuffle=True)
    val_loader = build_dataloader(val_records, config, vocabs, shuffle=False)

    model = URLOnlyDetector(config).to(device)
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

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                path=os.path.join(config.checkpoint_dir, "best_model_url_only.pt"),
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
                path=os.path.join(config.checkpoint_dir, "last_model_url_only.pt"),
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

    logger.info("url_only best_auc=%.4f best_epoch=%d", best_auc, best_epoch)


if __name__ == "__main__":
    main()
