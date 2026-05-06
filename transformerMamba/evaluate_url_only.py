"""Evaluate a pretrained URL-only detector checkpoint on a test set."""

from __future__ import annotations

import argparse
import logging

from config import PhishingConfig
from dataset import build_dataloader, load_records, load_vocab_for_runtime
from loss import build_criterion
from models import URLOnlyDetector
from utils import evaluate, get_device, load_checkpoint, save_json, setup_logging

logger = logging.getLogger(__name__)
__test__ = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved URL-only phishing detector checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_path", type=str, default=PhishingConfig.test_path)
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model_url_only.pt")
    parser.add_argument("--output_file", type=str, default="eval_results/url_only_test_metrics.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("logs", "evaluate_url_only")
    logger.info(
        "Starting URL-only evaluation | model_path=%s | test_path=%s | output_file=%s | device=%s",
        args.model_path,
        args.test_path,
        args.output_file,
        args.device,
    )
    device = get_device(args.device)

    checkpoint = load_checkpoint(args.model_path, device)
    config = PhishingConfig.from_dict(checkpoint["config"])
    config.use_traffic = False
    config.test_path = args.test_path
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)

    model = URLOnlyDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = build_criterion(config)
    records = load_records(config.test_path)
    vocabs = load_vocab_for_runtime(config, checkpoint.get("vocabs"))
    logger.info(
        "loaded vocab sizes | 1gram=%d | 2gram=%d | 3gram=%d",
        len(vocabs["1gram"]),
        len(vocabs["2gram"]),
        len(vocabs["3gram"]),
    )
    logger.info("test_samples=%d batch_size=%d", len(records), config.batch_size)

    dataloader = build_dataloader(records, config, vocabs, shuffle=False)
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("metrics", {}).get("threshold", 0.5))
    metrics = evaluate(
        model,
        dataloader,
        device=device,
        criterion=criterion,
        threshold=threshold,
        threshold_metric=config.threshold_metric,
    )
    serializable_metrics = {key: value for key, value in metrics.items() if key not in {"labels", "probs", "preds"}}
    save_json(args.output_file, serializable_metrics)
    logger.info("saved evaluation metrics to %s", args.output_file)
    logger.info(
        "evaluation summary | accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | auc=%.4f | threshold=%.2f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc"],
        metrics["threshold"],
    )
    print(
        f"accuracy={metrics['accuracy']:.4f}\n"
        f"precision={metrics['precision']:.4f}\n"
        f"recall={metrics['recall']:.4f}\n"
        f"f1={metrics['f1']:.4f}\n"
        f"auc={metrics['auc']:.4f}\n"
        f"threshold={metrics['threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
