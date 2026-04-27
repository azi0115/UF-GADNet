"""模型评估入口脚本。"""

from __future__ import annotations

import argparse
import logging

from config import PhishingConfig
from dataset import build_dataloader, load_records, load_vocab_for_runtime
from loss import build_criterion
from models import PhishingDetector
from utils import evaluate, get_device, load_checkpoint, save_json, setup_logging

logger = logging.getLogger(__name__)
__test__ = False


def parse_args() -> argparse.Namespace:
    """解析命令行评估参数。

    Returns:
        argparse.Namespace: 评估阶段所需参数集合。
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a saved phishing detector checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_path", type=str, default=PhishingConfig.test_path)
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output_file", type=str, default="eval_results/test_metrics.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    """执行测试集评估主流程。"""
    args = parse_args()
    setup_logging("logs", "evaluate_test")
    device = get_device(args.device)

    checkpoint = load_checkpoint(args.model_path, device)
    config = PhishingConfig.from_dict(checkpoint["config"])
    config.test_path = args.test_path

    model = PhishingDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = build_criterion(config)
    records = load_records(args.test_path)
    if getattr(config, "use_position_ngram_vocab", False):
        logger.info(
            "Position-aware vocabulary is enabled | ngram_range=%s | include_boundary_tokens=%s",
            config.ngram_range,
            config.include_boundary_tokens,
        )
    vocabs = load_vocab_for_runtime(config, checkpoint.get("vocabs"))
    logger.info(
        "loaded vocab sizes | 1gram=%d | 2gram=%d | 3gram=%d",
        len(vocabs["1gram"]),
        len(vocabs["2gram"]),
        len(vocabs["3gram"]),
    )
    metadata = vocabs.get("__metadata__", {})
    if getattr(config, "use_position_ngram_vocab", False):
        logger.info("position class counts: %s", metadata.get("position_class_frequency", {}))
        logger.info("n-gram length counts: %s", metadata.get("ngram_length_frequency", {}))
        for sample in metadata.get("sample_examples", [])[:3]:
            logger.info("position-aware tokenization example | url=%s | tokens=%s", sample["url"], sample["tokens"])
    dataloader = build_dataloader(records, config, vocabs, shuffle=False)

    # 默认沿用 checkpoint 中保存的阈值，也支持命令行显式覆盖。
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("metrics", {}).get("threshold", 0.5))
    metrics = evaluate(
        model,
        dataloader,
        device=device,
        criterion=criterion,
        threshold=threshold,
        threshold_metric=config.threshold_metric,
    )
    save_json(args.output_file, {key: value for key, value in metrics.items() if key not in {"labels", "probs", "preds"}})
    logger.info("saved evaluation metrics to %s", args.output_file)
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
