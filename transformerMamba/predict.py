"""Model prediction entrypoint."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List

import torch

from config import PhishingConfig
from dataset import PhishingDataset, collate_fn, load_records, load_vocab_for_runtime
from models import PhishingDetector
from utils import (
    apply_binary_threshold,
    extract_binary_probabilities,
    get_device,
    load_checkpoint,
    save_json,
    setup_logging,
)

logger = logging.getLogger(__name__)

PHISH_TYPE_NAMES = {
    0: "benign",
    1: "credential-phish",
    2: "malware",
    3: "banking-phish",
    4: "other-phish",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phishing predictions from a checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="predictions.json")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_model_bundle(
    model_path: str,
    device: torch.device,
) -> tuple[PhishingDetector, PhishingConfig, Dict[str, Dict[str, int]], float]:
    checkpoint = load_checkpoint(model_path, device)
    config = PhishingConfig.from_dict(checkpoint["config"])
    model = PhishingDetector(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(checkpoint.get("metrics", {}).get("threshold", 0.5))
    vocabs = load_vocab_for_runtime(config, checkpoint.get("vocabs"))
    logger.info(
        "loaded vocab sizes | 1gram=%d | 2gram=%d | 3gram=%d",
        len(vocabs["1gram"]),
        len(vocabs["2gram"]),
        len(vocabs["3gram"]),
    )
    return model, config, vocabs, threshold


def predict_records(
    model: PhishingDetector,
    config: PhishingConfig,
    vocabs: Dict[str, Dict[str, int]],
    records: List[Dict[str, Any]],
    device: torch.device,
    threshold: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    results = []
    effective_batch_size = max(int(batch_size), 1)

    for start in range(0, len(records), effective_batch_size):
        chunk_records = records[start : start + effective_batch_size]
        dataset = PhishingDataset(
            chunk_records,
            vocabs,
            config.max_url_len,
            config.max_traffic_len,
            lowercase_url=config.lowercase_url,
            require_targets=False,
            allow_missing_traffic=True,
        )
        batch = collate_fn([dataset[index] for index in range(len(dataset))])
        tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}

        with torch.no_grad():
            outputs = model(
                ids_1gram=tensor_batch["ids_1gram"],
                ids_2gram=tensor_batch["ids_2gram"],
                ids_3gram=tensor_batch["ids_3gram"],
                url_mask=tensor_batch["url_mask"],
                traffic_feats=tensor_batch["traffic_feats"],
                traffic_mask=tensor_batch["traffic_mask"],
            )

        probabilities = extract_binary_probabilities(outputs).cpu()
        predictions = apply_binary_threshold(probabilities, threshold).cpu().tolist()
        type_ids = outputs["type_logits"].argmax(dim=-1).cpu().tolist()
        risk_scores = outputs["risk_score"].cpu().tolist()

        for idx, record in enumerate(chunk_records):
            probability = float(probabilities[idx].item())
            results.append(
                {
                    "url": record["url"],
                    "probability": probability,
                    "is_phishing": bool(predictions[idx]),
                    "predicted_type_id": int(type_ids[idx]),
                    "predicted_type": PHISH_TYPE_NAMES.get(int(type_ids[idx]), "unknown"),
                    "risk_score": float(risk_scores[idx]),
                    "threshold": threshold,
                }
            )
    return results


def main() -> None:
    args = parse_args()
    setup_logging("logs", "predict")
    device = get_device(args.device)

    if bool(args.url) == bool(args.input_file):
        raise ValueError("Provide exactly one of --url or --input_file.")

    model, config, vocabs, default_threshold = load_model_bundle(args.model_path, device)
    threshold = args.threshold if args.threshold is not None else default_threshold
    batch_size = args.batch_size if args.batch_size is not None else config.predict_batch_size
    records = [{"url": args.url, "traffic": []}] if args.url else load_records(args.input_file)
    logger.info("Traffic input mode: %s", getattr(config, "traffic_input_mode", "raw_sequence"))
    logger.info("Using TrafficMambaEncoder=%s", getattr(config, "use_traffic", True))
    results = predict_records(model, config, vocabs, records, device, threshold, batch_size)

    if args.input_file:
        save_json(args.output_file, results)
        logger.info("saved %d predictions to %s", len(results), args.output_file)
    else:
        result = results[0]
        print(
            f"url={result['url']}\n"
            f"probability={result['probability']:.4f}\n"
            f"is_phishing={result['is_phishing']}\n"
            f"predicted_type={result['predicted_type']}\n"
            f"risk_score={result['risk_score']:.4f}\n"
            f"threshold={result['threshold']:.2f}"
        )


if __name__ == "__main__":
    main()
