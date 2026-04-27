"""Project-wide configuration definitions and CLI parsing."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class PhishingConfig:
    """Shared configuration used by train, evaluation, and prediction."""

    train_path: str = "data/b_data/data/split1/train.pkl"
    val_path: str = "data/b_data/data/split1/valid.pkl"
    test_path: str = "data/b_data/data/split1/test.pkl"

    ngram_min_freq: int = 1
    use_position_ngram_vocab: bool = True
    ngram_range: Tuple[int, int] = (1, 3)
    position_granularity: str = "fine"
    include_boundary_tokens: bool = True
    save_position_vocab_meta: bool = True
    lowercase_url: bool = False

    url_embed_dim: int = 96
    url_num_heads: int = 4
    url_num_layers: int = 2
    url_ffn_dim: int = 384

    traffic_input_dim: int = 2
    traffic_input_mode: str = "raw_sequence"
    num_phish_types: int = 5

    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    vocab_path: str = "checkpoints/url_vocabs.json"
    threshold_metric: str = "f1"
    save_last_checkpoint: bool = True
    predict_batch_size: int = 32
    use_traffic: bool = True

    max_url_len: int = 512
    max_traffic_len: int = 1024
    vocab_1gram_max_size: int = 10000
    vocab_2gram_max_size: int = 1000000
    vocab_3gram_max_size: int = 1000000

    traffic_embed_dim: int = 96
    traffic_num_layers: int = 2
    traffic_expand_factor: int = 2
    traffic_kernel_size: int = 5

    fusion_dim: int = 96
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 20

    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    loss_beta: float = 0.1
    loss_gamma: float = 0.03

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhishingConfig":
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        return cls(**filtered)


def build_parser() -> argparse.ArgumentParser:
    """Build the shared command-line parser."""
    parser = argparse.ArgumentParser(
        description="Phishing detector configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_path", type=str, default=PhishingConfig.train_path)
    parser.add_argument("--val_path", type=str, default=PhishingConfig.val_path)
    parser.add_argument("--test_path", type=str, default=PhishingConfig.test_path)

    parser.add_argument("--max_url_len", type=int, default=PhishingConfig.max_url_len)
    parser.add_argument("--max_traffic_len", type=int, default=PhishingConfig.max_traffic_len)
    parser.add_argument("--vocab_1gram_max_size", type=int, default=PhishingConfig.vocab_1gram_max_size)
    parser.add_argument("--vocab_2gram_max_size", type=int, default=PhishingConfig.vocab_2gram_max_size)
    parser.add_argument("--vocab_3gram_max_size", type=int, default=PhishingConfig.vocab_3gram_max_size)
    parser.add_argument("--ngram_min_freq", type=int, default=PhishingConfig.ngram_min_freq)

    parser.add_argument(
        "--use_position_ngram_vocab",
        action=argparse.BooleanOptionalAction,
        default=PhishingConfig.use_position_ngram_vocab,
    )
    parser.add_argument(
        "--ngram_range",
        type=int,
        nargs=2,
        metavar=("MIN_N", "MAX_N"),
        default=PhishingConfig.ngram_range,
    )
    parser.add_argument("--position_granularity", type=str, default=PhishingConfig.position_granularity)
    parser.add_argument(
        "--include_boundary_tokens",
        action=argparse.BooleanOptionalAction,
        default=PhishingConfig.include_boundary_tokens,
    )
    parser.add_argument(
        "--save_position_vocab_meta",
        action=argparse.BooleanOptionalAction,
        default=PhishingConfig.save_position_vocab_meta,
    )
    parser.add_argument("--lowercase_url", action=argparse.BooleanOptionalAction, default=PhishingConfig.lowercase_url)

    parser.add_argument("--url_embed_dim", type=int, default=PhishingConfig.url_embed_dim)
    parser.add_argument("--url_num_heads", type=int, default=PhishingConfig.url_num_heads)
    parser.add_argument("--url_num_layers", type=int, default=PhishingConfig.url_num_layers)
    parser.add_argument("--url_ffn_dim", type=int, default=PhishingConfig.url_ffn_dim)

    parser.add_argument("--traffic_input_dim", type=int, default=PhishingConfig.traffic_input_dim)
    parser.add_argument(
        "--traffic_input_mode",
        type=str,
        choices=["raw_sequence"],
        default=PhishingConfig.traffic_input_mode,
    )
    parser.add_argument("--traffic_embed_dim", type=int, default=PhishingConfig.traffic_embed_dim)
    parser.add_argument("--traffic_num_layers", type=int, default=PhishingConfig.traffic_num_layers)
    parser.add_argument("--traffic_expand_factor", type=int, default=PhishingConfig.traffic_expand_factor)
    parser.add_argument("--traffic_kernel_size", type=int, default=PhishingConfig.traffic_kernel_size)

    parser.add_argument("--fusion_dim", type=int, default=PhishingConfig.fusion_dim)
    parser.add_argument("--num_phish_types", type=int, default=PhishingConfig.num_phish_types)
    parser.add_argument("--dropout", type=float, default=PhishingConfig.dropout)
    parser.add_argument("--batch_size", type=int, default=PhishingConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=PhishingConfig.epochs)
    parser.add_argument("--lr", type=float, default=PhishingConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=PhishingConfig.weight_decay)
    parser.add_argument("--grad_clip", type=float, default=PhishingConfig.grad_clip)
    parser.add_argument("--patience", type=int, default=PhishingConfig.patience)

    parser.add_argument("--focal_alpha", type=float, default=PhishingConfig.focal_alpha)
    parser.add_argument("--focal_gamma", type=float, default=PhishingConfig.focal_gamma)
    parser.add_argument("--loss_beta", type=float, default=PhishingConfig.loss_beta)
    parser.add_argument("--loss_gamma", type=float, default=PhishingConfig.loss_gamma)

    parser.add_argument("--num_workers", type=int, default=PhishingConfig.num_workers)
    parser.add_argument("--seed", type=int, default=PhishingConfig.seed)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=PhishingConfig.device)
    parser.add_argument("--checkpoint_dir", type=str, default=PhishingConfig.checkpoint_dir)
    parser.add_argument("--log_dir", type=str, default=PhishingConfig.log_dir)
    parser.add_argument("--vocab_path", type=str, default=PhishingConfig.vocab_path)
    parser.add_argument(
        "--threshold_metric",
        type=str,
        choices=["f1", "precision", "recall"],
        default=PhishingConfig.threshold_metric,
    )
    parser.add_argument(
        "--save_last_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=PhishingConfig.save_last_checkpoint,
    )
    parser.add_argument("--predict_batch_size", type=int, default=PhishingConfig.predict_batch_size)
    parser.add_argument("--use_traffic", action=argparse.BooleanOptionalAction, default=PhishingConfig.use_traffic)
    return parser


def get_config(argv: list[str] | None = None) -> PhishingConfig:
    """Parse CLI arguments into a normalized config object."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    config = PhishingConfig.from_dict(vars(namespace))
    config.ngram_range = tuple(int(value) for value in config.ngram_range)
    return config
