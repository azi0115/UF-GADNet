"""Training launcher for the 30-feature flow FT-Transformer experiment."""

from __future__ import annotations

import argparse
import json
import os

from flow_30_feature_ft_transformer_experiment import run_flow_30_feature_ft_transformer_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an FT-Transformer on 30 flow-only features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_path", type=str, default="data/ubuntu/split1/train.pkl")
    parser.add_argument("--val_path", type=str, default="data/ubuntu/split1/valid.pkl")
    parser.add_argument("--test_path", type=str, default="data/ubuntu/split1/test.pkl")
    parser.add_argument("--output_dir", type=str, default="outputs/flow_30_feature_ft_transformer")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaler", type=str, choices=["standard", "robust"], default="standard")
    parser.add_argument("--best_metric", type=str, choices=["f1", "auc"], default="f1")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--d_token", type=int, default=192)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--attention_n_heads", type=int, default=8)
    parser.add_argument("--attention_dropout", type=float, default=0.2)
    parser.add_argument("--ffn_d_hidden", type=int, default=256)
    parser.add_argument("--ffn_dropout", type=float, default=0.1)
    parser.add_argument("--residual_dropout", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = run_flow_30_feature_ft_transformer_experiment(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
        scaler_name=args.scaler,
        best_metric=args.best_metric,
        device=args.device,
        d_token=args.d_token,
        n_blocks=args.n_blocks,
        attention_n_heads=args.attention_n_heads,
        attention_dropout=args.attention_dropout,
        ffn_d_hidden=args.ffn_d_hidden,
        ffn_dropout=args.ffn_dropout,
        residual_dropout=args.residual_dropout,
        head_dropout=args.head_dropout,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
