"""Independent training launcher for the flow 30-feature LightGBM experiment."""

from __future__ import annotations

import argparse
import json
import os

from flow_30_feature_transformer import run_flow_30_feature_transformer_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 30-feature flow-only LightGBM classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_path", type=str, default="data/ubuntu/split1/train.pkl")
    parser.add_argument("--val_path", type=str, default="data/ubuntu/split1/valid.pkl")
    parser.add_argument("--test_path", type=str, default="data/ubuntu/split1/test.pkl")
    parser.add_argument("--output_dir", type=str, default="outputs/flow_30_feature_transformer")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaler", type=str, choices=["standard", "robust"], default="standard")
    parser.add_argument("--best_metric", type=str, choices=["f1", "auc"], default="f1")
    parser.add_argument("--refit_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overfit_100_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_estimators", type=int, default=1000)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = run_flow_30_feature_transformer_experiment(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        scaler_name=args.scaler,
        best_metric=args.best_metric,
        refit_features=args.refit_features,
        overfit_100_debug=args.overfit_100_debug,
        num_estimators=args.num_estimators,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
