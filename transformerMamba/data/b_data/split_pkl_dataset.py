"""
split_pkl_dataset.py

功能：
把一个已经生成好的 pkl 数据集按比例划分为 train.pkl、valid.pkl、test.pkl。

输入 pkl 要求：
反序列化后是 List[Dict]，每条样本至少包含：
url, traffic, label, phish_type, risk_score

默认划分比例：
train : valid : test = 8 : 1 : 1

推荐用法：
python split_pkl_dataset.py --input data/train10w.pkl --output_dir data/b_data/data/split1

指定比例：
python split_pkl_dataset.py --input data/train10w.pkl --output_dir data/b_data/data/split1 --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1

按 label 分层划分，保持正常/钓鱼比例尽量一致：
python split_pkl_dataset.py --input train2w.pkl --output_dir data/split1 --stratify_label
"""

from __future__ import annotations

import argparse
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REQUIRED_KEYS = {"url", "traffic", "label", "phish_type", "risk_score"}


def load_pkl(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise TypeError(f"PKL top-level object must be list, got {type(data).__name__}")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"record[{i}] must be dict, got {type(item).__name__}")

    return data


def validate_records(records: List[Dict[str, Any]], strict: bool = True) -> None:
    if not records:
        raise ValueError("Input dataset is empty.")

    for i, item in enumerate(records):
        missing = REQUIRED_KEYS - set(item.keys())
        if missing:
            raise ValueError(f"record[{i}] missing required keys: {sorted(missing)}")

        if strict:
            url = item.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f"record[{i}] url must be a non-empty string")

            traffic = item.get("traffic")
            if not isinstance(traffic, (list, tuple)):
                raise TypeError(f"record[{i}] traffic must be list or tuple")

            try:
                int(item["label"])
                int(item["phish_type"])
                float(item["risk_score"])
            except Exception as exc:
                raise ValueError(
                    f"record[{i}] label/phish_type/risk_score type conversion failed: {exc}"
                ) from exc


def normalize_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    if train_ratio < 0 or valid_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative.")

    total = train_ratio + valid_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one ratio must be positive.")

    return train_ratio / total, valid_ratio / total, test_ratio / total


def split_one_group(
    records: Sequence[Dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = list(records)
    rng.shuffle(items)

    n = len(items)

    # round 后可能导致总和偏差，这里先算 train 和 valid，剩余全部给 test，保证不丢样本。
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))

    # 防止小样本组 round 后超过总数。
    if n_train + n_valid > n:
        overflow = n_train + n_valid - n
        reduce_valid = min(overflow, n_valid)
        n_valid -= reduce_valid
        overflow -= reduce_valid
        if overflow > 0:
            n_train -= min(overflow, n_train)

    train = items[:n_train]
    valid = items[n_train:n_train + n_valid]
    test = items[n_train + n_valid:]

    return train, valid, test


def split_records(
    records: List[Dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_label: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)

    if not stratify_label:
        return split_one_group(records, train_ratio, valid_ratio, test_ratio, rng)

    grouped = defaultdict(list)
    for item in records:
        grouped[int(item["label"])].append(item)

    train_all: List[Dict[str, Any]] = []
    valid_all: List[Dict[str, Any]] = []
    test_all: List[Dict[str, Any]] = []

    for label, group in sorted(grouped.items(), key=lambda x: x[0]):
        train, valid, test = split_one_group(group, train_ratio, valid_ratio, test_ratio, rng)
        train_all.extend(train)
        valid_all.extend(valid)
        test_all.extend(test)

    # 各集合内部再打乱一次，避免 label 分组顺序残留。
    rng.shuffle(train_all)
    rng.shuffle(valid_all)
    rng.shuffle(test_all)

    return train_all, valid_all, test_all


def dump_pkl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def label_counter(records: List[Dict[str, Any]]) -> Counter:
    return Counter(int(x["label"]) for x in records)


def print_summary(name: str, records: List[Dict[str, Any]]) -> None:
    labels = label_counter(records)
    total = len(records)
    print(f"{name}: samples={total}, labels={dict(sorted(labels.items()))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a model-ready PKL dataset into train/valid/test PKL files.")

    parser.add_argument("--input", type=str, required=True, help="Input pkl path, e.g. data/train10w.pkl")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory. It will contain train.pkl, valid.pkl and test.pkl.",
    )

    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split1 ratio. Default: 0.8")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="Valid split1 ratio. Default: 0.1")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split1 ratio. Default: 0.1")

    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--stratify_label",
        action="store_true",
        help="Stratify by label to keep label distribution approximately consistent.",
    )
    parser.add_argument(
        "--no_strict_validate",
        action="store_true",
        help="Only check required keys, skip strict type validation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    train_ratio, valid_ratio, test_ratio = normalize_ratios(
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
    )

    records = load_pkl(input_path)
    validate_records(records, strict=not args.no_strict_validate)

    train_records, valid_records, test_records = split_records(
        records=records,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
        stratify_label=args.stratify_label,
    )

    # 再次确保数量完全一致，没有丢样本。
    if len(train_records) + len(valid_records) + len(test_records) != len(records):
        raise RuntimeError("Split size mismatch: some records may be lost.")

    dump_pkl(train_records, output_dir / "train.pkl")
    dump_pkl(valid_records, output_dir / "valid.pkl")
    dump_pkl(test_records, output_dir / "test.pkl")

    print(f"input={input_path.resolve()}")
    print(f"output_dir={output_dir.resolve()}")
    print(f"ratios=train:{train_ratio:.4f}, valid:{valid_ratio:.4f}, test:{test_ratio:.4f}")
    print(f"seed={args.seed}")
    print(f"stratify_label={args.stratify_label}")
    print_summary("all", records)
    print_summary("train", train_records)
    print_summary("valid", valid_records)
    print_summary("test", test_records)
    print("saved files:")
    print(f"  {output_dir / 'train.pkl'}")
    print(f"  {output_dir / 'valid.pkl'}")
    print(f"  {output_dir / 'test.pkl'}")


if __name__ == "__main__":
    main()
