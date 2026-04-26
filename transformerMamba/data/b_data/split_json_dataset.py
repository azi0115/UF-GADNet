# split_json_dataset.py

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入文件必须是 JSON 数组格式，例如：[ {...}, {...} ]")

    return data


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_list(items, train_ratio, test_ratio, valid_ratio):
    n = len(items)

    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    train = items[:n_train]
    test = items[n_train:n_train + n_test]
    valid = items[n_train + n_test:]

    return train, test, valid


def random_split(data, train_ratio, test_ratio, valid_ratio, seed):
    random.seed(seed)
    data = data[:]
    random.shuffle(data)

    return split_list(data, train_ratio, test_ratio, valid_ratio)


def stratified_split(data, train_ratio, test_ratio, valid_ratio, seed, label_key="label"):
    random.seed(seed)

    groups = defaultdict(list)

    for item in data:
        if label_key not in item:
            raise ValueError(f"样本缺少字段：{label_key}")
        groups[item[label_key]].append(item)

    train, test, valid = [], [], []

    for label, items in groups.items():
        random.shuffle(items)

        g_train, g_test, g_valid = split_list(
            items,
            train_ratio,
            test_ratio,
            valid_ratio
        )

        train.extend(g_train)
        test.extend(g_test)
        valid.extend(g_valid)

    random.shuffle(train)
    random.shuffle(test)
    random.shuffle(valid)

    return train, test, valid


def check_ratios(train_ratio, test_ratio, valid_ratio):
    total = train_ratio + test_ratio + valid_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"比例之和必须等于 1，目前是 {total}"
        )


def print_stats(name, data):
    label_count = defaultdict(int)

    for item in data:
        label_count[item.get("label", "missing")] += 1

    print(f"{name}: {len(data)} samples")
    print(f"{name} label distribution: {dict(label_count)}")


def main():
    parser = argparse.ArgumentParser(
        description="Split JSON dataset into train/test/valid sets"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSON 文件路径，例如 data/all.json"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/split",
        help="输出目录，默认 data/split"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例，默认 0.8"
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="测试集比例，默认 0.1"
    )

    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="验证集比例，默认 0.1"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42"
    )

    parser.add_argument(
        "--stratify",
        action="store_true",
        help="是否按 label 分层抽样"
    )

    args = parser.parse_args()

    check_ratios(
        args.train_ratio,
        args.test_ratio,
        args.valid_ratio
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    data = load_json(input_path)

    if args.stratify:
        train, test, valid = stratified_split(
            data,
            args.train_ratio,
            args.test_ratio,
            args.valid_ratio,
            args.seed
        )
    else:
        train, test, valid = random_split(
            data,
            args.train_ratio,
            args.test_ratio,
            args.valid_ratio,
            args.seed
        )

    save_json(train, output_dir / "train.json")
    save_json(test, output_dir / "test.json")
    save_json(valid, output_dir / "valid.json")

    print("Split finished.")
    print_stats("train", train)
    print_stats("test", test)
    print_stats("valid", valid)


if __name__ == "__main__":
    main()