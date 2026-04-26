"""生成用于训练与验证的模拟钓鱼检测数据。"""

from __future__ import annotations

import json
import os
import random
from typing import List, Tuple

import numpy as np

NORMAL_DOMAINS = [
    "google.com",
    "github.com",
    "stackoverflow.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "reddit.com",
    "wikipedia.org",
]

PHISHING_BRANDS = [
    "paypal",
    "apple",
    "amazon",
    "google",
    "facebook",
    "microsoft",
    "chase",
    "bankofamerica",
]

RISKY_TLDS = ["xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work"]


def generate_normal_url() -> str:
    """生成一条模拟正常网站 URL。

    Returns:
        str: 合成后的正常 URL。
    """
    domain = random.choice(NORMAL_DOMAINS)
    path = random.choice(
        [
            "",
            "/",
            "/index.html",
            f"/page/{random.randint(1, 200)}",
            f"/search?q={''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))}",
        ]
    )
    return f"https://{domain}{path}"


def generate_phishing_url() -> Tuple[str, int]:
    """生成一条模拟钓鱼 URL 及其类型标签。

    Returns:
        Tuple[str, int]: 钓鱼 URL 与对应的钓鱼类型 ID。
    """
    phish_type = random.choice([1, 2, 3, 4])
    brand = random.choice(PHISHING_BRANDS)
    tld = random.choice(RISKY_TLDS)

    if phish_type == 1:
        url = f"https://{brand}-secure-{random.randint(100, 9999)}.{tld}/login"
    elif phish_type == 2:
        url = f"https://free-download-{random.randint(100, 9999)}.{tld}/setup"
    elif phish_type == 3:
        url = f"https://{brand}-banking-{random.randint(10, 99)}.{tld}/verify"
    else:
        url = f"https://{random.randint(10000, 99999)}-{brand}.{tld}/confirm"
    return url, phish_type


def generate_traffic(is_phishing: bool, n_packets: int | None = None) -> List[List[float]]:
    """生成与 URL 对应的模拟流量序列。

    Args:
        is_phishing: 是否生成钓鱼样本风格的流量。
        n_packets: 可选的包数量；不传则随机生成。

    Returns:
        List[List[float]]: 由 ``[timestamp, size]`` 组成的流量序列。
    """
    packet_count = n_packets or random.randint(20, 160)
    traffic: List[List[float]] = []
    timestamp = 0.0

    for _ in range(packet_count):
        timestamp += random.uniform(0.002, 0.06) * (1.7 if is_phishing else 1.0)
        if is_phishing:
            size = max(40, int(abs(random.gauss(850, 350))))
            if random.random() < 0.12:
                size += random.randint(900, 3000)
        else:
            size = max(20, min(1500, int(abs(random.gauss(280, 120)))))
        traffic.append([round(timestamp, 4), float(size)])
    return traffic


def generate_split(total: int, phishing_ratio: float) -> List[dict]:
    """根据给定规模和比例生成一个数据划分。

    Args:
        total: 样本总数。
        phishing_ratio: 钓鱼样本占比。

    Returns:
        List[dict]: 混洗后的样本列表。
    """
    phishing_count = int(total * phishing_ratio)
    benign_count = total - phishing_count
    data = []

    for _ in range(benign_count):
        data.append(
            {
                "url": generate_normal_url(),
                "traffic": generate_traffic(False),
                "label": 0,
                "phish_type": 0,
                "risk_score": round(random.uniform(0.02, 0.25), 3),
            }
        )

    for _ in range(phishing_count):
        url, phish_type = generate_phishing_url()
        data.append(
            {
                "url": url,
                "traffic": generate_traffic(True),
                "label": 1,
                "phish_type": phish_type,
                "risk_score": round(random.uniform(0.55, 0.98), 3),
            }
        )

    random.shuffle(data)
    return data


def main() -> None:
    """生成 train、val、test 三个模拟数据文件。"""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    for split_name, count in [("train", 5000), ("val", 500), ("test", 300)]:
        data = generate_split(count, phishing_ratio=0.35)
        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        print(f"saved {split_name} split to {output_path} ({len(data)} samples)")


if __name__ == "__main__":
    main()
