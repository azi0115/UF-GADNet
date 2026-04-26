"""数据集与模型输入输出的基础测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from config import PhishingConfig
from dataset import (
    PhishingDataset,
    build_url_vocabs,
    collate_fn,
    debug_position_aware_tokenization,
    load_records,
    parse_url_to_char_labels,
)
from models import PhishingDetector


def _make_sample(label: int = 0) -> dict:
    """构造最小可用测试样本。

    Args:
        label: 样本标签。

    Returns:
        dict: 含 URL、流量与标签字段的样本字典。
    """
    return {
        "url": "https://example.com/login" if label else "https://github.com/openai",
        "label": label,
        "phish_type": label,
        "risk_score": 0.8 if label else 0.1,
        "traffic": [[0.01, 120.0], [0.02, 250.0], [0.03, 512.0]],
    }


@pytest.fixture()
def sample_file() -> Path:
    """生成临时样本文件供测试用例复用。

    Returns:
        Path: 临时 JSON 文件路径。
    """
    samples = [_make_sample(label=index % 2) for index in range(8)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(samples, handle)
        return Path(handle.name)


def test_load_records(sample_file: Path) -> None:
    """验证样本文件能够被完整加载。"""
    records = load_records(str(sample_file))
    assert len(records) == 8


def test_dataset_item(sample_file: Path) -> None:
    """验证数据集单样本编码结果的结构与类型。"""
    config = PhishingConfig()
    records = load_records(str(sample_file))
    vocabs = build_url_vocabs((item["url"] for item in records), config)
    dataset = PhishingDataset(records, vocabs, config.max_url_len, config.max_traffic_len)
    item = dataset[0]

    assert {"ids_1gram", "ids_2gram", "ids_3gram", "traffic", "label", "phish_type", "risk_score"} <= set(item.keys())
    assert item["ids_1gram"].dtype == torch.long
    assert item["traffic"].dim() == 2
    assert item["traffic"].shape[1] == 2


def test_collate_fn_shapes(sample_file: Path) -> None:
    """验证批处理拼装后的张量形状与掩码类型。"""
    config = PhishingConfig()
    records = load_records(str(sample_file))
    vocabs = build_url_vocabs((item["url"] for item in records), config)
    dataset = PhishingDataset(records, vocabs, config.max_url_len, config.max_traffic_len)
    batch = collate_fn([dataset[0], dataset[1], dataset[2]])

    assert batch["ids_1gram"].shape[0] == 3
    assert batch["traffic_feats"].shape[0] == 3
    assert batch["traffic_feats"].shape[-1] == 2
    assert batch["url_mask"].dtype == torch.bool
    assert batch["traffic_mask"].dtype == torch.bool


def test_model_forward(sample_file: Path) -> None:
    """验证模型前向传播的输出形状。"""
    config = PhishingConfig(batch_size=2)
    records = load_records(str(sample_file))
    vocabs = build_url_vocabs((item["url"] for item in records), config)
    dataset = PhishingDataset(records, vocabs, config.max_url_len, config.max_traffic_len)
    batch = collate_fn([dataset[0], dataset[1]])
    model = PhishingDetector(config)

    outputs = model(
        ids_1gram=batch["ids_1gram"],
        ids_2gram=batch["ids_2gram"],
        ids_3gram=batch["ids_3gram"],
        url_mask=batch["url_mask"],
        traffic_feats=batch["traffic_feats"],
        traffic_mask=batch["traffic_mask"],
    )

    assert outputs["logits"].shape == (2,)
    assert outputs["type_logits"].shape == (2, config.num_phish_types)
    assert outputs["risk_score"].shape == (2,)


def test_position_aware_tokenization_debug_example() -> None:
    config = PhishingConfig(use_position_ngram_vocab=True)
    url = "https://login.pay.com:443/auth?id=12&tk=ab#top"
    labeled = parse_url_to_char_labels(url, granularity=config.position_granularity)
    assert len(labeled) == len(url)

    grouped = debug_position_aware_tokenization(
        url,
        ngram_range=config.ngram_range,
        granularity=config.position_granularity,
        include_boundary_tokens=config.include_boundary_tokens,
    )
    tokens = set(grouped["1gram"] + grouped["2gram"] + grouped["3gram"])

    expected_subset = {
        "SCHEME::h",
        "SCHEME::ht",
        "SCHEME::htt",
        "SUBDOMAIN::log",
        "DOMAIN::pay",
        "DOMAIN+TLD::y.c",
        "TLD::com",
        "PORT::443",
        "PORT+PATH::3/a",
        "PATH::aut",
        "PATH+QUERY_KEY::h?i",
        "QUERY_KEY+QUERY_VALUE::d=1",
        "QUERY_VALUE+QUERY_KEY::2&t",
        "QUERY_VALUE+FRAGMENT::b#t",
        "FRAGMENT::top",
    }
    assert expected_subset <= tokens
