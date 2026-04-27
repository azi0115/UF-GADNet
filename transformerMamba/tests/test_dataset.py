"""Dataset and model smoke tests."""

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


def _make_sample(label: int = 0, suffix: str = "") -> dict:
    base = 0.01 if label else 0.02
    return {
        "url": f"https://example{suffix}.com/login" if label else f"https://github{suffix}.com/openai",
        "label": label,
        "phish_type": label,
        "risk_score": 0.8 if label else 0.1,
        "traffic": [
            [base, 64.0 + 10 * label],
            [base + 0.01, 128.0 + 20 * label],
            [base + 0.03, 512.0 + 30 * label],
            [base + 0.08, 256.0 + 10 * label],
            [base + 0.12, 1024.0 + 10 * label],
        ],
    }

@pytest.fixture()
def sample_file() -> Path:
    samples = [_make_sample(label=index % 2, suffix=str(index)) for index in range(8)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(samples, handle)
        return Path(handle.name)


def test_load_records(sample_file: Path) -> None:
    records = load_records(str(sample_file))
    assert len(records) == 8


def test_dataset_item(sample_file: Path) -> None:
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


def test_model_forward_raw_sequence(sample_file: Path) -> None:
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
