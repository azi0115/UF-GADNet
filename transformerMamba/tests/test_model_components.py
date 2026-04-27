"""Component-level model tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config import PhishingConfig
from models.detector import PhishingDetector
from models.fusion import GateCrossModalFusion


def test_gate_cross_modal_fusion_diagnostics() -> None:
    fusion = GateCrossModalFusion(url_dim=16, traffic_dim=16, hidden_dim=32, dropout=0.0)
    url_repr = torch.randn(4, 16)
    traffic_repr = torch.randn(4, 16)

    output = fusion(url_repr, traffic_repr, return_gate_stats=False)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 32)
    assert fusion.get_last_gate_stats() is None

    output_with_stats = fusion(url_repr, traffic_repr, return_gate_stats=True)
    assert isinstance(output_with_stats, tuple)
    fused, stats = output_with_stats
    assert fused.shape == (4, 32)
    assert stats == fusion.get_last_gate_stats()
    for key in (
        "gate_mean",
        "gate_std",
        "gate_min",
        "gate_max",
        "url_weight_mean",
        "traffic_weight_mean",
        "gate_lt_0_1_ratio",
        "gate_gt_0_9_ratio",
        "sample_gate_mean_min",
        "sample_gate_mean_max",
    ):
        assert key in stats


def _make_dummy_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "ids_1gram": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        "ids_2gram": torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
        "ids_3gram": torch.tensor([[1, 0], [2, 0]], dtype=torch.long),
        "url_mask": torch.tensor([[True, True, True, False], [True, True, False, False]], dtype=torch.bool),
        "traffic_feats_raw": torch.randn(batch_size, 5, 2),
        "traffic_mask_raw": torch.tensor([[True, True, True, False, False], [True, True, False, False, False]], dtype=torch.bool),
    }


def _assert_detector_outputs(outputs: dict[str, torch.Tensor], batch_size: int, num_phish_types: int) -> None:
    assert {"binary_logit", "binary_probability", "logits", "main_logits", "type_logits", "risk_score"} <= set(outputs.keys())
    assert outputs["binary_logit"].shape == (batch_size,)
    assert outputs["binary_probability"].shape == (batch_size,)
    assert outputs["logits"].shape == (batch_size,)
    assert outputs["main_logits"].shape == (batch_size,)
    assert outputs["type_logits"].shape == (batch_size, num_phish_types)
    assert outputs["risk_score"].shape == (batch_size,)


def test_phishing_detector_forward_raw_sequence() -> None:
    config = PhishingConfig(batch_size=2, traffic_input_mode="raw_sequence")
    model = PhishingDetector(config)
    batch = _make_dummy_inputs()

    outputs = model(
        ids_1gram=batch["ids_1gram"],
        ids_2gram=batch["ids_2gram"],
        ids_3gram=batch["ids_3gram"],
        url_mask=batch["url_mask"],
        traffic_feats=batch["traffic_feats_raw"],
        traffic_mask=batch["traffic_mask_raw"],
        return_diagnostics=True,
    )

    _assert_detector_outputs(outputs, batch_size=2, num_phish_types=config.num_phish_types)
    assert "fusion_gate_stats" in outputs
