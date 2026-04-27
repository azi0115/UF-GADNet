"""Top-level phishing detector model."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .fusion import GateCrossModalFusion
from .traffic_encoder import TrafficMambaEncoder
from .url_encoder import URLTransformerEncoder


class PhishingDetector(nn.Module):
    """Joint URL and traffic phishing detector."""

    def __init__(self, config) -> None:
        super().__init__()
        self.use_traffic = getattr(config, "use_traffic", True)
        self.url_encoder = URLTransformerEncoder(
            vocab_1gram=config.vocab_1gram_max_size,
            vocab_2gram=config.vocab_2gram_max_size,
            vocab_3gram=config.vocab_3gram_max_size,
            embed_dim=config.url_embed_dim,
            num_heads=config.url_num_heads,
            num_layers=config.url_num_layers,
            ff_dim=config.url_ffn_dim,
            max_url_len=config.max_url_len,
            dropout=config.dropout,
        )
        self.traffic_encoder = TrafficMambaEncoder(
            input_dim=config.traffic_input_dim,
            embed_dim=config.traffic_embed_dim,
            num_layers=config.traffic_num_layers,
            expand_factor=config.traffic_expand_factor,
            kernel_size=config.traffic_kernel_size,
            max_len=config.max_traffic_len,
            dropout=config.dropout,
        )
        self.fusion = GateCrossModalFusion(
            url_dim=config.url_embed_dim,
            traffic_dim=config.traffic_embed_dim,
            hidden_dim=config.fusion_dim,
            dropout=config.dropout,
        )
        self.binary_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.num_phish_types),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, 1),
        )

    def forward(
        self,
        ids_1gram: torch.Tensor,
        ids_2gram: torch.Tensor,
        ids_3gram: torch.Tensor,
        url_mask: torch.Tensor,
        traffic_feats: torch.Tensor,
        traffic_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        url_repr, _ = self.url_encoder(ids_1gram, ids_2gram, ids_3gram, url_mask)
        if self.use_traffic:
            traffic_repr, _ = self.traffic_encoder(traffic_feats, traffic_mask)
        else:
            traffic_repr = torch.zeros(
                ids_1gram.size(0),
                self.traffic_encoder.output_norm.normalized_shape[0],
                device=ids_1gram.device,
                dtype=url_repr.dtype,
            )

        fusion_result = self.fusion(url_repr, traffic_repr, return_gate_stats=return_diagnostics)
        if return_diagnostics:
            fused, gate_stats = fusion_result
        else:
            fused = fusion_result
            gate_stats = None

        binary_logit = self.binary_head(fused).squeeze(-1)
        binary_probability = torch.sigmoid(binary_logit)
        outputs = {
            "binary_logit": binary_logit,
            "binary_probability": binary_probability,
            "logits": binary_logit,
            "main_logits": binary_logit,
            "type_logits": self.type_head(fused),
            "risk_score": torch.sigmoid(self.risk_head(fused).squeeze(-1)),
        }
        if gate_stats is not None:
            outputs["fusion_gate_stats"] = gate_stats
        return outputs
