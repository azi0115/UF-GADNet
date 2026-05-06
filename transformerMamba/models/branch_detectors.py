"""Single-branch detector variants."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .traffic_encoder import TrafficMambaEncoder
from .url_encoder import URLTransformerEncoder


class URLOnlyDetector(nn.Module):
    """Detector that only uses the URL branch."""

    def __init__(self, config) -> None:
        super().__init__()
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
        self.binary_head = nn.Sequential(
            nn.Linear(config.url_embed_dim, config.url_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.url_embed_dim, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(config.url_embed_dim, config.url_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.url_embed_dim, config.num_phish_types),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(config.url_embed_dim, config.url_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.url_embed_dim // 2, 1),
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
        del traffic_feats, traffic_mask, return_diagnostics
        url_repr, _ = self.url_encoder(ids_1gram, ids_2gram, ids_3gram, url_mask)
        binary_logit = self.binary_head(url_repr).squeeze(-1)
        binary_probability = torch.sigmoid(binary_logit)
        return {
            "binary_logit": binary_logit,
            "binary_probability": binary_probability,
            "logits": binary_logit,
            "main_logits": binary_logit,
            "type_logits": self.type_head(url_repr),
            "risk_score": torch.sigmoid(self.risk_head(url_repr).squeeze(-1)),
        }


class TrafficOnlyDetector(nn.Module):
    """Detector that only uses the traffic branch."""

    def __init__(self, config) -> None:
        super().__init__()
        self.traffic_encoder = TrafficMambaEncoder(
            input_dim=config.traffic_input_dim,
            embed_dim=config.traffic_embed_dim,
            num_layers=config.traffic_num_layers,
            num_heads=getattr(config, "traffic_num_heads", 4),
            expand_factor=config.traffic_expand_factor,
            kernel_size=config.traffic_kernel_size,
            patch_len=getattr(config, "traffic_patch_len", 8),
            patch_stride=getattr(config, "traffic_patch_stride", 8),
            max_len=config.max_traffic_len,
            dropout=config.dropout,
        )
        self.binary_head = nn.Sequential(
            nn.Linear(config.traffic_embed_dim, config.traffic_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.traffic_embed_dim, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(config.traffic_embed_dim, config.traffic_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.traffic_embed_dim, config.num_phish_types),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(config.traffic_embed_dim, config.traffic_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.traffic_embed_dim // 2, 1),
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
        del ids_1gram, ids_2gram, ids_3gram, url_mask, return_diagnostics
        traffic_repr, _ = self.traffic_encoder(traffic_feats, traffic_mask)
        binary_logit = self.binary_head(traffic_repr).squeeze(-1)
        binary_probability = torch.sigmoid(binary_logit)
        return {
            "binary_logit": binary_logit,
            "binary_probability": binary_probability,
            "logits": binary_logit,
            "main_logits": binary_logit,
            "type_logits": self.type_head(traffic_repr),
            "risk_score": torch.sigmoid(self.risk_head(traffic_repr).squeeze(-1)),
        }
