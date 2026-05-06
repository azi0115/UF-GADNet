"""Standalone traffic-transformer branch and classifier."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class TrafficSequenceEmbedding(nn.Module):
    """Embed raw traffic sequence values into transformer tokens."""

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 96,
        max_len: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.position = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, traffic_feats: torch.Tensor, traffic_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.input_proj(self.input_norm(traffic_feats))
        batch_size, sequence_length, _ = tokens.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.position[:, : sequence_length + 1]
        full_mask = torch.cat(
            [torch.ones(batch_size, 1, device=traffic_mask.device, dtype=torch.bool), traffic_mask],
            dim=1,
        )
        tokens = self.dropout(tokens) * full_mask.unsqueeze(-1).float()
        return tokens, full_mask


class TrafficTransformerBackbone(nn.Module):
    """Raw traffic encoder based on stacked transformer layers."""

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 384,
        max_len: int = 10000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = TrafficSequenceEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            max_len=max_len,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        traffic_feats: torch.Tensor,
        traffic_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, full_mask = self.embedding(traffic_feats, traffic_mask)
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        encoded = self.output_norm(encoded)
        encoded = encoded * full_mask.unsqueeze(-1).float()
        sequence_tokens = encoded[:, 1:]
        sequence_mask = traffic_mask.unsqueeze(-1).float()
        pooled = (sequence_tokens * sequence_mask).sum(dim=1) / sequence_mask.sum(dim=1).clamp_min(1.0)
        global_repr = self.pool_proj(torch.cat([encoded[:, 0], pooled], dim=-1))
        return global_repr, sequence_tokens


class ClassicTrafficClassifier(nn.Module):
    """Classic MLP classifier head for traffic representations."""

    def __init__(
        self,
        embed_dim: int = 96,
        num_phish_types: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_phish_types),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, traffic_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
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


class TrafficTransformerOnlyDetector(nn.Module):
    """Traffic-only detector compatible with the project's training loop."""

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 384,
        max_len: int = 10000,
        num_phish_types: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.traffic_encoder = TrafficTransformerBackbone(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            max_len=max_len,
            dropout=dropout,
        )
        self.classifier = ClassicTrafficClassifier(
            embed_dim=embed_dim,
            num_phish_types=num_phish_types,
            dropout=dropout,
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
        return self.classifier(traffic_repr)
