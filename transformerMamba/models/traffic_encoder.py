"""Traffic sequence encoder compatibility layer."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .uniTS import TrafficUniTSEncoder


class MambaStyleBlock(nn.Module):
    """Compatibility stub retained for older imports."""

    def __init__(
        self,
        embed_dim: int,
        expand_factor: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.identity(inputs)


class TrafficMambaEncoder(nn.Module):
    """Project traffic encoder backed by the UniTS sequence encoder blocks."""

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 96,
        num_layers: int = 2,
        expand_factor: int = 2,
        kernel_size: int = 5,
        max_len: int = 512,
        dropout: float = 0.1,
        num_heads: int = 4,
        patch_len: int = 8,
        patch_stride: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = TrafficUniTSEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_len=patch_len,
            stride=patch_stride,
            max_len=max_len,
            expand_factor=expand_factor,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        traffic_feats: torch.Tensor,
        traffic_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        global_repr, traffic_sequence = self.encoder(traffic_feats, traffic_mask)
        return self.output_norm(global_repr), traffic_sequence
