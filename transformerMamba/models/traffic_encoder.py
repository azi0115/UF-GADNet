"""流量时序编码器模块。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalConvStem(nn.Module):
    """Lightweight local temporal stem before the shallow Mamba stack."""

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,
        )
        self.pointwise_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = inputs
        states = self.norm(inputs)
        states = self.depthwise_conv(states.transpose(1, 2)).transpose(1, 2)
        outputs = self.pointwise_proj(F.gelu(states))
        outputs = self.dropout(outputs)
        if mask is not None:
            outputs = outputs * mask.unsqueeze(-1)
        result = residual + outputs
        if mask is not None:
            result = result * mask.unsqueeze(-1)
        return result


class MambaStyleBlock(nn.Module):
    """近似 Mamba 风格的局部卷积时序建模块。"""

    def __init__(
        self,
        embed_dim: int,
        expand_factor: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """初始化单个时序建模块。

        Args:
            embed_dim: 输入与输出隐藏维度。
            expand_factor: 内部扩展倍数。
            kernel_size: 深度可分离卷积核大小。
            dropout: Dropout 概率。
        """
        super().__init__()
        inner_dim = embed_dim * expand_factor
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=inner_dim,
        )
        self.state_proj = nn.Linear(inner_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """对流量序列做一次残差时序变换。

        Args:
            inputs: 输入序列张量。
            mask: 可选有效位置掩码。

        Returns:
            torch.Tensor: 与输入同形状的输出张量。
        """
        residual = inputs
        states = self.norm(inputs)
        values, gates = self.in_proj(states).chunk(2, dim=-1)
        values = self.depthwise_conv(values.transpose(1, 2))[..., : inputs.size(1)].transpose(1, 2)
        values = values + torch.tanh(self.state_proj(values))
        outputs = self.out_proj(F.silu(values) * torch.sigmoid(gates))
        outputs = self.dropout(outputs)
        if mask is not None:
            outputs = outputs * mask.unsqueeze(-1)
        result = residual + outputs
        if mask is not None:
            result = result * mask.unsqueeze(-1)
        return result


class TrafficMambaEncoder(nn.Module):
    """将流量时间序列编码为全局向量与局部序列表示。"""

    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 128,
        num_layers: int = 3,
        expand_factor: int = 2,
        kernel_size: int = 3,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """初始化流量编码器。

        Args:
            input_dim: 原始流量特征维度。
            embed_dim: 编码后的隐藏维度。
            num_layers: Mamba 风格模块层数。
            expand_factor: 每层内部扩展倍数。
            kernel_size: 深度卷积核大小。
            max_len: 最大时序长度。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.local_stem = LocalConvStem(
            embed_dim=embed_dim,
            kernel_size=3 if kernel_size >= 3 else kernel_size,
            dropout=dropout,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.position = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                MambaStyleBlock(
                    embed_dim=embed_dim,
                    expand_factor=expand_factor,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        traffic_feats: torch.Tensor,
        traffic_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码流量时序特征。

        Args:
            traffic_feats: 原始流量输入张量。
            traffic_mask: 流量有效位置掩码。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 全局 CLS 表示与逐步时序表示。
        """
        sequence = self.input_proj(self.input_norm(traffic_feats))
        sequence = self.local_stem(sequence, traffic_mask.float())
        batch_size, sequence_length, _ = sequence.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_token, sequence], dim=1)
        sequence = sequence + self.position[:, : sequence_length + 1]

        full_mask = torch.cat(
            [torch.ones(batch_size, 1, device=traffic_mask.device, dtype=torch.bool), traffic_mask],
            dim=1,
        )
        mask_float = full_mask.float()
        sequence = sequence * mask_float.unsqueeze(-1)

        for block in self.blocks:
            sequence = block(sequence, mask_float)

        sequence = self.output_norm(sequence)
        sequence = sequence * mask_float.unsqueeze(-1)
        traffic_sequence = sequence[:, 1:]
        traffic_mask_float = traffic_mask.unsqueeze(-1).float()
        pooled = (traffic_sequence * traffic_mask_float).sum(dim=1) / traffic_mask_float.sum(dim=1).clamp_min(1.0)
        global_repr = self.pool_proj(torch.cat([sequence[:, 0], pooled], dim=-1))
        return global_repr, traffic_sequence
