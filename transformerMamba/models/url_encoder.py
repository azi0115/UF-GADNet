"""URL 编码器模块。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_length(sequence: torch.Tensor, length: int) -> torch.Tensor:
    """将变长序列右侧补零到目标长度。

    Args:
        sequence: 输入张量，形状通常为 ``[B, T, D]``。
        length: 目标时间维长度。

    Returns:
        torch.Tensor: 时间维被补齐后的张量。
    """
    if sequence.size(1) >= length:
        return sequence[:, :length]
    pad_width = length - sequence.size(1)
    return F.pad(sequence, (0, 0, 0, pad_width))


class NGramEmbedding(nn.Module):
    """对 1/2/3-gram URL token 做嵌入并进行门控融合。"""

    def __init__(
        self,
        vocab_1gram: int,
        vocab_2gram: int,
        vocab_3gram: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """初始化 n-gram 嵌入与融合层。

        Args:
            vocab_1gram: 1-gram 词表大小。
            vocab_2gram: 2-gram 词表大小。
            vocab_3gram: 3-gram 词表大小。
            embed_dim: 嵌入维度。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.embed_1gram = nn.Embedding(vocab_1gram, embed_dim, padding_idx=0)
        self.embed_2gram = nn.Embedding(vocab_2gram, embed_dim, padding_idx=0)
        self.embed_3gram = nn.Embedding(vocab_3gram, embed_dim, padding_idx=0)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ids_1gram: torch.Tensor,
        ids_2gram: torch.Tensor,
        ids_3gram: torch.Tensor,
    ) -> torch.Tensor:
        """计算多粒度 URL token 的融合嵌入。

        Args:
            ids_1gram: 1-gram token ID 张量。
            ids_2gram: 2-gram token ID 张量。
            ids_3gram: 3-gram token ID 张量。

        Returns:
            torch.Tensor: 与 1-gram 序列长度对齐的融合嵌入张量。
        """
        embed_1 = self.embed_1gram(ids_1gram)
        embed_2 = _pad_to_length(self.embed_2gram(ids_2gram), embed_1.size(1))
        embed_3 = _pad_to_length(self.embed_3gram(ids_3gram), embed_1.size(1))
        gate_logits = self.gate(torch.cat([embed_1, embed_2, embed_3], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        fused = (
            gate[..., 0:1] * embed_1
            + gate[..., 1:2] * embed_2
            + gate[..., 2:3] * embed_3
        )
        return self.dropout(self.output_norm(fused))


class URLTransformerEncoder(nn.Module):
    """基于 Transformer 的 URL 序列编码器。"""

    def __init__(
        self,
        vocab_1gram: int,
        vocab_2gram: int,
        vocab_3gram: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        max_url_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """初始化 URL 编码器。

        Args:
            vocab_1gram: 1-gram 词表大小。
            vocab_2gram: 2-gram 词表大小。
            vocab_3gram: 3-gram 词表大小。
            embed_dim: URL 隐藏维度。
            num_heads: 多头注意力头数。
            num_layers: Transformer 堆叠层数。
            ff_dim: 前馈网络隐藏维度。
            max_url_len: URL 最大长度。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.ngram_embedding = NGramEmbedding(
            vocab_1gram=vocab_1gram,
            vocab_2gram=vocab_2gram,
            vocab_3gram=vocab_3gram,
            embed_dim=embed_dim,
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.position = nn.Parameter(torch.zeros(1, max_url_len + 1, embed_dim))
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ids_1gram: torch.Tensor,
        ids_2gram: torch.Tensor,
        ids_3gram: torch.Tensor,
        url_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码 URL 序列并输出全局与局部表示。

        Args:
            ids_1gram: 1-gram token ID 张量。
            ids_2gram: 2-gram token ID 张量。
            ids_3gram: 3-gram token ID 张量。
            url_mask: URL 有效位置掩码。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: CLS 全局表示与逐 token 序列表示。
        """
        sequence = self.ngram_embedding(ids_1gram, ids_2gram, ids_3gram)
        batch_size, sequence_length, _ = sequence.shape

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_token, sequence], dim=1)
        sequence = sequence + self.position[:, : sequence_length + 1]

        full_mask = torch.cat(
            [torch.ones(batch_size, 1, device=url_mask.device, dtype=torch.bool), url_mask],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=~full_mask)
        encoded = self.output_norm(encoded)
        return encoded[:, 0], encoded[:, 1:]
