"""Minimal FT-Transformer for numerical tabular features.

This implementation is adapted for the project's flow 30-feature experiments and
follows the core design of the reference in `github/FT-transformer.py`:

- NumericalFeatureTokenizer
- trainable [CLS] token
- pre-normalization Transformer blocks
- ReGLU feed-forward layers
- classification head on top of the [CLS] representation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReGLU(nn.Module):
    """ReGLU activation used by the FT-Transformer baseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.relu(gates)


class NumericalFeatureTokenizer(nn.Module):
    """Convert each scalar feature to one learnable token."""

    def __init__(self, n_features: int, d_token: int, bias: bool = True) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        if d_token <= 0:
            raise ValueError("d_token must be positive.")

        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.weight.shape[-1])
        nn.init.uniform_(self.weight, -scale, scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -scale, scale)

    @property
    def n_tokens(self) -> int:
        return int(self.weight.shape[0])

    @property
    def d_token(self) -> int:
        return int(self.weight.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x to have shape [batch, features], got {tuple(x.shape)}.")
        tokens = self.weight.unsqueeze(0) * x.unsqueeze(-1)
        if self.bias is not None:
            tokens = tokens + self.bias.unsqueeze(0)
        return tokens


class CLSToken(nn.Module):
    """Append a trainable [CLS] token to the end of the token sequence."""

    def __init__(self, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.weight.numel())
        nn.init.uniform_(self.weight, -scale, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.weight.view(1, 1, -1).expand(x.shape[0], 1, -1)
        return torch.cat([x, cls], dim=1)


class FTTransformerBlock(nn.Module):
    """PreNorm Transformer block used by FT-Transformer."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads.")

        self.attention_norm = nn.LayerNorm(d_token)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_residual_dropout = nn.Dropout(residual_dropout)

        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden * 2),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attention_norm(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attention_residual_dropout(attn_output)

        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.ffn_residual_dropout(ffn_output)
        return x


@dataclass
class FTTransformerConfig:
    n_num_features: int
    d_token: int = 192
    n_blocks: int = 3
    attention_n_heads: int = 8
    attention_dropout: float = 0.2
    ffn_d_hidden: int = 256
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    head_dropout: float = 0.1


class FlowFTTransformer(nn.Module):
    """FT-Transformer binary classifier for 30 numerical flow features."""

    def __init__(self, config: FTTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = NumericalFeatureTokenizer(config.n_num_features, config.d_token, bias=True)
        self.cls_token = CLSToken(config.d_token)
        self.blocks = nn.ModuleList(
            [
                FTTransformerBlock(
                    d_token=config.d_token,
                    n_heads=config.attention_n_heads,
                    attention_dropout=config.attention_dropout,
                    ffn_d_hidden=config.ffn_d_hidden,
                    ffn_dropout=config.ffn_dropout,
                    residual_dropout=config.residual_dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_token),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.d_token, 1),
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x_num)
        x = self.cls_token(x)
        for block in self.blocks:
            x = block(x)
        cls_representation = x[:, -1]
        return self.head(cls_representation).squeeze(-1)
