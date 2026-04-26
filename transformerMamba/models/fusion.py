"""跨模态融合模块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class GateCrossModalFusion(nn.Module):
    """将 URL 表征与流量表征融合为统一语义向量。"""

    def __init__(
        self,
        url_dim: int,
        traffic_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """初始化跨模态门控融合层。

        Args:
            url_dim: URL 编码向量维度。
            traffic_dim: 流量编码向量维度。
            hidden_dim: 融合后的隐藏维度。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.url_proj = nn.Linear(url_dim, hidden_dim)
        self.traffic_proj = nn.Linear(traffic_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self._last_gate_stats: dict[str, float] | None = None

    @staticmethod
    def _summarize_gate(gate: torch.Tensor) -> dict[str, float]:
        """生成轻量级 gate 诊断统计信息。

        Args:
            gate: 门控张量。

        Returns:
            dict[str, float]: 便于日志或排障使用的统计摘要。
        """
        gate_detached = gate.detach()
        return {
            "gate_mean": float(gate_detached.mean().item()),
            "gate_min": float(gate_detached.min().item()),
            "gate_max": float(gate_detached.max().item()),
            "url_weight_mean": float(gate_detached.mean().item()),
            "traffic_weight_mean": float((1.0 - gate_detached).mean().item()),
        }

    def forward(
        self,
        url_repr: torch.Tensor,
        traffic_repr: torch.Tensor,
        return_gate_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """融合两种模态的全局表征。

        Args:
            url_repr: URL 编码器输出的全局向量。
            traffic_repr: 流量编码器输出的全局向量。
            return_gate_stats: 是否额外返回 gate 统计摘要。

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: 融合后的隐藏表示，必要时附带 gate 摘要。
        """
        url_state = self.url_proj(url_repr)
        traffic_state = self.traffic_proj(traffic_repr)
        # 门控向量用于自适应控制两种模态在最终表示中的占比。
        gate = self.gate(torch.cat([url_state, traffic_state], dim=-1))
        gate_stats = self._summarize_gate(gate)
        self._last_gate_stats = gate_stats
        fused = gate * url_state + (1.0 - gate) * traffic_state
        interaction = url_state * traffic_state
        output = self.output(torch.cat([fused, interaction], dim=-1))
        if return_gate_stats:
            return output, gate_stats
        return output

    def get_last_gate_stats(self) -> dict[str, float] | None:
        """获取最近一次前向传播记录的 gate 统计信息。

        Returns:
            dict[str, float] | None: 最近一次 gate 摘要；若尚未前向传播则返回 ``None``。
        """
        return self._last_gate_stats
