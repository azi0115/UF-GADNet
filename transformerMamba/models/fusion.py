"""Cross-modal fusion layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class GateCrossModalFusion(nn.Module):
    """Fuse URL and traffic representations into one hidden vector."""

    def __init__(
        self,
        url_dim: int,
        traffic_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.url_proj = nn.Sequential(
            nn.Linear(url_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.traffic_proj = nn.Sequential(
            nn.Linear(traffic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self._last_gate_stats: dict[str, float] | None = None

    @staticmethod
    def _summarize_gate(gate: torch.Tensor) -> dict[str, float]:
        """Summarize gate activations for diagnostics."""
        gate_detached = gate.detach()
        sample_mean = gate_detached.mean(dim=-1)
        return {
            "gate_mean": float(gate_detached.mean().item()),
            "gate_std": float(gate_detached.std().item()),
            "gate_min": float(gate_detached.min().item()),
            "gate_max": float(gate_detached.max().item()),
            "url_weight_mean": float(gate_detached.mean().item()),
            "traffic_weight_mean": float((1.0 - gate_detached).mean().item()),
            "gate_lt_0_1_ratio": float((gate_detached < 0.1).float().mean().item()),
            "gate_gt_0_9_ratio": float((gate_detached > 0.9).float().mean().item()),
            "sample_gate_mean_min": float(sample_mean.min().item()),
            "sample_gate_mean_max": float(sample_mean.max().item()),
        }

    def forward(
        self,
        url_repr: torch.Tensor,
        traffic_repr: torch.Tensor,
        return_gate_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Fuse two modality-level vectors."""
        url_state = self.url_proj(url_repr)
        traffic_state = self.traffic_proj(traffic_repr)
        diff = torch.abs(url_state - traffic_state)
        interaction = url_state * traffic_state
        gate = self.gate(torch.cat([url_state, traffic_state, diff, interaction], dim=-1))
        fused = gate * url_state + (1.0 - gate) * traffic_state
        output = self.output(torch.cat([url_state, traffic_state, fused, diff, interaction], dim=-1))

        if return_gate_stats:
            gate_stats = self._summarize_gate(gate)
            self._last_gate_stats = gate_stats
            return output, gate_stats
        return output

    def get_last_gate_stats(self) -> dict[str, float] | None:
        """Return the latest gate stats or ``None`` before any diagnostic forward."""
        return self._last_gate_stats
