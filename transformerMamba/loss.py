"""损失函数定义模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """实现面向二分类任务的 Focal Loss。"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """初始化 Focal Loss 参数。

        Args:
            alpha: 正样本类别平衡系数。
            gamma: 困难样本聚焦系数。
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算批量样本的 Focal Loss。

        Args:
            logits: 模型输出的二分类 logits。
            targets: 取值为 0/1 的标签张量。

        Returns:
            torch.Tensor: 单个标量损失值。
        """
        logits = logits.float().view(-1)
        targets = targets.float().view(-1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        return (focal_weight * bce).mean()


class WeightedMultiTaskLoss(nn.Module):
    """组合主任务、类型预测与风险回归损失。"""

    def __init__(
        self,
        lambda_main: float = 1.0,
        beta_type: float = 0.25,
        gamma_risk: float = 0.15,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        """初始化多任务损失的权重与子损失函数。

        Args:
            lambda_main: 主任务损失权重。
            beta_type: 类型分类损失权重。
            gamma_risk: 风险回归损失权重。
            focal_alpha: Focal Loss 的类别平衡系数。
            focal_gamma: Focal Loss 的聚焦系数。
        """
        super().__init__()
        self.lambda_main = lambda_main
        self.beta_type = beta_type
        self.gamma_risk = gamma_risk
        self.main_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.type_loss = nn.CrossEntropyLoss()
        self.risk_loss = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """根据模型输出与标签字典计算多任务损失。

        Args:
            outputs: 模型前向传播输出。
            batch: 当前批次的标签与输入字典。

        Returns:
            Dict[str, torch.Tensor]: 包含主损失、辅助损失和总损失的字典。
        """
        main = self.main_loss(outputs["logits"], batch["label"])
        type_component = self.type_loss(outputs["type_logits"], batch["phish_type"])
        risk_component = self.risk_loss(outputs["risk_score"], batch["risk_score"].float())
        total = self.lambda_main * main + self.beta_type * type_component + self.gamma_risk * risk_component
        return {
            "main": main,
            "type": type_component,
            "risk": risk_component,
            "total": total,
        }


def build_criterion(config) -> WeightedMultiTaskLoss:
    """根据配置对象构建多任务损失实例。

    Args:
        config: 提供损失权重与 Focal Loss 参数的配置对象。

    Returns:
        WeightedMultiTaskLoss: 已初始化的损失函数对象。
    """
    return WeightedMultiTaskLoss(
        lambda_main=1.0,
        beta_type=config.loss_beta,
        gamma_risk=config.loss_gamma,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
    )
