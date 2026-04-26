"""轻量级冒烟测试。"""

from __future__ import annotations

from config import PhishingConfig


def test_config_roundtrip() -> None:
    """验证配置对象能够完成字典往返序列化。"""
    config = PhishingConfig()
    restored = PhishingConfig.from_dict(config.to_dict())
    assert restored == config
