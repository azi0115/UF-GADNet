"""项目配置定义与命令行参数构建模块。"""  # 模块级说明：统一管理训练、评估、推理配置。

from __future__ import annotations  # 启用延迟类型注解，避免前向引用问题。

import argparse  # 命令行参数解析标准库。
from dataclasses import asdict, dataclass  # dataclass 用于配置结构定义，asdict 用于序列化。
from typing import Any, Dict, Tuple  # 类型标注：通用字典和值类型。


@dataclass  # 使用 dataclass 简化配置类定义。
class PhishingConfig:  # 项目总配置对象。
    """封装训练、评估与推理阶段共享的配置参数。"""  # 类说明：所有入口共用一套配置结构。

    train_path: str = "data/b_data/data/split/train.json"  # 训练集文件路径。
    val_path: str = "data/b_data/data/split/valid.json"  # 验证集文件路径。
    test_path: str = "data/b_data/data/split/test.json"  # 测试集文件路径。

    ngram_min_freq: int = 1  # URL n-gram 进入词表的最小出现频次。
    use_position_ngram_vocab: bool = True  # 是否默认启用位置感知 URL n-gram 词表。
    ngram_range: Tuple[int, int] = (1, 3)  # URL n-gram 范围，(1, 3) 表示使用 1/2/3-gram。
    position_granularity: str = "fine"  # URL 位置标签粒度，当前实现主要支持 fine。
    include_boundary_tokens: bool = True  # 是否保留跨 URL 组件边界的 n-gram token。
    save_position_vocab_meta: bool = True  # 是否额外保存位置感知词表元信息文件。
    lowercase_url: bool = False  # 是否统一将 URL 转小写；False 表示保留原始大小写。

    url_embed_dim: int = 128  # URL 编码分支的嵌入维度。
    url_num_heads: int = 4  # URL Transformer 的多头注意力头数。
    url_num_layers: int = 3  # URL Transformer 的层数。
    url_ffn_dim: int = 1024  # URL Transformer 前馈网络隐藏维度。

    traffic_input_dim: int = 2  # 流量输入的原始特征维度，当前是 [delta_time, size]。
    num_phish_types: int = 5  # 钓鱼子类型类别数。

    grad_clip: float = 1.0  # 训练时梯度裁剪阈值。
    num_workers: int = 0  # DataLoader 的 worker 数量。
    seed: int = 42  # 随机种子，控制训练和切分复现性。
    device: str = "auto"  # 运行设备选择，可为 auto/cpu/cuda。

    checkpoint_dir: str = "checkpoints"  # checkpoint 输出目录。
    log_dir: str = "logs"  # 日志输出目录。
    vocab_path: str = "checkpoints/url_vocabs.json"  # URL 词表保存路径。

    threshold_metric: str = "f1"  # 验证集自动选阈值时使用的指标。
    save_last_checkpoint: bool = True  # 是否保存最后一个 epoch 的 checkpoint。
    predict_batch_size: int = 32  # 推理阶段默认批大小。
    use_traffic: bool = True  # 是否启用流量分支；False 可做 URL-only 消融。

    max_url_len: int = 512  # URL 序列最大长度。
    max_traffic_len: int = 1024  # 流量时序最大长度。

    vocab_1gram_max_size: int = 10000  # 1-gram 词表大小上限。
    vocab_2gram_max_size: int = 1000000  # 2-gram 词表大小上限。
    vocab_3gram_max_size: int = 1000000  # 3-gram 词表大小上限。

    traffic_embed_dim: int = 128  # 流量编码器隐藏维度。
    traffic_num_layers: int = 4  # 流量编码器层数。
    traffic_expand_factor: int = 4  # 流量编码器内部扩展倍数。
    traffic_kernel_size: int = 7  # 流量编码器卷积核大小。

    fusion_dim: int = 128  # 多模态融合层隐藏维度。

    dropout: float = 0.15  # 全局 dropout 概率。
    batch_size: int = 16  # 训练与评估阶段 batch size。
    epochs: int = 30  # 训练总轮数。
    lr: float = 1e-4  # 学习率。
    weight_decay: float = 5e-5  # 权重衰减系数。
    patience: int = 20  # early stopping 的耐心轮数。

    focal_alpha: float = 0.75  # Focal Loss 的 alpha 参数。
    focal_gamma: float = 2.0  # Focal Loss 的 gamma 参数。
    loss_beta: float = 0.1  # 多任务损失中类型分类分支权重。
    loss_gamma: float = 0.03  # 多任务损失中风险回归分支权重。

    def to_dict(self) -> Dict[str, Any]:  # 将配置对象转换为字典。
        """将配置对象序列化为字典。"""  # 方法说明：便于写入 checkpoint 或 JSON。
        return asdict(self)  # dataclass 标准序列化。

    @classmethod  # 声明为类方法，从字典恢复配置对象。
    def from_dict(cls, data: Dict[str, Any]) -> "PhishingConfig":  # 从任意字典恢复配置。
        """从任意字典中过滤有效字段并恢复配置对象。"""  # 方法说明：自动忽略无关字段。
        valid_fields = cls.__dataclass_fields__.keys()  # 获取 dataclass 已定义字段名集合。
        filtered = {key: value for key, value in data.items() if key in valid_fields}  # 只保留配置类支持的键。
        return cls(**filtered)  # 用过滤后的字段构造配置实例。


def build_parser() -> argparse.ArgumentParser:  # 构建命令行参数解析器。
    """构建项目统一命令行参数解析器。"""  # 方法说明：所有入口共用此解析器。
    parser = argparse.ArgumentParser(description="Phishing detector configuration", formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建解析器并开启默认值展示。

    parser.add_argument("--train_path", type=str, default=PhishingConfig.train_path)  # 训练集路径参数。
    parser.add_argument("--val_path", type=str, default=PhishingConfig.val_path)  # 验证集路径参数。
    parser.add_argument("--test_path", type=str, default=PhishingConfig.test_path)  # 测试集路径参数。

    parser.add_argument("--max_url_len", type=int, default=PhishingConfig.max_url_len)  # URL 最大长度参数。
    parser.add_argument("--max_traffic_len", type=int, default=PhishingConfig.max_traffic_len)  # 流量最大长度参数。

    parser.add_argument("--vocab_1gram_max_size", type=int, default=PhishingConfig.vocab_1gram_max_size)  # 1-gram 词表上限参数。
    parser.add_argument("--vocab_2gram_max_size", type=int, default=PhishingConfig.vocab_2gram_max_size)  # 2-gram 词表上限参数。
    parser.add_argument("--vocab_3gram_max_size", type=int, default=PhishingConfig.vocab_3gram_max_size)  # 3-gram 词表上限参数。
    parser.add_argument("--ngram_min_freq", type=int, default=PhishingConfig.ngram_min_freq)  # 词表最小频次参数。

    parser.add_argument("--use_position_ngram_vocab", action=argparse.BooleanOptionalAction, default=PhishingConfig.use_position_ngram_vocab)  # 是否启用位置感知词表。
    parser.add_argument("--ngram_range", type=int, nargs=2, metavar=("MIN_N", "MAX_N"), default=PhishingConfig.ngram_range)  # n-gram 范围参数。
    parser.add_argument("--position_granularity", type=str, default=PhishingConfig.position_granularity)  # URL 位置粒度参数。
    parser.add_argument("--include_boundary_tokens", action=argparse.BooleanOptionalAction, default=PhishingConfig.include_boundary_tokens)  # 是否保留边界 token。
    parser.add_argument("--save_position_vocab_meta", action=argparse.BooleanOptionalAction, default=PhishingConfig.save_position_vocab_meta)  # 是否保存词表元信息。
    parser.add_argument("--lowercase_url", action=argparse.BooleanOptionalAction, default=PhishingConfig.lowercase_url)  # URL 是否统一小写。

    parser.add_argument("--url_embed_dim", type=int, default=PhishingConfig.url_embed_dim)  # URL 嵌入维度参数。
    parser.add_argument("--url_num_heads", type=int, default=PhishingConfig.url_num_heads)  # URL 注意力头数参数。
    parser.add_argument("--url_num_layers", type=int, default=PhishingConfig.url_num_layers)  # URL 编码器层数参数。
    parser.add_argument("--url_ffn_dim", type=int, default=PhishingConfig.url_ffn_dim)  # URL 前馈层维度参数。

    parser.add_argument("--traffic_input_dim", type=int, default=PhishingConfig.traffic_input_dim)  # 流量输入维度参数。
    parser.add_argument("--traffic_embed_dim", type=int, default=PhishingConfig.traffic_embed_dim)  # 流量嵌入维度参数。
    parser.add_argument("--traffic_num_layers", type=int, default=PhishingConfig.traffic_num_layers)  # 流量编码器层数参数。
    parser.add_argument("--traffic_expand_factor", type=int, default=PhishingConfig.traffic_expand_factor)  # 流量扩展倍数参数。
    parser.add_argument("--traffic_kernel_size", type=int, default=PhishingConfig.traffic_kernel_size)  # 流量卷积核大小参数。

    parser.add_argument("--fusion_dim", type=int, default=PhishingConfig.fusion_dim)  # 融合层维度参数。
    parser.add_argument("--num_phish_types", type=int, default=PhishingConfig.num_phish_types)  # 子类型类别数参数。

    parser.add_argument("--dropout", type=float, default=PhishingConfig.dropout)  # dropout 参数。
    parser.add_argument("--batch_size", type=int, default=PhishingConfig.batch_size)  # batch size 参数。
    parser.add_argument("--epochs", type=int, default=PhishingConfig.epochs)  # 训练轮数参数。
    parser.add_argument("--lr", type=float, default=PhishingConfig.lr)  # 学习率参数。
    parser.add_argument("--weight_decay", type=float, default=PhishingConfig.weight_decay)  # 权重衰减参数。
    parser.add_argument("--grad_clip", type=float, default=PhishingConfig.grad_clip)  # 梯度裁剪参数。
    parser.add_argument("--patience", type=int, default=PhishingConfig.patience)  # early stopping 耐心轮数参数。

    parser.add_argument("--focal_alpha", type=float, default=PhishingConfig.focal_alpha)  # Focal Loss alpha 参数。
    parser.add_argument("--focal_gamma", type=float, default=PhishingConfig.focal_gamma)  # Focal Loss gamma 参数。
    parser.add_argument("--loss_beta", type=float, default=PhishingConfig.loss_beta)  # 多任务分类辅助损失权重参数。
    parser.add_argument("--loss_gamma", type=float, default=PhishingConfig.loss_gamma)  # 多任务回归辅助损失权重参数。

    parser.add_argument("--num_workers", type=int, default=PhishingConfig.num_workers)  # DataLoader worker 数参数。
    parser.add_argument("--seed", type=int, default=PhishingConfig.seed)  # 随机种子参数。
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=PhishingConfig.device)  # 设备选择参数。

    parser.add_argument("--checkpoint_dir", type=str, default=PhishingConfig.checkpoint_dir)  # checkpoint 输出目录参数。
    parser.add_argument("--log_dir", type=str, default=PhishingConfig.log_dir)  # 日志目录参数。
    parser.add_argument("--vocab_path", type=str, default=PhishingConfig.vocab_path)  # 词表文件路径参数。

    parser.add_argument("--threshold_metric", type=str, choices=["f1", "precision", "recall"], default=PhishingConfig.threshold_metric)  # 阈值搜索指标参数。
    parser.add_argument("--save_last_checkpoint", action=argparse.BooleanOptionalAction, default=PhishingConfig.save_last_checkpoint)  # 是否保存最后 checkpoint 参数。
    parser.add_argument("--predict_batch_size", type=int, default=PhishingConfig.predict_batch_size)  # 推理批大小参数。
    parser.add_argument("--use_traffic", action=argparse.BooleanOptionalAction, default=PhishingConfig.use_traffic)  # 是否启用流量分支参数。
    return parser  # 返回配置完成的命令行解析器。


def get_config(argv: list[str] | None = None) -> PhishingConfig:  # 解析命令行并得到配置对象。
    """解析命令行参数并返回配置对象。"""  # 方法说明：供 train/evaluate/predict 统一调用。
    parser = build_parser()  # 先构建解析器。
    namespace = parser.parse_args(argv)  # 解析传入的命令行参数。
    config = PhishingConfig.from_dict(vars(namespace))  # 将解析结果转换为配置对象。
    config.ngram_range = tuple(int(value) for value in config.ngram_range)  # 确保 ngram_range 最终是 tuple[int, int]。
    return config  # 返回规范化后的配置对象。
