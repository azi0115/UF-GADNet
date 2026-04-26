# PhishingDetector v2
基于 subword+Transformer + N-gram + 门的特征融合钓鱼检测系统。

网络结构: URL 子序列特征 + Bi-gram + 多工设计特征 + 网络层序号序列特征。通过门控机制融合多方面信号，输出综合判断。包含负采样和风险分层。

# 新增功能:

## 核心模块
- **URL 编码**: FastEmbed [1/2]-gram
- **RoPE-Transformer Encoder**: 位置旋转编码
- **2D CNN 上下文增强**: Human Texts

## 网络拓扑结构 (架构图 1/1)
```
[URL子序列] ─────┐
                  ├→ GateCrossModalFusion ──→ PhishingDetector
[流量特征] ───────┘                          ├── 主任务: 二分类
                                             ├── 辅助1: 风险分数
                                             └── 辅助2: 风险分类
```

## v2 核心优化点:

| 组件 | v1 | v2 | 说明 |
|------|----|----|------|
| Embed | 256 | 128 | 10万词表256维度后加投影 |
| Transformer | 层 | 6→3 | 细粒度、减少过拟合 |
| Projection | 无 | 有 | 上层 |
| wrlgate_Decoder | 1→3 | 6→3 | Transformer 卷积压缩至低维 |
| local_game | 1→2, 4→5 | 4.0→6.0 | 门控注意力范围扩大 |
| Focal Loss | 无 | ✅ | 处理正负样本不平衡 |
| Aux tasks | 无 | risk/blanc_everything/MLP | 多任务学习 |
| warmup | 0 | 10~20 | 辅助损失暖机 |
| StandardScaler | 否 | 是 | 数值型特征归一化 |
| Optuna search | 无 | TPE | 自动找超参 |

## 快速开始

```bash
# --- 安装依赖 ---
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 完整流程演示

### python data/generate_mock_data.py
生成模拟数据: data/train.json (5000条), data/val.json (500条), data/test.json (300条)

### 训练
```bash
python train.py \
  --train_path data/train.json \
  --val_path data/val.json \
  --epochs 20 \
  --lr 1e-4 \
  --device gpu
```

### 评测 (单?)
# python3 evaluate_test.py
```bash
python3 evaluate_test.py --url "http://paypal-secure-login.xyz/account/verify"
```

### 批量推理
# python3 predict.py
```bash
python predict.py --input_file data/test.json --output_file results.json
```

### 预测阈值调整
# python3 predict.py --url "http://example.com" --threshold 0.42

## 项目结构

phishing_detector/
├── README.md          # 项目文档 (本文件)
├── config.py          # PhishingConfig / N-gram / 2D辅助工作流后 / 配置管理器
├── dataset.py         # PhishingsDataset / Fusion (门的特征拼接逻辑)
├── loss.py           # WeightingLoss (Focal Loss + 辅助学习各阶段权重)
├── model.py          # 模型定义入口(GateCrossModalFusion + RoPE-Transfomer + WRL-gate)
├── traffic_encoder.py # TrafficTransformerEncoder (N-gram + Transformer + WRL-gate)
├── train.py          # 训练 (主流程)
├── val.json          # 验证集 (可选)
├── test_json         # 测试集 (可选)
├── generate_mock_data.py # 模拟数据 (生成)
├── train.json        # 训练集 (生成)
├── val.json          # 验证集 (可选)
├── test_json         # 测试集 (可选)
├── device_cpu        # 设备配置
├── model.pt          # 模型权重
├── scaler.pkl        # 归一化参数
├── Makefile          # 常用命令快捷入口
├── eval/             # 评测结果 / eval/evalluate/ / 评测速度 / checkpoint
├── checkpoints/      # 模型检查点
├── logs/             # 训练日志
└── ...