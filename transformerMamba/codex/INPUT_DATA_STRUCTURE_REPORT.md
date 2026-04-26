# 项目输入数据结构说明

## 1. 项目输入数据结构总览

本项目是一个“基于 URL 字符串与流量时序的多模态钓鱼网站检测系统”。从代码实现看，项目的输入数据可以分成两层：

- **原始输入层**：来自 JSON / JSONL 文件，或推理阶段直接由命令行提供的单条 URL
- **模型输入层**：由 `dataset.py` 解析、校验、编码、padding 后生成的张量字典

当前代码中的主要输入来源如下：

- 训练：
  - 默认读取 `data/train.json`
  - 验证读取 `data/val.json`
  - 入口：`train.py -> load_records -> build_dataloader`
- 评估：
  - 默认读取 `data/test.json`
  - 入口：`evaluate_test.py -> load_records -> build_dataloader`
- 推理：
  - 单条模式：`predict.py --url ...`
  - 批量模式：`predict.py --input_file ...`
  - 入口：`predict.py -> PhishingDataset / collate_fn`

### 1.1 输入数据的整体组织方式

从真实代码看，输入数据存在三种外层组织形式：

1. **JSON 数组文件**
   - 例如 `data/train.json`
   - 文件顶层是 `List[Dict]`
2. **JSONL 文件**
   - 每行一个 JSON 对象
   - 由 `dataset.py:load_records` 支持
3. **推理阶段临时构造的 Python 列表**
   - 单条 URL 推理时由 `predict.py` 临时构造：
     ```python
     [{"url": args.url, "traffic": []}]
     ```

> 基于代码推断：项目默认示例数据和训练流程主要使用 JSON 数组格式，JSONL 是兼容性支持而不是主路径。

---

## 2. 原始输入数据结构

## 2.1 原始样本顶层结构

单条样本在代码中的原始结构是一个 `dict`，典型形式如下：

```json
{
  "url": "https://example.com/login",
  "traffic": [
    [0.01, 120.0],
    [0.02, 250.0],
    [0.03, 512.0]
  ],
  "label": 1,
  "phish_type": 1,
  "risk_score": 0.8
}
```

### 2.1.1 字段说明

| 字段名 | 类型 | 训练是否必填 | 推理是否必填 | 默认值 / 缺省行为 | 含义 |
|---|---|---:|---:|---|---|
| `url` | `str` | 是 | 是 | 无默认值；为空或非字符串会报错 | 待检测网址字符串 |
| `traffic` | `list` / `tuple` | 是 | 否 | 推理缺失时可补为空列表 `[]` | 与 URL 对应的流量时序 |
| `label` | `int` | 是 | 否 | 推理时内部补为 `0` | 二分类标签，通常 `0=正常, 1=钓鱼` |
| `phish_type` | `int` | 是 | 否 | 推理时内部补为 `0` | 钓鱼子类型标签 |
| `risk_score` | `int` / `float` | 是 | 否 | 推理时内部补为 `0.0` | 风险分数字段 |

### 2.1.2 代码证据

- 样本读取：
  - `dataset.py -> load_records`
- schema 校验：
  - `dataset.py -> validate_record_schema`
- 训练强制要求标签：
  - `build_dataloader(..., require_targets=True)` 默认值
- 推理允许缺省 `traffic`：
  - `predict.py -> PhishingDataset(..., require_targets=False, allow_missing_traffic=True)`

---

## 2.2 `traffic` 字段的嵌套结构

`traffic` 是一个序列，序列中的每个元素支持以下几种形式：

### 2.2.1 标准形式

```json
"traffic": [
  [timestamp, size],
  [timestamp, size],
  ...
]
```

例如：

```json
"traffic": [
  [0.01, 120.0],
  [0.02, 250.0],
  [0.03, 512.0]
]
```

这是当前项目最核心、最推荐、也最符合主链路设计的输入形式。

### 2.2.2 兼容形式 1：单元素列表/元组

```json
"traffic": [
  [0.01],
  [0.02],
  [0.03]
]
```

代码会把缺失的 `size` 自动补成 `0.0`。

### 2.2.3 兼容形式 2：纯标量列表

```json
"traffic": [0.01, 0.02, 0.03]
```

代码会将每个标量视为时间戳，并自动补 `size = 0.0`。

### 2.2.4 不允许的形式

以下情况会在 schema 校验阶段直接失败：

- `traffic` 不是 `list` 或 `tuple`
- `traffic` 中某个元素是长度不是 1 或 2 的列表/元组
- `traffic` 中某个值不是 `int` / `float`

### 2.2.5 `traffic` 校验规则

由 `dataset.py -> _validate_traffic_payload` 实现，规则如下：

- 如果训练/评估阶段缺少 `traffic`，报错
- 如果推理阶段缺少 `traffic`，允许并按空流量处理
- 每个 packet 元素必须是：
  - 标量数值
  - 或长度为 1/2 的列表或元组

---

## 3. 多种输入格式之间的关系与使用场景

## 3.1 训练 / 验证 / 测试输入

### 3.1.1 训练

- 来源：
  - `config.train_path`
  - 默认值：`data/train.json`
- 使用路径：
  - `train.py -> load_records(config.train_path)`
- 要求：
  - `url`
  - `traffic`
  - `label`
  - `phish_type`
  - `risk_score`
  都必须存在并通过类型校验

### 3.1.2 验证

- 来源：
  - `config.val_path`
  - 默认值：`data/val.json`
- 使用路径：
  - `train.py -> load_records(config.val_path)`
- 结构要求：
  - 与训练集相同

### 3.1.3 测试评估

- 来源：
  - `config.test_path`
  - 或 `evaluate_test.py --test_path`
- 使用路径：
  - `evaluate_test.py -> load_records(args.test_path)`
- 结构要求：
  - 与训练集相同

## 3.2 推理输入

### 3.2.1 单条 URL 推理

来源：

```bash
python predict.py --url "https://example.com"
```

代码中会构造成：

```python
[{"url": args.url, "traffic": []}]
```

特点：

- 不要求 `label`
- 不要求 `phish_type`
- 不要求 `risk_score`
- 允许没有真实流量，默认按空流量处理

### 3.2.2 批量文件推理

来源：

```bash
python predict.py --input_file data/test.json
```

要求：

- 每条样本至少需要 `url`
- `traffic` 可缺失，缺失时按空流量处理
- 其他监督字段不是必须的

> 基于代码推断：批量推理可以直接复用训练/测试样本文件，也可以传入只包含 `url` 与可选 `traffic` 的轻量文件。

---

## 4. 关键数据类型 / 字段说明

## 4.1 配置层输入

配置本身也是项目的“输入”之一，影响数据读取与编码方式。关键字段如下：

| 配置项 | 类型 | 默认值 | 作用 |
|---|---|---:|---|
| `train_path` | `str` | `data/train.json` | 训练集路径 |
| `val_path` | `str` | `data/val.json` | 验证集路径 |
| `test_path` | `str` | `data/test.json` | 测试集路径 |
| `max_url_len` | `int` | `256` | URL 截断长度 |
| `max_traffic_len` | `int` | `512` | 流量时序截断长度 |
| `ngram_min_freq` | `int` | `1` | 词表最小频次 |
| `vocab_1gram_max_size` | `int` | `256` | 1-gram 词表大小上限 |
| `vocab_2gram_max_size` | `int` | `4096` | 2-gram 词表大小上限 |
| `vocab_3gram_max_size` | `int` | `8192` | 3-gram 词表大小上限 |
| `lowercase_url` | `bool` | `False` | 是否统一将 URL 转为小写 |
| `batch_size` | `int` | `64` | 训练/评估 DataLoader batch size |
| `predict_batch_size` | `int` | `128` | 推理分批大小 |

配置来源：

- 默认值：`config.py -> PhishingConfig`
- CLI 覆盖：`config.py -> build_parser`
- checkpoint 恢复：`PhishingConfig.from_dict(checkpoint["config"])`

---

## 4.2 原始输入字段的语义说明

### `url`

- 类型：`str`
- 是否必填：
  - 训练：是
  - 推理：是
- 校验规则：
  - 必须是字符串
  - `strip()` 后不能为空
- 作用：
  - 用于提取 1/2/3-gram
  - 是 URL 编码器的源输入

### `traffic`

- 类型：`Sequence[Any]`
- 是否必填：
  - 训练：是
  - 推理：否
- 支持内容：
  - `[timestamp, size]`
  - `[timestamp]`
  - `timestamp`
- 作用：
  - 经 `parse_traffic_sequence` 转为 `[delta_time, size]`
  - 是流量编码器的源输入

### `label`

- 类型：`int`
- 是否必填：
  - 训练：是
  - 评估：是
  - 推理：否
- 作用：
  - 二分类监督信号

### `phish_type`

- 类型：`int`
- 是否必填：
  - 训练：是
  - 评估：是
  - 推理：否
- 作用：
  - 多分类辅助监督

### `risk_score`

- 类型：`int` 或 `float`
- 是否必填：
  - 训练：是
  - 评估：是
  - 推理：否
- 作用：
  - 回归辅助监督

---

## 5. 示例输入

## 5.1 训练/评估样本示例

```json
{
  "url": "https://PayPal-secure-123.xyz/login",
  "traffic": [
    [0.012, 180.0],
    [0.018, 920.0],
    [0.034, 1440.0]
  ],
  "label": 1,
  "phish_type": 1,
  "risk_score": 0.87
}
```

## 5.2 推理样本示例（带流量）

```json
{
  "url": "https://example.com/login",
  "traffic": [
    [0.01, 120.0],
    [0.02, 250.0]
  ]
}
```

## 5.3 推理样本示例（仅 URL）

```json
{
  "url": "https://example.com/login"
}
```

> 基于代码推断：若推理阶段缺少 `traffic`，代码会将其当作空流量处理。

---

## 6. 模型输入层的数据结构

原始输入不会直接送入模型，而是先在 `PhishingDataset.__getitem__` 和 `collate_fn` 中变成模型消费的 batch 字典。

## 6.1 单样本编码后的结构

`PhishingDataset.__getitem__` 返回：

```python
{
  "ids_1gram": torch.LongTensor,
  "ids_2gram": torch.LongTensor,
  "ids_3gram": torch.LongTensor,
  "traffic": torch.FloatTensor,      # [T, 2]
  "label": torch.LongTensor,
  "phish_type": torch.LongTensor,
  "risk_score": torch.FloatTensor,
  "url": str
}
```

### 说明

- `ids_1gram` / `ids_2gram` / `ids_3gram`
  - 来自 `encode_url_to_ngrams`
- `traffic`
  - 来自 `parse_traffic_sequence`
- 标签字段
  - 训练/评估使用真实值
  - 推理场景填充为零占位值

## 6.2 批量拼装后的结构

`collate_fn` 返回：

```python
{
  "ids_1gram": Tensor[B, L1],
  "ids_2gram": Tensor[B, L2],
  "ids_3gram": Tensor[B, L3],
  "url_mask": BoolTensor[B, L1],
  "traffic_feats": FloatTensor[B, T, 2],
  "traffic_mask": BoolTensor[B, T],
  "label": Tensor[B],
  "phish_type": Tensor[B],
  "risk_score": Tensor[B],
  "urls": List[str]
}
```

### 字段语义

| 字段名 | 类型 | 说明 |
|---|---|---|
| `ids_1gram` | `Tensor[B, L1]` | 1-gram token ID 批量 |
| `ids_2gram` | `Tensor[B, L2]` | 2-gram token ID 批量 |
| `ids_3gram` | `Tensor[B, L3]` | 3-gram token ID 批量 |
| `url_mask` | `BoolTensor[B, L1]` | URL padding 掩码，`True` 表示有效 token |
| `traffic_feats` | `FloatTensor[B, T, 2]` | 流量序列批量，最后一维固定为 `[delta_time, size]` |
| `traffic_mask` | `BoolTensor[B, T]` | 流量有效位置掩码 |
| `label` | `Tensor[B]` | 二分类标签 |
| `phish_type` | `Tensor[B]` | 子类型标签 |
| `risk_score` | `Tensor[B]` | 风险分数标签 |
| `urls` | `List[str]` | 原始 URL 列表，便于调试或输出 |

---

## 7. 数据流说明

## 7.1 输入在哪里被读取

### 训练

```text
train.py
  -> get_config()
  -> load_records(config.train_path)
  -> load_records(config.val_path)
```

### 评估

```text
evaluate_test.py
  -> load_checkpoint(...)
  -> load_records(args.test_path)
```

### 推理

```text
predict.py
  -> --url            -> 构造 [{"url": ..., "traffic": []}]
  -> --input_file     -> load_records(args.input_file)
```

## 7.2 输入如何被解析、校验、转换和传递

### 第一步：读取

- 函数：`dataset.py -> load_records`
- 行为：
  - 先读取文件文本
  - 若首字符是 `[`，按 JSON 数组解析
  - 否则按 JSONL 逐行解析

### 第二步：schema 校验

- 函数：`dataset.py -> validate_record_schema`
- 校验内容：
  - 顶层必须是 `dict`
  - `url` 必须是非空字符串
  - `traffic` 类型和内部元素格式要合法
  - 训练/评估阶段要求 `label`、`phish_type`、`risk_score` 存在

### 第三步：URL 编码

- 函数链：
  - `normalize_url`
  - `extract_ngrams`
  - `encode_url_to_ngrams`
- 行为：
  - `strip()` 去空白
  - 根据配置决定是否小写
  - 按 1/2/3-gram 切分
  - 根据词表映射成 ID 序列

### 第四步：流量转换

- 函数：
  - `parse_traffic_sequence`
- 行为：
  - 将原始 `traffic` 统一转换成 `[delta_time, size]`
  - 若只有时间值，则 `size = 0.0`
  - 若流量为空，则输出 `[[0.0, 0.0]]`

### 第五步：批量组装

- 函数：
  - `collate_fn`
- 行为：
  - 对 URL token 序列做 padding
  - 对流量序列做 padding
  - 构造 `url_mask`
  - 构造 `traffic_mask`

### 第六步：被核心模块消费

最终被以下核心模块消费：

- `models/url_encoder.py -> URLTransformerEncoder`
  - 消费：
    - `ids_1gram`
    - `ids_2gram`
    - `ids_3gram`
    - `url_mask`
- `models/traffic_encoder.py -> TrafficMambaEncoder`
  - 消费：
    - `traffic_feats`
    - `traffic_mask`
- `models/detector.py -> PhishingDetector.forward`
  - 汇总消费以上全部模型输入

---

## 8. 需要特别注意的约束、边界条件与隐含假设

## 8.1 URL 大小写策略

- 当前默认：
  - `lowercase_url = False`
- 含义：
  - 默认保留 URL 原始大小写模式
- 特别注意：
  - 如果加载的是历史 checkpoint，而该模型是在“小写 URL”策略下训练的，则虽然代码能跑，但输入分布可能不一致

## 8.2 `traffic` 的容错性较强

- 允许：
  - `[timestamp, size]`
  - `[timestamp]`
  - `timestamp`
- 影响：
  - 这提高了兼容性
  - 但也意味着输入语义可能并不总是严格一致

## 8.3 训练与推理对字段要求不同

- 训练/评估：
  - 要求监督字段完整
- 推理：
  - 只强制要求 `url`
  - `traffic` 可缺省
  - 标签字段不是必须

## 8.4 空流量的隐含处理

- 若 `traffic` 为空：
  - `parse_traffic_sequence` 返回 `[[0.0, 0.0]]`
- 隐含假设：
  - 模型可以接受“无流量信息”的占位输入

## 8.5 词表构建依赖训练集 URL

- 词表来源：
  - `train.py -> build_url_vocabs((record.get("url", "") for record in train_records), config)`
- 当前代码已将可迭代对象转成列表，避免多次消费 generator
- 隐含假设：
  - 训练集 URL 分布能够代表推理阶段 URL token 分布

## 8.6 基于代码推断与不确定之处

### 基于代码推断

- `label` 很可能约定为：
  - `0 = 正常`
  - `1 = 钓鱼`
- `phish_type` 的数值含义在代码中没有集中枚举定义，但从数据生成脚本可推断：
  - `0` 表示正常
  - `1~4` 表示不同钓鱼子类型

### 不确定之处

- `risk_score` 的精确定义（人工标注、规则打分、还是外部模型得分）在当前代码中没有完整说明
- `traffic` 的真实来源在当前仓库中没有体现，仓库仅提供了模拟数据生成脚本
- JSONL 是否在真实项目使用中广泛使用，当前只能确认代码支持，无法确认实际主用场景

---

## 9. 总结

项目输入可以概括为：

- **原始层**：`dict` 样本，核心字段是 `url` 与 `traffic`
- **监督层**：训练/评估额外要求 `label`、`phish_type`、`risk_score`
- **模型层**：URL 被编码成 1/2/3-gram ID 序列，流量被编码成 `[delta_time, size]` 张量，再通过 `collate_fn` 生成 batch 输入

最重要的理解点有三个：

1. 训练、评估、推理共用同一套数据编码主链路，但字段要求严格度不同
2. `traffic` 支持多种兼容格式，但模型最终只消费统一的二维时序形式
3. URL 大小写是否保留由配置控制，当前默认保留原始大小写
