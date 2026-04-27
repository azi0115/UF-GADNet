# PKL数据文件输入格式说明

## 1. 总览

当前项目已经修改为**优先使用 `pkl` 文件作为输入数据**。

默认配置路径如下：

- 训练集：`data/b_data/data/split/train.pkl`
- 验证集：`data/b_data/data/split/valid.pkl`
- 测试集：`data/b_data/data/split/test.pkl`

核心数据读取入口在：

- [dataset.py](E:\EDevelop\WF\Website-Fingerprinting-Library-master\transformerMamba\dataset.py) 的 `load_records(path)`

当前支持的输入文件格式：

1. `pkl` / `pickle`
2. `json` 数组文件
3. `jsonl` 文件

其中 **`pkl` 现在是默认使用格式**。

---

## 2. pkl 文件的顶层结构

`pkl` 文件反序列化后，顶层必须是：

```python
List[Dict[str, Any]]
```

也就是：

- 一个 `list`
- 列表中的每个元素都是一条样本 `dict`

如果反序列化得到的不是 `list`，当前代码会直接报错。

---

## 3. 单条样本结构

每条样本的推荐结构如下：

```python
{
    "url": "https://login.pay.com/auth?id=12",
    "traffic": [
        [0.01, 120.0],
        [0.02, 256.0],
        [0.03, 512.0]
    ],
    "label": 1,
    "phish_type": 1,
    "risk_score": 0.87
}
```

---

## 4. 字段说明

### 4.1 `url`

- 类型：`str`
- 训练是否必填：是
- 推理是否必填：是
- 作用：
  - URL n-gram 切分
  - 位置感知 URL n-gram 切分
  - URL 编码器输入来源

约束：

- 必须是字符串
- `strip()` 后不能为空

---

### 4.2 `traffic`

- 类型：`list` 或 `tuple`
- 训练是否必填：是
- 推理是否必填：否
- 作用：
  - 转换为 `[delta_time, size]`
  - 作为流量分支输入

支持的内容形式：

#### 标准形式

```python
"traffic": [
    [timestamp, size],
    [timestamp, size],
]
```

例如：

```python
"traffic": [
    [0.01, 120.0],
    [0.02, 256.0],
    [0.03, 512.0]
]
```

#### 兼容形式 1：单元素列表

```python
"traffic": [
    [0.01],
    [0.02],
]
```

此时缺失的 `size` 会被补为 `0.0`。

#### 兼容形式 2：纯标量序列

```python
"traffic": [0.01, 0.02, 0.03]
```

此时会视为只有时间戳，同样自动补 `size=0.0`。

推理阶段如果缺失 `traffic`：

- 允许
- 会被按空流量处理
- 最终编码为 `[[0.0, 0.0]]`

---

### 4.3 `label`

- 类型：`int`
- 训练是否必填：是
- 验证/测试是否必填：是
- 推理是否必填：否

作用：

- 二分类监督标签

基于当前代码与项目语义推断：

- `0` 通常表示正常
- `1` 通常表示钓鱼

---

### 4.4 `phish_type`

- 类型：`int`
- 训练是否必填：是
- 验证/测试是否必填：是
- 推理是否必填：否

作用：

- 多分类辅助标签

基于代码推断：

- `0` 通常表示正常样本
- `1~4` 表示不同类型的钓鱼样本

---

### 4.5 `risk_score`

- 类型：`int` 或 `float`
- 训练是否必填：是
- 验证/测试是否必填：是
- 推理是否必填：否

作用：

- 回归辅助监督标签

注意：

- 代码中不会强制检查它是否在 `[0, 1]`
- 但模型输出端使用 `sigmoid`，因此从设计上通常默认它应接近 `[0, 1]`

---

## 5. 训练 / 评估 / 推理的输入差异

## 5.1 训练与评估

训练、验证、测试阶段要求字段完整：

必须包含：

- `url`
- `traffic`
- `label`
- `phish_type`
- `risk_score`

否则会在 `validate_record_schema()` 阶段直接报错。

---

## 5.2 推理

推理阶段要求更宽松：

必须包含：

- `url`

可选：

- `traffic`

不要求：

- `label`
- `phish_type`
- `risk_score`

如果推理输入没有 `traffic`，会自动使用空流量占位。

---

## 6. 数据加载流程

数据加载主路径如下：

1. `train.py` / `evaluate_test.py` / `predict.py`
2. 调用 `load_records(path)`
3. 若后缀是 `.pkl` 或 `.pickle`
   - 使用 `pickle.load(...)`
4. 若后缀不是 `pkl`
   - 回退到 `json` / `jsonl` 解析逻辑
5. 得到 `List[Dict]`
6. 进入 `PhishingDataset`
7. 执行 schema 校验
8. 执行 URL 编码与流量转换
9. 经 `collate_fn` 变成模型输入 batch

---

## 7. 模型真正消费的中间结构

原始样本不会直接送入模型，而会被编码成如下结构：

### 单条样本

```python
{
    "ids_1gram": torch.LongTensor,
    "ids_2gram": torch.LongTensor,
    "ids_3gram": torch.LongTensor,
    "traffic": torch.FloatTensor,   # [T, 2]
    "label": torch.LongTensor,
    "phish_type": torch.LongTensor,
    "risk_score": torch.FloatTensor,
    "url": str
}
```

### 批量样本

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

---

## 8. pkl 文件示例生成方式

如果你已经有一份 JSON 数组格式数据，例如：

```python
records = [
    {
        "url": "https://example.com",
        "traffic": [[0.01, 100.0], [0.02, 200.0]],
        "label": 0,
        "phish_type": 0,
        "risk_score": 0.05,
    }
]
```

可以这样写成 `pkl`：

```python
import pickle

with open("train.pkl", "wb") as f:
    pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
```

---

## 9. 当前默认 pkl 文件位置

我已经在当前工作区中生成了这些默认 pkl 文件：

- `data/b_data/data/split/train.pkl`
- `data/b_data/data/split/valid.pkl`
- `data/b_data/data/split/test.pkl`
- `data/mock_data/train.pkl`
- `data/mock_data/val.pkl`
- `data/mock_data/test.pkl`

因此当前默认配置已经可以直接走 pkl 数据读取路径。

---

## 10. 特别注意事项

### 10.1 顶层必须是列表

不支持：

- 单个 dict 直接作为顶层
- DataFrame 直接 pickle 后作为顶层
- 自定义对象列表但内部不是 dict

推荐统一使用：

```python
List[Dict[str, Any]]
```

### 10.2 推理阶段也建议保留 `traffic`

虽然推理允许省略 `traffic`，但如果你的模型训练时启用了流量分支，推理时缺失流量会让该分支只看到空占位输入。

### 10.3 URL 不应为空

无论训练还是推理：

- `url` 必须存在
- `url.strip()` 后不能是空字符串

### 10.4 当前代码仍兼容 json/jsonl

虽然默认已经切到 `pkl`，但为了兼容旧数据：

- `.json`
- `.jsonl`

仍然可以继续使用。

也就是说，本次修改是：

- **默认使用 pkl**
- **同时兼容旧 json/jsonl**

---

## 11. 不确定与基于代码推断的部分

### 基于代码推断

- `label=0/1` 的业务语义是“正常/钓鱼”
- `phish_type=0~4` 的语义是“正常 + 4 类钓鱼”
- `risk_score` 设计上通常应接近 `[0, 1]`

### 不确定之处

- `risk_score` 是否一定来自人工标注或规则打分，当前代码未给出来源说明
- 真实生产数据中 `traffic` 是否始终是 `[timestamp, size]` 形式，当前代码只体现了兼容解析逻辑
