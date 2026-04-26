# CODEX_REFACTOR_REPORT

## 1. 实际修改了哪些文件

### `config.py`
- 修改目的：
  - 新增 `lowercase_url: bool = False`
  - 新增 `predict_batch_size: int = 128`
  - 让 URL 大小写策略与推理分批大小都能通过统一配置控制
- 兼容性风险：
  - 低
  - 旧 checkpoint 缺少这两个字段时，会通过 dataclass 默认值补齐

### `dataset.py`
- 修改目的：
  - 取消“默认无条件小写 URL”的行为，改为受显式配置控制
  - 增加样本 schema 校验
  - 区分训练/评估与推理场景的字段要求
  - 修复 `build_url_vocabs` 对 generator 重复消费的问题
- 兼容性风险：
  - 中
  - 若切换 `lowercase_url=False` 后重新训练，会改变词表与输入分布

### `utils.py`
- 修改目的：
  - 增加统一的二分类 logit / probability 提取函数
  - 增加统一阈值决策函数
  - 让评估逻辑和推理逻辑共用同一套“概率 -> 标签”转换语义
- 兼容性风险：
  - 低
  - 保留了旧输出字段的兼容读取路径

### `predict.py`
- 修改目的：
  - 增加推理分批处理
  - 增加 `--batch_size`
  - 推理阶段显式走统一 threshold 决策逻辑
  - 推理数据集使用非训练型 schema 规则
- 兼容性风险：
  - 低
  - 现有命令行接口仍可运行，只是新增了可选参数

### `models/detector.py`
- 修改目的：
  - 去掉模型内部硬编码 0.5 的最终离散 `pred`
  - 新增更清晰的 `binary_logit` / `binary_probability`
  - 暂时保留 `logits` / `main_logits` 兼容别名
  - 增加可选诊断输出入口
- 兼容性风险：
  - 中
  - 如果仓库外部代码显式依赖 `outputs["pred"]`，会受到影响

### `models/fusion.py`
- 修改目的：
  - 增加最小可观测性支持
  - 可选返回 gate 统计摘要
  - 保存最近一次 gate 统计结果
- 兼容性风险：
  - 低
  - 默认前向行为不变，只有显式开启诊断时才返回额外信息

---

## 2. 每项修改对应解决了哪个问题

### 问题 1：URL 默认统一小写，与设计目标不一致
- 修改：
  - 在 `config.py` 增加 `lowercase_url`
  - 在 `dataset.py` 中将 `normalize_url`、`extract_ngrams`、`build_ngram_vocab`、`build_url_vocabs`、`encode_url_to_ngrams` 全部接入该配置
  - 默认行为改为保留原始大小写，仅 `strip()`
- 影响范围：
  - 训练词表构建
  - 推理 URL 编码
  - 评估 URL 编码
- 风险：
  - 若切换大小写策略，会改变输入分布和词表统计

### 问题 2：模型内部 0.5 阈值和外部 threshold 语义并存
- 修改：
  - `models/detector.py` 不再输出内部固定 0.5 的 `pred`
  - 新增 `binary_logit` / `binary_probability`
  - `utils.py` 增加统一的 `extract_binary_probabilities` 与 `apply_binary_threshold`
  - `predict.py` 和 `utils.evaluate` 都使用同一套阈值决策函数
- 影响范围：
  - 推理
  - 评估
  - 模型输出字段语义
- 风险：
  - 若外部依赖历史 `pred` 字段，需要适配

### 问题 3：`predict_records` 一次性处理全部样本
- 修改：
  - `predict.py` 增加 `batch_size` 参数
  - `predict_records` 内部改为按 chunk 分批构建 `PhishingDataset + collate_fn`
- 影响范围：
  - 批量推理内存占用
  - 单样本与多样本推理路径
- 风险：
  - 低，输出顺序保持不变

### 问题 4：输入 schema 校验薄弱
- 修改：
  - `dataset.py` 新增 `_describe_value`、`_raise_schema_error`、`_validate_traffic_payload`、`validate_record_schema`
  - `PhishingDataset.__init__` 对全部样本做预校验
  - 训练/评估默认要求标签字段存在
  - 推理允许缺省 `traffic`
- 影响范围：
  - 训练数据加载
  - 评估数据加载
  - 推理输入样本
- 风险：
  - 一些过去被默认值“吞掉”的脏数据现在会尽早失败

### 问题 5：融合层缺少最小可观测性
- 修改：
  - `models/fusion.py` 增加 gate 统计摘要
  - `models/detector.py` 增加 `return_diagnostics`
  - 诊断信息以 `fusion_gate_stats` 形式可选返回
- 影响范围：
  - 排障与分析
- 风险：
  - 默认路径几乎无影响

### 问题 6：`main_logits` 命名语义不清
- 修改：
  - `models/detector.py` 新增清晰字段 `binary_logit`
  - 保留 `logits` / `main_logits` 作为兼容别名
- 影响范围：
  - 模型输出语义
  - 后续代码维护成本
- 风险：
  - 低到中
  - 当前仓库内兼容性已保留，但外部调用方需逐步迁移到新字段

### 额外修复：`build_url_vocabs` 对 generator 的重复消费
- 修改：
  - 在 `dataset.py` 中先将 `urls` 转成 `list`
- 影响范围：
  - 训练阶段 1/2/3-gram 词表构建正确性
- 风险：
  - 低
  - 这是一个明确的正确性修复

---

## 3. 哪些地方保持不变

以下模块我刻意没有重写：

### URL 主干编码器
- 未更换 `URLTransformerEncoder`
- 原因：
  - 本次目标是接口收敛和结构修正，不是替换主模型

### traffic 主干编码器
- 未更换 `TrafficMambaEncoder` / `MambaStyleBlock`
- 原因：
  - 当前问题不在 backbone 本身，而在输入、接口和诊断能力

### fusion 核心计算图
- 未改成 cross-attention 或更复杂融合器
- 原因：
  - 会超出最小改动范围

### 多任务损失主语义
- 未修改 `FocalLoss + CrossEntropy + SmoothL1` 的总体方向
- 原因：
  - 当前优先级不在损失策略重构

### 训练主流程
- `train.py` 的训练循环、优化器、调度器、保存策略没有大改
- 原因：
  - 本轮重点是输入/输出与接口一致性，不是重做训练框架

---

## 4. 兼容性说明

### checkpoint 是否兼容

#### 旧 checkpoint -> 新代码
- **大体兼容**
- 原因：
  - `PhishingConfig.from_dict` 会为缺失的新字段使用默认值
  - 模型权重结构未改动层形状
  - 输出字段新增不影响 `load_state_dict`

#### 需要注意的点
- 若旧 checkpoint 是在 URL 全部小写策略下训练出来的，而新运行配置默认 `lowercase_url=False`：
  - 运行时输入分布会变化
  - 虽然“能加载”，但不保证效果可比

### 配置是否新增字段
- 是
- 新增：
  - `lowercase_url`
  - `predict_batch_size`

### 旧接口是否还能运行
- 仓库内主链路仍能运行：
  - `train.py`
  - `evaluate_test.py`
  - `predict.py`
- 模型输出兼容策略：
  - 新字段：`binary_logit`、`binary_probability`
  - 兼容别名：`logits`、`main_logits`

### 若不兼容，如何迁移

#### 场景 1：依赖旧 `pred`
- 迁移方式：
  - 改为：
    1. 先读取 `binary_probability`（或 `logits` + `sigmoid`）
    2. 再显式应用外部 threshold

#### 场景 2：历史模型在“小写 URL”策略下训练
- 迁移方式：
  - 方案 A：加载旧 checkpoint 时显式设置 `lowercase_url=True`
  - 方案 B：重建词表并重新训练新模型

---

## 5. 自检结果

### train 流程是否仍可走通
- 静态上可走通
- 证据：
  - `python -m py_compile ...` 通过

### evaluate 流程是否仍可走通
- 静态上可走通
- 证据：
  - `evaluate_test.py` 语法检查通过
  - `utils.evaluate` 与新输出字段兼容

### predict 流程是否仍可走通
- 静态上可走通
- 证据：
  - `predict.py` 语法检查通过
  - 批量推理路径已改为分批

### threshold 是否只剩一套真实决策逻辑
- 是
- 说明：
  - 真实生效的最终决策现在统一为“概率 + 外部 threshold”
  - 模型内部不再输出固定 0.5 的最终 `pred`

### URL 是否默认保留大小写
- 是
- 说明：
  - 默认 `lowercase_url=False`

### 大批量 records 是否支持分批推理
- 是
- 说明：
  - `predict.py` 已按 `batch_size` 分 chunk 推理

### 静态检查结果
- `python -m py_compile ...`：通过
- `python -m pytest -q`：`1 passed, 1 skipped`

---

## 6. 仍未解决的问题

### 6.1 历史文档尚未全面同步
- 未处理内容：
  - `技术说明文档.md`
  - `ALGORITHM_DESIGN_LOGIC_REPORT.md`
  - `VALIDATION_AND_MINIMAL_CHANGE_PLAN.md`
- 原因：
  - 本轮任务重点在代码修正与报告交付，不在历史分析文档回写

### 6.2 `main_logits` 兼容别名仍然存在
- 未完全解决内容：
  - 字段语义已明确，但兼容别名尚未删除
- 原因：
  - 为了避免一次性破坏已有调用

### 6.3 融合层可观测性仍是“最小可用”
- 未处理内容：
  - 没有引入系统化日志、监控面板或训练期统计汇总
- 原因：
  - 遵循最小改动原则，只提供轻量级诊断接口

### 6.4 schema 校验仍是轻量实现
- 未处理内容：
  - 没有引入更严格的 schema 框架
  - 没有做跨字段语义校验
- 原因：
  - 避免对主链路造成过大扰动

### 6.5 多任务损失权重仍为静态常数
- 未处理内容：
  - 未引入动态加权、warmup 或任务平衡
- 原因：
  - 该项在当前优先级中属于次级问题
