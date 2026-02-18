# LLM4GraphGen 复现仓库

本仓库从零开始复现论文 arXiv:2403.14358（LLM4GraphGen），对齐论文的实验设计。

## 改进特性（v0.3.0）

基于 v2 复现分析报告的 P0-P3 全部改进建议 + 方案 C 分类器对齐：

| 改进项 | 优先级 | 说明 |
|--------|--------|------|
| **GIN 分类器（方案 C）** | P0+ | OGB 官方 GIN baseline，与论文 "GNN classifier" 架构一致 |
| **论文 Table 10 TPR/FPR** | P0+ | `--paper-tpr-fpr` 使用论文精确值 TPR=0.5615/FPR=0.1045 作为参照 |
| OGBG-MolHIV 全量 LR 分类器 | P0 | 替换 24 样本过拟合分类器，在 ~41K 分子上训练 |
| Stage4 样本量提升至 100+ | P1 | 多次 LLM 调用自动拼接，默认生成 100 个分子 |
| Stage3 全策略 + 多重复 | P1/P3 | `--strategy all` 一键运行 4 种策略，`--num-repeats` 计算均值±标准差 |
| 图同构去重 (WL hash) | P2 | 替代边排序签名，正确处理节点重标号，对齐论文 Unique Rate 定义 |
| Small/Medium/Large 消融 | P2 | `--size small/medium/large` 对齐论文 Table 4/8 |
| 多模型支持 | P3 | 支持 GPT-4/GPT-3.5-Turbo/开源模型切换 |

---

## 快速验证（Mock 模式，无需 API Key）

以下命令在 PowerShell 和 Linux/Mac 终端均可直接使用：

```
python -X utf8 -m llm4graphgen.smoke
python -X utf8 -m llm4graphgen.stage2_rule_based --num-samples 10
python -X utf8 -m llm4graphgen.stage3_distribution --p-values 0.5
python -X utf8 -m llm4graphgen.stage4_property
python -X utf8 -m pytest -q
```

---

## 真实 LLM 实验（v3 完整命令）

### Windows PowerShell

```powershell
# 设置 API Key
$env:OPENAI_API_KEY = "sk-你的key"

# ===== Stage2: Rule-based 评测 =====

# 单策略运行（对齐论文默认配置）
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot --num-samples 100 --num-repeats 3

# 4 种策略逐个运行（Medium 规模）
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot --num-samples 100
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy few_shot --num-samples 100
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot_cot --num-samples 100
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy few_shot_cot --num-samples 100

# [v3 新增] Small/Large 规模消融
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot --num-samples 100 --size small
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot --num-samples 100 --size large

# [v3 新增] 使用图同构去重（默认）或边排序签名
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot --num-samples 100 --unique-method isomorphism
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot --num-samples 100 --unique-method signature

# ===== Stage3: Distribution-based =====

# 单策略 p 值扫描
python -m llm4graphgen.stage3_distribution --provider openai --model gpt-4 --temperature 0.5 --strategy zero_shot_cot --p-values 0.2,0.4,0.6,0.8

# [v3 新增] 一键运行全部 4 种策略
python -m llm4graphgen.stage3_distribution --provider openai --model gpt-4 --temperature 0.5 --strategy all --p-values 0.2,0.4,0.6,0.8

# [v3 新增] 多次重复 + 统计
python -m llm4graphgen.stage3_distribution --provider openai --model gpt-4 --temperature 0.5 --strategy zero_shot_cot --p-values 0.2,0.4,0.6,0.8 --num-repeats 3

# ===== Stage4: Property-based (MolHIV) =====

# [v3 推荐 - 方案C] GIN 分类器 + 论文 Table 10 TPR/FPR 参照
pip install ogb pandas torch torch_geometric
python -m llm4graphgen.stage4_property --provider openai --model gpt-4 --temperature 0.5 --strategy few_shot_cot --num-generate 100 --classifier gin --paper-tpr-fpr

# [v3 推荐] GIN 分类器，使用自身 TPR/FPR
python -m llm4graphgen.stage4_property --provider openai --model gpt-4 --temperature 0.5 --strategy few_shot_cot --num-generate 100 --classifier gin

# OGBG-MolHIV 全量 LR 分类器
python -m llm4graphgen.stage4_property --provider openai --model gpt-4 --temperature 0.5 --strategy few_shot_cot --num-generate 100 --classifier ogbg

# 使用 proxy 分类器（向后兼容，不推荐）
python -m llm4graphgen.stage4_property --provider openai --model gpt-4 --temperature 0.5 --strategy few_shot_cot --num-generate 100 --classifier proxy

# ===== [v3 新增] 多模型对比 =====
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-3.5-turbo --temperature 0.8 --strategy zero_shot --num-samples 100
```

### Linux / macOS (Bash)

```bash
# 设置 API Key
export OPENAI_API_KEY="sk-你的key"

# Stage2: 4 种策略 × 3 种规模完整消融
for size in small medium large; do
  for strategy in zero_shot few_shot zero_shot_cot few_shot_cot; do
    python -m llm4graphgen.stage2_rule_based \
      --provider openai --model gpt-4 --temperature 0.8 \
      --strategy $strategy --num-samples 100 --size $size
  done
done

# Stage3: 全策略 + 3 次重复
python -m llm4graphgen.stage3_distribution \
  --provider openai --model gpt-4 --temperature 0.5 \
  --strategy all --p-values 0.2,0.4,0.6,0.8 --num-repeats 3

# Stage4: GIN 分类器 + 论文 TPR/FPR 参照 (方案C，最完整)
python -m llm4graphgen.stage4_property \
  --provider openai --model gpt-4 --temperature 0.5 \
  --strategy few_shot_cot --num-generate 100 \
  --classifier gin --paper-tpr-fpr

# Stage4: GIN 分类器，使用自身 TPR/FPR
python -m llm4graphgen.stage4_property \
  --provider openai --model gpt-4 --temperature 0.5 \
  --strategy few_shot_cot --num-generate 100 --classifier gin
```

---

## 一键批量运行（推荐）

使用实验运行器自动化执行全部实验，支持并行、断点续跑、API 错误自动检测：

```bash
# 1. 先预览实验计划（不运行）
python -m llm4graphgen.experiment_runner --plan standard --dry-run

# 2. 开始运行（2 个实验并行）
python -m llm4graphgen.experiment_runner --plan standard --parallel 2

# 3. 中断后续跑（直接重新运行同一命令，已完成的自动跳过）
python -m llm4graphgen.experiment_runner --plan standard --parallel 2

# 只运行某个 Stage
python -m llm4graphgen.experiment_runner --plan standard --stage stage2

# 用其他模型
python -m llm4graphgen.experiment_runner --plan standard --model gpt-3.5-turbo

# 完整消融（4策略 × 3规模 + 多模型对比，共 19 个实验）
python -m llm4graphgen.experiment_runner --plan full --parallel 2

# 从头开始（清除续跑记录）
python -m llm4graphgen.experiment_runner --plan standard --reset
```

三种实验计划：

| 计划 | 实验数 | 内容 |
|------|--------|------|
| `minimal` | 3 | 最小验证：1 策略 × medium，快速跑通流程 |
| `standard` | 7 | 标准复现：4 策略 × medium × 3 repeats，对齐论文核心结果 |
| `full` | 19 | 完整消融：4 策略 × 3 规模 × 3 repeats + GPT-3.5-Turbo 对比 |

特性：
- 遇到 API 额度不足/认证失败自动停止并报告
- 断点续跑：状态记录在 `runs/experiment_status.json`
- Ctrl+C 安全中断，下次续跑

---

## 运行测试

```
python -X utf8 -m pytest -q
```

---

## 环境变量设置速查

| 终端 | 设置环境变量 |
|------|------------|
| **Windows PowerShell** | `$env:OPENAI_API_KEY = "sk-xxx"` |
| **Windows CMD** | `set OPENAI_API_KEY=sk-xxx` |
| **Linux / macOS** | `export OPENAI_API_KEY="sk-xxx"` |

---

## 依赖安装

```bash
# 基础依赖（Stage2 Rule-based）
pip install networkx

# Stage4 分子评测（proxy 分类器）
pip install rdkit numpy scikit-learn

# Stage4 OGBG-MolHIV LR 分类器
pip install ogb pandas torch

# Stage4 GIN 分类器（推荐，最忠实复现）
pip install ogb pandas torch torch_geometric

# 开发依赖
pip install pytest
```

---

## v0.2.0 改进特性

- **图规模对齐论文 Table 7**：Tree/Cycle/Planar/Wheel=15节点, k-regular=16节点, Bipartite=10节点
- **精确平面性检测**：使用 NetworkX Boyer-Myrvold 算法替代近似检查
- **4 种 Prompting 策略**：Zero-shot / Few-shot / Zero-shot+CoT / Few-shot+CoT
- **p 值扫描实验**：Distribution-based 任务支持 p=0.2,0.4,0.6,0.8
- **修正 rectified_C 公式**：`C(G) = (C_M(G) - FPR) / (TPR - FPR)`
- **支持真实 LLM 调用**：通过 `--provider openai` 接入 GPT-4 等模型
- **多次重复实验**：支持 `--num-repeats` 计算均值±标准差
