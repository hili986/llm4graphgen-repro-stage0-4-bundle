# SPEC.md — LLM4GraphGen（arXiv:2403.14358）复现规格（中文）

## 1. 复现目标
- 先达成：流程复现（任务定义/格式/解析/评测/可追溯日志）
- 再追求：现象复现（趋势接近论文；不强求数值一致）
- 数值级复现：仅在模型/提示词/采样完全对齐时尝试

## 2. 输出格式
### 2.1 图结构输出
统一格式：`(n, [(u,v), (u,v), ...])`
解析要求：容错空格/换行/尾逗号/重复边；失败必须给中文原因。

### 2.2 分子输出
SMILES 字符串；用 RDKit 校验合法性；不合法必须记录原因与样例。

## 3. 任务范围
### 3.1 Rule-based（8 约束）
Tree / Cycle / Planar / #Components / k-regular / Wheel / Bipartite / k-coloring
指标：valid / unique / novel（novel 判定口径在代码与报告中写清）

### 3.2 Distribution-based（3 子任务）
Trees-or-cycles / Union-of-components / Motif
指标：p_pred（解析自 LLM 输出）、p_gen（统计生成样本）
必须统计：解析失败率、判别失败率（不得吞错）

### 3.3 Property-based（MolHIV）
数据：OGBG-MolHIV（正类 few-shot）
指标：CM、rectified C、Unique、Novel
注意：分类器细节论文可能不完整 → 报告必须说明偏差来源。
