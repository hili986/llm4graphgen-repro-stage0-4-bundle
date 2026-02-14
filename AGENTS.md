# AGENTS.md — Codex 项目规则（必须遵守）

本仓库目标：从零开始复现 arXiv:2403.14358（LLM4GraphGen）。
硬约束：分阶段交付、每阶段可视验收、中文日志与报告、客观自评、可返工。

## 0) 语言与输出（硬性）

- 所有面向用户的产物必须中文：验收报告/日志/总结/报错说明（代码变量名与库名除外）。
- 每个阶段结束必须生成：
  - `runs/stageX/验收报告.md`（中文）
  - `runs/stageX/关键输出/`（可直接打开检查：CSV/样例输出/失败案例汇总等）
- 完成一个阶段后必须停止，等待用户验收；不得自行进入下一阶段。

## 1) 规格来源（硬性）

- `SPEC.md` 是本项目“零歧义规范”。实现、指标口径、输出格式以 SPEC 为准。
- 若论文信息不完整，必须在验收报告中明确写出“不确定/偏差来源”，不得臆测“官方开源”。

## 2) 分阶段范围（硬性）

Stage0：仓库骨架 + 复现规格 + 中文日志规范（只建框架与最小测试）
Stage1：最小闭环（不依赖真实 LLM）：Provider 抽象 + MockProvider + 解析器 + smoke CLI
Stage2：Rule-based 8 任务：valid/unique/novel + 单测 + 出表（CSV）
Stage3：Distribution-based 3 任务：ppred/pgen + motif 匹配 + 出表（CSV）
Stage4：Property-based（MolHIV）：RDKit 合法性 + Unique/Novel + baseline 分类器 + CM/C + 总报告

## 3) 工程与可追溯（硬性）

- 必须有 `make check`（跑 pytest）且每阶段验收报告包含“我如何复核”的命令清单。
- 每次运行输出到 `runs/<timestamp>/`，并写 `run.log`（中文）。
- 所有 LLM 输入/输出必须落盘（JSONL），至少包含：prompt、模型名、温度、原始输出、解析结果、解析失败原因。

## 4) LLM 与密钥（硬性）

- API key 只从环境变量读取（如 `OPENAI_API_KEY`），不得写入代码/日志/报告。
- 必须实现 Provider 抽象：至少支持 MockProvider + OpenAIProvider。

## 5) 每阶段验收报告模板（硬性）

`runs/stageX/验收报告.md` 必须包含以下栏目（缺一不可）：

1) 阶段目标（一句话）
2) 完成清单（逐条）
3) 复核方法（给出可复制命令）
4) 关键输出（路径 + 摘要）
5) 偏差与不确定性（明确说明原因）
6) 客观自评：
   - 完成度：0-100（说明依据）
   - 复现对齐度：高/中/低（说明依据）
   - 风险清单（按严重程度排序）
   - 建议用户重点检查的 5 项 checklist
   - 若需返工：最小改动路径（不要推倒重来）
