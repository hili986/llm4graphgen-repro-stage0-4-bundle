# LLM4GraphGen 复现仓库（Stage0）

本仓库用于从零开始复现论文 arXiv:2403.14358（LLM4GraphGen）。

当前进度：已完成 Stage4 Property-based（MolHIV：RDKit 合法性 + Unique/Novel + baseline 分类器 + CM/C）。

## 快速验证

```bash
python -m llm4graphgen.smoke
python -m llm4graphgen.stage1_smoke
python -m llm4graphgen.stage2_rule_based
python -m llm4graphgen.stage3_distribution
python -m llm4graphgen.stage4_property
python -m pytest -q
```

## Make 目标

```bash
make check
make stage0_summary
make stage1_smoke
make stage2_rule
make stage3_dist
make stage4_property
```
