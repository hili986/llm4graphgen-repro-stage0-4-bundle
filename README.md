# LLM4GraphGen 复现仓库

本仓库从零开始复现论文 arXiv:2403.14358（LLM4GraphGen），对齐论文的实验设计。

## 改进特性（v0.2.0）

- **图规模对齐论文 Table 7**：Tree/Cycle/Planar/Wheel=15节点, k-regular=16节点, Bipartite=10节点
- **精确平面性检测**：使用 NetworkX Boyer-Myrvold 算法替代近似检查
- **4 种 Prompting 策略**：Zero-shot / Few-shot / Zero-shot+CoT / Few-shot+CoT
- **p 值扫描实验**：Distribution-based 任务支持 p=0.2,0.4,0.6,0.8
- **修正 rectified_C 公式**：`C(G) = (C_M(G) - FPR) / (TPR - FPR)`
- **支持真实 LLM 调用**：通过 `--provider openai` 接入 GPT-4 等模型
- **多次重复实验**：支持 `--num-repeats` 计算均值±标准差

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

## 真实 LLM 实验

### Windows PowerShell

```powershell
# 设置 API Key
$env:OPENAI_API_KEY = "sk-你的key"

# Stage2: Rule-based 评测（对齐论文配置）
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot --num-samples 100 --num-repeats 3

# Stage2: 4 种策略逐个运行
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot --num-samples 100
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy few_shot --num-samples 100
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy zero_shot_cot --num-samples 100
python -m llm4graphgen.stage2_rule_based --provider openai --model gpt-4 --temperature 0.8 --strategy few_shot_cot --num-samples 100

# Stage3: Distribution-based（p 值扫描）
python -m llm4graphgen.stage3_distribution --provider openai --model gpt-4 --temperature 0.5 --strategy zero_shot_cot --p-values 0.2,0.4,0.6,0.8

# Stage4: Property-based（MolHIV）
python -m llm4graphgen.stage4_property --provider openai --model gpt-4 --temperature 0.5 --strategy few_shot_cot --num-generate 20
```

### Linux / macOS (Bash)

```bash
# 设置 API Key
export OPENAI_API_KEY="sk-你的key"

# Stage2: Rule-based 评测（对齐论文配置）
python -m llm4graphgen.stage2_rule_based \
  --provider openai --model gpt-4 --temperature 0.8 \
  --strategy zero_shot --num-samples 100 --num-repeats 3

# Stage2: 4 种策略循环运行
for strategy in zero_shot few_shot zero_shot_cot few_shot_cot; do
  python -m llm4graphgen.stage2_rule_based \
    --provider openai --model gpt-4 --temperature 0.8 \
    --strategy $strategy --num-samples 100
done

# Stage3: Distribution-based（p 值扫描）
python -m llm4graphgen.stage3_distribution \
  --provider openai --model gpt-4 --temperature 0.5 \
  --strategy zero_shot_cot --p-values 0.2,0.4,0.6,0.8

# Stage4: Property-based（MolHIV）
python -m llm4graphgen.stage4_property \
  --provider openai --model gpt-4 --temperature 0.5 \
  --strategy few_shot_cot --num-generate 20
```

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
