# LLM4GraphGen 复现教程

本仓库从零开始复现论文 **arXiv:2403.14358**（*Exploring the Potential of Large Language Models in Graph Generation*）。

本教程面向组内同学，包含两条实验路径：
- **路径 A**：使用 OpenAI API（GPT-4 / GPT-3.5-Turbo），本地即可运行
- **路径 B**：在 GPU 服务器上部署 LLaMA-2-13B-Chat，从零开始搭建推理环境

> 注意：教程中涉及 API Key、HuggingFace Token、服务器 IP 等敏感信息的地方，均用占位符表示，请替换为你自己的实际值。

---

## 目录

1. [项目结构概览](#1-项目结构概览)
2. [环境准备（本地）](#2-环境准备本地)
3. [路径 A：使用 OpenAI API 跑实验](#3-路径-a使用-openai-api-跑实验)
4. [路径 B：GPU 服务器部署 LLaMA-2](#4-路径-bgpu-服务器部署-llama-2)
   - [4.1 SSH 连接与 Miniconda 安装](#41-ssh-连接与-miniconda-安装)
   - [4.2 创建 Python 环境与安装依赖](#42-创建-python-环境与安装依赖)
   - [4.3 下载模型](#43-下载模型)
   - [4.4 启动 vLLM 推理服务](#44-启动-vllm-推理服务)
   - [4.5 验证服务](#45-验证服务)
   - [4.6 上传仓库与运行实验](#46-上传仓库与运行实验)
   - [4.7 下载结果到本地](#47-下载结果到本地)
5. [实验体系说明](#5-实验体系说明)
6. [踩坑记录与常见问题](#6-踩坑记录与常见问题)
7. [附录：完整 CLI 参数参考](#7-附录完整-cli-参数参考)

---

## 1. 项目结构概览

```
llm4graphgen/
├── src/llm4graphgen/
│   ├── providers/          # LLM 调用层（OpenAI / Mock）
│   │   ├── openai_provider.py   # OpenAI 兼容 API 调用
│   │   └── mock_provider.py     # 离线测试用 mock
│   ├── prompts/            # 4 种 prompting 策略的模板
│   ├── parsers/            # LLM 输出解析器（图结构提取）
│   ├── metrics/            # 评测指标计算
│   ├── stage2_rule_based.py    # Stage 2: 规则图生成评测
│   ├── stage3_distribution.py  # Stage 3: 分布图生成评测
│   └── stage4_property.py      # Stage 4: 属性图（分子）生成评测
├── tests/                  # 单元测试
├── run_remaining.sh        # 服务器批量运行脚本示例
└── README.md               # 本文件
```

---

## 2. 环境准备（本地）

**Python 版本**：3.10+

```bash
# 基础依赖（Stage 2）
pip install networkx openai

# Stage 3/4 额外依赖
pip install rdkit numpy scikit-learn

# Stage 4 GIN 分类器（最忠实复现）
pip install ogb pandas torch torch_geometric

# 开发/测试
pip install pytest
```

---

## 3. 路径 A：使用 OpenAI API 跑实验

这是最简单的路径，只需要一个 OpenAI API Key，在本地电脑上即可运行。

### 3.1 设置 API Key

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-你的key"
```

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-你的key"
```

### 3.2 先跑 Mock 验证（无需 API Key）

建议先用 mock 模式跑通流程，确认环境没问题：

```bash
python -X utf8 -m llm4graphgen.smoke
python -X utf8 -m llm4graphgen.stage2_rule_based --num-samples 10
```

### 3.3 运行真实实验

```bash
# Stage 2: Zero-shot, Medium 规模, GPT-4, 100 样本, 3 次重复
python -m llm4graphgen.stage2_rule_based \
  --provider openai \
  --model gpt-4 \
  --temperature 0.8 \
  --strategy zero_shot \
  --num-samples 100 \
  --num-repeats 3

# Stage 2: 换成 Few-shot 策略
python -m llm4graphgen.stage2_rule_based \
  --provider openai \
  --model gpt-4 \
  --temperature 0.8 \
  --strategy few_shot \
  --num-samples 100 \
  --num-repeats 3

# Stage 2: Small 规模
python -m llm4graphgen.stage2_rule_based \
  --provider openai \
  --model gpt-4 \
  --temperature 0.8 \
  --strategy zero_shot \
  --num-samples 100 \
  --size small

# Stage 3: 分布图评测
python -m llm4graphgen.stage3_distribution \
  --provider openai \
  --model gpt-4 \
  --temperature 0.5 \
  --strategy zero_shot_cot \
  --p-values 0.2,0.4,0.6,0.8

# Stage 4: 分子属性评测（GIN 分类器 + 论文 TPR/FPR）
python -m llm4graphgen.stage4_property \
  --provider openai \
  --model gpt-4 \
  --temperature 0.5 \
  --strategy few_shot_cot \
  --num-generate 100 \
  --classifier gin \
  --paper-tpr-fpr
```

### 3.4 一键批量运行（推荐）

```bash
# 预览实验计划（不运行）
python -m llm4graphgen.experiment_runner --plan standard --dry-run

# 开始运行（2 个实验并行）
python -m llm4graphgen.experiment_runner --plan standard --parallel 2

# 中断后续跑（已完成的自动跳过）
python -m llm4graphgen.experiment_runner --plan standard --parallel 2
```

| 计划 | 实验数 | 内容 |
|------|--------|------|
| `minimal` | 3 | 最小验证：1 策略 × medium |
| `standard` | 7 | 标准复现：4 策略 × medium × 3 repeats |
| `full` | 19 | 完整消融：4 策略 × 3 规模 × 3 repeats + GPT-3.5 对比 |

---

## 4. 路径 B：GPU 服务器部署 LLaMA-2

这条路径适用于在 GPU 服务器上用开源模型 LLaMA-2-13B-Chat 跑实验。整个过程我们踩了不少坑，下面是总结出来的**正确路径**。

**硬件需求**：至少 2 张 GPU，单卡显存 ≥ 20GB（如 RTX 6000 / A100 / 3090 等）。LLaMA-2-13B 约需 26GB 显存，单卡放不下，需要双卡张量并行。

### 4.1 SSH 连接与 Miniconda 安装

```bash
# SSH 登录服务器
ssh 你的用户名@服务器IP

# 下载并安装 Miniconda（不需要 root 权限）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
eval "$(~/miniconda3/bin/conda shell.bash hook)"
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

> **为什么用 Miniconda 而不是 venv？**
> 服务器通常没有 root 权限，系统 Python 版本可能太低，venv 无法安装 C 扩展依赖（如 vLLM 需要的 CUDA 绑定）。Miniconda 是无 root 安装的最佳选择。

### 4.2 创建 Python 环境与安装依赖

```bash
# 如果 conda 提示需要同意服务条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建独立环境
conda create -n llm4graphgen python=3.10 -y
conda activate llm4graphgen

# 安装依赖
pip install vllm
pip install networkx rdkit numpy scikit-learn huggingface_hub
```

### 4.3 下载模型

> **关键：必须下载 Chat 版本**（`Llama-2-13b-chat-hf`），不是 Base 版本（`Llama-2-13b-hf`）。
> Base 版本只会做文本续写，无法遵循指令，生成的内容不可用。

**前置条件**：
1. 注册 [HuggingFace](https://huggingface.co) 账号
2. 打开 [Llama-2-13b-chat-hf 模型页面](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)，点 "Request access" 申请 Meta 授权（通常几小时内通过）
3. 在 [HuggingFace Token 页面](https://huggingface.co/settings/tokens) 创建 access token

```bash
# 登录 HuggingFace
huggingface-cli login
# 按提示粘贴你的 token

# === 国内服务器需要使用镜像站 ===
export HF_ENDPOINT=https://hf-mirror.com

# 先下载小文件（config、tokenizer 等）
huggingface-cli download meta-llama/Llama-2-13b-chat-hf \
  --local-dir ~/models/Llama-2-13b-chat-hf \
  --exclude "pytorch_model*" "model-*.safetensors"

# 大文件（3 个 safetensors，共约 25GB）用 wget 并行下载，防止超时
TOKEN=$(cat ~/.cache/huggingface/token)
cd ~/models/Llama-2-13b-chat-hf

nohup wget -c --header="Authorization: Bearer $TOKEN" \
  "https://hf-mirror.com/meta-llama/Llama-2-13b-chat-hf/resolve/main/model-00001-of-00003.safetensors" \
  -o wget-log1 &

nohup wget -c --header="Authorization: Bearer $TOKEN" \
  "https://hf-mirror.com/meta-llama/Llama-2-13b-chat-hf/resolve/main/model-00002-of-00003.safetensors" \
  -o wget-log2 &

nohup wget -c --header="Authorization: Bearer $TOKEN" \
  "https://hf-mirror.com/meta-llama/Llama-2-13b-chat-hf/resolve/main/model-00003-of-00003.safetensors" \
  -o wget-log3 &
```

**查看下载进度**：

```bash
ls -lh ~/models/Llama-2-13b-chat-hf/*.safetensors   # 查看文件大小
ps aux | grep wget                                     # 确认进程还在
```

**下载完成标志**：3 个 safetensors 文件分别约 9.3GB、9.3GB、5.8GB。

> **如果服务器可以直接访问 HuggingFace**（海外服务器），则不需要设 `HF_ENDPOINT`，直接用 `huggingface-cli download` 即可。

### 4.4 启动 vLLM 推理服务

vLLM 提供 OpenAI 兼容的 API，这样我们的代码可以用同一套 `--provider openai` 参数同时支持 GPT-4 和 LLaMA-2。

```bash
conda activate llm4graphgen

# 设置 CUDA 环境（每次新开终端都要设）
export CUDA_HOME=$CONDA_PREFIX

# 创建 libcuda.so 符号链接（只需做一次）
mkdir -p $CONDA_PREFIX/lib64/stubs
ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 $CONDA_PREFIX/lib64/stubs/libcuda.so

# 后台启动 vLLM（nohup 确保关闭 SSH 不中断）
nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Llama-2-13b-chat-hf \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0 \
  > ~/vllm.log 2>&1 &

# 查看启动日志
tail -f ~/vllm.log
# 看到 "Application startup complete" 表示成功，按 Ctrl+C 退出查看
```

> **为什么需要 CUDA_HOME 和 libcuda.so 符号链接？**
> vLLM 的 FlashInfer 组件需要 JIT 编译 CUDA kernel，如果找不到 CUDA 开发头文件和 `libcuda.so`，会报编译错误。这两行命令就是解决这个问题的。

### 4.5 验证服务

```bash
# 写测试 JSON 到文件（避免终端换行破坏 JSON）
cat > /tmp/test.json << 'EOF'
{
  "model": "/home/你的用户名/models/Llama-2-13b-chat-hf",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "max_tokens": 50
}
EOF

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/test.json
```

返回 JSON 中包含模型回答就说明服务正常。

> **注意**：`model` 字段必须填 vLLM 启动时 `--model` 指定的**完整路径**。路径不一致会报 404。

### 4.6 上传仓库与运行实验

**Step 1：修改代码让 base_url 支持环境变量**

在服务器上，需要修改 `openai_provider.py`，让 API 请求指向本地 vLLM 而不是 OpenAI：

```bash
cd ~/llm4graphgen
sed -i 's|base_url: str = "https://api.openai.com/v1"|base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")|' \
  src/llm4graphgen/providers/openai_provider.py
```

**Step 2：上传仓库（本地终端执行）**

```powershell
# Windows PowerShell
scp -r "本地仓库路径" 用户名@服务器IP:~/llm4graphgen
```

**Step 3：设置环境变量并运行实验**

```bash
conda activate llm4graphgen
cd ~/llm4graphgen

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="none"   # vLLM 不需要真实 key，但参数必须传

# 运行单个实验
python -m llm4graphgen.stage2_rule_based \
  --provider openai \
  --model /home/你的用户名/models/Llama-2-13b-chat-hf \
  --temperature 0.8 \
  --strategy zero_shot \
  --num-samples 100 \
  --num-repeats 3 \
  --size small
```

> **关键**：`--model` 路径必须与 vLLM 启动时的 `--model` 参数完全一致。

**Step 4：批量运行全部策略（推荐）**

参考仓库中的 `run_remaining.sh`，可以一次性跑 4 种 prompting 策略：

```bash
# 后台运行，SSH 关闭不受影响
nohup bash run_remaining.sh > run_rerun.log 2>&1 &

# 查看进度
tail -20 run_rerun.log
# 看到 "ALL DONE" 表示全部完成
```

### 4.7 下载结果到本地

```powershell
# 在本地 PowerShell 执行
# 下载全部 Stage 2 结果
scp -r 用户名@服务器IP:~/llm4graphgen/results/stage2_* C:\Users\你的用户名\Desktop\llm4graphgen-results\

# 只下载某个策略的结果
scp -r 用户名@服务器IP:~/llm4graphgen/results/stage2_*zero_shot_small* C:\Users\你的用户名\Desktop\llm4graphgen-results\
```

---

## 5. 实验体系说明

### 5.1 论文的 4 个评测阶段

| 阶段 | 名称 | 评什么 | 关键指标 |
|------|------|--------|----------|
| Stage 2 | Rule-based | LLM 能否生成满足规则约束的图 | Valid Rate, Unique Rate, Novel Rate |
| Stage 3 | Distribution-based | LLM 生成图的统计分布是否合理 | MMD (degree/clustering/orbit) |
| Stage 4 | Property-based | LLM 生成的分子是否具有目标属性 | Rectified C 分类指标 |

### 5.2 4 种 Prompting 策略

| 策略 | 描述 |
|------|------|
| `zero_shot` | 直接给任务描述，不给示例 |
| `few_shot` | 给 2-3 个示例 + 任务描述 |
| `zero_shot_cot` | 零样本 + Chain-of-Thought 推理 |
| `few_shot_cot` | 少样本 + CoT 推理 |

### 5.3 3 种图规模

| 规模 | 说明 | 对应论文 |
|------|------|----------|
| `small` | 小规模图（5-16 节点） | Table 8 |
| `medium` | 中规模图（默认） | Table 3 / Table 7 |
| `large` | 大规模图（50-100 节点） | Table 8 |

### 5.4 评测指标

- **Valid Rate**：LLM 输出中能成功解析且满足图类型约束的比例
- **Unique Rate**：有效图中去重后的比例（使用 WL 图同构哈希）
- **Novel Rate**：有效图中不在 few-shot 示例里出现的比例
- **MMD**：Maximum Mean Discrepancy，衡量生成图分布与真实图分布的差距

### 5.5 Stage 2 涉及的图类型

Tree, Cycle, Planar, Bipartite, k-Regular, k-Coloring, Wheel

---

## 6. 踩坑记录与常见问题

以下是我们在部署过程中实际遇到的问题，总结出来帮大家避坑。

### Q1：用 venv 安装 vLLM 失败

**症状**：`pip install vllm` 编译报错，找不到 CUDA 头文件。

**原因**：服务器上没有 root 权限，系统 Python 和 CUDA SDK 不完整。

**解决**：用 Miniconda 创建独立环境（见 [4.1](#41-ssh-连接与-miniconda-安装)），不依赖系统包。

### Q2：下载了 Base 模型而不是 Chat 模型

**症状**：模型可以启动，但生成的内容完全不可用——只会做无关的文本续写，不遵循指令。

**原因**：`Llama-2-13b-hf`（Base）是预训练模型，没有经过指令微调。

**解决**：必须下载 `Llama-2-13b-chat-hf`（Chat），这是经过 RLHF 微调的指令遵循版本。

### Q3：HuggingFace 下载超时 / 无法访问

**症状**：`huggingface-cli download` 卡住或报网络错误。

**原因**：国内服务器无法直接访问 huggingface.co。

**解决**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
大文件（safetensors）建议用 `wget -c` 并行下载，防止超时中断（见 [4.3](#43-下载模型)）。

### Q4：vLLM 启动报 FlashInfer JIT 编译错误

**症状**：启动 vLLM 时报 CUDA kernel 编译失败。

**原因**：缺少 `CUDA_HOME` 环境变量和 `libcuda.so` 符号链接。

**解决**：
```bash
export CUDA_HOME=$CONDA_PREFIX
mkdir -p $CONDA_PREFIX/lib64/stubs
ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 $CONDA_PREFIX/lib64/stubs/libcuda.so
```

### Q5：curl 测试 vLLM 返回格式错误

**症状**：`curl -d '{"model":...}'` 报 JSON 解析错误。

**原因**：终端换行/引号转义破坏了内联 JSON。

**解决**：把 JSON 写入文件，用 `curl -d @/tmp/test.json` 发送（见 [4.5](#45-验证服务)）。

### Q6：实验报 404 Model Not Found

**症状**：运行实验脚本时报模型不存在。

**原因**：`--model` 参数的路径与 vLLM 启动时不一致。例如 vLLM 用了 `~/models/...`，但实验脚本写了 `/home/xxx/models/...`。

**解决**：确保两边路径完全相同。建议都用绝对路径。

### Q7：LLaMA 输出解析失败率很高（40-70%）

**症状**：实验 Valid Rate 极低，大量 "格式错误" 解析失败。

**原因**：LLaMA 生成图时，节点 ID 常带引号，如 `('0', '1')` 而不是 `(0, 1)`，原始 parser 不接受字符串。

**解决**：本仓库的 `graph_parser.py` 已修复，能容忍字符串格式数字（自动转 int）。如果你用的是旧版代码，需要更新 parser。

### Q8：SIZE_PRESETS 与论文不对齐

**症状**：实验结果与论文 Table 8 差异过大。

**原因**：代码中 `small`/`large` 的节点数设置与论文表格不一致（如 cycle、k_regular、k_coloring 的节点数有误）。

**解决**：本仓库已修正 SIZE_PRESETS 与论文 Table 8 对齐。请确保使用最新版代码。

### Q9：SSH 断开后实验中断

**症状**：关闭 SSH 终端后，实验进程被杀死。

**解决**：所有后台任务用 `nohup ... &` 运行：
```bash
# vLLM 服务
nohup python -m vllm.entrypoints.openai.api_server ... > ~/vllm.log 2>&1 &

# 实验脚本
nohup bash run_remaining.sh > run_rerun.log 2>&1 &
```

### Q10：`base_url` 硬编码导致无法连接 vLLM

**症状**：实验脚本尝试连接 `api.openai.com` 而不是本地 vLLM。

**原因**：`openai_provider.py` 中 `base_url` 硬编码为 OpenAI 地址。

**解决**：用 `sed` 修改代码读取环境变量（见 [4.6 Step 1](#46-上传仓库与运行实验)），然后设置：
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

---

## 7. 附录：完整 CLI 参数参考

### Stage 2: `llm4graphgen.stage2_rule_based`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--provider` | `openai` 或 `mock` | `mock` |
| `--model` | 模型名称/路径 | `gpt-4` |
| `--temperature` | 生成温度 | `0.8` |
| `--strategy` | 策略：`zero_shot` / `few_shot` / `zero_shot_cot` / `few_shot_cot` | `zero_shot` |
| `--num-samples` | 每种图类型的生成样本数 | `100` |
| `--num-repeats` | 重复实验次数 | `1` |
| `--size` | 图规模：`small` / `medium` / `large` | `medium` |
| `--unique-method` | 去重方法：`isomorphism`（WL 哈希）/ `signature` | `isomorphism` |

### Stage 3: `llm4graphgen.stage3_distribution`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--provider` | `openai` 或 `mock` | `mock` |
| `--model` | 模型名称/路径 | `gpt-4` |
| `--temperature` | 生成温度 | `0.5` |
| `--strategy` | 策略（同上，或 `all` 一键运行 4 种） | `zero_shot_cot` |
| `--p-values` | ER 图 p 值列表 | `0.2,0.4,0.6,0.8` |
| `--num-repeats` | 重复实验次数 | `1` |

### Stage 4: `llm4graphgen.stage4_property`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--provider` | `openai` 或 `mock` | `mock` |
| `--model` | 模型名称/路径 | `gpt-4` |
| `--temperature` | 生成温度 | `0.5` |
| `--strategy` | 策略 | `few_shot_cot` |
| `--num-generate` | 生成分子数 | `100` |
| `--classifier` | 分类器：`gin`（推荐）/ `ogbg` / `proxy` | `gin` |
| `--paper-tpr-fpr` | 使用论文 Table 10 的 TPR/FPR 参照 | `False` |

### 实验运行器: `llm4graphgen.experiment_runner`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--plan` | 实验计划：`minimal` / `standard` / `full` | `standard` |
| `--parallel` | 并行实验数 | `1` |
| `--model` | 覆盖模型 | — |
| `--stage` | 只运行某个阶段 | 全部 |
| `--dry-run` | 只预览不运行 | `False` |
| `--reset` | 清除续跑记录从头开始 | `False` |

---

## 运行测试

```bash
python -X utf8 -m pytest -q
```
