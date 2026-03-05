#!/bin/bash
# LLaMA-2-13B Stage4 Property-based (MolHIV) experiment
# Matches GPT-4 Stage4 setup: GIN classifier, few_shot + few_shot_cot, 100 molecules
#
# Prerequisites:
#   1. vLLM server running: python -m vllm.entrypoints.openai.api_server \
#        --model /home/sheng-xiang/models/Llama-2-13b-chat-hf --max-model-len 4096
#   2. GIN cache exists (auto-downloads ogbg-molhiv on first run)
#   3. Run from project root directory
#
# Usage (on GPU server):
#   cd ~/llm4graphgen
#   nohup bash run_llama2_stage4.sh > run_llama2_stage4.log 2>&1 &

export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=none
export LLM_MAX_TOKENS=2048

MODEL="/home/sheng-xiang/models/Llama-2-13b-chat-hf"

echo "========================================"
echo "LLaMA-2-13B Stage4 Property-based (MolHIV)"
echo "Start time: $(date)"
echo "Model: $MODEL"
echo "Classifier: GIN"
echo "Strategies: few_shot, few_shot_cot"
echo "Generate: 100 molecules each"
echo "max_tokens: $LLM_MAX_TOKENS"
echo "========================================"

# --- Run 1: few_shot + GIN (classifier own TPR/FPR) ---
echo ""
echo "----------------------------------------"
echo "[$(date)] Run 1/4: few_shot + GIN (own TPR/FPR)"
echo "----------------------------------------"
START_SEC=$SECONDS

python -m llm4graphgen.stage4_property \
    --strategy few_shot \
    --num-generate 100 \
    --provider openai \
    --model "$MODEL" \
    --classifier gin \
    --temperature 0.5

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START_SEC ))
echo "Exit code: $EXIT_CODE, Duration: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "========================================"

# --- Run 2: few_shot + GIN (paper TPR/FPR) ---
echo ""
echo "----------------------------------------"
echo "[$(date)] Run 2/4: few_shot + GIN (paper TPR/FPR)"
echo "----------------------------------------"
START_SEC=$SECONDS

python -m llm4graphgen.stage4_property \
    --strategy few_shot \
    --num-generate 100 \
    --provider openai \
    --model "$MODEL" \
    --classifier gin \
    --paper-tpr-fpr \
    --temperature 0.5

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START_SEC ))
echo "Exit code: $EXIT_CODE, Duration: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "========================================"

# --- Run 3: few_shot_cot + GIN (classifier own TPR/FPR) ---
echo ""
echo "----------------------------------------"
echo "[$(date)] Run 3/4: few_shot_cot + GIN (own TPR/FPR)"
echo "----------------------------------------"
START_SEC=$SECONDS

python -m llm4graphgen.stage4_property \
    --strategy few_shot_cot \
    --num-generate 100 \
    --provider openai \
    --model "$MODEL" \
    --classifier gin \
    --temperature 0.5

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START_SEC ))
echo "Exit code: $EXIT_CODE, Duration: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "========================================"

# --- Run 4: few_shot_cot + GIN (paper TPR/FPR) ---
echo ""
echo "----------------------------------------"
echo "[$(date)] Run 4/4: few_shot_cot + GIN (paper TPR/FPR)"
echo "----------------------------------------"
START_SEC=$SECONDS

python -m llm4graphgen.stage4_property \
    --strategy few_shot_cot \
    --num-generate 100 \
    --provider openai \
    --model "$MODEL" \
    --classifier gin \
    --paper-tpr-fpr \
    --temperature 0.5

EXIT_CODE=$?
ELAPSED=$(( SECONDS - START_SEC ))
echo "Exit code: $EXIT_CODE, Duration: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "========================================"

echo ""
echo "========================================"
echo "ALL DONE at $(date)"
echo "Total runs: 4 (2 strategies x 2 TPR/FPR modes)"
echo ""
echo "Results:"
for dir in runs/stage4_Llama-2-13b-chat-hf_*; do
    if [ -d "$dir" ]; then
        echo "  $dir"
        if [ -f "$dir/property_metrics.csv" ]; then
            echo "    $(head -2 "$dir/property_metrics.csv" | tail -1 | cut -d',' -f1-10)"
        fi
    fi
done
echo "========================================"
