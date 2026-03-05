#!/bin/bash
# LLaMA-2-13B Stage2 scale ablation: Medium + Large (论文 Table 4/8 对齐)
# Only runs the 3 tasks defined in Table 8: Cycle, k-regular, k-coloring
#
# Paper Table 8 parameters:
#   Medium: Cycle n=15, k-regular n=16 k=3, k-coloring n=15 m=32 k=3
#   Large:  Cycle n=20, k-regular n=20 k=3, k-coloring n=18 m=39 k=3
#
# Usage: nohup bash run_llama2_medium.sh > run_llama2_medium.log 2>&1 &

export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=none
export LLM_MAX_TOKENS=2048

MODEL="/home/sheng-xiang/models/Llama-2-13b-chat-hf"

TASKS="cycle k_regular k_coloring"
STRATEGIES=("zero_shot" "few_shot" "zero_shot_cot" "few_shot_cot")
SIZES=("medium" "large")

echo "========================================"
echo "LLaMA-2-13B Stage2 Scale Ablation"
echo "Start time: $(date)"
echo "Tasks: $TASKS"
echo "Sizes: ${SIZES[*]}"
echo "Strategies: ${STRATEGIES[*]}"
echo "Samples: 100, Repeats: 3"
echo "max_tokens: $LLM_MAX_TOKENS"
echo "========================================"

for size in "${SIZES[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "[$(date)] Starting: $strategy @ $size"
        echo "----------------------------------------"

        START_SEC=$SECONDS

        python -m llm4graphgen.stage2_rule_based \
            --strategy "$strategy" \
            --size "$size" \
            --tasks $TASKS \
            --num-samples 100 \
            --num-repeats 3 \
            --provider openai \
            --model "$MODEL"

        EXIT_CODE=$?
        ELAPSED=$(( SECONDS - START_SEC ))
        MINS=$(( ELAPSED / 60 ))
        SECS=$(( ELAPSED % 60 ))

        echo ""
        echo "========================================"
        echo "REPORT: $strategy @ $size"
        echo "Exit code: $EXIT_CODE"
        echo "Duration: ${MINS}m ${SECS}s"
        echo "Finished: $(date)"

        if [ $EXIT_CODE -eq 0 ]; then
            echo "Status: SUCCESS"
            RESULT_DIR=$(ls -td runs/stage2_*_${strategy}_${size}_* \
                2>/dev/null | head -1)
            if [ -n "$RESULT_DIR" ]; then
                echo "Result dir: $RESULT_DIR"
                SUMMARY="$RESULT_DIR/rule_based_summary.csv"
                if [ -f "$SUMMARY" ]; then
                    echo ""
                    echo "--- Summary ---"
                    cat "$SUMMARY"
                fi
            fi
        else
            echo "Status: FAILED"
        fi
        echo "========================================"
    done
done

echo ""
echo "========================================"
echo "ALL DONE at $(date)"
echo "Total runs: ${#SIZES[@]} sizes x ${#STRATEGIES[@]} strategies = $(( ${#SIZES[@]} * ${#STRATEGIES[@]} ))"
echo "========================================"
