#!/bin/bash
# Rerun FS and FS+CoT with max_tokens=2048 (previously 512, truncated CoT output)
# Usage: nohup bash run_rerun_maxtok2048.sh > run_rerun_maxtok2048.log 2>&1 &

export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=none
export LLM_MAX_TOKENS=2048

MODEL="/home/sheng-xiang/models/Llama-2-13b-chat-hf"

COMMON_ARGS="--size small \
--num-samples 100 \
--num-repeats 3 \
--provider openai \
--model $MODEL"

STRATEGIES=("few_shot" "few_shot_cot")

echo "========================================"
echo "Start time: $(date)"
echo "Rerun with max_tokens=2048"
echo "Strategies: ${STRATEGIES[*]}"
echo "========================================"

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "[$(date)] Starting: $strategy"
    echo "----------------------------------------"

    START_SEC=$SECONDS

    python -m llm4graphgen.stage2_rule_based \
        --strategy "$strategy" \
        $COMMON_ARGS

    EXIT_CODE=$?
    ELAPSED=$(( SECONDS - START_SEC ))
    MINS=$(( ELAPSED / 60 ))
    SECS=$(( ELAPSED % 60 ))

    echo ""
    echo "========================================"
    echo "REPORT: $strategy"
    echo "Exit code: $EXIT_CODE"
    echo "Duration: ${MINS}m ${SECS}s"
    echo "Finished: $(date)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS"
        RESULT_DIR=$(ls -td results/stage2_*_${strategy}_* \
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

echo ""
echo "========================================"
echo "ALL DONE at $(date)"
echo "========================================"
