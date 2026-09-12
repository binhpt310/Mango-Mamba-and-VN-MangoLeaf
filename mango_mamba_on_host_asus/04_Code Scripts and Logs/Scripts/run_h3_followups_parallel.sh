#!/usr/bin/env bash
# Completed runs are skipped; interrupted runs resume from last.pth.tar. At most two trainers use GPU 0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-mango/bin/python"
RUNNER="$ROOT/scripts/run_h3_followups.py"
OUTPUT="$ROOT/results/h3_320_aligned_followup"
LOG_DIR="$ROOT/logs/h3_320_aligned_followup"

mkdir -p "$LOG_DIR"

launch() {
    local task="$1" seed="$2"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" --task "$task" --seed "$seed" --output "$OUTPUT" \
        >"$LOG_DIR/${task}_seed_${seed}.log" 2>&1 &
    PIDS+=("$!")
    LABELS+=("${task}:seed_${seed}")
}

run_pair() {
    PIDS=()
    LABELS=()
    launch "$1" "$2"
    launch "$3" "$4"
    local index
    for index in "${!PIDS[@]}"; do
        wait "${PIDS[$index]}"
        echo "DONE ${LABELS[$index]}"
    done
}

run_pair no-inter 42 direct-vn 42
run_pair curriculum 42 no-inter 123
run_pair direct-vn 123 curriculum 123
run_pair no-inter 2024 direct-vn 2024
run_pair curriculum 2024 no-inter 3333
run_pair direct-vn 3333 curriculum 3333
