#!/usr/bin/env bash
# At most two trainers share GPU 0; each trainer writes its own run log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv-mango/bin/python}"

for seed in 42 123 2024 3333; do
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/code/run_curriculum.py" --task curriculum --seed "$seed" "$@" &
    curriculum_pid=$!
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/code/run_curriculum.py" --task direct-vn --seed "$seed" "$@" &
    direct_pid=$!
    status=0
    wait "$curriculum_pid" || status=$?
    wait "$direct_pid" || status=$?
    if (( status != 0 )); then exit "$status"; fi
done
