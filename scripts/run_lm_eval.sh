#!/usr/bin/env bash
# Optional standard evaluation; requires 'python -m pip install lm-eval[hf]'.
# Usage:
#   bash scripts/run_lm_eval.sh Qwen/Qwen2.5-0.5B '' base
#   bash scripts/run_lm_eval.sh outputs/sft_merged outputs/dpo_adapter dpo
set -euo pipefail
BASE="${1:?base model/path required}"
ADAPTER="${2:-}"
NAME="${3:-gsm8k}"
ARGS="pretrained=${BASE},dtype=float16"
if [[ -n "$ADAPTER" ]]; then ARGS="${ARGS},peft=${ADAPTER}"; fi
mkdir -p "results/lm_eval/${NAME}"
# This uses the lm-evaluation-harness task prompt / parser and is distinct
# from scripts/evaluate.py custom greedy prompt evaluation.
lm_eval run --model hf --model_args "$ARGS" --tasks gsm8k --num_fewshot 5 --batch_size 1 \
  --output_path "results/lm_eval/${NAME}" --log_samples
