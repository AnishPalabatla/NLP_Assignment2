#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${MODEL_DIR:-data/models/Qwen2.5-Math-1.5B}"
MATH_TRAIN="${MATH_TRAIN:-data/MATH/train.jsonl}"
MATH_VAL="${MATH_VAL:-data/MATH/validation.jsonl}"
SFT_DATA="${SFT_DATA:-data/sft.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-cs336_alignment/prompts/r1_zero.prompt}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/assignment_${RUN_ID}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
VLLM_DEVICE="${VLLM_DEVICE:-cuda:1}"

BASELINE_LIMIT="${BASELINE_LIMIT:-0}"
SFT_TRAIN_SAMPLES="${SFT_TRAIN_SAMPLES:-0}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-2000}"
SFT_EVAL_INTERVAL="${SFT_EVAL_INTERVAL:-200}"
SFT_EVAL_MAX_EXAMPLES="${SFT_EVAL_MAX_EXAMPLES:-500}"
EI_STEPS="${EI_STEPS:-5}"
EI_DATA_SIZE="${EI_DATA_SIZE:-512}"
EI_ROLLOUTS="${EI_ROLLOUTS:-2}"
EI_EPOCHS="${EI_EPOCHS:-1}"
EI_EVAL_MAX_EXAMPLES="${EI_EVAL_MAX_EXAMPLES:-500}"
GRPO_STEPS="${GRPO_STEPS:-200}"
GRPO_ROLLOUT_BATCH_SIZE="${GRPO_ROLLOUT_BATCH_SIZE:-256}"
GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-8}"
GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-256}"
GRPO_GRAD_ACCUMULATION_STEPS="${GRPO_GRAD_ACCUMULATION_STEPS:-128}"
GRPO_EVAL_INTERVAL="${GRPO_EVAL_INTERVAL:-10}"
GRPO_EVAL_MAX_EXAMPLES="${GRPO_EVAL_MAX_EXAMPLES:-1024}"
GRPO_SAVE_INTERVAL="${GRPO_SAVE_INTERVAL:-50}"

for required_file in "$MATH_TRAIN" "$MATH_VAL" "$SFT_DATA" "$PROMPT_FILE"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Required file not found: %s\n' "$required_file" >&2
    exit 1
  fi
done

if [[ ! -d "$MODEL_DIR" ]]; then
  printf 'Model directory not found: %s\nSet MODEL_DIR to the Qwen 2.5 Math 1.5B model directory.\n' "$MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/launcher_logs"

run_stage() {
  local stage="$1"
  shift
  local log_file="$RUN_ROOT/launcher_logs/${stage}.log"

  printf '\n===== Starting %s =====\n' "$stage"
  printf 'Log: %s\n' "$log_file"
  "$@" 2>&1 | tee "$log_file"
  printf '===== Finished %s =====\n' "$stage"
}

run_stage preflight \
  uv run --no-sync python -c \
  'import torch; count = torch.cuda.device_count(); print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; visible GPUs: {count}"); assert torch.cuda.is_available() and count >= 2, "These experiment scripts require at least two visible CUDA GPUs (cuda:0 and cuda:1)."'

run_stage math_baseline \
  uv run --no-sync python scripts/math_baseline.py \
  --model "$MODEL_DIR" \
  --data "$MATH_VAL" \
  --prompt_file "$PROMPT_FILE" \
  --out_dir "$RUN_ROOT/math_baseline" \
  --limit "$BASELINE_LIMIT"

run_stage sft_unfiltered \
  uv run --no-sync python scripts/sft_experiment.py \
  --model_id "$MODEL_DIR" \
  --sft_path "$SFT_DATA" \
  --val_path "$MATH_VAL" \
  --prompt_file "$PROMPT_FILE" \
  --train_device "$TRAIN_DEVICE" \
  --vllm_device "$VLLM_DEVICE" \
  --train_samples "$SFT_TRAIN_SAMPLES" \
  --max_steps "$SFT_MAX_STEPS" \
  --eval_interval "$SFT_EVAL_INTERVAL" \
  --eval_max_examples "$SFT_EVAL_MAX_EXAMPLES" \
  --out_dir "$RUN_ROOT/sft_unfiltered"

run_stage sft_filtered \
  uv run --no-sync python scripts/sft_experiment.py \
  --model_id "$MODEL_DIR" \
  --sft_path "$SFT_DATA" \
  --val_path "$MATH_VAL" \
  --prompt_file "$PROMPT_FILE" \
  --train_device "$TRAIN_DEVICE" \
  --vllm_device "$VLLM_DEVICE" \
  --train_samples "$SFT_TRAIN_SAMPLES" \
  --max_steps "$SFT_MAX_STEPS" \
  --eval_interval "$SFT_EVAL_INTERVAL" \
  --eval_max_examples "$SFT_EVAL_MAX_EXAMPLES" \
  --out_dir "$RUN_ROOT/sft_filtered" \
  --filter_correct

run_stage expert_iteration \
  uv run --no-sync python scripts/expert_iteration_experiment.py \
  --model_id "$MODEL_DIR" \
  --train_path "$MATH_TRAIN" \
  --val_path "$MATH_VAL" \
  --prompt_file "$PROMPT_FILE" \
  --train_device "$TRAIN_DEVICE" \
  --vllm_device "$VLLM_DEVICE" \
  --n_ei_steps "$EI_STEPS" \
  --D_i "$EI_DATA_SIZE" \
  --G "$EI_ROLLOUTS" \
  --epochs "$EI_EPOCHS" \
  --eval_max_examples "$EI_EVAL_MAX_EXAMPLES" \
  --out_dir "$RUN_ROOT/expert_iteration"

run_stage grpo \
  uv run --no-sync python scripts/grpo_experiment.py \
  --model-id "$MODEL_DIR" \
  --train-path "$MATH_TRAIN" \
  --val-path "$MATH_VAL" \
  --prompt-file "$PROMPT_FILE" \
  --n-grpo-steps "$GRPO_STEPS" \
  --rollout-batch-size "$GRPO_ROLLOUT_BATCH_SIZE" \
  --group-size "$GRPO_GROUP_SIZE" \
  --train-batch-size "$GRPO_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRPO_GRAD_ACCUMULATION_STEPS" \
  --eval-interval "$GRPO_EVAL_INTERVAL" \
  --eval-max-examples "$GRPO_EVAL_MAX_EXAMPLES" \
  --save-interval "$GRPO_SAVE_INTERVAL" \
  --log-dir "$RUN_ROOT/grpo"

printf '\nAll experiment stages completed.\nResults and launcher logs: %s\n' "$RUN_ROOT"
