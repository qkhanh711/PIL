#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Optional: activate conda env before running
# conda activate PIL

# ------------------------
# Global knobs (override via env vars)
# ------------------------
SEEDS_EXP_A="${SEEDS_EXP_A:-7,11,23,31,43,59,71,89}"
SEEDS_EXP_B="${SEEDS_EXP_B:-7,11,23,31,43,59,71,89,97,101,107,109,113,127,131,137,139,149,151,157}"
SEEDS_EXP_D="${SEEDS_EXP_D:-7,11,23,31,43,59,71,89,97,101,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211}"
SEEDS_MPE="${SEEDS_MPE:-7,11,23}"

SYN_NUM_BLOCKS="${SYN_NUM_BLOCKS:-50}"
SYN_INNER_UPDATES="${SYN_INNER_UPDATES:-50}"
SYN_TRAIN_BATCH="${SYN_TRAIN_BATCH:-64}"
SYN_EVAL_BATCH="${SYN_EVAL_BATCH:-256}"
SYN_EPISODE_LENGTH="${SYN_EPISODE_LENGTH:-8}"
SYN_HIDDEN_DIM="${SYN_HIDDEN_DIM:-64}"
SYN_MESSAGE_DIM="${SYN_MESSAGE_DIM:-2}"
SYN_PRIVACY_ALPHA="${SYN_PRIVACY_ALPHA:-8.0}"
SYN_DELTA="${SYN_DELTA:-1e-5}"
SYN_GAMMA="${SYN_GAMMA:-0.95}"

MPE_EPISODES="${MPE_EPISODES:-100}"
MPE_SCENARIOS="${MPE_SCENARIOS:-cn,ccn,pp}"
MPE_ALGOS="${MPE_ALGOS:-pil,dpmac,i2c,tarmac,maddpg}"

RUN_EXP_A="${RUN_EXP_A:-1}"
RUN_EXP_BCD="${RUN_EXP_BCD:-1}"
RUN_MPE="${RUN_MPE:-1}"

OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/experiments/exp_runs/new_pil}"
OUT_LOG="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$OUT_LOG"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_LOG/new_run_${STAMP}.log"
SCRIPT_START_TS=$(date +%s)

format_seconds() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02dh:%02dm:%02ds" "$h" "$m" "$s"
}

STEPS_TOTAL=0
[[ "$RUN_EXP_A" == "1" ]] && STEPS_TOTAL=$((STEPS_TOTAL + 1))
[[ "$RUN_EXP_BCD" == "1" ]] && STEPS_TOTAL=$((STEPS_TOTAL + 3))
[[ "$RUN_MPE" == "1" ]] && STEPS_TOTAL=$((STEPS_TOTAL + 1))

STEP_INDEX=0

run_cmd() {
  STEP_INDEX=$((STEP_INDEX + 1))
  local step_start_ts
  step_start_ts=$(date +%s)

  echo -e "\n[STEP ${STEP_INDEX}/${STEPS_TOTAL}] >>> $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"

  local now_ts step_elapsed total_elapsed remaining_steps avg_per_step eta_sec
  now_ts=$(date +%s)
  step_elapsed=$((now_ts - step_start_ts))
  total_elapsed=$((now_ts - SCRIPT_START_TS))
  remaining_steps=$((STEPS_TOTAL - STEP_INDEX))

  if [[ "$STEP_INDEX" -gt 0 ]]; then
    avg_per_step=$((total_elapsed / STEP_INDEX))
  else
    avg_per_step=0
  fi
  eta_sec=$((avg_per_step * remaining_steps))

  echo "[STEP ${STEP_INDEX}/${STEPS_TOTAL}] done in $(format_seconds "$step_elapsed") | elapsed $(format_seconds "$total_elapsed") | ETA ~$(format_seconds "$eta_sec")" | tee -a "$LOG_FILE"
}

echo "[INFO] Root: $ROOT_DIR" | tee -a "$LOG_FILE"
echo "[INFO] Output root: $OUT_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] Planned steps: $STEPS_TOTAL" | tee -a "$LOG_FILE"
echo "[INFO] Started at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

# --------------------------------------------------
# Exp A: Multi-agent heterogeneous budgets
# Paper intent: compare adaptive schedules vs DPMAC
# Repo mapping: synthetic bundle with clip_la / exact_wf / naive_la / dpmac + baselines
# --------------------------------------------------
if [[ "$RUN_EXP_A" == "1" ]]; then
  run_cmd python experiments/new_experiments.py \
    --sections synthetic \
    --seeds "$SEEDS_EXP_A" \
    --output_root "$OUT_ROOT/expA_multi_agent_heterogeneous" \
    --num_blocks "$SYN_NUM_BLOCKS" \
    --inner_updates "$SYN_INNER_UPDATES" \
    --train_batch_size "$SYN_TRAIN_BATCH" \
    --eval_batch_size "$SYN_EVAL_BATCH" \
    --episode_length "$SYN_EPISODE_LENGTH" \
    --hidden_dim "$SYN_HIDDEN_DIM" \
    --message_dim "$SYN_MESSAGE_DIM" \
    --privacy_alpha "$SYN_PRIVACY_ALPHA" \
    --delta "$SYN_DELTA" \
    --discount_gamma "$SYN_GAMMA"
fi

# --------------------------------------------------
# Exp B: Learning-augmented posterior-driven scheduler
# Repo mapping: synthetic CLIP-LA / exact_wf / naive_la comparison with single-agent setting
# --------------------------------------------------
if [[ "$RUN_EXP_BCD" == "1" ]]; then
  run_cmd python experiments/new_experiments.py \
    --sections synthetic \
    --seeds "$SEEDS_EXP_B" \
    --output_root "$OUT_ROOT/expB_learning_augmented" \
    --num_agents 1 \
    --num_blocks 60 \
    --inner_updates "$SYN_INNER_UPDATES" \
    --train_batch_size "$SYN_TRAIN_BATCH" \
    --eval_batch_size "$SYN_EVAL_BATCH" \
    --episode_length 1 \
    --hidden_dim "$SYN_HIDDEN_DIM" \
    --message_dim 4 \
    --total_rho_budget 4 \
    --privacy_alpha "$SYN_PRIVACY_ALPHA" \
    --delta "$SYN_DELTA" \
    --discount_gamma "$SYN_GAMMA"

  # ------------------------------------------------
  # Exp C: DP-safety audit
  # Repo mapping: same synthetic setup, summarized through naive_la overspend ratios
  # ------------------------------------------------
  run_cmd python experiments/new_experiments.py \
    --sections synthetic \
    --seeds "$SEEDS_EXP_B" \
    --output_root "$OUT_ROOT/expC_dp_audit" \
    --num_agents 1 \
    --num_blocks 60 \
    --inner_updates "$SYN_INNER_UPDATES" \
    --train_batch_size "$SYN_TRAIN_BATCH" \
    --eval_batch_size "$SYN_EVAL_BATCH" \
    --episode_length 1 \
    --hidden_dim "$SYN_HIDDEN_DIM" \
    --message_dim 4 \
    --total_rho_budget 4 \
    --privacy_alpha "$SYN_PRIVACY_ALPHA" \
    --delta "$SYN_DELTA" \
    --discount_gamma "$SYN_GAMMA"

  # ------------------------------------------------
  # Exp D: CLIP-LA fix
  # Repo mapping: synthetic setup at larger budgets to compare clip_la / exact_wf / dpmac
  # ------------------------------------------------
  run_cmd python experiments/new_experiments.py \
    --sections synthetic \
    --seeds "$SEEDS_EXP_D" \
    --output_root "$OUT_ROOT/expD_clip_la_fix" \
    --num_agents 1 \
    --num_blocks 60 \
    --inner_updates "$SYN_INNER_UPDATES" \
    --train_batch_size "$SYN_TRAIN_BATCH" \
    --eval_batch_size "$SYN_EVAL_BATCH" \
    --episode_length 1 \
    --hidden_dim "$SYN_HIDDEN_DIM" \
    --message_dim 4 \
    --total_rho_budget 64 \
    --clip_multiplier 1.2 \
    --privacy_alpha "$SYN_PRIVACY_ALPHA" \
    --delta "$SYN_DELTA" \
    --discount_gamma "$SYN_GAMMA"
fi

# --------------------------------------------------
# PettingZoo MPE stress test
# Paper intent: CN / CCN / PP with private communication baselines
# Repo mapping: MPE section of new_experiments
# --------------------------------------------------
if [[ "$RUN_MPE" == "1" ]]; then
  run_cmd python experiments/new_experiments.py \
    --sections mpe \
    --seeds "$SEEDS_MPE" \
    --output_root "$OUT_ROOT/mpe_stress_test" \
    --mpe_scenarios "$MPE_SCENARIOS" \
    --mpe_algorithms "$MPE_ALGOS" \
    --episodes "$MPE_EPISODES"
fi

echo -e "\n[DONE] new_PIL experiment run completed." | tee -a "$LOG_FILE"
TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START_TS ))
echo "[DONE] Total elapsed: $(format_seconds "$TOTAL_ELAPSED")" | tee -a "$LOG_FILE"
echo "[DONE] Log file: $LOG_FILE" | tee -a "$LOG_FILE"
