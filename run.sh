#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Optional: activate conda env before running
# conda activate PIL

# ------------------------
# Global knobs (override via env vars)
# ------------------------
SEEDS="${SEEDS:-7,11,23}"
NUM_BLOCKS="${NUM_BLOCKS:-50}"
INNER_UPDATES="${INNER_UPDATES:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EPISODE_LENGTH="${EPISODE_LENGTH:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
MESSAGE_DIM="${MESSAGE_DIM:-2}"

MPE_EPISODES="${MPE_EPISODES:-100}"
MPE_SCENARIOS="${MPE_SCENARIOS:-cn,ccn,pp}"
MPE_ALGOS="${MPE_ALGOS:-pil,dpmac,i2c,tarmac,maddpg}"

MATRIX_GAMES="${MATRIX_GAMES:-binary_sum,multi_round_sum}"
MATRIX_ALGOS="${MATRIX_ALGOS:-pil,dpmac,i2c,tarmac,maddpg}"

RUN_SINGLE="${RUN_SINGLE:-1}"
RUN_COMPARE_PIL_DPMAC="${RUN_COMPARE_PIL_DPMAC:-1}"
RUN_COMPARE_BASELINES="${RUN_COMPARE_BASELINES:-1}"
RUN_MATRIX_SUITE="${RUN_MATRIX_SUITE:-1}"
RUN_MPE_SUITE="${RUN_MPE_SUITE:-1}"

OUT_ROOT="$ROOT_DIR/experiments/exp_runs"
OUT_SINGLE="$OUT_ROOT/01_single_trainers"
OUT_COMPARE_PVSD="$OUT_ROOT/02_compare/pil_vs_dpmac"
OUT_COMPARE_BASE="$OUT_ROOT/02_compare/baselines"
OUT_MATRIX="$OUT_ROOT/03_matrix_suite"
OUT_MPE="$OUT_ROOT/04_mpe_suite"
OUT_LOG="$OUT_ROOT/logs"

PLOT_ROOT="$ROOT_DIR/plots/exp_runs"
PLOT_COMPARE_PVSD="$PLOT_ROOT/02_compare/pil_vs_dpmac"
PLOT_COMPARE_BASE="$PLOT_ROOT/02_compare/baselines"
PLOT_MATRIX="$PLOT_ROOT/03_matrix_suite"
PLOT_MPE="$PLOT_ROOT/04_mpe_suite"

mkdir -p "$OUT_SINGLE" "$OUT_COMPARE_PVSD" "$OUT_COMPARE_BASE" "$OUT_MATRIX" "$OUT_MPE" "$OUT_LOG"
mkdir -p "$PLOT_COMPARE_PVSD" "$PLOT_COMPARE_BASE" "$PLOT_MATRIX" "$PLOT_MPE"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_LOG/run_${STAMP}.log"

SCRIPT_START_TS=$(date +%s)

format_seconds() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02dh:%02dm:%02ds" "$h" "$m" "$s"
}

STEPS_TOTAL=0
[[ "$RUN_SINGLE" == "1" ]] && STEPS_TOTAL=$((STEPS_TOTAL + 4))
[[ "$RUN_COMPARE_PIL_DPMAC" == "1" ]] && STEPS_TOTAL=$((STEPS_TOTAL + 1))
[[ "$RUN_COMPARE_BASELINES" == "1" ]] && STEPS_TOTAL=$((STEPS_TOTAL + 1))
[[ "$RUN_MATRIX_SUITE" == "1" ]] && STEPS_TOTAL=$((STEPS_TOTAL + 1))
[[ "$RUN_MPE_SUITE" == "1" ]] && STEPS_TOTAL=$((STEPS_TOTAL + 1))

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
echo "[INFO] Seeds: $SEEDS" | tee -a "$LOG_FILE"
echo "[INFO] Planned steps: $STEPS_TOTAL" | tee -a "$LOG_FILE"
echo "[INFO] Started at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

if [[ "$RUN_SINGLE" == "1" ]]; then
  run_cmd python experiments/run_pil_aps.py \
    --num_blocks "$NUM_BLOCKS" \
    --inner_updates "$INNER_UPDATES" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --episode_length "$EPISODE_LENGTH" \
    --hidden_dim "$HIDDEN_DIM" \
    --message_dim "$MESSAGE_DIM" \
    --output "$OUT_SINGLE/pil_aps_metrics.json"

  run_cmd python experiments/run_dpmac.py \
    --num_blocks "$NUM_BLOCKS" \
    --inner_updates "$INNER_UPDATES" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --episode_length "$EPISODE_LENGTH" \
    --hidden_dim "$HIDDEN_DIM" \
    --message_dim "$MESSAGE_DIM" \
    --output "$OUT_SINGLE/dpmac_metrics.json"

  run_cmd python experiments/run_i2c.py \
    --num_blocks "$NUM_BLOCKS" \
    --inner_updates "$INNER_UPDATES" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --episode_length "$EPISODE_LENGTH" \
    --hidden_dim "$HIDDEN_DIM" \
    --message_dim "$MESSAGE_DIM" \
    --output "$OUT_SINGLE/i2c_metrics.json"

  run_cmd python experiments/run_maddpg.py \
    --num_blocks "$NUM_BLOCKS" \
    --inner_updates "$INNER_UPDATES" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --episode_length "$EPISODE_LENGTH" \
    --hidden_dim "$HIDDEN_DIM" \
    --message_dim "$MESSAGE_DIM" \
    --output "$OUT_SINGLE/maddpg_metrics.json"
fi

if [[ "$RUN_COMPARE_PIL_DPMAC" == "1" ]]; then
  run_cmd python experiments/compare_pil_vs_dpmac.py \
    --num_blocks "$NUM_BLOCKS" \
    --inner_updates "$INNER_UPDATES" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --episode_length "$EPISODE_LENGTH" \
    --hidden_dim "$HIDDEN_DIM" \
    --message_dim "$MESSAGE_DIM" \
    --seeds "$SEEDS" \
    --pil_output "$OUT_COMPARE_PVSD/pil_aps_metrics.json" \
    --dpmac_output "$OUT_COMPARE_PVSD/dpmac_metrics.json" \
    --summary_output "$OUT_COMPARE_PVSD/pil_vs_dpmac_summary.json" \
    --plots_dir "$PLOT_COMPARE_PVSD"
fi

if [[ "$RUN_COMPARE_BASELINES" == "1" ]]; then
  run_cmd python experiments/compare_baselines.py \
    --num_blocks "$NUM_BLOCKS" \
    --inner_updates "$INNER_UPDATES" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --episode_length "$EPISODE_LENGTH" \
    --hidden_dim "$HIDDEN_DIM" \
    --message_dim "$MESSAGE_DIM" \
    --seeds "$SEEDS" \
    --baselines pil,dpmac,i2c,maddpg \
    --output_dir "$OUT_COMPARE_BASE" \
    --summary_output "$OUT_COMPARE_BASE/baseline_summary.json" \
    --plots_dir "$PLOT_COMPARE_BASE"
fi

if [[ "$RUN_MATRIX_SUITE" == "1" ]]; then
  run_cmd python experiments/compare_matrix_games.py \
    --games "$MATRIX_GAMES" \
    --algorithms "$MATRIX_ALGOS" \
    --seeds "$SEEDS" \
    --num_blocks 25 \
    --inner_updates "$INNER_UPDATES" \
    --output_dir "$OUT_MATRIX" \
    --summary_output "$OUT_MATRIX/matrix_games_summary.json" \
    --plots_dir "$PLOT_MATRIX"
fi

if [[ "$RUN_MPE_SUITE" == "1" ]]; then
  run_cmd python experiments/compare_mpe_suite.py \
    --scenarios "$MPE_SCENARIOS" \
    --algorithms "$MPE_ALGOS" \
    --seeds "$SEEDS" \
    --episodes "$MPE_EPISODES" \
    --output_dir "$OUT_MPE" \
    --summary_output "$OUT_MPE/mpe_suite_summary.json" \
    --plots_dir "$PLOT_MPE"
fi

echo -e "\n[DONE] Full experiment run completed." | tee -a "$LOG_FILE"
TOTAL_ELAPSED=$(( $(date +%s) - SCRIPT_START_TS ))
echo "[DONE] Total elapsed: $(format_seconds "$TOTAL_ELAPSED")" | tee -a "$LOG_FILE"
echo "[DONE] Log file: $LOG_FILE" | tee -a "$LOG_FILE"
