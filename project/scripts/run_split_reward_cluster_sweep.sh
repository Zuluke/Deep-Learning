#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

TARGETS="${TARGETS:-gf_2pow2_mult,hamming_weight_n4,hamming_weight_n5}"
MODES="${MODES:-none,v2_progress,v3_frontier,v4_sticky_frontier}"
PROFILE="${PROFILE:-cpu}"
USE_GADGETS="${USE_GADGETS:-on}"
TRAINING_STEPS="${TRAINING_STEPS:-5000}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_MCTS_SIMULATIONS="${NUM_MCTS_SIMULATIONS:-16}"
ACTION_DICTIONARY="${ACTION_DICTIONARY:-low-weight}"
MAX_ACTION_WEIGHT="${MAX_ACTION_WEIGHT:-2}"
TENSOR_OVERLAP_MAX_WEIGHT="${TENSOR_OVERLAP_MAX_WEIGHT:-5}"
TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET="${TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET:-128}"
GADGET_CLOSURE_MAX_WEIGHT="${GADGET_CLOSURE_MAX_WEIGHT:-4}"
MAX_NUM_MOVES="${MAX_NUM_MOVES:-0}"
NUM_PAST_FACTORS_TO_OBSERVE="${NUM_PAST_FACTORS_TO_OBSERVE:-0}"
LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-0.10}"
LAMBDA_FRONTIER="${LAMBDA_FRONTIER:-0.10}"
ACTION_PRIOR="${ACTION_PRIOR:-none}"
ACTION_PRIOR_BETA="${ACTION_PRIOR_BETA:-1.0}"
ACTION_PRIOR_RESIDUAL_WEIGHT="${ACTION_PRIOR_RESIDUAL_WEIGHT:-1.0}"
ACTION_PRIOR_MIXED_DROP_WEIGHT="${ACTION_PRIOR_MIXED_DROP_WEIGHT:-1.0}"
ACTION_PRIOR_MIXED_MASS_WEIGHT="${ACTION_PRIOR_MIXED_MASS_WEIGHT:-0.25}"
ACTION_PRIOR_HAMMING_WEIGHT="${ACTION_PRIOR_HAMMING_WEIGHT:-0.05}"
ACTION_PRIOR_GADGET_BONUS="${ACTION_PRIOR_GADGET_BONUS:-0.25}"
ACTION_PRIOR_TOP_K="${ACTION_PRIOR_TOP_K:-0}"
ACTION_PRIOR_STANDARDIZE="${ACTION_PRIOR_STANDARDIZE:-1}"
ACTION_PRIOR_CANONICAL_ONLY="${ACTION_PRIOR_CANONICAL_ONLY:-1}"
FRONTIER_REPLAY_FRACTION="${FRONTIER_REPLAY_FRACTION:-0.0}"
FRONTIER_REPLAY_MIN_MOVES="${FRONTIER_REPLAY_MIN_MOVES:-0}"
FRONTIER_REPLAY_MIN_RESIDUAL_DROP="${FRONTIER_REPLAY_MIN_RESIDUAL_DROP:-0.0}"
PARTITION_PRESET="${PARTITION_PRESET:-balanced}"
MASK_PADDED_ACTIONS="${MASK_PADDED_ACTIONS:-1}"
FORCE_CANONICAL_BASIS="${FORCE_CANONICAL_BASIS:-1}"
SEED="${SEED:-2024}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
MATERIALIZE_TIMEOUT_SEC="${MATERIALIZE_TIMEOUT_SEC:-600}"
BASELINE_T_COSTS="${BASELINE_T_COSTS:-}"
RUN_LABEL="${RUN_LABEL:-cluster_candidate}"

cd "$PROJECT_ROOT"
mkdir -p results/csv results/reports results/logs/split_reward_target_sweep

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected Python interpreter at $PYTHON_BIN" >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN" scripts/run_split_reward_target_sweep.py
  --targets "$TARGETS"
  --modes "$MODES"
  --profile "$PROFILE"
  --use-gadgets "$USE_GADGETS"
  --training-steps "$TRAINING_STEPS"
  --eval-frequency "$EVAL_FREQUENCY"
  --batch-size "$BATCH_SIZE"
  --num-mcts-simulations "$NUM_MCTS_SIMULATIONS"
  --action-dictionary "$ACTION_DICTIONARY"
  --max-action-weight "$MAX_ACTION_WEIGHT"
  --tensor-overlap-max-weight "$TENSOR_OVERLAP_MAX_WEIGHT"
  --tensor-overlap-max-actions-per-target "$TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET"
  --gadget-closure-max-weight "$GADGET_CLOSURE_MAX_WEIGHT"
  --max-num-moves "$MAX_NUM_MOVES"
  --num-past-factors-to-observe "$NUM_PAST_FACTORS_TO_OBSERVE"
  --lambda-residual "$LAMBDA_RESIDUAL"
  --lambda-frontier "$LAMBDA_FRONTIER"
  --action-prior "$ACTION_PRIOR"
  --action-prior-beta "$ACTION_PRIOR_BETA"
  --action-prior-residual-weight "$ACTION_PRIOR_RESIDUAL_WEIGHT"
  --action-prior-mixed-drop-weight "$ACTION_PRIOR_MIXED_DROP_WEIGHT"
  --action-prior-mixed-mass-weight "$ACTION_PRIOR_MIXED_MASS_WEIGHT"
  --action-prior-hamming-weight "$ACTION_PRIOR_HAMMING_WEIGHT"
  --action-prior-gadget-bonus "$ACTION_PRIOR_GADGET_BONUS"
  --action-prior-top-k "$ACTION_PRIOR_TOP_K"
  --frontier-replay-fraction "$FRONTIER_REPLAY_FRACTION"
  --frontier-replay-min-moves "$FRONTIER_REPLAY_MIN_MOVES"
  --frontier-replay-min-residual-drop "$FRONTIER_REPLAY_MIN_RESIDUAL_DROP"
  --partition-preset "$PARTITION_PRESET"
  --seed "$SEED"
  --timeout-sec "$TIMEOUT_SEC"
  --materialize-timeout-sec "$MATERIALIZE_TIMEOUT_SEC"
  --output-csv "results/csv/split_reward_target_sweep_${RUN_LABEL}.csv"
  --report-path "results/reports/split_reward_target_sweep_${RUN_LABEL}.md"
)

if [[ "$MASK_PADDED_ACTIONS" == "1" || "$MASK_PADDED_ACTIONS" == "true" ]]; then
  cmd+=(--mask-padded-actions)
else
  cmd+=(--no-mask-padded-actions)
fi

if [[ "$FORCE_CANONICAL_BASIS" == "1" || "$FORCE_CANONICAL_BASIS" == "true" ]]; then
  cmd+=(--force-canonical-basis)
fi

if [[ "$ACTION_PRIOR_STANDARDIZE" == "1" || "$ACTION_PRIOR_STANDARDIZE" == "true" ]]; then
  cmd+=(--action-prior-standardize)
else
  cmd+=(--no-action-prior-standardize)
fi

if [[ "$ACTION_PRIOR_CANONICAL_ONLY" == "1" || "$ACTION_PRIOR_CANONICAL_ONLY" == "true" ]]; then
  cmd+=(--action-prior-canonical-only)
else
  cmd+=(--no-action-prior-canonical-only)
fi

if [[ -n "$BASELINE_T_COSTS" ]]; then
  cmd+=(--baseline-t-costs "$BASELINE_T_COSTS")
fi

printf '[split-reward-sweep] %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
