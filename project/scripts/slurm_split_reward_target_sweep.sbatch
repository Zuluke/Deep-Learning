#!/usr/bin/env bash
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=atq_split_sweep
#SBATCH --output=results/logs/slurm/%x_%j.out
#SBATCH --error=results/logs/slurm/%x_%j.err

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Deep-Learning/project}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

export PROFILE="${PROFILE:-cuda}"
export USE_GADGETS="${USE_GADGETS:-on}"
export TARGETS="${TARGETS:-gf_2pow2_mult,hamming_weight_n4,hamming_weight_n5}"
export MODES="${MODES:-none,v2_progress,v3_frontier,v4_sticky_frontier}"
export TRAINING_STEPS="${TRAINING_STEPS:-5000}"
export EVAL_FREQUENCY="${EVAL_FREQUENCY:-100}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_MCTS_SIMULATIONS="${NUM_MCTS_SIMULATIONS:-16}"
export ACTION_DICTIONARY="${ACTION_DICTIONARY:-low-weight}"
export MAX_ACTION_WEIGHT="${MAX_ACTION_WEIGHT:-2}"
export TENSOR_OVERLAP_MAX_WEIGHT="${TENSOR_OVERLAP_MAX_WEIGHT:-5}"
export TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET="${TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET:-128}"
export GADGET_CLOSURE_MAX_WEIGHT="${GADGET_CLOSURE_MAX_WEIGHT:-4}"
export MAX_NUM_MOVES="${MAX_NUM_MOVES:-0}"
export NUM_PAST_FACTORS_TO_OBSERVE="${NUM_PAST_FACTORS_TO_OBSERVE:-0}"
export LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-0.10}"
export LAMBDA_FRONTIER="${LAMBDA_FRONTIER:-0.10}"
export ACTION_PRIOR="${ACTION_PRIOR:-none}"
export ACTION_PRIOR_BETA="${ACTION_PRIOR_BETA:-1.0}"
export ACTION_PRIOR_RESIDUAL_WEIGHT="${ACTION_PRIOR_RESIDUAL_WEIGHT:-1.0}"
export ACTION_PRIOR_MIXED_DROP_WEIGHT="${ACTION_PRIOR_MIXED_DROP_WEIGHT:-1.0}"
export ACTION_PRIOR_MIXED_MASS_WEIGHT="${ACTION_PRIOR_MIXED_MASS_WEIGHT:-0.25}"
export ACTION_PRIOR_HAMMING_WEIGHT="${ACTION_PRIOR_HAMMING_WEIGHT:-0.05}"
export ACTION_PRIOR_GADGET_BONUS="${ACTION_PRIOR_GADGET_BONUS:-0.25}"
export ACTION_PRIOR_TOP_K="${ACTION_PRIOR_TOP_K:-0}"
export ACTION_PRIOR_STANDARDIZE="${ACTION_PRIOR_STANDARDIZE:-1}"
export ACTION_PRIOR_CANONICAL_ONLY="${ACTION_PRIOR_CANONICAL_ONLY:-1}"
export FRONTIER_REPLAY_FRACTION="${FRONTIER_REPLAY_FRACTION:-0.0}"
export FRONTIER_REPLAY_MIN_MOVES="${FRONTIER_REPLAY_MIN_MOVES:-0}"
export FRONTIER_REPLAY_MIN_RESIDUAL_DROP="${FRONTIER_REPLAY_MIN_RESIDUAL_DROP:-0.0}"
export PARTITION_PRESET="${PARTITION_PRESET:-balanced}"
export MASK_PADDED_ACTIONS="${MASK_PADDED_ACTIONS:-1}"
export FORCE_CANONICAL_BASIS="${FORCE_CANONICAL_BASIS:-1}"
export SEED="${SEED:-2024}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-10800}"
export MATERIALIZE_TIMEOUT_SEC="${MATERIALIZE_TIMEOUT_SEC:-900}"
export RUN_LABEL="${RUN_LABEL:-slurm_${SLURM_JOB_ID:-manual}}"

cd "$PROJECT_ROOT"
mkdir -p results/logs/slurm results/csv results/reports

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected Python interpreter at $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$PROFILE" == "cuda" ]]; then
  nvidia-smi || true
  "$PYTHON_BIN" - <<'PY'
import json
import jax

payload = {
    "backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
}
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["backend"] != "gpu":
    raise SystemExit("PROFILE=cuda but JAX backend is not gpu.")
PY
fi

exec scripts/run_split_reward_cluster_sweep.sh
