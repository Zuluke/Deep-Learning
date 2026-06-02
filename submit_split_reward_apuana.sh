#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/project/scripts/slurm_split_reward_target_sweep.sbatch"

MODE="submit"
PRESET="split4"
FOLLOW=1
WATCH_MODE=0
LINES=80
INTERVAL=5
ATTACH_JOBID=""
DRY_RUN=0

JOB_NAME=""
PARTITION="${SLURM_PARTITION:-short-complex}"
QOS="${SLURM_QOS:-complex}"
GRES="${SLURM_GRES:-gpu:1}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
MEMORY="${SLURM_MEM:-48G}"
TIME_LIMIT="${SLURM_TIME:-04:00:00}"
LOGS_DIR="${SLURM_LOGS_DIR:-${SCRIPT_DIR}/project/results/logs/slurm}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Deep-Learning/project}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

PROFILE="cuda"
USE_GADGETS="on"
TARGETS="mod_5_4,gf_2pow2_mult,hamming_weight_n4,hamming_weight_n5"
MODES="none,v2_progress,v3_frontier,v4_sticky_frontier"
TRAINING_STEPS="5000"
EVAL_FREQUENCY="100"
BATCH_SIZE="32"
NUM_MCTS_SIMULATIONS="16"
ACTION_DICTIONARY="low-weight"
MAX_ACTION_WEIGHT="2"
TENSOR_OVERLAP_MAX_WEIGHT="5"
TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET="128"
GADGET_CLOSURE_MAX_WEIGHT="4"
MAX_NUM_MOVES="0"
NUM_PAST_FACTORS_TO_OBSERVE="0"
LAMBDA_RESIDUAL="0.10"
LAMBDA_FRONTIER="0.10"
ACTION_PRIOR="none"
ACTION_PRIOR_BETA="1.0"
ACTION_PRIOR_RESIDUAL_WEIGHT="1.0"
ACTION_PRIOR_MIXED_DROP_WEIGHT="1.0"
ACTION_PRIOR_MIXED_MASS_WEIGHT="0.25"
ACTION_PRIOR_HAMMING_WEIGHT="0.05"
ACTION_PRIOR_GADGET_BONUS="0.25"
ACTION_PRIOR_TOP_K="0"
ACTION_PRIOR_STANDARDIZE="1"
ACTION_PRIOR_CANONICAL_ONLY="1"
FRONTIER_REPLAY_FRACTION="0.0"
FRONTIER_REPLAY_MIN_MOVES="0"
FRONTIER_REPLAY_MIN_RESIDUAL_DROP="0.0"
PARTITION_PRESET="balanced"
MASK_PADDED_ACTIONS="1"
MASK_REPEATED_ACTIONS="0"
FORCE_CANONICAL_BASIS="1"
SEED="2024"
TIMEOUT_SEC="10800"
MATERIALIZE_TIMEOUT_SEC="900"
RUN_LABEL=""
BASELINE_T_COSTS=""

EXTRA_EXPORTS=()
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./submit_split_reward_apuana.sh [options]
  ./submit_split_reward_apuana.sh --attach JOBID [options]

Main presets:
  --preset smoke          Small sanity run: mod_5_4, none/v3, 200 steps
  --preset split4         Default run: four trainable targets, none/v2/v3/v4
  --preset split4-prior   Same as split4 with AlphaQuantum-only split action prior
  --preset production     Longer split4 run with more search budget

Cluster options:
  --job-name NAME         Slurm job name
  --partition NAME        Slurm partition (default: short-complex)
  --qos NAME              Slurm QoS (default: complex)
  --gres VALUE            Slurm GRES (default: gpu:1)
  --cpus N                CPUs per task (default: 8)
  --mem SIZE              Memory (default: 48G)
  --time HH:MM:SS         Time limit (default: 04:00:00)
  --logs-dir DIR          Local log directory used for following logs
  --sbatch-arg ARG        Extra sbatch argument, repeatable
  --dry-run               Print sbatch command without submitting

Sweep options:
  --targets LIST          Comma-separated target list
  --modes LIST            Comma-separated split reward modes
  --steps N               Training steps
  --eval-frequency N      Evaluation frequency
  --batch-size N          Batch size
  --mcts N                Number of MCTS simulations
  --action-dictionary X   low-weight, tensor-overlap, gadget-closure, or full
  --max-action-weight N   Max low-weight action weight
  --action-prior X        none, residual, or split
  --action-prior-beta X   Split/residual prior strength
  --action-prior-top-k N  Optional prior-guided action narrowing
  --lambda-residual X     v3/v4 residual progress weight
  --lambda-frontier X     v4 frontier regret weight
  --partition-preset X    Split partition preset
  --mask-repeated-actions Mask factors already used in the current episode
  --seed N                Random seed
  --timeout-sec N         Per-run timeout used by Python sweep wrapper
  --materialize-timeout-sec N
  --run-label LABEL       Output CSV/report suffix
  --baseline-t-costs MAP  Optional target=cost,target=cost map for budgeted modes
  --export NAME=VALUE     Extra environment variable passed to the job

Runtime paths:
  --project-root DIR      Project root on the cluster node
  --python-bin PATH       Python interpreter on the cluster node
  --profile cpu|cuda      Runtime profile (default: cuda)

Log options:
  --no-follow             Submit and exit after printing job id
  --watch                 Watch last log lines instead of tail -F
  --lines N               Lines for --watch (default: 80)
  --interval S            Seconds between watch refreshes (default: 5)
  --attach JOBID          Attach to logs of an existing job

Examples:
  ./submit_split_reward_apuana.sh --preset smoke --dry-run
  ./submit_split_reward_apuana.sh --preset split4 --time 08:00:00
  ./submit_split_reward_apuana.sh --preset split4-prior --action-prior-beta 0.5
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

to_abs_path() {
  local value="$1"
  if [[ "$value" == /* || "$value" == "\$HOME"* || "$value" == "~"* ]]; then
    echo "$value"
    return
  fi
  echo "$PWD/${value#./}"
}

apply_preset() {
  case "$PRESET" in
    smoke)
      JOB_NAME="${JOB_NAME:-atq_split_smoke}"
      TARGETS="mod_5_4"
      MODES="none,v3_frontier"
      TRAINING_STEPS="200"
      EVAL_FREQUENCY="50"
      BATCH_SIZE="16"
      NUM_MCTS_SIMULATIONS="8"
      TIME_LIMIT="${SLURM_TIME:-00:30:00}"
      TIMEOUT_SEC="1200"
      MATERIALIZE_TIMEOUT_SEC="300"
      RUN_LABEL="${RUN_LABEL:-smoke}"
      ;;
    split4)
      JOB_NAME="${JOB_NAME:-atq_split4}"
      RUN_LABEL="${RUN_LABEL:-split4}"
      ;;
    split4-prior)
      JOB_NAME="${JOB_NAME:-atq_split4_prior}"
      ACTION_PRIOR="split"
      ACTION_PRIOR_BETA="1.0"
      ACTION_PRIOR_TOP_K="0"
      RUN_LABEL="${RUN_LABEL:-split4_prior}"
      ;;
    production)
      JOB_NAME="${JOB_NAME:-atq_split4_prod}"
      TRAINING_STEPS="20000"
      EVAL_FREQUENCY="200"
      BATCH_SIZE="64"
      NUM_MCTS_SIMULATIONS="32"
      TIME_LIMIT="${SLURM_TIME:-24:00:00}"
      TIMEOUT_SEC="43200"
      MATERIALIZE_TIMEOUT_SEC="1200"
      RUN_LABEL="${RUN_LABEL:-production}"
      ;;
    *)
      die "Unknown preset: $PRESET"
      ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preset) PRESET="${2:-}"; shift 2 ;;
      --job-name) JOB_NAME="${2:-}"; shift 2 ;;
      --partition) PARTITION="${2:-}"; shift 2 ;;
      --qos) QOS="${2:-}"; shift 2 ;;
      --gres) GRES="${2:-}"; shift 2 ;;
      --cpus) CPUS="${2:-}"; shift 2 ;;
      --mem) MEMORY="${2:-}"; shift 2 ;;
      --time) TIME_LIMIT="${2:-}"; shift 2 ;;
      --logs-dir) LOGS_DIR="${2:-}"; shift 2 ;;
      --project-root) PROJECT_ROOT="${2:-}"; shift 2 ;;
      --python-bin) PYTHON_BIN="${2:-}"; shift 2 ;;
      --profile) PROFILE="${2:-}"; shift 2 ;;
      --targets) TARGETS="${2:-}"; shift 2 ;;
      --modes) MODES="${2:-}"; shift 2 ;;
      --steps) TRAINING_STEPS="${2:-}"; shift 2 ;;
      --eval-frequency) EVAL_FREQUENCY="${2:-}"; shift 2 ;;
      --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
      --mcts) NUM_MCTS_SIMULATIONS="${2:-}"; shift 2 ;;
      --action-dictionary) ACTION_DICTIONARY="${2:-}"; shift 2 ;;
      --max-action-weight) MAX_ACTION_WEIGHT="${2:-}"; shift 2 ;;
      --action-prior) ACTION_PRIOR="${2:-}"; shift 2 ;;
      --action-prior-beta) ACTION_PRIOR_BETA="${2:-}"; shift 2 ;;
      --action-prior-top-k) ACTION_PRIOR_TOP_K="${2:-}"; shift 2 ;;
      --lambda-residual) LAMBDA_RESIDUAL="${2:-}"; shift 2 ;;
      --lambda-frontier) LAMBDA_FRONTIER="${2:-}"; shift 2 ;;
      --partition-preset) PARTITION_PRESET="${2:-}"; shift 2 ;;
      --mask-repeated-actions) MASK_REPEATED_ACTIONS="1"; shift ;;
      --no-mask-repeated-actions) MASK_REPEATED_ACTIONS="0"; shift ;;
      --seed) SEED="${2:-}"; shift 2 ;;
      --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
      --materialize-timeout-sec) MATERIALIZE_TIMEOUT_SEC="${2:-}"; shift 2 ;;
      --run-label) RUN_LABEL="${2:-}"; shift 2 ;;
      --baseline-t-costs) BASELINE_T_COSTS="${2:-}"; shift 2 ;;
      --export) EXTRA_EXPORTS+=("${2:-}"); shift 2 ;;
      --sbatch-arg) EXTRA_SBATCH_ARGS+=("${2:-}"); shift 2 ;;
      --no-follow) FOLLOW=0; shift ;;
      --watch) WATCH_MODE=1; shift ;;
      --lines) LINES="${2:-80}"; shift 2 ;;
      --interval) INTERVAL="${2:-5}"; shift 2 ;;
      --attach) ATTACH_JOBID="${2:-}"; MODE="attach"; shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown argument: $1" ;;
    esac
  done
}

discover_preset() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preset)
        PRESET="${2:-}"
        shift 2
        ;;
      --)
        break
        ;;
      *)
        shift
        ;;
    esac
  done
}

append_export() {
  local name="$1"
  local value="$2"
  EXPORTS+=("${name}=${value}")
}

build_exports() {
  EXPORTS=()
  append_export PROJECT_ROOT "$PROJECT_ROOT"
  append_export PYTHON_BIN "$PYTHON_BIN"
  append_export PROFILE "$PROFILE"
  append_export USE_GADGETS "$USE_GADGETS"
  append_export TARGETS "$TARGETS"
  append_export MODES "$MODES"
  append_export TRAINING_STEPS "$TRAINING_STEPS"
  append_export EVAL_FREQUENCY "$EVAL_FREQUENCY"
  append_export BATCH_SIZE "$BATCH_SIZE"
  append_export NUM_MCTS_SIMULATIONS "$NUM_MCTS_SIMULATIONS"
  append_export ACTION_DICTIONARY "$ACTION_DICTIONARY"
  append_export MAX_ACTION_WEIGHT "$MAX_ACTION_WEIGHT"
  append_export TENSOR_OVERLAP_MAX_WEIGHT "$TENSOR_OVERLAP_MAX_WEIGHT"
  append_export TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET "$TENSOR_OVERLAP_MAX_ACTIONS_PER_TARGET"
  append_export GADGET_CLOSURE_MAX_WEIGHT "$GADGET_CLOSURE_MAX_WEIGHT"
  append_export MAX_NUM_MOVES "$MAX_NUM_MOVES"
  append_export NUM_PAST_FACTORS_TO_OBSERVE "$NUM_PAST_FACTORS_TO_OBSERVE"
  append_export LAMBDA_RESIDUAL "$LAMBDA_RESIDUAL"
  append_export LAMBDA_FRONTIER "$LAMBDA_FRONTIER"
  append_export ACTION_PRIOR "$ACTION_PRIOR"
  append_export ACTION_PRIOR_BETA "$ACTION_PRIOR_BETA"
  append_export ACTION_PRIOR_RESIDUAL_WEIGHT "$ACTION_PRIOR_RESIDUAL_WEIGHT"
  append_export ACTION_PRIOR_MIXED_DROP_WEIGHT "$ACTION_PRIOR_MIXED_DROP_WEIGHT"
  append_export ACTION_PRIOR_MIXED_MASS_WEIGHT "$ACTION_PRIOR_MIXED_MASS_WEIGHT"
  append_export ACTION_PRIOR_HAMMING_WEIGHT "$ACTION_PRIOR_HAMMING_WEIGHT"
  append_export ACTION_PRIOR_GADGET_BONUS "$ACTION_PRIOR_GADGET_BONUS"
  append_export ACTION_PRIOR_TOP_K "$ACTION_PRIOR_TOP_K"
  append_export ACTION_PRIOR_STANDARDIZE "$ACTION_PRIOR_STANDARDIZE"
  append_export ACTION_PRIOR_CANONICAL_ONLY "$ACTION_PRIOR_CANONICAL_ONLY"
  append_export FRONTIER_REPLAY_FRACTION "$FRONTIER_REPLAY_FRACTION"
  append_export FRONTIER_REPLAY_MIN_MOVES "$FRONTIER_REPLAY_MIN_MOVES"
  append_export FRONTIER_REPLAY_MIN_RESIDUAL_DROP "$FRONTIER_REPLAY_MIN_RESIDUAL_DROP"
  append_export PARTITION_PRESET "$PARTITION_PRESET"
  append_export MASK_PADDED_ACTIONS "$MASK_PADDED_ACTIONS"
  append_export MASK_REPEATED_ACTIONS "$MASK_REPEATED_ACTIONS"
  append_export FORCE_CANONICAL_BASIS "$FORCE_CANONICAL_BASIS"
  append_export SEED "$SEED"
  append_export TIMEOUT_SEC "$TIMEOUT_SEC"
  append_export MATERIALIZE_TIMEOUT_SEC "$MATERIALIZE_TIMEOUT_SEC"
  append_export RUN_LABEL "$RUN_LABEL"
  if [[ -n "$BASELINE_T_COSTS" ]]; then
    append_export BASELINE_T_COSTS "$BASELINE_T_COSTS"
  fi
  if [[ ${#EXTRA_EXPORTS[@]} -gt 0 ]]; then
    EXPORTS+=("${EXTRA_EXPORTS[@]}")
  fi
}

resolve_log_file() {
  local jobid="$1"
  local extension="$2"
  local match

  match="$(find "$LOGS_DIR" -maxdepth 1 -type f -name "*_${jobid}.${extension}" | head -n 1 || true)"
  if [[ -n "$match" ]]; then
    echo "$match"
    return
  fi
  echo "${LOGS_DIR}/${JOB_NAME}_${jobid}.${extension}"
}

wait_for_logs() {
  local out_file="$1"
  local err_file="$2"
  local waited=0
  while [[ ! -f "$out_file" || ! -f "$err_file" ]]; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= 180 )); then
      break
    fi
  done
}

follow_logs() {
  local out_file="$1"
  local err_file="$2"
  echo "Following logs:"
  echo "  ERR: $err_file"
  echo "  OUT: $out_file"
  tail -F -n 200 "$err_file" | sed -u 's/^/[ERR] /' &
  local pid_err=$!
  tail -F -n 200 "$out_file" | sed -u 's/^/[OUT] /' &
  local pid_out=$!
  trap 'kill "$pid_err" "$pid_out" 2>/dev/null || true' INT TERM EXIT
  wait
}

watch_logs() {
  local out_file="$1"
  local err_file="$2"
  local watch_cmd
  watch_cmd="echo 'ERR: ${err_file}'; tail -n ${LINES} '${err_file}' 2>/dev/null || true; echo; echo 'OUT: ${out_file}'; tail -n ${LINES} '${out_file}' 2>/dev/null || true"
  if command -v watch >/dev/null 2>&1; then
    exec watch -n "$INTERVAL" bash -lc "$watch_cmd"
  fi
  while true; do
    clear
    bash -lc "$watch_cmd"
    sleep "$INTERVAL"
  done
}

submit_job() {
  [[ -f "$SBATCH_SCRIPT" ]] || die "Slurm script not found: $SBATCH_SCRIPT"
  mkdir -p "$LOGS_DIR"

  build_exports
  local sbatch_cmd=(
    sbatch
    --parsable
    --job-name "$JOB_NAME"
    --partition "$PARTITION"
    --qos "$QOS"
    --gres "$GRES"
    --cpus-per-task "$CPUS"
    --mem "$MEMORY"
    --time "$TIME_LIMIT"
    --output "${LOGS_DIR}/%x_%j.out"
    --error "${LOGS_DIR}/%x_%j.err"
    --export "ALL"
  )
  if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
    sbatch_cmd+=("${EXTRA_SBATCH_ARGS[@]}")
  fi
  sbatch_cmd+=("$SBATCH_SCRIPT")

  echo "Submitting AlphaQuantum split-reward sweep"
  echo "Preset: $PRESET"
  echo "Targets: $TARGETS"
  echo "Modes: $MODES"
  echo "Run label: $RUN_LABEL"
  echo "Project root on node: $PROJECT_ROOT"
  echo "Logs dir: $LOGS_DIR"

  if (( DRY_RUN == 1 )); then
    echo "Dry run command:"
    printf '  env '
    printf '%q ' "${EXPORTS[@]}"
    printf '%q ' "${sbatch_cmd[@]}"
    echo
    return 0
  fi

  local submit_out
  if ! submit_out="$(env "${EXPORTS[@]}" "${sbatch_cmd[@]}" 2>&1)"; then
    die "sbatch submission failed: $submit_out"
  fi
  echo "$submit_out"
  JOBID="${submit_out%%;*}"
  [[ "$JOBID" =~ ^[0-9]+$ ]] || die "Could not parse job id from sbatch output: $submit_out"
  echo "Parsed Job ID: $JOBID"
}

attach_logs() {
  local jobid="$1"
  mkdir -p "$LOGS_DIR"
  local out_file
  local err_file
  out_file="$(resolve_log_file "$jobid" out)"
  err_file="$(resolve_log_file "$jobid" err)"
  wait_for_logs "$out_file" "$err_file"
  if (( WATCH_MODE == 1 )); then
    watch_logs "$out_file" "$err_file"
  else
    follow_logs "$out_file" "$err_file"
  fi
}

discover_preset "$@"
apply_preset
parse_args "$@"
LOGS_DIR="$(to_abs_path "$LOGS_DIR")"

if [[ "$MODE" == "attach" ]]; then
  [[ -n "$ATTACH_JOBID" ]] || die "--attach requires a job id"
  JOBID="$ATTACH_JOBID"
else
  submit_job
fi

if (( DRY_RUN == 1 || FOLLOW == 0 )); then
  exit 0
fi

attach_logs "$JOBID"
