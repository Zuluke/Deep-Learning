#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

MODE="submit"
WATCH_MODE=false
LINES=20
INTERVAL=1
ATTACH_JOBID=""
LOGS_DIR="./logs"
SCRIPT_PATH=""
PROJECT_DIR="${SLURM_PROJECT_DIR:-$PWD}"
JOB_NAME=""
JOB_NAME_SET=0
PARTITION="${SLURM_PARTITION:-}"
TIME_LIMIT="${SLURM_TIME:-}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-}"
MEMORY="${SLURM_MEM:-}"
MAIL_TYPE="${SLURM_MAIL_TYPE:-}"
MAIL_USER="${SLURM_MAIL_USER:-}"
VENV_DIR="${SLURM_VENV_DIR:-.venv}"
PYTHON_VERSION="${SLURM_PYTHON_VERSION:-3.12}"
SYNC_CMD="${SLURM_SYNC_CMD:-}"
USE_UV=1
DRY_RUN=0

PYTHON_ARGS=()
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash slurm/submit.sh --script PATH_TO_SCRIPT.py [options] [-- python_args...]
  bash slurm/submit.sh --attach JOBID [options]

Submit options:
  --script PATH           Python script path to execute on Slurm node
  --project-dir DIR       Working directory on compute node (default: current dir)
  --job-name NAME         Slurm job name (default: script basename)
  --logs-dir DIR          Directory for Slurm .out/.err logs (default: ./logs)
  --partition NAME        Slurm partition
  --time HH:MM:SS         Slurm time limit (if omitted, partition default is used)
  --cpus N                Slurm CPUs per task
  --mem SIZE              Slurm memory, e.g. 16G
  --mail-type TYPES       Slurm mail type, e.g. FAIL,END
  --mail-user EMAIL       Slurm mail user
  --sbatch-arg ARG        Extra sbatch argument (repeatable)
  --watch                 Periodic watch mode instead of tail -F follow mode
  --lines N               Lines for watch mode (default: 20)
  --interval S            Refresh interval for watch mode (default: 1)
  --attach JOBID          Skip submit and attach to logs of an existing job
  --dry-run               Print sbatch command and exit

Runtime options (applied inside the Slurm job):
  --use-uv                Prefer uv flow (default)
  --no-uv                 Force python fallback flow
  --venv-dir PATH         Virtual environment path for uv flow (default: .venv)
  --python-version VER    Python version for uv venv (default: 3.12)
  --sync-cmd CMD          Command run before execution in uv flow (optional)

Notes:
  - Everything after '--' is passed to the target Python script.
  - This script supports an internal mode and should not be called manually with --internal-run.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

to_abs_path() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    echo "$value"
    return
  fi
  echo "$PWD/${value#./}"
}

ensure_logs_dir() {
  mkdir -p "$LOGS_DIR"
}

build_default_job_name() {
  local stem="${SCRIPT_PATH##*/}"
  stem="${stem%.py}"
  stem="${stem//[^a-zA-Z0-9_-]/-}"
  if [[ -z "$stem" ]]; then
    stem="python-job"
  fi
  echo "$stem"
}

wait_for_logs() {
  local out_file="$1"
  local err_file="$2"
  local timeout_secs=180
  local waited=0
  while [[ ! -f "$out_file" || ! -f "$err_file" ]]; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= timeout_secs )); then
      break
    fi
  done
}

follow_mode() {
  local out_file="$1"
  local err_file="$2"
  echo "Following logs:"
  echo "  ERR: $err_file"
  echo "  OUT: $out_file"
  echo
  tail -F -n 200 "$err_file" | sed -u 's/^/[ERR] /' &
  local pid_err=$!
  tail -F -n 200 "$out_file" | sed -u 's/^/[OUT] /' &
  local pid_out=$!
  trap 'kill "$pid_err" "$pid_out" 2>/dev/null || true' INT TERM EXIT
  wait
}

watch_mode() {
  local out_file="$1"
  local err_file="$2"
  echo "Watching logs (every ${INTERVAL}s, last ${LINES} lines):"
  echo "  ERR: $err_file"
  echo "  OUT: $out_file"
  echo

  local watch_cmd="echo '==== ${err_file} ===='; tail -n ${LINES} '${err_file}' 2>/dev/null || true; \

echo; echo '==== ${out_file} ===='; tail -n ${LINES} '${out_file}' 2>/dev/null || true"
  if command -v watch >/dev/null 2>&1; then
    exec watch -n "$INTERVAL" bash -lc "$watch_cmd"
  fi

  while true; do
    clear
    bash -lc "$watch_cmd"
    sleep "$INTERVAL"
  done
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

  if [[ -n "$JOB_NAME" ]]; then
    echo "${LOGS_DIR}/${JOB_NAME}_${jobid}.${extension}"
    return
  fi

  echo "${LOGS_DIR}/job_${jobid}.${extension}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --internal-run)
        MODE="internal"
        shift
        ;;
      --watch)
        WATCH_MODE=true
        shift
        ;;
      --lines)
        LINES="${2:-20}"
        shift 2
        ;;
      --interval)
        INTERVAL="${2:-1}"
        shift 2
        ;;
      --attach)
        ATTACH_JOBID="${2:-}"
        shift 2
        ;;
      --script)
        SCRIPT_PATH="${2:-}"
        shift 2
        ;;
      --project-dir)
        PROJECT_DIR="${2:-$PWD}"
        shift 2
        ;;
      --logs-dir)
        LOGS_DIR="${2:-./logs}"
        shift 2
        ;;
      --job-name)
        JOB_NAME="${2:-}"
        JOB_NAME_SET=1
        shift 2
        ;;
      --partition)
        PARTITION="${2:-}"
        shift 2
        ;;
      --time)
        TIME_LIMIT="${2:-}"
        shift 2
        ;;
      --cpus)
        CPUS_PER_TASK="${2:-}"
        shift 2
        ;;
      --mem)
        MEMORY="${2:-}"
        shift 2
        ;;
      --mail-type)
        MAIL_TYPE="${2:-}"
        shift 2
        ;;
      --mail-user)
        MAIL_USER="${2:-}"
        shift 2
        ;;
      --venv-dir)
        VENV_DIR="${2:-.venv}"
        shift 2
        ;;
      --python-version)
        PYTHON_VERSION="${2:-3.12}"
        shift 2
        ;;
      --sync-cmd)
        SYNC_CMD="${2:-}"
        shift 2
        ;;
      --use-uv)
        USE_UV=1
        shift
        ;;
      --no-uv)
        USE_UV=0
        shift
        ;;
      --sbatch-arg)
        EXTRA_SBATCH_ARGS+=("${2:-}")
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --)
        shift
        PYTHON_ARGS=("$@")
        break
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

run_internal_mode() {
  [[ -n "$SCRIPT_PATH" ]] || die "--script is required in internal mode"
  [[ -d "$PROJECT_DIR" ]] || die "Project directory not found: $PROJECT_DIR"

  if [[ "$SCRIPT_PATH" != /* ]]; then
    SCRIPT_PATH="${PROJECT_DIR}/${SCRIPT_PATH}"
  fi

  [[ -f "$SCRIPT_PATH" ]] || die "Python script not found: $SCRIPT_PATH"

  cd "$PROJECT_DIR"

  echo "==========================================="
  echo "Starting Slurm job runtime"
  echo "Host: $(hostname)"
  echo "User: ${USER:-unknown}"
  echo "Date: $(date)"
  echo "Project dir: $PROJECT_DIR"
  echo "Script: $SCRIPT_PATH"
  echo "Working mode: internal"
  echo "Use uv: $USE_UV"
  echo "==========================================="

  local run_cmd=()
  if (( USE_UV == 1 )) && command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    if [[ -n "$SYNC_CMD" ]]; then
      # User-provided setup command intentionally runs in current shell context.
      eval "$SYNC_CMD"
    fi
    run_cmd=(uv run python "$SCRIPT_PATH")
  else
    if (( USE_UV == 1 )); then
      echo "uv not found in PATH, falling back to python interpreter."
    fi
    if command -v python3 >/dev/null 2>&1; then
      run_cmd=(python3 "$SCRIPT_PATH")
    elif command -v python >/dev/null 2>&1; then
      run_cmd=(python "$SCRIPT_PATH")
    else
      die "No python interpreter found in PATH"
    fi
  fi

  if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
    run_cmd+=("${PYTHON_ARGS[@]}")
  fi

  echo "Executing:"
  printf '  %q ' "${run_cmd[@]}"
  echo
  exec "${run_cmd[@]}"
}

submit_mode() {
  if [[ -n "$ATTACH_JOBID" ]]; then
    echo "Attaching to existing job: $ATTACH_JOBID"
  else
    [[ -n "$SCRIPT_PATH" ]] || die "--script is required when not using --attach"
    SCRIPT_PATH="$(to_abs_path "$SCRIPT_PATH")"
    [[ -f "$SCRIPT_PATH" ]] || die "Python script not found: $SCRIPT_PATH"
  fi

  PROJECT_DIR="$(to_abs_path "$PROJECT_DIR")"
  [[ -d "$PROJECT_DIR" ]] || die "Project directory not found: $PROJECT_DIR"

  LOGS_DIR="$(to_abs_path "$LOGS_DIR")"
  ensure_logs_dir

  if (( JOB_NAME_SET == 0 )) && [[ -n "$SCRIPT_PATH" ]]; then
    JOB_NAME="$(build_default_job_name)"
  fi

  local jobid=""
  if [[ -z "$ATTACH_JOBID" ]]; then
    local sbatch_cmd=(
      sbatch
      --parsable
      --job-name "$JOB_NAME"
      --output "${LOGS_DIR}/%x_%j.out"
      --error "${LOGS_DIR}/%x_%j.err"
    )

    if [[ -n "$PARTITION" ]]; then
      sbatch_cmd+=(--partition "$PARTITION")
    fi
    if [[ -n "$TIME_LIMIT" ]]; then
      sbatch_cmd+=(--time "$TIME_LIMIT")
    fi
    if [[ -n "$CPUS_PER_TASK" ]]; then
      sbatch_cmd+=(--cpus-per-task "$CPUS_PER_TASK")
    fi
    if [[ -n "$MEMORY" ]]; then
      sbatch_cmd+=(--mem "$MEMORY")
    fi
    if [[ -n "$MAIL_TYPE" ]]; then
      sbatch_cmd+=(--mail-type "$MAIL_TYPE")
    fi
    if [[ -n "$MAIL_USER" ]]; then
      sbatch_cmd+=(--mail-user "$MAIL_USER")
    fi

    if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
      sbatch_cmd+=("${EXTRA_SBATCH_ARGS[@]}")
    fi

    sbatch_cmd+=(
      "$SCRIPT_SELF"
      --internal-run
      --script "$SCRIPT_PATH"
      --project-dir "$PROJECT_DIR"
      --venv-dir "$VENV_DIR"
      --python-version "$PYTHON_VERSION"
      --logs-dir "$LOGS_DIR"
    )

    if (( USE_UV == 0 )); then
      sbatch_cmd+=(--no-uv)
    else
      sbatch_cmd+=(--use-uv)
    fi
    if [[ -n "$SYNC_CMD" ]]; then
      sbatch_cmd+=(--sync-cmd "$SYNC_CMD")
    fi

    if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
      sbatch_cmd+=(-- "${PYTHON_ARGS[@]}")
    fi

    echo "Submitting job"
    echo "Script: $SCRIPT_PATH"
    echo "Project dir: $PROJECT_DIR"
    echo "Logs dir: $LOGS_DIR"

    if (( DRY_RUN == 1 )); then
      echo "Dry run command:"
      printf '  %q ' "${sbatch_cmd[@]}"
      echo
      exit 0
    fi

    local submit_out
    if ! submit_out="$("${sbatch_cmd[@]}" 2>&1)"; then
      die "sbatch submission failed: $submit_out"
    fi

    echo "$submit_out"
    jobid="${submit_out%%;*}"
    if [[ ! "$jobid" =~ ^[0-9]+$ ]]; then
      die "Could not parse job ID from sbatch output: $submit_out"
    fi
    echo "Parsed Job ID: $jobid"
  else
    jobid="$ATTACH_JOBID"
  fi

  local out_file
  local err_file
  out_file="$(resolve_log_file "$jobid" out)"
  err_file="$(resolve_log_file "$jobid" err)"

  wait_for_logs "$out_file" "$err_file"

  if [[ "$WATCH_MODE" == true ]]; then
    watch_mode "$out_file" "$err_file"
  fi
  follow_mode "$out_file" "$err_file"
}

parse_args "$@"

if [[ "$MODE" == "internal" ]]; then
  run_internal_mode
fi
submit_mode
