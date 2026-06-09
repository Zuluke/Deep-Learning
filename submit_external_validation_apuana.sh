#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOB_NAME="${SLURM_JOB_NAME:-alphaq_external_validation}"
PARTITION="${SLURM_PARTITION:-short-complex}"
QOS="${SLURM_QOS:-complex}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
MEMORY="${SLURM_MEM:-64G}"
SLURM_TIME_LIMIT="${SLURM_TIME:-06:00:00}"
LOGS_DIR="${SLURM_LOGS_DIR:-${SCRIPT_DIR}/project/results/logs/slurm}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Deep-Learning/project}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

TARGETS="nc_tof_4,barenco_tof_4,vbe_adder_3"
TIME_LIMIT_SEC="900"
BEAM_WIDTHS="4,16"
OBJECTIVE_VARIANTS=""
OUTPUT_SUFFIX=""
FOLLOW=0
DRY_RUN=0
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  ./submit_external_validation_apuana.sh [options]

Submits the AlphaQ external-validation pipeline:
  1. objective decompositions with explicit per-objective failure rows;
  2. objective-by-beam materialization grid;
  3. combined internal+external grid;
  4. guarded selector transfer report.

Options:
  --targets LIST          Comma-separated targets (default: nc_tof_4,barenco_tof_4,vbe_adder_3)
  --time-limit-sec N      Per-MILP objective time limit in seconds (default: 900)
  --beam-widths LIST      Beam widths for materialization (default: 4,16)
  --objective-variants LIST
                          Objective variants to run, e.g. factor_count_pair_cap
  --output-suffix NAME    Write all pipeline outputs with this suffix, e.g. night_long
  --job-name NAME
  --partition NAME
  --qos NAME
  --cpus N
  --mem SIZE
  --time HH:MM:SS         Slurm wall time (default: 06:00:00)
  --logs-dir DIR
  --project-root DIR      Project root on Apuana (default: $HOME/Deep-Learning/project)
  --python-bin PATH       Python binary on Apuana (default: $PROJECT_ROOT/.venv/bin/python)
  --force                 Recompute existing outputs
  --follow                Tail logs after submission
  --dry-run               Print generated sbatch script and command, do not submit
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --targets) TARGETS="${2:-}"; shift 2 ;;
    --time-limit-sec) TIME_LIMIT_SEC="${2:-}"; shift 2 ;;
    --beam-widths) BEAM_WIDTHS="${2:-}"; shift 2 ;;
    --objective-variants) OBJECTIVE_VARIANTS="${2:-}"; shift 2 ;;
    --output-suffix) OUTPUT_SUFFIX="${2:-}"; shift 2 ;;
    --job-name) JOB_NAME="${2:-}"; shift 2 ;;
    --partition) PARTITION="${2:-}"; shift 2 ;;
    --qos) QOS="${2:-}"; shift 2 ;;
    --cpus) CPUS="${2:-}"; shift 2 ;;
    --mem) MEMORY="${2:-}"; shift 2 ;;
    --time) SLURM_TIME_LIMIT="${2:-}"; shift 2 ;;
    --logs-dir) LOGS_DIR="${2:-}"; shift 2 ;;
    --project-root) PROJECT_ROOT="${2:-}"; shift 2 ;;
    --python-bin) PYTHON_BIN="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --follow) FOLLOW=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
esac
done

if [[ -n "$OUTPUT_SUFFIX" && ! "$OUTPUT_SUFFIX" =~ ^[A-Za-z0-9_-]+$ ]]; then
  die "--output-suffix may contain only letters, numbers, underscores, and dashes."
fi

mkdir -p "$LOGS_DIR"
SBATCH_PATH="${LOGS_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).sbatch"

FORCE_ARG=""
if (( FORCE == 1 )); then
  FORCE_ARG="--force"
fi

cat > "$SBATCH_PATH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${SLURM_TIME_LIMIT}
#SBATCH --output=${LOGS_DIR}/%x_%j.out
#SBATCH --error=${LOGS_DIR}/%x_%j.err

set -euo pipefail

cd $(printf "%q" "$PROJECT_ROOT")
export PYTHONPATH=".:external:\${PYTHONPATH:-}"

PYTHON_BIN=$(printf "%q" "$PYTHON_BIN")
if [[ ! -x "\$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi

echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo "Project: \$(pwd)"
echo "Targets: ${TARGETS}"
echo "Per-objective time limit: ${TIME_LIMIT_SEC}s"
echo "Objective variants: ${OBJECTIVE_VARIANTS:-all}"
echo "Output suffix: ${OUTPUT_SUFFIX:-standard}"

PIPELINE_CMD=("\$PYTHON_BIN" scripts/run_alphaq_external_validation_pipeline.py \\
  --targets $(printf "%q" "$TARGETS") \\
  --time-limit-sec $(printf "%q" "$TIME_LIMIT_SEC") \\
  --beam-widths $(printf "%q" "$BEAM_WIDTHS"))
if [[ -n "$(printf "%s" "$OBJECTIVE_VARIANTS")" ]]; then
  PIPELINE_CMD+=(--objective-variants "$(printf "%s" "$OBJECTIVE_VARIANTS")")
fi
if [[ -n "$(printf "%s" "$OUTPUT_SUFFIX")" ]]; then
  PIPELINE_CMD+=(\\
    --output-root "results/alphaq_decomposition_objective_external_validation_$(printf "%s" "$OUTPUT_SUFFIX")" \\
    --output-csv "results/csv/alphaq_decomposition_objective_external_validation_$(printf "%s" "$OUTPUT_SUFFIX").csv" \\
    --report-path "results/reports/alphaq_decomposition_objective_external_validation_$(printf "%s" "$OUTPUT_SUFFIX").md" \\
    --figure-path "results/figures/alphaq_decomposition_objective_external_validation_$(printf "%s" "$OUTPUT_SUFFIX").png" \\
    --beam-output-root "results/alphaq_objective_beam_policy_external_validation_$(printf "%s" "$OUTPUT_SUFFIX")" \\
    --beam-grid-csv "results/csv/alphaq_objective_beam_policy_external_validation_$(printf "%s" "$OUTPUT_SUFFIX")_grid.csv" \\
    --beam-policy-csv "results/csv/alphaq_objective_beam_policy_external_validation_$(printf "%s" "$OUTPUT_SUFFIX")_summary.csv" \\
    --beam-report-path "results/reports/alphaq_objective_beam_policy_external_validation_$(printf "%s" "$OUTPUT_SUFFIX").md" \\
    --beam-figure-path "results/figures/alphaq_objective_beam_policy_external_validation_$(printf "%s" "$OUTPUT_SUFFIX").png" \\
    --combined-grid-csv "results/csv/alphaq_objective_beam_policy_grid_plus_external_validation_$(printf "%s" "$OUTPUT_SUFFIX").csv" \\
    --transfer-summary-csv "results/csv/alphaq_external_selector_transfer_$(printf "%s" "$OUTPUT_SUFFIX")_summary.csv" \\
    --transfer-detail-csv "results/csv/alphaq_external_selector_transfer_$(printf "%s" "$OUTPUT_SUFFIX")_details.csv" \\
    --transfer-report-path "results/reports/alphaq_external_selector_transfer_$(printf "%s" "$OUTPUT_SUFFIX").md" \\
    --status-csv "results/csv/alphaq_external_validation_status_$(printf "%s" "$OUTPUT_SUFFIX").csv" \\
    --status-report-path "results/reports/alphaq_external_validation_status_$(printf "%s" "$OUTPUT_SUFFIX").md" \\
    --paper-zx-csv "results/csv/alphaq_external_validation_paper_zx_audit_$(printf "%s" "$OUTPUT_SUFFIX").csv" \\
    --paper-zx-report-path "results/reports/alphaq_external_validation_paper_zx_audit_$(printf "%s" "$OUTPUT_SUFFIX").md")
fi
if [[ -n "${FORCE_ARG}" ]]; then
  PIPELINE_CMD+=("${FORCE_ARG}")
fi
"\${PIPELINE_CMD[@]}"
EOF

if (( DRY_RUN == 1 )); then
  echo "Generated sbatch script: $SBATCH_PATH"
  echo
  sed -n '1,220p' "$SBATCH_PATH"
  echo
  echo "Dry-run command:"
  printf 'sbatch %q\n' "$SBATCH_PATH"
  exit 0
fi

SUBMIT_OUT="$(sbatch --parsable "$SBATCH_PATH")"
echo "$SUBMIT_OUT"
JOBID="${SUBMIT_OUT%%;*}"
echo "Parsed Job ID: $JOBID"
echo "Logs:"
echo "  ${LOGS_DIR}/${JOB_NAME}_${JOBID}.out"
echo "  ${LOGS_DIR}/${JOB_NAME}_${JOBID}.err"

if (( FOLLOW == 1 )); then
  tail -F "${LOGS_DIR}/${JOB_NAME}_${JOBID}.err" "${LOGS_DIR}/${JOB_NAME}_${JOBID}.out"
fi
