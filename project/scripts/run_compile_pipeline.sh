#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/run_compile_pipeline.sh --input <qasm> --output-dir <dir> --zx-preopt on|off
EOF
}

input=""
output_dir=""
zx_preopt=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      input="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --zx-preopt)
      zx_preopt="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$input" || -z "$output_dir" || ( "$zx_preopt" != "on" && "$zx_preopt" != "off" ) ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$PROJECT_ROOT/external/circuit-to-tensor/target/release/circuit-to-tensor"

if [[ ! -x "$BIN" ]]; then
  echo "Missing compiled binary at $BIN. Run ./scripts/setup_env.sh first." >&2
  exit 1
fi

mkdir -p "$output_dir"

cmd=("$BIN" compile -e tensor,matrix,circuit-qasm,log)
if [[ "$zx_preopt" == "on" ]]; then
  cmd+=(-z)
fi
cmd+=("$output_dir" "$input")
"${cmd[@]}"

PROJECT_ROOT="$PROJECT_ROOT" OUTPUT_DIR="$output_dir" INPUT_QASM="$input" ZX_PREOPT="$zx_preopt" PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python3 - <<'PY'
import os

from scripts._manifest import append_command

project_root = os.environ["PROJECT_ROOT"]
output_dir = os.environ["OUTPUT_DIR"]
input_qasm = os.environ["INPUT_QASM"]
zx_preopt = os.environ["ZX_PREOPT"]

append_command(
    {
        "tool": "run_compile_pipeline.sh",
        "command": f"./scripts/run_compile_pipeline.sh --input {input_qasm} --output-dir {output_dir} --zx-preopt {zx_preopt}",
        "cwd": project_root,
        "input_qasm": input_qasm,
        "output_dir": output_dir,
        "zx_preopt": zx_preopt,
        "exit_code": 0,
    }
)
PY
