#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/run_resynth.sh --decomposition <npy> --mapping <txt> --original <npy> --output-dir <dir> --gadgets on|off
EOF
}

decomposition=""
mapping=""
original=""
output_dir=""
gadgets=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --decomposition)
      decomposition="${2:-}"
      shift 2
      ;;
    --mapping)
      mapping="${2:-}"
      shift 2
      ;;
    --original)
      original="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --gadgets)
      gadgets="${2:-}"
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

if [[ -z "$decomposition" || -z "$mapping" || -z "$original" || -z "$output_dir" || ( "$gadgets" != "on" && "$gadgets" != "off" ) ]]; then
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

cmd=("$BIN" resynth -e circuit-qasm,log -m "$mapping" -O "$original")
if [[ "$gadgets" == "on" ]]; then
  cmd+=(-g)
fi
cmd+=("$output_dir" "$decomposition")
"${cmd[@]}"

PROJECT_ROOT="$PROJECT_ROOT" OUTPUT_DIR="$output_dir" DECOMPOSITION="$decomposition" MAPPING="$mapping" ORIGINAL="$original" GADGETS="$gadgets" PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python3 - <<'PY'
import os

from scripts._manifest import append_command

project_root = os.environ["PROJECT_ROOT"]
output_dir = os.environ["OUTPUT_DIR"]
decomposition = os.environ["DECOMPOSITION"]
mapping = os.environ["MAPPING"]
original = os.environ["ORIGINAL"]
gadgets = os.environ["GADGETS"]

append_command(
    {
        "tool": "run_resynth.sh",
        "command": (
            "./scripts/run_resynth.sh "
            f"--decomposition {decomposition} --mapping {mapping} --original {original} "
            f"--output-dir {output_dir} --gadgets {gadgets}"
        ),
        "cwd": project_root,
        "decomposition": decomposition,
        "mapping": mapping,
        "original": original,
        "output_dir": output_dir,
        "gadgets": gadgets,
        "exit_code": 0,
    }
)
PY
