#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_env.sh --profile cpu|cuda
EOF
}

profile=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile="${2:-}"
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

if [[ "$profile" != "cpu" && "$profile" != "cuda" ]]; then
  echo "Expected --profile cpu|cuda" >&2
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUSTUP_SH_URL="https://sh.rustup.rs"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"

case "$profile" in
  cpu)
    PROFILE_GROUP="demo-cpu"
    ;;
  cuda)
    PROFILE_GROUP="demo-cuda12"
    ;;
esac

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "uv not found and curl is unavailable for installation." >&2
    exit 1
  fi
  curl -LsSf "$UV_INSTALLER_URL" | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is still unavailable after installation." >&2
  exit 1
fi

MANAGED_PYTHON="$(find "$HOME/.local/share/uv/python" -maxdepth 3 -type f -path '*cpython-3.11*/bin/python3.11' 2>/dev/null | head -n 1 || true)"
if [[ -z "$MANAGED_PYTHON" ]]; then
  uv python install 3.11
  MANAGED_PYTHON="3.11"
fi

uv sync --project "$PROJECT_ROOT" --python "$MANAGED_PYTHON" --no-default-groups --group dev --group "$PROFILE_GROUP"

if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

if ! command -v cargo >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "cargo not found and curl is unavailable for rustup installation." >&2
    exit 1
  fi
  curl --proto '=https' --tlsv1.2 -sSf "$RUSTUP_SH_URL" | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is still unavailable after rustup installation." >&2
  exit 1
fi

cargo build --release --manifest-path "$PROJECT_ROOT/external/circuit-to-tensor/Cargo.toml"

PYTHONPATH="$PROJECT_ROOT/external${PYTHONPATH:+:$PYTHONPATH}" \
  uv run --project "$PROJECT_ROOT" python -c "import alphatensor_quantum.src.demo.run_demo as run_demo; print(run_demo.__name__)"

PROJECT_ROOT="$PROJECT_ROOT" BOOTSTRAP_PROFILE="$profile" PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  uv run --project "$PROJECT_ROOT" python - <<'PY'
import os
import platform
import subprocess
import sys

from scripts._manifest import append_command, update_environment

project_root = os.environ["PROJECT_ROOT"]
profile = os.environ["BOOTSTRAP_PROFILE"]

def capture(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unavailable"

environment = {
    "profile": profile,
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "uv_version": capture(["uv", "--version"]),
    "rustc_version": capture(["rustc", "--version"]),
    "cargo_version": capture(["cargo", "--version"]),
    "feynver_version": capture(["feynver", "--help"]),
    "project_root": project_root,
}
update_environment(environment)

append_command(
    {
        "tool": "setup_env.sh",
        "profile": profile,
        "command": f"./scripts/setup_env.sh --profile {profile}",
        "cwd": project_root,
        "exit_code": 0,
    }
)
PY
