#!/usr/bin/env bash
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=atq_mod54_quick
#SBATCH --output=results/logs/slurm/%x_%j.out
#SBATCH --error=results/logs/slurm/%x_%j.err

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
PROJECT_ROOT="$HOME/Deep-Learning/project"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT"
mkdir -p results/logs/slurm results/logs/demo

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected virtualenv interpreter at $PYTHON_BIN" >&2
  exit 1
fi

nvidia-smi

echo "[preflight] starting jax gpu check"
"$PYTHON_BIN" - <<'PY'
import json
import jax

payload = {
    "backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
}
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["backend"] != "gpu":
    raise SystemExit("JAX backend is not gpu.")
PY

echo "[train] starting quick MOD_5_4 run"
"$PYTHON_BIN" scripts/run_demo_train.py \
  --mode quick \
  --profile cuda \
  --use-gadgets on \
  --training-steps 500 \
  --eval-frequency 50 \
  --batch-size 128 \
  --num-mcts-simulations 80 \
  --seed 2024 \
  --log-dir results/logs/demo
