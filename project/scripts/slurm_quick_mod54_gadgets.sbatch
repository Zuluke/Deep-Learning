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

cd "$PROJECT_ROOT"
mkdir -p results/logs/slurm results/logs/demo

nvidia-smi

uv run --project . python - <<'PY'
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

uv run --project . python scripts/run_demo_train.py \
  --mode quick \
  --profile cuda \
  --use-gadgets on \
  --training-steps 500 \
  --eval-frequency 50 \
  --batch-size 128 \
  --num-mcts-simulations 80 \
  --seed 2024 \
  --log-dir results/logs/demo
