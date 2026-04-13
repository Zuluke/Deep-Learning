from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts._manifest import append_command


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_ROOT = PROJECT_ROOT / "external"
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "logs" / "demo"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pythonpath_entries() -> list[str]:
    entries = [str(EXTERNAL_ROOT)]
    current = os.environ.get("PYTHONPATH")
    if current:
        entries.append(current)
    return entries


def _env_for_subprocess(profile: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(_pythonpath_entries())
    env["ATQ_BOOTSTRAP_PROFILE"] = profile
    return env


def _load_demo_modules() -> tuple[object, object, object, object, object]:
    sys.path.insert(0, str(EXTERNAL_ROOT))
    import jax
    import jax.numpy as jnp
    from alphatensor_quantum.src import tensors
    from alphatensor_quantum.src.demo import agent as agent_lib
    from alphatensor_quantum.src.demo import demo_config

    return jax, jnp, tensors, agent_lib, demo_config


def _reference_best_tcount(target_name: str, use_gadgets: bool) -> int | None:
    if target_name == "mod_5_4" and use_gadgets:
        return 2
    return None


def run_control(profile: str, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"control_{profile}_{_timestamp()}.log"
    cmd = [sys.executable, "-m", "alphatensor_quantum.src.demo.run_demo"]
    env = _env_for_subprocess(profile)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()

    append_command(
        {
            "tool": "run_demo_train.py",
            "mode": "control",
            "profile": profile,
            "command": " ".join(cmd),
            "cwd": str(PROJECT_ROOT),
            "log_path": str(log_path),
            "exit_code": return_code,
        }
    )
    return return_code


def run_smoke(profile: str, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"smoke_{profile}_{_timestamp()}.json"

    jax, _, _, agent_lib, demo_config = _load_demo_modules()

    config = demo_config.get_demo_config(use_gadgets=True)
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(2024))

    payload = {
        "profile": profile,
        "devices": [str(device) for device in jax.devices()],
        "backend": jax.default_backend(),
        "target_circuits": [
            target.name.lower() for target in config.env_config.target_circuit_types
        ],
        "batch_size": config.exp_config.batch_size,
        "num_mcts_simulations": config.exp_config.num_mcts_simulations,
        "num_training_steps": config.exp_config.num_training_steps,
        "best_return_shape": list(run_state.game_stats.best_return.shape),
    }
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    append_command(
        {
            "tool": "run_demo_train.py",
            "mode": "smoke",
            "profile": profile,
            "command": f"{sys.executable} scripts/run_demo_train.py --mode smoke --profile {profile} --log-dir {log_dir}",
            "cwd": str(PROJECT_ROOT),
            "log_path": str(log_path),
            "exit_code": 0,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_quick(args: argparse.Namespace) -> int:
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = _timestamp()
    text_log_path = log_dir / f"quick_{args.profile}_{run_id}.log"
    json_log_path = log_dir / f"quick_{args.profile}_{run_id}.json"

    jax, jnp, tensors, agent_lib, demo_config = _load_demo_modules()
    use_gadgets = args.use_gadgets == "on"

    base_config = demo_config.get_demo_config(use_gadgets=use_gadgets)
    exp_config = dataclasses.replace(
        base_config.exp_config,
        batch_size=args.batch_size,
        num_mcts_simulations=args.num_mcts_simulations,
        num_training_steps=args.training_steps,
        eval_frequency_steps=args.eval_frequency,
    )
    env_config = dataclasses.replace(
        base_config.env_config,
        target_circuit_types=[tensors.CircuitType.MOD_5_4],
        use_gadgets=use_gadgets,
    )
    config = dataclasses.replace(
        base_config,
        exp_config=exp_config,
        env_config=env_config,
    )

    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(args.seed))
    history: list[dict[str, object]] = []

    with text_log_path.open("w", encoding="utf-8") as text_log:
        for step in range(
            0, exp_config.num_training_steps, exp_config.eval_frequency_steps
        ):
            time_start = time.time()
            run_state = agent.run_agent_env_interaction(step, run_state)
            time_taken = (time.time() - time_start) / exp_config.eval_frequency_steps
            num_games = run_state.game_stats.num_games
            avg_return = run_state.game_stats.avg_return
            avg_return = jnp.sum(
                jnp.where(
                    num_games > 0,
                    avg_return
                    / (1.0 - exp_config.avg_return_smoothing ** num_games),
                    0.0,
                ),
                axis=0,
            ) / jnp.sum(num_games > 0, axis=0)
            avg_return_value = float(avg_return[0])
            best_return = float(run_state.game_stats.best_return[0])
            best_tcount = None if not math.isfinite(best_return) else int(-best_return)

            headline = (
                f"Step: {step + exp_config.eval_frequency_steps} .. "
                f"Running Average Returns: {avg_return} .. "
                f"Time taken: {time_taken} seconds/step"
            )
            detail = f"  Best T-count for mod_5_4: {best_tcount}"
            print(headline)
            print(detail)
            text_log.write(headline + "\n")
            text_log.write(detail + "\n")
            history.append(
                {
                    "step": step + exp_config.eval_frequency_steps,
                    "avg_return": avg_return_value,
                    "best_tcount": best_tcount,
                    "time_per_step_sec": time_taken,
                }
            )

    target_name = config.env_config.target_circuit_types[0].name.lower()
    best_return_final = float(run_state.game_stats.best_return[0])
    best_tcount_final = (
        None if not math.isfinite(best_return_final) else int(-best_return_final)
    )
    reference_best_tcount = _reference_best_tcount(target_name, use_gadgets)
    summary = {
        "profile": args.profile,
        "mode": "quick",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "target_circuits": [target_name],
        "use_gadgets": use_gadgets,
        "training_steps": args.training_steps,
        "eval_frequency_steps": args.eval_frequency,
        "batch_size": args.batch_size,
        "num_mcts_simulations": args.num_mcts_simulations,
        "seed": args.seed,
        "reference_best_tcount": reference_best_tcount,
        "best_tcount_observed": best_tcount_final,
        "matched_reference": (
            best_tcount_final is not None
            and reference_best_tcount is not None
            and best_tcount_final <= reference_best_tcount
        ),
        "history": history,
        "text_log_path": str(text_log_path),
    }
    json_log_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    append_command(
        {
            "tool": "run_demo_train.py",
            "mode": "quick",
            "profile": args.profile,
            "command": (
                f"{sys.executable} scripts/run_demo_train.py --mode quick "
                f"--profile {args.profile} --use-gadgets {args.use_gadgets} "
                f"--training-steps {args.training_steps} --eval-frequency {args.eval_frequency} "
                f"--batch-size {args.batch_size} --num-mcts-simulations {args.num_mcts_simulations} "
                f"--seed {args.seed} --log-dir {args.log_dir}"
            ),
            "cwd": str(PROJECT_ROOT),
            "json_log_path": str(json_log_path),
            "text_log_path": str(text_log_path),
            "exit_code": 0,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AlphaTensor-Quantum demo wrappers.")
    parser.add_argument("--mode", choices=("control", "smoke", "quick"), required=True)
    parser.add_argument("--profile", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--use-gadgets",
        choices=("on", "off"),
        default="on",
        help="Only used in quick mode.",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=500,
        help="Only used in quick mode.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=50,
        help="Only used in quick mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Only used in quick mode.",
    )
    parser.add_argument(
        "--num-mcts-simulations",
        type=int,
        default=80,
        help="Only used in quick mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Only used in quick mode.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory where logs should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "control":
        return run_control(args.profile, args.log_dir)
    if args.mode == "smoke":
        return run_smoke(args.profile, args.log_dir)
    return run_quick(args)


if __name__ == "__main__":
    raise SystemExit(main())
