from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._manifest import append_command
EXTERNAL_ROOT = PROJECT_ROOT / "external"
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "logs" / "demo"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "results" / "alphaq_split_reward"


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


def _target_circuit_types(tensors_module: object, preset: str, default_targets):
    if preset == "demo":
        return list(default_targets)
    if preset == "mod_5_4":
        return [tensors_module.CircuitType.MOD_5_4]
    if preset == "gf_2pow2_mult":
        return [tensors_module.CircuitType.GF_2POW2_MULT]
    if preset == "hamming_weight_n4":
        return [tensors_module.CircuitType.HAMMING_WEIGHT_N4]
    if preset == "hamming_weight_n5":
        return [tensors_module.CircuitType.HAMMING_WEIGHT_N5]
    if preset == "split4":
        return [
            tensors_module.CircuitType.MOD_5_4,
            tensors_module.CircuitType.GF_2POW2_MULT,
            tensors_module.CircuitType.HAMMING_WEIGHT_N4,
            tensors_module.CircuitType.HAMMING_WEIGHT_N5,
        ]
    if preset == "qft_4":
        raise ValueError(
            "qft_4 has tensor size 43 and requires a restricted action "
            "dictionary; it is evaluation-only in this sprint."
        )
    raise ValueError(f"Unknown target preset: {preset}")


PARTITION_PRESETS = (
    "balanced",
    "shifted-contiguous",
    "tensor-spectral",
    "ensemble",
)


def _max_target_size(tensors_module: object, target_circuit_types) -> int:
    return max(
        tensors_module.get_signature_tensor(target).shape[0]
        for target in target_circuit_types
    )


def _pad_partition(blocks: list[int], max_size: int) -> list[int]:
    if len(blocks) > max_size:
        raise ValueError(
            f"Partition has length {len(blocks)} but max size is {max_size}."
        )
    return blocks + [1] * (max_size - len(blocks))


def _contiguous_partition(size: int, cut: int, max_size: int) -> list[int]:
    cut = min(max(1, cut), size - 1 if size > 1 else 1)
    return _pad_partition(
        [0 if index < cut else 1 for index in range(size)],
        max_size,
    )


def _spectral_partition(tensor: np.ndarray, max_size: int) -> list[int]:
    """Returns a deterministic tensor-graph bipartition."""
    size = int(tensor.shape[0])
    if size <= 1:
        return _pad_partition([0] * size, max_size)
    graph = np.zeros((size, size), dtype=float)
    for i, j, k in np.argwhere(tensor != 0):
        for a, b in ((i, j), (i, k), (j, k)):
            if a == b:
                continue
            graph[a, b] += 1.0
            graph[b, a] += 1.0
    if not np.any(graph):
        return _contiguous_partition(size, (size + 1) // 2, max_size)
    degree = np.diag(np.sum(graph, axis=1))
    laplacian = degree - graph
    _eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, 1] if size > 1 else eigenvectors[:, 0]
    order = np.argsort(fiedler, kind="mergesort")
    blocks = [1] * size
    for index in order[: max(1, size // 2)]:
        blocks[int(index)] = 0
    if len(set(blocks)) == 1:
        return _contiguous_partition(size, (size + 1) // 2, max_size)
    return _pad_partition(blocks, max_size)


def _partition_blocks_and_weights(
    tensors_module: object,
    target_circuit_types,
    preset: str,
) -> tuple[list[list[list[int]]], list[list[float]]]:
    if preset not in PARTITION_PRESETS:
        raise ValueError(
            f"Unknown partition preset: {preset}. Expected one of {PARTITION_PRESETS}."
        )
    max_size = _max_target_size(tensors_module, target_circuit_types)
    blocks_by_target: list[list[list[int]]] = []
    for target in target_circuit_types:
        tensor = np.asarray(tensors_module.get_signature_tensor(target), dtype=np.int32)
        size = int(tensor.shape[0])
        half = max(1, (size + 1) // 2)
        third = max(1, (size + 2) // 3)
        two_thirds = min(size - 1, max(1, (2 * size + 2) // 3))
        balanced = _contiguous_partition(size, half, max_size)
        shifted = [
            _contiguous_partition(size, third, max_size),
            balanced,
            _contiguous_partition(size, two_thirds, max_size),
        ]
        spectral = _spectral_partition(tensor, max_size)
        if preset == "balanced":
            target_blocks = [balanced]
        elif preset == "shifted-contiguous":
            target_blocks = shifted
        elif preset == "tensor-spectral":
            target_blocks = [spectral]
        else:
            target_blocks = [shifted[0], shifted[1], shifted[2], spectral]
        blocks_by_target.append(target_blocks)
    weights_by_target = [
        [1.0 / len(target_blocks)] * len(target_blocks)
        for target_blocks in blocks_by_target
    ]
    return blocks_by_target, weights_by_target


def _parse_baseline_t_costs(value: str | None, target_names: list[str]) -> list[float] | None:
    if not value:
        return None
    if "," in value or "=" in value:
        by_name: dict[str, float] = {}
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            name, raw_cost = item.split("=", maxsplit=1)
            by_name[name.strip()] = float(raw_cost)
        missing = [name for name in target_names if name not in by_name]
        if missing:
            raise ValueError(
                "Missing baseline T-cost entries for: " + ", ".join(missing)
            )
        return [by_name[name] for name in target_names]
    costs = [float(item) for item in value.split(":") if item]
    if len(costs) != len(target_names):
        raise ValueError(
            f"Expected {len(target_names)} baseline T-cost values, got {len(costs)}."
        )
    return costs


def _configured_demo_config(
    args: argparse.Namespace,
    *,
    use_gadgets: bool,
    baseline_t_costs: list[float] | None = None,
):
    _, _, tensors_module, _, demo_config = _load_demo_modules()
    base_config = demo_config.get_demo_config(use_gadgets=use_gadgets)
    target_circuit_types = _target_circuit_types(
        tensors_module,
        args.target_preset,
        base_config.env_config.target_circuit_types,
    )
    target_names = [target.name.lower() for target in target_circuit_types]
    split_reward = demo_config.config_lib.SplitRewardParams()
    change_of_basis = base_config.env_config.change_of_basis
    if args.force_canonical_basis:
        change_of_basis = dataclasses.replace(
            change_of_basis, prob_canonical_basis=1.0
        )
    needs_split_partitions = (
        args.split_reward_mode != "none" or args.action_prior == "split"
    )
    partition_blocks = None
    partition_weights = None
    if needs_split_partitions:
        partition_blocks, partition_weights = _partition_blocks_and_weights(
            tensors_module,
            target_circuit_types,
            args.partition_preset,
        )
    if needs_split_partitions:
        split_reward = demo_config.config_lib.SplitRewardParams(
            mode=args.split_reward_mode,
            lambda_drop=args.lambda_drop,
            lambda_auc=args.lambda_auc,
            lambda_mass=args.lambda_mass,
            lambda_residual=args.lambda_residual,
            lambda_frontier=args.lambda_frontier,
            lambda_budget=args.lambda_budget,
            drop_clip=args.drop_clip,
            canonical_basis_only=args.canonical_only,
            partition_blocks_by_target=partition_blocks,
            partition_weights_by_target=partition_weights,
            baseline_t_costs=baseline_t_costs
            if baseline_t_costs is not None
            else _parse_baseline_t_costs(args.baseline_t_costs, target_names),
            t_guard_delta=args.t_guard_delta,
            interim_budget_slack=args.interim_budget_slack,
            terminal_tiebreak_clip=args.terminal_tiebreak_clip,
        )
    env_config = dataclasses.replace(
        base_config.env_config,
        target_circuit_types=target_circuit_types,
        use_gadgets=use_gadgets,
        change_of_basis=change_of_basis,
        split_reward=split_reward,
    )
    if args.max_num_moves > 0:
        env_config = dataclasses.replace(
            env_config,
            max_num_moves=args.max_num_moves,
        )
    if args.num_past_factors_to_observe > 0:
        env_config = dataclasses.replace(
            env_config,
            num_past_factors_to_observe=args.num_past_factors_to_observe,
        )
    exp_config = dataclasses.replace(
        base_config.exp_config,
        action_dictionary=args.action_dictionary,
        max_action_weight=args.max_action_weight,
        tensor_overlap_max_weight=args.tensor_overlap_max_weight,
        tensor_overlap_max_actions_per_target=(
            args.tensor_overlap_max_actions_per_target
        ),
        gadget_closure_max_weight=args.gadget_closure_max_weight,
        mask_padded_actions=args.mask_padded_actions,
        mask_repeated_actions=args.mask_repeated_actions,
        action_prior=args.action_prior,
        action_prior_beta=args.action_prior_beta,
        action_prior_residual_weight=args.action_prior_residual_weight,
        action_prior_mixed_drop_weight=args.action_prior_mixed_drop_weight,
        action_prior_mixed_mass_weight=args.action_prior_mixed_mass_weight,
        action_prior_hamming_weight=args.action_prior_hamming_weight,
        action_prior_gadget_bonus=args.action_prior_gadget_bonus,
        action_prior_standardize=args.action_prior_standardize,
        action_prior_top_k=args.action_prior_top_k,
        action_prior_canonical_only=args.action_prior_canonical_only,
        frontier_replay_fraction=args.frontier_replay_fraction,
        frontier_replay_min_moves=args.frontier_replay_min_moves,
        frontier_replay_min_residual_drop=(
            args.frontier_replay_min_residual_drop
        ),
    )
    return dataclasses.replace(
        base_config,
        exp_config=exp_config,
        env_config=env_config,
    )


def _smoothed_stat(jnp, values, num_games, smoothing: float):
    active = num_games > 0
    corrected = jnp.where(
        active,
        values / (1.0 - smoothing ** jnp.maximum(num_games, 1)),
        0.0,
    )
    active_count = jnp.maximum(jnp.sum(active, axis=0), 1)
    return jnp.sum(corrected, axis=0) / active_count


def _finite_or_none(value: float) -> float | None:
    return None if not math.isfinite(value) else value


def _finite_list(values) -> list[float | None]:
    return [_finite_or_none(float(value)) for value in values]


def _int_list(values) -> list[int]:
    return [int(value) for value in values]


def _valid_factor_prefix(factors: np.ndarray, num_moves: int) -> np.ndarray:
    if num_moves <= 0:
        return factors[:0]
    return factors[-num_moves:]


def _write_factor_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "candidate_kind",
        "status",
        "num_moves",
        "return",
        "effective_t_cost",
        "residual_weight",
        "factor_path",
        "change_of_basis_path",
        "is_canonical_basis",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_candidate_factors(
    config,
    run_state,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_names = [
        target.name.lower() for target in config.env_config.target_circuit_types
    ]
    stats = run_state.game_stats
    best_returns = _finite_list(np.asarray(stats.best_return))
    best_return_effective_t_costs = _finite_list(
        np.asarray(stats.best_return_effective_t_cost)
    )
    best_return_num_moves = _int_list(np.asarray(stats.best_return_num_moves))
    best_return_residual_weight = _finite_list(
        np.asarray(stats.best_return_residual_weight)
    )
    best_effective_t_costs = _finite_list(np.asarray(stats.best_effective_t_cost))
    best_solved_num_moves = _int_list(np.asarray(stats.best_solved_num_moves))
    best_return_factors = np.asarray(stats.best_return_factors, dtype=np.int32)
    best_solved_factors = np.asarray(stats.best_solved_factors, dtype=np.int32)
    best_return_change_of_basis = np.asarray(
        stats.best_return_change_of_basis, dtype=np.int32
    )
    best_solved_change_of_basis = np.asarray(
        stats.best_solved_change_of_basis, dtype=np.int32
    )
    best_frontier_residual_weight = _finite_list(
        np.asarray(stats.best_frontier_residual_weight)
    )
    best_frontier_effective_t_cost = _finite_list(
        np.asarray(stats.best_frontier_effective_t_cost)
    )
    best_frontier_num_moves = _int_list(np.asarray(stats.best_frontier_num_moves))
    best_frontier_factors = np.asarray(stats.best_frontier_factors, dtype=np.int32)
    best_frontier_change_of_basis = np.asarray(
        stats.best_frontier_change_of_basis, dtype=np.int32
    )
    identity = np.eye(config.env_config.max_tensor_size, dtype=np.int32)

    rows: list[dict[str, object]] = []
    for index, target_name in enumerate(target_names):
        return_status = "missing" if best_returns[index] is None else "terminal"
        return_path = ""
        return_cob_path = ""
        return_is_canonical = ""
        if return_status == "terminal":
            return_factors = _valid_factor_prefix(
                best_return_factors[index], best_return_num_moves[index]
            )
            return_path = str(output_dir / f"{target_name}.best_return.npy")
            np.save(return_path, return_factors)
            return_cob_path = str(
                output_dir / f"{target_name}.best_return.change_of_basis.npy"
            )
            np.save(return_cob_path, best_return_change_of_basis[index])
            return_is_canonical = bool(
                np.array_equal(best_return_change_of_basis[index], identity)
            )
        rows.append(
            {
                "target": target_name,
                "candidate_kind": "best_return",
                "status": return_status,
                "num_moves": best_return_num_moves[index],
                "return": best_returns[index],
                "effective_t_cost": best_return_effective_t_costs[index],
                "residual_weight": best_return_residual_weight[index],
                "factor_path": return_path,
                "change_of_basis_path": return_cob_path,
                "is_canonical_basis": return_is_canonical,
            }
        )

        solved_status = (
            "missing" if best_effective_t_costs[index] is None else "solved"
        )
        solved_path = ""
        solved_cob_path = ""
        solved_is_canonical = ""
        if solved_status == "solved":
            solved_factors = _valid_factor_prefix(
                best_solved_factors[index], best_solved_num_moves[index]
            )
            solved_path = str(output_dir / f"{target_name}.best_solved.npy")
            np.save(solved_path, solved_factors)
            solved_cob_path = str(
                output_dir / f"{target_name}.best_solved.change_of_basis.npy"
            )
            np.save(solved_cob_path, best_solved_change_of_basis[index])
            solved_is_canonical = bool(
                np.array_equal(best_solved_change_of_basis[index], identity)
            )
        rows.append(
            {
                "target": target_name,
                "candidate_kind": "best_solved",
                "status": solved_status,
                "num_moves": best_solved_num_moves[index],
                "return": "",
                "effective_t_cost": best_effective_t_costs[index],
                "residual_weight": 0.0 if solved_status == "solved" else "",
                "factor_path": solved_path,
                "change_of_basis_path": solved_cob_path,
                "is_canonical_basis": solved_is_canonical,
            }
        )

        frontier_status = (
            "missing"
            if best_frontier_residual_weight[index] is None
            else (
                "solved"
                if best_frontier_residual_weight[index] == 0.0
                else "partial"
            )
        )
        frontier_path = ""
        frontier_cob_path = ""
        frontier_is_canonical = ""
        if frontier_status != "missing":
            frontier_factors = _valid_factor_prefix(
                best_frontier_factors[index], best_frontier_num_moves[index]
            )
            frontier_path = str(output_dir / f"{target_name}.best_frontier.npy")
            np.save(frontier_path, frontier_factors)
            frontier_cob_path = str(
                output_dir / f"{target_name}.best_frontier.change_of_basis.npy"
            )
            np.save(frontier_cob_path, best_frontier_change_of_basis[index])
            frontier_is_canonical = bool(
                np.array_equal(best_frontier_change_of_basis[index], identity)
            )
        rows.append(
            {
                "target": target_name,
                "candidate_kind": "best_frontier",
                "status": frontier_status,
                "num_moves": best_frontier_num_moves[index],
                "return": "",
                "effective_t_cost": best_frontier_effective_t_cost[index],
                "residual_weight": best_frontier_residual_weight[index],
                "factor_path": frontier_path,
                "change_of_basis_path": frontier_cob_path,
                "is_canonical_basis": frontier_is_canonical,
            }
        )

    manifest_path = output_dir / "candidate_factors_manifest.csv"
    _write_factor_manifest(rows, manifest_path)
    return manifest_path


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


def run_smoke(args: argparse.Namespace) -> int:
    profile = args.profile
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"smoke_{profile}_{_timestamp()}.json"

    jax, _, _, agent_lib, demo_config = _load_demo_modules()

    config = _configured_demo_config(
        args, use_gadgets=args.use_gadgets == "on"
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(2024))

    payload = {
        "profile": profile,
        "mode": "smoke",
        "devices": [str(device) for device in jax.devices()],
        "backend": jax.default_backend(),
        "target_circuits": [
            target.name.lower() for target in config.env_config.target_circuit_types
        ],
        "target_preset": args.target_preset,
        "split_reward_mode": args.split_reward_mode,
        "partition_preset": args.partition_preset,
        "max_num_moves": config.env_config.max_num_moves,
        "num_past_factors_to_observe": (
            config.env_config.num_past_factors_to_observe
        ),
        "batch_size": config.exp_config.batch_size,
        "num_mcts_simulations": config.exp_config.num_mcts_simulations,
        "num_training_steps": config.exp_config.num_training_steps,
        "action_dictionary": config.exp_config.action_dictionary,
        "max_action_weight": config.exp_config.max_action_weight,
        "tensor_overlap_max_weight": config.exp_config.tensor_overlap_max_weight,
        "tensor_overlap_max_actions_per_target": (
            config.exp_config.tensor_overlap_max_actions_per_target
        ),
        "gadget_closure_max_weight": config.exp_config.gadget_closure_max_weight,
        "mask_padded_actions": config.exp_config.mask_padded_actions,
        "mask_repeated_actions": config.exp_config.mask_repeated_actions,
        "action_prior": config.exp_config.action_prior,
        "action_prior_beta": config.exp_config.action_prior_beta,
        "action_prior_residual_weight": config.exp_config.action_prior_residual_weight,
        "action_prior_mixed_drop_weight": (
            config.exp_config.action_prior_mixed_drop_weight
        ),
        "action_prior_mixed_mass_weight": (
            config.exp_config.action_prior_mixed_mass_weight
        ),
        "action_prior_hamming_weight": config.exp_config.action_prior_hamming_weight,
        "action_prior_gadget_bonus": config.exp_config.action_prior_gadget_bonus,
        "action_prior_standardize": config.exp_config.action_prior_standardize,
        "action_prior_top_k": config.exp_config.action_prior_top_k,
        "action_prior_canonical_only": config.exp_config.action_prior_canonical_only,
        "frontier_replay_fraction": config.exp_config.frontier_replay_fraction,
        "frontier_replay_min_moves": config.exp_config.frontier_replay_min_moves,
        "frontier_replay_min_residual_drop": (
            config.exp_config.frontier_replay_min_residual_drop
        ),
        "num_actions": int(agent._num_actions),  # pylint: disable=protected-access
        "best_return_shape": list(run_state.game_stats.best_return.shape),
        "best_effective_t_cost_shape": list(
            run_state.game_stats.best_effective_t_cost.shape
        ),
        "best_frontier_residual_weight_shape": list(
            run_state.game_stats.best_frontier_residual_weight.shape
        ),
    }
    log_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    append_command(
        {
            "tool": "run_demo_train.py",
            "mode": "smoke",
            "profile": profile,
            "command": " ".join(sys.argv),
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

    jax, jnp, _, agent_lib, _ = _load_demo_modules()
    use_gadgets = args.use_gadgets == "on"

    base_config = _configured_demo_config(args, use_gadgets=use_gadgets)
    exp_config = dataclasses.replace(
        base_config.exp_config,
        batch_size=args.batch_size,
        num_mcts_simulations=args.num_mcts_simulations,
        num_training_steps=args.training_steps,
        eval_frequency_steps=args.eval_frequency,
    )
    config = dataclasses.replace(
        base_config,
        exp_config=exp_config,
    )

    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(args.seed))
    history: list[dict[str, object]] = []
    target_names = [
        target.name.lower() for target in config.env_config.target_circuit_types
    ]

    with text_log_path.open("w", encoding="utf-8") as text_log:
        for step in range(
            0, exp_config.num_training_steps, exp_config.eval_frequency_steps
        ):
            time_start = time.time()
            run_state = agent.run_agent_env_interaction(step, run_state)
            time_taken = (time.time() - time_start) / exp_config.eval_frequency_steps
            num_games = run_state.game_stats.num_games
            smoothing = exp_config.avg_return_smoothing
            avg_return = _smoothed_stat(
                jnp, run_state.game_stats.avg_return, num_games, smoothing
            )
            avg_split_sum_rewards = _smoothed_stat(
                jnp,
                run_state.game_stats.avg_split_sum_rewards,
                num_games,
                smoothing,
            )
            avg_split_mixed_auc_sum = _smoothed_stat(
                jnp,
                run_state.game_stats.avg_split_mixed_auc_sum,
                num_games,
                smoothing,
            )
            avg_split_mixed_mass_sum = _smoothed_stat(
                jnp,
                run_state.game_stats.avg_split_mixed_mass_sum,
                num_games,
                smoothing,
            )
            best_returns = [
                _finite_or_none(float(value))
                for value in run_state.game_stats.best_return
            ]
            best_effective_t_costs = [
                _finite_or_none(float(value))
                for value in run_state.game_stats.best_effective_t_cost
            ]
            best_frontier_residual_weights = _finite_list(
                np.asarray(run_state.game_stats.best_frontier_residual_weight)
            )
            best_frontier_effective_t_costs = _finite_list(
                np.asarray(run_state.game_stats.best_frontier_effective_t_cost)
            )
            best_frontier_num_moves = _int_list(
                np.asarray(run_state.game_stats.best_frontier_num_moves)
            )
            best_tcounts_from_return = [
                None if value is None else int(-value) for value in best_returns
            ]

            headline = (
                f"Step: {step + exp_config.eval_frequency_steps} .. "
                f"Running Average Returns: {avg_return} .. "
                f"Time taken: {time_taken} seconds/step"
            )
            if args.split_reward_mode == "none":
                best_detail = best_tcounts_from_return
            else:
                best_detail = best_effective_t_costs
            detail = (
                "  Per-target best "
                f"({'T-count' if args.split_reward_mode == 'none' else 'effective T-cost'}): "
                + ", ".join(
                    f"{name}={value}"
                    for name, value in zip(target_names, best_detail, strict=True)
                )
            )
            frontier_detail = (
                "  Per-target frontier residual: "
                + ", ".join(
                    f"{name}={residual}@cost{cost}/m{moves}"
                    for name, residual, cost, moves in zip(
                        target_names,
                        best_frontier_residual_weights,
                        best_frontier_effective_t_costs,
                        best_frontier_num_moves,
                        strict=True,
                    )
                )
            )
            print(headline)
            print(detail)
            print(frontier_detail)
            text_log.write(headline + "\n")
            text_log.write(detail + "\n")
            text_log.write(frontier_detail + "\n")
            history.append(
                {
                    "step": step + exp_config.eval_frequency_steps,
                    "avg_return": [
                        float(value) for value in np.asarray(avg_return)
                    ],
                    "avg_split_sum_rewards": [
                        float(value) for value in np.asarray(avg_split_sum_rewards)
                    ],
                    "avg_split_mixed_auc_sum": [
                        float(value) for value in np.asarray(avg_split_mixed_auc_sum)
                    ],
                    "avg_split_mixed_mass_sum": [
                        float(value)
                        for value in np.asarray(avg_split_mixed_mass_sum)
                    ],
                    "best_return": best_returns,
                    "best_tcount_from_return": (
                        best_tcounts_from_return
                        if args.split_reward_mode == "none"
                        else [None for _ in target_names]
                    ),
                    "best_effective_t_cost": best_effective_t_costs,
                    "best_frontier_residual_weight": (
                        best_frontier_residual_weights
                    ),
                    "best_frontier_effective_t_cost": (
                        best_frontier_effective_t_costs
                    ),
                    "best_frontier_num_moves": best_frontier_num_moves,
                    "time_per_step_sec": time_taken,
                }
            )

    best_returns_final = [
        _finite_or_none(float(value)) for value in run_state.game_stats.best_return
    ]
    best_effective_t_costs_final = [
        _finite_or_none(float(value))
        for value in run_state.game_stats.best_effective_t_cost
    ]
    best_return_effective_t_costs_final = _finite_list(
        np.asarray(run_state.game_stats.best_return_effective_t_cost)
    )
    best_return_num_moves_final = _int_list(
        np.asarray(run_state.game_stats.best_return_num_moves)
    )
    best_return_residual_weight_final = _finite_list(
        np.asarray(run_state.game_stats.best_return_residual_weight)
    )
    best_solved_num_moves_final = _int_list(
        np.asarray(run_state.game_stats.best_solved_num_moves)
    )
    best_frontier_residual_weight_final = _finite_list(
        np.asarray(run_state.game_stats.best_frontier_residual_weight)
    )
    best_frontier_effective_t_cost_final = _finite_list(
        np.asarray(run_state.game_stats.best_frontier_effective_t_cost)
    )
    best_frontier_num_moves_final = _int_list(
        np.asarray(run_state.game_stats.best_frontier_num_moves)
    )
    best_tcounts_final = [
        None if value is None else int(-value) for value in best_returns_final
    ]
    reference_best_tcounts = [
        _reference_best_tcount(target_name, use_gadgets)
        for target_name in target_names
    ]
    target_tensor_weights = [
        int(
            np.asarray(
                tensors.get_signature_tensor(target),
                dtype=np.int32,
            ).sum()
        )
        for target in config.env_config.target_circuit_types
    ]
    best_frontier_residual_drop_final = [
        (
            None
            if frontier is None
            else float(weight) - float(frontier)
        )
        for weight, frontier in zip(
            target_tensor_weights,
            best_frontier_residual_weight_final,
        )
    ]
    summary = {
        "profile": args.profile,
        "mode": "quick",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "target_circuits": target_names,
        "target_preset": args.target_preset,
        "use_gadgets": use_gadgets,
        "split_reward_mode": args.split_reward_mode,
        "split_reward": dataclasses.asdict(config.env_config.split_reward),
        "partition_preset": args.partition_preset,
        "canonical_only": args.canonical_only,
        "force_canonical_basis": args.force_canonical_basis,
        "max_num_moves": config.env_config.max_num_moves,
        "num_past_factors_to_observe": (
            config.env_config.num_past_factors_to_observe
        ),
        "training_steps": args.training_steps,
        "eval_frequency_steps": args.eval_frequency,
        "batch_size": args.batch_size,
        "num_mcts_simulations": args.num_mcts_simulations,
        "action_dictionary": config.exp_config.action_dictionary,
        "max_action_weight": config.exp_config.max_action_weight,
        "tensor_overlap_max_weight": config.exp_config.tensor_overlap_max_weight,
        "tensor_overlap_max_actions_per_target": (
            config.exp_config.tensor_overlap_max_actions_per_target
        ),
        "gadget_closure_max_weight": config.exp_config.gadget_closure_max_weight,
        "mask_padded_actions": config.exp_config.mask_padded_actions,
        "mask_repeated_actions": config.exp_config.mask_repeated_actions,
        "action_prior": config.exp_config.action_prior,
        "action_prior_beta": config.exp_config.action_prior_beta,
        "action_prior_residual_weight": config.exp_config.action_prior_residual_weight,
        "action_prior_mixed_drop_weight": (
            config.exp_config.action_prior_mixed_drop_weight
        ),
        "action_prior_mixed_mass_weight": (
            config.exp_config.action_prior_mixed_mass_weight
        ),
        "action_prior_hamming_weight": config.exp_config.action_prior_hamming_weight,
        "action_prior_gadget_bonus": config.exp_config.action_prior_gadget_bonus,
        "action_prior_standardize": config.exp_config.action_prior_standardize,
        "action_prior_top_k": config.exp_config.action_prior_top_k,
        "action_prior_canonical_only": config.exp_config.action_prior_canonical_only,
        "frontier_replay_fraction": config.exp_config.frontier_replay_fraction,
        "frontier_replay_min_moves": config.exp_config.frontier_replay_min_moves,
        "frontier_replay_min_residual_drop": (
            config.exp_config.frontier_replay_min_residual_drop
        ),
        "num_actions": int(agent._num_actions),  # pylint: disable=protected-access
        "seed": args.seed,
        "target_tensor_weight": target_tensor_weights,
        "reference_best_tcount": reference_best_tcounts,
        "best_return": best_returns_final,
        "best_tcount_from_return": (
            best_tcounts_final
            if args.split_reward_mode == "none"
            else [None for _ in target_names]
        ),
        "best_effective_t_cost": best_effective_t_costs_final,
        "best_return_effective_t_cost": best_return_effective_t_costs_final,
        "best_return_num_moves": best_return_num_moves_final,
        "best_return_residual_weight": best_return_residual_weight_final,
        "best_solved_num_moves": best_solved_num_moves_final,
        "best_frontier_residual_weight": best_frontier_residual_weight_final,
        "best_frontier_residual_drop": best_frontier_residual_drop_final,
        "best_frontier_effective_t_cost": best_frontier_effective_t_cost_final,
        "best_frontier_num_moves": best_frontier_num_moves_final,
        "matched_reference": [
            observed is not None
            and reference is not None
            and observed <= reference
            for observed, reference in zip(
                best_tcounts_final, reference_best_tcounts, strict=True
            )
        ],
        "history": history,
        "text_log_path": str(text_log_path),
        "json_log_path": str(json_log_path),
    }
    candidate_output_dir = (
        args.candidate_output_dir
        if args.candidate_output_dir is not None
        else DEFAULT_CANDIDATE_ROOT
        / f"{args.target_preset}_{args.split_reward_mode}_{run_id}"
    )
    candidate_manifest_path = _export_candidate_factors(
        config, run_state, candidate_output_dir
    )
    summary["candidate_output_dir"] = str(candidate_output_dir)
    summary["candidate_manifest_path"] = str(candidate_manifest_path)
    json_log_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    append_command(
        {
            "tool": "run_demo_train.py",
            "mode": "quick",
            "profile": args.profile,
            "command": " ".join(sys.argv),
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
    parser.add_argument("--profile", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--target-preset",
        choices=(
            "demo",
            "mod_5_4",
            "gf_2pow2_mult",
            "hamming_weight_n4",
            "hamming_weight_n5",
            "split4",
            "qft_4",
        ),
        default="mod_5_4",
        help="Target set used by smoke and quick modes.",
    )
    parser.add_argument(
        "--split-reward-mode",
        choices=(
            "none",
            "mixed_drop",
            "mixed_auc",
            "v1",
            "v2_progress",
            "v3_frontier",
            "v4_sticky_frontier",
            "v1_guarded",
            "v1_tiebreak",
        ),
        default="none",
        help="Optional AlphaQuantum-only splitting reward mode.",
    )
    parser.add_argument("--lambda-drop", type=float, default=0.05)
    parser.add_argument("--lambda-auc", type=float, default=0.05)
    parser.add_argument("--lambda-mass", type=float, default=0.05)
    parser.add_argument("--lambda-residual", type=float, default=0.10)
    parser.add_argument("--lambda-frontier", type=float, default=0.10)
    parser.add_argument("--lambda-budget", type=float, default=1.0)
    parser.add_argument("--drop-clip", type=float, default=1.0)
    parser.add_argument("--t-guard-delta", type=float, default=0.0)
    parser.add_argument("--interim-budget-slack", type=float, default=5.0)
    parser.add_argument("--terminal-tiebreak-clip", type=float, default=0.25)
    parser.add_argument(
        "--baseline-t-costs",
        default=None,
        help=(
            "Guard budgets, either name=value comma pairs or colon-separated "
            "values in target order."
        ),
    )
    parser.add_argument(
        "--canonical-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only apply split reward in the canonical basis.",
    )
    parser.add_argument(
        "--force-canonical-basis",
        action="store_true",
        help="Train only in the canonical basis instead of the default basis mix.",
    )
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
        "--action-dictionary",
        choices=("full", "low-weight", "tensor-overlap", "gadget-closure"),
        default="full",
        help="Action dictionary used by the demo agent.",
    )
    parser.add_argument(
        "--max-action-weight",
        type=int,
        default=3,
        help="Maximum Hamming weight for the low-weight dictionary base.",
    )
    parser.add_argument(
        "--tensor-overlap-max-weight",
        type=int,
        default=5,
        help="Maximum Hamming weight considered by tensor-overlap expansion.",
    )
    parser.add_argument(
        "--tensor-overlap-max-actions-per-target",
        type=int,
        default=128,
        help="Maximum target-guided extra actions per target.",
    )
    parser.add_argument(
        "--gadget-closure-max-weight",
        type=int,
        default=4,
        help="Maximum Hamming weight exposed by gadget-closure mode.",
    )
    parser.add_argument(
        "--mask-padded-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask actions that touch padded coordinates beyond each target size.",
    )
    parser.add_argument(
        "--mask-repeated-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mask rank-one factors already selected in the current episode, "
            "avoiding GF(2) cancellation cycles."
        ),
    )
    parser.add_argument(
        "--max-num-moves",
        type=int,
        default=0,
        help=(
            "Override the environment episode horizon. Zero preserves the "
            "demo default."
        ),
    )
    parser.add_argument(
        "--num-past-factors-to-observe",
        type=int,
        default=0,
        help=(
            "Override how many past factors are included in observations. "
            "Zero preserves the demo default."
        ),
    )
    parser.add_argument(
        "--action-prior",
        choices=("none", "residual", "split"),
        default="none",
        help="Optional state-aware prior added to MCTS policy logits.",
    )
    parser.add_argument("--action-prior-beta", type=float, default=1.0)
    parser.add_argument("--action-prior-residual-weight", type=float, default=1.0)
    parser.add_argument("--action-prior-mixed-drop-weight", type=float, default=1.0)
    parser.add_argument("--action-prior-mixed-mass-weight", type=float, default=0.25)
    parser.add_argument("--action-prior-hamming-weight", type=float, default=0.05)
    parser.add_argument("--action-prior-gadget-bonus", type=float, default=0.25)
    parser.add_argument(
        "--action-prior-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize action-prior scores over currently valid actions.",
    )
    parser.add_argument(
        "--action-prior-top-k",
        type=int,
        default=0,
        help=(
            "If positive, keep only the top-k prior-scored valid actions in "
            "MCTS. Zero preserves the full valid action set."
        ),
    )
    parser.add_argument(
        "--action-prior-canonical-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only apply the action prior in the canonical basis.",
    )
    parser.add_argument(
        "--frontier-replay-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of terminated acting episodes to restart from the current "
            "best residual frontier for the sampled target."
        ),
    )
    parser.add_argument(
        "--frontier-replay-min-moves",
        type=int,
        default=0,
        help=(
            "Minimum stored frontier depth required before frontier replay can "
            "restart from it."
        ),
    )
    parser.add_argument(
        "--frontier-replay-min-residual-drop",
        type=float,
        default=0.0,
        help=(
            "Minimum normalized residual reduction required before a frontier "
            "can be archived/replayed when frontier replay is active."
        ),
    )
    parser.add_argument(
        "--partition-preset",
        choices=PARTITION_PRESETS,
        default="balanced",
        help=(
            "Tensor partition preset used by split rewards and split action "
            "priors."
        ),
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
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional extra path for a machine-readable quick-mode summary.",
    )
    parser.add_argument(
        "--candidate-output-dir",
        type=Path,
        default=None,
        help="Optional directory where quick mode exports best factor arrays.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "control":
        return run_control(args.profile, args.log_dir)
    if args.mode == "smoke":
        return run_smoke(args)
    return run_quick(args)


if __name__ == "__main__":
    raise SystemExit(main())
