from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_demo_train


def _args(**overrides):
    defaults = {
        "target_preset": "split4",
        "split_reward_mode": "v1",
        "canonical_only": True,
        "force_canonical_basis": False,
        "lambda_drop": 0.05,
        "lambda_auc": 0.05,
        "lambda_mass": 0.05,
        "lambda_residual": 0.10,
        "lambda_frontier": 0.10,
        "lambda_budget": 1.0,
        "drop_clip": 1.0,
        "baseline_t_costs": None,
        "t_guard_delta": 0.0,
        "interim_budget_slack": 5.0,
        "terminal_tiebreak_clip": 0.25,
        "action_dictionary": "full",
        "max_action_weight": 3,
        "tensor_overlap_max_weight": 5,
        "tensor_overlap_max_actions_per_target": 128,
        "gadget_closure_max_weight": 4,
        "mask_padded_actions": False,
        "mask_repeated_actions": False,
        "max_num_moves": 0,
        "num_past_factors_to_observe": 0,
        "action_prior": "none",
        "action_prior_beta": 1.0,
        "action_prior_residual_weight": 1.0,
        "action_prior_mixed_drop_weight": 1.0,
        "action_prior_mixed_mass_weight": 0.25,
        "action_prior_hamming_weight": 0.05,
        "action_prior_gadget_bonus": 0.25,
        "action_prior_standardize": True,
        "action_prior_top_k": 0,
        "action_prior_canonical_only": True,
        "frontier_replay_fraction": 0.0,
        "frontier_replay_min_moves": 0,
        "frontier_replay_min_residual_drop": 0.0,
        "partition_preset": "balanced",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_qft_4_is_evaluation_only_without_restricted_action_dictionary():
    with pytest.raises(ValueError, match="qft_4.*evaluation-only"):
        run_demo_train._configured_demo_config(
            _args(target_preset="qft_4"), use_gadgets=True
        )


@pytest.mark.parametrize(
    ("preset", "expected_target", "expected_size"),
    [
        ("gf_2pow2_mult", "gf_2pow2_mult", 6),
        ("hamming_weight_n4", "hamming_weight_n4", 9),
        ("hamming_weight_n5", "hamming_weight_n5", 10),
    ],
)
def test_individual_split_targets_are_trainable_presets(
    preset: str,
    expected_target: str,
    expected_size: int,
):
    config = run_demo_train._configured_demo_config(
        _args(target_preset=preset),
        use_gadgets=True,
    )

    assert [target.name.lower() for target in config.env_config.target_circuit_types] == [
        expected_target
    ]
    assert config.env_config.max_tensor_size == expected_size


def test_low_weight_action_dictionary_reduces_agent_action_space():
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n5",
            action_dictionary="low-weight",
            max_action_weight=2,
        ),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)

    assert agent._full_num_actions == 1023
    assert agent._num_actions == 55


def test_training_config_can_extend_environment_horizon():
    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="gf_2pow2_mult",
            max_num_moves=60,
            num_past_factors_to_observe=10,
        ),
        use_gadgets=True,
    )

    assert config.env_config.max_num_moves == 60
    assert config.env_config.num_past_factors_to_observe == 10


def test_mask_repeated_actions_removes_previous_factor_from_valid_actions():
    import jax.numpy as jnp
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n5",
            action_dictionary="low-weight",
            max_action_weight=2,
            mask_repeated_actions=True,
        ),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jnp.array([0, 1], dtype=jnp.uint32))
    repeated_full_action = 0  # factor [1, 0, ...]
    repeated_restricted_action = int(
        agent._restricted_action_from_full[repeated_full_action]
    )
    repeated_factor = agent._action_factors[repeated_restricted_action]

    initial_valid = agent._action_valid_mask(run_state.env_states)
    assert bool(initial_valid[0, repeated_restricted_action])

    past_factors = run_state.env_states.past_factors.at[0, -1, :].set(
        repeated_factor
    )
    repeated_state = run_state.env_states._replace(
        past_factors=past_factors,
        num_moves=run_state.env_states.num_moves.at[0].set(1),
    )
    repeated_valid = agent._action_valid_mask(repeated_state)

    assert not bool(repeated_valid[0, repeated_restricted_action])


def test_tensor_overlap_dictionary_adds_target_guided_actions():
    from alphatensor_quantum.src.demo import agent as agent_lib

    low_weight_config = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n5",
            action_dictionary="low-weight",
            max_action_weight=2,
        ),
        use_gadgets=True,
    )
    overlap_config = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n5",
            action_dictionary="tensor-overlap",
            max_action_weight=2,
            tensor_overlap_max_weight=5,
            tensor_overlap_max_actions_per_target=64,
        ),
        use_gadgets=True,
    )
    low_weight_agent = agent_lib.Agent(low_weight_config)
    overlap_agent = agent_lib.Agent(overlap_config)

    assert low_weight_agent._num_actions == 55
    assert 55 < overlap_agent._num_actions < overlap_agent._full_num_actions
    assert any(
        int(action + 1).bit_count() > 2
        for action in overlap_agent._action_indices.tolist()
    )


def test_gadget_closure_dictionary_masks_completion_actions_until_prefix():
    import jax.numpy as jnp
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n4",
            action_dictionary="gadget-closure",
            max_action_weight=1,
            gadget_closure_max_weight=2,
        ),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jnp.array([0, 1], dtype=jnp.uint32))

    cs_completion_full_action = 2  # factor [1, 1, 0, ...]
    cs_completion_restricted = int(
        agent._restricted_action_from_full[cs_completion_full_action]
    )
    assert cs_completion_restricted >= 0
    assert not bool(agent._base_action_mask[cs_completion_restricted])

    logits = jnp.zeros((config.exp_config.batch_size, agent._num_actions))
    initial_masked = agent._mask_padded_action_logits(
        logits,
        run_state.env_states,
    )
    assert float(initial_masked[0, cs_completion_restricted]) < -1.0e8

    e0 = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    e1 = jnp.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    prefixed_factors = run_state.env_states.past_factors.at[0, -2, :].set(e0)
    prefixed_factors = prefixed_factors.at[0, -1, :].set(e1)
    prefixed_state = run_state.env_states._replace(
        past_factors=prefixed_factors,
        num_moves=run_state.env_states.num_moves.at[0].set(2),
    )
    prefixed_masked = agent._mask_padded_action_logits(
        logits,
        prefixed_state,
    )
    assert float(prefixed_masked[0, cs_completion_restricted]) == 0.0


def test_full_action_dictionary_preserves_original_action_space():
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(target_preset="hamming_weight_n5"),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)

    assert agent._full_num_actions == 1023
    assert agent._num_actions == 1023


def test_frontier_replay_restarts_from_stored_frontier_state():
    import dataclasses

    import jax
    import jax.numpy as jnp
    import numpy as np
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="mod_5_4",
            force_canonical_basis=True,
            frontier_replay_fraction=1.0,
        ),
        use_gadgets=True,
    )
    config = dataclasses.replace(
        config,
        exp_config=dataclasses.replace(config.exp_config, batch_size=2),
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jax.random.PRNGKey(0))
    base_states = agent._env.init_state(jax.random.split(jax.random.PRNGKey(1), 2))
    frontier_states = agent._env.step(jnp.zeros((2,), dtype=jnp.int32), base_states)
    nonzero_index = tuple(np.argwhere(np.asarray(frontier_states.tensor[0]) == 1)[0])
    reduced_tensor = frontier_states.tensor.at[
        (slice(None),) + nonzero_index
    ].set(0)
    frontier_states = frontier_states._replace(tensor=reduced_tensor)
    frontier_residual = jnp.sum(frontier_states.tensor[0].astype(jnp.float32))
    game_stats = run_state.game_stats._replace(
        best_frontier_residual_weight=jnp.array([frontier_residual]),
        best_frontier_effective_t_cost=frontier_states.effective_t_cost[:1],
        best_frontier_num_moves=frontier_states.num_moves[:1],
        best_frontier_factors=frontier_states.past_factors[:1],
        best_frontier_change_of_basis=frontier_states.change_of_basis[:1],
        best_frontier_factors_in_gadgets=(
            frontier_states.factors_in_gadgets[:1]
        ),
        best_frontier_tensor=frontier_states.tensor[:1],
        best_frontier_sum_rewards=frontier_states.sum_rewards[:1],
        best_frontier_split_sum_rewards=frontier_states.split_sum_rewards[:1],
        best_frontier_split_mixed_auc_sum=(
            frontier_states.split_mixed_auc_sum[:1]
        ),
        best_frontier_split_mixed_mass_sum=(
            frontier_states.split_mixed_mass_sum[:1]
        ),
    )
    fresh_states = agent._env.init_state(jax.random.split(jax.random.PRNGKey(2), 2))

    replay_states = agent._frontier_replay_states(
        fresh_states,
        game_stats,
        jax.random.PRNGKey(3),
    )

    np.testing.assert_array_equal(
        np.asarray(replay_states.tensor),
        np.broadcast_to(
            np.asarray(frontier_states.tensor[:1]),
            np.asarray(replay_states.tensor).shape,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(replay_states.past_factors),
        np.broadcast_to(
            np.asarray(frontier_states.past_factors[:1]),
            np.asarray(replay_states.past_factors).shape,
        ),
    )
    assert np.asarray(replay_states.num_moves).tolist() == [
        int(frontier_states.num_moves[0])
    ] * 2
    assert np.asarray(replay_states.effective_t_cost).tolist() == [
        float(frontier_states.effective_t_cost[0])
    ] * 2
    assert np.asarray(replay_states.sum_rewards).tolist() == [
        float(frontier_states.sum_rewards[0])
    ] * 2

    gated_config = dataclasses.replace(
        config,
        exp_config=dataclasses.replace(
            config.exp_config,
            frontier_replay_min_moves=int(frontier_states.num_moves[0]) + 1,
        ),
    )
    gated_agent = agent_lib.Agent(gated_config)
    gated_run_state = gated_agent.init_run_state(jax.random.PRNGKey(6))
    gated_fresh_states = gated_agent._env.init_state(
        jax.random.split(jax.random.PRNGKey(4), 2)
    )
    gated_replay_states = gated_agent._frontier_replay_states(
        gated_fresh_states,
        game_stats,
        jax.random.PRNGKey(5),
    )

    assert np.asarray(gated_replay_states.num_moves).tolist() == [0, 0]

    shallow_states = frontier_states._replace(
        num_moves=jnp.ones((2,), dtype=jnp.int32),
    )
    shallow_stats = gated_agent._update_game_stats(gated_run_state, shallow_states)
    assert np.isinf(float(shallow_stats.best_frontier_residual_weight[0]))

    deep_states = frontier_states._replace(
        num_moves=jnp.full((2,), 2, dtype=jnp.int32),
    )
    deep_stats = gated_agent._update_game_stats(gated_run_state, deep_states)
    assert np.isfinite(float(deep_stats.best_frontier_residual_weight[0]))
    assert int(deep_stats.best_frontier_num_moves[0]) == 2

    drop_gated_config = dataclasses.replace(
        config,
        exp_config=dataclasses.replace(
            config.exp_config,
            frontier_replay_min_residual_drop=0.99,
        ),
    )
    drop_gated_agent = agent_lib.Agent(drop_gated_config)
    drop_gated_run_state = drop_gated_agent.init_run_state(jax.random.PRNGKey(7))
    drop_gated_states = frontier_states
    drop_gated_stats = drop_gated_agent._update_game_stats(
        drop_gated_run_state,
        drop_gated_states,
    )
    assert np.isinf(float(drop_gated_stats.best_frontier_residual_weight[0]))


def test_mask_padded_actions_rejects_factors_outside_target_size():
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="split4",
            action_dictionary="low-weight",
            max_action_weight=2,
            mask_padded_actions=True,
        ),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)

    bit_9_full_action = (1 << 9) - 1
    restricted_action = int(
        agent._restricted_action_from_full[bit_9_full_action]
    )

    assert restricted_action >= 0
    # split4 target order is mod_5_4, gf_2pow2_mult, hamming_weight_n4,
    # hamming_weight_n5. Bit 9 is padding for gf_2pow2_mult(size 6), but active
    # for hamming_weight_n5(size 10).
    assert not bool(agent._action_valid_by_target[1, restricted_action])
    assert bool(agent._action_valid_by_target[3, restricted_action])


def test_default_action_prior_is_disabled_and_preserves_logits():
    import jax.numpy as jnp
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="mod_5_4",
            action_dictionary="low-weight",
            max_action_weight=2,
        ),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jnp.array([0, 1], dtype=jnp.uint32))
    logits = jnp.arange(
        config.exp_config.batch_size * agent._num_actions,
        dtype=jnp.float32,
    ).reshape(config.exp_config.batch_size, agent._num_actions)

    assert config.exp_config.action_prior == "none"
    assert jnp.allclose(
        agent._apply_action_prior_to_logits(logits, run_state.env_states),
        logits,
    )


def test_residual_action_prior_prefers_immediate_residual_drop():
    import jax.numpy as jnp
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="mod_5_4",
            action_dictionary="low-weight",
            max_action_weight=2,
            action_prior="residual",
            action_prior_canonical_only=False,
        ),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jnp.array([0, 1], dtype=jnp.uint32))
    tensor = jnp.zeros_like(run_state.env_states.tensor)
    tensor = tensor.at[:, 0, 0, 0].set(1)
    state = run_state.env_states._replace(
        tensor=tensor,
        init_tensor_index=jnp.zeros(
            (config.exp_config.batch_size,), dtype=jnp.int32
        ),
        is_terminal=jnp.zeros(
            (config.exp_config.batch_size,), dtype=jnp.bool_
        ),
    )
    e0_action = int(agent._restricted_action_from_full[0])
    e0e1_action = int(agent._restricted_action_from_full[2])

    prior = agent._state_action_prior_logits(state)

    assert e0_action >= 0
    assert e0e1_action >= 0
    assert float(prior[0, e0_action]) > float(prior[0, e0e1_action])


def test_action_prior_top_k_narrows_search_logits():
    import jax.numpy as jnp
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="mod_5_4",
            action_dictionary="low-weight",
            max_action_weight=2,
            action_prior="residual",
            action_prior_top_k=1,
            action_prior_canonical_only=False,
        ),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jnp.array([0, 1], dtype=jnp.uint32))
    tensor = jnp.zeros_like(run_state.env_states.tensor)
    tensor = tensor.at[:, 0, 0, 0].set(1)
    state = run_state.env_states._replace(
        tensor=tensor,
        init_tensor_index=jnp.zeros(
            (config.exp_config.batch_size,), dtype=jnp.int32
        ),
        is_terminal=jnp.zeros(
            (config.exp_config.batch_size,), dtype=jnp.bool_
        ),
    )
    logits = jnp.zeros((config.exp_config.batch_size, agent._num_actions))
    e0_action = int(agent._restricted_action_from_full[0])
    e0e1_action = int(agent._restricted_action_from_full[2])

    search_logits = agent._policy_logits_for_search(logits, state)

    assert e0_action >= 0
    assert e0e1_action >= 0
    assert float(search_logits[0, e0_action]) > -1.0e8
    assert float(search_logits[0, e0e1_action]) < -1.0e8


def test_split_action_prior_is_finite_on_gadget_closure_dictionary():
    import jax.numpy as jnp
    from alphatensor_quantum.src.demo import agent as agent_lib

    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n4",
            action_dictionary="gadget-closure",
            max_action_weight=2,
            gadget_closure_max_weight=4,
            action_prior="split",
            action_prior_canonical_only=False,
        ),
        use_gadgets=True,
    )
    agent = agent_lib.Agent(config)
    run_state = agent.init_run_state(jnp.array([0, 1], dtype=jnp.uint32))

    prior = agent._state_action_prior_logits(run_state.env_states)

    assert prior.shape == (config.exp_config.batch_size, agent._num_actions)
    assert bool(jnp.all(jnp.isfinite(prior)))


def test_split4_partitions_match_environment_size_and_weights():
    config = run_demo_train._configured_demo_config(_args(), use_gadgets=True)
    split_reward = config.env_config.split_reward
    max_size = config.env_config.max_tensor_size

    assert [
        target.name.lower() for target in config.env_config.target_circuit_types
    ] == [
        "mod_5_4",
        "gf_2pow2_mult",
        "hamming_weight_n4",
        "hamming_weight_n5",
    ]
    assert len(split_reward.partition_blocks_by_target) == 4
    assert len(split_reward.partition_weights_by_target) == 4
    for target_blocks, target_weights in zip(
        split_reward.partition_blocks_by_target,
        split_reward.partition_weights_by_target,
        strict=True,
    ):
        assert sum(target_weights) == pytest.approx(1.0)
        for blocks in target_blocks:
            assert len(blocks) == max_size


def test_split_action_prior_configures_partitions_even_without_reward():
    config = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n4",
            split_reward_mode="none",
            action_prior="split",
            partition_preset="ensemble",
        ),
        use_gadgets=True,
    )
    split_reward = config.env_config.split_reward

    assert split_reward.mode == "none"
    assert split_reward.partition_blocks_by_target is not None
    assert split_reward.partition_weights_by_target is not None
    assert len(split_reward.partition_blocks_by_target) == 1
    assert len(split_reward.partition_blocks_by_target[0]) == 4
    assert sum(split_reward.partition_weights_by_target[0]) == pytest.approx(1.0)


def test_tensor_spectral_partition_preset_is_deterministic_and_padded():
    config_a = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n4",
            split_reward_mode="none",
            action_prior="split",
            partition_preset="tensor-spectral",
        ),
        use_gadgets=True,
    )
    config_b = run_demo_train._configured_demo_config(
        _args(
            target_preset="hamming_weight_n4",
            split_reward_mode="none",
            action_prior="split",
            partition_preset="tensor-spectral",
        ),
        use_gadgets=True,
    )

    blocks_a = config_a.env_config.split_reward.partition_blocks_by_target
    blocks_b = config_b.env_config.split_reward.partition_blocks_by_target
    assert blocks_a == blocks_b
    assert len(blocks_a[0][0]) == config_a.env_config.max_tensor_size
    assert set(blocks_a[0][0]) == {0, 1}


def test_ablation_budgeted_modes_do_not_invent_unsolved_budgets():
    from scripts import run_split_reward_ablation

    summary = {
        "target_circuits": ["mod_5_4", "gf_2pow2_mult"],
        "best_effective_t_cost": [2.0, None],
    }

    with pytest.raises(RuntimeError, match="did not solve: gf_2pow2_mult"):
        run_split_reward_ablation._guard_budgets_from_none(summary)


def test_ablation_mode_selection_rejects_unknown_modes():
    from scripts import run_split_reward_ablation

    with pytest.raises(ValueError, match="Unknown modes"):
        run_split_reward_ablation._parse_modes("none,not_a_mode")


def test_target_sweep_rejects_unknown_targets_and_modes():
    from scripts import run_split_reward_target_sweep

    with pytest.raises(ValueError, match="Unknown targets"):
        run_split_reward_target_sweep.parse_csv_list(
            "mod_5_4,unknown",
            run_split_reward_target_sweep.TARGETS,
            "targets",
        )
    with pytest.raises(ValueError, match="Unknown modes"):
        run_split_reward_target_sweep.parse_csv_list(
            "none,unknown",
            run_split_reward_target_sweep.MODES,
            "modes",
        )


def test_target_sweep_tags_action_dictionary_and_basis_regime():
    from scripts import run_split_reward_target_sweep

    args = SimpleNamespace(
        action_dictionary="low-weight",
        max_action_weight=2,
        tensor_overlap_max_weight=5,
        tensor_overlap_max_actions_per_target=64,
        action_prior="none",
        action_prior_beta=1.0,
        action_prior_residual_weight=1.0,
        action_prior_mixed_drop_weight=1.0,
        action_prior_mixed_mass_weight=0.25,
        action_prior_top_k=0,
        partition_preset="balanced",
        max_num_moves=0,
        num_past_factors_to_observe=0,
        force_canonical_basis=True,
        frontier_replay_fraction=0.0,
        frontier_replay_min_moves=0,
        frontier_replay_min_residual_drop=0.0,
    )

    assert run_split_reward_target_sweep._action_tag(args) == "loww2"
    assert run_split_reward_target_sweep._prior_tag(args) == "noprior"
    assert run_split_reward_target_sweep._partition_tag(args) == "pi-balanced"
    assert run_split_reward_target_sweep._basis_tag(args) == "canonical"
    assert run_split_reward_target_sweep._replay_tag(args) == "noreplay"

    args.action_dictionary = "tensor-overlap"
    assert (
        run_split_reward_target_sweep._action_tag(args)
        == "tensoroverlap64_w5_base2"
    )

    args.action_dictionary = "gadget-closure"
    args.gadget_closure_max_weight = 4
    assert run_split_reward_target_sweep._action_tag(args) == "gadgetclosure4_base2"

    args.action_prior = "split"
    args.partition_preset = "ensemble"
    assert run_split_reward_target_sweep._prior_tag(args).startswith(
        "splitprior_beta1"
    )
    assert run_split_reward_target_sweep._partition_tag(args) == "pi-ensemble"
    args.frontier_replay_fraction = 0.25
    args.frontier_replay_min_moves = 8
    args.frontier_replay_min_residual_drop = 0.04
    assert run_split_reward_target_sweep._replay_tag(args) == (
        "replay0p25_minm8_mindrop0p04"
    )


def test_target_sweep_budget_map_is_explicit_only():
    from scripts import run_split_reward_target_sweep

    assert run_split_reward_target_sweep.parse_budget_map(None) == {}
    assert run_split_reward_target_sweep.parse_budget_map(
        "mod_5_4=2,gf_2pow2_mult=17"
    ) == {
        "mod_5_4": 2.0,
        "gf_2pow2_mult": 17.0,
    }


def test_split_prior_grid_builds_expected_combinations():
    from scripts import run_split_prior_grid

    args = SimpleNamespace(
        partition_presets="balanced,ensemble",
        action_prior_betas="0.5,1.0",
        action_prior_residual_weights="1.0",
        action_prior_mixed_drop_weights="1.0,2.0",
        action_prior_mixed_mass_weights="0.1",
        action_prior_top_k=16,
        action_prior_top_ks=None,
        frontier_replay_fraction=0.0,
        frontier_replay_fractions=None,
        frontier_replay_min_moves=0,
        frontier_replay_min_moves_values=None,
        frontier_replay_min_residual_drop=0.0,
        frontier_replay_min_residual_drops=None,
    )

    configs = run_split_prior_grid.build_grid_configs(args)

    assert len(configs) == 8
    assert configs[0].grid_id == "pi-balanced_beta0p5_r1_md1_mm0p1_topk16"
    assert configs[-1].grid_id == "pi-ensemble_beta1_r1_md2_mm0p1_topk16"


def test_split_prior_grid_can_sweep_top_k_values():
    from scripts import run_split_prior_grid

    args = SimpleNamespace(
        partition_presets="ensemble",
        action_prior_betas="1.0",
        action_prior_residual_weights="1.0",
        action_prior_mixed_drop_weights="1.0",
        action_prior_mixed_mass_weights="0.25",
        action_prior_top_k=0,
        action_prior_top_ks="0,16,32",
        frontier_replay_fraction=0.0,
        frontier_replay_fractions=None,
        frontier_replay_min_moves=0,
        frontier_replay_min_moves_values=None,
        frontier_replay_min_residual_drop=0.0,
        frontier_replay_min_residual_drops=None,
    )

    configs = run_split_prior_grid.build_grid_configs(args)

    assert [config.top_k for config in configs] == [0, 16, 32]
    assert configs[0].grid_id == "pi-ensemble_beta1_r1_md1_mm0p25"
    assert configs[1].grid_id == "pi-ensemble_beta1_r1_md1_mm0p25_topk16"
    assert configs[2].grid_id == "pi-ensemble_beta1_r1_md1_mm0p25_topk32"


def test_split_prior_grid_can_sweep_frontier_replay_values():
    from scripts import run_split_prior_grid

    args = SimpleNamespace(
        partition_presets="ensemble",
        action_prior_betas="1.0",
        action_prior_residual_weights="1.0",
        action_prior_mixed_drop_weights="1.0",
        action_prior_mixed_mass_weights="0.25",
        action_prior_top_k=16,
        action_prior_top_ks=None,
        frontier_replay_fraction=0.0,
        frontier_replay_fractions="0,0.25,0.5",
        frontier_replay_min_moves=8,
        frontier_replay_min_moves_values=None,
        frontier_replay_min_residual_drop=0.04,
        frontier_replay_min_residual_drops=None,
    )

    configs = run_split_prior_grid.build_grid_configs(args)

    assert [config.frontier_replay_fraction for config in configs] == [
        0.0,
        0.25,
        0.5,
    ]
    assert configs[0].grid_id == "pi-ensemble_beta1_r1_md1_mm0p25_topk16"
    assert configs[1].grid_id == (
        "pi-ensemble_beta1_r1_md1_mm0p25_topk16_replay0p25_minm8_mindrop0p04"
    )
    assert configs[2].grid_id == (
        "pi-ensemble_beta1_r1_md1_mm0p25_topk16_replay0p5_minm8_mindrop0p04"
    )


def test_split_prior_grid_rejects_unknown_partition():
    from scripts import run_split_prior_grid

    args = SimpleNamespace(
        partition_presets="balanced,unknown",
        action_prior_betas="1.0",
        action_prior_residual_weights="1.0",
        action_prior_mixed_drop_weights="1.0",
        action_prior_mixed_mass_weights="0.25",
        action_prior_top_k=0,
        action_prior_top_ks=None,
        frontier_replay_fraction=0.0,
        frontier_replay_fractions=None,
        frontier_replay_min_moves=0,
        frontier_replay_min_moves_values=None,
        frontier_replay_min_residual_drop=0.0,
        frontier_replay_min_residual_drops=None,
    )

    with pytest.raises(ValueError, match="Unknown partition presets"):
        run_split_prior_grid.build_grid_configs(args)


def test_split_prior_grid_selects_best_residual_then_cost():
    from scripts import run_split_prior_grid

    rows = [
        {
            "target": "hamming_weight_n4",
            "mode": "none",
            "status": "ok",
            "grid_id": "a",
            "best_frontier_residual_weight": "120",
            "best_frontier_effective_t_cost": "29",
            "best_return_residual_weight": "119",
        },
        {
            "target": "hamming_weight_n4",
            "mode": "none",
            "status": "ok",
            "grid_id": "b",
            "best_frontier_residual_weight": "117",
            "best_frontier_effective_t_cost": "30",
            "best_return_residual_weight": "118",
        },
        {
            "target": "hamming_weight_n4",
            "mode": "none",
            "status": "ok",
            "grid_id": "c",
            "best_frontier_residual_weight": "117",
            "best_frontier_effective_t_cost": "29",
            "best_return_residual_weight": "118",
        },
    ]

    [best] = run_split_prior_grid._best_rows(rows)

    assert best["grid_id"] == "c"


def test_split_prior_grid_does_not_fallback_to_terminal_residual():
    from scripts import run_split_prior_grid

    rows = [
        {
            "target": "hamming_weight_n4",
            "mode": "none",
            "status": "ok",
            "grid_id": "terminal-only",
            "best_return_residual_weight": "1",
            "best_return_effective_t_cost": "1",
        },
        {
            "target": "hamming_weight_n4",
            "mode": "none",
            "status": "ok",
            "grid_id": "frontier",
            "best_frontier_residual_weight": "117",
            "best_frontier_effective_t_cost": "29",
            "best_return_residual_weight": "118",
        },
    ]

    [best] = run_split_prior_grid._best_rows(rows)

    assert best["grid_id"] == "frontier"


def test_materialize_split_reward_canonicalizes_change_of_basis():
    import numpy as np

    from scripts.materialize_split_reward_candidate import canonicalize_factors
    from scripts.materialize_split_reward_candidate import rank_one_tensor_sum

    canonical = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )
    change_of_basis = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    changed_basis_factors = ((change_of_basis @ canonical.T) % 2).T

    recovered = canonicalize_factors(changed_basis_factors, change_of_basis)

    assert np.array_equal(recovered.astype(np.uint8), canonical)
    assert np.array_equal(
        rank_one_tensor_sum(recovered),
        rank_one_tensor_sum(canonical),
    )
