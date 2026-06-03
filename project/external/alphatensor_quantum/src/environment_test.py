# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src import change_of_basis as change_of_basis_lib
from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import demonstrations
from alphatensor_quantum.src import environment
from alphatensor_quantum.src import factors as factors_utils
from alphatensor_quantum.src import tensors


_SMALL_TCOUNT3_FACTORS = np.array(
    [[1, 1, 1], [0, 1, 1], [1, 0, 1]], dtype=np.int32
)


def _small_env_config(**kwargs):
  split_reward = kwargs.pop(
      'split_reward', config_lib.SplitRewardParams(mode='none')
  )
  change_of_basis = kwargs.pop(
      'change_of_basis',
      config_lib.ChangeOfBasisParams(
          num_change_of_basis_matrices=1,
          prob_canonical_basis=1.0,
      ),
  )
  return config_lib.EnvironmentParams(
      target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
      max_num_moves=kwargs.pop('max_num_moves', 10),
      change_of_basis=change_of_basis,
      split_reward=split_reward,
      **kwargs,
  )


class EnvironmentTest(parameterized.TestCase):

  def test_init_state(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.
    with self.subTest('tensor'):
      np.testing.assert_array_equal(
          env_state.tensor,
          tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3)[None]
      )
    with self.subTest('past_factors'):
      np.testing.assert_array_equal(
          env_state.past_factors, np.zeros((1, 10, 3), dtype=np.int32)
      )
    with self.subTest('num_moves'):
      np.testing.assert_array_equal(
          env_state.num_moves, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('last_reward'):
      np.testing.assert_array_equal(env_state.last_reward, np.zeros((1,)))
    with self.subTest('sum_rewards'):
      np.testing.assert_array_equal(env_state.sum_rewards, np.zeros((1,)))
    with self.subTest('is_terminal'):
      np.testing.assert_array_equal(
          env_state.is_terminal, np.zeros((1,), dtype=np.bool_)
      )
    with self.subTest('init_tensor_index'):
      np.testing.assert_array_equal(
          env_state.init_tensor_index, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('change_of_basis'):
      np.testing.assert_array_equal(
          env_state.change_of_basis, np.eye(3, dtype=np.int32)[None]
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets, np.zeros((1, 10), dtype=np.bool_)
      )
    with self.subTest('split_fields'):
      np.testing.assert_array_equal(env_state.effective_t_cost, np.zeros((1,)))
      np.testing.assert_array_equal(env_state.split_last_reward, np.zeros((1,)))
      np.testing.assert_array_equal(env_state.split_sum_rewards, np.zeros((1,)))
      np.testing.assert_array_equal(
          env_state.split_mixed_auc_sum, np.zeros((1,))
      )
      np.testing.assert_array_equal(
          env_state.split_mixed_mass_sum, np.zeros((1,))
      )
      np.testing.assert_array_equal(
          env_state.frontier_residual_weight,
          np.sum(env_state.tensor, axis=(-3, -2, -1)),
      )

  def test_init_state_change_of_basis(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            prob_zero_entry=0.3,
            num_change_of_basis_matrices=1,
            prob_canonical_basis=0.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.
    with self.subTest('cob_matrix'):
      np.testing.assert_array_equal(
          env_state.change_of_basis[0], env.change_of_basis[0]
      )
    expected_tensor = change_of_basis_lib.apply_change_of_basis(
        tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3),
        env.change_of_basis[0],
    )
    with self.subTest('tensor_is_in_non_canonical_basis'):
      np.testing.assert_array_equal(env_state.tensor[0], expected_tensor)

  def test_init_state_from_demonstration(self):
    env_config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    dem_config = config_lib.DemonstrationsParams(
        max_num_factors=10,
        max_num_gadgets=2,
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        3, dem_config, jax.random.PRNGKey(0)[None]  # Add batch dim.
    )
    env = environment.Environment(jax.random.PRNGKey(1), env_config)
    env_state = env.init_state_from_demonstration(demonstration)
    with self.subTest('tensor'):
      np.testing.assert_array_equal(env_state.tensor, demonstration.tensor)
    with self.subTest('past_factors'):
      np.testing.assert_array_equal(
          env_state.past_factors, np.zeros((1, 10, 3), dtype=np.int32)
      )
    with self.subTest('num_moves'):
      np.testing.assert_array_equal(
          env_state.num_moves, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('last_reward'):
      np.testing.assert_array_equal(env_state.last_reward, np.zeros((1,)))
    with self.subTest('sum_rewards'):
      np.testing.assert_array_equal(env_state.sum_rewards, np.zeros((1,)))
    with self.subTest('is_terminal'):
      np.testing.assert_array_equal(
          env_state.is_terminal, np.zeros((1,), dtype=np.bool_)
      )
    with self.subTest('init_tensor_index'):
      np.testing.assert_array_equal(
          env_state.init_tensor_index, -np.ones((1,), dtype=np.int32)
      )
    with self.subTest('change_of_basis'):
      np.testing.assert_array_equal(
          env_state.change_of_basis, np.eye(3, dtype=np.int32)[None]
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets, np.zeros((1, 10), dtype=np.bool_)
      )
    with self.subTest('split_fields'):
      np.testing.assert_array_equal(env_state.effective_t_cost, np.zeros((1,)))
      np.testing.assert_array_equal(env_state.split_last_reward, np.zeros((1,)))
      np.testing.assert_array_equal(env_state.split_sum_rewards, np.zeros((1,)))
      np.testing.assert_array_equal(
          env_state.split_mixed_auc_sum, np.zeros((1,))
      )
      np.testing.assert_array_equal(
          env_state.split_mixed_mass_sum, np.zeros((1,))
      )
      np.testing.assert_array_equal(
          env_state.frontier_residual_weight,
          np.sum(env_state.tensor, axis=(-3, -2, -1)),
      )

  def test_one_step(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    factor1 = jnp.array([1, 1, 1], dtype=jnp.int32)
    new_env_state = env.step(
        factors_utils.action_factor_to_index(factor1)[None], env_state
    )
    with self.subTest('tensor'):
      # The action of the factor [1, 1, 1] is to flip all the bits.
      np.testing.assert_array_equal(
          new_env_state.tensor,
          1 - tensors.get_signature_tensor(
              tensors.CircuitType.SMALL_TCOUNT_3
          )[None]
      )
    with self.subTest('past_factors'):
      np.testing.assert_array_equal(
          new_env_state.past_factors,
          np.concatenate([
              np.zeros((1, 9, 3), dtype=np.int32),
              np.ones((1, 1, 3), dtype=np.int32)
          ], axis=1)
      )
    with self.subTest('num_moves'):
      np.testing.assert_array_equal(
          new_env_state.num_moves, np.ones((1,), dtype=np.int32)
      )
    with self.subTest('last_reward'):
      np.testing.assert_array_equal(new_env_state.last_reward, np.array([-1.0]))
    with self.subTest('sum_rewards'):
      np.testing.assert_array_equal(new_env_state.sum_rewards, np.array([-1.0]))
    with self.subTest('effective_t_cost'):
      np.testing.assert_array_equal(
          new_env_state.effective_t_cost, np.array([1.0])
      )
    with self.subTest('is_terminal'):
      np.testing.assert_array_equal(
          new_env_state.is_terminal, np.array([False])
      )
    with self.subTest('init_tensor_index'):
      np.testing.assert_array_equal(
          new_env_state.init_tensor_index, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('change_of_basis'):
      np.testing.assert_array_equal(
          env_state.change_of_basis, np.eye(3, dtype=np.int32)[None]
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          new_env_state.factors_in_gadgets, np.zeros((1, 10), dtype=np.bool_)
      )

  def test_three_steps(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    for factor in _SMALL_TCOUNT3_FACTORS:
      env_state = env.step(
          factors_utils.action_factor_to_index(jnp.array(factor))[None],
          env_state
      )

    with self.subTest('tensor'):
      # After three steps, we should reach the all-zero tensor.
      np.testing.assert_array_equal(
          env_state.tensor, np.zeros((1, 3, 3, 3), dtype=np.int32)
      )
    with self.subTest('past_factors'):
      np.testing.assert_array_equal(
          env_state.past_factors,
          np.concatenate([
              np.zeros((1, 7, 3), dtype=np.int32), _SMALL_TCOUNT3_FACTORS[None]
          ], axis=1)
      )
    with self.subTest('num_moves'):
      np.testing.assert_array_equal(
          env_state.num_moves, np.array([3], dtype=np.int32)
      )
    with self.subTest('last_reward'):
      np.testing.assert_array_equal(env_state.last_reward, np.array([-1.0]))
    with self.subTest('sum_rewards'):
      np.testing.assert_array_equal(env_state.sum_rewards, np.array([-3.0]))
    with self.subTest('effective_t_cost'):
      np.testing.assert_array_equal(env_state.effective_t_cost, np.array([3.0]))
    with self.subTest('is_terminal'):
      np.testing.assert_array_equal(env_state.is_terminal, np.array([True]))
    with self.subTest('init_tensor_index'):
      np.testing.assert_array_equal(
          env_state.init_tensor_index, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('change_of_basis'):
      np.testing.assert_array_equal(
          env_state.change_of_basis, np.eye(3, dtype=np.int32)[None]
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets, np.zeros((1, 10), dtype=np.bool_)
      )

  def test_step_exhausts_max_num_moves(self):
    max_num_moves = 10
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=max_num_moves,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    factor = jnp.array([1, 1, 1], dtype=jnp.int32)
    for _ in range(max_num_moves):
      # Apply the same action repeatedly.
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    np.testing.assert_array_equal(env_state.is_terminal, np.array([True]))

  def test_step_completes_toffoli_gadget(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # These seven factors form a Toffoli gadget.
    factors = jnp.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=jnp.int32)
    for factor in factors:
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets,
          np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool_)[None]
      )
    with self.subTest('rewards'):
      np.testing.assert_array_equal(env_state.last_reward, np.array([4.0]))
      np.testing.assert_array_equal(env_state.sum_rewards, np.array([-2.0]))
    with self.subTest('effective_t_cost'):
      np.testing.assert_array_equal(env_state.effective_t_cost, np.array([2.0]))

  def test_step_without_gadgets_enabled(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        ),
        use_gadgets=False
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # These seven factors form a Toffoli gadget, but they should not be
    # recognized as such, since gadgets are disabled.
    factors = jnp.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=jnp.int32)
    for factor in factors:
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(env_state.factors_in_gadgets, False)
    with self.subTest('rewards'):
      np.testing.assert_array_equal(env_state.last_reward, np.array([-1.0]))
      np.testing.assert_array_equal(env_state.sum_rewards, np.array([-7.0]))
    with self.subTest('effective_t_cost'):
      np.testing.assert_array_equal(env_state.effective_t_cost, np.array([7.0]))

  def test_step_completes_cs_gadget(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # These three factors form a CS gadget.
    factors = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=jnp.int32)
    for factor in factors:
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets,
          np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.bool_)[None]
      )
    with self.subTest('rewards'):
      np.testing.assert_array_equal(env_state.last_reward, np.array([0.0]))
      np.testing.assert_array_equal(env_state.sum_rewards, np.array([-2.0]))
    with self.subTest('effective_t_cost'):
      np.testing.assert_array_equal(env_state.effective_t_cost, np.array([2.0]))

  def test_split_reward_none_preserves_original_reward(self):
    config_default = _small_env_config()
    config_split_none = _small_env_config(
        split_reward=config_lib.SplitRewardParams(mode='none')
    )
    env_default = environment.Environment(jax.random.PRNGKey(0), config_default)
    env_split_none = environment.Environment(
        jax.random.PRNGKey(0), config_split_none
    )
    env_state_default = env_default.init_state(jax.random.PRNGKey(1)[None])
    env_state_split_none = env_split_none.init_state(jax.random.PRNGKey(1)[None])

    action = factors_utils.action_factor_to_index(
        jnp.array([1, 1, 1], dtype=jnp.int32)
    )[None]
    new_default = env_default.step(action, env_state_default)
    new_split_none = env_split_none.step(action, env_state_split_none)

    np.testing.assert_array_equal(
        new_split_none.last_reward, new_default.last_reward
    )
    np.testing.assert_array_equal(
        new_split_none.sum_rewards, new_default.sum_rewards
    )
    np.testing.assert_array_equal(
        new_split_none.split_last_reward, np.array([0.0])
    )

  def test_split_reward_factor_mixed_mass(self):
    config = _small_env_config(
        split_reward=config_lib.SplitRewardParams(
            mode='v1',
            partition_blocks_by_target=[[[0, 1, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    local = env._factor_mixed_mass(
        jnp.array([1, 0, 0], dtype=jnp.int32), jnp.array(0)
    )
    bridge = env._factor_mixed_mass(
        jnp.array([1, 1, 0], dtype=jnp.int32), jnp.array(0)
    )

    np.testing.assert_allclose(local, 0.0)
    self.assertGreater(float(bridge), 0.0)

  def test_split_reward_mixed_drop_matches_direct_calculation(self):
    split_reward = config_lib.SplitRewardParams(
        mode='mixed_drop',
        lambda_drop=1.0,
        drop_clip=10.0,
        partition_blocks_by_target=[[[0, 1, 1]]],
        partition_weights_by_target=[[1.0]],
    )
    config = _small_env_config(split_reward=split_reward)
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])
    factor = jnp.array([1, 1, 1], dtype=jnp.int32)

    old_mixed = env._mixed_level(env_state.tensor[0], jnp.array(0))
    new_tensor = factors_utils.rank_one_update_to_tensor(
        env_state.tensor[0], factor
    )
    new_mixed = env._mixed_level(new_tensor, jnp.array(0))
    expected_drop = old_mixed - new_mixed
    new_env_state = env.step(
        factors_utils.action_factor_to_index(factor)[None], env_state
    )

    np.testing.assert_allclose(new_env_state.split_last_reward[0], expected_drop)
    np.testing.assert_allclose(
        new_env_state.split_mixed_auc_sum[0], new_mixed
    )

  def test_split_reward_v2_progress_is_positive_only(self):
    split_reward = config_lib.SplitRewardParams(
        mode='v2_progress',
        lambda_drop=1.0,
        lambda_mass=1.0,
        drop_clip=10.0,
        partition_blocks_by_target=[[[0, 1, 1]]],
        partition_weights_by_target=[[1.0]],
    )
    config = _small_env_config(split_reward=split_reward)
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])

    bridge_bad = jnp.array([1, 0, 1], dtype=jnp.int32)
    bridge_good = jnp.array([1, 1, 0], dtype=jnp.int32)
    bad_state = env.step(
        factors_utils.action_factor_to_index(bridge_bad)[None], env_state
    )
    good_state = env.step(
        factors_utils.action_factor_to_index(bridge_good)[None], env_state
    )

    np.testing.assert_allclose(bad_state.split_last_reward[0], 0.0)
    self.assertGreater(float(good_state.split_last_reward[0]), 0.0)

  def test_split_reward_v3_frontier_tracks_total_residual_progress(self):
    split_reward = config_lib.SplitRewardParams(
        mode='v3_frontier',
        lambda_drop=0.0,
        lambda_mass=0.0,
        lambda_residual=1.0,
        drop_clip=10.0,
        partition_blocks_by_target=[[[0, 1, 1]]],
        partition_weights_by_target=[[1.0]],
    )
    config = _small_env_config(split_reward=split_reward)
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])

    improving_factor = jnp.array([1, 1, 0], dtype=jnp.int32)
    worsening_factor = jnp.array([1, 0, 1], dtype=jnp.int32)
    improving_state = env.step(
        factors_utils.action_factor_to_index(improving_factor)[None],
        env_state,
    )
    worsening_state = env.step(
        factors_utils.action_factor_to_index(worsening_factor)[None],
        env_state,
    )

    np.testing.assert_allclose(
        improving_state.split_last_reward[0],
        4.0 / 13.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        worsening_state.split_last_reward[0],
        -6.0 / 13.0,
        rtol=1e-6,
    )

  def test_split_reward_v4_sticky_frontier_penalizes_frontier_regret(self):
    split_reward = config_lib.SplitRewardParams(
        mode='v4_sticky_frontier',
        lambda_drop=0.0,
        lambda_mass=0.0,
        lambda_residual=1.0,
        lambda_frontier=1.0,
        drop_clip=10.0,
        partition_blocks_by_target=[[[0, 1, 1]]],
        partition_weights_by_target=[[1.0]],
    )
    config = _small_env_config(split_reward=split_reward)
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])

    improving_factor = jnp.array([1, 1, 0], dtype=jnp.int32)
    worsening_factor = jnp.array([1, 0, 1], dtype=jnp.int32)
    improving_state = env.step(
        factors_utils.action_factor_to_index(improving_factor)[None],
        env_state,
    )
    worsening_state = env.step(
        factors_utils.action_factor_to_index(worsening_factor)[None],
        env_state,
    )

    np.testing.assert_allclose(
        improving_state.split_last_reward[0],
        8.0 / 13.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        improving_state.frontier_residual_weight[0],
        9.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        worsening_state.split_last_reward[0],
        -12.0 / 13.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        worsening_state.frontier_residual_weight[0],
        13.0,
        rtol=1e-6,
    )

  def test_split_reward_v5_barrier_frontier_ignores_temporary_worsening(self):
    split_reward = config_lib.SplitRewardParams(
        mode='v5_barrier_frontier',
        lambda_drop=0.0,
        lambda_mass=0.0,
        lambda_frontier=1.0,
        drop_clip=10.0,
        partition_blocks_by_target=[[[0, 1, 1]]],
        partition_weights_by_target=[[1.0]],
    )
    config = _small_env_config(split_reward=split_reward)
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])

    improving_factor = jnp.array([1, 1, 0], dtype=jnp.int32)
    worsening_factor = jnp.array([1, 0, 1], dtype=jnp.int32)
    improving_state = env.step(
        factors_utils.action_factor_to_index(improving_factor)[None],
        env_state,
    )
    worsening_state = env.step(
        factors_utils.action_factor_to_index(worsening_factor)[None],
        env_state,
    )

    np.testing.assert_allclose(
        improving_state.split_last_reward[0],
        4.0 / 13.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        worsening_state.split_last_reward[0],
        0.0,
        rtol=1e-6,
    )

  def test_split_reward_zero_for_synthetic_demonstrations(self):
    config = _small_env_config(
        split_reward=config_lib.SplitRewardParams(
            mode='v1',
            partition_blocks_by_target=[[[0, 1, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    dem_config = config_lib.DemonstrationsParams(
        max_num_factors=10,
        max_num_gadgets=2,
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        3, dem_config, jax.random.PRNGKey(0)[None]
    )
    env = environment.Environment(jax.random.PRNGKey(1), config)
    env_state = env.init_state_from_demonstration(demonstration)
    new_env_state = env.step(
        factors_utils.action_factor_to_index(
            jnp.array([1, 1, 0], dtype=jnp.int32)
        )[None],
        env_state,
    )

    np.testing.assert_array_equal(
        new_env_state.split_last_reward, np.array([0.0])
    )

  def test_split_reward_zero_for_noncanonical_basis_when_guarded(self):
    config = _small_env_config(
        split_reward=config_lib.SplitRewardParams(
            mode='v1',
            canonical_basis_only=True,
            partition_blocks_by_target=[[[0, 1, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])
    nonidentity = jnp.array(
        [[[0, 1, 0], [1, 0, 0], [0, 0, 1]]], dtype=jnp.int32
    )
    env_state = env_state._replace(change_of_basis=nonidentity)
    new_env_state = env.step(
        factors_utils.action_factor_to_index(
            jnp.array([1, 1, 0], dtype=jnp.int32)
        )[None],
        env_state,
    )

    np.testing.assert_array_equal(
        new_env_state.split_last_reward, np.array([0.0])
    )

  def test_split_reward_guarded_applies_solved_terminal_over_budget_penalty(self):
    config = _small_env_config(
        split_reward=config_lib.SplitRewardParams(
            mode='v1_guarded',
            lambda_drop=0.0,
            lambda_auc=0.0,
            lambda_mass=0.0,
            lambda_budget=1.0,
            baseline_t_costs=[0.0],
            interim_budget_slack=0.0,
            partition_blocks_by_target=[[[0, 1, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])
    factor = jnp.array([1, 0, 0], dtype=jnp.int32)
    solvable_tensor = jnp.einsum('i,j,k->ijk', factor, factor, factor)[None]
    env_state = env_state._replace(
        tensor=solvable_tensor,
        num_moves=jnp.array([9], dtype=jnp.int32),
    )
    new_env_state = env.step(
        factors_utils.action_factor_to_index(factor)[None],
        env_state,
    )

    np.testing.assert_allclose(new_env_state.split_last_reward, np.array([-1.0]))
    np.testing.assert_allclose(new_env_state.last_reward, np.array([-2.0]))

  def test_split_reward_guard_allows_interim_gadget_buildup(self):
    config = _small_env_config(
        split_reward=config_lib.SplitRewardParams(
            mode='v1_guarded',
            lambda_drop=0.0,
            lambda_auc=0.0,
            lambda_mass=0.0,
            lambda_budget=1.0,
            baseline_t_costs=[2.0],
            interim_budget_slack=5.0,
            partition_blocks_by_target=[[[0, 1, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])

    for factor in jnp.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=jnp.int32,
    ):
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )

    np.testing.assert_allclose(env_state.effective_t_cost, np.array([3.0]))
    np.testing.assert_allclose(env_state.split_last_reward, np.array([0.0]))

  def test_split_reward_guard_does_not_budget_penalize_unsolved_terminal(self):
    config = _small_env_config(
        max_num_moves=10,
        split_reward=config_lib.SplitRewardParams(
            mode='v1_guarded',
            lambda_drop=0.0,
            lambda_auc=0.0,
            lambda_mass=0.0,
            lambda_budget=1.0,
            baseline_t_costs=[0.0],
            interim_budget_slack=0.0,
            partition_blocks_by_target=[[[0, 1, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])
    env_state = env_state._replace(num_moves=jnp.array([9], dtype=jnp.int32))
    new_env_state = env.step(
        factors_utils.action_factor_to_index(
            jnp.array([1, 0, 0], dtype=jnp.int32)
        )[None],
        env_state,
    )

    np.testing.assert_array_equal(new_env_state.is_terminal, np.array([True]))
    self.assertGreater(float(jnp.sum(new_env_state.tensor[0])), 0.0)
    np.testing.assert_allclose(new_env_state.split_last_reward, np.array([0.0]))

  def test_split_reward_tiebreak_is_terminal_only_and_clipped(self):
    config = _small_env_config(
        split_reward=config_lib.SplitRewardParams(
            mode='v1_tiebreak',
            lambda_auc=0.0,
            lambda_mass=10.0,
            baseline_t_costs=[2.0],
            terminal_tiebreak_clip=0.2,
            partition_blocks_by_target=[[[0, 1, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])

    nonterminal_state = env.step(
        factors_utils.action_factor_to_index(
            jnp.array([1, 1, 0], dtype=jnp.int32)
        )[None],
        env_state,
    )
    np.testing.assert_array_equal(
        nonterminal_state.is_terminal, np.array([False])
    )
    np.testing.assert_allclose(
        nonterminal_state.split_last_reward, np.array([0.0])
    )

    factor = jnp.array([1, 1, 0], dtype=jnp.int32)
    solvable_tensor = jnp.einsum('i,j,k->ijk', factor, factor, factor)[None]
    env_state = env_state._replace(tensor=solvable_tensor)
    terminal_state = env.step(
        factors_utils.action_factor_to_index(factor)[None],
        env_state,
    )

    np.testing.assert_array_equal(terminal_state.is_terminal, np.array([True]))
    np.testing.assert_allclose(
        terminal_state.split_last_reward, np.array([-0.2])
    )
    np.testing.assert_allclose(terminal_state.last_reward, np.array([-1.2]))

  def test_split_reward_tiebreak_does_not_pay_over_budget(self):
    config = _small_env_config(
        split_reward=config_lib.SplitRewardParams(
            mode='v1_tiebreak',
            lambda_auc=1.0,
            lambda_mass=1.0,
            baseline_t_costs=[0.0],
            terminal_tiebreak_clip=0.2,
            partition_blocks_by_target=[[[0, 1, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])
    factor = jnp.array([1, 1, 0], dtype=jnp.int32)
    solvable_tensor = jnp.einsum('i,j,k->ijk', factor, factor, factor)[None]
    env_state = env_state._replace(tensor=solvable_tensor)
    terminal_state = env.step(
        factors_utils.action_factor_to_index(factor)[None],
        env_state,
    )

    np.testing.assert_array_equal(terminal_state.is_terminal, np.array([True]))
    np.testing.assert_allclose(terminal_state.split_last_reward, np.array([0.0]))
    np.testing.assert_allclose(terminal_state.last_reward, np.array([-1.0]))

  def test_split_reward_local_gadget_has_zero_mixed_mass(self):
    config = _small_env_config(
        split_reward=config_lib.SplitRewardParams(
            mode='v1',
            lambda_drop=0.0,
            lambda_auc=0.0,
            lambda_mass=1.0,
            partition_blocks_by_target=[[[0, 0, 1]]],
            partition_weights_by_target=[[1.0]],
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])

    for factor in jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=jnp.int32):
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )

    np.testing.assert_allclose(env_state.split_sum_rewards, np.array([0.0]))
    np.testing.assert_allclose(env_state.split_mixed_mass_sum, np.array([0.0]))

  def test_get_observation(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=100,
        num_past_factors_to_observe=2,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # Apply an action.
    factor = jnp.array([1, 1, 1], dtype=jnp.int32)
    new_env_state = env.step(
        factors_utils.action_factor_to_index(factor)[None], env_state
    )

    # Get the observation.
    obs = env.get_observation(new_env_state)
    with self.subTest('tensor'):
      # The action of the factor [1, 1, 1] is to flip all the bits.
      np.testing.assert_array_equal(
          obs.tensor,
          1 - tensors.get_signature_tensor(
              tensors.CircuitType.SMALL_TCOUNT_3
          )[None]
      )
    with self.subTest('past_factors_as_planes'):
      np.testing.assert_array_equal(
          obs.past_factors_as_planes,
          np.concatenate(
              [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))], axis=1
          )
      )
    with self.subTest('sqrt_played_fraction'):
      # The played fraction is 1/100, so `sqrt_played_fraction` should be 0.1.
      np.testing.assert_allclose(
          obs.sqrt_played_fraction, np.array([0.1]), rtol=1e-6
      )

  def test_get_observation_with_factors_in_gadgets(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        num_past_factors_to_observe=7,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # Apply the actions that complete the Toffoli gadget.
    factors = jnp.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=jnp.int32)
    for factor in factors:
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    observations = env.get_observation(env_state)
    # All past factors are part of a gadget, so they should be masked out.
    np.testing.assert_array_equal(
        observations.past_factors_as_planes, np.zeros((1, 7, 3, 3))
    )


if __name__ == '__main__':
  absltest.main()
