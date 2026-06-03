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

"""Environment for AlphaTensor-Quantum."""

import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt

from alphatensor_quantum.src import change_of_basis as change_of_basis_lib
from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import demonstrations
from alphatensor_quantum.src import factors
from alphatensor_quantum.src import tensors


class EnvState(NamedTuple):
  """State of the environment (or states, if considering batch dimensions).

  Attributes:
    tensor: The residual tensor.
    past_factors: The past played factors. Initially, `past_factors` contains
      all-zero factors, and as the game progresses, the factors are inserted
      from the back (i.e., in the last row), shifting the previous factors
      accordingly.
    num_moves: The current number of moves.
    last_reward: The immediate reward (corresponding to the last played action).
    sum_rewards: The sum of the rewards so far.
    is_terminal: Whether the current environment state is terminal (i.e., the
      game has ended), as a boolean.
    init_tensor_index: The index of the chosen initial tensor from
      `target_circuit_types` (useful for identifying which game is being
      played). If <0, the state does not correspond to any target tensor.
    change_of_basis: The change of basis matrix applied at the beginning of the
      game (or the identity matrix if no change of basis was applied).
    factors_in_gadgets: Which factors are part of a gadget of any type.
    effective_t_cost: The accumulated gadget-aware effective T-cost.
    split_last_reward: The splitting-only component of the last reward.
    split_sum_rewards: The accumulated splitting-only reward.
    split_mixed_auc_sum: The accumulated post-action mixed residual level.
    split_mixed_mass_sum: The accumulated mixed factor mass after gadget refund.
    frontier_residual_weight: The lowest total residual tensor weight observed
      so far in this episode.
  """
  tensor: jt.Integer[jt.Array, '*batch size size size']
  past_factors: jt.Integer[jt.Array, '*batch max_num_moves size']
  num_moves: jt.Integer[jt.Array, '*batch']
  last_reward: jt.Float[jt.Array, '*batch']
  sum_rewards: jt.Float[jt.Array, '*batch']
  is_terminal: jt.Bool[jt.Array, '*batch']
  init_tensor_index: jt.Integer[jt.Array, '*batch']
  change_of_basis: jt.Integer[jt.Array, '*batch size size']
  factors_in_gadgets: jt.Bool[jt.Array, '*batch max_num_moves']
  effective_t_cost: jt.Float[jt.Array, '*batch']
  split_last_reward: jt.Float[jt.Array, '*batch']
  split_sum_rewards: jt.Float[jt.Array, '*batch']
  split_mixed_auc_sum: jt.Float[jt.Array, '*batch']
  split_mixed_mass_sum: jt.Float[jt.Array, '*batch']
  frontier_residual_weight: jt.Float[jt.Array, '*batch']


class Observation(NamedTuple):
  """Observation to be passed to the network (possibly with batch dimensions).

  Attributes:
    tensor: The residual tensor.
    past_factors_as_planes: The outer products of the past played factors.
    sqrt_played_fraction: The square root of the ratio of played moves and the
      maximum number of allowed moves.
  """
  tensor: jt.Float[jt.Array, '*batch size size size']
  past_factors_as_planes: jt.Float[jt.Array, '*batch num_factors size size']
  sqrt_played_fraction: jt.Float[jt.Array, '*batch']


class Environment:
  """Environment for AlphaTensor-Quantum."""

  def __init__(self, rng: chex.PRNGKey, config: config_lib.EnvironmentParams):
    """Initializes the environment."""
    self._config = config
    self._split_config = self._config.split_reward
    self._split_enabled = self._split_config.mode != 'none'

    # Obtain the target signature tensors.
    unpadded_target_tensors = [
        tensors.get_signature_tensor(circuit_type)
        for circuit_type in self._config.target_circuit_types
    ]
    self._target_tensors = jnp.stack([
        tensors.zero_pad_tensor(tensor, self._config.max_tensor_size)
        for tensor in unpadded_target_tensors
    ], axis=0)  # Shape (num_target_tensors, size, size, size).
    self._target_tensor_weights = jnp.maximum(
        jnp.sum(self._target_tensors.astype(jnp.float32), axis=(1, 2, 3)),
        1.0,
    )
    self._init_split_reward_tables(unpadded_target_tensors)

    # Generate a set of change of basis matrices.
    self._change_of_basis = change_of_basis_lib.generate_change_of_basis(
        self._config.max_tensor_size,
        self._config.change_of_basis.prob_zero_entry,
        jax.random.split(
            rng, self._config.change_of_basis.num_change_of_basis_matrices
        )
    )

  @property
  def change_of_basis(self) -> jt.Integer[jt.Array, 'num_matrices size size']:
    return self._change_of_basis

  def _init_split_reward_tables(
      self,
      unpadded_target_tensors: list[jt.Integer[jt.Array, 'size size size']],
  ) -> None:
    """Builds static split-reward tables used by JAX reward code."""
    num_targets = len(unpadded_target_tensors)
    max_tensor_size = self._config.max_tensor_size
    if self._split_config.partition_blocks_by_target is None:
      target_sizes = [int(tensor.shape[0]) for tensor in unpadded_target_tensors]
      blocks_by_target = self._balanced_partition_blocks(
          target_sizes, max_tensor_size
      )
    else:
      blocks_by_target = self._normalize_partition_blocks(
          self._split_config.partition_blocks_by_target,
          num_targets,
          max_tensor_size,
      )

    num_partitions = len(blocks_by_target[0])
    if self._split_config.partition_weights_by_target is None:
      weights_by_target = [
          [1.0 / num_partitions] * num_partitions for _ in range(num_targets)
      ]
    else:
      weights_by_target = self._normalize_partition_weights(
          self._split_config.partition_weights_by_target,
          num_targets,
          num_partitions,
      )

    masks = []
    for target_blocks in blocks_by_target:
      target_masks = []
      for blocks in target_blocks:
        block_array = jnp.array(blocks)
        local_mask = jnp.logical_and(
            block_array[:, None, None] == block_array[None, :, None],
            block_array[:, None, None] == block_array[None, None, :],
        )
        target_masks.append(jnp.logical_not(local_mask).astype(jnp.float32))
      masks.append(jnp.stack(target_masks, axis=0))
    self._split_mixed_masks = jnp.stack(masks, axis=0)

    denoms = []
    for target_index, tensor in enumerate(self._target_tensors):
      counts = jnp.sum(
          tensor.astype(jnp.float32)[None] * self._split_mixed_masks[target_index],
          axis=(1, 2, 3),
      )
      denoms.append(jnp.maximum(counts, 1.0))
    self._split_denoms = jnp.stack(denoms, axis=0)
    self._split_partition_weights = jnp.array(weights_by_target, dtype=jnp.float32)

    if self._split_config.baseline_t_costs is None:
      baseline_t_costs = [jnp.inf] * num_targets
    else:
      baseline_t_costs = list(self._split_config.baseline_t_costs)
      if len(baseline_t_costs) != num_targets:
        raise ValueError(
            'baseline_t_costs must have one value per target tensor. Got '
            f'{len(baseline_t_costs)} for {num_targets} targets.'
        )
    self._split_baseline_t_costs = jnp.array(baseline_t_costs, dtype=jnp.float32)

  @staticmethod
  def _balanced_partition_blocks(
      target_sizes: list[int],
      max_tensor_size: int,
  ) -> list[list[list[int]]]:
    """Returns one balanced contiguous partition for each target."""
    blocks_by_target = []
    for target_size in target_sizes:
      split = max(1, (target_size + 1) // 2)
      blocks = [
          0 if index < split else 1 for index in range(max_tensor_size)
      ]
      blocks_by_target.append([blocks])
    return blocks_by_target

  @staticmethod
  def _normalize_partition_blocks(
      blocks_by_target,
      num_targets: int,
      max_tensor_size: int,
  ) -> list[list[list[int]]]:
    """Validates and pads configured partition blocks."""
    if len(blocks_by_target) != num_targets:
      raise ValueError(
          'partition_blocks_by_target must have one entry per target. Got '
          f'{len(blocks_by_target)} for {num_targets} targets.'
      )
    normalized = []
    num_partitions = None
    for target_blocks in blocks_by_target:
      if num_partitions is None:
        num_partitions = len(target_blocks)
      elif len(target_blocks) != num_partitions:
        raise ValueError('All targets must have the same number of partitions.')
      normalized_target = []
      for blocks in target_blocks:
        blocks = list(blocks)
        if len(blocks) > max_tensor_size:
          raise ValueError(
              'Partition block length cannot exceed max_tensor_size. Got '
              f'{len(blocks)} > {max_tensor_size}.'
          )
        if len(blocks) < max_tensor_size:
          blocks = blocks + [1] * (max_tensor_size - len(blocks))
        normalized_target.append(blocks)
      normalized.append(normalized_target)
    return normalized

  @staticmethod
  def _normalize_partition_weights(
      weights_by_target,
      num_targets: int,
      num_partitions: int,
  ) -> list[list[float]]:
    """Validates and normalizes configured partition weights."""
    if len(weights_by_target) != num_targets:
      raise ValueError(
          'partition_weights_by_target must have one entry per target. Got '
          f'{len(weights_by_target)} for {num_targets} targets.'
      )
    normalized = []
    for weights in weights_by_target:
      if len(weights) != num_partitions:
        raise ValueError(
            'partition_weights_by_target must match the partition count.'
        )
      total = float(sum(weights))
      if total <= 0.0:
        raise ValueError('Partition weights must have positive sum.')
      normalized.append([float(weight) / total for weight in weights])
    return normalized

  def _mixed_level(
      self,
      tensor: jt.Integer[jt.Array, 'size size size'],
      init_tensor_index: jt.Integer[jt.Scalar, ''],
  ) -> jt.Float[jt.Scalar, '']:
    """Returns the weighted normalized mixed weight of a tensor."""
    safe_index = jnp.maximum(init_tensor_index, 0)
    counts = jnp.sum(
        tensor.astype(jnp.float32)[None] * self._split_mixed_masks[safe_index],
        axis=(1, 2, 3),
    )
    return jnp.sum(
        self._split_partition_weights[safe_index]
        * counts
        / self._split_denoms[safe_index]
    )

  def _factor_mixed_mass(
      self,
      factor: jt.Integer[jt.Array, 'size'],
      init_tensor_index: jt.Integer[jt.Scalar, ''],
  ) -> jt.Float[jt.Scalar, '']:
    """Returns the weighted mixed mass of one rank-one factor."""
    rank_one_tensor = jnp.einsum('i,j,k->ijk', factor, factor, factor)
    return self._mixed_level(rank_one_tensor, init_tensor_index)

  def _group_mixed_mass(
      self,
      group: jt.Integer[jt.Array, 'num_factors size'],
      init_tensor_index: jt.Integer[jt.Scalar, ''],
  ) -> jt.Float[jt.Scalar, '']:
    """Returns the mixed mass of a GF(2) sum of rank-one factors."""
    rank_one_tensors = jnp.einsum('fi,fj,fk->fijk', group, group, group)
    group_tensor = jnp.mod(jnp.sum(rank_one_tensors, axis=0), 2)
    return self._mixed_level(group_tensor, init_tensor_index)

  def _group_factor_mixed_mass(
      self,
      group: jt.Integer[jt.Array, 'num_factors size'],
      init_tensor_index: jt.Integer[jt.Scalar, ''],
  ) -> jt.Float[jt.Scalar, '']:
    """Returns the sum of individual mixed masses in a group."""
    return jnp.sum(jax.vmap(
        lambda factor: self._factor_mixed_mass(factor, init_tensor_index)
    )(group))

  def _gadget_mixed_refund(
      self,
      new_past_factors: jt.Integer[jt.Array, 'max_num_moves size'],
      init_tensor_index: jt.Integer[jt.Scalar, ''],
      action_completed_toffoli_gadget: jt.Bool[jt.Scalar, ''],
      action_completed_cs_gadget: jt.Bool[jt.Scalar, ''],
  ) -> jt.Float[jt.Scalar, '']:
    """Refunds mixed mass when consecutive factors complete a gadget."""
    toffoli_group = new_past_factors[-7:]
    toffoli_refund = (
        self._group_factor_mixed_mass(toffoli_group, init_tensor_index)
        - self._group_mixed_mass(toffoli_group, init_tensor_index)
    )
    cs_group = new_past_factors[-3:]
    cs_refund = (
        self._group_factor_mixed_mass(cs_group, init_tensor_index)
        - self._group_mixed_mass(cs_group, init_tensor_index)
    )
    return jnp.maximum(
        jnp.where(
            action_completed_toffoli_gadget,
            toffoli_refund,
            jnp.where(action_completed_cs_gadget, cs_refund, 0.0),
        ),
        0.0,
    )

  def _split_reward_applicable(
      self,
      init_tensor_index: jt.Integer[jt.Scalar, ''],
      change_of_basis: jt.Integer[jt.Array, 'size size'],
  ) -> jt.Bool[jt.Scalar, '']:
    """Returns whether split reward should affect this state."""
    applicable = init_tensor_index >= 0
    if self._split_config.canonical_basis_only:
      identity = jnp.eye(self._config.max_tensor_size, dtype=jnp.int32)
      applicable = jnp.logical_and(
          applicable, jnp.all(change_of_basis == identity)
      )
    return applicable

  def _split_reward(
      self,
      old_tensor: jt.Integer[jt.Array, 'size size size'],
      new_tensor: jt.Integer[jt.Array, 'size size size'],
      factor: jt.Integer[jt.Array, 'size'],
      new_past_factors: jt.Integer[jt.Array, 'max_num_moves size'],
      init_tensor_index: jt.Integer[jt.Scalar, ''],
      change_of_basis: jt.Integer[jt.Array, 'size size'],
      action_completed_toffoli_gadget: jt.Bool[jt.Scalar, ''],
      action_completed_cs_gadget: jt.Bool[jt.Scalar, ''],
      new_effective_t_cost: jt.Float[jt.Scalar, ''],
      is_terminal: jt.Bool[jt.Scalar, ''],
      prior_mixed_auc_sum: jt.Float[jt.Scalar, ''],
      prior_mixed_mass_sum: jt.Float[jt.Scalar, ''],
      prior_frontier_residual_weight: jt.Float[jt.Scalar, ''],
  ) -> tuple[jt.Float[jt.Scalar, ''], jt.Float[jt.Scalar, ''],
             jt.Float[jt.Scalar, '']]:
    """Computes split reward, mixed AUC level, and net mixed mass."""
    if not self._split_enabled:
      return jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

    current_mixed = self._mixed_level(old_tensor, init_tensor_index)
    next_mixed = self._mixed_level(new_tensor, init_tensor_index)
    mixed_drop = current_mixed - next_mixed
    safe_index = jnp.maximum(init_tensor_index, 0)
    residual_drop = (
        jnp.sum(old_tensor.astype(jnp.float32))
        - jnp.sum(new_tensor.astype(jnp.float32))
    ) / self._target_tensor_weights[safe_index]
    next_residual_weight = jnp.sum(new_tensor.astype(jnp.float32))
    frontier_bonus = (
        jnp.maximum(prior_frontier_residual_weight - next_residual_weight, 0.0)
        / self._target_tensor_weights[safe_index]
    )
    frontier_regret = (
        jnp.maximum(next_residual_weight - prior_frontier_residual_weight, 0.0)
        / self._target_tensor_weights[safe_index]
    )
    factor_mass = self._factor_mixed_mass(factor, init_tensor_index)
    gadget_refund = self._gadget_mixed_refund(
        new_past_factors,
        init_tensor_index,
        action_completed_toffoli_gadget,
        action_completed_cs_gadget,
    )
    drop_reward = (
        self._split_config.lambda_drop
        * jnp.clip(
            mixed_drop,
            -self._split_config.drop_clip,
            self._split_config.drop_clip,
        )
    )
    residual_progress_reward = (
        self._split_config.lambda_residual
        * jnp.clip(
            residual_drop,
            -self._split_config.drop_clip,
            self._split_config.drop_clip,
        )
    )
    auc_penalty = self._split_config.lambda_auc * next_mixed
    mass_penalty = self._split_config.lambda_mass * factor_mass
    refund_reward = self._split_config.lambda_mass * gadget_refund
    net_mixed_mass = factor_mass - gadget_refund
    new_mixed_auc_sum = prior_mixed_auc_sum + next_mixed
    new_mixed_mass_sum = prior_mixed_mass_sum + net_mixed_mass
    solved_terminal = jnp.logical_and(is_terminal, jnp.all(new_tensor == 0))

    if self._split_config.mode == 'mixed_drop':
      split_reward = drop_reward
    elif self._split_config.mode == 'mixed_auc':
      split_reward = -auc_penalty
    elif self._split_config.mode == 'v1':
      split_reward = (
          drop_reward - auc_penalty - mass_penalty + refund_reward
      )
    elif self._split_config.mode == 'v2_progress':
      split_reward = (
          jnp.maximum(drop_reward, 0.0)
          + jnp.maximum(refund_reward, 0.0)
      )
    elif self._split_config.mode == 'v3_frontier':
      split_reward = (
          residual_progress_reward
          + jnp.maximum(drop_reward, 0.0)
          + jnp.maximum(refund_reward, 0.0)
      )
    elif self._split_config.mode == 'v4_sticky_frontier':
      split_reward = (
          residual_progress_reward
          + self._split_config.lambda_frontier
          * (frontier_bonus - frontier_regret)
          + jnp.maximum(drop_reward, 0.0)
          + jnp.maximum(refund_reward, 0.0)
      )
    elif self._split_config.mode == 'v5_barrier_frontier':
      split_reward = (
          self._split_config.lambda_frontier * frontier_bonus
          + jnp.maximum(drop_reward, 0.0)
          + jnp.maximum(refund_reward, 0.0)
      )
    elif self._split_config.mode == 'v1_guarded':
      target_budget = (
          self._split_baseline_t_costs[jnp.maximum(init_tensor_index, 0)]
          + self._split_config.t_guard_delta
      )
      active_budget = jnp.where(
          is_terminal,
          target_budget,
          target_budget + self._split_config.interim_budget_slack,
      )
      budget_penalty_applies = solved_terminal
      inside_budget = new_effective_t_cost <= active_budget
      over_budget = jnp.where(
          budget_penalty_applies,
          jnp.maximum(new_effective_t_cost - active_budget, 0.0),
          0.0,
      )
      positive_bonus = jnp.maximum(drop_reward, 0.0) + jnp.maximum(
          refund_reward, 0.0
      )
      always_on_terms = (
          jnp.minimum(drop_reward, 0.0) - auc_penalty - mass_penalty
      )
      split_reward = (
          jnp.where(inside_budget, positive_bonus, 0.0)
          + always_on_terms
          - self._split_config.lambda_budget * over_budget
      )
    elif self._split_config.mode == 'v1_tiebreak':
      target_budget = (
          self._split_baseline_t_costs[jnp.maximum(init_tensor_index, 0)]
          + self._split_config.t_guard_delta
      )
      inside_budget = new_effective_t_cost <= target_budget
      structural_penalty = (
          self._split_config.lambda_auc * new_mixed_auc_sum
          + self._split_config.lambda_mass
          * jnp.maximum(new_mixed_mass_sum, 0.0)
      )
      split_reward = jnp.where(
          jnp.logical_and(solved_terminal, inside_budget),
          -jnp.minimum(
              structural_penalty,
              self._split_config.terminal_tiebreak_clip,
          ),
          0.0,
      )
    else:
      split_reward = jnp.array(0.0)

    applicable = self._split_reward_applicable(
        init_tensor_index, change_of_basis
    )
    return (
        jnp.where(applicable, split_reward, 0.0),
        jnp.where(applicable, next_mixed, 0.0),
        jnp.where(applicable, net_mixed_mass, 0.0),
    )

  @functools.partial(jax.vmap, in_axes=(None, 0, 0))
  def step(
      self,
      action: jt.Integer[jt.Scalar, ''],
      env_state: EnvState
  ) -> EnvState:
    """Advances the environment state by applying the given action.

    Args:
      action: The action to apply, as an integer in {0, ..., num_actions - 1}.
      env_state: The current environment state.

    Returns:
      The new environment state.
    """
    factor = factors.action_index_to_factor(
        action, self._config.max_tensor_size
    )
    # Obtain the new environment state and past factors.
    new_tensor = factors.rank_one_update_to_tensor(env_state.tensor, factor)
    new_past_factors = jnp.concatenate(
        [env_state.past_factors[1:], factor[None]], axis=0
    )
    new_num_moves = env_state.num_moves + 1
    # The episode terminates when either we reach the all-zero tensor, or we
    # exceed the maximum number of moves.
    is_terminal = jnp.logical_or(
        jnp.all(new_tensor == 0), new_num_moves >= self._config.max_num_moves
    )
    # Determine whether the last action completed a gadget. Due to the specific
    # ordering of the actions defining the Toffoli and CS gadgets, at most one
    # of the two gadgets can be completed at any given step.
    action_completed_toffoli_gadget = jnp.logical_and(
        self._config.use_gadgets,
        jnp.logical_and(
            jnp.logical_and(
                new_num_moves >= 7,
                jnp.all(jnp.logical_not(env_state.factors_in_gadgets[-6:]))
            ),
            factors.factors_form_toffoli_gadget(new_past_factors[-7:])
        )
    )
    action_completed_cs_gadget = jnp.logical_and(
        self._config.use_gadgets,
        jnp.logical_and(
            jnp.logical_and(
                new_num_moves >= 3,
                jnp.all(jnp.logical_not(env_state.factors_in_gadgets[-2:]))
            ),
            factors.factors_form_cs_gadget(new_past_factors[-3:])
        )
    )
    new_factors_in_gadgets = jnp.concatenate([
        env_state.factors_in_gadgets[1:], jnp.zeros((1,), dtype=jnp.bool_)
    ], axis=0)
    new_factors_in_gadgets = new_factors_in_gadgets.at[-7:].set(
        jnp.where(
            action_completed_toffoli_gadget, True, new_factors_in_gadgets[-7:]
        )
    )
    new_factors_in_gadgets = new_factors_in_gadgets.at[-3:].set(
        jnp.where(action_completed_cs_gadget, True, new_factors_in_gadgets[-3:])
    )
    # In TensorGame, the reward is -1 per move, plus an additional penalty for
    # terminal games that exceeded the maximum number of moves without reaching
    # the all-zero tensor.
    reward = jnp.array(-1.0)
    reward -= jnp.where(is_terminal, jnp.sum(new_tensor), 0.0)
    # Adjust the reward if the last action completed a gadget.
    reward += jnp.where(
        action_completed_toffoli_gadget,
        # The 7 actions in the Toffoli gadget get a net reward of -2.0.
        factors.TOFFOLI_REWARD_SAVING,
        jnp.where(
            action_completed_cs_gadget, factors.CS_REWARD_SAVING, 0.0
        )  # The 3 actions in the CS gadget get a net reward of -2.0.
    )
    new_effective_t_cost = env_state.effective_t_cost + 1.0
    new_effective_t_cost -= jnp.where(
        action_completed_toffoli_gadget, factors.TOFFOLI_REWARD_SAVING, 0.0
    )
    new_effective_t_cost -= jnp.where(
        action_completed_cs_gadget, factors.CS_REWARD_SAVING, 0.0
    )
    split_reward, mixed_auc_level, net_mixed_mass = self._split_reward(
        old_tensor=env_state.tensor,
        new_tensor=new_tensor,
        factor=factor,
        new_past_factors=new_past_factors,
        init_tensor_index=env_state.init_tensor_index,
        change_of_basis=env_state.change_of_basis,
        action_completed_toffoli_gadget=action_completed_toffoli_gadget,
        action_completed_cs_gadget=action_completed_cs_gadget,
        new_effective_t_cost=new_effective_t_cost,
        is_terminal=is_terminal,
        prior_mixed_auc_sum=env_state.split_mixed_auc_sum,
        prior_mixed_mass_sum=env_state.split_mixed_mass_sum,
        prior_frontier_residual_weight=env_state.frontier_residual_weight,
    )
    new_frontier_residual_weight = jnp.minimum(
        env_state.frontier_residual_weight,
        jnp.sum(new_tensor.astype(jnp.float32)),
    )
    reward += split_reward
    return EnvState(
        tensor=new_tensor,
        past_factors=new_past_factors,
        num_moves=new_num_moves,
        is_terminal=is_terminal,
        last_reward=reward,
        sum_rewards=env_state.sum_rewards + reward,
        init_tensor_index=env_state.init_tensor_index,
        change_of_basis=env_state.change_of_basis,
        factors_in_gadgets=new_factors_in_gadgets,
        effective_t_cost=new_effective_t_cost,
        split_last_reward=split_reward,
        split_sum_rewards=env_state.split_sum_rewards + split_reward,
        split_mixed_auc_sum=(
            env_state.split_mixed_auc_sum + mixed_auc_level
        ),
        split_mixed_mass_sum=(
            env_state.split_mixed_mass_sum + net_mixed_mass
        ),
        frontier_residual_weight=new_frontier_residual_weight,
    )

  def _get_init_tensor(
      self, rng: chex.PRNGKey
  ) -> tuple[jt.Integer[jt.Array, 'size size size'],
             jt.Integer[jt.Scalar, '']]:
    """Returns a tensor from the set of target signature tensors.

    Args:
      rng: A Jax random key.

    Returns:
      A 2-tuple:
      - The target tensor, randomly chosen from the set of target signature
        tensors.
      - The index of that tensor in the set of target signature tensors.
    """
    num_target_tensors = len(self._config.target_circuit_types)
    tensor_index = jax.random.choice(
        rng,
        jnp.arange(num_target_tensors),
        p=(
            None if self._config.target_circuit_probabilities is None
            else jnp.array(self._config.target_circuit_probabilities)
        ),
    )
    return self._target_tensors[tensor_index], tensor_index

  def _apply_random_change_of_basis(
      self,
      tensor: jt.Integer[jt.Array, 'size size size'],
      rng: chex.PRNGKey
  ) -> tuple[jt.Integer[jt.Array, 'size size size'],
             jt.Integer[jt.Array, 'size size']]:
    """Applies a randomly chosen change of basis to the given tensor.

    Args:
      tensor: The tensor to apply the change of basis to.
      rng: A Jax random key.

    Returns:
      A 2-tuple:
      - The tensor after applying the change of basis.
      - The applied change of basis matrix.
    """
    rng_canonical, rng_cob = jax.random.split(rng)
    use_canonical_basis = jax.random.bernoulli(
        rng_canonical, self._config.change_of_basis.prob_canonical_basis
    )
    cob_matrix = jax.random.choice(rng_cob, self._change_of_basis)
    matrix = jnp.where(
        use_canonical_basis,
        jnp.eye(self._config.max_tensor_size, dtype=jnp.int32),
        cob_matrix
    )
    return change_of_basis_lib.apply_change_of_basis(tensor, matrix), matrix

  @functools.partial(jax.vmap, in_axes=(None, 0))
  def init_state(self, rng: chex.PRNGKey) -> EnvState:
    """Initializes and returns an environment state.

    Args:
      rng: A Jax random key.

    Returns:
      A new environment state.
    """
    rng_init_tensor, rng_cob = jax.random.split(rng)
    init_tensor, tensor_index = self._get_init_tensor(rng_init_tensor)
    init_tensor_cob, cob_matrix = self._apply_random_change_of_basis(
        init_tensor, rng_cob
    )
    init_residual_weight = jnp.sum(init_tensor_cob.astype(jnp.float32))
    return EnvState(
        tensor=init_tensor_cob,
        past_factors=jnp.zeros(
            (self._config.max_num_moves, self._config.max_tensor_size),
            dtype=jnp.int32
        ),
        num_moves=jnp.zeros((), dtype=jnp.int32),
        is_terminal=jnp.zeros((), dtype=jnp.bool_),
        last_reward=jnp.zeros(()),
        sum_rewards=jnp.zeros(()),
        init_tensor_index=tensor_index,
        change_of_basis=cob_matrix,
        factors_in_gadgets=jnp.zeros(
            (self._config.max_num_moves,), dtype=jnp.bool_
        ),
        effective_t_cost=jnp.zeros(()),
        split_last_reward=jnp.zeros(()),
        split_sum_rewards=jnp.zeros(()),
        split_mixed_auc_sum=jnp.zeros(()),
        split_mixed_mass_sum=jnp.zeros(()),
        frontier_residual_weight=init_residual_weight,
    )

  @functools.partial(jax.vmap, in_axes=(None, 0))
  def init_state_from_demonstration(
      self, demonstration: demonstrations.Demonstration
  ) -> EnvState:
    """Initializes an environment state from a demonstration.

    Args:
      demonstration: A synthetic demonstration.

    Returns:
      A newly initialized environment state.
    """
    return EnvState(
        tensor=demonstration.tensor,
        past_factors=jnp.zeros(
            (self._config.max_num_moves, self._config.max_tensor_size),
            dtype=jnp.int32
        ),
        num_moves=jnp.zeros((), dtype=jnp.int32),
        is_terminal=jnp.zeros((), dtype=jnp.bool_),
        last_reward=jnp.zeros(()),
        sum_rewards=jnp.zeros(()),
        init_tensor_index=-1,  # Dummy value to indicate that the state does not
                               # correspond to any target from
                               # `target_circuit_types`.
        change_of_basis=jnp.eye(self._config.max_tensor_size, dtype=jnp.int32),
        factors_in_gadgets=jnp.zeros(
            (self._config.max_num_moves,), dtype=jnp.bool_
        ),
        effective_t_cost=jnp.zeros(()),
        split_last_reward=jnp.zeros(()),
        split_sum_rewards=jnp.zeros(()),
        split_mixed_auc_sum=jnp.zeros(()),
        split_mixed_mass_sum=jnp.zeros(()),
        frontier_residual_weight=jnp.sum(demonstration.tensor.astype(jnp.float32)),
    )

  @functools.partial(jax.vmap, in_axes=(None, 0))
  def get_observation(self, env_state: EnvState) -> Observation:
    """Returns the observation that will be passed to the neural network.

    Args:
      env_state: The current environment state.

    Returns:
      The observation that will be passed to the neural network.
    """
    past_active_factors = env_state.past_factors[
        -self._config.num_past_factors_to_observe:
    ].astype(jnp.float_)  # (num_factors, size).
    past_factors_not_in_gadgets = jnp.expand_dims(jnp.logical_not(
        env_state.factors_in_gadgets[-self._config.num_past_factors_to_observe:]
    ), axis=(-1, -2))  # (num_factors, 1, 1).
    return Observation(
        tensor=env_state.tensor.astype(jnp.float_),
        past_factors_as_planes=past_factors_not_in_gadgets * jnp.einsum(
            'fu,fv->fuv', past_active_factors, past_active_factors
        ),
        sqrt_played_fraction=jnp.sqrt(
            env_state.num_moves / self._config.max_num_moves
        ),
    )
