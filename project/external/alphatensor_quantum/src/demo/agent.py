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

"""Agent and states for the AlphaTensor-Quantum demo."""

import functools
from typing import NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jaxtyping as jt
import mctx
import numpy as np
import optax

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import demonstrations as demonstrations_lib
from alphatensor_quantum.src import environment
from alphatensor_quantum.src import networks
from alphatensor_quantum.src import tensors as tensors_lib
from alphatensor_quantum.src.demo import demo_config


class GameStats(NamedTuple):
  """Statistics of the played games.

  Attributes:
    num_games: The number of played games for each considered target. It
      includes a batch dimension.
    best_return: The best return (sum of rewards) for each considered target.
    best_effective_t_cost: The best solved gadget-aware effective T-cost for
      each considered target.
    best_return_effective_t_cost: The effective T-cost of the terminal episode
      with the best return, even when it did not solve the tensor exactly.
    best_return_num_moves: Number of moves in the terminal episode with the best
      return.
    best_return_residual_weight: Residual tensor weight of the terminal episode
      with the best return.
    best_return_factors: Factors of the terminal episode with the best return.
    best_return_change_of_basis: Change-of-basis matrix for the best-return
      episode.
    best_solved_num_moves: Number of moves in the best exactly solved episode.
    best_solved_factors: Factors of the best exactly solved episode.
    best_solved_change_of_basis: Change-of-basis matrix for the best solved
      episode.
    best_frontier_residual_weight: Lowest residual tensor weight observed at
      any point in acting episodes for each target.
    best_frontier_effective_t_cost: Effective T-cost at the best frontier state.
    best_frontier_num_moves: Number of raw moves at the best frontier state.
    best_frontier_factors: Factors of the best frontier state.
    best_frontier_change_of_basis: Change-of-basis matrix for the best frontier
      state.
    best_frontier_factors_in_gadgets: Gadget-membership mask for the best
      frontier state.
    best_frontier_tensor: Residual tensor of the best frontier state.
    best_frontier_sum_rewards: Accumulated reward at the best frontier state.
    best_frontier_split_sum_rewards: Accumulated split-only reward at the best
      frontier state.
    best_frontier_split_mixed_auc_sum: Accumulated mixed residual level at the
      best frontier state.
    best_frontier_split_mixed_mass_sum: Accumulated mixed factor mass at the
      best frontier state.
    avg_return: The average return (sum of rewards) for each considered target.
      Like `num_games`, `avg_return` includes a batch dimension; this is solely
      for convenience, as it makes it possible to filter out elements in the
      batch for which `num_games == 0` when computing the effective average
      return.
    avg_split_sum_rewards: Smoothed terminal splitting-only return.
    avg_split_mixed_auc_sum: Smoothed terminal mixed residual AUC diagnostic.
    avg_split_mixed_mass_sum: Smoothed terminal mixed mass diagnostic.
  """
  num_games: jt.Integer[jt.Array, 'batch_size num_target_tensors']
  best_return: jt.Float[jt.Array, 'num_target_tensors']
  best_effective_t_cost: jt.Float[jt.Array, 'num_target_tensors']
  best_return_effective_t_cost: jt.Float[jt.Array, 'num_target_tensors']
  best_return_num_moves: jt.Integer[jt.Array, 'num_target_tensors']
  best_return_residual_weight: jt.Float[jt.Array, 'num_target_tensors']
  best_return_factors: jt.Integer[
      jt.Array, 'num_target_tensors max_num_moves tensor_size'
  ]
  best_return_change_of_basis: jt.Integer[
      jt.Array, 'num_target_tensors tensor_size tensor_size'
  ]
  best_solved_num_moves: jt.Integer[jt.Array, 'num_target_tensors']
  best_solved_factors: jt.Integer[
      jt.Array, 'num_target_tensors max_num_moves tensor_size'
  ]
  best_solved_change_of_basis: jt.Integer[
      jt.Array, 'num_target_tensors tensor_size tensor_size'
  ]
  best_frontier_residual_weight: jt.Float[jt.Array, 'num_target_tensors']
  best_frontier_effective_t_cost: jt.Float[jt.Array, 'num_target_tensors']
  best_frontier_num_moves: jt.Integer[jt.Array, 'num_target_tensors']
  best_frontier_factors: jt.Integer[
      jt.Array, 'num_target_tensors max_num_moves tensor_size'
  ]
  best_frontier_change_of_basis: jt.Integer[
      jt.Array, 'num_target_tensors tensor_size tensor_size'
  ]
  best_frontier_factors_in_gadgets: jt.Bool[
      jt.Array, 'num_target_tensors max_num_moves'
  ]
  best_frontier_tensor: jt.Integer[
      jt.Array, 'num_target_tensors tensor_size tensor_size tensor_size'
  ]
  best_frontier_sum_rewards: jt.Float[jt.Array, 'num_target_tensors']
  best_frontier_split_sum_rewards: jt.Float[jt.Array, 'num_target_tensors']
  best_frontier_split_mixed_auc_sum: jt.Float[jt.Array, 'num_target_tensors']
  best_frontier_split_mixed_mass_sum: jt.Float[jt.Array, 'num_target_tensors']
  avg_return: jt.Float[jt.Array, 'batch_size num_target_tensors']
  avg_split_sum_rewards: jt.Float[jt.Array, 'batch_size num_target_tensors']
  avg_split_mixed_auc_sum: jt.Float[jt.Array, 'batch_size num_target_tensors']
  avg_split_mixed_mass_sum: jt.Float[jt.Array, 'batch_size num_target_tensors']


class RunState(NamedTuple):
  """The state of the experiment run.

  Attributes:
    params: The network parameters.
    env_states: The environment states.
    demonstrations: The current synthetic demonstrations.
    demonstrations_states: The environment states for the synthetic
      demonstrations.
    opt_state: The optimizer state.
    game_stats: The game statistics.
    rng: A Jax random key.
    training_step: The current optimizer step, stored in the JAX state so
      repeated evaluation windows do not recompile with different loop bounds.
  """
  params: chex.ArrayTree
  env_states: environment.EnvState
  demonstrations: demonstrations_lib.Demonstration
  demonstrations_states: environment.EnvState
  opt_state: optax.OptState
  game_stats: GameStats
  rng: chex.PRNGKey
  training_step: jt.Integer[jt.Scalar, '']


class NeuralNetwork(hk.Module):
  """Neural network with a simplified policy and value heads."""

  def __init__(
      self,
      num_actions: int,
      net_config: config_lib.NetworkParams,
      name: str = 'NeuralNetwork'
  ):
    """Initializes the module.

    Args:
      num_actions: The number of possible actions.
      net_config: The hyperparameters of the neural network.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._num_actions = num_actions
    self._torso = networks.TorsoNetwork(net_config)

  def __call__(
      self, observations: environment.Observation
  ) -> tuple[jt.Float[jt.Array, 'batch_size num_actions'],
             jt.Float[jt.Array, 'batch_size']]:
    """Applies the network.

    Args:
      observations: The (batched) observed environment state.

    Returns:
      A 2-tuple:
      - The policy logits.
      - The output of the value head.
    """
    embeddings = self._torso(observations)
    batch_size = embeddings.shape[0]
    reshaped_embeddings = jnp.reshape(embeddings, (batch_size, -1))
    outputs = hk.Linear(self._num_actions + 1)(reshaped_embeddings)
    return outputs[..., :-1], outputs[..., -1]


def _broadcast_shapes(
    x: jt.Shaped[jt.Array, 'batch_size'],
    y: jt.Shaped[jt.Array, 'batch_size ...'],
) -> jt.Shaped[jt.Array, 'batch_size ...']:
  """Broadcasts `x` to a shape compatible with `y`.

  Args:
    x: The array to be broadcasted.
    y: The array whose shape is used as a reference for broadcasting.

  Returns:
    The array `x` reshaped to (batch_size, 1, ..., 1) so that it has the same
    number of dimensions as `y`.
  """
  batch_size = y.shape[0]
  return jnp.reshape(x, [batch_size] + [1] * (len(y.shape) - 1))


class Agent:
  """Simplified version of an AlphaTensor-Quantum agent."""

  def __init__(self, config: demo_config.DemoConfig):
    """Initializes the agent.

    Args:
      config: The config hyperparameters for the demo.
    """
    self._env: environment.Environment  # Initialized in `init_run_state`.
    self._config = config

    self._full_num_actions = 2 ** config.env_config.max_tensor_size - 1
    self._action_indices = self._build_action_indices()
    self._num_actions = int(self._action_indices.shape[0])
    self._action_factors = self._build_action_factors()
    self._base_action_mask = self._build_base_action_mask()
    self._restricted_action_from_full = self._build_restricted_action_lookup()
    self._action_valid_by_target = self._build_action_valid_by_target()
    self._init_action_prior_tables()
    self._network = hk.transform(
        lambda obs: NeuralNetwork(self._num_actions, config.net_config)(obs)  # pylint: disable=unnecessary-lambda
    )
    # Inialize the optimizer.
    opt_scheduler = optax.exponential_decay(
        init_value=config.opt_config.init_lr,
        transition_steps=config.opt_config.lr_scheduler_transition_steps,
        decay_rate=config.opt_config.lr_scheduler_decay_factor,
        staircase=True,
    )
    self._opt = optax.chain(
        optax.adamw(
            learning_rate=opt_scheduler,
            weight_decay=config.opt_config.weight_decay
        ),
        optax.clip_by_global_norm(config.opt_config.clip_by_global_norm),
    )

  def _build_action_indices(self) -> jnp.ndarray:
    """Returns full environment action indices exposed to the agent."""
    dictionary = self._config.exp_config.action_dictionary
    if dictionary == "full":
      return jnp.arange(self._full_num_actions, dtype=jnp.int32)
    if dictionary not in ("low-weight", "tensor-overlap", "gadget-closure"):
      raise ValueError(
          f"Unknown action_dictionary {dictionary!r}. Expected 'full' or "
          "'low-weight' or 'tensor-overlap' or 'gadget-closure'."
      )
    max_weight = int(self._config.exp_config.max_action_weight)
    if max_weight <= 0:
      raise ValueError("max_action_weight must be positive.")
    action_indices = {
        action
        for action in range(self._full_num_actions)
        if int(action + 1).bit_count() <= max_weight
    }
    if dictionary == "tensor-overlap":
      action_indices.update(self._build_tensor_overlap_action_indices())
    if dictionary == "gadget-closure":
      closure_max_weight = int(
          self._config.exp_config.gadget_closure_max_weight
      )
      if closure_max_weight < max_weight:
        raise ValueError(
            "gadget_closure_max_weight must be at least max_action_weight."
        )
      action_indices.update({
          action
          for action in range(self._full_num_actions)
          if int(action + 1).bit_count() <= closure_max_weight
      })
    if not action_indices:
      raise ValueError("Restricted action dictionary cannot be empty.")
    return jnp.array(sorted(action_indices), dtype=jnp.int32)

  def _build_action_factors(self) -> jnp.ndarray:
    """Returns the exposed action factors as a dense lookup table."""
    action_indices = [int(action) for action in self._action_indices.tolist()]
    factors = [
        [
            ((action + 1) >> index) & 1
            for index in range(self._config.env_config.max_tensor_size)
        ]
        for action in action_indices
    ]
    return jnp.array(factors, dtype=jnp.int32)

  def _build_base_action_mask(self) -> jnp.ndarray:
    """Marks actions that are always valid in restricted dictionaries."""
    max_weight = int(self._config.exp_config.max_action_weight)
    return jnp.array(
        [
            int(action + 1).bit_count() <= max_weight
            for action in self._action_indices.tolist()
        ],
        dtype=jnp.bool_,
    )

  def _build_tensor_overlap_action_indices(self) -> set[int]:
    """Returns target-guided high-overlap action indices.

    This dictionary is AlphaQuantum-only: it scores candidate rank-one factors
    by how many non-zero entries of the target signature tensor lie inside the
    factor support. It does not use any external circuit or ZX metrics.
    """
    max_weight = int(self._config.exp_config.tensor_overlap_max_weight)
    per_target_limit = int(
        self._config.exp_config.tensor_overlap_max_actions_per_target
    )
    if max_weight <= 0:
      raise ValueError("tensor_overlap_max_weight must be positive.")
    if per_target_limit <= 0:
      raise ValueError("tensor_overlap_max_actions_per_target must be positive.")

    selected: set[int] = set()
    for target in self._config.env_config.target_circuit_types:
      tensor = np.asarray(tensors_lib.get_signature_tensor(target), dtype=np.int32)
      target_size = int(tensor.shape[0])
      candidates = []
      for action in range((1 << target_size) - 1):
        factor_bits = action + 1
        weight = factor_bits.bit_count()
        if weight > max_weight:
          continue
        support = [
            index for index in range(target_size)
            if (factor_bits >> index) & 1
        ]
        overlap = int(tensor[np.ix_(support, support, support)].sum())
        if overlap == 0:
          continue
        density = overlap / float(max(len(support) ** 3, 1))
        candidates.append((overlap, density, -weight, -action, action))
      candidates.sort(reverse=True)
      selected.update(
          action for *_score, action in candidates[:per_target_limit]
      )
    return selected

  def _build_restricted_action_lookup(self) -> jnp.ndarray:
    lookup = -jnp.ones((self._full_num_actions,), dtype=jnp.int32)
    restricted_indices = jnp.arange(self._num_actions, dtype=jnp.int32)
    return lookup.at[self._action_indices].set(restricted_indices)

  def _build_action_valid_by_target(self) -> jnp.ndarray:
    """Builds a target-indexed mask for actions touching only active indices."""
    if not self._config.exp_config.mask_padded_actions:
      return jnp.ones(
          (len(self._config.env_config.target_circuit_types), self._num_actions),
          dtype=jnp.bool_,
      )
    action_indices = [int(action) for action in self._action_indices.tolist()]
    rows = []
    for target in self._config.env_config.target_circuit_types:
      target_size = tensors_lib.get_signature_tensor(target).shape[0]
      rows.append([
          ((action + 1) >> target_size) == 0
          for action in action_indices
      ])
    return jnp.array(rows, dtype=jnp.bool_)

  def _init_action_prior_tables(self) -> None:
    """Builds static tensors used by the optional state-aware action prior."""
    factors_float = self._action_factors.astype(jnp.float32)
    self._action_tensors = jnp.einsum(
        'ai,aj,ak->aijk', factors_float, factors_float, factors_float
    )
    self._action_tensor_weights = jnp.sum(
        self._action_tensors, axis=(1, 2, 3)
    )
    self._action_factor_weights = jnp.sum(factors_float, axis=-1)

    target_tensors = []
    unpadded_target_tensors = []
    for target in self._config.env_config.target_circuit_types:
      tensor = tensors_lib.get_signature_tensor(target)
      unpadded_target_tensors.append(tensor)
      target_tensors.append(
          tensors_lib.zero_pad_tensor(
              tensor, self._config.env_config.max_tensor_size
          )
      )
    self._target_tensors = jnp.stack(target_tensors, axis=0).astype(jnp.float32)
    self._target_tensor_weights = jnp.maximum(
        jnp.sum(self._target_tensors, axis=(1, 2, 3)),
        1.0,
    )

    split_config = self._config.env_config.split_reward
    num_targets = len(unpadded_target_tensors)
    max_tensor_size = self._config.env_config.max_tensor_size
    if split_config.partition_blocks_by_target is None:
      target_sizes = [int(tensor.shape[0]) for tensor in unpadded_target_tensors]
      blocks_by_target = environment.Environment._balanced_partition_blocks(
          target_sizes, max_tensor_size
      )
    else:
      blocks_by_target = environment.Environment._normalize_partition_blocks(
          split_config.partition_blocks_by_target,
          num_targets,
          max_tensor_size,
      )

    num_partitions = len(blocks_by_target[0])
    if split_config.partition_weights_by_target is None:
      weights_by_target = [
          [1.0 / num_partitions] * num_partitions
          for _ in range(num_targets)
      ]
    else:
      weights_by_target = environment.Environment._normalize_partition_weights(
          split_config.partition_weights_by_target,
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
    self._prior_mixed_masks = jnp.stack(masks, axis=0)

    mixed_target_weights = jnp.einsum(
        'tijk,tpijk->tp', self._target_tensors, self._prior_mixed_masks
    )
    self._prior_split_denoms = jnp.maximum(mixed_target_weights, 1.0)
    self._prior_partition_weights = jnp.array(
        weights_by_target, dtype=jnp.float32
    )
    self._prior_action_mixed_weights = jnp.einsum(
        'aijk,tpijk->tap', self._action_tensors, self._prior_mixed_masks
    )
    self._prior_action_mixed_levels = jnp.sum(
        self._prior_partition_weights[:, None, :]
        * self._prior_action_mixed_weights
        / self._prior_split_denoms[:, None, :],
        axis=-1,
    )

  def _action_valid_mask(
      self,
      env_states: environment.EnvState,
  ) -> jt.Bool[jt.Array, 'batch_size num_actions']:
    """Returns the current action mask before it is applied to logits."""
    if not self._config.exp_config.mask_padded_actions:
      valid_actions = jnp.ones(
          (env_states.tensor.shape[0], self._num_actions),
          dtype=jnp.bool_,
      )
    else:
      target_indices = env_states.init_tensor_index
      safe_indices = jnp.maximum(target_indices, 0)
      valid_actions = self._action_valid_by_target[safe_indices]
      valid_actions = jnp.where(
          target_indices[:, None] >= 0,
          valid_actions,
          jnp.ones_like(valid_actions),
      )
    if self._config.exp_config.action_dictionary == "gadget-closure":
      valid_actions = jnp.logical_and(
          valid_actions,
          self._gadget_closure_valid_actions(env_states),
      )
    if self._config.exp_config.mask_repeated_actions:
      selected_before = jnp.any(
          jnp.all(
              env_states.past_factors[:, None, :, :]
              == self._action_factors[None, :, None, :],
              axis=-1,
          ),
          axis=-1,
      )
      valid_actions = jnp.logical_and(valid_actions, jnp.logical_not(selected_before))
    return valid_actions

  def _mask_padded_action_logits(
      self,
      policy_logits: jt.Float[jt.Array, 'batch_size num_actions'],
      env_states: environment.EnvState,
  ) -> jt.Float[jt.Array, 'batch_size num_actions']:
    """Masks actions disallowed by target size or dynamic gadget closure."""
    valid_actions = self._action_valid_mask(env_states)
    return jnp.where(valid_actions, policy_logits, -1.0e9)

  def _action_prior_active_mask(
      self,
      env_states: environment.EnvState,
  ) -> jt.Bool[jt.Array, 'batch_size']:
    """Returns states where the optional action prior should be active."""
    exp_config = self._config.exp_config
    active = jnp.logical_and(
        env_states.init_tensor_index >= 0,
        jnp.logical_not(env_states.is_terminal),
    )
    if exp_config.action_prior_canonical_only:
      identity = jnp.eye(
          self._config.env_config.max_tensor_size, dtype=jnp.int32
      )
      canonical_basis = jnp.all(
          env_states.change_of_basis == identity, axis=(-2, -1)
      )
      active = jnp.logical_and(active, canonical_basis)
    return active

  def _state_action_prior_logits(
      self,
      env_states: environment.EnvState,
  ) -> jt.Float[jt.Array, 'batch_size num_actions']:
    """Returns a state-aware tensor prior over exposed actions.

    The prior is AlphaQuantum-only: it scores the GF(2) effect of each
    candidate rank-one factor on the current residual tensor and, in split
    mode, on the current mixed residual defined by the configured partition.
    """
    exp_config = self._config.exp_config
    batch_size = env_states.tensor.shape[0]
    if exp_config.action_prior == "none":
      return jnp.zeros((batch_size, self._num_actions), dtype=jnp.float32)

    target_indices = env_states.init_tensor_index
    safe_indices = jnp.maximum(target_indices, 0)
    residual = env_states.tensor.astype(jnp.float32)
    overlaps = jnp.einsum(
        'bijk,aijk->ba', residual, self._action_tensors
    )
    residual_drop = 2.0 * overlaps - self._action_tensor_weights[None, :]
    residual_drop = residual_drop / self._target_tensor_weights[safe_indices, None]
    score = exp_config.action_prior_residual_weight * residual_drop

    if exp_config.action_prior == "split":
      masks = self._prior_mixed_masks[safe_indices]
      mixed_overlaps = jnp.einsum(
          'bijk,bpijk,aijk->bap',
          residual,
          masks,
          self._action_tensors,
      )
      action_mixed_weights = self._prior_action_mixed_weights[safe_indices]
      mixed_drop = 2.0 * mixed_overlaps - action_mixed_weights
      partition_weights = self._prior_partition_weights[safe_indices]
      split_denoms = self._prior_split_denoms[safe_indices]
      mixed_drop_level = jnp.sum(
          partition_weights[:, None, :] * mixed_drop / split_denoms[:, None, :],
          axis=-1,
      )
      current_mixed_counts = jnp.sum(
          residual[:, None, :, :, :] * masks,
          axis=(2, 3, 4),
      )
      current_mixed_level = jnp.sum(
          partition_weights * current_mixed_counts / split_denoms,
          axis=-1,
      )
      action_mixed_level = self._prior_action_mixed_levels[safe_indices]
      score = (
          score
          + exp_config.action_prior_mixed_drop_weight
          * current_mixed_level[:, None]
          * mixed_drop_level
          - exp_config.action_prior_mixed_mass_weight * action_mixed_level
      )

    score = (
        score
        - exp_config.action_prior_hamming_weight
        * self._action_factor_weights[None, :]
        / float(self._config.env_config.max_tensor_size)
    )
    if exp_config.action_dictionary == "gadget-closure":
      closure_valid = jnp.logical_and(
          self._gadget_closure_valid_actions(env_states),
          jnp.logical_not(self._base_action_mask)[None, :],
      )
      score = score + exp_config.action_prior_gadget_bonus * closure_valid

    valid_actions = self._action_valid_mask(env_states)
    score = jnp.where(valid_actions, score, 0.0)
    if exp_config.action_prior_standardize:
      valid_float = valid_actions.astype(jnp.float32)
      counts = jnp.maximum(jnp.sum(valid_float, axis=-1, keepdims=True), 1.0)
      mean = jnp.sum(score * valid_float, axis=-1, keepdims=True) / counts
      variance = (
          jnp.sum(jnp.square(score - mean) * valid_float, axis=-1, keepdims=True)
          / counts
      )
      score = jnp.where(
          valid_actions,
          (score - mean) / (jnp.sqrt(variance) + 1.0e-6),
          0.0,
      )

    active = self._action_prior_active_mask(env_states)
    return jnp.where(active[:, None], score, 0.0)

  def _apply_action_prior_to_logits(
      self,
      policy_logits: jt.Float[jt.Array, 'batch_size num_actions'],
      env_states: environment.EnvState,
  ) -> jt.Float[jt.Array, 'batch_size num_actions']:
    """Adds the optional state-aware prior before action masking."""
    exp_config = self._config.exp_config
    if exp_config.action_prior == "none" or exp_config.action_prior_beta == 0.0:
      return policy_logits
    return (
        policy_logits
        + exp_config.action_prior_beta * self._state_action_prior_logits(env_states)
    )

  def _apply_action_prior_top_k_mask(
      self,
      valid_actions: jt.Bool[jt.Array, 'batch_size num_actions'],
      prior_logits: jt.Float[jt.Array, 'batch_size num_actions'],
      env_states: environment.EnvState,
  ) -> jt.Bool[jt.Array, 'batch_size num_actions']:
    """Optionally narrows valid actions to the top-k prior-scored actions."""
    exp_config = self._config.exp_config
    top_k = min(int(exp_config.action_prior_top_k), self._num_actions)
    if (
        top_k <= 0
        or exp_config.action_prior == "none"
        or exp_config.action_prior_beta == 0.0
    ):
      return valid_actions

    masked_scores = jnp.where(valid_actions, prior_logits, -1.0e9)
    threshold = jnp.sort(masked_scores, axis=-1)[:, -top_k]
    top_k_actions = jnp.logical_and(
        valid_actions,
        masked_scores >= threshold[:, None],
    )
    valid_counts = jnp.sum(valid_actions, axis=-1)
    narrowed_actions = jnp.where(
        valid_counts[:, None] > top_k,
        top_k_actions,
        valid_actions,
    )
    active = self._action_prior_active_mask(env_states)
    return jnp.where(active[:, None], narrowed_actions, valid_actions)

  def _policy_logits_for_search(
      self,
      policy_logits: jt.Float[jt.Array, 'batch_size num_actions'],
      env_states: environment.EnvState,
  ) -> jt.Float[jt.Array, 'batch_size num_actions']:
    """Applies the optional prior and all action masks used by MCTS."""
    exp_config = self._config.exp_config
    valid_actions = self._action_valid_mask(env_states)
    if exp_config.action_prior == "none" or exp_config.action_prior_beta == 0.0:
      return jnp.where(valid_actions, policy_logits, -1.0e9)
    prior_logits = self._state_action_prior_logits(env_states)
    search_logits = policy_logits + exp_config.action_prior_beta * prior_logits
    valid_actions = self._apply_action_prior_top_k_mask(
        valid_actions,
        prior_logits,
        env_states,
    )
    return jnp.where(valid_actions, search_logits, -1.0e9)

  def _matches_action_factor(
      self,
      desired_factor: jt.Integer[jt.Array, 'batch_size size'],
      valid_prefix: jt.Bool[jt.Array, 'batch_size'],
  ) -> jt.Bool[jt.Array, 'batch_size num_actions']:
    """Returns action matches for a desired next factor."""
    factor_matches = jnp.all(
        self._action_factors[None, :, :] == desired_factor[:, None, :],
        axis=-1,
    )
    nonzero = jnp.any(desired_factor != 0, axis=-1)
    return jnp.logical_and(factor_matches, (valid_prefix & nonzero)[:, None])

  def _unused_recent_factors(
      self,
      env_states: environment.EnvState,
      count: int,
  ) -> jt.Bool[jt.Array, 'batch_size']:
    return jnp.logical_and(
        env_states.num_moves >= count,
        jnp.all(jnp.logical_not(env_states.factors_in_gadgets[:, -count:]), axis=1),
    )

  def _linearly_independent_batch(
      self,
      factor1: jt.Integer[jt.Array, 'batch_size size'],
      factor2: jt.Integer[jt.Array, 'batch_size size'],
      factor3: jt.Integer[jt.Array, 'batch_size size'],
  ) -> jt.Bool[jt.Array, 'batch_size']:
    distinct = jnp.logical_and(
        jnp.any(factor1 != factor2, axis=1),
        jnp.logical_and(
            jnp.any(factor1 != factor3, axis=1),
            jnp.any(factor2 != factor3, axis=1),
        ),
    )
    return jnp.logical_and(
        distinct,
        jnp.any(factor3 != jnp.mod(factor1 + factor2, 2), axis=1),
    )

  def _gadget_closure_valid_actions(
      self,
      env_states: environment.EnvState,
  ) -> jt.Bool[jt.Array, 'batch_size num_actions']:
    """Allows high-weight actions only when they continue a gadget prefix."""
    always_valid = jnp.broadcast_to(
        self._base_action_mask[None, :],
        (env_states.past_factors.shape[0], self._num_actions),
    )
    past = env_states.past_factors

    a2 = past[:, -2, :]
    b2 = past[:, -1, :]
    cs_prefix = jnp.logical_and(
        self._unused_recent_factors(env_states, 2),
        jnp.any(a2 != b2, axis=1),
    )
    cs_next = jnp.mod(a2 + b2, 2)
    cs_valid = self._matches_action_factor(cs_next, cs_prefix)

    a3 = past[:, -3, :]
    b3 = past[:, -2, :]
    c3 = past[:, -1, :]
    toffoli3_prefix = jnp.logical_and(
        self._unused_recent_factors(env_states, 3),
        self._linearly_independent_batch(a3, b3, c3),
    )
    toffoli3_valid = self._matches_action_factor(
        jnp.mod(a3 + b3, 2),
        toffoli3_prefix,
    )

    a4 = past[:, -4, :]
    b4 = past[:, -3, :]
    c4 = past[:, -2, :]
    ab4 = past[:, -1, :]
    toffoli4_prefix = jnp.logical_and(
        self._unused_recent_factors(env_states, 4),
        jnp.logical_and(
            self._linearly_independent_batch(a4, b4, c4),
            jnp.all(ab4 == jnp.mod(a4 + b4, 2), axis=1),
        ),
    )
    toffoli4_valid = self._matches_action_factor(
        jnp.mod(a4 + c4, 2),
        toffoli4_prefix,
    )

    a5 = past[:, -5, :]
    b5 = past[:, -4, :]
    c5 = past[:, -3, :]
    ab5 = past[:, -2, :]
    ac5 = past[:, -1, :]
    toffoli5_prefix = jnp.logical_and(
        self._unused_recent_factors(env_states, 5),
        jnp.logical_and(
            self._linearly_independent_batch(a5, b5, c5),
            jnp.logical_and(
                jnp.all(ab5 == jnp.mod(a5 + b5, 2), axis=1),
                jnp.all(ac5 == jnp.mod(a5 + c5, 2), axis=1),
            ),
        ),
    )
    toffoli5_valid = self._matches_action_factor(
        jnp.mod(a5 + b5 + c5, 2),
        toffoli5_prefix,
    )

    a6 = past[:, -6, :]
    b6 = past[:, -5, :]
    c6 = past[:, -4, :]
    ab6 = past[:, -3, :]
    ac6 = past[:, -2, :]
    abc6 = past[:, -1, :]
    toffoli6_prefix = jnp.logical_and(
        self._unused_recent_factors(env_states, 6),
        jnp.logical_and(
            self._linearly_independent_batch(a6, b6, c6),
            jnp.logical_and(
                jnp.all(ab6 == jnp.mod(a6 + b6, 2), axis=1),
                jnp.logical_and(
                    jnp.all(ac6 == jnp.mod(a6 + c6, 2), axis=1),
                    jnp.all(abc6 == jnp.mod(a6 + b6 + c6, 2), axis=1),
                ),
            ),
        ),
    )
    toffoli6_valid = self._matches_action_factor(
        jnp.mod(b6 + c6, 2),
        toffoli6_prefix,
    )

    return jnp.logical_or(
        always_valid,
        jnp.logical_or(
            cs_valid,
            jnp.logical_or(
                toffoli3_valid,
                jnp.logical_or(
                    toffoli4_valid,
                    jnp.logical_or(toffoli5_valid, toffoli6_valid),
                ),
            ),
        ),
    )

  def _to_full_actions(
      self, restricted_actions: jt.Integer[jt.Array, 'batch_size']
  ) -> jt.Integer[jt.Array, 'batch_size']:
    return self._action_indices[restricted_actions]

  def _to_restricted_actions(
      self, full_actions: jt.Integer[jt.Array, 'batch_size']
  ) -> jt.Integer[jt.Array, 'batch_size']:
    return self._restricted_action_from_full[full_actions]

  def init_run_state(self, rng: chex.PRNGKey) -> RunState:
    """Initializes the run state.

    Args:
      rng: A Jax random key.

    Returns:
      A run state.
    """
    (
        rng_env,
        rng_env_states,
        rng_demonstrations,
        rng_params,
        rng_run_state
    ) = jax.random.split(rng, num=5)

    # Initialize the environment, the environment states, the synthetic
    # demonstrations, and the network parameters.
    self._env = environment.Environment(rng_env, self._config.env_config)
    env_states = self._env.init_state(
        jax.random.split(rng_env_states, self._config.exp_config.batch_size)
    )
    demonstrations = demonstrations_lib.generate_synthetic_demonstrations(
        self._config.env_config.max_tensor_size,
        self._config.dem_config,
        jax.random.split(
            rng_demonstrations, num=self._config.exp_config.batch_size
        )
    )
    params = self._network.init(
        rng_params, self._env.get_observation(env_states)
    )
    # Initialize the game statistics.
    num_target_tensors = len(self._config.env_config.target_circuit_types)
    identity = jnp.eye(self._config.env_config.max_tensor_size, dtype=jnp.int32)
    initial_change_of_basis = jnp.broadcast_to(
        identity,
        (
            num_target_tensors,
            self._config.env_config.max_tensor_size,
            self._config.env_config.max_tensor_size,
        ),
    )
    game_stats = GameStats(
        num_games=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors,),
            dtype=jnp.int32
        ),
        best_return=jnp.array([-jnp.inf] * num_target_tensors),
        best_effective_t_cost=jnp.array([jnp.inf] * num_target_tensors),
        best_return_effective_t_cost=jnp.array([jnp.inf] * num_target_tensors),
        best_return_num_moves=jnp.zeros(
            (num_target_tensors,), dtype=jnp.int32
        ),
        best_return_residual_weight=jnp.array([jnp.inf] * num_target_tensors),
        best_return_factors=jnp.zeros(
            (
                num_target_tensors,
                self._config.env_config.max_num_moves,
                self._config.env_config.max_tensor_size,
            ),
            dtype=jnp.int32,
        ),
        best_return_change_of_basis=initial_change_of_basis,
        best_solved_num_moves=jnp.zeros(
            (num_target_tensors,), dtype=jnp.int32
        ),
        best_solved_factors=jnp.zeros(
            (
                num_target_tensors,
                self._config.env_config.max_num_moves,
                self._config.env_config.max_tensor_size,
            ),
            dtype=jnp.int32,
        ),
        best_solved_change_of_basis=initial_change_of_basis,
        best_frontier_residual_weight=jnp.array(
            [jnp.inf] * num_target_tensors
        ),
        best_frontier_effective_t_cost=jnp.array(
            [jnp.inf] * num_target_tensors
        ),
        best_frontier_num_moves=jnp.zeros(
            (num_target_tensors,), dtype=jnp.int32
        ),
        best_frontier_factors=jnp.zeros(
            (
                num_target_tensors,
                self._config.env_config.max_num_moves,
                self._config.env_config.max_tensor_size,
            ),
            dtype=jnp.int32,
        ),
        best_frontier_change_of_basis=initial_change_of_basis,
        best_frontier_factors_in_gadgets=jnp.zeros(
            (
                num_target_tensors,
                self._config.env_config.max_num_moves,
            ),
            dtype=jnp.bool_,
        ),
        best_frontier_tensor=jnp.zeros(
            (
                num_target_tensors,
                self._config.env_config.max_tensor_size,
                self._config.env_config.max_tensor_size,
                self._config.env_config.max_tensor_size,
            ),
            dtype=jnp.int32,
        ),
        best_frontier_sum_rewards=jnp.array([-jnp.inf] * num_target_tensors),
        best_frontier_split_sum_rewards=jnp.zeros((num_target_tensors,)),
        best_frontier_split_mixed_auc_sum=jnp.zeros((num_target_tensors,)),
        best_frontier_split_mixed_mass_sum=jnp.zeros((num_target_tensors,)),
        avg_return=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors)
        ),
        avg_split_sum_rewards=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors)
        ),
        avg_split_mixed_auc_sum=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors)
        ),
        avg_split_mixed_mass_sum=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors)
        ),
    )
    return RunState(
        params=params,
        env_states=env_states,
        demonstrations=demonstrations,
        demonstrations_states=self._env.init_state_from_demonstration(
            demonstrations
        ),
        opt_state=self._opt.init(params),
        game_stats=game_stats,
        rng=rng_run_state,
        training_step=jnp.array(0, dtype=jnp.int32),
    )

  def _recurrent_fn(
      self,
      params: chex.ArrayTree,
      rng: chex.PRNGKey,
      actions: jt.Integer[jt.Array, 'batch_size'],
      env_states: environment.EnvState
  ) -> tuple[mctx.RecurrentFnOutput, environment.EnvState]:
    """Implements the recurrent policy.

    In AlphaTensor-Quantum, the environment is deterministic, so there is no
    need for a recurrent function that captures the environment dynamics.
    Instead of a neural network that predicts some embeddings representing the
    environment state, we return the environment state itself.

    Args:
      params: The network parameters.
      rng: A Jax random key.
      actions: The batched action indices.
      env_states: The batched environment states.

    Returns:
      A 2-tuple:
      - The output of the recurrent function.
      - The new environment states.
    """
    env_states = self._env.step(self._to_full_actions(actions), env_states)
    observations = self._env.get_observation(env_states)
    policy_logits, values = self._network.apply(params, rng, observations)
    policy_logits = self._policy_logits_for_search(
        policy_logits, env_states
    )
    recurrent_fn_output = mctx.RecurrentFnOutput(
        prior_logits=policy_logits,
        value=values,
        reward=env_states.last_reward,
        discount=1.0 - env_states.is_terminal
    )
    return recurrent_fn_output, env_states

  def _loss_fn(
      self,
      params: chex.ArrayTree,
      global_step: int,
      acting_observations: environment.Observation,
      acting_policy_targets: jt.Float[jt.Array, 'batch_size num_actions'],
      acting_value_targets: jt.Float[jt.Array, 'batch_size'],
      demonstrations_observations: environment.Observation,
      demonstrations_policy_targets: jt.Float[jt.Array,
                                              'batch_size num_actions'],
      demonstrations_value_targets: jt.Float[jt.Array, 'batch_size'],
      rng: chex.PRNGKey,
  ) -> jt.Float[jt.Scalar, '']:
    """Obtains the loss.

    Args:
      params: The network parameters.
      global_step: The training step.
      acting_observations: The (batched) observed environment state.
      acting_policy_targets: The (batched) policy targets from the actors.
      acting_value_targets: The (batched) value targets from the actors.
      demonstrations_observations: The (batched) observed environment state from
        the synthetic demonstrations.
      demonstrations_policy_targets: The (batched) policy targets for the
        synthetic demonstrations.
      demonstrations_value_targets: The (batched) value targets for the
        synthetic demonstrations.
      rng: A Jax random key.

    Returns:
      The sum of the policy and value losses.
    """
    rng_acting, rng_demonstrations = jax.random.split(rng, num=2)

    # Loss corresponding to the episodes from acting.
    acting_policy_logits, acting_values = self._network.apply(
        params, rng_acting, acting_observations
    )
    acting_policy_logprobs = jax.nn.log_softmax(acting_policy_logits)
    acting_policy_loss = jnp.sum(acting_policy_targets * (
        jnp.log(acting_policy_targets) - acting_policy_logprobs
    ), axis=-1)
    acting_value_loss = jnp.square(acting_values - acting_value_targets)
    acting_loss = jnp.mean(acting_policy_loss + acting_value_loss)

    # Loss corresponding to the episodes from synthetic demonstrations.
    demonstrations_policy_logits, demonstrations_values = self._network.apply(
        params, rng_demonstrations, demonstrations_observations
    )
    demonstrations_policy_logprobs = jax.nn.log_softmax(
        demonstrations_policy_logits
    )
    demonstrations_policy_loss = -jnp.sum(
        demonstrations_policy_targets * demonstrations_policy_logprobs,
        axis=-1
    )
    demonstrations_value_loss = jnp.square(
        demonstrations_values - demonstrations_value_targets
    )
    demonstrations_loss = jnp.mean(
        demonstrations_policy_loss + demonstrations_value_loss
    )

    # Obtain the weight for the two terms in the loss.
    demonstrations_weight = optax.piecewise_constant_schedule(
        init_value=self._config.exp_config.loss.init_demonstrations_weight,
        boundaries_and_scales=(
            self._config.exp_config.loss.demonstrations_boundaries_and_scales
        )
    )(global_step)
    return (
        (1.0 - demonstrations_weight) * acting_loss
        + demonstrations_weight * demonstrations_loss
    )

  def _update_game_stats(
      self, run_state: RunState, new_env_states: environment.EnvState
  ) -> GameStats:
    """Returns the new game statistics."""
    is_terminal = new_env_states.is_terminal
    new_num_games_if_terminal = jax.vmap(
        lambda x, idx: x.at[idx].set(x[idx] + 1)
    )(run_state.game_stats.num_games, new_env_states.init_tensor_index)
    new_num_games = jnp.where(
        _broadcast_shapes(is_terminal, run_state.game_stats.num_games),
        new_num_games_if_terminal,
        run_state.game_stats.num_games
    )
    smoothing = self._config.exp_config.avg_return_smoothing
    new_avg_return_if_terminal = jax.vmap(
        lambda x, v, i: x.at[i].set(smoothing * x[i] + (1 - smoothing) * v)
    )(
        run_state.game_stats.avg_return,
        new_env_states.sum_rewards,
        new_env_states.init_tensor_index
    )
    new_avg_return = jnp.where(
        _broadcast_shapes(is_terminal, run_state.game_stats.avg_return),
        new_avg_return_if_terminal,
        run_state.game_stats.avg_return
    )
    new_avg_split_sum_rewards = self._smoothed_target_update(
        run_state.game_stats.avg_split_sum_rewards,
        new_env_states.split_sum_rewards,
        new_env_states.init_tensor_index,
        is_terminal,
        smoothing,
    )
    new_avg_split_mixed_auc_sum = self._smoothed_target_update(
        run_state.game_stats.avg_split_mixed_auc_sum,
        new_env_states.split_mixed_auc_sum,
        new_env_states.init_tensor_index,
        is_terminal,
        smoothing,
    )
    new_avg_split_mixed_mass_sum = self._smoothed_target_update(
        run_state.game_stats.avg_split_mixed_mass_sum,
        new_env_states.split_mixed_mass_sum,
        new_env_states.init_tensor_index,
        is_terminal,
        smoothing,
    )
    num_target_tensors = len(self._config.env_config.target_circuit_types)
    negative_inf = -jnp.inf * jnp.ones(
        (self._config.exp_config.batch_size, num_target_tensors)
    )
    new_best_return_if_terminal = jax.vmap(lambda x, v, i: x.at[i].set(v))(
        negative_inf,
        new_env_states.sum_rewards,
        new_env_states.init_tensor_index
    )
    terminal_return_candidates = jnp.where(
        _broadcast_shapes(is_terminal, new_best_return_if_terminal),
        new_best_return_if_terminal,
        negative_inf
    )
    candidate_best_return = jnp.max(terminal_return_candidates, axis=0)
    candidate_best_return_index = jnp.argmax(terminal_return_candidates, axis=0)
    improved_return = candidate_best_return > run_state.game_stats.best_return
    selected_return_factors = new_env_states.past_factors[
        candidate_best_return_index
    ]
    selected_return_change_of_basis = new_env_states.change_of_basis[
        candidate_best_return_index
    ]
    selected_return_residual_weight = jnp.sum(
        new_env_states.tensor[candidate_best_return_index],
        axis=(-3, -2, -1),
    )
    new_best_return = jnp.where(
        improved_return,
        candidate_best_return,
        run_state.game_stats.best_return,
    )
    new_best_return_effective_t_cost = jnp.where(
        improved_return,
        new_env_states.effective_t_cost[candidate_best_return_index],
        run_state.game_stats.best_return_effective_t_cost,
    )
    new_best_return_num_moves = jnp.where(
        improved_return,
        new_env_states.num_moves[candidate_best_return_index],
        run_state.game_stats.best_return_num_moves,
    )
    new_best_return_residual_weight = jnp.where(
        improved_return,
        selected_return_residual_weight,
        run_state.game_stats.best_return_residual_weight,
    )
    new_best_return_factors = jnp.where(
        improved_return[:, None, None],
        selected_return_factors,
        run_state.game_stats.best_return_factors,
    )
    new_best_return_change_of_basis = jnp.where(
        improved_return[:, None, None],
        selected_return_change_of_basis,
        run_state.game_stats.best_return_change_of_basis,
    )
    positive_inf = jnp.inf * jnp.ones(
        (self._config.exp_config.batch_size, num_target_tensors)
    )
    is_solved = jnp.logical_and(
        is_terminal, jnp.all(new_env_states.tensor == 0, axis=(-3, -2, -1))
    )
    new_best_t_cost_if_solved = jax.vmap(lambda x, v, i: x.at[i].set(v))(
        positive_inf,
        new_env_states.effective_t_cost,
        new_env_states.init_tensor_index
    )
    solved_t_cost_candidates = jnp.where(
        _broadcast_shapes(is_solved, new_best_t_cost_if_solved),
        new_best_t_cost_if_solved,
        positive_inf,
    )
    candidate_best_solved_t_cost = jnp.min(solved_t_cost_candidates, axis=0)
    candidate_best_solved_index = jnp.argmin(solved_t_cost_candidates, axis=0)
    improved_solved = (
        candidate_best_solved_t_cost < run_state.game_stats.best_effective_t_cost
    )
    new_best_effective_t_cost = jnp.minimum(
        run_state.game_stats.best_effective_t_cost,
        candidate_best_solved_t_cost,
    )
    new_best_solved_num_moves = jnp.where(
        improved_solved,
        new_env_states.num_moves[candidate_best_solved_index],
        run_state.game_stats.best_solved_num_moves,
    )
    new_best_solved_factors = jnp.where(
        improved_solved[:, None, None],
        new_env_states.past_factors[candidate_best_solved_index],
        run_state.game_stats.best_solved_factors,
    )
    new_best_solved_change_of_basis = jnp.where(
        improved_solved[:, None, None],
        new_env_states.change_of_basis[candidate_best_solved_index],
        run_state.game_stats.best_solved_change_of_basis,
    )
    state_residual_weight = jnp.sum(new_env_states.tensor, axis=(-3, -2, -1))
    frontier_filter_active = self._config.exp_config.frontier_replay_fraction > 0.0
    frontier_min_moves = jnp.where(
        frontier_filter_active,
        self._config.exp_config.frontier_replay_min_moves,
        0,
    )
    frontier_min_residual_drop = jnp.where(
        frontier_filter_active,
        self._config.exp_config.frontier_replay_min_residual_drop,
        0.0,
    )
    target_weights = self._target_tensor_weights[new_env_states.init_tensor_index]
    normalized_residual_drop = (
        target_weights - state_residual_weight
    ) / target_weights
    frontier_drop_satisfied = jnp.where(
        frontier_min_residual_drop > 0.0,
        normalized_residual_drop >= frontier_min_residual_drop,
        normalized_residual_drop > 0.0,
    )
    frontier_candidate_is_eligible = jnp.logical_and(
        new_env_states.num_moves >= frontier_min_moves,
        frontier_drop_satisfied,
    )
    frontier_score = (
        state_residual_weight * 1.0e6 + new_env_states.effective_t_cost
    )
    frontier_score = jnp.where(
        frontier_candidate_is_eligible,
        frontier_score,
        positive_inf[:, 0],
    )
    frontier_score_by_target = jax.vmap(lambda x, v, i: x.at[i].set(v))(
        positive_inf,
        frontier_score,
        new_env_states.init_tensor_index,
    )
    candidate_best_frontier_score = jnp.min(frontier_score_by_target, axis=0)
    candidate_best_frontier_index = jnp.argmin(frontier_score_by_target, axis=0)
    current_best_frontier_score = (
        run_state.game_stats.best_frontier_residual_weight * 1.0e6
        + run_state.game_stats.best_frontier_effective_t_cost
    )
    improved_frontier = candidate_best_frontier_score < current_best_frontier_score
    new_best_frontier_residual_weight = jnp.where(
        improved_frontier,
        state_residual_weight[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_residual_weight,
    )
    new_best_frontier_effective_t_cost = jnp.where(
        improved_frontier,
        new_env_states.effective_t_cost[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_effective_t_cost,
    )
    new_best_frontier_num_moves = jnp.where(
        improved_frontier,
        new_env_states.num_moves[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_num_moves,
    )
    new_best_frontier_factors = jnp.where(
        improved_frontier[:, None, None],
        new_env_states.past_factors[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_factors,
    )
    new_best_frontier_change_of_basis = jnp.where(
        improved_frontier[:, None, None],
        new_env_states.change_of_basis[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_change_of_basis,
    )
    new_best_frontier_factors_in_gadgets = jnp.where(
        improved_frontier[:, None],
        new_env_states.factors_in_gadgets[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_factors_in_gadgets,
    )
    new_best_frontier_tensor = jnp.where(
        improved_frontier[:, None, None, None],
        new_env_states.tensor[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_tensor,
    )
    new_best_frontier_sum_rewards = jnp.where(
        improved_frontier,
        new_env_states.sum_rewards[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_sum_rewards,
    )
    new_best_frontier_split_sum_rewards = jnp.where(
        improved_frontier,
        new_env_states.split_sum_rewards[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_split_sum_rewards,
    )
    new_best_frontier_split_mixed_auc_sum = jnp.where(
        improved_frontier,
        new_env_states.split_mixed_auc_sum[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_split_mixed_auc_sum,
    )
    new_best_frontier_split_mixed_mass_sum = jnp.where(
        improved_frontier,
        new_env_states.split_mixed_mass_sum[candidate_best_frontier_index],
        run_state.game_stats.best_frontier_split_mixed_mass_sum,
    )
    return GameStats(
        num_games=new_num_games,
        avg_return=new_avg_return,
        best_return=new_best_return,
        best_effective_t_cost=new_best_effective_t_cost,
        best_return_effective_t_cost=new_best_return_effective_t_cost,
        best_return_num_moves=new_best_return_num_moves,
        best_return_residual_weight=new_best_return_residual_weight,
        best_return_factors=new_best_return_factors,
        best_return_change_of_basis=new_best_return_change_of_basis,
        best_solved_num_moves=new_best_solved_num_moves,
        best_solved_factors=new_best_solved_factors,
        best_solved_change_of_basis=new_best_solved_change_of_basis,
        best_frontier_residual_weight=new_best_frontier_residual_weight,
        best_frontier_effective_t_cost=new_best_frontier_effective_t_cost,
        best_frontier_num_moves=new_best_frontier_num_moves,
        best_frontier_factors=new_best_frontier_factors,
        best_frontier_change_of_basis=new_best_frontier_change_of_basis,
        best_frontier_factors_in_gadgets=new_best_frontier_factors_in_gadgets,
        best_frontier_tensor=new_best_frontier_tensor,
        best_frontier_sum_rewards=new_best_frontier_sum_rewards,
        best_frontier_split_sum_rewards=new_best_frontier_split_sum_rewards,
        best_frontier_split_mixed_auc_sum=new_best_frontier_split_mixed_auc_sum,
        best_frontier_split_mixed_mass_sum=(
            new_best_frontier_split_mixed_mass_sum
        ),
        avg_split_sum_rewards=new_avg_split_sum_rewards,
        avg_split_mixed_auc_sum=new_avg_split_mixed_auc_sum,
        avg_split_mixed_mass_sum=new_avg_split_mixed_mass_sum,
    )

  def _smoothed_target_update(
      self,
      old_values: jt.Float[jt.Array, 'batch_size num_target_tensors'],
      terminal_values: jt.Float[jt.Array, 'batch_size'],
      target_indices: jt.Integer[jt.Array, 'batch_size'],
      is_terminal: jt.Bool[jt.Array, 'batch_size'],
      smoothing: float,
  ) -> jt.Float[jt.Array, 'batch_size num_target_tensors']:
    """Updates a per-target smoothed terminal statistic."""
    updated_if_terminal = jax.vmap(
        lambda x, v, i: x.at[i].set(smoothing * x[i] + (1 - smoothing) * v)
    )(old_values, terminal_values, target_indices)
    return jnp.where(
        _broadcast_shapes(is_terminal, old_values),
        updated_if_terminal,
        old_values
    )

  def _frontier_replay_states(
      self,
      fresh_states: environment.EnvState,
      game_stats: GameStats,
      rng: chex.PRNGKey,
  ) -> environment.EnvState:
    """Replaces some fresh restarts by stored best frontier states."""
    replay_fraction = self._config.exp_config.frontier_replay_fraction
    if replay_fraction <= 0.0:
      return fresh_states

    target_indices = fresh_states.init_tensor_index
    frontier_residual = game_stats.best_frontier_residual_weight[target_indices]
    frontier_num_moves = game_stats.best_frontier_num_moves[target_indices]
    target_weights = self._target_tensor_weights[target_indices]
    normalized_residual_drop = (
        target_weights - frontier_residual
    ) / target_weights
    mature_frontier = (
        frontier_num_moves >= self._config.exp_config.frontier_replay_min_moves
    )
    progressed_frontier = (
        normalized_residual_drop
        >= self._config.exp_config.frontier_replay_min_residual_drop
    )
    has_frontier = jnp.logical_and(
        jnp.isfinite(frontier_residual),
        jnp.logical_and(
            frontier_residual > 0.0,
            jnp.logical_and(
                jnp.logical_and(mature_frontier, progressed_frontier),
                frontier_num_moves < self._config.env_config.max_num_moves,
            ),
        ),
    )
    replay_draw = (
        jax.random.uniform(
            rng,
            shape=(self._config.exp_config.batch_size,),
        )
        < replay_fraction
    )
    use_frontier = jnp.logical_and(has_frontier, replay_draw)

    replay_states = environment.EnvState(
        tensor=game_stats.best_frontier_tensor[target_indices],
        past_factors=game_stats.best_frontier_factors[target_indices],
        num_moves=frontier_num_moves,
        last_reward=jnp.zeros_like(fresh_states.last_reward),
        sum_rewards=game_stats.best_frontier_sum_rewards[target_indices],
        is_terminal=jnp.zeros_like(fresh_states.is_terminal),
        init_tensor_index=target_indices,
        change_of_basis=(
            game_stats.best_frontier_change_of_basis[target_indices]
        ),
        factors_in_gadgets=(
            game_stats.best_frontier_factors_in_gadgets[target_indices]
        ),
        effective_t_cost=(
            game_stats.best_frontier_effective_t_cost[target_indices]
        ),
        split_last_reward=jnp.zeros_like(fresh_states.split_last_reward),
        split_sum_rewards=(
            game_stats.best_frontier_split_sum_rewards[target_indices]
        ),
        split_mixed_auc_sum=(
            game_stats.best_frontier_split_mixed_auc_sum[target_indices]
        ),
        split_mixed_mass_sum=(
            game_stats.best_frontier_split_mixed_mass_sum[target_indices]
        ),
        frontier_residual_weight=frontier_residual,
    )
    return jax.tree_util.tree_map(
        lambda replay, fresh: jnp.where(
            _broadcast_shapes(use_frontier, replay), replay, fresh
        ),
        replay_states,
        fresh_states,
    )

  def _update_demonstrations_and_states(
      self,
      demonstrations_actions: jt.Integer[jt.Array, 'batch_size'],
      run_state: RunState,
      rng: chex.PRNGKey
  ) -> tuple[demonstrations_lib.Demonstration, environment.EnvState]:
    """Updates the synthetic demonstrations and their states."""

    # Take a step for the environment states.
    new_demonstrations_states = self._env.step(
        demonstrations_actions, run_state.demonstrations_states
    )

    # Update the demonstrations if their corresponding episodes have terminated.
    new_demonstrations_if_terminal = (
        demonstrations_lib.generate_synthetic_demonstrations(
            self._config.env_config.max_tensor_size,
            self._config.dem_config,
            jax.random.split(rng, num=self._config.exp_config.batch_size),
        )
    )
    new_demonstrations = jax.tree_util.tree_map(
        lambda x, y: jnp.where(
            _broadcast_shapes(new_demonstrations_states.is_terminal, x), x, y
        ),
        new_demonstrations_if_terminal,
        run_state.demonstrations
    )

    # Update the demonstrations states for terminated episodes.
    new_demonstrations_states_if_terminal = (
        self._env.init_state_from_demonstration(new_demonstrations_if_terminal)
    )
    new_demonstrations_states = jax.tree_util.tree_map(
        lambda x, y: jnp.where(
            _broadcast_shapes(new_demonstrations_states.is_terminal, x), x, y
        ),
        new_demonstrations_states_if_terminal,
        new_demonstrations_states
    )
    return new_demonstrations, new_demonstrations_states

  def _run_iteration_agent_env_interaction(
      self, global_step: int, run_state: RunState
  ) -> RunState:
    """Runs one iteration of the agent-environment interaction loop.

    Args:
      global_step: The training step.
      run_state: The run state.

    Returns:
      The new run state.
    """
    rngs = jax.random.split(run_state.rng, num=8)

    acting_observations = self._env.get_observation(run_state.env_states)
    policy_logits, values = self._network.apply(
        run_state.params, rngs[0], acting_observations
    )
    policy_logits = self._policy_logits_for_search(
        policy_logits, run_state.env_states
    )
    root = mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=values,
        embedding=run_state.env_states,
    )
    policy_output = mctx.muzero_policy(
        params=run_state.params,
        rng_key=rngs[1],
        root=root,
        recurrent_fn=self._recurrent_fn,
        num_simulations=self._config.exp_config.num_mcts_simulations,
        qtransform=mctx.qtransform_by_parent_and_siblings,
    )
    search_value = policy_output.search_tree.node_values[
        :, policy_output.search_tree.ROOT_INDEX
    ]

    # Obtain the observations and the policy and value targets for the synthetic
    # demonstrations.
    demonstrations_observations = self._env.get_observation(
        run_state.demonstrations_states
    )
    (
        demonstrations_actions,
        demonstrations_value_targets
    ) = demonstrations_lib.get_action_and_value(
        run_state.demonstrations,
        run_state.demonstrations_states.num_moves,
    )

    # Compute the gradient of the loss and take a grad step.
    restricted_demonstrations_actions = self._to_restricted_actions(
        demonstrations_actions
    )
    demonstrations_policy_targets = jax.nn.one_hot(
        restricted_demonstrations_actions, self._num_actions
    )

    grads = jax.grad(self._loss_fn)(
        run_state.params,
        run_state.training_step,
        acting_observations,
        policy_output.action_weights,
        search_value,
        demonstrations_observations,
        demonstrations_policy_targets,
        demonstrations_value_targets,
        rngs[2]
    )
    updates, new_opt_state = self._opt.update(
        grads, run_state.opt_state, run_state.params
    )
    new_params = optax.apply_updates(run_state.params, updates)

    # Select next action probabilistically based on visit counts.
    actions = jax.vmap(
        lambda r, p: jax.random.choice(r, a=self._num_actions, p=p)
    )(
        jax.random.split(rngs[3], self._config.exp_config.batch_size),
        policy_output.action_weights
    )
    new_env_states = self._env.step(
        self._to_full_actions(actions), run_state.env_states
    )
    is_terminal = new_env_states.is_terminal

    # Update game statistics.
    new_game_stats = self._update_game_stats(run_state, new_env_states)

    # Reset the environment state if the episode has terminated. Optionally,
    # restart from a stored frontier for the sampled target.
    fresh_env_states = self._env.init_state(
        jax.random.split(rngs[4], num=self._config.exp_config.batch_size)
    )
    fresh_env_states = self._frontier_replay_states(
        fresh_env_states,
        new_game_stats,
        rngs[5],
    )
    new_env_states = jax.tree_util.tree_map(
        lambda x, y: jnp.where(_broadcast_shapes(is_terminal, x), x, y),
        fresh_env_states,
        new_env_states
    )

    # Reset the demonstrations and their states if the corresponding episodes
    # have terminated.
    (
        new_demonstrations, new_demonstrations_states
    ) = self._update_demonstrations_and_states(
        demonstrations_actions, run_state, rngs[6]
    )

    return RunState(
        params=new_params,
        env_states=new_env_states,
        demonstrations=new_demonstrations,
        demonstrations_states=new_demonstrations_states,
        opt_state=new_opt_state,
        game_stats=new_game_stats,
        rng=rngs[7],
        training_step=run_state.training_step + 1,
    )

  @functools.partial(jax.jit, static_argnums=(0,))
  def run_agent_env_interaction(
      self, global_step: int, run_state: RunState
  ) -> RunState:
    """Runs a few iterations of the agent-environment interaction loop.

    Args:
      global_step: Deprecated external step hint kept for API compatibility.
      run_state: The run state.

    Returns:
      The new run state, after running `eval_frequency_steps` tranining steps.
    """
    return jax.lax.fori_loop(
        lower=0,
        upper=self._config.exp_config.eval_frequency_steps,
        body_fun=self._run_iteration_agent_env_interaction,
        init_val=run_state,
    )
