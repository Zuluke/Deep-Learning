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

"""Configuration hyperparameters for the AlphaTensor-Quantum demo."""

import dataclasses

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import tensors


@dataclasses.dataclass(frozen=True, kw_only=True)
class LossParams:
  """Hyperparameters for the loss.

  Attributes:
    init_demonstrations_weight: The initial weight of the loss corresponding to
      the episodes from synthetic demonstrations.
    demonstrations_boundaries_and_scales: The boundaries and scales for the
      synthetic demonstrations weight, to be used in a
      `piecewise_constant_schedule` Optax schedule.
  """
  init_demonstrations_weight: float
  demonstrations_boundaries_and_scales: dict[int, float]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentParams:
  """Hyperparameters for the experiment.

  Attributes:
    batch_size: The batch size.
    num_mcts_simulations: The number of MCTS simulations to run per each action
      taken.
    num_training_steps: The total number of training steps.
    avg_return_smoothing: The smoothing factor for the average return, for
      reporting purposes only.
    eval_frequency_steps: The frequency (expressed in number of training steps)
      to report the running statistics. This is for reporting purposes only.
    action_dictionary: Which action dictionary the demo agent should use.
      "full" preserves the original action space. "low-weight" restricts MCTS
      and the policy head to factors with Hamming weight at most
      `max_action_weight`. "tensor-overlap" additionally includes a target
      tensor guided union of higher-weight factors. "gadget-closure" includes
      higher-weight linear combinations but masks them unless they continue or
      complete a CS/Toffoli gadget prefix already present in the state.
    max_action_weight: Maximum Hamming weight for the "low-weight" action
      dictionary.
    tensor_overlap_max_weight: Maximum Hamming weight considered for the
      tensor-overlap dictionary expansion.
    tensor_overlap_max_actions_per_target: Maximum number of target-guided
      extra actions added per target in tensor-overlap mode.
    gadget_closure_max_weight: Maximum Hamming weight exposed by
      gadget-closure mode before dynamic gadget-prefix masking.
    mask_padded_actions: Whether to mask actions that touch padded coordinates
      beyond the active target tensor size in multi-target runs.
    action_prior: Optional state-aware logit prior added before MCTS. "none"
      preserves the learned policy logits. "residual" scores actions by their
      immediate residual-weight drop. "split" additionally uses the configured
      tensor partition to reward mixed-residual progress and penalize mixed
      factor mass.
    action_prior_beta: Multiplicative scale applied to the standardized prior
      before it is added to policy logits.
    action_prior_residual_weight: Weight of the residual-drop component.
    action_prior_mixed_drop_weight: Weight of the mixed-residual-drop
      component in split mode.
    action_prior_mixed_mass_weight: Penalty weight for mixed mass introduced by
      the candidate rank-one factor.
    action_prior_hamming_weight: Small penalty on factor Hamming weight.
    action_prior_gadget_bonus: Bonus for currently valid high-weight gadget
      closure actions.
    action_prior_standardize: Whether to standardize prior scores over valid
      actions before adding them to policy logits.
    action_prior_top_k: Optional prior-guided action narrowing. Zero preserves
      the full currently valid action set; positive values keep only the best
      scored actions before MCTS search.
    action_prior_canonical_only: If true, disable the prior when the current
      change of basis is not the identity.
    frontier_replay_fraction: Fraction of terminated acting episodes to restart
      from the current best residual frontier for the newly sampled target. Zero
      preserves ordinary target-tensor restarts.
    frontier_replay_min_moves: Minimum number of moves a stored frontier must
      have before it can be used for frontier replay. This avoids over-replaying
      shallow prefixes that only make trivial residual progress.
    frontier_replay_min_residual_drop: Minimum normalized total-residual drop a
      stored frontier must have before it can be archived/replayed when frontier
      replay is active. For example, 0.04 means at least 4% residual reduction.
    loss: The loss parameters.
  """
  batch_size: int = 2_048
  num_mcts_simulations: int = 800
  num_training_steps: int = 1_000_000
  avg_return_smoothing: float = 0.9
  eval_frequency_steps: int = 1_000
  action_dictionary: str = "full"
  max_action_weight: int = 3
  tensor_overlap_max_weight: int = 5
  tensor_overlap_max_actions_per_target: int = 128
  gadget_closure_max_weight: int = 4
  mask_padded_actions: bool = False
  action_prior: str = "none"
  action_prior_beta: float = 1.0
  action_prior_residual_weight: float = 1.0
  action_prior_mixed_drop_weight: float = 1.0
  action_prior_mixed_mass_weight: float = 0.25
  action_prior_hamming_weight: float = 0.05
  action_prior_gadget_bonus: float = 0.25
  action_prior_standardize: bool = True
  action_prior_top_k: int = 0
  action_prior_canonical_only: bool = True
  frontier_replay_fraction: float = 0.0
  frontier_replay_min_moves: int = 0
  frontier_replay_min_residual_drop: float = 0.0
  loss: LossParams

  def __post_init__(self):
    if self.action_prior not in ("none", "residual", "split"):
      raise ValueError(
          f"Unknown action_prior {self.action_prior!r}. Expected 'none', "
          "'residual' or 'split'."
      )
    if self.action_prior_top_k < 0:
      raise ValueError("action_prior_top_k must be non-negative.")
    if not 0.0 <= self.frontier_replay_fraction <= 1.0:
      raise ValueError("frontier_replay_fraction must be between 0 and 1.")
    if self.frontier_replay_min_moves < 0:
      raise ValueError("frontier_replay_min_moves must be non-negative.")
    if self.frontier_replay_min_residual_drop < 0.0:
      raise ValueError(
          "frontier_replay_min_residual_drop must be non-negative."
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class DemoConfig:
  """All the hyperparameters for the demo."""
  exp_config: ExperimentParams
  env_config: config_lib.EnvironmentParams
  net_config: config_lib.NetworkParams
  opt_config: config_lib.OptimizerParams
  dem_config: config_lib.DemonstrationsParams


def get_demo_config(use_gadgets: bool) -> DemoConfig:
  """Returns the config hyperparameters for the demo.

  Args:
    use_gadgets: Whether to consider gadgetization. This parameter affects not
      only the environment, but also the default target circuits.

  Returns:
    The hyperparameters for the demo.
  """
  if use_gadgets:
    target_circuit_types = [
        # A tensor of size 5. The optimal decomposition has a single Toffoli
        # gadget, i.e., its equivalent T-count is 2.
        tensors.CircuitType.MOD_5_4,
    ]
  else:
    target_circuit_types = [
        # A tensor of size 5 and rank 7.
        tensors.CircuitType.MOD_5_4,
        # A tensor of size 8 and rank 13.
        tensors.CircuitType.BARENCO_TOFF_3,
        # A tensor of size 7 and rank 13.
        tensors.CircuitType.NC_TOFF_3,
    ]

  exp_config = ExperimentParams(
      batch_size=128,
      num_mcts_simulations=80,
      num_training_steps=50_000,
      eval_frequency_steps=50,
      loss=LossParams(
          init_demonstrations_weight=1.0,
          # Progressively reduce the weight of the demonstrations in favour of
          # the acting episodes.
          demonstrations_boundaries_and_scales={
              60: 0.99, 200: 0.5, 5_000: 0.2, 10_000: 0.1
          },
      ),
  )
  env_config = config_lib.EnvironmentParams(
      max_num_moves=30,
      target_circuit_types=target_circuit_types,
      num_past_factors_to_observe=6,
      change_of_basis=config_lib.ChangeOfBasisParams(
          prob_zero_entry=0.9,
          num_change_of_basis_matrices=80,
          prob_canonical_basis=0.16,
      ),
      use_gadgets=use_gadgets,
  )
  net_config = config_lib.NetworkParams(
      num_layers_torso=4,
      attention_params=config_lib.AttentionParams(
          num_heads=8,
          head_depth=8,
          mlp_widening_factor=2,
      ),
  )
  opt_config = config_lib.OptimizerParams(
      init_lr=1e-3,
      lr_scheduler_transition_steps=5_000,
  )
  dem_config = config_lib.DemonstrationsParams(
      max_num_factors=30,
      max_num_gadgets=5,
      prob_include_gadget=0.9 if use_gadgets else 0.0,
  )
  return DemoConfig(
      exp_config=exp_config,
      env_config=env_config,
      net_config=net_config,
      opt_config=opt_config,
      dem_config=dem_config,
  )
