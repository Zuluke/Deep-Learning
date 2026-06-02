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

"""Tensor utilities for AlphaTensor-Quantum."""

import enum
from pathlib import Path

import immutabledict

import jax.numpy as jnp
import jaxtyping as jt
import numpy as np


_CIRCUIT_TO_TENSOR_BENCHMARKS = (
    Path(__file__).resolve().parents[2] / 'circuit-to-tensor' / 'benchmarks'
)


def _load_benchmark_tensor(*relative_parts: str) -> np.ndarray:
  """Loads a circuit-to-tensor benchmark tensor as an int32 NumPy array."""
  return np.load(_CIRCUIT_TO_TENSOR_BENCHMARKS.joinpath(*relative_parts)).astype(
      np.int32
  )


_SMALL_TCOUNT_3 = np.array(
    [
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    ],
    dtype=np.int32,
)

_BARENCO_TOFF_3 = np.array(
    [
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

_MOD_5_4 = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        [
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

_NC_TOFF_3 = np.array(
    [
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    ],
    dtype=np.int32,
)

_GF_2POW2_MULT = _load_benchmark_tensor(
    'arithmetic', 'gf_2pow2_mult', 'gf_2pow2_mult.tensor.npy'
)
_QFT_4 = _load_benchmark_tensor('arithmetic', 'qft_4', 'qft_4.tensor.npy')
_HAMMING_WEIGHT_N4 = _load_benchmark_tensor(
    'applications', 'hamming_weight_n4', 'hamming_weight_n4.tensor.npy'
)
_HAMMING_WEIGHT_N5 = _load_benchmark_tensor(
    'applications', 'hamming_weight_n5', 'hamming_weight_n5.tensor.npy'
)


class CircuitType(enum.Enum):
  """Types of circuits."""
  # Some circuits taken from the "Benchmarks" section of the paper.
  BARENCO_TOFF_3 = 1
  MOD_5_4 = 2
  NC_TOFF_3 = 3
  # A small 3-qubit circuit with optimal T-count of 3, useful for testing.
  SMALL_TCOUNT_3 = 4
  # Local circuit-to-tensor benchmarks used for split-reward experiments.
  GF_2POW2_MULT = 5
  HAMMING_WEIGHT_N4 = 6
  HAMMING_WEIGHT_N5 = 7
  # Evaluation-only for the current full-action-space environment.
  QFT_4 = 8


_TENSORS_DICT = immutabledict.immutabledict({
    CircuitType.BARENCO_TOFF_3: _BARENCO_TOFF_3,
    CircuitType.MOD_5_4: _MOD_5_4,
    CircuitType.NC_TOFF_3: _NC_TOFF_3,
    CircuitType.SMALL_TCOUNT_3: _SMALL_TCOUNT_3,
    CircuitType.GF_2POW2_MULT: _GF_2POW2_MULT,
    CircuitType.HAMMING_WEIGHT_N4: _HAMMING_WEIGHT_N4,
    CircuitType.HAMMING_WEIGHT_N5: _HAMMING_WEIGHT_N5,
    CircuitType.QFT_4: _QFT_4,
})


def zero_pad_tensor(
    tensor: jt.Integer[jt.Array, 'size size size'],
    pad_to_size: int
) -> jt.Integer[jt.Array, '{pad_to_size} {pad_to_size} {pad_to_size}']:
  """Zero-pads the given tensor to the given size.

  Args:
    tensor: The tensor to pad.
    pad_to_size: The size to pad to. It must be at least as large as the tensor
      size.

  Returns:
    The padded tensor, such that the original tensor can be recovered by keeping
    the first `size` entries of each dimension.
  """
  size = tensor.shape[0]
  padding_width = pad_to_size - size
  return jnp.pad(tensor, (0, padding_width))


def get_signature_tensor(
    circuit_type: CircuitType
) -> jt.Integer[jt.Array, 'size size size']:
  """Returns the signature tensor for the given quantum circuit.

  Args:
    circuit_type: The circuit type.

  Returns:
    The (symmetric) target signature tensor, with entries in {0, 1}.
  """
  if circuit_type not in _TENSORS_DICT:
    raise ValueError(f'Unsupported circuit type: {circuit_type}')
  return jnp.array(_TENSORS_DICT[circuit_type])
