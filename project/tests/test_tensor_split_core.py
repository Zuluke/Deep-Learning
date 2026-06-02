from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tensor_split_core import balanced_contiguous_partition
from scripts.tensor_split_core import canonicalize_factors
from scripts.tensor_split_core import gadget_aware_mixed_cost
from scripts.tensor_split_core import gadget_aware_mixed_stats
from scripts.tensor_split_core import group_multiset_gadgets
from scripts.tensor_split_core import is_bridge_factor
from scripts.tensor_split_core import load_factor_array
from scripts.tensor_split_core import mixed_auc
from scripts.tensor_split_core import mixed_auc_greedy
from scripts.tensor_split_core import mixed_weight
from scripts.tensor_split_core import outer3
from scripts.tensor_split_core import project_mixed_factor
from scripts.tensor_split_core import raw_bridge_count
from scripts.tensor_split_core import semantic_partitions_from_qasm
from scripts.tensor_split_core import singleton_bridge_count
from scripts.tensor_split_core import stable_factor_hash
from scripts.tensor_split_core import tensor_from_factors
from scripts.tensor_split_core import tensor_split_v3_stats


def test_project_mixed_factor_vanishes_for_local_factor() -> None:
    partition = [0, 0, 1, 1]
    local_factor = np.asarray([1, 1, 0, 0], dtype=np.uint8)

    mixed = project_mixed_factor(local_factor, partition)

    assert not np.any(mixed)
    assert not is_bridge_factor(local_factor, partition)


def test_project_mixed_factor_detects_bridge_factor() -> None:
    partition = [0, 0, 1, 1]
    bridge_factor = np.asarray([1, 0, 1, 0], dtype=np.uint8)

    mixed = project_mixed_factor(bridge_factor, partition)

    assert np.any(mixed)
    assert is_bridge_factor(bridge_factor, partition)


def test_mixed_auc_rewards_resolving_mixed_residual_early() -> None:
    partition = balanced_contiguous_partition(4)
    bridge_factor = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    local_factor = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    target = outer3(bridge_factor) ^ outer3(local_factor)

    early = np.stack([bridge_factor, local_factor], axis=0)
    late = np.stack([local_factor, bridge_factor], axis=0)

    assert mixed_weight(target ^ tensor_from_factors(early), partition) == 0
    assert mixed_auc(target, early, partition) < mixed_auc(target, late, partition)


def test_gadget_aware_mixed_cost_uses_effective_toffoli_cost() -> None:
    partition = [0, 0, 1]
    a = np.asarray([1, 0, 0], dtype=np.uint8)
    b = np.asarray([0, 1, 0], dtype=np.uint8)
    c = np.asarray([0, 0, 1], dtype=np.uint8)
    toffoli = np.stack(
        [
            a,
            b,
            c,
            (a + b) % 2,
            (a + c) % 2,
            (a + b + c) % 2,
            (b + c) % 2,
        ],
        axis=0,
    )

    assert raw_bridge_count(toffoli, partition) == 3
    assert gadget_aware_mixed_cost(toffoli, partition) == 2


def test_factor_loader_accepts_resynth_orientation(tmp_path: Path) -> None:
    factors = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    resynth_oriented = factors.T
    npy_path = tmp_path / "candidate.npy"
    np.save(npy_path, resynth_oriented)

    loaded = load_factor_array(npy_path, tensor_size=3)

    np.testing.assert_array_equal(loaded, factors)
    np.testing.assert_array_equal(canonicalize_factors(factors, tensor_size=3), factors)


def test_semantic_register_partition_separates_gf_inputs_from_output(tmp_path: Path) -> None:
    qasm_path = tmp_path / "gf.qasm"
    mapping_path = tmp_path / "gf.mapping.txt"
    qasm_path.write_text(
        "\n".join(
            [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                "qreg a[2];",
                "qreg b[2];",
                "qreg c[2];",
            ]
        ),
        encoding="utf-8",
    )
    mapping_path.write_text("[0, 1, 2, 3, 4, 5]", encoding="utf-8")

    partitions = semantic_partitions_from_qasm(
        circuit_id="gf_2pow2_mult",
        qasm_path=qasm_path,
        mapping_path=mapping_path,
        tensor_size=6,
    )
    register_partition = next(
        partition
        for partition in partitions
        if partition.partition_id == "semantic_register_k2"
    )

    assert register_partition.semantic_partition_status == "ok"
    assert register_partition.block_of.tolist() == [0, 0, 0, 0, 1, 1]


def test_semantic_hamming_partition_sends_extra_indices_to_work(tmp_path: Path) -> None:
    qasm_path = tmp_path / "hamming.qasm"
    mapping_path = tmp_path / "hamming.mapping.txt"
    qasm_path.write_text(
        "\n".join(["OPENQASM 2.0;", "qreg q[7];", 'include "qelib1.inc";']),
        encoding="utf-8",
    )
    mapping_path.write_text("[0, 1, 2, 3, 4, 5, 6, 7, 8]", encoding="utf-8")

    partitions = semantic_partitions_from_qasm(
        circuit_id="hamming_weight_n4",
        qasm_path=qasm_path,
        mapping_path=mapping_path,
        tensor_size=9,
    )
    role_partition = next(
        partition
        for partition in partitions
        if partition.partition_id == "semantic_role_k2"
    )

    assert role_partition.semantic_partition_status == "partial-mapping"
    assert role_partition.block_of.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_multiset_gadget_detector_finds_permuted_toffoli() -> None:
    a = np.asarray([1, 0, 0], dtype=np.uint8)
    b = np.asarray([0, 1, 0], dtype=np.uint8)
    c = np.asarray([0, 0, 1], dtype=np.uint8)
    toffoli = np.stack(
        [
            (a + b + c) % 2,
            a,
            (a + c) % 2,
            b,
            (b + c) % 2,
            c,
            (a + b) % 2,
        ],
        axis=0,
    )

    groups = group_multiset_gadgets(toffoli)

    assert len(groups) == 1
    assert groups[0].group_type == "toffoli"
    assert gadget_aware_mixed_cost(toffoli, [0, 0, 1]) == 2


def test_gadget_mixed_stats_separate_local_and_mixed_gadgets() -> None:
    local_a = np.asarray([1, 0, 0, 0], dtype=np.uint8)
    local_b = np.asarray([0, 1, 0, 0], dtype=np.uint8)
    local_cs = np.stack([local_a, local_b, (local_a + local_b) % 2], axis=0)
    mixed_a = np.asarray([1, 0, 0, 0], dtype=np.uint8)
    mixed_b = np.asarray([0, 0, 1, 0], dtype=np.uint8)
    mixed_cs = np.stack([mixed_a, mixed_b, (mixed_a + mixed_b) % 2], axis=0)

    local_stats = gadget_aware_mixed_stats(local_cs, [0, 0, 1, 1])
    mixed_stats = gadget_aware_mixed_stats(mixed_cs, [0, 0, 1, 1])

    assert local_stats["gadget_aware_effective_mixed_cost"] == 0
    assert local_stats["gadget_mixed_weight"] == 0
    assert mixed_stats["gadget_aware_effective_mixed_cost"] == 2
    assert mixed_stats["gadget_mixed_weight"] > 0


def test_v3_stats_normalize_by_target_mixed_weight() -> None:
    partition = [0, 1]
    bridge = np.asarray([1, 1], dtype=np.uint8)
    target = outer3(bridge)
    factors = np.stack([bridge], axis=0)

    stats = tensor_split_v3_stats(target, factors, partition)

    assert stats["target_mixed_weight"] == mixed_weight(target, partition)
    assert stats["gadget_mixed_weight_norm"] == 1.0
    assert stats["mixed_excess_norm"] == 0.0


def test_mixed_excess_norm_grows_with_redundant_mixed_mass() -> None:
    partition = [0, 1]
    bridge = np.asarray([1, 1], dtype=np.uint8)
    target = outer3(bridge)
    redundant = np.stack([bridge, bridge, bridge], axis=0)

    stats = tensor_split_v3_stats(target, redundant, partition)

    assert stats["target_mixed_weight"] == 6
    assert stats["gadget_mixed_weight_norm"] == 3.0
    assert stats["mixed_excess_norm"] == 2.0


def test_mixed_auc_greedy_is_invariant_to_factor_order() -> None:
    partition = balanced_contiguous_partition(4)
    bridge = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    local = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    target = outer3(bridge) ^ outer3(local)
    early = np.stack([bridge, local], axis=0)
    late = np.stack([local, bridge], axis=0)

    assert mixed_auc(target, early, partition) < mixed_auc(target, late, partition)
    assert mixed_auc_greedy(target, early, partition) == mixed_auc_greedy(
        target, late, partition
    )


def test_singleton_bridge_count_ignores_gadget_factors() -> None:
    partition = [0, 1]
    a = np.asarray([1, 0], dtype=np.uint8)
    b = np.asarray([0, 1], dtype=np.uint8)
    bridge = (a + b) % 2
    mixed_cs = np.stack([a, b, bridge], axis=0)
    with_singleton = np.concatenate([mixed_cs, bridge[None, :]], axis=0)

    assert singleton_bridge_count(mixed_cs, partition) == 0
    assert singleton_bridge_count(with_singleton, partition) == 1


def test_stable_factor_hash_is_reload_stable_and_content_sensitive(tmp_path: Path) -> None:
    factors = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    changed = np.asarray([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    npy_path = tmp_path / "factors.npy"
    np.save(npy_path, factors)

    reloaded = load_factor_array(npy_path, tensor_size=3)

    assert stable_factor_hash(factors) == stable_factor_hash(reloaded)
    assert stable_factor_hash(factors) != stable_factor_hash(changed)
