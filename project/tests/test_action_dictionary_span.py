from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_action_dictionary_span import analyze_target
from scripts.analyze_action_dictionary_span import gf2_rank_and_membership


def test_gf2_rank_and_membership_detects_representable_target():
    columns = [
        np.array([1, 0, 1], dtype=np.uint8),
        np.array([0, 1, 1], dtype=np.uint8),
    ]
    target = np.array([1, 1, 0], dtype=np.uint8)

    rank, in_span = gf2_rank_and_membership(columns, target)

    assert rank == 2
    assert in_span


def test_gf2_rank_and_membership_rejects_out_of_span_target():
    columns = [
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([0, 1, 0], dtype=np.uint8),
    ]
    target = np.array([0, 0, 1], dtype=np.uint8)

    rank, in_span = gf2_rank_and_membership(columns, target)

    assert rank == 2
    assert not in_span


def test_hamming_loww3_is_first_viable_low_weight_dictionary():
    rows = analyze_target(
        "hamming_weight_n4",
        low_weight_values=[2, 3],
        tensor_overlap_limits=[],
        tensor_overlap_max_weight=5,
        tensor_overlap_base_weight=2,
    )
    by_dictionary = {row["dictionary"]: row for row in rows}

    assert not by_dictionary["loww2"]["target_in_span"]
    assert by_dictionary["loww3"]["target_in_span"]


def test_hamming_n5_tensor_overlap_needs_k175_for_span():
    rows = analyze_target(
        "hamming_weight_n5",
        low_weight_values=[],
        tensor_overlap_limits=[128, 175],
        tensor_overlap_max_weight=5,
        tensor_overlap_base_weight=2,
    )
    by_dictionary = {row["dictionary"]: row for row in rows}

    assert not by_dictionary["tensoroverlap_w5_k128_base2"]["target_in_span"]
    assert by_dictionary["tensoroverlap_w5_k175_base2"]["target_in_span"]
