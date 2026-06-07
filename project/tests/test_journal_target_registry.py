from __future__ import annotations

from scripts.run_shared_parity_study import STUDY_CASES


JOURNAL_FULL_ACTION_TARGETS = (
    "barenco_tof_4",
    "mod_mult_55",
    "cuccaro_adder_n4",
    "gf_2pow4_mult",
    "hamming_weight_n6",
    "hamming_weight_n7",
    "gf_2pow5_mult",
    "nc_tof_5",
    "cuccaro_adder_n5",
)


def test_journal_full_action_targets_are_registered() -> None:
    missing = [target for target in JOURNAL_FULL_ACTION_TARGETS if target not in STUDY_CASES]

    assert missing == []


def test_journal_target_cases_have_nonfallback_materialization_parameters() -> None:
    for target in JOURNAL_FULL_ACTION_TARGETS:
        case = STUDY_CASES[target]
        assert case.target == target
        assert case.max_action_weight > 0
        assert case.objective == "mixed-pair"
        assert case.factor_order
        assert case.target_strategy
