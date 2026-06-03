from __future__ import annotations


def test_shared_parity_study_selects_named_cases() -> None:
    from scripts import run_shared_parity_study

    cases = run_shared_parity_study.selected_cases("mod_5_4,hamming_weight_n5")

    assert [case.target for case in cases] == ["mod_5_4", "hamming_weight_n5"]
    assert cases[0].factor_order == "greedy-cnot"
    assert cases[1].target_strategy == "min-change"


def test_shared_parity_study_presets_keep_core_default_small() -> None:
    from scripts import run_shared_parity_study

    core = run_shared_parity_study.selected_cases(None, preset="core")
    expanded = run_shared_parity_study.selected_cases(None, preset="expanded")

    assert [case.target for case in core] == list(run_shared_parity_study.CORE_TARGETS)
    assert len(expanded) > len(core)
    assert "cuccaro_adder_n3" in [case.target for case in expanded]


def test_shared_parity_study_output_paths_are_stable(tmp_path) -> None:
    from scripts import run_shared_parity_study

    case = run_shared_parity_study.STUDY_CASES["gf_2pow2_mult"]

    assert (
        run_shared_parity_study.case_output_dir(tmp_path, case).name
        == "gf_2pow2_mult_milp_span_mixed_pair_pairo6_wfull_greedy-cnot_max-change"
    )
    assert (
        run_shared_parity_study.optimization_output_dir(tmp_path, case).name
        == "gf_2pow2_mult_low-weight_w6_k175_mixed-pair"
    )
