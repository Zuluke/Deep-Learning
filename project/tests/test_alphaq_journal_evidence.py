from __future__ import annotations

from scripts.analyze_alphaq_journal_evidence import OBJECTIVES
from scripts.analyze_alphaq_journal_evidence import evidence_rows
from scripts.analyze_alphaq_journal_evidence import external_effect_gate
from scripts.analyze_alphaq_journal_evidence import external_target_status
from scripts.analyze_alphaq_journal_evidence import next_battery_rows

NON_BASELINE_OBJECTIVES = tuple(
    objective for objective in OBJECTIVES if objective != "factor_count"
)


def test_evidence_rows_block_journal_when_external_and_formal_are_insufficient() -> None:
    gates = {
        row["gate"]: row
        for row in evidence_rows(
            split_rows=split_rows(),
            dataset_rows=dataset_rows(train_ready_groups=13, external_groups=5),
            external_rows=external_rows(complete=2, partial=True),
            readiness_rows=[],
            verification_rows=verification_rows(equal=8, inconclusive=2),
        )
    }

    assert gates["selector_loto"]["status"] == "pass"
    assert gates["dataset_scale_and_label_diversity"]["status"] == "partial"
    assert gates["external_generalization_coverage"]["status"] == "partial"
    assert gates["formal_verification_coverage"]["status"] == "partial"
    assert gates["overall_journal_readiness"]["status"] == "not-yet-journal-ready"


def test_evidence_rows_pass_journal_when_all_core_gates_pass() -> None:
    gates = {
        row["gate"]: row
        for row in evidence_rows(
            split_rows=split_rows(),
            dataset_rows=dataset_rows(train_ready_groups=30, external_groups=10, families=("arith", "app", "logic")),
            external_rows=external_rows(complete=10, partial=False, nonbaseline_improvements=3),
            readiness_rows=[],
            verification_rows=verification_rows(equal=24, inconclusive=0),
        )
    }

    assert gates["dataset_scale_and_label_diversity"]["status"] == "pass"
    assert gates["external_generalization_coverage"]["status"] == "pass"
    assert gates["external_nonbaseline_effect"]["status"] == "pass"
    assert gates["formal_verification_coverage"]["status"] == "pass"
    assert gates["overall_journal_readiness"]["status"] == "journal-ready"


def test_external_target_status_distinguishes_complete_partial_and_failed() -> None:
    rows = (
        [make_external("a", objective, "ok") for objective in OBJECTIVES]
        + [make_external("b", "factor_count", "ok")]
        + [
            make_external("b", objective, "failed")
            for objective in NON_BASELINE_OBJECTIVES
        ]
        + [make_external("c", objective, "failed") for objective in OBJECTIVES]
    )

    assert external_target_status(rows) == {"a": "complete", "b": "partial", "c": "failed"}


def test_external_target_status_accepts_consolidated_best_schema() -> None:
    rows = [
        {"target": "a", "objective_variant": objective, "best_status": "ok"}
        for objective in OBJECTIVES
    ] + [
        {"target": "b", "objective_variant": "factor_count", "best_status": "ok"},
        {"target": "b", "objective_variant": "factor_count_pair_cap", "best_status": "failed"},
        {"target": "b", "objective_variant": "mixed_pair", "best_status": "missing"},
    ]

    assert external_target_status(rows) == {"a": "complete", "b": "partial"}


def test_external_effect_gate_accepts_consolidated_best_schema() -> None:
    rows = [
        {"target": "a", "objective_variant": "factor_count", "best_status": "ok", "best_tcount": "10", "best_beam_qasm_depth": "50"},
        {"target": "a", "objective_variant": "factor_count_pair_cap", "best_status": "ok", "best_tcount": "8", "best_beam_qasm_depth": "45"},
        {"target": "a", "objective_variant": "mixed_pair", "best_status": "ok", "best_tcount": "9", "best_beam_qasm_depth": "40"},
    ]

    gate = external_effect_gate(rows)

    assert gate["status"] == "partial"
    assert "a:factor_count_pair_cap" in gate["evidence"]


def test_next_battery_prioritizes_repair_then_full_action_expansion() -> None:
    readiness = [
        make_readiness("validated", "ready-full-action", priority=110),
        make_readiness("partial", "ready-full-action", priority=120),
        make_readiness("new_small", "needs-tensor-v3-screen", priority=125),
        make_readiness("restricted", "ready-restricted-action", priority=80, action="restricted-action-recommended"),
        make_readiness("control", "current-grid-control", priority=200),
    ]
    external = [
        make_external("validated", objective, "ok") for objective in OBJECTIVES
    ] + [
        make_external("partial", "factor_count", "ok"),
        make_external("partial", "factor_count_pair_cap", "failed"),
        make_external("partial", "mixed_pair", "ok"),
    ]

    rows = next_battery_rows(readiness, external)

    assert rows[0]["target"] == "partial"
    assert rows[0]["recommended_stage"] == "full-action-repair"
    assert rows[1]["target"] == "new_small"
    assert rows[1]["recommended_stage"] == "tensor-v3-screen"
    assert rows[-1]["target"] == "restricted"


def split_rows() -> list[dict[str, str]]:
    return [
        {
            "policy": "baseline_factor_count",
            "ok_groups": "13",
            "exact_oracle_matches": "6",
            "tcount_wins_vs_baseline": "0",
            "qasm_nonworse_vs_baseline": "13",
            "median_qasm_ratio_vs_baseline": "1.0",
        },
        {
            "policy": "split_select_linear_alphaq_decomp",
            "ok_groups": "13",
            "exact_oracle_matches": "9",
            "tcount_wins_vs_baseline": "4",
            "qasm_nonworse_vs_baseline": "12",
            "median_qasm_ratio_vs_baseline": "1.0",
        },
    ]


def dataset_rows(
    *,
    train_ready_groups: int,
    external_groups: int,
    families: tuple[str, ...] = ("arith",),
) -> list[dict[str, str]]:
    rows = []
    labels = ("factor_count", "factor_count_pair_cap", "mixed_pair")
    for index in range(train_ready_groups):
        source = "external_eval" if index < external_groups else "internal"
        family = families[index % len(families)]
        rows.append(
            {
                "source_split": source,
                "target": f"target_{index}",
                "family": family,
                "train_ready": "True",
                "oracle_objective": labels[index % len(labels)],
            }
        )
    return rows


def external_rows(
    *,
    complete: int,
    partial: bool,
    nonbaseline_improvements: int = 1,
) -> list[dict[str, str]]:
    rows = []
    for index in range(complete):
        target = f"ext_{index}"
        mixed_wins = index < nonbaseline_improvements
        rows.append(make_external(target, "factor_count", "ok", tcount=20, qasm=100))
        rows.append(make_external(target, "factor_count_pair_cap", "ok", tcount=22, qasm=110))
        rows.append(
            make_external(
                target,
                "mixed_pair",
                "ok",
                tcount=10 if mixed_wins else 25,
                qasm=90 if mixed_wins else 120,
            )
        )
        for objective in OBJECTIVES:
            if objective in {"factor_count", "factor_count_pair_cap", "mixed_pair"}:
                continue
            rows.append(make_external(target, objective, "ok", tcount=24, qasm=115))
    if partial:
        rows.extend(
            [
                make_external("partial", "factor_count", "ok", tcount=30, qasm=100),
                make_external("partial", "factor_count_pair_cap", "failed"),
                make_external("partial", "mixed_pair", "ok", tcount=20, qasm=90),
            ]
        )
    return rows


def make_external(
    target: str,
    objective: str,
    status: str,
    *,
    tcount: int = 10,
    qasm: int = 100,
) -> dict[str, str]:
    return {
        "target": target,
        "objective_variant": objective,
        "night_status": status,
        "night_tcount": str(tcount) if status == "ok" else "",
        "night_best_beam_qasm_depth": str(qasm) if status == "ok" else "",
    }


def verification_rows(*, equal: int, inconclusive: int) -> list[dict[str, str]]:
    return [
        {"target": f"equal_{index}", "verification_status": "equal"}
        for index in range(equal)
    ] + [
        {"target": f"inconclusive_{index}", "verification_status": "inconclusive"}
        for index in range(inconclusive)
    ]


def make_readiness(
    target: str,
    status: str,
    *,
    priority: int,
    action: str = "full-action-feasible",
) -> dict[str, str]:
    return {
        "target": target,
        "family": "arith",
        "tensor_size": "12",
        "tcount_original": "50",
        "action_space_class": action,
        "readiness_status": status,
        "priority_score": str(priority),
        "next_action": "run",
    }
