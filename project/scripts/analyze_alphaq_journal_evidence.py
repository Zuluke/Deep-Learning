from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.structural_target import coerce_float


DEFAULT_SPLIT_SELECT = PROJECT_ROOT / "results" / "csv" / "alphaq_split_select_summary.csv"
DEFAULT_DATASET = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_selection_dataset.csv"
DEFAULT_EXTERNAL_CLUSTER = PROJECT_ROOT / "results" / "csv" / "alphaq_external_runs_consolidated.csv"
DEFAULT_READINESS = PROJECT_ROOT / "results" / "csv" / "alphaq_external_validation_readiness.csv"
DEFAULT_VERIFICATION_CSVS = (
    PROJECT_ROOT / "results" / "verification" / "alphaq_best_objective_beam_ablation" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_objective_beam_policy_factor_count_pair_cap" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_beam_materializer_ablation" / "verification_summary.csv",
    PROJECT_ROOT / "results" / "verification" / "alphaq_beam_materializer_holdout_ablation" / "verification_summary.csv",
)
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_journal_evidence_gates.csv"
DEFAULT_BATTERY_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_journal_next_battery.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_journal_evidence.md"

OBJECTIVES = ("factor_count", "factor_count_pair_cap", "mixed_pair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the AlphaQuantum Split-Select evidence is journal-ready."
    )
    parser.add_argument("--split-select-csv", type=Path, default=DEFAULT_SPLIT_SELECT)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--external-cluster-csv", type=Path, default=DEFAULT_EXTERNAL_CLUSTER)
    parser.add_argument("--readiness-csv", type=Path, default=DEFAULT_READINESS)
    parser.add_argument(
        "--verification-csvs",
        default=",".join(str(path) for path in DEFAULT_VERIFICATION_CSVS),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--battery-csv", type=Path, default=DEFAULT_BATTERY_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_paths(value: str) -> list[Path]:
    return [Path(item) for item in value.split(",") if item.strip()]


def evidence_rows(
    *,
    split_rows: list[dict[str, str]],
    dataset_rows: list[dict[str, str]],
    external_rows: list[dict[str, str]],
    readiness_rows: list[dict[str, str]],
    verification_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = [
        selector_gate(split_rows),
        dataset_gate(dataset_rows),
        external_coverage_gate(external_rows, dataset_rows),
        external_effect_gate(external_rows),
        verification_gate(verification_rows),
        depth_gate(split_rows),
    ]
    rows.append(overall_gate(rows))
    rows.append(next_battery_gate(next_battery_rows(readiness_rows, external_rows)))
    return rows


def selector_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    split = best_split_policy(rows)
    baseline = row_by_policy(rows, "baseline_factor_count")
    if split is None or baseline is None:
        return gate("selector_loto", "missing", 0.0, "Missing Split-Select or baseline rows.", "Regenerate `alphaq_split_select_summary.csv`.")
    split_exact = int_value(split.get("exact_oracle_matches"))
    base_exact = int_value(baseline.get("exact_oracle_matches"))
    split_t_wins = int_value(split.get("tcount_wins_vs_baseline"))
    groups = int_value(split.get("ok_groups"))
    score = 0.0 if groups == 0 else (split_exact - base_exact + split_t_wins) / max(groups, 1)
    status = "pass" if split_exact > base_exact and split_t_wins > 0 else "fail"
    evidence = (
        f"best Split-Select `{split['policy']}` has oracle matches {split_exact}/{groups}; "
        f"baseline has {base_exact}/{groups}; T-count wins over baseline {split_t_wins}/{groups}."
    )
    return gate(
        "selector_loto",
        status,
        score,
        evidence,
        "Keep the trained selector as prototype policy; expand held-out targets before journal claims.",
    )


def dataset_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    groups = grouped_dataset(rows)
    train_ready = [items for items in groups.values() if truthy(items[0].get("train_ready"))]
    external = [key for key, items in groups.items() if key[0].startswith("external") and truthy(items[0].get("train_ready"))]
    labels = Counter(items[0].get("oracle_objective", "") for items in train_ready)
    status = "pass" if len(train_ready) >= 30 and len(external) >= 10 and len(labels) >= 3 else "partial"
    evidence = (
        f"train-ready groups={len(train_ready)}/{len(groups)}; external train-ready groups={len(external)}; "
        f"oracle labels={format_counts(labels)}."
    )
    return gate(
        "dataset_scale_and_label_diversity",
        status,
        min(1.0, len(train_ready) / 30.0),
        evidence,
        "Grow to at least 30 train-ready target/run groups with at least 10 external groups and all three objective labels represented.",
    )


def external_coverage_gate(rows: list[dict[str, str]], dataset_rows: list[dict[str, str]]) -> dict[str, Any]:
    target_status = external_target_status(rows)
    complete = [target for target, status in target_status.items() if status == "complete"]
    partial = [target for target, status in target_status.items() if status == "partial"]
    external_families = external_family_count(dataset_rows)
    status = "pass" if len(complete) >= 10 and external_families >= 3 else "partial" if complete or partial else "fail"
    evidence = (
        f"complete external targets={len(complete)} ({', '.join(complete) or '-'}); "
        f"partial external targets={len(partial)} ({', '.join(partial) or '-'}); "
        f"external families in dataset={external_families}."
    )
    return gate(
        "external_generalization_coverage",
        status,
        min(1.0, len(complete) / 10.0),
        evidence,
        "Run a broader external battery: at least 10 complete external targets spanning at least 3 families.",
    )


def external_effect_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    improvements = []
    regressions = []
    for target, items in external_rows_by_target(rows).items():
        factor = objective_row(items, "factor_count")
        candidates = [row for row in items if run_status(row) == "ok"]
        if factor is None or run_status(factor) != "ok" or not candidates:
            continue
        best = min(candidates, key=lambda row: metric_tuple(row, "night"))
        if best["objective_variant"] != "factor_count":
            if metric_tuple(best, "night") < metric_tuple(factor, "night"):
                improvements.append(f"{target}:{best['objective_variant']}")
            else:
                regressions.append(f"{target}:{best['objective_variant']}")
    status = "pass" if len(improvements) >= 3 and not regressions else "partial" if improvements else "fail"
    evidence = (
        f"non-baseline external improvements={len(improvements)} ({', '.join(improvements) or '-'}); "
        f"non-baseline regressions={len(regressions)} ({', '.join(regressions) or '-'})."
    )
    return gate(
        "external_nonbaseline_effect",
        status,
        min(1.0, len(improvements) / 3.0),
        evidence,
        "Find repeated external cases where Split-Select chooses a non-factor-count objective and improves T-count/depth.",
    )


def verification_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    status_counts = Counter(row.get("verification_status", "missing") for row in rows)
    total = sum(status_counts.values())
    equal = status_counts.get("equal", 0)
    inconclusive = status_counts.get("inconclusive", 0)
    failures = total - equal - inconclusive
    status = "pass" if total >= 20 and equal == total else "partial" if equal > 0 and failures == 0 else "fail"
    evidence = f"formal verification rows={total}; equal={equal}; inconclusive={inconclusive}; failures={failures}."
    return gate(
        "formal_verification_coverage",
        status,
        0.0 if total == 0 else equal / total,
        evidence,
        "Resolve inconclusive proofs and verify all promoted candidates, including the expanded external battery.",
    )


def depth_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    split = best_split_policy(rows)
    if split is None:
        return gate("depth_control", "missing", 0.0, "Missing Split-Select rows.", "Regenerate Split-Select summary.")
    median_qasm = float_value(split.get("median_qasm_ratio_vs_baseline"), default=999.0)
    qasm_nonworse = int_value(split.get("qasm_nonworse_vs_baseline"))
    groups = int_value(split.get("ok_groups"))
    status = "pass" if median_qasm <= 1.05 and qasm_nonworse >= max(1, int(0.8 * groups)) else "partial"
    evidence = f"best Split-Select median QASM ratio={median_qasm:.3g}; QASM non-worse={qasm_nonworse}/{groups}."
    return gate(
        "depth_control",
        status,
        1.0 if median_qasm <= 1.0 else max(0.0, 1.05 - median_qasm),
        evidence,
        "Keep depth as a hard audit metric; avoid claiming T-count wins alone.",
    )


def next_battery_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [row for row in rows if row["recommended_stage"] == "full-action-expansion"]
    screen = [row for row in rows if row["recommended_stage"] == "tensor-v3-screen"]
    restricted = [row for row in rows if row["recommended_stage"] == "restricted-action-pilot"]
    status = "planned" if full or screen or restricted else "missing"
    evidence = (
        f"recommended full-action expansion targets={len(full)}; "
        f"tensor-v3 screening targets={len(screen)}; "
        f"restricted-action pilot targets={len(restricted)}."
    )
    return gate(
        "next_decisive_battery",
        status,
        min(1.0, (len(full) + len(screen) + len(restricted)) / 8.0),
        evidence,
        "Run tensor-v3 screening before repeating failed full-action targets; then implement the restricted-action pilot if the signal survives.",
    )


def overall_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate_status = {row["gate"]: row["status"] for row in rows}
    pass_count = sum(status == "pass" for status in gate_status.values())
    blocking = [
        gate_name
        for gate_name in (
            "dataset_scale_and_label_diversity",
            "external_generalization_coverage",
            "external_nonbaseline_effect",
            "formal_verification_coverage",
        )
        if gate_status.get(gate_name) != "pass"
    ]
    status = "journal-ready" if not blocking else "not-yet-journal-ready"
    evidence = f"passed gates={pass_count}/{len(rows)}; blocking gates={', '.join(blocking) or '-'}."
    return gate(
        "overall_journal_readiness",
        status,
        pass_count / max(1, len(rows)),
        evidence,
        "Do not frame as journal-ready until blocking gates pass; use current results as prototype evidence.",
    )


def next_battery_rows(readiness_rows: list[dict[str, str]], external_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    already_external = set(external_rows_by_target(external_rows))
    rows = []
    for row in readiness_rows:
        target = row.get("target", "")
        status = row.get("readiness_status", "")
        if status == "current-grid-control":
            continue
        if target in already_external and external_target_status(external_rows).get(target) == "complete":
            continue
        if status == "ready-full-action":
            stage = "full-action-repair" if target in already_external else "full-action-expansion"
        elif status == "needs-tensor-v3-screen" and row.get("action_space_class") == "full-action-feasible":
            stage = "tensor-v3-screen"
        elif status == "ready-restricted-action":
            stage = "restricted-action-pilot"
        else:
            continue
        rows.append(
            {
                "target": target,
                "family": row.get("family", ""),
                "tensor_size": row.get("tensor_size", ""),
                "tcount_original": row.get("tcount_original", ""),
                "action_space_class": row.get("action_space_class", ""),
                "readiness_status": status,
                "recommended_stage": stage,
                "priority_score": int_value(row.get("priority_score")),
                "next_action": row.get("next_action", ""),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            stage_rank(row["recommended_stage"]),
            -int_value(row["priority_score"]),
            int_value(row.get("tensor_size"), default=999),
            row["target"],
        ),
    )


def stage_rank(stage: str) -> int:
    return {
        "full-action-repair": 0,
        "full-action-expansion": 1,
        "tensor-v3-screen": 2,
        "restricted-action-pilot": 3,
    }.get(stage, 99)


def best_split_policy(rows: list[dict[str, str]]) -> dict[str, str] | None:
    split_rows = [row for row in rows if row.get("policy", "").startswith("split_select_")]
    if not split_rows:
        return None
    return max(
        split_rows,
        key=lambda row: (
            int_value(row.get("exact_oracle_matches")),
            int_value(row.get("tcount_wins_vs_baseline")),
            -float_value(row.get("median_qasm_ratio_vs_baseline"), default=999.0),
            row.get("policy", ""),
        ),
    )


def row_by_policy(rows: list[dict[str, str]], policy: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("policy") == policy), None)


def grouped_dataset(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row.get("source_split", ""), row.get("target", "")), []).append(row)
    return groups


def external_family_count(rows: list[dict[str, str]]) -> int:
    families = {
        items[0].get("family", "")
        for key, items in grouped_dataset(rows).items()
        if key[0].startswith("external") and truthy(items[0].get("train_ready"))
    }
    return len({family for family in families if family})


def external_rows_by_target(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("target", ""), []).append(row)
    return grouped


def external_target_status(rows: list[dict[str, str]]) -> dict[str, str]:
    statuses = {}
    for target, items in external_rows_by_target(rows).items():
        objective_statuses = {row.get("objective_variant"): run_status(row) for row in items}
        ok_count = sum(objective_statuses.get(objective) == "ok" for objective in OBJECTIVES)
        if ok_count == len(OBJECTIVES):
            statuses[target] = "complete"
        elif ok_count > 0:
            statuses[target] = "partial"
        else:
            statuses[target] = "failed"
    return statuses


def objective_row(items: list[dict[str, str]], objective: str) -> dict[str, str] | None:
    return next((row for row in items if row.get("objective_variant") == objective), None)


def metric_tuple(row: dict[str, str], prefix: str) -> tuple[float, float]:
    return (
        float_value(row_tcount(row, prefix), default=999999.0),
        float_value(row_qasm_depth(row, prefix), default=999999.0),
    )


def run_status(row: dict[str, str]) -> str:
    return row.get("best_status") or row.get("night_status") or row.get("execution_status") or "missing"


def row_tcount(row: dict[str, str], prefix: str) -> str:
    return row.get("best_tcount") or row.get(f"{prefix}_tcount") or row.get("tcount") or ""


def row_qasm_depth(row: dict[str, str], prefix: str) -> str:
    return (
        row.get("best_beam_qasm_depth")
        or row.get(f"{prefix}_best_beam_qasm_depth")
        or row.get("qasm_depth")
        or ""
    )


def gate(
    gate_name: str,
    status: str,
    score: float,
    evidence: str,
    required_next: str,
) -> dict[str, Any]:
    return {
        "gate": gate_name,
        "status": status,
        "score": score,
        "evidence": evidence,
        "required_next": required_next,
    }


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def int_value(value: Any, default: int = 0) -> int:
    parsed = coerce_float(value)
    return default if parsed is None else int(parsed)


def float_value(value: Any, default: float = 0.0) -> float:
    parsed = coerce_float(value)
    return default if parsed is None else float(parsed)


def format_counts(counts: Counter[str]) -> str:
    clean = {key: value for key, value in counts.items() if key}
    return ", ".join(f"{key}={value}" for key, value in sorted(clean.items())) or "-"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_gate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, ["gate", "status", "score", "evidence", "required_next"])


def write_battery_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "target",
            "family",
            "tensor_size",
            "tcount_original",
            "action_space_class",
            "readiness_status",
            "recommended_stage",
            "priority_score",
            "next_action",
        ],
    )


def write_report(
    path: Path,
    gates: list[dict[str, Any]],
    battery: list[dict[str, Any]],
    gate_csv: Path,
    battery_csv: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = next(row for row in gates if row["gate"] == "overall_journal_readiness")
    lines = [
        "# AlphaQuantum Journal Evidence Audit",
        "",
        f"Gate CSV: `{gate_csv}`.",
        f"Next battery CSV: `{battery_csv}`.",
        "",
        "## Decision",
        "",
        f"Decision: `{overall['status']}`.",
        "",
        overall["evidence"],
        "",
        "The current evidence is strong enough to justify a prototype integration experiment, but it is not yet enough for a journal-level robustness claim. The blocking issues are external coverage, repeated non-baseline external wins, and complete formal verification over the promoted/expanded candidates.",
        "",
        "## Evidence Gates",
        "",
        "| gate | status | score | evidence | required next |",
        "|---|---|---:|---|---|",
    ]
    for row in gates:
        lines.append(
            f"| {row['gate']} | {row['status']} | {float_value(row['score']):.3g} | {row['evidence']} | {row['required_next']} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Next Battery",
            "",
            "| target | family | tensor size | T original | stage | status | action |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in battery[:15]:
        lines.append(
            f"| {row['target']} | {row['family']} | {row['tensor_size']} | {row['tcount_original']} | {row['recommended_stage']} | {row['readiness_status']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Practical Interpretation",
            "",
            "The immediate journal path is not to change the reward again. It is to run the next external battery, keep failures explicit, and then rerun this audit. If the Split-Select policy keeps the Barenco-like gains while avoiding VBE-like regressions across a larger set, then the result starts looking like a journal claim rather than a promising engineering observation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    verification_rows = [
        row
        for path in parse_paths(args.verification_csvs)
        for row in read_csv(path)
    ]
    external_rows = read_csv(args.external_cluster_csv)
    readiness = read_csv(args.readiness_csv)
    battery = next_battery_rows(readiness, external_rows)
    gates = evidence_rows(
        split_rows=read_csv(args.split_select_csv),
        dataset_rows=read_csv(args.dataset_csv),
        external_rows=external_rows,
        readiness_rows=readiness,
        verification_rows=verification_rows,
    )
    write_gate_csv(args.output_csv, gates)
    write_battery_csv(args.battery_csv, battery)
    write_report(args.report_path, gates, battery, args.output_csv, args.battery_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.battery_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
