from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json


DEFAULT_PROFILE_CSV = DEFAULT_CSV_ROOT / "tensor_v3_profile_comparison.csv"
DEFAULT_GUARD_SURFACE_CSV = DEFAULT_CSV_ROOT / "tensor_v3_guard_surface_summary.csv"
DEFAULT_LOCO_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_loco_eval.csv"
DEFAULT_SEED_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_seed_robustness.csv"
DEFAULT_ABLATION_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_ablations.csv"
DEFAULT_ALPHAQ_FINAL_VERIFICATION_CSV = (
    DEFAULT_RESULTS_ROOT / "verification" / "alphaq_final_verification_summary.csv"
)
DEFAULT_TENSOR_V3_VERIFICATION_CSV = (
    DEFAULT_RESULTS_ROOT
    / "verification"
    / "entrega1"
    / "tensor_v3_phase_slack_verification_summary.csv"
)
DEFAULT_FRONTIER_VERIFICATION_CSV = (
    DEFAULT_RESULTS_ROOT
    / "verification"
    / "frontier"
    / "alphaq_candidate_frontier_verification.csv"
)
DEFAULT_TABLE_MANIFEST_JSON = REPO_ROOT / "paper" / "tables" / "tensor_v3_guard_table.json"
DEFAULT_TENSOR_V3_SELECTION_MANIFEST_CSV = (
    DEFAULT_RESULTS_ROOT
    / "public_resynth_tensor_v3_phase_slack_guarded"
    / "tensor_v3_selection_manifest.csv"
)
DEFAULT_OUTPUT_CSV = DEFAULT_CSV_ROOT / "submission_readiness_audit.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_CSV_ROOT / "submission_readiness_audit.json"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_ROOT / "submission_readiness_audit.md"

PASS = "pass"
WARN = "warn"
FAIL = "fail"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    numeric = coerce_float(value)
    return None if numeric is None else int(numeric)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt(value: Any, digits: int = 3) -> str:
    numeric = coerce_float(value)
    return "NA" if numeric is None else f"{numeric:.{digits}f}"


def finite_values(values: Any) -> list[float]:
    output = []
    for value in values:
        numeric = coerce_float(value)
        if numeric is not None:
            output.append(numeric)
    return output


def mean(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def count_values(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def csv_row(
    *,
    evidence_id: str,
    status: str,
    claim: str,
    result: str,
    scope: str,
    artifact_path: Path,
    recommendation: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "status": status,
        "claim": claim,
        "result": result,
        "scope": scope,
        "artifact_path": str(artifact_path),
        "recommendation": recommendation,
        "details_json": json.dumps(details or {}, sort_keys=True),
    }


def audit_tensor_v3_profile(rows: list[dict[str, Any]], artifact_path: Path) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    guarded_counts = count_values(ok_rows, "guarded_relation_vs_conservative")
    aggressive_counts = count_values(ok_rows, "aggressive_relation_vs_conservative")
    guarded_worse = guarded_counts.get("worse", 0)
    guarded_better = guarded_counts.get("better", 0)
    guarded_tie = guarded_counts.get("tie", 0)
    changed = sum(truthy(row.get("guarded_uses_aggressive")) for row in ok_rows)
    t_overhead = sum(
        int(coerce_int(row.get("guarded_delta_tcount_vs_conservative")) or 0)
        for row in ok_rows
        if (coerce_int(row.get("guarded_delta_tcount_vs_conservative")) or 0) > 0
    )
    global_hits = sum(truthy(row.get("guarded_is_global_oracle")) for row in ok_rows)
    status = PASS if ok_rows and guarded_worse == 0 and guarded_better > 0 else WARN
    if not ok_rows or guarded_worse > 0:
        status = FAIL
    return csv_row(
        evidence_id="tensor_v3_guarded_profile",
        status=status,
        claim="The guarded tensor-v3 selector preserves conservative behavior while retaining observed structural improvements.",
        result=(
            f"guarded B/T/W={guarded_better}/{guarded_tie}/{guarded_worse}; "
            f"accepted aggressive candidates={changed}/{len(ok_rows)}; "
            f"global-oracle hits={global_hits}/{len(ok_rows)}; "
            f"T overhead=+{t_overhead}."
        ),
        scope="expanded tensor-v3 audit circuits with available candidate frontiers",
        artifact_path=artifact_path,
        recommendation=(
            "Use guarded tensor-v3 as the conservative manuscript baseline; keep the "
            "unguarded aggressive profile as an ablation."
        ),
        details={
            "num_ok_rows": len(ok_rows),
            "guarded_relation_counts": guarded_counts,
            "aggressive_relation_counts": aggressive_counts,
            "guarded_candidate_changes": changed,
            "guarded_tcount_overhead_total": t_overhead,
            "guarded_global_oracle_hits": global_hits,
        },
    )


def forbidden_selection_columns(columns: list[str]) -> list[str]:
    forbidden = []
    exact_forbidden = {"qasm_depth_ratio"}
    contains_forbidden = (
        "primary",
        "zx_",
        "feynver",
        "formal",
        "verification",
        "oracle",
        "alphaq_",
        "structural",
    )
    for column in columns:
        lowered = column.lower()
        if lowered in exact_forbidden or any(token in lowered for token in contains_forbidden):
            forbidden.append(column)
    return forbidden


def audit_tensor_v3_selection_manifest(
    rows: list[dict[str, Any]], artifact_path: Path
) -> dict[str, Any]:
    columns = list(rows[0]) if rows else []
    forbidden_columns = forbidden_selection_columns(columns)
    selected_rows = [
        row
        for row in rows
        if str(row.get("tensor_v3_selected") or "").strip().lower() in {"1", "true"}
    ]
    status = PASS if rows and selected_rows and not forbidden_columns else FAIL
    return csv_row(
        evidence_id="tensor_v3_selection_manifest_purity",
        status=status,
        claim="The deployable guarded tensor-v3 selector is materialized before external ZX/feynver audit labels are attached.",
        result=(
            f"rows={len(rows)}; selected={len(selected_rows)}; "
            f"forbidden_columns={len(forbidden_columns)}."
        ),
        scope="guarded tensor-v3 selected-candidate manifest",
        artifact_path=artifact_path,
        recommendation=(
            "Use this manifest as the source of selected candidates in the manuscript "
            "pipeline; keep primary/Zx/feynver columns in downstream audit tables only."
        ),
        details={
            "columns": columns,
            "forbidden_columns": forbidden_columns,
            "selected_candidate_ids": [
                row.get("candidate_id") for row in selected_rows if row.get("candidate_id")
            ],
        },
    )


def audit_guard_surface(rows: list[dict[str, Any]], artifact_path: Path) -> dict[str, Any]:
    parsed = [
        {
            **row,
            "_qasm_gain": coerce_float(row.get("qasm_depth_gain")),
            "_mixed_drop": coerce_float(row.get("mixed_drop_fraction")),
            "_better": coerce_int(row.get("better_vs_conservative")) or 0,
            "_worse": coerce_int(row.get("worse_vs_conservative")) or 0,
            "_hits": coerce_int(row.get("global_oracle_hits")) or 0,
            "_t_overhead": coerce_int(row.get("tcount_overhead_total")) or 0,
            "_max_regret": coerce_float(row.get("max_global_oracle_regret")),
        }
        for row in rows
    ]
    non_worse = [
        row
        for row in parsed
        if row["_worse"] == 0 and row["_qasm_gain"] is not None and row["_better"] > 0
    ]
    robust = [row for row in non_worse if (row["_qasm_gain"] or 0.0) >= 0.10]
    qasm_values = [row["_qasm_gain"] for row in non_worse if row["_qasm_gain"] is not None]
    best = min(
        robust or non_worse,
        key=lambda row: (
            -row["_better"],
            -row["_hits"],
            row["_max_regret"] if row["_max_regret"] is not None else float("inf"),
            row["_t_overhead"],
            row["_qasm_gain"] if row["_qasm_gain"] is not None else float("inf"),
        ),
        default=None,
    )
    status = PASS if robust else WARN if non_worse else FAIL
    if best is None:
        result = "No non-worse guard setting found."
        details: dict[str, Any] = {"non_worse_count": 0}
    else:
        result = (
            f"{len(non_worse)} non-worse settings; robust settings with qasm_gain>=0.10: "
            f"{len(robust)}; selected plateau row qasm_gain={fmt(best['_qasm_gain'])}, "
            f"mixed_drop={fmt(best['_mixed_drop'])}, B/T/W="
            f"{best['_better']}/{best.get('tie_vs_conservative')}/{best['_worse']}."
        )
        details = {
            "non_worse_count": len(non_worse),
            "robust_count": len(robust),
            "non_worse_qasm_gain_min": min(qasm_values) if qasm_values else None,
            "non_worse_qasm_gain_max": max(qasm_values) if qasm_values else None,
            "selected_row": {
                key: best.get(key)
                for key in (
                    "qasm_depth_gain",
                    "mixed_drop_fraction",
                    "better_vs_conservative",
                    "tie_vs_conservative",
                    "worse_vs_conservative",
                    "global_oracle_hits",
                    "max_global_oracle_regret",
                    "tcount_overhead_total",
                    "accepted_circuits",
                    "regressed_circuits",
                )
            },
        }
    return csv_row(
        evidence_id="tensor_v3_guard_surface",
        status=status,
        claim="The QASM-depth guard has a stable non-worse plateau rather than a single hand-picked threshold.",
        result=result,
        scope="grid sweep over guarded qasm-depth gain and mixed-drop thresholds",
        artifact_path=artifact_path,
        recommendation=(
            "Report the plateau and not only the chosen tau_d=0.10 setting; this makes "
            "the guard calibration harder to dismiss as cherry-picking."
        ),
        details=details,
    )


def audit_alphaq_loco(rows: list[dict[str, Any]], artifact_path: Path) -> dict[str, Any]:
    regrets = finite_values(row.get("primary_regret_vs_true_best") for row in rows)
    gains = finite_values(row.get("primary_gain_vs_tcount_best") for row in rows)
    hits = sum(truthy(row.get("hit_true_best")) for row in rows)
    singleton_tests = sum((coerce_int(row.get("num_test_candidates")) or 0) <= 1 for row in rows)
    hit_rate = hits / len(rows) if rows else 0.0
    mean_regret = mean(regrets)
    status = WARN
    if not rows or mean_regret is None:
        status = FAIL
    elif hit_rate < 0.5 or mean_regret > 0.25:
        status = FAIL
    return csv_row(
        evidence_id="alphaq_final_loco_generalization",
        status=status,
        claim="The AlphaQ-only learned ranker generalizes beyond full-fit selection under leave-one-circuit-out evaluation.",
        result=(
            f"LOCO hit rate={fmt(hit_rate)}; mean regret={fmt(mean_regret)}; "
            f"max regret={fmt(max(regrets) if regrets else None)}; "
            f"mean gain vs T-count={fmt(mean(gains))}; singleton tests={singleton_tests}/{len(rows)}."
        ),
        scope="five formally retained circuits; some held-out circuits have only one verified candidate",
        artifact_path=artifact_path,
        recommendation=(
            "Present this as supportive but scoped learning evidence. The next robustness "
            "step is expanding formally equal candidate frontiers, not claiming universal generalization."
        ),
        details={
            "num_circuits": len(rows),
            "hit_rate": hit_rate,
            "mean_regret": mean_regret,
            "max_regret": max(regrets) if regrets else None,
            "mean_gain_vs_tcount_best": mean(gains),
            "singleton_test_circuits": singleton_tests,
        },
    )


def audit_seed_robustness(rows: list[dict[str, Any]], artifact_path: Path) -> dict[str, Any]:
    circuits = sorted({row.get("circuit_id", "") for row in rows if row.get("circuit_id")}, key=natural_sort_key)
    seed_count = len({row.get("seed") for row in rows if row.get("seed")})
    unstable: list[str] = []
    max_mean_regret = 0.0
    per_circuit: dict[str, Any] = {}
    for circuit_id in circuits:
        subset = [row for row in rows if row.get("circuit_id") == circuit_id]
        selected = {row.get("selected_candidate_id") for row in subset}
        regrets = finite_values(row.get("primary_regret_vs_structural_best") for row in subset)
        circuit_mean = mean(regrets) or 0.0
        max_mean_regret = max(max_mean_regret, circuit_mean)
        per_circuit[circuit_id] = {
            "unique_selections": len(selected),
            "mean_regret": circuit_mean,
        }
        if len(selected) > 1 or circuit_mean > 1e-9:
            unstable.append(circuit_id)
    status = PASS if rows and not unstable and seed_count >= 5 else WARN if rows else FAIL
    return csv_row(
        evidence_id="alphaq_final_seed_robustness",
        status=status,
        claim="The final AlphaQ-only ranker selection is stable across random seeds.",
        result=(
            f"{seed_count} seeds; {len(circuits)} circuits; unstable circuits={len(unstable)}; "
            f"max per-circuit mean regret={fmt(max_mean_regret)}."
        ),
        scope="seed robustness sweep for the final pairwise MLP ranker",
        artifact_path=artifact_path,
        recommendation=(
            "Use this as a reproducibility argument for the learned selector; keep the "
            "small-circuit-count limitation separate."
        ),
        details={"seed_count": seed_count, "per_circuit": per_circuit, "unstable": unstable},
    )


def audit_ablation(rows: list[dict[str, Any]], artifact_path: Path) -> dict[str, Any]:
    by_name = {row.get("ablation"): row for row in rows}
    final_row = by_name.get("pairwise_mlp_all")
    tcount_row = by_name.get("baseline:tcount")
    no_dependency_row = by_name.get("pairwise_mlp_no_dependency")
    final_regret = coerce_float(final_row.get("mean_regret") if final_row else None)
    tcount_regret = coerce_float(tcount_row.get("mean_regret") if tcount_row else None)
    no_dependency_regret = coerce_float(
        no_dependency_row.get("mean_regret") if no_dependency_row else None
    )
    status = WARN
    if final_regret is None or tcount_regret is None or no_dependency_regret is None:
        status = FAIL
    elif final_regret < tcount_regret and final_regret < no_dependency_regret:
        status = PASS
    return csv_row(
        evidence_id="alphaq_final_feature_ablation",
        status=status,
        claim="Dependency/border structural features add signal beyond T-count-only selection.",
        result=(
            f"final mean regret={fmt(final_regret)}; T-count baseline={fmt(tcount_regret)}; "
            f"no-dependency ablation={fmt(no_dependency_regret)}."
        ),
        scope="same five-circuit AlphaQ-final evaluation frontier",
        artifact_path=artifact_path,
        recommendation=(
            "Use the ablation to justify why the proxy is structural, not just a disguised "
            "T-count heuristic."
        ),
        details={
            "pairwise_mlp_all": final_row,
            "baseline_tcount": tcount_row,
            "pairwise_mlp_no_dependency": no_dependency_row,
        },
    )


def audit_verification(
    rows: list[dict[str, Any]],
    artifact_path: Path,
    *,
    evidence_id: str,
    claim: str,
    scope: str,
    min_equal: int,
    allow_partial: bool = False,
) -> dict[str, Any]:
    counts = count_values(rows, "verification_status")
    equal = counts.get("equal", 0)
    total = len(rows)
    status = PASS if total >= min_equal and equal == total else WARN if allow_partial and equal >= min_equal else FAIL
    non_equal = [
        f"{row.get('circuit_id')}:{row.get('method')}={row.get('verification_status')}"
        for row in rows
        if row.get("verification_status") != "equal"
    ]
    return csv_row(
        evidence_id=evidence_id,
        status=status,
        claim=claim,
        result=f"equal={equal}/{total}; status counts={dict(sorted(counts.items()))}.",
        scope=scope,
        artifact_path=artifact_path,
        recommendation=(
            "Treat equal rows as formally audited evidence; inconclusive/timeout rows are "
            "not counterexamples, but they cannot support the main claim until resolved."
        ),
        details={
            "status_counts": counts,
            "non_equal": non_equal,
            "circuits": sorted({row.get("circuit_id") for row in rows if row.get("circuit_id")}, key=natural_sort_key),
        },
    )


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_paper_table_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return csv_row(
            evidence_id="paper_tensor_v3_table_manifest",
            status=FAIL,
            claim="The manuscript tensor-v3 table is backed by a current manifest.",
            result="Manifest is missing.",
            scope="paper table artifact",
            artifact_path=manifest_path,
            recommendation="Regenerate the paper table before relying on it.",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison_csv = Path(manifest.get("comparison_csv", ""))
    output_tex = Path(manifest.get("output_tex", ""))
    comparison_current = sha256_file(comparison_csv) == manifest.get("comparison_csv_sha256")
    table_current = sha256_file(output_tex) == manifest.get("output_tex_sha256")
    counts = manifest.get("guarded_relation_counts", {})
    status = PASS if comparison_current and table_current and counts.get("worse") == 0 else FAIL
    return csv_row(
        evidence_id="paper_tensor_v3_table_manifest",
        status=status,
        claim="The manuscript tensor-v3 table is synchronized with the generated comparison CSV.",
        result=(
            f"comparison_current={comparison_current}; table_current={table_current}; "
            f"guarded B/T/W={counts.get('better')}/{counts.get('tie')}/{counts.get('worse')}."
        ),
        scope="paper table and JSON manifest",
        artifact_path=manifest_path,
        recommendation="Run the table exporter in --check mode before paper submission.",
        details={
            "comparison_csv": str(comparison_csv),
            "output_tex": str(output_tex),
            "comparison_current": comparison_current,
            "table_current": table_current,
            "manifest": manifest,
        },
    )


def build_audit_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [
        audit_tensor_v3_profile(
            load_csv_rows(args.profile_csv),
            args.profile_csv,
        ),
        audit_tensor_v3_selection_manifest(
            load_csv_rows(args.tensor_v3_selection_manifest_csv),
            args.tensor_v3_selection_manifest_csv,
        ),
        audit_guard_surface(
            load_csv_rows(args.guard_surface_csv),
            args.guard_surface_csv,
        ),
        audit_alphaq_loco(
            load_csv_rows(args.loco_csv),
            args.loco_csv,
        ),
        audit_seed_robustness(
            load_csv_rows(args.seed_robustness_csv),
            args.seed_robustness_csv,
        ),
        audit_ablation(
            load_csv_rows(args.ablation_csv),
            args.ablation_csv,
        ),
        audit_verification(
            load_csv_rows(args.alphaq_final_verification_csv),
            args.alphaq_final_verification_csv,
            evidence_id="alphaq_final_formal_verification",
            claim="The AlphaQ-final selected circuits are formally equivalent to the originals.",
            scope="five AlphaQ-final selected circuits",
            min_equal=5,
        ),
        audit_verification(
            load_csv_rows(args.tensor_v3_verification_csv),
            args.tensor_v3_verification_csv,
            evidence_id="tensor_v3_formal_verification",
            claim="The tensor-v3 selected circuits used in the current claim are formally equivalent to the originals.",
            scope="five tensor-v3 phase-slack selected circuits",
            min_equal=5,
        ),
        audit_verification(
            load_csv_rows(args.frontier_verification_csv),
            args.frontier_verification_csv,
            evidence_id="candidate_frontier_formal_coverage",
            claim="The learned AlphaQ-final model trains/evaluates on a formally audited frontier.",
            scope="candidate frontier before requiring equal candidates",
            min_equal=5,
            allow_partial=True,
        ),
        audit_paper_table_manifest(args.table_manifest_json),
    ]
    return rows


def readiness_verdict(rows: list[dict[str, Any]]) -> str:
    counts = count_values(rows, "status")
    if counts.get(FAIL, 0):
        return "not-ready"
    if counts.get(WARN, 0):
        return "ready-with-caveats"
    return "ready"


def write_report(
    rows: list[dict[str, Any]],
    *,
    report_path: Path,
    output_csv: Path,
    output_json: Path,
) -> Path:
    counts = count_values(rows, "status")
    verdict = readiness_verdict(rows)
    table_lines = [
        "| evidence | status | result | recommendation |",
        "|---|---|---|---|",
    ]
    for row in rows:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['evidence_id']}`",
                    f"`{row['status']}`",
                    str(row["result"]).replace("|", "\\|"),
                    str(row["recommendation"]).replace("|", "\\|"),
                ]
            )
            + " |"
        )

    limiting_rows = [row for row in rows if row["status"] != PASS]
    limitations = [
        f"- `{row['evidence_id']}`: {row['scope']} -> {row['result']}"
        for row in limiting_rows
    ] or ["- None under the current audit gates."]
    text = [
        "# Submission Readiness Audit",
        "",
        f"- Audit CSV: `{output_csv}`.",
        f"- Audit JSON: `{output_json}`.",
        f"- Verdict: `{verdict}`.",
        f"- Status counts: pass={counts.get(PASS, 0)}, warn={counts.get(WARN, 0)}, fail={counts.get(FAIL, 0)}.",
        "",
        "## Scientific Reading",
        "",
        (
            "The current evidence supports a controlled, formally audited claim: an "
            "AlphaQuantum-only structural selector can expose splitting-aware candidate "
            "choices beyond plain T-count on the retained benchmark subset. The evidence "
            "does not yet support a universal optimization claim, mainly because the "
            "learned-ranker frontier is small and some candidate-frontier verification "
            "remains inconclusive."
        ),
        "",
        "## Evidence Gates",
        "",
        *table_lines,
        "",
        "## Remaining Limitations",
        "",
        *limitations,
        "",
        "## Next Robustness Moves",
        "",
        "1. Expand formally equal candidate frontiers for circuits where the verifier currently returns inconclusive or timeout.",
        "2. Re-run leave-one-circuit-out after frontier expansion so every held-out circuit has multiple verified candidates.",
        "3. Keep the guarded tensor-v3 selector as the conservative default and report the aggressive selector only as an ablation.",
        "4. Use ZX/feynver strictly as external audits while preserving AlphaQuantum-only features and labels inside the selector.",
        "",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate current AlphaQ/tensor-v3 evidence into a submission-readiness audit."
    )
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILE_CSV)
    parser.add_argument("--guard-surface-csv", type=Path, default=DEFAULT_GUARD_SURFACE_CSV)
    parser.add_argument("--loco-csv", type=Path, default=DEFAULT_LOCO_CSV)
    parser.add_argument("--seed-robustness-csv", type=Path, default=DEFAULT_SEED_CSV)
    parser.add_argument("--ablation-csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument(
        "--alphaq-final-verification-csv",
        type=Path,
        default=DEFAULT_ALPHAQ_FINAL_VERIFICATION_CSV,
    )
    parser.add_argument(
        "--tensor-v3-verification-csv",
        type=Path,
        default=DEFAULT_TENSOR_V3_VERIFICATION_CSV,
    )
    parser.add_argument(
        "--frontier-verification-csv",
        type=Path,
        default=DEFAULT_FRONTIER_VERIFICATION_CSV,
    )
    parser.add_argument("--table-manifest-json", type=Path, default=DEFAULT_TABLE_MANIFEST_JSON)
    parser.add_argument(
        "--tensor-v3-selection-manifest-csv",
        type=Path,
        default=DEFAULT_TENSOR_V3_SELECTION_MANIFEST_CSV,
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_audit_rows(args)
    write_csv_rows(rows, args.output_csv)
    payload = {
        "verdict": readiness_verdict(rows),
        "status_counts": count_values(rows, "status"),
        "output_csv": str(args.output_csv),
        "report_path": str(args.report_path),
        "evidence": rows,
    }
    write_json(payload, args.output_json)
    write_report(
        rows,
        report_path=args.report_path,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(json.dumps(payload | {"evidence": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
