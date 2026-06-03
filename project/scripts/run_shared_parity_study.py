from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import ensure_dir
from scripts._analysis_common import write_json
from scripts.materialize_split_reward_candidate import find_benchmark_dir
from scripts.run_formal_verification import circuit_to_tensor_binary


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "alphaq_shared_parity_study"
DEFAULT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_shared_parity_study.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_shared_parity_study.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_shared_parity_study.png"
DEFAULT_PROOF_ROOT = PROJECT_ROOT / "results" / "verification" / "alphaq_shared_parity_study"


@dataclass(frozen=True)
class StudyCase:
    target: str
    max_action_weight: int
    objective: str
    candidate_kind: str
    factor_order: str
    target_strategy: str
    mixed_weight_scale: float = 5.0
    pair_weight_scale: float = 0.25
    max_pair_overlap: int | None = 6


STUDY_CASES: dict[str, StudyCase] = {
    "mod_5_4": StudyCase(
        target="mod_5_4",
        max_action_weight=5,
        objective="factor-count",
        candidate_kind="milp_span_factor_count_wfull",
        factor_order="greedy-cnot",
        target_strategy="max-change",
        mixed_weight_scale=1.0,
        pair_weight_scale=0.0,
        max_pair_overlap=None,
    ),
    "gf_2pow2_mult": StudyCase(
        target="gf_2pow2_mult",
        max_action_weight=6,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6_wfull",
        factor_order="greedy-cnot",
        target_strategy="max-change",
    ),
    "hamming_weight_n4": StudyCase(
        target="hamming_weight_n4",
        max_action_weight=5,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6",
        factor_order="greedy-cnot",
        target_strategy="max-row-weight",
    ),
    "hamming_weight_n5": StudyCase(
        target="hamming_weight_n5",
        max_action_weight=5,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6",
        factor_order="given",
        target_strategy="min-change",
    ),
    "barenco_tof_3": StudyCase(
        target="barenco_tof_3",
        max_action_weight=8,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6_wfull",
        factor_order="greedy-cnot",
        target_strategy="max-change",
    ),
    "nc_tof_3": StudyCase(
        target="nc_tof_3",
        max_action_weight=7,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6_wfull",
        factor_order="greedy-cnot",
        target_strategy="max-change",
    ),
    "cuccaro_adder_n3": StudyCase(
        target="cuccaro_adder_n3",
        max_action_weight=8,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6_wfull",
        factor_order="greedy-cnot",
        target_strategy="max-change",
    ),
    "gf_2pow3_mult": StudyCase(
        target="gf_2pow3_mult",
        max_action_weight=9,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6_wfull",
        factor_order="greedy-cnot",
        target_strategy="max-change",
    ),
    "mod_mult_55": StudyCase(
        target="mod_mult_55",
        max_action_weight=11,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6_wfull",
        factor_order="greedy-cnot",
        target_strategy="max-change",
    ),
    "nc_tof_4": StudyCase(
        target="nc_tof_4",
        max_action_weight=11,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6_wfull",
        factor_order="greedy-cnot",
        target_strategy="max-change",
    ),
    "hamming_weight_n6": StudyCase(
        target="hamming_weight_n6",
        max_action_weight=5,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6",
        factor_order="given",
        target_strategy="min-change",
    ),
    "vbe_adder_3": StudyCase(
        target="vbe_adder_3",
        max_action_weight=5,
        objective="mixed-pair",
        candidate_kind="milp_span_mixed_pair_pairo6",
        factor_order="greedy-cnot",
        target_strategy="max-change",
    ),
}

CORE_TARGETS = (
    "mod_5_4",
    "gf_2pow2_mult",
    "hamming_weight_n4",
    "hamming_weight_n5",
)
EXPANDED_TARGETS = (
    *CORE_TARGETS,
    "barenco_tof_3",
    "nc_tof_3",
    "cuccaro_adder_n3",
    "gf_2pow3_mult",
    "mod_mult_55",
    "nc_tof_4",
    "hamming_weight_n6",
    "vbe_adder_3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the AlphaQ MILP + shared-parity candidate study."
    )
    parser.add_argument(
        "--preset",
        choices=("core", "expanded"),
        default="core",
        help="Target preset used when --targets is omitted.",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated targets to run. Overrides --preset.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--proof-root", type=Path, default=DEFAULT_PROOF_ROOT)
    parser.add_argument("--time-limit-sec", type=float, default=300.0)
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def case_output_dir(output_root: Path, case: StudyCase) -> Path:
    return output_root / (
        f"{case.target}_{case.candidate_kind}_{case.factor_order}_"
        f"{case.target_strategy}"
    )


def optimization_output_dir(output_root: Path, case: StudyCase) -> Path:
    return output_root / "linear_span" / (
        f"{case.target}_low-weight_w{case.max_action_weight}_k175_"
        f"{case.objective}"
    )


def run_optimization(case: StudyCase, output_root: Path, time_limit_sec: float) -> Path:
    linear_root = output_root / "linear_span"
    cmd = [
        sys.executable,
        "scripts/optimize_linear_span_candidate.py",
        "--target",
        case.target,
        "--action-dictionary",
        "low-weight",
        "--max-action-weight",
        str(case.max_action_weight),
        "--objective",
        case.objective,
        "--mixed-weight-scale",
        str(case.mixed_weight_scale),
        "--support-weight-scale",
        "0.25",
        "--pair-weight-scale",
        str(case.pair_weight_scale),
        "--candidate-kind",
        case.candidate_kind,
        "--output-root",
        str(linear_root),
        "--time-limit-sec",
        str(time_limit_sec),
    ]
    if case.max_pair_overlap is not None:
        cmd.extend(["--max-pair-overlap", str(case.max_pair_overlap)])
    run_cmd(cmd)
    return optimization_output_dir(output_root, case) / "candidate_factors_manifest.csv"


def run_materialization(case: StudyCase, manifest: Path, output_root: Path) -> Path:
    output_dir = case_output_dir(output_root, case)
    cmd = [
        sys.executable,
        "scripts/materialize_shared_parity_candidate.py",
        "--target",
        case.target,
        "--manifest-csv",
        str(manifest),
        "--candidate-kind",
        case.candidate_kind,
        "--factor-order",
        case.factor_order,
        "--target-strategy",
        case.target_strategy,
        "--output-root",
        str(output_dir),
    ]
    run_cmd(cmd)
    return output_dir / "summary.json"


def run_verification(case: StudyCase, summary_path: Path, proof_root: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    benchmark_dir = find_benchmark_dir(case.target)
    original = benchmark_dir / f"{case.target}.qasm"
    candidate = Path(summary["assembled_qasm"])
    proof_dir = proof_root / case.target
    ensure_dir(proof_dir)
    proof_path = proof_dir / f"{case.candidate_kind}.verify.txt"
    cmd = [str(circuit_to_tensor_binary()), "verify", str(original), str(candidate)]
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = completed.stdout + completed.stderr
    proof_path.write_text(output, encoding="utf-8")
    if completed.stdout.startswith("Equal"):
        status = "equal"
    elif completed.stdout.startswith("Inconclusive"):
        status = "inconclusive"
    else:
        status = "failed"
    return {
        "target": case.target,
        "candidate_kind": case.candidate_kind,
        "verification_status": status,
        "returncode": completed.returncode,
        "proof_path": str(proof_path),
    }


def run_analysis(summary_paths: list[Path], csv_path: Path, report_path: Path, figure_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/analyze_materialized_candidates.py",
        *[str(path) for path in summary_paths],
        "--output-csv",
        str(csv_path),
        "--report-path",
        str(report_path),
        "--figure-path",
        str(figure_path),
    ]
    run_cmd(cmd)


def selected_cases(targets: str | None, *, preset: str = "core") -> list[StudyCase]:
    if targets is None:
        targets = ",".join(EXPANDED_TARGETS if preset == "expanded" else CORE_TARGETS)
    result = []
    for target in [item.strip() for item in targets.split(",") if item.strip()]:
        if target not in STUDY_CASES:
            raise SystemExit(f"Unknown target {target!r}; expected one of {sorted(STUDY_CASES)}.")
        result.append(STUDY_CASES[target])
    return result


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    ensure_dir(output_root)
    summary_paths: list[Path] = []
    verification_rows: list[dict[str, Any]] = []
    for case in selected_cases(args.targets, preset=args.preset):
        manifest = optimization_output_dir(output_root, case) / "candidate_factors_manifest.csv"
        if not args.skip_optimization:
            manifest = run_optimization(case, output_root, args.time_limit_sec)
        summary_path = run_materialization(case, manifest, output_root)
        summary_paths.append(summary_path)
        if not args.skip_verification:
            verification_rows.append(run_verification(case, summary_path, args.proof_root))
    run_analysis(summary_paths, args.csv_path, args.report_path, args.figure_path)
    if verification_rows:
        write_json(verification_rows, args.proof_root / "verification_summary.json")
        failed = [row for row in verification_rows if row["verification_status"] != "equal"]
        if failed:
            raise SystemExit(f"Formal verification failed for {failed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
