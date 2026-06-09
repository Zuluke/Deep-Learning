from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.structural_target import coerce_float
from scripts.zx_splitting import compute_paper_zx_splitting_metrics_from_qasm
from scripts.zx_splitting import compute_zx_splitting_metrics_from_qasm


DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "zx_border_detector_comparison.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "zx_border_detector_comparison.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the existing row-border ZX splitting detector with the "
            "paper-style ZX border detector inspired by arXiv:2504.16004."
        )
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        type=Path,
        default=None,
        help="Input CSV. Can be passed multiple times. Defaults to active AlphaQ grids plus archived tensor-v3 frontier when present.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--rewrite-level",
        choices=("graphlike", "clifford", "full"),
        default="clifford",
        help="ZX rewrite level used by the paper-style detector.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional debugging cap. 0 means no cap.",
    )
    return parser.parse_args()


def default_input_csvs() -> list[Path]:
    paths = sorted(
        (PROJECT_ROOT / "results" / "csv").glob(
            "alphaq_objective_beam_policy_external_validation*_grid.csv"
        )
    )
    archive_frontier = (
        PROJECT_ROOT
        / "results"
        / "archive"
        / "old_experiments"
        / "public_resynth_tensor_v3_phase_slack_expanded"
        / "candidate_frontier.csv"
    )
    if archive_frontier.exists():
        paths.append(archive_frontier)
    return paths


def read_rows(paths: list[Path], max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item = dict(row)
                item["_source_csv"] = str(path)
                rows.append(item)
                if max_rows and len(rows) >= max_rows:
                    return rows
    return rows


def comparison_rows(
    rows: list[dict[str, Any]],
    *,
    rewrite_level: str,
) -> list[dict[str, Any]]:
    qasm_cache: dict[str, dict[str, Any]] = {}
    original_cache: dict[str, dict[str, Any] | None] = {}
    output = []
    for row in rows:
        target = target_name(row)
        objective = objective_name(row)
        materializer = row.get("materializer") or row.get("candidate_id") or ""
        qasm_path = resolve_candidate_qasm(row)
        base = {
            "source_csv": row.get("_source_csv", ""),
            "target": target,
            "objective_variant": objective,
            "materializer": materializer,
            "candidate_id": row.get("candidate_id", ""),
            "qasm_path": "" if qasm_path is None else str(qasm_path),
            "qasm_status": "missing-qasm" if qasm_path is None else "ok",
            "tcount": row.get("tcount") or row.get("tcount_after") or "",
            "tdepth": row.get("tdepth") or row.get("tdepth_after") or "",
            "qasm_depth": row.get("qasm_depth") or row.get("depth_after") or "",
            "legacy_primary_nc_depth_ratio_input": row.get("primary_nc_depth_ratio", ""),
            "legacy_structural_status_input": row.get("structural_target_status", ""),
        }
        if qasm_path is None:
            output.append(base)
            continue

        metrics = qasm_cache.get(str(qasm_path))
        if metrics is None:
            legacy = compute_zx_splitting_metrics_from_qasm(qasm_path)
            paper = compute_paper_zx_splitting_metrics_from_qasm(
                qasm_path,
                rewrite_level=rewrite_level,  # type: ignore[arg-type]
            )
            metrics = {
                **legacy_fields(legacy),
                **paper_fields(paper),
            }
            qasm_cache[str(qasm_path)] = metrics

        original = original_cache.get(target)
        if target not in original_cache:
            original_path = resolve_original_qasm(target)
            if original_path is None:
                original = None
            else:
                original_metrics = compute_paper_zx_splitting_metrics_from_qasm(
                    original_path,
                    rewrite_level=rewrite_level,  # type: ignore[arg-type]
                )
                original = {
                    "original_qasm_path": str(original_path),
                    "original_paper_zx_split_status": original_metrics.get(
                        "paper_zx_split_status"
                    ),
                    "original_paper_zx_total_depth": original_metrics.get(
                        "paper_zx_total_depth"
                    ),
                }
            original_cache[target] = original

        paper_primary = ""
        paper_primary_status = "missing-original"
        if original is not None:
            denom = coerce_float(original.get("original_paper_zx_total_depth"))
            numer = coerce_float(metrics.get("paper_zx_best_nonclifford_depth"))
            if original.get("original_paper_zx_split_status") != "ok":
                paper_primary_status = "missing-zx"
            elif denom is None or denom <= 0 or numer is None:
                paper_primary_status = "invalid-depth"
            else:
                paper_primary = numer / max(denom, 1.0)
                paper_primary_status = "ok"

        output.append(
            {
                **base,
                **metrics,
                "original_qasm_path": "" if original is None else original.get("original_qasm_path", ""),
                "paper_primary_nc_depth_ratio": paper_primary,
                "paper_structural_target_status": paper_primary_status,
            }
        )
    return output


def target_name(row: dict[str, Any]) -> str:
    return row.get("target") or row.get("circuit_id") or ""


def objective_name(row: dict[str, Any]) -> str:
    return (
        row.get("objective_variant")
        or row.get("tensor_v3_profile")
        or row.get("source_methods")
        or row.get("method")
        or ""
    )


def legacy_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "legacy_zx_split_status": metrics.get("zx_split_status", ""),
        "legacy_zx_total_depth": metrics.get("zx_total_depth", ""),
        "legacy_zx_best_side": metrics.get("zx_best_side", ""),
        "legacy_zx_best_clifford_depth": metrics.get("zx_best_clifford_depth", ""),
        "legacy_zx_best_nonclifford_depth": metrics.get("zx_best_nonclifford_depth", ""),
        "legacy_zx_best_clifford_fraction": metrics.get("zx_best_clifford_fraction", ""),
        "legacy_zx_num_nonclifford_spiders": metrics.get("zx_num_nonclifford_spiders", ""),
        "legacy_zx_left_closure_iters": metrics.get("zx_left_closure_iters", ""),
        "legacy_zx_right_closure_iters": metrics.get("zx_right_closure_iters", ""),
        "legacy_zx_split_error": metrics.get("zx_split_error", ""),
    }


def paper_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_zx_split_status": metrics.get("paper_zx_split_status", ""),
        "paper_zx_detector_variant": metrics.get("paper_zx_detector_variant", ""),
        "paper_zx_rewrite_status": metrics.get("paper_zx_rewrite_status", ""),
        "paper_zx_total_depth": metrics.get("paper_zx_total_depth", ""),
        "paper_zx_best_side": metrics.get("paper_zx_best_side", ""),
        "paper_zx_best_clifford_depth": metrics.get("paper_zx_best_clifford_depth", ""),
        "paper_zx_best_nonclifford_depth": metrics.get("paper_zx_best_nonclifford_depth", ""),
        "paper_zx_best_clifford_fraction": metrics.get("paper_zx_best_clifford_fraction", ""),
        "paper_zx_num_nonclifford_spiders": metrics.get("paper_zx_num_nonclifford_spiders", ""),
        "paper_zx_left_closure_iters": metrics.get("paper_zx_left_closure_iters", ""),
        "paper_zx_right_closure_iters": metrics.get("paper_zx_right_closure_iters", ""),
        "paper_zx_vertices_before_rewrite": metrics.get("paper_zx_vertices_before_rewrite", ""),
        "paper_zx_vertices_after_rewrite": metrics.get("paper_zx_vertices_after_rewrite", ""),
        "paper_zx_clifford_rewrite_rounds": metrics.get("paper_zx_clifford_rewrite_rounds", ""),
        "paper_zx_split_error": metrics.get("paper_zx_split_error", ""),
    }


def resolve_candidate_qasm(row: dict[str, Any]) -> Path | None:
    for key in ("candidate_qasm_path", "qasm_path", "assembled_qasm_path"):
        path = resolve_path(row.get(key, ""), allow_basename_search=True)
        if path is not None:
            return path

    summary_path = resolve_path(row.get("summary_path", ""), allow_basename_search=False)
    if summary_path is not None:
        for raw in qasm_values_from_json(summary_path):
            path = resolve_path(raw, allow_basename_search=True)
            if path is not None:
                return path

    candidate_dir = resolve_path(row.get("candidate_dir", ""), allow_basename_search=False)
    if candidate_dir is not None and candidate_dir.is_dir():
        qasm_files = sorted(candidate_dir.glob("*.qasm"))
        assembled = [path for path in qasm_files if path.name == "assembled.qasm"]
        if assembled:
            return assembled[0]
        if qasm_files:
            return qasm_files[0]
    return None


def resolve_path(raw: Any, *, allow_basename_search: bool = False) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path]
    cluster_prefix = Path("/home/CIN/cacl2/Deep-Learning/project")
    if path.is_absolute():
        try:
            rel = path.relative_to(cluster_prefix)
            candidates.append(PROJECT_ROOT / rel)
        except ValueError:
            pass
    else:
        candidates.append(PROJECT_ROOT / path)
        parts = path.parts
        if parts and parts[0] == "results":
            candidates.append(PROJECT_ROOT / "results" / "archive" / "old_experiments" / Path(*parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if allow_basename_search and path.name and path.suffix == ".qasm":
        return find_by_basename(path.name, hint=text)
    return None


_BASENAME_CACHE: dict[str, list[Path]] = {}


def find_by_basename(name: str, *, hint: str = "") -> Path | None:
    if name not in _BASENAME_CACHE:
        _BASENAME_CACHE[name] = sorted((PROJECT_ROOT / "results").rglob(name))
    matches = _BASENAME_CACHE[name]
    if not matches:
        return None
    if hint:
        hint_parts = [part for part in Path(hint).parts if part not in {"results", "."}]
        scored = []
        for match in matches:
            score = sum(1 for part in hint_parts if part in match.parts)
            scored.append((score, len(str(match)), match))
        return max(scored)[2]
    return matches[0]


def qasm_values_from_json(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    values: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and ("qasm" in key.lower() or value.endswith(".qasm")):
            values.append(value)

    visit(data)
    return values


def resolve_original_qasm(target: str) -> Path | None:
    if not target:
        return None
    candidates = sorted((PROJECT_ROOT / "results").rglob(f"{target}.qasm"))
    filtered = [
        path
        for path in candidates
        if "/candidates/" not in str(path)
        and "/structural_combinations/" not in str(path)
        and path.name == f"{target}.qasm"
    ]
    if not filtered:
        return None

    def key(path: Path) -> tuple[int, int, str]:
        text = str(path)
        archive_penalty = 1 if "/archive/" in text else 0
        assembled_penalty = 1 if "assembled" in text else 0
        return (archive_penalty, assembled_penalty, text)

    return sorted(filtered, key=key)[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_csv",
        "target",
        "objective_variant",
        "materializer",
        "candidate_id",
        "qasm_status",
        "qasm_path",
        "original_qasm_path",
        "tcount",
        "tdepth",
        "qasm_depth",
        "legacy_primary_nc_depth_ratio_input",
        "legacy_structural_status_input",
        "legacy_zx_split_status",
        "legacy_zx_total_depth",
        "legacy_zx_best_side",
        "legacy_zx_best_clifford_depth",
        "legacy_zx_best_nonclifford_depth",
        "legacy_zx_best_clifford_fraction",
        "legacy_zx_num_nonclifford_spiders",
        "legacy_zx_left_closure_iters",
        "legacy_zx_right_closure_iters",
        "paper_zx_split_status",
        "paper_zx_detector_variant",
        "paper_zx_rewrite_status",
        "paper_zx_total_depth",
        "paper_zx_best_side",
        "paper_zx_best_clifford_depth",
        "paper_zx_best_nonclifford_depth",
        "paper_zx_best_clifford_fraction",
        "paper_zx_num_nonclifford_spiders",
        "paper_zx_left_closure_iters",
        "paper_zx_right_closure_iters",
        "paper_zx_vertices_before_rewrite",
        "paper_zx_vertices_after_rewrite",
        "paper_zx_clifford_rewrite_rounds",
        "paper_primary_nc_depth_ratio",
        "paper_structural_target_status",
        "legacy_zx_split_error",
        "paper_zx_split_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_display = display_path(csv_path)
    computed = [
        row
        for row in rows
        if row.get("legacy_zx_split_status") == "ok"
        and row.get("paper_zx_split_status") == "ok"
    ]
    detector_delta = classify_detector_delta(computed)
    pairwise = pairwise_objective_summary(computed)

    lines = [
        "# ZX Border Detector Comparison",
        "",
        f"CSV: `{csv_display}`.",
        "",
        "This compares the existing row-border ZX detector with a paper-style detector inspired by arXiv:2504.16004. The paper-style detector first converts the PyZX graph to graph-like form, applies Clifford simplification as an operational non-Clifford pushing stage, and then applies the same recursive crossing-gate closure used by the baseline detector.",
        "",
        f"Rows read: {len(rows)}.",
        f"Rows with local QASM and both detectors OK: {len(computed)}.",
        f"Rows missing local QASM: {sum(1 for row in rows if row.get('qasm_status') == 'missing-qasm')}.",
        "",
        "## Detector Delta",
        "",
        "| relation | rows |",
        "|---|---:|",
    ]
    for key in ("paper-smaller-core", "tie", "paper-larger-core"):
        lines.append(f"| {key} | {detector_delta[key]} |")

    lines.extend(
        [
            "",
            "## Objective-Level Comparison",
            "",
            "Lower non-Clifford depth/ratio is better. Each row compares the best available candidate for `factor_count` against a split-aware objective on the same target.",
            "",
            "| objective | legacy better/tie/worse | paper better/tie/worse | comparable targets |",
            "|---|---:|---:|---:|",
        ]
    )
    for objective, stats in sorted(pairwise.items()):
        lines.append(
            f"| {objective} | {stats['legacy_better']}/{stats['legacy_tie']}/{stats['legacy_worse']} | "
            f"{stats['paper_better']}/{stats['paper_tie']}/{stats['paper_worse']} | {stats['targets']} |"
        )

    examples = strongest_examples(computed)
    if examples:
        lines.extend(["", "## Largest Paper-Style Core Reductions", "", "| target | objective | materializer | legacy NC depth | paper NC depth |", "|---|---|---|---:|---:|"])
        for row in examples:
            lines.append(
                "| {target} | {objective_variant} | {materializer} | {legacy_zx_best_nonclifford_depth} | {paper_zx_best_nonclifford_depth} |".format(
                    **row
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this report as an external audit, not as a replacement for the AlphaQ objective. If split-aware objectives improve under `paper_primary_nc_depth_ratio` or `paper_zx_best_nonclifford_depth`, then the internal AlphaQ-only objective is producing candidates that look better under a ZX-calculus splitting lens.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def classify_detector_delta(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"paper-smaller-core": 0, "tie": 0, "paper-larger-core": 0}
    for row in rows:
        legacy = coerce_float(row.get("legacy_zx_best_nonclifford_depth"))
        paper = coerce_float(row.get("paper_zx_best_nonclifford_depth"))
        if legacy is None or paper is None:
            continue
        if paper < legacy:
            counts["paper-smaller-core"] += 1
        elif math.isclose(paper, legacy):
            counts["tie"] += 1
        else:
            counts["paper-larger-core"] += 1
    return counts


def pairwise_objective_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_target_objective: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = row.get("target", "")
        objective = row.get("objective_variant", "")
        if target and objective:
            by_target_objective[(target, objective)].append(row)

    best = {
        key: min(items, key=ranking_key)
        for key, items in by_target_objective.items()
    }
    objectives = sorted(
        {
            objective
            for _target, objective in best
            if objective != "factor_count"
            and ("mixed" in objective or "pair" in objective or "tensor" in objective)
        }
    )
    summary = {
        objective: {
            "legacy_better": 0,
            "legacy_tie": 0,
            "legacy_worse": 0,
            "paper_better": 0,
            "paper_tie": 0,
            "paper_worse": 0,
            "targets": 0,
        }
        for objective in objectives
    }
    targets = {target for target, objective in best if objective == "factor_count"}
    for target in targets:
        baseline = best.get((target, "factor_count"))
        if baseline is None:
            continue
        for objective in objectives:
            contender = best.get((target, objective))
            if contender is None:
                continue
            summary[objective]["targets"] += 1
            compare_metric(
                summary[objective],
                contender,
                baseline,
                "legacy_zx_best_nonclifford_depth",
                "legacy",
            )
            compare_metric(
                summary[objective],
                contender,
                baseline,
                "paper_primary_nc_depth_ratio",
                "paper",
                fallback_key="paper_zx_best_nonclifford_depth",
            )
    return summary


def ranking_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        finite_or_inf(row.get("paper_primary_nc_depth_ratio")),
        finite_or_inf(row.get("paper_zx_best_nonclifford_depth")),
        finite_or_inf(row.get("tcount")),
        row.get("materializer", ""),
    )


def compare_metric(
    stats: dict[str, int],
    contender: dict[str, Any],
    baseline: dict[str, Any],
    key: str,
    prefix: str,
    *,
    fallback_key: str | None = None,
) -> None:
    contender_value = coerce_float(contender.get(key))
    baseline_value = coerce_float(baseline.get(key))
    if (contender_value is None or baseline_value is None) and fallback_key is not None:
        contender_value = coerce_float(contender.get(fallback_key))
        baseline_value = coerce_float(baseline.get(fallback_key))
    if contender_value is None or baseline_value is None:
        return
    if contender_value < baseline_value:
        stats[f"{prefix}_better"] += 1
    elif math.isclose(contender_value, baseline_value):
        stats[f"{prefix}_tie"] += 1
    else:
        stats[f"{prefix}_worse"] += 1


def finite_or_inf(value: Any) -> float:
    numeric = coerce_float(value)
    return math.inf if numeric is None else numeric


def strongest_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        legacy = coerce_float(row.get("legacy_zx_best_nonclifford_depth"))
        paper = coerce_float(row.get("paper_zx_best_nonclifford_depth"))
        if legacy is None or paper is None:
            continue
        scored.append((legacy - paper, row))
    return [row for score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:8] if score > 0]


def main() -> None:
    args = parse_args()
    input_csvs = args.input_csv or default_input_csvs()
    rows = read_rows(input_csvs, max_rows=args.max_rows)
    compared = comparison_rows(rows, rewrite_level=args.rewrite_level)
    write_csv(args.output_csv, compared)
    write_report(args.report_path, compared, args.output_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")


if __name__ == "__main__":
    main()
