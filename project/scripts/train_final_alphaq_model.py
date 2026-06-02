from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._analysis_common import DEFAULT_CSV_ROOT
from scripts._analysis_common import DEFAULT_REPORTS_ROOT
from scripts._analysis_common import DEFAULT_RESULTS_ROOT
from scripts._analysis_common import ensure_dir
from scripts._analysis_common import natural_sort_key
from scripts._analysis_common import write_csv_rows
from scripts._analysis_common import write_json
from scripts._manifest import append_command
from scripts.alphatensor_reranker import DEFAULT_SEED_STRIDE
from scripts.alphatensor_reranker import FEATURE_COLUMNS
from scripts.alphatensor_reranker import FINAL_RERANKER_METHOD
from scripts.alphatensor_reranker import LABEL_COLUMN
from scripts.alphatensor_reranker import MODEL_KIND_MLP
from scripts.alphatensor_reranker import MODEL_KIND_PAIRWISE
from scripts.alphatensor_reranker import MODEL_KIND_PAIRWISE_MLP
from scripts.alphatensor_reranker import NO_DEPENDENCY_FEATURE_COLUMNS
from scripts.alphatensor_reranker import DEPENDENCY_FEATURE_COLUMNS
from scripts.alphatensor_reranker import PREDICTION_COLUMN
from scripts.alphatensor_reranker import baseline_rows
from scripts.alphatensor_reranker import coerce_float
from scripts.alphatensor_reranker import comparison_rows
from scripts.alphatensor_reranker import leave_one_circuit_out_eval
from scripts.alphatensor_reranker import materialize_reranker_selection
from scripts.alphatensor_reranker import model_payload
from scripts.alphatensor_reranker import predictions_rows
from scripts.alphatensor_reranker import train_ensemble
from scripts.alphatensor_reranker import valid_candidate_rows


DEFAULT_FRONTIER_CSV = (
    DEFAULT_RESULTS_ROOT / "public_resynth_structural" / "candidate_frontier.csv"
)
DEFAULT_MODEL_JSON = DEFAULT_RESULTS_ROOT / "models" / "alphaq_final_pairwise_ranker.json"
DEFAULT_PREDICTIONS_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_predictions.csv"
DEFAULT_EVAL_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_loco_eval.csv"
DEFAULT_COMPARISON_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_comparison.csv"
DEFAULT_BASELINES_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_baselines.csv"
DEFAULT_ABLATION_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_ablations.csv"
DEFAULT_FRONTIER_VERIFICATION_CSV = (
    DEFAULT_RESULTS_ROOT
    / "verification"
    / "frontier"
    / "alphaq_candidate_frontier_verification.csv"
)
DEFAULT_TOLERANCE_SWEEP_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_tolerance_sweep.csv"
DEFAULT_SEED_ROBUSTNESS_CSV = DEFAULT_CSV_ROOT / "alphaq_final_model_seed_robustness.csv"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_ROOT / "alphaq_final_model_report.md"
DEFAULT_SELECTION_ROOT = DEFAULT_RESULTS_ROOT / "public_resynth_alphaq_final"
DEFAULT_TOLERANCES = (0.0, 0.025, 0.05, 0.10, 0.20)
DEFAULT_ROBUSTNESS_SEEDS = tuple(2026 + index * DEFAULT_SEED_STRIDE for index in range(11))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def attach_frontier_verification(
    rows: list[dict[str, str]],
    verification_csv: Path | None,
) -> list[dict[str, Any]]:
    if verification_csv is None or not verification_csv.exists():
        return [
            {
                **row,
                "frontier_verification_status": "not-run",
                "frontier_verification_error": None,
                "frontier_proof_path": None,
            }
            for row in rows
        ]
    verification_index = {
        row["candidate_id"]: row for row in load_csv_rows(verification_csv)
    }
    enriched = []
    for row in rows:
        verification = verification_index.get(row.get("candidate_id"))
        enriched.append(
            {
                **row,
                "frontier_verification_status": (
                    None if verification is None else verification.get("verification_status")
                )
                or "not-run",
                "frontier_verification_error": (
                    None if verification is None else verification.get("verification_error")
                ),
                "frontier_proof_path": (
                    None if verification is None else verification.get("proof_path")
                ),
            }
        )
    return enriched


def selected_by_circuit(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in prediction_rows if row.get("reranker_selected")]


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = _finite_values(row.get("primary_regret_vs_structural_best") for row in rows)
    gains = _finite_values(row.get("primary_gain_vs_tcount_best") for row in rows)
    t_delta_structural = _finite_values(row.get("tcount_delta_vs_structural_best") for row in rows)
    t_delta_tcount = _finite_values(row.get("tcount_delta_vs_tcount_best") for row in rows)
    hits = [
        row.get("reranker_candidate_id") == row.get("structural_best_candidate_id")
        for row in rows
    ]
    return {
        "num_circuits": len(rows),
        "hit_structural_best_rate": 0.0 if not hits else sum(hits) / len(hits),
        "mean_structural_regret": _mean(regrets),
        "max_structural_regret": max(regrets) if regrets else None,
        "mean_gain_vs_tcount_best": _mean(gains),
        "total_tcount_delta_vs_structural_best": sum(t_delta_structural)
        if t_delta_structural
        else None,
        "total_tcount_delta_vs_tcount_best": sum(t_delta_tcount)
        if t_delta_tcount
        else None,
    }


def tolerance_sweep_rows(
    rows: list[dict[str, Any]],
    model: Any,
    tolerances: tuple[float, ...],
) -> list[dict[str, Any]]:
    sweep_rows: list[dict[str, Any]] = []
    for tolerance in tolerances:
        prediction_rows = predictions_rows(rows, model, prediction_tolerance=tolerance)
        comparison = comparison_rows(prediction_rows)
        summary = summarize_comparison(comparison)
        sweep_rows.append(
            {
                "prediction_tolerance": tolerance,
                **summary,
                "selected_candidates": ";".join(
                    f"{row['circuit_id']}={row['candidate_id']}"
                    for row in selected_by_circuit(prediction_rows)
                ),
            }
        )
    return sweep_rows


def seed_robustness_rows(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    model_kind: str,
    hidden_size: int,
    ensemble_size: int,
    prediction_tolerance: float,
) -> list[dict[str, Any]]:
    robustness_rows: list[dict[str, Any]] = []
    for seed in seeds:
        model = train_ensemble(
            rows,
            model_kind=model_kind,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            ensemble_size=ensemble_size,
        )
        prediction_rows = predictions_rows(
            rows,
            model,
            prediction_tolerance=prediction_tolerance,
        )
        comparison = comparison_rows(prediction_rows)
        comparison_by_circuit = {row["circuit_id"]: row for row in comparison}
        for selected in selected_by_circuit(prediction_rows):
            comparison_row = comparison_by_circuit[selected["circuit_id"]]
            robustness_rows.append(
                {
                    "seed": seed,
                    "circuit_id": selected["circuit_id"],
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_structural_cost": selected.get(LABEL_COLUMN),
                    "selected_predicted_structural_cost": selected.get(PREDICTION_COLUMN),
                    "selected_tcount": selected.get("tcount_after"),
                    "structural_best_candidate_id": comparison_row.get(
                        "structural_best_candidate_id"
                    ),
                    "primary_regret_vs_structural_best": comparison_row.get(
                        "primary_regret_vs_structural_best"
                    ),
                    "tcount_delta_vs_structural_best": comparison_row.get(
                        "tcount_delta_vs_structural_best"
                    ),
                }
            )
    return robustness_rows


def seed_stability_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    circuits = sorted({row["circuit_id"] for row in rows}, key=natural_sort_key)
    for circuit_id in circuits:
        subset = [row for row in rows if row["circuit_id"] == circuit_id]
        counts: dict[str, int] = {}
        regrets = _finite_values(row.get("primary_regret_vs_structural_best") for row in subset)
        for row in subset:
            selected = row["selected_candidate_id"]
            counts[selected] = counts.get(selected, 0) + 1
        winner, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        lines.append(
            f"- `{circuit_id}`: {len(counts)} unique selections; most frequent "
            f"`{winner}` in {count}/{len(subset)} seeds; mean regret {_fmt(_mean(regrets))}."
        )
    return lines


def baseline_summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for objective in sorted({row["objective"] for row in rows}):
        subset = [row for row in rows if row["objective"] == objective]
        regrets = _finite_values(row.get("primary_regret_vs_structural_best") for row in subset)
        t_delta = _finite_values(row.get("tcount_delta_vs_structural_best") for row in subset)
        lines.append(
            f"- `{objective}`: mean regret {_fmt(_mean(regrets))}; "
            f"total T-delta vs structural {_fmt(sum(t_delta), 0)}."
        )
    return lines


def ablation_rows(
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    ensemble_size: int,
    hidden_size: int,
    prediction_tolerance: float,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for objective, subset in _group_by_objective(baseline_rows(rows)).items():
        regrets = _finite_values(
            row.get("primary_regret_vs_structural_best") for row in subset
        )
        gains = _finite_values(row.get("primary_gain_vs_tcount_best") for row in subset)
        hits = [
            row.get("selected_candidate_id") == row.get("structural_best_candidate_id")
            for row in subset
        ]
        output_rows.append(
            {
                "ablation": f"baseline:{objective}",
                "model_kind": "baseline",
                "feature_set": objective,
                "num_circuits": len(subset),
                "hit_rate": 0.0 if not hits else sum(hits) / len(hits),
                "mean_regret": _mean(regrets),
                "max_regret": max(regrets) if regrets else None,
                "mean_gain_vs_tcount_best": _mean(gains),
            }
        )

    variants = [
        ("mlp_regressor_all", MODEL_KIND_MLP, FEATURE_COLUMNS),
        ("linear_pairwise_all", MODEL_KIND_PAIRWISE, FEATURE_COLUMNS),
        ("pairwise_mlp_no_dependency", MODEL_KIND_PAIRWISE_MLP, NO_DEPENDENCY_FEATURE_COLUMNS),
        ("pairwise_mlp_dependency_only", MODEL_KIND_PAIRWISE_MLP, DEPENDENCY_FEATURE_COLUMNS),
        ("pairwise_mlp_all", MODEL_KIND_PAIRWISE_MLP, FEATURE_COLUMNS),
    ]
    for name, model_kind, feature_columns in variants:
        if not feature_columns:
            continue
        eval_rows = leave_one_circuit_out_eval(
            rows,
            model_kind=model_kind,
            feature_columns=feature_columns,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            prediction_tolerance=prediction_tolerance,
            ensemble_size=ensemble_size,
        )
        regrets = _finite_values(row.get("primary_regret_vs_true_best") for row in eval_rows)
        gains = _finite_values(row.get("primary_gain_vs_tcount_best") for row in eval_rows)
        hits = [bool(row.get("hit_true_best")) for row in eval_rows]
        output_rows.append(
            {
                "ablation": name,
                "model_kind": model_kind,
                "feature_set": _feature_set_name(feature_columns),
                "num_features": len(feature_columns),
                "num_circuits": len(eval_rows),
                "hit_rate": 0.0 if not hits else sum(hits) / len(hits),
                "mean_regret": _mean(regrets),
                "max_regret": max(regrets) if regrets else None,
                "mean_gain_vs_tcount_best": _mean(gains),
            }
        )
    return output_rows


def ablation_summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    return [
        f"- `{row['ablation']}`: hit {_fmt(row.get('hit_rate'))}, "
        f"mean regret {_fmt(row.get('mean_regret'))}, "
        f"gain-vs-T {_fmt(row.get('mean_gain_vs_tcount_best'))}."
        for row in rows
    ]


def _group_by_objective(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("objective")), []).append(row)
    return grouped


def _feature_set_name(feature_columns: tuple[str, ...]) -> str:
    if feature_columns == FEATURE_COLUMNS:
        return "all"
    if feature_columns == NO_DEPENDENCY_FEATURE_COLUMNS:
        return "no_dependency"
    if feature_columns == DEPENDENCY_FEATURE_COLUMNS:
        return "dependency_only"
    return "custom"


def selected_lines(prediction_rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in selected_by_circuit(prediction_rows):
        lines.append(
            f"- `{row['circuit_id']}`: `{row['candidate_id']}`, "
            f"cost {_fmt(row.get(LABEL_COLUMN))}, pred {_fmt(row.get(PREDICTION_COLUMN))}, "
            f"ZX audit {_fmt(row.get('primary_nc_depth_ratio'))}, T={row.get('tcount_after')}."
        )
    return lines


def eval_lines(rows: list[dict[str, Any]]) -> list[str]:
    return [
        f"- `{row['circuit_id']}`: selected `{row['selected_candidate_id']}`, "
        f"oracle `{row['true_best_candidate_id']}`, regret "
        f"{_fmt(row.get('primary_regret_vs_true_best'))}, "
        f"gain-vs-T-best {_fmt(row.get('primary_gain_vs_tcount_best'))}."
        for row in rows
    ]


def write_report(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    tolerance_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    model_json: Path,
    selection_summary_csv: Path,
    args: argparse.Namespace,
) -> Path:
    circuits = sorted({row["circuit_id"] for row in rows}, key=natural_sort_key)
    full_summary = summarize_comparison(comparison)
    loco_regrets = _finite_values(row.get("primary_regret_vs_true_best") for row in eval_rows)
    loco_hits = [
        row.get("selected_candidate_id") == row.get("true_best_candidate_id")
        for row in eval_rows
    ]
    best_tolerance = min(
        tolerance_rows,
        key=lambda row: (
            _rank_value(row.get("mean_structural_regret")),
            abs(coerce_float(row.get("total_tcount_delta_vs_tcount_best")) or 0.0),
            _rank_value(row.get("prediction_tolerance")),
        ),
    )
    text = [
        "# AlphaQ Final Pairwise Reranker",
        "",
        "## Training Setup",
        "",
        f"- Frontier: `{args.frontier_csv}`.",
        f"- Frontier verification: `{args.frontier_verification_csv}`.",
        f"- Require formal equal candidates: `{args.require_formal_equal}`.",
        f"- Candidates: {len(rows)} across {len(circuits)} circuits.",
        f"- Circuits: {', '.join(f'`{item}`' for item in circuits)}.",
        f"- Model: `{args.model_kind}` ensemble.",
        f"- Epochs: {args.epochs}; ensemble size: {args.ensemble_size}.",
        f"- Learning rate: {args.learning_rate}; weight decay: {args.weight_decay}.",
        f"- Label: `{LABEL_COLUMN}` = AlphaQ dependency-closed core area ratio.",
        "- Features exclude ZX/PyZX targets and exclude candidate enumeration.",
        f"- Feature columns: {', '.join(f'`{item}`' for item in FEATURE_COLUMNS)}.",
        f"- Model JSON: `{model_json}`.",
        f"- Selection summary: `{selection_summary_csv}`.",
        "",
        "## Full-Fit Selection",
        "",
        f"- Hit structural oracle rate: {_fmt(full_summary['hit_structural_best_rate'])}.",
        f"- Mean structural regret: {_fmt(full_summary['mean_structural_regret'])}.",
        f"- Max structural regret: {_fmt(full_summary['max_structural_regret'])}.",
        f"- Mean gain vs T-count best: {_fmt(full_summary['mean_gain_vs_tcount_best'])}.",
        *selected_lines(prediction_rows),
        "",
        "## Leave-One-Circuit-Out",
        "",
        f"- Hit rate: {_fmt(0.0 if not loco_hits else sum(loco_hits) / len(loco_hits))}.",
        f"- Mean regret: {_fmt(_mean(loco_regrets))}.",
        f"- Max regret: {_fmt(max(loco_regrets) if loco_regrets else None)}.",
        *eval_lines(eval_rows),
        "",
        "## Tolerance Sweep",
        "",
        (
            f"- Best sweep row by structural regret: tolerance "
            f"{_fmt(best_tolerance['prediction_tolerance'])}, mean regret "
            f"{_fmt(best_tolerance['mean_structural_regret'])}, selected "
            f"{best_tolerance['selected_candidates']}."
        ),
        *[
            f"- tol={_fmt(row['prediction_tolerance'])}: mean regret "
            f"{_fmt(row['mean_structural_regret'])}, hit "
            f"{_fmt(row['hit_structural_best_rate'])}, total T-delta vs T-best "
            f"{_fmt(row['total_tcount_delta_vs_tcount_best'], 0)}."
            for row in tolerance_rows
        ],
        "",
        "## Seed Robustness",
        "",
        f"- Seeds: {len(set(row['seed'] for row in robustness_rows))}.",
        *seed_stability_summary(robustness_rows),
        "",
        "## Baselines",
        "",
        *baseline_summary_lines(baselines),
        "",
        "## Ablations",
        "",
        *ablation_summary_lines(ablations),
        "",
        "## Scientific Reading",
        "",
        (
            "This final model is a ranking model, not a scalar-only regressor: the training "
            "objective is the within-benchmark order induced by the AlphaQ Clifford-splitting "
            "proxy. ZX/PyZX-derived columns are carried through as external audits only; they "
            "are not features, labels, or tie-breakers."
        ),
        (
            "Because the frontier is still curated, the strongest claim remains a controlled "
            "mechanistic result on the formally auditable benchmark subset. The artifacts above "
            "are meant to make that claim falsifiable: exact structural oracle, learned ranker, "
            "T-count baseline, tolerance sensitivity, seed stability, and ZX audit all live in "
            "separate tables."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the final robust AlphaQ-only pairwise reranker."
    )
    parser.add_argument("--frontier-csv", type=Path, default=DEFAULT_FRONTIER_CSV)
    parser.add_argument(
        "--frontier-verification-csv",
        type=Path,
        default=DEFAULT_FRONTIER_VERIFICATION_CSV,
    )
    parser.add_argument("--require-formal-equal", action="store_true")
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--baselines-csv", type=Path, default=DEFAULT_BASELINES_CSV)
    parser.add_argument("--ablation-csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument("--tolerance-sweep-csv", type=Path, default=DEFAULT_TOLERANCE_SWEEP_CSV)
    parser.add_argument("--seed-robustness-csv", type=Path, default=DEFAULT_SEED_ROBUSTNESS_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--selection-output-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument(
        "--model-kind",
        choices=(MODEL_KIND_MLP, MODEL_KIND_PAIRWISE, MODEL_KIND_PAIRWISE_MLP),
        default=MODEL_KIND_PAIRWISE_MLP,
    )
    parser.add_argument("--hidden-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12_000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ensemble-size", type=int, default=23)
    parser.add_argument("--prediction-tolerance", type=float, default=0.05)
    parser.add_argument("--tolerances", type=parse_float_tuple, default=DEFAULT_TOLERANCES)
    parser.add_argument("--robustness-seeds", type=parse_int_tuple, default=DEFAULT_ROBUSTNESS_SEEDS)
    parser.add_argument("--robustness-epochs", type=int, default=8_000)
    parser.add_argument("--robustness-ensemble-size", type=int, default=9)
    parser.add_argument("--ablation-epochs", type=int, default=4_000)
    parser.add_argument("--ablation-ensemble-size", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = attach_frontier_verification(
        valid_candidate_rows(load_csv_rows(args.frontier_csv)),
        args.frontier_verification_csv,
    )
    if args.require_formal_equal:
        rows = [
            row for row in rows if row.get("frontier_verification_status") == "equal"
        ]
    if not rows:
        raise SystemExit(f"No valid AlphaQ frontier rows found in {args.frontier_csv}")

    eval_rows = leave_one_circuit_out_eval(
        rows,
        model_kind=args.model_kind,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        prediction_tolerance=args.prediction_tolerance,
        ensemble_size=args.ensemble_size,
    )
    model = train_ensemble(
        rows,
        model_kind=args.model_kind,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        ensemble_size=args.ensemble_size,
    )
    prediction_rows = predictions_rows(
        rows,
        model,
        prediction_tolerance=args.prediction_tolerance,
    )
    comparison = comparison_rows(prediction_rows)
    baselines = baseline_rows(rows)
    ablations = ablation_rows(
        rows,
        epochs=args.ablation_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        ensemble_size=args.ablation_ensemble_size,
        hidden_size=args.hidden_size,
        prediction_tolerance=args.prediction_tolerance,
    )
    tolerance_rows = tolerance_sweep_rows(rows, model, args.tolerances)
    robustness_rows = seed_robustness_rows(
        rows,
        seeds=args.robustness_seeds,
        epochs=args.robustness_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_kind=args.model_kind,
        hidden_size=args.hidden_size,
        ensemble_size=args.robustness_ensemble_size,
        prediction_tolerance=args.prediction_tolerance,
    )
    selection_rows, selection_summary_csv, selection_summary_json = (
        materialize_reranker_selection(
            prediction_rows,
            output_root=args.selection_output_root,
            model_path=args.model_json,
            prediction_tolerance=args.prediction_tolerance,
            selection_objective=f"alphaq_final_{args.model_kind}",
            method_name=FINAL_RERANKER_METHOD,
        )
    )

    write_csv_rows(prediction_rows, args.predictions_csv)
    write_csv_rows(eval_rows, args.eval_csv)
    write_csv_rows(comparison, args.comparison_csv)
    write_csv_rows(baselines, args.baselines_csv)
    write_csv_rows(ablations, args.ablation_csv)
    write_csv_rows(tolerance_rows, args.tolerance_sweep_csv)
    write_csv_rows(robustness_rows, args.seed_robustness_csv)
    write_json(model_payload(model, args, rows), args.model_json)
    report_path = write_report(
        output_path=args.report_path,
        rows=rows,
        prediction_rows=prediction_rows,
        eval_rows=eval_rows,
        comparison=comparison,
        baselines=baselines,
        ablations=ablations,
        tolerance_rows=tolerance_rows,
        robustness_rows=robustness_rows,
        model_json=args.model_json,
        selection_summary_csv=selection_summary_csv,
        args=args,
    )
    write_json(
        {
            "frontier_csv": str(args.frontier_csv),
            "frontier_verification_csv": str(args.frontier_verification_csv),
            "require_formal_equal": args.require_formal_equal,
            "model_json": str(args.model_json),
            "predictions_csv": str(args.predictions_csv),
            "eval_csv": str(args.eval_csv),
            "comparison_csv": str(args.comparison_csv),
            "baselines_csv": str(args.baselines_csv),
            "ablation_csv": str(args.ablation_csv),
            "tolerance_sweep_csv": str(args.tolerance_sweep_csv),
            "seed_robustness_csv": str(args.seed_robustness_csv),
            "report_path": str(report_path),
            "selection_summary_csv": str(selection_summary_csv),
            "selection_summary_json": str(selection_summary_json),
            "num_candidates": len(rows),
            "num_selected": len(selection_rows),
        },
        args.selection_output_root / "alphaq_final_model_manifest.json",
    )
    append_command(
        {
            "tool": "train_final_alphaq_model.py",
            "command": " ".join(sys.argv),
            "cwd": str(PROJECT_ROOT),
            "frontier_csv": str(args.frontier_csv),
            "model_json": str(args.model_json),
            "selection_summary_csv": str(selection_summary_csv),
            "report_path": str(report_path),
            "exit_code": 0,
        }
    )
    print(
        json.dumps(
            {
                "num_candidates": len(rows),
                "num_selected": len(selection_rows),
                "model_json": str(args.model_json),
                "report_path": str(report_path),
                "selection_summary_csv": str(selection_summary_csv),
            },
            indent=2,
        )
    )
    return 0


def _finite_values(values: Any) -> list[float]:
    finite: list[float] = []
    for value in values:
        numeric = coerce_float(value)
        if numeric is not None:
            finite.append(numeric)
    return finite


def _mean(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = coerce_float(value)
    return "NA" if numeric is None else f"{numeric:.{digits}f}"


def _rank_value(value: Any) -> float:
    numeric = coerce_float(value)
    return float("inf") if numeric is None else numeric


if __name__ == "__main__":
    raise SystemExit(main())
