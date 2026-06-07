from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_oracle_equivalence import oracle_info
from scripts.run_best_objective_beam_ablation import parse_paths
from scripts.run_best_objective_beam_ablation import read_csv
from scripts.run_best_objective_beam_ablation import safe_ratio
from scripts.run_objective_beam_policy_grid import best_current_beams
from scripts.run_objective_beam_policy_grid import best_policy_beams
from scripts.structural_target import coerce_float


DEFAULT_DECOMP_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_holdout_ablation.csv",
)
DEFAULT_GRID_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid.csv"
DEFAULT_CURRENT_BEAM_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_beam_materializer_holdout_ablation.csv",
)
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_learned_objective_selector_summary.csv"
DEFAULT_DETAIL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_learned_objective_selector_details.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_learned_objective_selector.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_learned_objective_selector.png"

ALPHAQ_FEATURES = (
    "factor_count",
    "factor_qubit_concentration_index",
    "factor_support_weight_mean",
    "factor_pairwise_support_overlap_mean",
    "factor_pairwise_jaccard_mean",
)
DEPTH_AWARE_FEATURES = (*ALPHAQ_FEATURES, "tdepth", "qasm_depth_ratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe small learned AlphaQ-only objective selectors with leave-one-target-out validation."
    )
    parser.add_argument(
        "--decomposition-csvs",
        default=",".join(str(path) for path in DEFAULT_DECOMP_CSVS),
    )
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument(
        "--current-beam-csvs",
        default=",".join(str(path) for path in DEFAULT_CURRENT_BEAM_CSVS),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def selector_profiles() -> list[dict[str, Any]]:
    return [
        {
            "selector": "loto_linear_alphaq_nonnegative",
            "features": ALPHAQ_FEATURES,
            "levels": (0, 1, 2, 4),
            "feature_scope": "alphaq-only",
        },
        {
            "selector": "loto_linear_alphaq_signed",
            "features": ALPHAQ_FEATURES,
            "levels": (-2, -1, 0, 1, 2),
            "feature_scope": "alphaq-only",
        },
        {
            "selector": "loto_linear_alphaq_depth_signed",
            "features": DEPTH_AWARE_FEATURES,
            "levels": (-2, -1, 0, 1, 2),
            "feature_scope": "alphaq-plus-qasm-depth",
        },
    ]


def normalize_rows(rows: list[dict[str, str]], features: Iterable[str]) -> list[dict[str, Any]]:
    result = []
    features = tuple(features)
    for target in sorted({row["target"] for row in rows}):
        target_rows = [row for row in rows if row["target"] == target]
        ranges = {feature: feature_range(target_rows, feature) for feature in features}
        for row in target_rows:
            normalized = dict(row)
            for feature in features:
                low, high = ranges[feature]
                value = numeric_feature(row, feature)
                normalized[f"norm_{feature}"] = 0.0 if high == low else (value - low) / (high - low)
            result.append(normalized)
    return result


def feature_range(rows: list[dict[str, str]], feature: str) -> tuple[float, float]:
    values = [numeric_feature(row, feature) for row in rows]
    return min(values), max(values)


def numeric_feature(row: dict[str, str], feature: str) -> float:
    value = coerce_float(row.get(feature))
    if value is None:
        return float("inf")
    return value


def selected_objective(
    rows: list[dict[str, Any]],
    target: str,
    features: tuple[str, ...],
    weights: tuple[int, ...],
) -> str:
    candidates = [row for row in rows if row["target"] == target]
    selected = min(
        candidates,
        key=lambda row: (
            linear_score(row, features, weights),
            row.get("objective_variant", ""),
        ),
    )
    return selected["objective_variant"]


def linear_score(row: dict[str, Any], features: tuple[str, ...], weights: tuple[int, ...]) -> float:
    return sum(
        weight * float(row[f"norm_{feature}"])
        for feature, weight in zip(features, weights)
    )


def weight_candidates(levels: tuple[int, ...], size: int) -> Iterable[tuple[int, ...]]:
    return (
        weights
        for weights in itertools.product(levels, repeat=size)
        if any(weight != 0 for weight in weights)
    )


def train_weights(
    *,
    rows: list[dict[str, Any]],
    targets: list[str],
    features: tuple[str, ...],
    levels: tuple[int, ...],
    oracle: dict[str, dict[str, Any]],
) -> tuple[int, ...]:
    best_weights: tuple[int, ...] | None = None
    best_key: tuple[Any, ...] | None = None
    for weights in weight_candidates(levels, len(features)):
        selected = [
            (
                target,
                selected_objective(rows, target, features, weights),
            )
            for target in targets
        ]
        exact = sum(objective == oracle[target]["oracle_objective"] for target, objective in selected)
        equivalent = sum(objective in oracle[target]["equivalent_objectives"] for target, objective in selected)
        key = (
            equivalent,
            exact,
            -sum(weight != 0 for weight in weights),
            -sum(abs(weight) for weight in weights),
            tuple(-weight for weight in weights),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
    if best_weights is None:
        raise ValueError("No non-zero weight candidate available.")
    return best_weights


def best_grid_row_for_objective(
    grid_rows: list[dict[str, str]],
    target: str,
    objective_variant: str,
) -> dict[str, str] | None:
    return best_policy_beams(grid_rows, objective_variant).get(target)


def detail_rows(
    *,
    decomp_rows: list[dict[str, str]],
    grid_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    current = best_current_beams(current_rows)
    oracle = oracle_info(grid_rows)
    targets = sorted({row["target"] for row in decomp_rows})
    details: list[dict[str, Any]] = []
    all_features = tuple(sorted({feature for profile in selector_profiles() for feature in profile["features"]}))
    normalized = normalize_rows(decomp_rows, all_features)
    for profile in selector_profiles():
        features = tuple(profile["features"])
        for holdout in targets:
            train_targets = [target for target in targets if target != holdout]
            weights = train_weights(
                rows=normalized,
                targets=train_targets,
                features=features,
                levels=tuple(profile["levels"]),
                oracle=oracle,
            )
            selected = selected_objective(normalized, holdout, features, weights)
            beam = best_grid_row_for_objective(grid_rows, holdout, selected)
            baseline = current.get(holdout)
            if beam is None or baseline is None:
                continue
            details.append(
                {
                    "selector": profile["selector"],
                    "feature_scope": profile["feature_scope"],
                    "fold": holdout,
                    "target": holdout,
                    "selected_objective": selected,
                    "oracle_objective": oracle[holdout]["oracle_objective"],
                    "equivalent_objectives": ",".join(oracle[holdout]["equivalent_objectives"]),
                    "exact_oracle_match": selected == oracle[holdout]["oracle_objective"],
                    "oracle_equivalent": selected in oracle[holdout]["equivalent_objectives"],
                    "weights": format_weights(features, weights),
                    "tcount_ratio": safe_ratio(beam.get("tcount"), baseline.get("tcount")),
                    "primary_ratio": safe_ratio(
                        beam.get("primary_nc_depth_ratio"),
                        baseline.get("primary_nc_depth_ratio"),
                    ),
                    "qasm_ratio": safe_ratio(beam.get("qasm_depth"), baseline.get("qasm_depth")),
                    "selected_beam_materializer": beam.get("materializer"),
                    "summary_path": beam.get("summary_path"),
                }
            )
    return details


def summary_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for selector in sorted({row["selector"] for row in details}):
        items = [row for row in details if row["selector"] == selector]
        total = len(items)
        rows.append(
            {
                "selector": selector,
                "feature_scope": items[0]["feature_scope"] if items else "",
                "targets": total,
                "exact_oracle_matches": sum(bool(row["exact_oracle_match"]) for row in items),
                "oracle_equivalent_matches": sum(bool(row["oracle_equivalent"]) for row in items),
                "tcount_nonworse": count(items, "tcount_ratio", strict=False),
                "tcount_wins": count(items, "tcount_ratio", strict=True),
                "primary_wins": count(items, "primary_ratio", strict=True),
                "qasm_wins": count(items, "qasm_ratio", strict=True),
                "joint_nonworse": sum(
                    metric_ok(row.get("tcount_ratio"), strict=False)
                    and metric_ok(row.get("primary_ratio"), strict=False)
                    and metric_ok(row.get("qasm_ratio"), strict=False)
                    for row in items
                ),
                "median_tcount_ratio": median_metric(items, "tcount_ratio"),
                "median_primary_ratio": median_metric(items, "primary_ratio"),
                "median_qasm_ratio": median_metric(items, "qasm_ratio"),
                "target_details": "; ".join(
                    f"{row['target']}->{row['selected_objective']} exact={row['exact_oracle_match']} equiv={row['oracle_equivalent']}"
                    for row in items
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["oracle_equivalent_matches"]),
            -int(row["exact_oracle_matches"]),
            -int(row["joint_nonworse"]),
            row["selector"],
        ),
    )


def count(rows: list[dict[str, Any]], key: str, *, strict: bool) -> int:
    return sum(metric_ok(row.get(key), strict=strict) for row in rows)


def metric_ok(value: Any, *, strict: bool) -> bool:
    numeric = coerce_float(value)
    if numeric is None:
        return False
    return numeric < 1.0 if strict else numeric <= 1.0


def median_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = sorted(value for row in rows if (value := coerce_float(row.get(key))) is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def format_weights(features: tuple[str, ...], weights: tuple[int, ...]) -> str:
    return ",".join(
        f"{feature}:{weight}"
        for feature, weight in zip(features, weights)
        if weight != 0
    )


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "selector",
            "feature_scope",
            "fold",
            "target",
            "selected_objective",
            "oracle_objective",
            "equivalent_objectives",
            "exact_oracle_match",
            "oracle_equivalent",
            "weights",
            "tcount_ratio",
            "primary_ratio",
            "qasm_ratio",
            "selected_beam_materializer",
            "summary_path",
        ],
    )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "selector",
            "feature_scope",
            "targets",
            "exact_oracle_matches",
            "oracle_equivalent_matches",
            "tcount_nonworse",
            "tcount_wins",
            "primary_wins",
            "qasm_wins",
            "joint_nonworse",
            "median_tcount_ratio",
            "median_primary_ratio",
            "median_qasm_ratio",
            "target_details",
        ],
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], output_csv: Path, detail_csv: Path, figure_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = rows[0] if rows else None
    lines = [
        "# AlphaQ learned objective selector probe",
        "",
        f"Summary CSV: `{output_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This probe asks whether a small linear ranker over normalized per-target candidate features can learn the post-beam objective oracle under leave-one-target-out validation. The selector never sees ZX/feynver metrics as input; the labels come from the already materialized beam oracle.",
        "",
        "## Bottom line",
        "",
    ]
    if best is None:
        lines.append("No selector rows were generated.")
    else:
        lines.append(
            f"The best learned profile is `{best['selector']}` with "
            f"{best['exact_oracle_matches']}/{best['targets']} exact matches and "
            f"{best['oracle_equivalent_matches']}/{best['targets']} metric-equivalent matches."
        )
        lines.append(
            "In the current data, these small learned rules do not improve over the simple metric-equivalent `min_factor_count` baseline. Depth-aware inputs can recover some exact labels but do not resolve the robust selection problem."
        )
    lines.extend(
        [
            "",
            "| selector | scope | exact oracle | oracle-equivalent | T <= current | T < current | primary < current | QASM < current | all non-worse | median T | median primary | median QASM |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {selector} | {scope} | {exact}/{n} | {equiv}/{n} | {tn}/{n} | {tw}/{n} | {p}/{n} | {q}/{n} | {j}/{n} | {mt} | {mp} | {mq} |".format(
                selector=row["selector"],
                scope=row["feature_scope"],
                exact=row["exact_oracle_matches"],
                equiv=row["oracle_equivalent_matches"],
                n=row["targets"],
                tn=row["tcount_nonworse"],
                tw=row["tcount_wins"],
                p=row["primary_wins"],
                q=row["qasm_wins"],
                j=row["joint_nonworse"],
                mt=fmt(row.get("median_tcount_ratio")),
                mp=fmt(row.get("median_primary_ratio")),
                mq=fmt(row.get("median_qasm_ratio")),
            )
        )
    lines.extend(["", "## Target Details", ""])
    for row in rows:
        lines.append(f"- `{row['selector']}`: {row['target_details']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["selector"].replace("loto_linear_", "").replace("_", "\n") for row in rows]
    fields = [
        ("exact_oracle_matches", "exact oracle"),
        ("oracle_equivalent_matches", "oracle-equivalent"),
        ("joint_nonworse", "all non-worse"),
    ]
    x = range(len(rows))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    colors = ["#4c78a8", "#1b9e77", "#d95f02"]
    for offset, (field, label) in enumerate(fields):
        positions = [item + (offset - 1) * width for item in x]
        ax.bar(positions, [int(row[field]) for row in rows], width=width, label=label, color=colors[offset])
    ax.set_title("Leave-one-target-out learned objective selector probe")
    ax.set_ylabel("targets")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(int(row["targets"]) for row in rows) + 1 if rows else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    decomp_rows = [
        row
        for path in parse_paths(args.decomposition_csvs)
        for row in read_csv(path)
    ]
    current_rows = [
        row
        for path in parse_paths(args.current_beam_csvs)
        for row in read_csv(path)
    ]
    return decomp_rows, read_csv(args.grid_csv), current_rows


def main() -> int:
    args = parse_args()
    decomp_rows, grid_rows, current_rows = load_rows(args)
    details = detail_rows(decomp_rows=decomp_rows, grid_rows=grid_rows, current_rows=current_rows)
    summaries = summary_rows(details)
    write_detail_csv(args.detail_csv, details)
    write_summary_csv(args.output_csv, summaries)
    write_report(args.report_path, summaries, args.output_csv, args.detail_csv, args.figure_path)
    write_figure(args.figure_path, summaries)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.detail_csv}")
    print(f"Wrote {args.report_path}")
    print(f"Wrote {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
