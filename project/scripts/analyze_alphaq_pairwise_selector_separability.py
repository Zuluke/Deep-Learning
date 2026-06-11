from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_alphaq_learned_objective_selector import ALPHAQ_FEATURES
from scripts.analyze_alphaq_learned_objective_selector import DEPTH_AWARE_FEATURES
from scripts.analyze_alphaq_learned_objective_selector import normalize_rows
from scripts.analyze_alphaq_oracle_equivalence import oracle_info
from scripts.run_best_objective_beam_ablation import parse_paths
from scripts.run_best_objective_beam_ablation import read_csv
from scripts.structural_target import coerce_float


DEFAULT_DECOMP_CSVS = (
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_ablation.csv",
    PROJECT_ROOT / "results" / "csv" / "alphaq_decomposition_objective_holdout_ablation.csv",
)
DEFAULT_GRID_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_objective_beam_policy_grid.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_pairwise_selector_separability_summary.csv"
DEFAULT_DETAIL_CSV = PROJECT_ROOT / "results" / "csv" / "alphaq_pairwise_selector_separability_details.csv"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "reports" / "alphaq_pairwise_selector_separability.md"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "alphaq_pairwise_selector_separability.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether current AlphaQ descriptors separate post-beam oracle decisions pairwise."
    )
    parser.add_argument(
        "--decomposition-csvs",
        default=",".join(str(path) for path in DEFAULT_DECOMP_CSVS),
    )
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def selector_profiles() -> list[dict[str, Any]]:
    return [
        {
            "selector": "loto_pairwise_single_alphaq",
            "features": ALPHAQ_FEATURES,
            "kind": "single",
            "levels": (-1, 1),
            "feature_scope": "alphaq-only",
        },
        {
            "selector": "loto_pairwise_linear_alphaq",
            "features": ALPHAQ_FEATURES,
            "kind": "linear",
            "levels": (-2, -1, 0, 1, 2),
            "feature_scope": "alphaq-only",
        },
        {
            "selector": "loto_pairwise_linear_alphaq_depth",
            "features": DEPTH_AWARE_FEATURES,
            "kind": "linear",
            "levels": (-2, -1, 0, 1, 2),
            "feature_scope": "alphaq-plus-qasm-depth",
        },
    ]


def clear_pair_rows(
    *,
    normalized_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, str]],
    features: tuple[str, ...],
) -> list[dict[str, Any]]:
    info = oracle_info(grid_rows)
    rows_by_target_objective = {
        (row["target"], row["objective_variant"]): row
        for row in normalized_rows
    }
    pairs: list[dict[str, Any]] = []
    for target, target_info in sorted(info.items()):
        equivalent = set(target_info["equivalent_objectives"])
        objectives = sorted(
            objective
            for row_target, objective in rows_by_target_objective
            if row_target == target
        )
        for preferred in objectives:
            if preferred not in equivalent:
                continue
            for rejected in objectives:
                if rejected in equivalent or rejected == preferred:
                    continue
                preferred_row = rows_by_target_objective[(target, preferred)]
                rejected_row = rows_by_target_objective[(target, rejected)]
                pair = {
                    "target": target,
                    "preferred_objective": preferred,
                    "rejected_objective": rejected,
                    "oracle_objective": target_info["oracle_objective"],
                    "equivalent_objectives": ",".join(target_info["equivalent_objectives"]),
                }
                for feature in features:
                    pair[f"delta_{feature}"] = (
                        float(preferred_row[f"norm_{feature}"])
                        - float(rejected_row[f"norm_{feature}"])
                    )
                pairs.append(pair)
    return pairs


def candidate_weights(profile: dict[str, Any]) -> list[tuple[int, ...]]:
    features = tuple(profile["features"])
    levels = tuple(profile["levels"])
    if profile["kind"] == "single":
        result = []
        for index in range(len(features)):
            for direction in levels:
                weights = [0] * len(features)
                weights[index] = direction
                result.append(tuple(weights))
        return result
    return [
        weights
        for weights in itertools.product(levels, repeat=len(features))
        if any(weight != 0 for weight in weights)
    ]


def pairwise_margin(pair: dict[str, Any], features: tuple[str, ...], weights: tuple[int, ...]) -> float:
    return sum(weight * float(pair[f"delta_{feature}"]) for feature, weight in zip(features, weights))


def pair_correct(pair: dict[str, Any], features: tuple[str, ...], weights: tuple[int, ...]) -> bool:
    return pairwise_margin(pair, features, weights) < 0


def train_weights(
    *,
    pairs: list[dict[str, Any]],
    train_targets: list[str],
    profile: dict[str, Any],
) -> tuple[int, ...]:
    features = tuple(profile["features"])
    train_pairs = [pair for pair in pairs if pair["target"] in train_targets]
    best_weights: tuple[int, ...] | None = None
    best_key: tuple[Any, ...] | None = None
    for weights in candidate_weights(profile):
        correct = sum(pair_correct(pair, features, weights) for pair in train_pairs)
        margins = [
            -pairwise_margin(pair, features, weights)
            for pair in train_pairs
            if pair_correct(pair, features, weights)
        ]
        mean_margin = sum(margins) / len(margins) if margins else 0.0
        key = (
            correct,
            mean_margin,
            -sum(weight != 0 for weight in weights),
            -sum(abs(weight) for weight in weights),
            tuple(-weight for weight in weights),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
    if best_weights is None:
        raise ValueError("No pairwise selector candidate available.")
    return best_weights


def detail_rows(decomp_rows: list[dict[str, str]], grid_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    all_features = tuple(sorted({feature for profile in selector_profiles() for feature in profile["features"]}))
    normalized = normalize_rows(decomp_rows, all_features)
    pairs = clear_pair_rows(normalized_rows=normalized, grid_rows=grid_rows, features=all_features)
    targets = sorted({pair["target"] for pair in pairs})
    details: list[dict[str, Any]] = []
    for profile in selector_profiles():
        features = tuple(profile["features"])
        for holdout in targets:
            weights = train_weights(
                pairs=pairs,
                train_targets=[target for target in targets if target != holdout],
                profile=profile,
            )
            for pair in [pair for pair in pairs if pair["target"] == holdout]:
                margin = pairwise_margin(pair, features, weights)
                details.append(
                    {
                        "selector": profile["selector"],
                        "feature_scope": profile["feature_scope"],
                        "fold": holdout,
                        "target": pair["target"],
                        "preferred_objective": pair["preferred_objective"],
                        "rejected_objective": pair["rejected_objective"],
                        "oracle_objective": pair["oracle_objective"],
                        "equivalent_objectives": pair["equivalent_objectives"],
                        "pair_correct": margin < 0,
                        "margin": margin,
                        "weights": format_weights(features, weights),
                    }
                )
    return details


def summary_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for selector in sorted({row["selector"] for row in details}):
        items = [row for row in details if row["selector"] == selector]
        targets = sorted({row["target"] for row in items})
        correct_pairs = sum(bool(row["pair_correct"]) for row in items)
        all_correct_targets = sum(
            all(bool(row["pair_correct"]) for row in items if row["target"] == target)
            for target in targets
        )
        rows.append(
            {
                "selector": selector,
                "feature_scope": items[0]["feature_scope"] if items else "",
                "targets": len(targets),
                "pairs": len(items),
                "pairwise_correct": correct_pairs,
                "target_all_pairs_correct": all_correct_targets,
                "mean_margin": mean([coerce_float(row.get("margin")) for row in items]),
                "target_details": "; ".join(
                    f"{target}:{sum(bool(row['pair_correct']) for row in items if row['target'] == target)}/{sum(1 for row in items if row['target'] == target)}"
                    for target in targets
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["target_all_pairs_correct"]),
            -int(row["pairwise_correct"]),
            row["selector"],
        ),
    )


def mean(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


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
            "preferred_objective",
            "rejected_objective",
            "oracle_objective",
            "equivalent_objectives",
            "pair_correct",
            "margin",
            "weights",
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
            "pairs",
            "pairwise_correct",
            "target_all_pairs_correct",
            "mean_margin",
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
        "# AlphaQ pairwise selector separability",
        "",
        f"Summary CSV: `{output_csv}`.",
        f"Detail CSV: `{detail_csv}`.",
        f"Figure: `{figure_path}`.",
        "",
        "This audit converts objective selection into clear pairwise comparisons: an oracle-equivalent objective must score better than a non-equivalent objective for the same target. Metric ties are excluded, so the audit tests separability of real post-beam decisions.",
        "",
        "## Bottom line",
        "",
    ]
    if best is None:
        lines.append("No clear pairwise decisions were generated.")
    else:
        lines.append(
            f"The best profile `{best['selector']}` solves all clear pairs for "
            f"{best['target_all_pairs_correct']}/{best['targets']} targets and "
            f"{best['pairwise_correct']}/{best['pairs']} individual pairs."
        )
        lines.append(
            "This shows whether the current descriptors separate oracle decisions locally. If target-level pair accuracy remains below the min-factor-count equivalence rate, the selector issue is feature/class insufficiency rather than a single bad tie-break."
        )
    lines.extend(
        [
            "",
            "| selector | scope | target all-pair correct | pairwise correct | mean margin | target details |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {selector} | {scope} | {tc}/{targets} | {pc}/{pairs} | {margin} | {details} |".format(
                selector=row["selector"],
                scope=row["feature_scope"],
                tc=row["target_all_pairs_correct"],
                targets=row["targets"],
                pc=row["pairwise_correct"],
                pairs=row["pairs"],
                margin=fmt(row.get("mean_margin")),
                details=row["target_details"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["selector"].replace("loto_pairwise_", "").replace("_", "\n") for row in rows]
    fields = [
        ("target_all_pairs_correct", "targets"),
        ("pairwise_correct", "pairs"),
    ]
    denominators = [
        ("targets", "targets"),
        ("pairs", "pairs"),
    ]
    x = range(len(rows))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    colors = ["#4c78a8", "#1b9e77"]
    for offset, ((field, label), (denominator, _)) in enumerate(zip(fields, denominators)):
        positions = [item + (offset - 0.5) * width for item in x]
        values = [
            0.0 if int(row[denominator]) == 0 else int(row[field]) / int(row[denominator])
            for row in rows
        ]
        ax.bar(positions, values, width=width, label=label, color=colors[offset])
    ax.set_title("Pairwise separability of post-beam objective decisions")
    ax.set_ylabel("accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fmt(value: Any) -> str:
    numeric = coerce_float(value)
    return "" if numeric is None else f"{numeric:.3g}"


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    decomp_rows = [
        row
        for path in parse_paths(args.decomposition_csvs)
        for row in read_csv(path)
    ]
    return decomp_rows, read_csv(args.grid_csv)


def main() -> int:
    args = parse_args()
    decomp_rows, grid_rows = load_rows(args)
    details = detail_rows(decomp_rows, grid_rows)
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
