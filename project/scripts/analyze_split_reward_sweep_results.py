from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "results" / "csv" / "split_reward_sweep_analysis.csv"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "results" / "reports" / "split_reward_sweep_analysis.md"
)
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "split_reward_sweep_analysis"
KNOWN_TARGET_WEIGHTS = {
    "mod_5_4": 34.0,
    "gf_2pow2_mult": 66.0,
    "hamming_weight_n4": 120.0,
    "hamming_weight_n5": 144.0,
}


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        target_weight = parse_float(row.get("target_tensor_weight"))
        if target_weight is None:
            target_weight = KNOWN_TARGET_WEIGHTS.get(str(row.get("target", "")))
        terminal = parse_float(row.get("best_return_residual_weight"))
        frontier = parse_float(row.get("best_frontier_residual_weight"))
        frontier_drop = parse_float(row.get("best_frontier_residual_drop"))
        if frontier_drop is None and target_weight is not None and frontier is not None:
            frontier_drop = target_weight - frontier
        terminal_drop = None
        if target_weight is not None and terminal is not None:
            terminal_drop = target_weight - terminal
        normalized_frontier_drop = None
        if target_weight and frontier_drop is not None:
            normalized_frontier_drop = frontier_drop / target_weight
        normalized_terminal_drop = None
        if target_weight and terminal_drop is not None:
            normalized_terminal_drop = terminal_drop / target_weight
        solved_cost = parse_float(row.get("best_effective_t_cost"))
        enriched_row: dict[str, Any] = {
            **row,
            "target_tensor_weight": target_weight,
            "terminal_residual": terminal,
            "terminal_residual_drop": terminal_drop,
            "normalized_terminal_residual_drop": normalized_terminal_drop,
            "frontier_residual": frontier,
            "frontier_residual_drop": frontier_drop,
            "normalized_frontier_residual_drop": normalized_frontier_drop,
            "has_positive_frontier": (
                frontier_drop is not None and frontier_drop > 0.0
            ),
            "is_solved": solved_cost is not None,
            "solved_effective_t_cost": solved_cost,
            "max_num_moves_numeric": parse_float(row.get("max_num_moves")),
        }
        enriched.append(enriched_row)
    return enriched


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_csv",
        "target",
        "mode",
        "status",
        "is_solved",
        "solved_effective_t_cost",
        "target_tensor_weight",
        "terminal_residual",
        "terminal_residual_drop",
        "normalized_terminal_residual_drop",
        "frontier_residual",
        "frontier_residual_drop",
        "normalized_frontier_residual_drop",
        "has_positive_frontier",
        "best_frontier_effective_t_cost",
        "best_frontier_num_moves",
        "max_num_moves",
        "action_dictionary",
        "max_action_weight",
        "tensor_overlap_max_weight",
        "tensor_overlap_max_actions_per_target",
        "gadget_closure_max_weight",
        "action_prior",
        "action_prior_beta",
        "action_prior_top_k",
        "frontier_replay_fraction",
        "frontier_replay_min_moves",
        "frontier_replay_min_residual_drop",
        "mask_repeated_actions",
        "candidate_output_dir",
        "candidate_manifest_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collision_groups(rows: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    by_dir: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        candidate_dir = str(row.get("candidate_output_dir") or "").strip()
        source = str(row.get("source_csv") or "").strip()
        max_moves = str(row.get("max_num_moves") or "").strip()
        if candidate_dir:
            by_dir[candidate_dir].add(f"{Path(source).name}:m{max_moves}")
    return [
        (candidate_dir, sorted(sources))
        for candidate_dir, sources in sorted(by_dir.items())
        if len(sources) > 1
    ]


def plot_results(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    positive_rows = [
        row for row in rows
        if row["has_positive_frontier"]
    ]
    if not positive_rows:
        return []

    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    labels = [
        f"{row.get('target')}\n{row.get('mode')}\nH={row.get('max_num_moves')}"
        for row in positive_rows
    ]
    values = [
        float(row["normalized_frontier_residual_drop"])
        for row in positive_rows
    ]
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.65 * len(labels))))
    ax.barh(range(len(labels)), values, color="#3b7c9f")
    ax.set_xlabel("normalized frontier residual drop")
    ax.set_title("Positive partial progress found by split-reward sweeps")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    path = figure_dir / "positive_frontier_drop.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("max_num_moves_numeric") is not None:
            by_target[str(row.get("target"))].append(row)
    plotted = False
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for target, target_rows in sorted(by_target.items()):
        points = [
            (
                float(row["max_num_moves_numeric"]),
                float(row["terminal_residual"]),
                str(row.get("mode")),
            )
            for row in target_rows
            if row.get("terminal_residual") is not None
        ]
        if len({(h, mode) for h, _value, mode in points}) < 2:
            continue
        for mode in sorted({mode for _h, _value, mode in points}):
            mode_points = sorted(
                (h, value) for h, value, point_mode in points if point_mode == mode
            )
            if len(mode_points) < 2:
                continue
            xs, ys = zip(*mode_points)
            ax.plot(xs, ys, marker="o", linewidth=1.6, label=f"{target} {mode}")
            plotted = True
    if plotted:
        ax.set_xlabel("max moves")
        ax.set_ylabel("terminal residual weight")
        ax.set_title("Terminal failures versus horizon")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        path = figure_dir / "terminal_residual_by_horizon.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)

    return paths


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    output_csv: Path,
    figure_paths: list[Path],
    collisions: list[tuple[str, list[str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    solved = [row for row in rows if row["is_solved"]]
    positive = [row for row in rows if row["has_positive_frontier"]]
    lines = [
        "# Split reward sweep analysis",
        "",
        f"Rows analyzed: {len(rows)}.",
        f"Analysis CSV: `{output_csv}`.",
        "",
        "## Summary",
        "",
        f"- Solved rows: {len(solved)}.",
        f"- Rows with positive frontier progress: {len(positive)}.",
        f"- Candidate-directory collision groups: {len(collisions)}.",
        "",
    ]
    if positive:
        lines.extend([
            "## Positive Frontiers",
            "",
            "| target | mode | horizon | terminal residual | frontier residual | frontier drop | normalized drop |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(
            positive,
            key=lambda item: (
                str(item.get("target")),
                -float(item.get("normalized_frontier_residual_drop") or 0.0),
            ),
        ):
            lines.append(
                "| {target} | {mode} | {horizon} | {terminal} | {frontier} | {drop} | {ndrop:.4f} |".format(
                    target=row.get("target", ""),
                    mode=row.get("mode", ""),
                    horizon=row.get("max_num_moves", ""),
                    terminal=row.get("terminal_residual", ""),
                    frontier=row.get("frontier_residual", ""),
                    drop=row.get("frontier_residual_drop", ""),
                    ndrop=float(row.get("normalized_frontier_residual_drop") or 0.0),
                )
            )
        lines.append("")
    if collisions:
        lines.extend([
            "## Artifact Warnings",
            "",
            "Some rows point to the same candidate directory across distinct sources or horizons. Their scalar CSV metrics can still be read, but factor manifests from those paths are not safe for horizon-specific replay.",
            "",
        ])
        for candidate_dir, sources in collisions[:20]:
            lines.append(f"- `{candidate_dir}`: {', '.join(sources)}")
        if len(collisions) > 20:
            lines.append(f"- ... {len(collisions) - 20} additional groups omitted.")
        lines.append("")
    if figure_paths:
        lines.extend(["## Figures", ""])
        for figure in figure_paths:
            lines.append(f"- `{figure}`")
        lines.append("")
    if not positive:
        lines.append("No figure was generated because no positive frontier progress was present.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze split-reward sweep CSVs and generate informative figures."
    )
    parser.add_argument("input_csvs", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = enrich_rows(read_rows(args.input_csvs))
    write_csv(args.output_csv, rows)
    figures = plot_results(rows, args.figure_dir)
    collisions = collision_groups(rows)
    write_report(args.report_path, rows, args.output_csv, figures, collisions)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    for figure in figures:
        print(f"Wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
