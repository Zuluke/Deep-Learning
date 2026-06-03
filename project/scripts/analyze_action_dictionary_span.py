from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_ROOT = PROJECT_ROOT / "external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

from alphatensor_quantum.src import tensors

DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "results" / "csv" / "action_dictionary_span.csv"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "results" / "reports" / "action_dictionary_span.md"
)

TARGETS = {
    "mod_5_4": tensors.CircuitType.MOD_5_4,
    "gf_2pow2_mult": tensors.CircuitType.GF_2POW2_MULT,
    "hamming_weight_n4": tensors.CircuitType.HAMMING_WEIGHT_N4,
    "hamming_weight_n5": tensors.CircuitType.HAMMING_WEIGHT_N5,
}


def action_vector(size: int, action: int) -> np.ndarray:
    bits = action + 1
    factor = np.array([(bits >> index) & 1 for index in range(size)], dtype=np.uint8)
    return np.einsum("i,j,k->ijk", factor, factor, factor).reshape(-1)


def low_weight_actions(size: int, max_weight: int) -> list[int]:
    return [
        action
        for action in range((1 << size) - 1)
        if (action + 1).bit_count() <= max_weight
    ]


def tensor_overlap_actions(
    tensor: np.ndarray,
    *,
    base_max_weight: int,
    overlap_max_weight: int,
    per_target_limit: int,
) -> list[int]:
    size = tensor.shape[0]
    selected = set(low_weight_actions(size, base_max_weight))
    candidates = []
    for action in range((1 << size) - 1):
        bits = action + 1
        weight = bits.bit_count()
        if weight > overlap_max_weight:
            continue
        support = [index for index in range(size) if (bits >> index) & 1]
        overlap = int(tensor[np.ix_(support, support, support)].sum())
        if overlap == 0:
            continue
        density = overlap / float(max(len(support) ** 3, 1))
        candidates.append((overlap, density, -weight, -action, action))
    candidates.sort(reverse=True)
    selected.update(action for *_score, action in candidates[:per_target_limit])
    return sorted(selected)


def gf2_rank_and_membership(columns: Iterable[np.ndarray], target: np.ndarray) -> tuple[int, bool]:
    columns = list(columns)
    num_rows = target.size
    rows = [0] * num_rows
    rhs = [int(value) for value in target.reshape(-1)]
    for col_index, column in enumerate(columns):
        bit = 1 << col_index
        for row_index, value in enumerate(column.reshape(-1)):
            if value:
                rows[row_index] ^= bit

    rank = 0
    for col_index in range(len(columns)):
        mask = 1 << col_index
        pivot = None
        for row_index in range(rank, num_rows):
            if rows[row_index] & mask:
                pivot = row_index
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        rhs[rank], rhs[pivot] = rhs[pivot], rhs[rank]
        for row_index in range(num_rows):
            if row_index != rank and (rows[row_index] & mask):
                rows[row_index] ^= rows[rank]
                rhs[row_index] ^= rhs[rank]
        rank += 1

    in_span = not any(rows[row] == 0 and rhs[row] for row in range(rank, num_rows))
    return rank, in_span


def analyze_target(
    target_name: str,
    *,
    low_weight_values: list[int],
    tensor_overlap_limits: list[int],
    tensor_overlap_max_weight: int,
    tensor_overlap_base_weight: int,
) -> list[dict[str, object]]:
    tensor = np.asarray(tensors.get_signature_tensor(TARGETS[target_name]), dtype=np.uint8)
    rows: list[dict[str, object]] = []
    configs: list[tuple[str, list[int]]] = []
    for max_weight in low_weight_values:
        configs.append((f"loww{max_weight}", low_weight_actions(tensor.shape[0], max_weight)))
    for limit in tensor_overlap_limits:
        configs.append(
            (
                f"tensoroverlap_w{tensor_overlap_max_weight}_k{limit}_base{tensor_overlap_base_weight}",
                tensor_overlap_actions(
                    tensor,
                    base_max_weight=tensor_overlap_base_weight,
                    overlap_max_weight=tensor_overlap_max_weight,
                    per_target_limit=limit,
                ),
            )
        )

    for dictionary, actions in configs:
        columns = [action_vector(tensor.shape[0], action) for action in actions]
        rank, in_span = gf2_rank_and_membership(columns, tensor.reshape(-1))
        rows.append(
            {
                "target": target_name,
                "tensor_size": tensor.shape[0],
                "target_weight": int(tensor.sum()),
                "dictionary": dictionary,
                "num_actions": len(actions),
                "span_rank": rank,
                "target_in_span": in_span,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Action dictionary span audit",
        "",
        f"CSV: `{output_csv}`.",
        "",
        "This AlphaQuantum-only audit checks whether each restricted action dictionary spans the target signature tensor over GF(2). If `target_in_span` is false, training cannot solve that target with that dictionary.",
        "",
        "| target | dictionary | actions | rank | target in span |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {dictionary} | {num_actions} | {span_rank} | {target_in_span} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit restricted action dictionary span.")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--low-weight-values", default="1,2,3,4,5")
    parser.add_argument("--tensor-overlap-limits", default="64,128,175,256")
    parser.add_argument("--tensor-overlap-max-weight", type=int, default=5)
    parser.add_argument("--tensor-overlap-base-weight", type=int, default=2)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    rows: list[dict[str, object]] = []
    for target in selected_targets:
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}. Expected one of {sorted(TARGETS)}.")
        rows.extend(
            analyze_target(
                target,
                low_weight_values=parse_int_list(args.low_weight_values),
                tensor_overlap_limits=parse_int_list(args.tensor_overlap_limits),
                tensor_overlap_max_weight=args.tensor_overlap_max_weight,
                tensor_overlap_base_weight=args.tensor_overlap_base_weight,
            )
        )
    write_csv(args.output_csv, rows)
    write_report(args.report_path, rows, args.output_csv)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
