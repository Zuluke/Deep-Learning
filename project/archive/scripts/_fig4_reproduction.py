from __future__ import annotations

import csv
import dataclasses
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BINARY_ADDITION_DECOMPOSITIONS_PATH = (
    PROJECT_ROOT
    / "external"
    / "alphatensor_quantum"
    / "decompositions"
    / "binary_addition.npz"
)
MULTIPLICATION_DECOMPOSITIONS_PATH = (
    PROJECT_ROOT
    / "external"
    / "alphatensor_quantum"
    / "decompositions"
    / "multiplication_finite_fields_gadgets.npz"
)
APPLICATION_BENCHMARKS_ROOT = (
    PROJECT_ROOT
    / "external"
    / "circuit-to-tensor"
    / "benchmarks"
    / "applications"
)
ARITHMETIC_BENCHMARKS_ROOT = (
    PROJECT_ROOT
    / "external"
    / "circuit-to-tensor"
    / "benchmarks"
    / "arithmetic"
)

DEFAULT_FIG4B_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "reproducibility" / "fig4b_binary_addition"
)
DEFAULT_FIG4_OUTPUT_DIR = PROJECT_ROOT / "results" / "reproducibility" / "fig4"

FIG4A_BEST_CLASSICAL_UPPER_BOUNDS = {
    2: 3,
    3: 6,
    4: 9,
    5: 13,
    6: 17,
    7: 22,
    8: 26,
    9: 31,
    10: 35,
}
FIG4A_UPPER_BOUND_MATCH_EXPONENTS = {2, 3, 4, 5, 7}
FIG4A_LOWER_BOUND_MATCH_EXPONENTS = {2, 3, 4, 5}


@dataclasses.dataclass(frozen=True)
class BinaryAdditionPoint:
    bits: int
    target: str
    tensor_size: int
    num_decompositions: int
    num_factors: int
    num_toffoli_gadgets: int
    num_cs_gadgets: int
    effective_tcount: int
    alphatensor_toffoli_count: int
    gidney_toffoli_count: int
    cuccaro_toffoli_count: int
    vendored_qasm_ccx_count: int
    decomposition_key: str
    qasm_path: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FiniteFieldPoint:
    exponent_m: int
    target: str
    tensor_size: int
    num_decompositions: int
    num_factors: int
    num_toffoli_gadgets: int
    num_cs_gadgets: int
    effective_tcount: int
    alphatensor_toffoli_count: int
    circuit_before_optimization_toffoli_count: int
    karatsuba_toffoli_estimate: float
    best_classical_toffoli_upper_bound: int
    matches_best_known_upper_bound: bool
    matches_best_known_lower_bound: bool
    decomposition_key: str
    qasm_path: str

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["karatsuba_toffoli_estimate"] = round(self.karatsuba_toffoli_estimate, 6)
        return payload


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _bits_from_binary_addition_target(target: str) -> int:
    match = re.fullmatch(r"cuccaro_adder_n(\d+)", target)
    if match is None:
        raise ValueError(f"Unexpected binary addition target: {target}")
    return int(match.group(1))


def _exponent_from_finite_field_target(target: str) -> int:
    match = re.fullmatch(r"gf_2pow(\d+)_mult_comp2", target)
    if match is None:
        raise ValueError(f"Unexpected finite-field target: {target}")
    return int(match.group(1))


def _count_ccx_in_qasm(qasm_path: Path) -> int:
    return len(re.findall(r"^\s*ccx\s", qasm_path.read_text(), flags=re.MULTILINE))


def _factors_are_linearly_independent(
    factor1: np.ndarray,
    factor2: np.ndarray,
    factor3: np.ndarray,
) -> bool:
    distinct = (
        np.any(factor1 != factor2)
        and np.any(factor1 != factor3)
        and np.any(factor2 != factor3)
    )
    return bool(distinct and np.any(factor3 != np.mod(factor1 + factor2, 2)))


def _factors_form_toffoli_gadget(factors: np.ndarray) -> bool:
    if factors.shape[0] != 7:
        raise ValueError(f"Expected 7 factors for a Toffoli gadget, got {factors.shape}")
    a, b, c, ab, ac, abc, bc = factors
    return bool(
        _factors_are_linearly_independent(a, b, c)
        and np.array_equal(ab, np.mod(a + b, 2))
        and np.array_equal(ac, np.mod(a + c, 2))
        and np.array_equal(abc, np.mod(a + b + c, 2))
        and np.array_equal(bc, np.mod(b + c, 2))
    )


def _factors_form_cs_gadget(factors: np.ndarray) -> bool:
    if factors.shape[0] != 3:
        raise ValueError(f"Expected 3 factors for a CS gadget, got {factors.shape}")
    a, b, ab = factors
    return bool(np.any(a != b) and np.array_equal(ab, np.mod(a + b, 2)))


def analyze_factorization(factors: np.ndarray) -> tuple[int, int, int]:
    num_factors = factors.shape[0]
    factors_in_toffoli = np.zeros(num_factors, dtype=bool)
    factors_in_cs = np.zeros(num_factors, dtype=bool)
    num_toffoli_gadgets = 0
    num_cs_gadgets = 0

    for index in range(num_factors):
        factors_not_in_gadgets = np.logical_not(factors_in_toffoli | factors_in_cs)
        if (
            index >= 6
            and np.all(factors_not_in_gadgets[(index - 6) : (index + 1)])
            and _factors_form_toffoli_gadget(factors[(index - 6) : (index + 1)])
        ):
            factors_in_toffoli[(index - 6) : (index + 1)] = True
            num_toffoli_gadgets += 1
        if (
            index >= 2
            and np.all(factors_not_in_gadgets[(index - 2) : (index + 1)])
            and _factors_form_cs_gadget(factors[(index - 2) : (index + 1)])
        ):
            factors_in_cs[(index - 2) : (index + 1)] = True
            num_cs_gadgets += 1

    effective_tcount = num_factors - 5 * num_toffoli_gadgets - num_cs_gadgets
    return effective_tcount, num_toffoli_gadgets, num_cs_gadgets


def _paper_cuccaro_toffoli_count(bits: int) -> int:
    return 2 * bits - 2


def build_binary_addition_points() -> list[BinaryAdditionPoint]:
    with np.load(BINARY_ADDITION_DECOMPOSITIONS_PATH, allow_pickle=True) as decompositions:
        expected_targets = [f"cuccaro_adder_n{bits}" for bits in range(3, 11)]
        missing_targets = sorted(set(expected_targets) - set(decompositions.files))
        if missing_targets:
            raise ValueError(f"Missing binary addition targets: {missing_targets}")

        points: list[BinaryAdditionPoint] = []
        for target in expected_targets:
            bits = _bits_from_binary_addition_target(target)
            target_decompositions = np.asarray(decompositions[target], dtype=np.int32)
            if target_decompositions.ndim != 3:
                raise ValueError(
                    f"Unexpected decomposition shape for {target}: {target_decompositions.shape}"
                )
            first_decomposition = target_decompositions[0]
            effective_tcount, num_toffoli_gadgets, num_cs_gadgets = analyze_factorization(
                first_decomposition
            )
            qasm_path = APPLICATION_BENCHMARKS_ROOT / target / f"{target}.qasm"
            if not qasm_path.exists():
                raise FileNotFoundError(f"Missing baseline QASM for {target}: {qasm_path}")

            vendored_qasm_ccx_count = _count_ccx_in_qasm(qasm_path)
            paper_cuccaro_toffoli_count = _paper_cuccaro_toffoli_count(bits)

            points.append(
                BinaryAdditionPoint(
                    bits=bits,
                    target=target,
                    tensor_size=int(first_decomposition.shape[1]),
                    num_decompositions=int(target_decompositions.shape[0]),
                    num_factors=int(first_decomposition.shape[0]),
                    num_toffoli_gadgets=num_toffoli_gadgets,
                    num_cs_gadgets=num_cs_gadgets,
                    effective_tcount=effective_tcount,
                    alphatensor_toffoli_count=num_toffoli_gadgets,
                    gidney_toffoli_count=bits - 1,
                    cuccaro_toffoli_count=paper_cuccaro_toffoli_count,
                    vendored_qasm_ccx_count=vendored_qasm_ccx_count,
                    decomposition_key=target,
                    qasm_path=str(qasm_path.relative_to(PROJECT_ROOT)),
                )
            )
    return points


def build_finite_field_points() -> list[FiniteFieldPoint]:
    with np.load(MULTIPLICATION_DECOMPOSITIONS_PATH, allow_pickle=True) as decompositions:
        expected_targets = [f"gf_2pow{exponent}_mult_comp2" for exponent in range(2, 11)]
        missing_targets = sorted(set(expected_targets) - set(decompositions.files))
        if missing_targets:
            raise ValueError(f"Missing finite-field targets: {missing_targets}")

        points: list[FiniteFieldPoint] = []
        for target in expected_targets:
            exponent_m = _exponent_from_finite_field_target(target)
            target_decompositions = np.asarray(decompositions[target], dtype=np.int32)
            if target_decompositions.ndim != 3:
                raise ValueError(
                    f"Unexpected decomposition shape for {target}: {target_decompositions.shape}"
                )
            first_decomposition = target_decompositions[0]
            effective_tcount, num_toffoli_gadgets, num_cs_gadgets = analyze_factorization(
                first_decomposition
            )
            qasm_path = (
                ARITHMETIC_BENCHMARKS_ROOT
                / f"gf_2pow{exponent_m}_mult"
                / f"gf_2pow{exponent_m}_mult.qasm"
            )
            if not qasm_path.exists():
                raise FileNotFoundError(f"Missing arithmetic QASM for {target}: {qasm_path}")

            best_classical = FIG4A_BEST_CLASSICAL_UPPER_BOUNDS[exponent_m]
            points.append(
                FiniteFieldPoint(
                    exponent_m=exponent_m,
                    target=target,
                    tensor_size=int(first_decomposition.shape[1]),
                    num_decompositions=int(target_decompositions.shape[0]),
                    num_factors=int(first_decomposition.shape[0]),
                    num_toffoli_gadgets=num_toffoli_gadgets,
                    num_cs_gadgets=num_cs_gadgets,
                    effective_tcount=effective_tcount,
                    alphatensor_toffoli_count=num_toffoli_gadgets,
                    circuit_before_optimization_toffoli_count=_count_ccx_in_qasm(qasm_path),
                    karatsuba_toffoli_estimate=exponent_m ** math.log2(3),
                    best_classical_toffoli_upper_bound=best_classical,
                    matches_best_known_upper_bound=exponent_m in FIG4A_UPPER_BOUND_MATCH_EXPONENTS,
                    matches_best_known_lower_bound=exponent_m in FIG4A_LOWER_BOUND_MATCH_EXPONENTS,
                    decomposition_key=target,
                    qasm_path=str(qasm_path.relative_to(PROJECT_ROOT)),
                )
            )
    return points


def validate_binary_addition_points(points: list[BinaryAdditionPoint]) -> None:
    if [point.bits for point in points] != list(range(3, 11)):
        raise ValueError(f"Unexpected bit counts: {[point.bits for point in points]}")

    expected_cuccaro = [4, 6, 8, 10, 12, 14, 16, 18]
    expected_alphatensor = [2, 3, 4, 5, 6, 7, 8, 9]
    expected_effective_tcount = [4, 6, 8, 10, 12, 14, 16, 18]
    expected_vendored_qasm = [6, 8, 10, 12, 14, 16, 18, 20]

    if [point.cuccaro_toffoli_count for point in points] != expected_cuccaro:
        raise ValueError("Unexpected Cuccaro paper baseline series.")
    if [point.vendored_qasm_ccx_count for point in points] != expected_vendored_qasm:
        raise ValueError("Unexpected vendored Cuccaro QASM series.")
    if [point.alphatensor_toffoli_count for point in points] != expected_alphatensor:
        raise ValueError("Unexpected AlphaTensor-Quantum series.")
    if [point.gidney_toffoli_count for point in points] != expected_alphatensor:
        raise ValueError("Unexpected Gidney series.")
    if [point.effective_tcount for point in points] != expected_effective_tcount:
        raise ValueError("Unexpected effective T-count series.")

    for point in points:
        if point.num_cs_gadgets != 0:
            raise ValueError(f"{point.target} unexpectedly contains CS gadgets.")
        if point.num_toffoli_gadgets != point.bits - 1:
            raise ValueError(f"{point.target} has an unexpected Toffoli gadget count.")
        if point.vendored_qasm_ccx_count != point.cuccaro_toffoli_count + 2:
            raise ValueError(
                f"{point.target} does not match the expected opt=False to opt=True offset."
            )


def validate_finite_field_points(points: list[FiniteFieldPoint]) -> None:
    if [point.exponent_m for point in points] != list(range(2, 11)):
        raise ValueError(f"Unexpected exponents: {[point.exponent_m for point in points]}")

    expected_original = [4, 9, 16, 25, 36, 49, 64, 81, 100]
    expected_alphatensor = [3, 6, 9, 13, 18, 22, 29, 35, 46]
    expected_upper_bounds = [3, 6, 9, 13, 17, 22, 26, 31, 35]
    expected_effective_tcount = [6, 12, 18, 26, 36, 44, 58, 70, 92]

    if [point.circuit_before_optimization_toffoli_count for point in points] != expected_original:
        raise ValueError("Unexpected pre-optimization finite-field series.")
    if [point.alphatensor_toffoli_count for point in points] != expected_alphatensor:
        raise ValueError("Unexpected AlphaTensor-Quantum finite-field series.")
    if [point.best_classical_toffoli_upper_bound for point in points] != expected_upper_bounds:
        raise ValueError("Unexpected best classical finite-field series.")
    if [point.effective_tcount for point in points] != expected_effective_tcount:
        raise ValueError("Unexpected effective T-count finite-field series.")

    for point in points:
        if point.num_cs_gadgets != 0:
            raise ValueError(f"{point.target} unexpectedly contains CS gadgets.")


def _write_csv_rows(rows: list[dict[str, Any]], output_path: Path) -> Path:
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_binary_addition_csv(points: list[BinaryAdditionPoint], output_dir: Path) -> Path:
    return _write_csv_rows(
        [point.as_dict() for point in points],
        output_dir / "fig4b_data.csv",
    )


def write_finite_field_csv(points: list[FiniteFieldPoint], output_dir: Path) -> Path:
    return _write_csv_rows(
        [point.as_dict() for point in points],
        output_dir / "fig4a_data.csv",
    )


def write_fig4_json(
    finite_field_points: list[FiniteFieldPoint],
    binary_addition_points: list[BinaryAdditionPoint],
    output_dir: Path,
) -> Path:
    payload = {
        "figure": "Fig. 4",
        "title": "Reproduction of Fig. 4 from published AlphaTensor-Quantum artifacts",
        "article_url": "https://doi.org/10.1038/s42256-025-01001-1",
        "panel_a": {
            "title": "Multiplication in finite fields",
            "x_axis": "Exponent m",
            "y_axis": "Number of Toffoli gates",
            "series": {
                "circuit_before_optimization": [
                    point.circuit_before_optimization_toffoli_count
                    for point in finite_field_points
                ],
                "alphatensor_quantum": [
                    point.alphatensor_toffoli_count for point in finite_field_points
                ],
                "karatsuba_algorithm": [
                    round(point.karatsuba_toffoli_estimate, 6) for point in finite_field_points
                ],
                "best_classical_algorithm": [
                    point.best_classical_toffoli_upper_bound for point in finite_field_points
                ],
            },
            "points": [point.as_dict() for point in finite_field_points],
        },
        "panel_b": {
            "title": "Binary addition",
            "x_axis": "Number of bits",
            "y_axis": "Number of Toffoli gates",
            "series": {
                "cuccaro_2004": [point.cuccaro_toffoli_count for point in binary_addition_points],
                "gidney_2018": [point.gidney_toffoli_count for point in binary_addition_points],
                "alphatensor_quantum": [
                    point.alphatensor_toffoli_count for point in binary_addition_points
                ],
                "vendored_qasm_audit": [
                    point.vendored_qasm_ccx_count for point in binary_addition_points
                ],
                "effective_tcount_audit": [
                    point.effective_tcount for point in binary_addition_points
                ],
            },
            "points": [point.as_dict() for point in binary_addition_points],
        },
        "classical_reference_notes": {
            "panel_a_best_classical_algorithm": (
                "Best-known upper bounds reconstructed from refs. 48-49 in the article: "
                "Montgomery (2005) for m=2..7 and Fan & Hasan (2007) improvements for m=8..10."
            ),
            "panel_b_cuccaro_baseline": (
                "The vendored cuccaro_adder_n*.qasm files correspond to cuccaro_ripple_adder(opt=False) "
                "with 2n Toffolis. The article panel uses the optimized opt=True variant with 2n-2 Toffolis."
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = output_dir / "fig4_data.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_fig4b_json(points: list[BinaryAdditionPoint], output_dir: Path) -> Path:
    payload = {
        "figure": "Fig. 4b",
        "title": "Corrected binary addition reproduction from published AlphaTensor-Quantum artifacts",
        "article_url": "https://doi.org/10.1038/s42256-025-01001-1",
        "x_axis": "Number of bits",
        "y_axis": "Number of Toffoli gates",
        "series": {
            "cuccaro_2004": [point.cuccaro_toffoli_count for point in points],
            "gidney_2018": [point.gidney_toffoli_count for point in points],
            "alphatensor_quantum": [point.alphatensor_toffoli_count for point in points],
            "vendored_qasm_audit": [point.vendored_qasm_ccx_count for point in points],
            "effective_tcount_audit": [point.effective_tcount for point in points],
        },
        "points": [point.as_dict() for point in points],
        "note": (
            "The article panel uses the optimized cuccaro_ripple_adder(opt=True) baseline with 2n-2 "
            "Toffolis, whereas the vendored cuccaro_adder_n*.qasm files correspond to opt=False and "
            "contain 2n Toffolis."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = output_dir / "fig4b_data.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_fig4b_readme(points: list[BinaryAdditionPoint], output_dir: Path) -> Path:
    readme_path = output_dir / "README.md"
    readme = f"""# Fig. 4b Reproduction

This directory contains a corrected reproduction of the binary-addition panel
(`Fig. 4b`) from *Quantum circuit optimization with AlphaTensor*.

## Data sources

- Decompositions: `/{BINARY_ADDITION_DECOMPOSITIONS_PATH.relative_to(PROJECT_ROOT)}`
- Benchmark notebook/QASM: `/{APPLICATION_BENCHMARKS_ROOT.relative_to(PROJECT_ROOT)}`

## Important audit note

The vendored `cuccaro_adder_n{{3..10}}.qasm` circuits were generated from
`cuccaro_ripple_adder(..., opt=False)` and therefore contain `2n` Toffoli gates.
The paper panel uses the optimized `opt=True` variant of the same Cuccaro family,
which contains `2n - 2` Toffoli gates. This is why the plotted `Cuccaro et al. (2004)`
series is `[4, 6, 8, 10, 12, 14, 16, 18]`, while the vendored QASM audit series is
`[6, 8, 10, 12, 14, 16, 18, 20]`.

## Reproduced series

- `Cuccaro et al. (2004)`: {[point.cuccaro_toffoli_count for point in points]}
- `Gidney (2018)`: {[point.gidney_toffoli_count for point in points]}
- `AlphaTensor-Quantum`: {[point.alphatensor_toffoli_count for point in points]}
- `Vendored QASM audit`: {[point.vendored_qasm_ccx_count for point in points]}
"""
    readme_path.write_text(readme, encoding="utf-8")
    return readme_path


def write_fig4_readme(
    finite_field_points: list[FiniteFieldPoint],
    binary_addition_points: list[BinaryAdditionPoint],
    output_dir: Path,
) -> Path:
    readme_path = output_dir / "README.md"
    readme = f"""# Fig. 4 Reproduction

This directory contains a two-panel reproduction of `Fig. 4` from
*Quantum circuit optimization with AlphaTensor* using the local vendored
artifacts already present in this repository.

## Panel a: multiplication in finite fields

- Original circuits (`Circuit before optimization`) are reconstructed by counting
  `ccx` gates in `gf_2pow{{2..10}}_mult.qasm`.
- `AlphaTensor-Quantum` uses the first published decomposition for each target in
  `multiplication_finite_fields_gadgets.npz`, with the plotted value given by the
  number of detected Toffoli gadgets.
- `Karatsuba's algorithm` is plotted as `m^log2(3)`.
- `Best classical algorithm` uses the best-known classical upper bounds from refs.
  48-49 in the paper, yielding:
  `{[point.best_classical_toffoli_upper_bound for point in finite_field_points]}`.

## Panel b: binary addition

- `AlphaTensor-Quantum` and `Gidney (2018)` coincide at
  `{[point.alphatensor_toffoli_count for point in binary_addition_points]}`.
- The paper's `Cuccaro et al. (2004)` series is
  `{[point.cuccaro_toffoli_count for point in binary_addition_points]}`.
- The vendored QASM audit series is
  `{[point.vendored_qasm_ccx_count for point in binary_addition_points]}` because
  those files correspond to `opt=False`, whereas the article panel uses `opt=True`.
"""
    readme_path.write_text(readme, encoding="utf-8")
    return readme_path


def _apply_article_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def plot_binary_addition_panel(points: list[BinaryAdditionPoint], output_dir: Path) -> tuple[Path, Path]:
    _apply_article_rcparams()

    bits = [point.bits for point in points]
    cuccaro = [point.cuccaro_toffoli_count for point in points]
    gidney = [point.gidney_toffoli_count for point in points]
    alphatensor = [point.alphatensor_toffoli_count for point in points]

    fig, ax = plt.subplots(figsize=(3.05, 2.45))
    ax.plot(
        bits,
        cuccaro,
        color="#56B4E9",
        marker="o",
        linewidth=1.1,
        markersize=3.0,
        label="Cuccaro et al. (2004)",
    )
    ax.plot(
        bits,
        gidney,
        color="#8FC7FF",
        marker="s",
        linewidth=1.0,
        markersize=2.8,
        label="Gidney (2018)",
    )
    ax.plot(
        bits,
        alphatensor,
        color="#D55E00",
        marker="x",
        linewidth=1.0,
        markersize=3.8,
        label="AlphaTensor-Quantum",
    )

    ax.set_xlabel("Number of bits")
    ax.set_ylabel("Number of Toffoli gates")
    ax.set_xticks(bits)
    ax.set_yticks(list(range(2, 19, 2)))
    ax.set_xlim(2.8, 10.2)
    ax.set_ylim(1.5, 18.5)
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)

    png_path = output_dir / "fig4b_binary_addition.png"
    pdf_path = output_dir / "fig4b_binary_addition.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def plot_combined_fig4(
    finite_field_points: list[FiniteFieldPoint],
    binary_addition_points: list[BinaryAdditionPoint],
    output_dir: Path,
) -> tuple[Path, Path]:
    _apply_article_rcparams()

    exponents = [point.exponent_m for point in finite_field_points]
    original = [point.circuit_before_optimization_toffoli_count for point in finite_field_points]
    alphatensor_a = [point.alphatensor_toffoli_count for point in finite_field_points]
    karatsuba = [point.karatsuba_toffoli_estimate for point in finite_field_points]
    best_classical = [point.best_classical_toffoli_upper_bound for point in finite_field_points]

    bits = [point.bits for point in binary_addition_points]
    cuccaro = [point.cuccaro_toffoli_count for point in binary_addition_points]
    gidney = [point.gidney_toffoli_count for point in binary_addition_points]
    alphatensor_b = [point.alphatensor_toffoli_count for point in binary_addition_points]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.35, 2.8))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.92, wspace=0.30)

    line_original, = ax_a.plot(
        exponents,
        original,
        color="#56B4E9",
        marker="o",
        linewidth=1.1,
        markersize=3.0,
        label="Circuit before optimization",
    )
    line_alpha_a, = ax_a.plot(
        exponents,
        alphatensor_a,
        color="#D55E00",
        marker="x",
        linewidth=1.0,
        markersize=3.8,
        label="AlphaTensor-Quantum",
    )
    line_karatsuba, = ax_a.plot(
        exponents,
        karatsuba,
        color="#7F7F7F",
        linewidth=1.0,
        linestyle=(0, (2, 2)),
        label="Karatsuba's algorithm",
    )
    line_best_classical, = ax_a.plot(
        exponents,
        best_classical,
        color="#000000",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label="Best classical algorithm",
    )
    ax_a.set_xlabel("Exponent m")
    ax_a.set_ylabel("Number of Toffoli gates")
    ax_a.set_xticks(exponents)
    ax_a.set_yticks(list(range(0, 101, 20)))
    ax_a.set_xlim(1.8, 10.2)
    ax_a.set_ylim(0, 102)
    _style_axes(ax_a)
    legend_primary = ax_a.legend(
        handles=[line_original, line_alpha_a],
        frameon=False,
        loc="upper left",
        handlelength=2.0,
        borderaxespad=0.0,
    )
    ax_a.add_artist(legend_primary)
    ax_a.text(0.01, 0.53, "Classical baselines:", transform=ax_a.transAxes, fontsize=6.5)
    ax_a.legend(
        handles=[line_karatsuba, line_best_classical],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.50),
        handlelength=2.0,
        borderaxespad=0.0,
    )

    ax_b.plot(
        bits,
        cuccaro,
        color="#56B4E9",
        marker="o",
        linewidth=1.1,
        markersize=3.0,
        label="Cuccaro et al. (2004)",
    )
    ax_b.plot(
        bits,
        gidney,
        color="#8FC7FF",
        marker="s",
        linewidth=1.0,
        markersize=2.8,
        label="Gidney (2018)",
    )
    ax_b.plot(
        bits,
        alphatensor_b,
        color="#D55E00",
        marker="x",
        linewidth=1.0,
        markersize=3.8,
        label="AlphaTensor-Quantum",
    )
    ax_b.set_xlabel("Number of bits")
    ax_b.set_ylabel("Number of Toffoli gates")
    ax_b.set_xticks(bits)
    ax_b.set_yticks(list(range(2, 19, 2)))
    ax_b.set_xlim(2.8, 10.2)
    ax_b.set_ylim(1.5, 18.5)
    _style_axes(ax_b)
    ax_b.legend(frameon=False, loc="upper left", handlelength=2.0)

    ax_a.text(-0.12, 1.03, "a", transform=ax_a.transAxes, fontweight="bold", fontsize=8)
    ax_b.text(-0.12, 1.03, "b", transform=ax_b.transAxes, fontweight="bold", fontsize=8)

    png_path = output_dir / "fig4_reproduction.png"
    pdf_path = output_dir / "fig4_reproduction.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def generate_fig4b_artifacts(output_dir: Path) -> dict[str, Path]:
    points = build_binary_addition_points()
    validate_binary_addition_points(points)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = write_binary_addition_csv(points, output_dir)
    json_path = write_fig4b_json(points, output_dir)
    readme_path = write_fig4b_readme(points, output_dir)
    png_path, pdf_path = plot_binary_addition_panel(points, output_dir)
    return {
        "csv": csv_path,
        "json": json_path,
        "readme": readme_path,
        "png": png_path,
        "pdf": pdf_path,
    }


def generate_fig4_artifacts(output_dir: Path) -> dict[str, Path]:
    finite_field_points = build_finite_field_points()
    binary_addition_points = build_binary_addition_points()
    validate_finite_field_points(finite_field_points)
    validate_binary_addition_points(binary_addition_points)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig4a_csv_path = write_finite_field_csv(finite_field_points, output_dir)
    fig4b_csv_path = write_binary_addition_csv(binary_addition_points, output_dir)
    json_path = write_fig4_json(finite_field_points, binary_addition_points, output_dir)
    readme_path = write_fig4_readme(finite_field_points, binary_addition_points, output_dir)
    png_path, pdf_path = plot_combined_fig4(
        finite_field_points,
        binary_addition_points,
        output_dir,
    )
    return {
        "fig4a_csv": fig4a_csv_path,
        "fig4b_csv": fig4b_csv_path,
        "json": json_path,
        "readme": readme_path,
        "png": png_path,
        "pdf": pdf_path,
    }
