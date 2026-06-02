from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path
import shutil
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


DEFAULT_FRONTIER_CSV = (
    DEFAULT_RESULTS_ROOT / "public_resynth_structural" / "candidate_frontier.csv"
)
DEFAULT_MODEL_JSON = DEFAULT_RESULTS_ROOT / "models" / "alphatensor_structural_reranker.json"
DEFAULT_PREDICTIONS_CSV = DEFAULT_CSV_ROOT / "alphatensor_reranker_predictions.csv"
DEFAULT_EVAL_CSV = DEFAULT_CSV_ROOT / "alphatensor_reranker_eval.csv"
DEFAULT_COMPARISON_CSV = DEFAULT_CSV_ROOT / "alphatensor_reranker_comparison.csv"
DEFAULT_BASELINES_CSV = DEFAULT_CSV_ROOT / "alphatensor_reranker_baselines.csv"
DEFAULT_REPORT_PATH = DEFAULT_REPORTS_ROOT / "alphatensor_reranker_report.md"
DEFAULT_SELECTION_ROOT = DEFAULT_RESULTS_ROOT / "public_resynth_reranker"
LABEL_COLUMN = "structural_cost"
PREDICTION_COLUMN = "predicted_structural_cost"
RERANKER_METHOD = "public_resynth_reranker"
FINAL_RERANKER_METHOD = "public_resynth_alphaq_final"
DEFAULT_PREDICTION_TOLERANCE = 0.10
DEFAULT_ENSEMBLE_SIZE = 7
DEFAULT_SEED_STRIDE = 9_973
MODEL_KIND_MLP = "mlp_regressor"
MODEL_KIND_PAIRWISE = "pairwise_ranker"
MODEL_KIND_PAIRWISE_MLP = "pairwise_mlp_ranker"
MODEL_KINDS = (MODEL_KIND_MLP, MODEL_KIND_PAIRWISE, MODEL_KIND_PAIRWISE_MLP)

FEATURE_COLUMNS = (
    "tcount_after",
    "tdepth_after",
    "depth_after",
    "gate_count_after",
    "rho_t",
    "rho_w",
    "n_clifford_blocks",
    "n_nonclifford_blocks",
    "avg_nonclifford_block_len",
    "hadamard_boundary_density",
    "tdepth_over_tcount",
    "tcount_ratio",
    "tdepth_ratio",
    "qasm_depth_ratio",
    "gate_count_ratio",
    "num_blocks",
    "num_gadget_blocks",
    "num_no_gadget_blocks",
    "gadget_block_fraction",
    "uses_any_gadget_source",
    "block_tcount_sum",
    "block_tdepth_sum",
    "alphaq_nc_core_area_ratio",
    "alphaq_nc_core_depth_ratio",
    "alphaq_dependency_core_area_ratio",
    "alphaq_dependency_core_depth_ratio",
    "alphaq_dependency_core_width_ratio",
    "alphaq_dependency_core_size",
    "alphaq_dependency_core_tcount",
    "alphaq_dependency_entangling_count",
    "alphaq_dependency_closure_rounds",
    "alphaq_dependency_internal_edge_count",
    "alphaq_dependency_boundary_edge_count",
    "alphaq_dependency_component_count",
    "alphaq_dependency_largest_component_fraction",
    "alphaq_dependency_chain_depth",
    "alphaq_dependency_edge_density",
)

DEPENDENCY_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column.startswith("alphaq_dependency_")
)
NO_DEPENDENCY_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if not column.startswith("alphaq_dependency_")
)


@dataclasses.dataclass(frozen=True)
class FeatureStats:
    means: np.ndarray
    stds: np.ndarray
    impute_values: np.ndarray


@dataclasses.dataclass(frozen=True)
class MLPModel:
    feature_columns: tuple[str, ...]
    stats: FeatureStats
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclasses.dataclass(frozen=True)
class PairwiseRankerModel:
    feature_columns: tuple[str, ...]
    stats: FeatureStats
    weights: np.ndarray
    bias: float
    output_transform: str = "identity"


@dataclasses.dataclass(frozen=True)
class PairwiseMLPRankerModel:
    feature_columns: tuple[str, ...]
    stats: FeatureStats
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    output_scale: float = 1.0
    output_bias: float = 0.0
    output_transform: str = "expm1"


@dataclasses.dataclass(frozen=True)
class EnsembleModel:
    models: tuple[MLPModel | PairwiseRankerModel | PairwiseMLPRankerModel, ...]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.models[0].feature_columns


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def coerce_int(value: Any) -> int | None:
    numeric = coerce_float(value)
    return None if numeric is None else int(numeric)


def rank_float(value: Any) -> float:
    numeric = coerce_float(value)
    return float("inf") if numeric is None else numeric


def valid_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("selection_status") == "ok"
        and coerce_float(row.get(LABEL_COLUMN)) is not None
        and row.get("alphaq_target_status") == "ok"
        and (
            coerce_float(row.get("alphaq_dependency_core_area_ratio")) is not None
            or coerce_float(row.get("alphaq_nc_core_area_ratio")) is not None
        )
    ]


def feature_matrix(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> np.ndarray:
    matrix = np.empty((len(rows), len(feature_columns)), dtype=float)
    for row_index, row in enumerate(rows):
        for feature_index, column in enumerate(feature_columns):
            value = coerce_float(row.get(column))
            matrix[row_index, feature_index] = np.nan if value is None else value
    return matrix


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([coerce_float(row[LABEL_COLUMN]) for row in rows], dtype=float)


def fit_feature_stats(matrix: np.ndarray) -> FeatureStats:
    impute_values = np.array(
        [
            float(np.median(values)) if len(values) else 0.0
            for values in (column[np.isfinite(column)] for column in matrix.T)
        ],
        dtype=float,
    )
    imputed = np.where(np.isnan(matrix), impute_values, matrix)
    means = imputed.mean(axis=0)
    stds = imputed.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    return FeatureStats(means=means, stds=stds, impute_values=impute_values)


def transform_features(matrix: np.ndarray, stats: FeatureStats) -> np.ndarray:
    imputed = np.where(np.isnan(matrix), stats.impute_values, matrix)
    return (imputed - stats.means) / stats.stds


def train_mlp(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    hidden_size: int = 8,
    epochs: int = 2_000,
    learning_rate: float = 0.03,
    weight_decay: float = 1e-4,
    seed: int = 2026,
) -> MLPModel:
    raw_features = feature_matrix(rows, feature_columns)
    stats = fit_feature_stats(raw_features)
    x = transform_features(raw_features, stats)
    y = labels(rows).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.15, size=(x.shape[1], hidden_size))
    b1 = np.zeros((hidden_size,), dtype=float)
    w2 = rng.normal(0.0, 0.15, size=(hidden_size, 1))
    b2 = np.zeros((1,), dtype=float)

    n = max(len(rows), 1)
    for _ in range(epochs):
        hidden = np.tanh(x @ w1 + b1)
        pred = hidden @ w2 + b2
        grad_pred = 2.0 * (pred - y) / n
        grad_w2 = hidden.T @ grad_pred + weight_decay * w2
        grad_b2 = grad_pred.sum(axis=0)
        grad_hidden = grad_pred @ w2.T
        grad_z1 = grad_hidden * (1.0 - hidden**2)
        grad_w1 = x.T @ grad_z1 + weight_decay * w1
        grad_b1 = grad_z1.sum(axis=0)

        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2

    return MLPModel(
        feature_columns=feature_columns,
        stats=stats,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )


def train_pairwise_ranker(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    epochs: int = 10_000,
    learning_rate: float = 0.02,
    weight_decay: float = 1e-3,
    seed: int = 2026,
) -> PairwiseRankerModel:
    raw_features = feature_matrix(rows, feature_columns)
    stats = fit_feature_stats(raw_features)
    x = transform_features(raw_features, stats)
    pair_winners, pair_losers, pair_weights = pairwise_training_pairs(rows)
    if len(pair_winners) == 0:
        raise ValueError("Pairwise ranker needs at least one within-circuit ordered pair.")

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, size=(x.shape[1],))
    bias = 0.0
    total_pair_weight = max(float(np.sum(pair_weights)), 1.0)
    diffs = x[pair_winners] - x[pair_losers]

    for _ in range(epochs):
        margins = diffs @ weights
        probs = _sigmoid(margins)
        weighted = pair_weights * probs / total_pair_weight
        grad_weights = weighted @ diffs + weight_decay * weights
        weights -= learning_rate * grad_weights

    raw_scores = x @ weights
    y = np.log1p(labels(rows))
    raw_variance = float(np.var(raw_scores))
    if raw_variance > 1e-12:
        slope = float(np.mean((raw_scores - raw_scores.mean()) * (y - y.mean())) / raw_variance)
        if slope > 0:
            bias = float(y.mean() - slope * raw_scores.mean())
            weights = weights * slope

    return PairwiseRankerModel(
        feature_columns=feature_columns,
        stats=stats,
        weights=weights,
        bias=bias,
        output_transform="expm1",
    )


def train_pairwise_mlp_ranker(
    rows: list[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    hidden_size: int = 12,
    epochs: int = 6_000,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-3,
    seed: int = 2026,
) -> PairwiseMLPRankerModel:
    raw_features = feature_matrix(rows, feature_columns)
    stats = fit_feature_stats(raw_features)
    x = transform_features(raw_features, stats)
    pair_winners, pair_losers, pair_weights = pairwise_training_pairs(rows)
    if len(pair_winners) == 0:
        raise ValueError("Pairwise MLP ranker needs at least one ordered pair.")

    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.12, size=(x.shape[1], hidden_size))
    b1 = np.zeros((hidden_size,), dtype=float)
    w2 = rng.normal(0.0, 0.12, size=(hidden_size, 1))
    b2 = np.zeros((1,), dtype=float)
    total_pair_weight = max(float(np.sum(pair_weights)), 1.0)

    for _ in range(epochs):
        hidden = np.tanh(x @ w1 + b1)
        scores = (hidden @ w2 + b2).reshape(-1)
        margins = scores[pair_winners] - scores[pair_losers]
        grad_margin = pair_weights * _sigmoid(margins) / total_pair_weight
        grad_scores = np.zeros_like(scores)
        np.add.at(grad_scores, pair_winners, grad_margin)
        np.add.at(grad_scores, pair_losers, -grad_margin)

        grad_w2 = hidden.T @ grad_scores.reshape(-1, 1) + weight_decay * w2
        grad_b2 = np.array([grad_scores.sum()])
        grad_hidden = grad_scores.reshape(-1, 1) @ w2.T
        grad_z1 = grad_hidden * (1.0 - hidden**2)
        grad_w1 = x.T @ grad_z1 + weight_decay * w1
        grad_b1 = grad_z1.sum(axis=0)

        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2

    raw_scores = (np.tanh(x @ w1 + b1) @ w2 + b2).reshape(-1)
    y = np.log1p(labels(rows))
    raw_variance = float(np.var(raw_scores))
    output_scale = 1.0
    output_bias = 0.0
    if raw_variance > 1e-12:
        slope = float(np.mean((raw_scores - raw_scores.mean()) * (y - y.mean())) / raw_variance)
        if slope > 0:
            output_scale = slope
            output_bias = float(y.mean() - slope * raw_scores.mean())

    return PairwiseMLPRankerModel(
        feature_columns=feature_columns,
        stats=stats,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        output_scale=output_scale,
        output_bias=output_bias,
        output_transform="expm1",
    )


def pairwise_training_pairs(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    winners: list[int] = []
    losers: list[int] = []
    weights: list[float] = []
    labels_array = labels(rows)
    positive_gaps: list[float] = []
    grouped_indices = [
        [index for index, row in enumerate(rows) if row["circuit_id"] == circuit_id]
        for circuit_id in sorted({row["circuit_id"] for row in rows}, key=natural_sort_key)
    ]
    for indices in grouped_indices:
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                left_value = float(labels_array[left_index])
                right_value = float(labels_array[right_index])
                if abs(left_value - right_value) < 1e-12:
                    continue
                winner, loser = (
                    (left_index, right_index)
                    if left_value < right_value
                    else (right_index, left_index)
                )
                gap = abs(left_value - right_value)
                winners.append(winner)
                losers.append(loser)
                positive_gaps.append(gap)

    scale = float(np.median(positive_gaps)) if positive_gaps else 1.0
    scale = max(scale, 1e-9)
    for gap in positive_gaps:
        weights.append(float(np.clip(gap / scale, 0.25, 4.0)))

    return (
        np.array(winners, dtype=int),
        np.array(losers, dtype=int),
        np.array(weights, dtype=float),
    )


def train_model(
    rows: list[dict[str, Any]],
    *,
    model_kind: str,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    hidden_size: int = 8,
    epochs: int = 2_000,
    learning_rate: float = 0.03,
    weight_decay: float = 1e-4,
    seed: int = 2026,
) -> MLPModel | PairwiseRankerModel | PairwiseMLPRankerModel:
    if model_kind == MODEL_KIND_MLP:
        return train_mlp(
            rows,
            feature_columns=feature_columns,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )
    if model_kind == MODEL_KIND_PAIRWISE:
        return train_pairwise_ranker(
            rows,
            feature_columns=feature_columns,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )
    if model_kind == MODEL_KIND_PAIRWISE_MLP:
        return train_pairwise_mlp_ranker(
            rows,
            feature_columns=feature_columns,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def train_ensemble(
    rows: list[dict[str, Any]],
    *,
    model_kind: str = MODEL_KIND_MLP,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    hidden_size: int = 8,
    epochs: int = 2_000,
    learning_rate: float = 0.03,
    weight_decay: float = 1e-4,
    seed: int = 2026,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
) -> EnsembleModel:
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be at least 1")
    models = tuple(
        train_model(
            rows,
            model_kind=model_kind,
            feature_columns=feature_columns,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + index * DEFAULT_SEED_STRIDE,
        )
        for index in range(ensemble_size)
    )
    return EnsembleModel(models=models)


def predict(model: MLPModel, rows: list[dict[str, Any]]) -> np.ndarray:
    x = transform_features(feature_matrix(rows, model.feature_columns), model.stats)
    hidden = np.tanh(x @ model.w1 + model.b1)
    return (hidden @ model.w2 + model.b2).reshape(-1)


def predict_pairwise(model: PairwiseRankerModel, rows: list[dict[str, Any]]) -> np.ndarray:
    x = transform_features(feature_matrix(rows, model.feature_columns), model.stats)
    raw = (x @ model.weights + model.bias).reshape(-1)
    if model.output_transform == "expm1":
        return np.expm1(np.clip(raw, 0.0, None))
    return raw


def predict_pairwise_mlp(
    model: PairwiseMLPRankerModel,
    rows: list[dict[str, Any]],
) -> np.ndarray:
    x = transform_features(feature_matrix(rows, model.feature_columns), model.stats)
    hidden = np.tanh(x @ model.w1 + model.b1)
    raw = (hidden @ model.w2 + model.b2).reshape(-1)
    calibrated = model.output_scale * raw + model.output_bias
    if model.output_transform == "expm1":
        return np.expm1(np.clip(calibrated, 0.0, None))
    return calibrated


def predict_ensemble(model: EnsembleModel, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack([predict_any(member, rows) for member in model.models])
    return stacked.mean(axis=0), stacked.std(axis=0)


def predict_any(
    model: MLPModel | PairwiseRankerModel | PairwiseMLPRankerModel,
    rows: list[dict[str, Any]],
) -> np.ndarray:
    if isinstance(model, PairwiseMLPRankerModel):
        return predict_pairwise_mlp(model, rows)
    if isinstance(model, PairwiseRankerModel):
        return predict_pairwise(model, rows)
    return predict(model, rows)


def predict_with_uncertainty(
    model: MLPModel | PairwiseRankerModel | PairwiseMLPRankerModel | EnsembleModel,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model, EnsembleModel):
        return predict_ensemble(model, rows)
    predictions = predict_any(model, rows)
    return predictions, np.zeros_like(predictions)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def select_with_prediction_tolerance(
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    *,
    prediction_tolerance: float,
) -> int:
    best_prediction = float(np.min(predictions))
    eligible = [
        index
        for index, prediction in enumerate(predictions)
        if float(prediction) <= best_prediction + prediction_tolerance
    ]

    def tolerance_key(index: int) -> tuple[float, float, float, float, float, int]:
        row = rows[index]
        tcount = coerce_float(row.get("tcount_after"))
        crossing_closure = coerce_float(row.get("alphaq_crossing_closure_count"))
        alphaq_total_depth = coerce_float(row.get("alphaq_total_depth_ratio"))
        qasm_depth = coerce_float(row.get("qasm_depth_ratio"))
        combo_index = coerce_int(row.get("combo_index"))
        return (
            float("inf") if tcount is None else tcount,
            float("inf") if crossing_closure is None else crossing_closure,
            float("inf") if alphaq_total_depth is None else alphaq_total_depth,
            float("inf") if qasm_depth is None else qasm_depth,
            float(predictions[index]),
            10**9 if combo_index is None else combo_index,
        )

    return min(eligible, key=tolerance_key)


def leave_one_circuit_out_eval(
    rows: list[dict[str, str]],
    *,
    model_kind: str = MODEL_KIND_MLP,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    prediction_tolerance: float = DEFAULT_PREDICTION_TOLERANCE,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
) -> list[dict[str, Any]]:
    circuits = sorted({row["circuit_id"] for row in rows}, key=natural_sort_key)
    eval_rows = []
    for circuit_id in circuits:
        train_rows = [row for row in rows if row["circuit_id"] != circuit_id]
        test_rows = [row for row in rows if row["circuit_id"] == circuit_id]
        if not train_rows or not test_rows:
            continue
        model = train_ensemble(
            train_rows,
            model_kind=model_kind,
            feature_columns=feature_columns,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            ensemble_size=ensemble_size,
        )
        predictions, prediction_std = predict_ensemble(model, test_rows)
        selected_index = select_with_prediction_tolerance(
            test_rows,
            predictions,
            prediction_tolerance=prediction_tolerance,
        )
        true_values = labels(test_rows)
        true_best_index = int(np.argmin(true_values))
        tcount_best_index = min(
            range(len(test_rows)),
            key=lambda index: (
                rank_float(test_rows[index].get("tcount_after")),
                float(true_values[index]),
                (
                    10**9
                    if coerce_int(test_rows[index].get("combo_index")) is None
                    else coerce_int(test_rows[index].get("combo_index"))
                ),
            ),
        )
        selected = test_rows[selected_index]
        true_best = test_rows[true_best_index]
        tcount_best = test_rows[tcount_best_index]
        selected_true = float(true_values[selected_index])
        true_best_value = float(true_values[true_best_index])
        tcount_best_value = float(true_values[tcount_best_index])
        eval_rows.append(
            {
                "circuit_id": circuit_id,
                "num_train_candidates": len(train_rows),
                "num_test_candidates": len(test_rows),
                "selected_candidate_id": selected["candidate_id"],
                "selected_predicted_primary": float(predictions[selected_index]),
                "selected_prediction_std": float(prediction_std[selected_index]),
                "selected_prediction_margin_from_best": float(
                    predictions[selected_index] - np.min(predictions)
                ),
                "prediction_tolerance": prediction_tolerance,
                "ensemble_size": ensemble_size,
                "model_kind": model_kind,
                "selected_true_primary": selected_true,
                "selected_tcount": selected.get("tcount_after"),
                "true_best_candidate_id": true_best["candidate_id"],
                "true_best_primary": true_best_value,
                "tcount_best_candidate_id": tcount_best["candidate_id"],
                "tcount_best_primary": tcount_best_value,
                "tcount_best_tcount": tcount_best.get("tcount_after"),
                "primary_regret_vs_true_best": selected_true - true_best_value,
                "primary_gain_vs_tcount_best": tcount_best_value - selected_true,
                "hit_true_best": selected["candidate_id"] == true_best["candidate_id"],
            }
        )
    return eval_rows


def predictions_rows(
    rows: list[dict[str, str]],
    model: MLPModel | PairwiseRankerModel | EnsembleModel,
    *,
    prediction_tolerance: float = DEFAULT_PREDICTION_TOLERANCE,
) -> list[dict[str, Any]]:
    preds, prediction_std = predict_with_uncertainty(model, rows)
    selected_by_circuit = {}
    for circuit_id in sorted({row["circuit_id"] for row in rows}, key=natural_sort_key):
        indices = [idx for idx, row in enumerate(rows) if row["circuit_id"] == circuit_id]
        local_rows = [rows[index] for index in indices]
        local_predictions = np.array([preds[index] for index in indices], dtype=float)
        selected_local_index = select_with_prediction_tolerance(
            local_rows,
            local_predictions,
            prediction_tolerance=prediction_tolerance,
        )
        selected_by_circuit[circuit_id] = indices[selected_local_index]
    return [
        {
            **row,
            PREDICTION_COLUMN: float(preds[index]),
            "predicted_primary_nc_depth_ratio": None,
            "prediction_std": float(prediction_std[index]),
            "prediction_margin_from_best": float(
                preds[index]
                - min(
                    preds[idx]
                    for idx, item in enumerate(rows)
                    if item["circuit_id"] == row["circuit_id"]
                )
            ),
            "prediction_tolerance": prediction_tolerance,
            "reranker_selected": index == selected_by_circuit[row["circuit_id"]],
        }
        for index, row in enumerate(rows)
    ]


def selected_row_by_key(
    rows: list[dict[str, Any]],
    key: tuple[str, ...],
) -> dict[str, Any]:
    def row_key(row: dict[str, Any]) -> tuple[float, ...]:
        return tuple(rank_float(row.get(column)) for column in key)

    return min(rows, key=row_key)


def baseline_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    objectives = {
        "structural_oracle": (LABEL_COLUMN, "tcount_after", "qasm_depth_ratio"),
        "tcount": ("tcount_after", LABEL_COLUMN, "qasm_depth_ratio"),
        "tdepth": ("tdepth_after", LABEL_COLUMN, "tcount_after"),
        "qasm_depth": ("qasm_depth_ratio", LABEL_COLUMN, "tcount_after"),
        "zx_total_depth": ("zx_total_depth_ratio", LABEL_COLUMN, "tcount_after"),
    }
    baseline_results = []
    for circuit_id in sorted({row["circuit_id"] for row in rows}, key=natural_sort_key):
        circuit_rows = [row for row in rows if row["circuit_id"] == circuit_id]
        structural_best = selected_row_by_key(
            circuit_rows, (LABEL_COLUMN, "tcount_after", "qasm_depth_ratio")
        )
        tcount_best = selected_row_by_key(
            circuit_rows, ("tcount_after", LABEL_COLUMN, "qasm_depth_ratio")
        )
        structural_primary = coerce_float(structural_best.get(LABEL_COLUMN))
        tcount_primary = coerce_float(tcount_best.get(LABEL_COLUMN))
        for objective, key in objectives.items():
            selected = selected_row_by_key(circuit_rows, key)
            selected_primary = coerce_float(selected.get(LABEL_COLUMN))
            selected_tcount = coerce_float(selected.get("tcount_after"))
            structural_tcount = coerce_float(structural_best.get("tcount_after"))
            tcount_tcount = coerce_float(tcount_best.get("tcount_after"))
            baseline_results.append(
                {
                    "circuit_id": circuit_id,
                    "objective": objective,
                    "selected_candidate_id": selected.get("candidate_id"),
                    "selected_primary": selected_primary,
                    "selected_tcount": selected_tcount,
                    "structural_best_candidate_id": structural_best.get("candidate_id"),
                    "structural_best_primary": structural_primary,
                    "structural_best_tcount": structural_tcount,
                    "tcount_best_candidate_id": tcount_best.get("candidate_id"),
                    "tcount_best_primary": tcount_primary,
                    "tcount_best_tcount": tcount_tcount,
                    "primary_regret_vs_structural_best": (
                        None
                        if selected_primary is None or structural_primary is None
                        else selected_primary - structural_primary
                    ),
                    "primary_gain_vs_tcount_best": (
                        None
                        if selected_primary is None or tcount_primary is None
                        else tcount_primary - selected_primary
                    ),
                    "tcount_delta_vs_structural_best": (
                        None
                        if selected_tcount is None or structural_tcount is None
                        else selected_tcount - structural_tcount
                    ),
                    "tcount_delta_vs_tcount_best": (
                        None
                        if selected_tcount is None or tcount_tcount is None
                        else selected_tcount - tcount_tcount
                    ),
                }
            )
    return baseline_results


def comparison_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for circuit_id in sorted(
        {row["circuit_id"] for row in prediction_rows}, key=natural_sort_key
    ):
        circuit_rows = [
            row for row in prediction_rows if row["circuit_id"] == circuit_id
        ]
        selected = next(row for row in circuit_rows if row["reranker_selected"])
        true_best = min(
            circuit_rows,
            key=lambda row: rank_float(row.get(LABEL_COLUMN)),
        )
        tcount_best = min(
            circuit_rows,
            key=lambda row: (
                rank_float(row.get("tcount_after")),
                rank_float(row.get(LABEL_COLUMN)),
            ),
        )
        selected_primary = coerce_float(selected.get(LABEL_COLUMN))
        true_best_primary = coerce_float(true_best.get(LABEL_COLUMN))
        tcount_best_primary = coerce_float(tcount_best.get(LABEL_COLUMN))
        selected_tcount = coerce_float(selected.get("tcount_after"))
        true_best_tcount = coerce_float(true_best.get("tcount_after"))
        tcount_best_tcount = coerce_float(tcount_best.get("tcount_after"))
        rows.append(
            {
                "circuit_id": circuit_id,
                "reranker_candidate_id": selected.get("candidate_id"),
                "reranker_predicted_primary": selected.get(
                    PREDICTION_COLUMN
                ),
                "reranker_primary": selected_primary,
                "reranker_tcount": selected_tcount,
                "structural_best_candidate_id": true_best.get("candidate_id"),
                "structural_best_primary": true_best_primary,
                "structural_best_tcount": true_best_tcount,
                "tcount_best_candidate_id": tcount_best.get("candidate_id"),
                "tcount_best_primary": tcount_best_primary,
                "tcount_best_tcount": tcount_best_tcount,
                "primary_regret_vs_structural_best": (
                    None
                    if selected_primary is None or true_best_primary is None
                    else selected_primary - true_best_primary
                ),
                "primary_gain_vs_tcount_best": (
                    None
                    if selected_primary is None or tcount_best_primary is None
                    else tcount_best_primary - selected_primary
                ),
                "tcount_delta_vs_structural_best": (
                    None
                    if selected_tcount is None or true_best_tcount is None
                    else selected_tcount - true_best_tcount
                ),
                "tcount_delta_vs_tcount_best": (
                    None
                    if selected_tcount is None or tcount_best_tcount is None
                    else selected_tcount - tcount_best_tcount
                ),
            }
        )
    return rows


def project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def materialize_reranker_selection(
    prediction_rows: list[dict[str, Any]],
    *,
    output_root: Path,
    model_path: Path,
    prediction_tolerance: float,
    selection_objective: str = "mlp_reranker",
    method_name: str = RERANKER_METHOD,
) -> tuple[list[dict[str, Any]], Path, Path]:
    summary_rows = []
    method_root = ensure_dir(output_root / method_name)
    for row in prediction_rows:
        if not row.get("reranker_selected"):
            continue
        circuit_id = row["circuit_id"]
        circuit_dir = ensure_dir(method_root / circuit_id)
        source_qasm = project_path(row.get("candidate_qasm_path"))
        destination_qasm = circuit_dir / "assembled.qasm"
        status = "ok"
        error = None
        if source_qasm is None or not source_qasm.exists():
            status = "missing-qasm"
            error = f"Missing selected candidate QASM: {row.get('candidate_qasm_path')}"
            assembled_qasm_path = None
        else:
            shutil.copyfile(source_qasm, destination_qasm)
            assembled_qasm_path = str(destination_qasm)
        summary_row = {
            "circuit_id": circuit_id,
            "method": method_name,
            "status": status,
            "selection_objective": selection_objective,
            "selection_status": row.get("selection_status"),
            "selection_error": error or row.get("selection_error"),
            "source_candidate_id": row.get("candidate_id"),
            PREDICTION_COLUMN: row.get(PREDICTION_COLUMN),
            "predicted_primary_nc_depth_ratio": row.get(
                "predicted_primary_nc_depth_ratio"
            ),
            "prediction_std": row.get("prediction_std"),
            "prediction_tolerance": prediction_tolerance,
            "prediction_margin_from_best": row.get("prediction_margin_from_best"),
            "structural_cost": row.get("structural_cost"),
            "alphaq_target_status": row.get("alphaq_target_status"),
            "alphaq_target_error": row.get("alphaq_target_error"),
            "alphaq_nc_core_area_ratio": row.get("alphaq_nc_core_area_ratio"),
            "alphaq_dependency_core_area_ratio": row.get(
                "alphaq_dependency_core_area_ratio"
            ),
            "alphaq_nc_core_area_delta_vs_original": row.get(
                "alphaq_nc_core_area_delta_vs_original"
            ),
            "alphaq_dependency_core_area_delta_vs_original": row.get(
                "alphaq_dependency_core_area_delta_vs_original"
            ),
            "alphaq_nc_core_depth_ratio": row.get("alphaq_nc_core_depth_ratio"),
            "alphaq_nc_core_width_ratio": row.get("alphaq_nc_core_width_ratio"),
            "alphaq_dependency_core_depth_ratio": row.get(
                "alphaq_dependency_core_depth_ratio"
            ),
            "alphaq_dependency_core_width_ratio": row.get(
                "alphaq_dependency_core_width_ratio"
            ),
            "alphaq_total_depth_ratio": row.get("alphaq_total_depth_ratio"),
            "alphaq_total_area_ratio": row.get("alphaq_total_area_ratio"),
            "alphaq_border_status": row.get("alphaq_border_status"),
            "alphaq_border_error": row.get("alphaq_border_error"),
            "alphaq_total_depth": row.get("alphaq_total_depth"),
            "alphaq_total_width": row.get("alphaq_total_width"),
            "alphaq_total_area": row.get("alphaq_total_area"),
            "alphaq_nc_core_depth": row.get("alphaq_nc_core_depth"),
            "alphaq_nc_core_width": row.get("alphaq_nc_core_width"),
            "alphaq_nc_core_area": row.get("alphaq_nc_core_area"),
            "alphaq_core_tcount": row.get("alphaq_core_tcount"),
            "alphaq_crossing_closure_count": row.get(
                "alphaq_crossing_closure_count"
            ),
            "alphaq_dependency_core_depth": row.get("alphaq_dependency_core_depth"),
            "alphaq_dependency_core_width": row.get("alphaq_dependency_core_width"),
            "alphaq_dependency_core_area": row.get("alphaq_dependency_core_area"),
            "alphaq_dependency_core_size": row.get("alphaq_dependency_core_size"),
            "alphaq_dependency_core_tcount": row.get("alphaq_dependency_core_tcount"),
            "alphaq_dependency_entangling_count": row.get(
                "alphaq_dependency_entangling_count"
            ),
            "alphaq_dependency_closure_rounds": row.get(
                "alphaq_dependency_closure_rounds"
            ),
            "alphaq_dependency_internal_edge_count": row.get(
                "alphaq_dependency_internal_edge_count"
            ),
            "alphaq_dependency_boundary_edge_count": row.get(
                "alphaq_dependency_boundary_edge_count"
            ),
            "alphaq_dependency_component_count": row.get(
                "alphaq_dependency_component_count"
            ),
            "alphaq_dependency_largest_component_size": row.get(
                "alphaq_dependency_largest_component_size"
            ),
            "alphaq_dependency_largest_component_fraction": row.get(
                "alphaq_dependency_largest_component_fraction"
            ),
            "alphaq_dependency_chain_depth": row.get("alphaq_dependency_chain_depth"),
            "alphaq_dependency_edge_density": row.get("alphaq_dependency_edge_density"),
            "alphaq_left_nc_core_depth": row.get("alphaq_left_nc_core_depth"),
            "alphaq_left_nc_core_width": row.get("alphaq_left_nc_core_width"),
            "alphaq_left_nc_core_area": row.get("alphaq_left_nc_core_area"),
            "alphaq_left_crossing_closure_count": row.get(
                "alphaq_left_crossing_closure_count"
            ),
            "alphaq_right_nc_core_depth": row.get("alphaq_right_nc_core_depth"),
            "alphaq_right_nc_core_width": row.get("alphaq_right_nc_core_width"),
            "alphaq_right_nc_core_area": row.get("alphaq_right_nc_core_area"),
            "alphaq_right_crossing_closure_count": row.get(
                "alphaq_right_crossing_closure_count"
            ),
            "alphaq_prefix_clifford_depth": row.get("alphaq_prefix_clifford_depth"),
            "alphaq_suffix_clifford_depth": row.get("alphaq_suffix_clifford_depth"),
            "alphaq_clifford_shaved_depth_fraction": row.get(
                "alphaq_clifford_shaved_depth_fraction"
            ),
            "primary_nc_depth_ratio": row.get("primary_nc_depth_ratio"),
            "zx_total_depth_ratio": row.get("zx_total_depth_ratio"),
            "qasm_depth_ratio": row.get("qasm_depth_ratio"),
            "tcount_ratio": row.get("tcount_ratio"),
            "tcount_after_selection": row.get("tcount_after"),
            "tdepth_after_selection": row.get("tdepth_after"),
            "combo_index": row.get("combo_index"),
            "assembled_qasm_path": assembled_qasm_path,
            "model_json": str(model_path),
            "error": error,
        }
        write_json(summary_row, circuit_dir / "summary.json")
        summary_rows.append(summary_row)

    summary_csv = output_root / "public_resynth_summary.csv"
    summary_json = output_root / "public_resynth_summary.json"
    write_csv_rows(summary_rows, summary_csv)
    write_json(
        {
            "method": method_name,
            "selection_objective": selection_objective,
            "model_json": str(model_path),
            "prediction_tolerance": prediction_tolerance,
            "num_rows": len(summary_rows),
            "summary_csv": str(summary_csv),
        },
        summary_json,
    )
    return summary_rows, summary_csv, summary_json


def mlp_payload(model: MLPModel) -> dict[str, Any]:
    return {
        "feature_means": model.stats.means.tolist(),
        "feature_stds": model.stats.stds.tolist(),
        "feature_impute_values": model.stats.impute_values.tolist(),
        "w1": model.w1.tolist(),
        "b1": model.b1.tolist(),
        "w2": model.w2.tolist(),
        "b2": model.b2.tolist(),
    }


def pairwise_payload(model: PairwiseRankerModel) -> dict[str, Any]:
    return {
        "feature_means": model.stats.means.tolist(),
        "feature_stds": model.stats.stds.tolist(),
        "feature_impute_values": model.stats.impute_values.tolist(),
        "weights": model.weights.tolist(),
        "bias": model.bias,
        "output_transform": model.output_transform,
    }


def pairwise_mlp_payload(model: PairwiseMLPRankerModel) -> dict[str, Any]:
    return {
        "feature_means": model.stats.means.tolist(),
        "feature_stds": model.stats.stds.tolist(),
        "feature_impute_values": model.stats.impute_values.tolist(),
        "w1": model.w1.tolist(),
        "b1": model.b1.tolist(),
        "w2": model.w2.tolist(),
        "b2": model.b2.tolist(),
        "output_scale": model.output_scale,
        "output_bias": model.output_bias,
        "output_transform": model.output_transform,
    }


def member_payload(
    model: MLPModel | PairwiseRankerModel | PairwiseMLPRankerModel,
) -> dict[str, Any]:
    if isinstance(model, PairwiseMLPRankerModel):
        return pairwise_mlp_payload(model)
    if isinstance(model, PairwiseRankerModel):
        return pairwise_payload(model)
    return mlp_payload(model)


def model_payload(model: EnsembleModel, args: argparse.Namespace, rows: list[dict[str, str]]) -> dict[str, Any]:
    first_model = model.models[0]
    model_type = (
        "pairwise_linear_ranker_ensemble"
        if isinstance(first_model, PairwiseRankerModel)
        else "pairwise_mlp_ranker_ensemble"
        if isinstance(first_model, PairwiseMLPRankerModel)
        else "one_hidden_layer_tanh_mlp_ensemble"
    )
    return {
        "model_type": model_type,
        "model_kind": args.model_kind,
        "feature_columns": list(model.feature_columns),
        "label_column": LABEL_COLUMN,
        "num_training_candidates": len(rows),
        "circuits": sorted({row["circuit_id"] for row in rows}, key=natural_sort_key),
        "hidden_size": (
            int(first_model.w1.shape[1])
            if isinstance(first_model, (MLPModel, PairwiseMLPRankerModel))
            else None
        ),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "seed_stride": DEFAULT_SEED_STRIDE,
        "ensemble_size": args.ensemble_size,
        "prediction_tolerance": args.prediction_tolerance,
        "members": [member_payload(member) for member in model.models],
    }


def write_report(
    *,
    output_path: Path,
    rows: list[dict[str, str]],
    eval_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    model_path: Path,
    eval_csv: Path,
    predictions_csv: Path,
    comparison_csv: Path,
    comparison_rows_for_report: list[dict[str, Any]],
    baseline_csv: Path,
    baseline_rows_for_report: list[dict[str, Any]],
    selection_summary_csv: Path,
    prediction_tolerance: float,
    ensemble_size: int,
    model_kind: str,
) -> Path:
    circuits = sorted({row["circuit_id"] for row in rows}, key=natural_sort_key)
    hit_rate = (
        0.0
        if not eval_rows
        else sum(bool(row["hit_true_best"]) for row in eval_rows) / len(eval_rows)
    )
    mean_regret = (
        0.0
        if not eval_rows
        else float(np.mean([row["primary_regret_vs_true_best"] for row in eval_rows]))
    )
    mean_gain_vs_tcount = (
        0.0
        if not eval_rows
        else float(np.mean([row["primary_gain_vs_tcount_best"] for row in eval_rows]))
    )
    selected_lines = []
    for row in prediction_rows:
        if row["reranker_selected"]:
            selected_lines.append(
                (
                    f"- `{row['circuit_id']}`: `{row['candidate_id']}` "
                    f"pred={float(row[PREDICTION_COLUMN]):.3f}, "
                    f"std={float(row['prediction_std']):.3f}, "
                    f"true={float(row[LABEL_COLUMN]):.3f}, T={row['tcount_after']}."
                )
            )
    eval_lines = [
        (
            f"- `{row['circuit_id']}`: selected `{row['selected_candidate_id']}`, "
            f"true best `{row['true_best_candidate_id']}`, "
            f"regret={float(row['primary_regret_vs_true_best']):.3f}, "
            f"gain-vs-T-best={float(row['primary_gain_vs_tcount_best']):.3f}."
        )
        for row in eval_rows
    ]
    comparison_lines = [
        (
            f"- `{row['circuit_id']}`: reranker `{row['reranker_candidate_id']}` "
            f"primary={float(row['reranker_primary']):.3f}, T={row['reranker_tcount']}; "
            f"structural-best `{row['structural_best_candidate_id']}` "
            f"primary={float(row['structural_best_primary']):.3f}, "
            f"T={row['structural_best_tcount']}; "
            f"T-best `{row['tcount_best_candidate_id']}` "
            f"primary={float(row['tcount_best_primary']):.3f}, "
            f"T={row['tcount_best_tcount']}."
        )
        for row in comparison_rows_for_report
    ]
    objective_summaries = []
    for objective in sorted({row["objective"] for row in baseline_rows_for_report}):
        subset = [row for row in baseline_rows_for_report if row["objective"] == objective]
        regrets = [coerce_float(row["primary_regret_vs_structural_best"]) for row in subset]
        regrets = [item for item in regrets if item is not None]
        t_delta = [coerce_float(row["tcount_delta_vs_structural_best"]) for row in subset]
        t_delta = [item for item in t_delta if item is not None]
        objective_summaries.append(
            f"- `{objective}`: mean regret={float(np.mean(regrets)):.3f}, "
            f"total T-delta={float(np.sum(t_delta)):.0f}."
        )
    text = [
        "# AlphaTensor Structural Reranker",
        "",
        "## Dataset",
        "",
        f"- Candidates: {len(rows)}.",
        f"- Circuits: {', '.join(f'`{item}`' for item in circuits)}.",
        f"- Label: `{LABEL_COLUMN}`.",
        f"- Model kind: `{model_kind}`.",
        "- Features are cheap QASM/decomposition statistics; ZX-derived target columns are not used as features.",
        f"- Predictions CSV: `{predictions_csv}`.",
        f"- Evaluation CSV: `{eval_csv}`.",
        f"- Comparison CSV: `{comparison_csv}`.",
        f"- Baseline CSV: `{baseline_csv}`.",
        f"- Model JSON: `{model_path}`.",
        f"- Reranker selection summary: `{selection_summary_csv}`.",
        f"- Prediction tolerance: {prediction_tolerance:.3f}.",
        f"- Ensemble size: {ensemble_size}.",
        "",
        "## Leave-one-circuit-out evaluation",
        "",
        f"- Hit rate against true best candidate: {hit_rate:.3f}.",
        f"- Mean primary regret vs true best: {mean_regret:.3f}.",
        f"- Mean primary gain vs T-count-best: {mean_gain_vs_tcount:.3f}.",
        *eval_lines,
        "",
        "## Full-dataset fitted selections",
        "",
        *selected_lines,
        "",
        "## Selection comparison",
        "",
        *comparison_lines,
        "",
        "## Objective baselines",
        "",
        *objective_summaries,
        "",
        "## Caveat",
        "",
        (
            "This is a small proof-of-concept dataset. Treat the model as a reranking "
            "probe for whether cheap features carry splitting signal, not as a mature "
            "general-purpose predictor yet."
        ),
        "",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a small MLP reranker for AlphaTensor structural candidates."
    )
    parser.add_argument("--frontier-csv", type=Path, default=DEFAULT_FRONTIER_CSV)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--baselines-csv", type=Path, default=DEFAULT_BASELINES_CSV)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--selection-output-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument(
        "--model-kind",
        choices=MODEL_KINDS,
        default=MODEL_KIND_MLP,
    )
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    parser.add_argument(
        "--prediction-tolerance",
        type=float,
        default=DEFAULT_PREDICTION_TOLERANCE,
        help=(
            "Prefer lower T-count among candidates whose predicted structural cost "
            "is within this absolute margin of the best prediction."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = valid_candidate_rows(load_csv_rows(args.frontier_csv))
    if not rows:
        raise SystemExit(f"No valid candidate rows found in {args.frontier_csv}")

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
    selection_rows, selection_summary_csv, selection_summary_json = (
        materialize_reranker_selection(
            prediction_rows,
            output_root=args.selection_output_root,
            model_path=args.model_json,
            prediction_tolerance=args.prediction_tolerance,
            selection_objective=args.model_kind,
        )
    )

    write_csv_rows(eval_rows, args.eval_csv)
    write_csv_rows(prediction_rows, args.predictions_csv)
    write_csv_rows(comparison, args.comparison_csv)
    write_csv_rows(baselines, args.baselines_csv)
    write_json(model_payload(model, args, rows), args.model_json)
    report_path = write_report(
        output_path=args.report_path,
        rows=rows,
        eval_rows=eval_rows,
        prediction_rows=prediction_rows,
        model_path=args.model_json,
        eval_csv=args.eval_csv,
        predictions_csv=args.predictions_csv,
        comparison_csv=args.comparison_csv,
        comparison_rows_for_report=comparison,
        baseline_csv=args.baselines_csv,
        baseline_rows_for_report=baselines,
        selection_summary_csv=selection_summary_csv,
        prediction_tolerance=args.prediction_tolerance,
        ensemble_size=args.ensemble_size,
        model_kind=args.model_kind,
    )
    print(
        {
            "num_candidates": len(rows),
            "eval_csv": str(args.eval_csv),
            "predictions_csv": str(args.predictions_csv),
            "comparison_csv": str(args.comparison_csv),
            "baselines_csv": str(args.baselines_csv),
            "model_json": str(args.model_json),
            "report_path": str(report_path),
            "selection_summary_csv": str(selection_summary_csv),
            "selection_summary_json": str(selection_summary_json),
            "num_selected": len(selection_rows),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
