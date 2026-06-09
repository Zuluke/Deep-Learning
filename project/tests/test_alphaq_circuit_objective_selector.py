from __future__ import annotations

import scripts.train_alphaq_circuit_objective_selector as selector


def test_knn_selector_uses_circuit_features(monkeypatch) -> None:
    monkeypatch.setattr(selector, "tensor_features", fake_tensor_features)
    groups = {
        ("unit", "near_factor"): make_group("near_factor", "factor_count", tensor_size=1),
        ("unit", "near_mixed"): make_group("near_mixed", "mixed_pair", tensor_size=10),
        ("unit", "holdout"): make_group("holdout", "mixed_pair", tensor_size=9),
    }
    ranges = selector.feature_ranges(groups)

    chosen = selector.knn_selector(
        train_items=[groups[("unit", "near_factor")], groups[("unit", "near_mixed")]],
        holdout_items=groups[("unit", "holdout")],
        ranges=ranges,
        k=1,
    )

    assert chosen == "mixed_pair"


def test_softmax_selector_learns_simple_feature_boundary(monkeypatch) -> None:
    monkeypatch.setattr(selector, "tensor_features", fake_tensor_features)
    groups = {
        ("unit", "factor_a"): make_group("factor_a", "factor_count", tensor_size=1),
        ("unit", "factor_b"): make_group("factor_b", "factor_count", tensor_size=2),
        ("unit", "mixed_a"): make_group("mixed_a", "mixed_pair", tensor_size=9),
        ("unit", "mixed_b"): make_group("mixed_b", "mixed_pair", tensor_size=10),
        ("unit", "holdout"): make_group("holdout", "mixed_pair", tensor_size=8),
    }
    ranges = selector.feature_ranges(groups)

    chosen = selector.softmax_selector(
        train_items=[
            groups[("unit", "factor_a")],
            groups[("unit", "factor_b")],
            groups[("unit", "mixed_a")],
            groups[("unit", "mixed_b")],
        ],
        holdout_items=groups[("unit", "holdout")],
        ranges=ranges,
        seed=3,
        steps=300,
    )

    assert chosen == "mixed_pair"


def test_evaluation_rows_include_supervised_policies(monkeypatch) -> None:
    monkeypatch.setattr(selector, "tensor_features", fake_tensor_features)
    groups = {
        ("unit", "factor_a"): make_group("factor_a", "factor_count", tensor_size=1),
        ("unit", "factor_b"): make_group("factor_b", "factor_count", tensor_size=2),
        ("unit", "mixed_a"): make_group("mixed_a", "mixed_pair", tensor_size=9),
        ("unit", "mixed_b"): make_group("mixed_b", "mixed_pair", tensor_size=10),
    }

    rows = selector.evaluation_rows(groups, shuffle_seed=5)
    policies = {row["policy"] for row in rows}

    assert "circuit_1nn" in policies
    assert "circuit_knn3" in policies
    assert "circuit_softmax" in policies


def fake_tensor_features(target: str) -> dict[str, float]:
    return {
        "tensor_size": 0.0,
        "tensor_weight": 0.0,
        "tensor_density": 0.0,
        "tensor_index_degree_mean": 0.0,
        "tensor_index_degree_max": 0.0,
        "tensor_index_degree_std": 0.0,
        "tensor_pair_graph_edges": 0.0,
        "tensor_pair_graph_density": 0.0,
        "tensor_pair_graph_degree_mean": 0.0,
        "tensor_pair_graph_degree_max": 0.0,
        "tensor_pair_graph_degree_std": 0.0,
    }


def make_group(target: str, oracle: str, *, tensor_size: int) -> list[dict[str, str]]:
    return [
        make_row(target, "factor_count", oracle, tensor_size=tensor_size, tcount=1 if oracle == "factor_count" else 2),
        make_row(target, "mixed_pair", oracle, tensor_size=tensor_size, tcount=1 if oracle == "mixed_pair" else 2),
    ]


def make_row(
    target: str,
    objective: str,
    oracle: str,
    *,
    tensor_size: int,
    tcount: int,
) -> dict[str, str]:
    return {
        "source_split": "unit",
        "target": target,
        "objective_variant": objective,
        "oracle_objective": oracle,
        "train_ready": "true",
        "execution_status": "ok",
        "has_beam_candidate": "true",
        "n_qubits": "3",
        "tensor_size": str(tensor_size),
        "original_tcount": "4",
        "best_beam_tcount": str(tcount),
        "best_beam_primary_nc_depth_ratio": str(tcount),
        "best_beam_qasm_depth": str(10 * tcount),
        "objective_elapsed_sec": str(tcount),
    }
