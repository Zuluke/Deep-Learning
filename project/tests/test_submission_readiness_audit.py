from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_submission_readiness import PASS
from scripts.audit_submission_readiness import WARN
from scripts.audit_submission_readiness import audit_alphaq_loco
from scripts.audit_submission_readiness import audit_guard_surface
from scripts.audit_submission_readiness import audit_tensor_v3_selection_manifest
from scripts.audit_submission_readiness import audit_tensor_v3_profile
from scripts.audit_submission_readiness import forbidden_selection_columns
from scripts.audit_submission_readiness import readiness_verdict


def test_submission_audit_accepts_non_worse_guarded_profile() -> None:
    rows = [
        {
            "status": "ok",
            "guarded_relation_vs_conservative": "better",
            "aggressive_relation_vs_conservative": "better",
            "guarded_uses_aggressive": "1",
            "guarded_delta_tcount_vs_conservative": "5",
            "guarded_is_global_oracle": "0",
        },
        {
            "status": "ok",
            "guarded_relation_vs_conservative": "tie",
            "aggressive_relation_vs_conservative": "worse",
            "guarded_uses_aggressive": "0",
            "guarded_delta_tcount_vs_conservative": "0",
            "guarded_is_global_oracle": "1",
        },
    ]

    audit_row = audit_tensor_v3_profile(rows, Path("profile.csv"))

    assert audit_row["status"] == PASS
    assert "B/T/W=1/1/0" in audit_row["result"]


def test_submission_audit_guard_surface_requires_robust_non_worse_plateau() -> None:
    rows = [
        {
            "qasm_depth_gain": "0.0",
            "mixed_drop_fraction": "0.0",
            "better_vs_conservative": "1",
            "tie_vs_conservative": "1",
            "worse_vs_conservative": "1",
            "global_oracle_hits": "1",
            "max_global_oracle_regret": "0.1",
            "tcount_overhead_total": "4",
        },
        {
            "qasm_depth_gain": "0.1",
            "mixed_drop_fraction": "0.0",
            "better_vs_conservative": "1",
            "tie_vs_conservative": "2",
            "worse_vs_conservative": "0",
            "global_oracle_hits": "3",
            "max_global_oracle_regret": "0.0",
            "tcount_overhead_total": "5",
        },
    ]

    audit_row = audit_guard_surface(rows, Path("surface.csv"))

    assert audit_row["status"] == PASS
    assert "robust settings with qasm_gain>=0.10: 1" in audit_row["result"]


def test_submission_audit_loco_keeps_small_scope_as_warning() -> None:
    rows = [
        {
            "hit_true_best": "True",
            "primary_regret_vs_true_best": "0.0",
            "primary_gain_vs_tcount_best": "0.2",
            "num_test_candidates": "3",
        },
        {
            "hit_true_best": "False",
            "primary_regret_vs_true_best": "0.1",
            "primary_gain_vs_tcount_best": "0.7",
            "num_test_candidates": "2",
        },
        {
            "hit_true_best": "True",
            "primary_regret_vs_true_best": "0.0",
            "primary_gain_vs_tcount_best": "0.0",
            "num_test_candidates": "1",
        },
    ]

    audit_row = audit_alphaq_loco(rows, Path("loco.csv"))

    assert audit_row["status"] == WARN
    assert "singleton tests=1/3" in audit_row["result"]


def test_submission_audit_selection_manifest_rejects_external_columns() -> None:
    rows = [
        {
            "circuit_id": "toy",
            "candidate_id": "toy:combo0",
            "tensor_v3_selected": "1",
            "tensor_v3_mixed_excess_norm": "0.2",
            "primary_nc_depth_ratio": "0.4",
        }
    ]

    audit_row = audit_tensor_v3_selection_manifest(rows, Path("manifest.csv"))

    assert audit_row["status"] != PASS
    assert "forbidden_columns=1" in audit_row["result"]
    assert forbidden_selection_columns(list(rows[0])) == ["primary_nc_depth_ratio"]


def test_submission_audit_selection_manifest_accepts_pure_guarded_manifest() -> None:
    rows = [
        {
            "circuit_id": "toy",
            "candidate_id": "toy:combo0",
            "tensor_v3_selected": "1",
            "tensor_v3_profile": "guarded-aggressive",
            "guarded_qasm_depth_gain_threshold": "0.1",
            "tensor_v3_mixed_excess_norm": "0.2",
        }
    ]

    audit_row = audit_tensor_v3_selection_manifest(rows, Path("manifest.csv"))

    assert audit_row["status"] == PASS
    assert "forbidden_columns=0" in audit_row["result"]


def test_submission_readiness_verdict_reflects_warnings() -> None:
    rows = [
        {"status": PASS},
        {"status": WARN},
    ]

    assert readiness_verdict(rows) == "ready-with-caveats"
