from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, command: list[str]) -> dict[str, Any]:
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.time() - start
    result = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    print(f"[{name}] returncode={completed.returncode} elapsed={elapsed:.2f}s")
    if completed.stdout.strip():
        print(completed.stdout[-1200:])
    if completed.stderr.strip():
        print(completed.stderr[-1200:], file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducibility pipeline used by Draft 1."
    )
    parser.add_argument(
        "--run-formal",
        action="store_true",
        help="Run feynver formal verification before rebuilding Entrega 1 tables.",
    )
    parser.add_argument("--formal-timeout-sec", type=int, default=30)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-notebook", action="store_true")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "reports" / "draft1_pipeline_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps: list[dict[str, Any]] = []
    python = sys.executable

    steps.append(
        run_step("paper-reproduction", [python, "scripts/reproduce_paper_results.py"])
    )
    if args.run_formal:
        steps.append(
            run_step(
                "formal-verification",
                [
                    python,
                    "scripts/run_formal_verification.py",
                    "--scope",
                    "entrega1",
                    "--timeout-sec",
                    str(args.formal_timeout_sec),
                ],
            )
        )
    steps.append(
        run_step("entrega1-outputs", [python, "scripts/make_entrega1_outputs.py"])
    )
    if not args.skip_tests:
        steps.append(run_step("tests", [python, "-m", "pytest", "tests", "-q"]))
    if not args.skip_notebook:
        steps.append(
            run_step(
                "notebook",
                [
                    python,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "notebooks/entrega1_reproducao_alphatensor_quantum.ipynb",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=1200",
                ],
            )
        )

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_root": str(PROJECT_ROOT),
        "run_formal": args.run_formal,
        "formal_timeout_sec": args.formal_timeout_sec,
        "steps": steps,
        "artifacts": {
            "paper_summary": "results/reproducibility/paper/paper_summary.json",
            "paper_comparison": "results/reproducibility/paper/paper_benchmark_comparison.csv",
            "entrega1_metrics": "results/csv/entrega1_metrics.csv",
            "entrega1_formal_metrics": "results/csv/entrega1_metrics_formally_verified.csv",
            "notebook": "notebooks/entrega1_reproducao_alphatensor_quantum.ipynb",
        },
    }
    args.manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_path": str(args.manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
