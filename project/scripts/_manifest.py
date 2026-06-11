from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "results" / "reproducibility" / "bootstrap_manifest.json"
LOCK_PATH = MANIFEST_PATH.with_suffix(".lock")
VENDOR_METADATA_PATH = PROJECT_ROOT / "external" / ".vendor-metadata.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "vendor_sources": {},
        "environment": {},
        "commands": [],
    }
    if VENDOR_METADATA_PATH.exists():
        manifest["vendor_sources"] = json.loads(VENDOR_METADATA_PATH.read_text())
    return manifest


@contextmanager
def _manifest_lock() -> Iterator[None]:
    """Serialize manifest read-modify-write cycles across processes.

    Without this, parallel batteries interleave non-atomic writes and corrupt
    the manifest, which then crashes every subsequent writer at the end of an
    otherwise successful run.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            # A corrupted manifest must never poison new runs: preserve the
            # damaged file for forensics and restart from a clean state.
            backup = MANIFEST_PATH.with_name(
                f"bootstrap_manifest.corrupt-{os.getpid()}.json"
            )
            MANIFEST_PATH.replace(backup)
            return _default_manifest()
    return _default_manifest()


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _timestamp()
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    tmp_path = MANIFEST_PATH.with_name(f"{MANIFEST_PATH.name}.tmp-{os.getpid()}")
    tmp_path.write_text(payload)
    tmp_path.replace(MANIFEST_PATH)


def update_environment(environment: dict[str, Any]) -> None:
    with _manifest_lock():
        manifest = load_manifest()
        manifest.setdefault("environment", {}).update(environment)
        save_manifest(manifest)


def append_command(command: dict[str, Any]) -> None:
    with _manifest_lock():
        manifest = load_manifest()
        command = dict(command)
        command.setdefault("timestamp", _timestamp())
        manifest.setdefault("commands", []).append(command)
        save_manifest(manifest)
