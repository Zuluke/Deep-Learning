from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "results" / "reproducibility" / "bootstrap_manifest.json"
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


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return _default_manifest()


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _timestamp()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def update_environment(environment: dict[str, Any]) -> None:
    manifest = load_manifest()
    manifest.setdefault("environment", {}).update(environment)
    save_manifest(manifest)


def append_command(command: dict[str, Any]) -> None:
    manifest = load_manifest()
    command = dict(command)
    command.setdefault("timestamp", _timestamp())
    manifest.setdefault("commands", []).append(command)
    save_manifest(manifest)
