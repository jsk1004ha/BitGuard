"""Lightweight BitGuard Modal status reader.

This file intentionally does not import the training image definition, so
checking bootstrap state does not rebuild the CUDA/PyTorch image.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "bitguard-status"
VOLUME_NAME = "bitguard-bnn"
VOLUME_MOUNT = "/bitguard"
SNAPSHOT_ROOT = "/bitguard/snapshot-v2"
DATA_ROOT = f"{SNAPSHOT_ROOT}/BitGuardData"
REPORT_PATH = f"{DATA_ROOT}/.bitguard/bootstrap-report.json"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _read_report() -> dict[str, object]:
    path = Path(REPORT_PATH)
    if not path.is_file():
        return {
            "status": "no_snapshot",
            "bootstrap_report": REPORT_PATH,
            "message": "No durable BitGuard bootstrap report exists yet.",
        }

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "unreadable_report",
            "bootstrap_report": REPORT_PATH,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not isinstance(value, dict):
        return {
            "status": "invalid_report",
            "bootstrap_report": REPORT_PATH,
        }

    keys = (
        "status",
        "last_completed_stage",
        "failed_stage",
        "error",
        "report_error",
        "lock_release_error",
        "next_stage",
        "dataset_statuses",
        "trained_runs",
        "prepared_datasets",
        "recovery_command",
    )
    return {
        "volume": VOLUME_NAME,
        "bootstrap_report": REPORT_PATH,
        **{key: value[key] for key in keys if key in value},
    }


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOLUME_MOUNT: volume.with_mount_options(read_only=True)},
    timeout=120,
)
def status() -> dict[str, object]:
    volume.reload()
    result = _read_report()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = status.remote()
    print(json.dumps(result, ensure_ascii=False, indent=2))
