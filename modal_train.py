"""Run BitGuard's verified full-data pipeline on Modal with durable storage.

Usage from the repository root::

    pip install "modal>=1.5,<1.6"
    modal setup
    modal run modal_train.py --gpu T4 --dataset all --accept-botiot-license

All research data, prepared Parquet shards, bootstrap state, checkpoints, and
training outputs live on the ``bitguard-bnn`` Modal Volume. Re-running the same
command reuses the repository's normal bootstrap/checkpoint recovery contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal

import modal


APP_NAME = "bitguard-bnn"
VOLUME_NAME = "bitguard-bnn"
VOLUME_MOUNT = "/bitguard"
DATA_ROOT = "/bitguard/BitGuardData"
RUNS_ROOT = "/bitguard/BitGuardRuns"
REPOSITORY_ROOT = "/opt/BitGuard"
BOOTSTRAP = "/opt/BitGuard/bootstrap.sh"
COMMIT_INTERVAL_SECONDS = 60
MAX_MODAL_RUNTIME_SECONDS = 24 * 60 * 60

GpuName = Literal["T4", "L4", "A10", "A100", "L40S", "H100"]
DatasetName = Literal["all", "nbaiot", "botiot"]

_LOCAL_REPOSITORY = Path(__file__).resolve().parent

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "p7zip-full")
    .add_local_dir(
        str(_LOCAL_REPOSITORY),
        REPOSITORY_ROOT,
        copy=True,
        ignore=[
            ".git/**",
            ".venv/**",
            "data/**",
            "runs/**",
            "__pycache__/**",
            "*.pyc",
            ".pytest_cache/**",
        ],
    )
    .run_commands(
        f"chmod +x {BOOTSTRAP}",
        f"python -m venv {REPOSITORY_ROOT}/.venv",
        f"{REPOSITORY_ROOT}/.venv/bin/python -m pip install --upgrade pip setuptools wheel",
        (
            f"{REPOSITORY_ROOT}/.venv/bin/python -m pip install "
            f"--requirement {REPOSITORY_ROOT}/requirements/locks/torch-cu128.txt"
        ),
        (
            f"{REPOSITORY_ROOT}/.venv/bin/python -m pip install --no-build-isolation "
            f"--editable {REPOSITORY_ROOT} --no-deps"
        ),
        (
            f"{REPOSITORY_ROOT}/.venv/bin/python -m pip install "
            f"--requirement {REPOSITORY_ROOT}/requirements/locks/full-base.txt"
        ),
    )
)

app = modal.App(APP_NAME)


def _load_report() -> dict[str, object] | None:
    report_path = Path(DATA_ROOT) / ".bitguard" / "bootstrap-report.json"
    if not report_path.is_file():
        return None
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _report_summary(returncode: int) -> dict[str, object]:
    report = _load_report() or {}
    keys = (
        "status",
        "failed_stage",
        "next_stage",
        "dataset_statuses",
        "trained_runs",
        "prepared_datasets",
        "reports",
        "recovery_command",
    )
    return {
        "returncode": int(returncode),
        "volume": VOLUME_NAME,
        "data_root": DATA_ROOT,
        "runs_root": RUNS_ROOT,
        "bootstrap_report": f"{DATA_ROOT}/.bitguard/bootstrap-report.json",
        **{key: report[key] for key in keys if key in report},
    }


def _resolve_volume_source(value: str) -> Path:
    supplied = Path(value)
    mount = Path(VOLUME_MOUNT).resolve()
    candidate = supplied if supplied.is_absolute() else mount / supplied
    resolved = candidate.resolve()
    try:
        resolved.relative_to(mount)
    except ValueError as exc:
        raise ValueError("botiot_source must be stored inside the BitGuard Modal Volume") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"BoT-IoT source does not exist in the Modal Volume: {resolved}")
    return resolved


def _commit_volume_periodically(stop: threading.Event) -> None:
    while not stop.wait(COMMIT_INTERVAL_SECONDS):
        try:
            volume.commit()
            print("[modal] committed BitGuard Volume checkpoint", flush=True)
        except Exception as exc:  # keep training alive if a diagnostic commit fails
            print(f"[modal] periodic Volume commit failed: {exc}", file=sys.stderr, flush=True)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    cpu=8.0,
    memory=32768,
    timeout=MAX_MODAL_RUNTIME_SECONDS,
    max_containers=1,
)
def run_bitguard(
    dataset: str = "all",
    accept_botiot_license: bool = False,
    prepare_only: bool = False,
    restart_stage: str = "",
    botiot_source: str = "",
) -> dict[str, object]:
    """Run BitGuard's existing bootstrap pipeline on one durable Modal Volume."""

    if dataset not in {"all", "nbaiot", "botiot"}:
        raise ValueError("dataset must be one of: all, nbaiot, botiot")
    if dataset in {"all", "botiot"} and not accept_botiot_license:
        raise ValueError(
            "BoT-IoT requires explicit acceptance after reviewing the UNSW academic-use terms; "
            "rerun with --accept-botiot-license"
        )

    # Pull in the newest committed Volume state before opening any files.
    volume.reload()
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(RUNS_ROOT).mkdir(parents=True, exist_ok=True)

    command = [
        BOOTSTRAP,
        "--full",
        "--dataset",
        dataset,
        "--compute",
        "cu128",
        "--data-root",
        DATA_ROOT,
        "--runs-root",
        RUNS_ROOT,
        "--no-install-system-tools",
    ]
    if dataset in {"all", "botiot"}:
        command.append("--accept-botiot-academic-license")
    if prepare_only:
        command.append("--prepare-only")
    if restart_stage:
        command.extend(["--restart-stage", restart_stage])
    if botiot_source:
        command.extend(["--botiot-source", str(_resolve_volume_source(botiot_source))])

    print("[modal] BitGuard command:", " ".join(command), flush=True)
    print(f"[modal] persistent Volume: {VOLUME_NAME} mounted at {VOLUME_MOUNT}", flush=True)

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    stop = threading.Event()
    committer = threading.Thread(
        target=_commit_volume_periodically,
        args=(stop,),
        name="bitguard-volume-committer",
        daemon=True,
    )
    committer.start()
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        )
    finally:
        stop.set()
        committer.join(timeout=COMMIT_INTERVAL_SECONDS + 5)
        # Bootstrap/checkpoint writes are atomic. Commit the final closed-file state.
        volume.commit()

    summary = _report_summary(completed.returncode)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "BitGuard bootstrap did not complete successfully. Durable state was committed; "
            "inspect the bootstrap report and rerun the same Modal command to resume."
        )
    return summary


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOLUME_MOUNT: volume.with_mount_options(read_only=True)},
    timeout=120,
)
def read_status() -> dict[str, object]:
    """Read the durable bootstrap status without allocating a GPU."""

    volume.reload()
    return _report_summary(0)


@app.local_entrypoint()
def main(
    gpu: GpuName = "T4",
    dataset: DatasetName = "all",
    accept_botiot_license: bool = False,
    prepare_only: bool = False,
    restart_stage: str = "",
    botiot_source: str = "",
) -> None:
    """Start or resume the full BitGuard job with a selectable Modal GPU."""

    result = run_bitguard.with_options(gpu=gpu).remote(
        dataset=dataset,
        accept_botiot_license=accept_botiot_license,
        prepare_only=prepare_only,
        restart_stage=restart_stage,
        botiot_source=botiot_source,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
