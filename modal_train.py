"""Run BitGuard's verified full-data pipeline on Modal with durable snapshots.

BitGuard's bootstrap intentionally uses POSIX hard links and no-clobber rename
operations as part of its crash-safety contract. Modal Volumes do not provide
all of those semantics, so the scientific pipeline runs on the Function's
ordinary ephemeral Linux filesystem. A persistent Modal Volume is used only as
an rsync snapshot/restore target.

Usage from the repository root::

    pip install "modal>=1.5,<1.6"
    modal setup
    modal run -d -n bitguard-bnn modal_train.py --gpu T4 --dataset all --accept-botiot-license

Re-running the same command restores the most recent durable snapshot and lets
BitGuard reuse completed bootstrap stages and compatible training checkpoints.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal

import modal


APP_NAME = "bitguard-bnn"
VOLUME_NAME = "bitguard-bnn"
VOLUME_MOUNT = "/bitguard"

SNAPSHOT_ROOT = "/bitguard/snapshot-v2"
PERSISTENT_DATA_ROOT = f"{SNAPSHOT_ROOT}/BitGuardData"
PERSISTENT_RUNS_ROOT = f"{SNAPSHOT_ROOT}/BitGuardRuns"

WORK_ROOT = "/work/bitguard"
DATA_ROOT = f"{WORK_ROOT}/BitGuardData"
RUNS_ROOT = f"{WORK_ROOT}/BitGuardRuns"
USER_SOURCES_ROOT = f"{WORK_ROOT}/UserSources"

REPOSITORY_ROOT = "/opt/BitGuard"
BOOTSTRAP = "/opt/BitGuard/bootstrap.sh"

SYNC_INTERVAL_SECONDS = 180
MAX_MODAL_RUNTIME_SECONDS = 24 * 60 * 60
EPHEMERAL_DISK_MIB = 128 * 1024

GpuName = Literal["T4", "L4", "A10", "A100", "L40S", "H100"]
DatasetName = Literal["all", "nbaiot", "botiot"]

_LOCAL_REPOSITORY = Path(__file__).resolve().parent

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "p7zip-full", "rsync")
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


def _load_report(data_root: str) -> dict[str, object] | None:
    report_path = Path(data_root) / ".bitguard" / "bootstrap-report.json"
    if not report_path.is_file():
        return None
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _report_summary(returncode: int, *, persistent: bool) -> dict[str, object]:
    data_root = PERSISTENT_DATA_ROOT if persistent else DATA_ROOT
    report = _load_report(data_root) or {}
    keys = (
        "status",
        "last_completed_stage",
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
        "working_data_root": DATA_ROOT,
        "working_runs_root": RUNS_ROOT,
        "snapshot_data_root": PERSISTENT_DATA_ROOT,
        "snapshot_runs_root": PERSISTENT_RUNS_ROOT,
        "bootstrap_report": f"{data_root}/.bitguard/bootstrap-report.json",
        **{key: report[key] for key in keys if key in report},
    }


def _rsync_directory(source: str, destination: str, *, delete: bool) -> None:
    """Copy a tree without preserving hard-link identity across filesystems."""

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        return
    command = [
        "rsync",
        "-rlt",
        "--partial",
        "--omit-dir-times",
    ]
    if delete:
        command.append("--delete")
    command.extend([f"{source_path}/", f"{destination_path}/"])
    subprocess.run(command, check=True)


def _restore_snapshot() -> None:
    """Restore the last committed Volume snapshot onto local Linux storage."""

    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(RUNS_ROOT).mkdir(parents=True, exist_ok=True)
    _rsync_directory(PERSISTENT_DATA_ROOT, DATA_ROOT, delete=False)
    _rsync_directory(PERSISTENT_RUNS_ROOT, RUNS_ROOT, delete=False)


def _sync_snapshot() -> None:
    """Mirror current local state to the persistent Volume and commit it."""

    Path(PERSISTENT_DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(PERSISTENT_RUNS_ROOT).mkdir(parents=True, exist_ok=True)
    _rsync_directory(DATA_ROOT, PERSISTENT_DATA_ROOT, delete=True)
    _rsync_directory(RUNS_ROOT, PERSISTENT_RUNS_ROOT, delete=True)
    volume.commit()


def _sync_periodically(stop: threading.Event) -> None:
    while not stop.wait(SYNC_INTERVAL_SECONDS):
        try:
            _sync_snapshot()
            print("[modal] committed durable BitGuard snapshot", flush=True)
        except Exception as exc:
            print(
                f"[modal] periodic snapshot failed: {exc}",
                file=sys.stderr,
                flush=True,
            )


def _resolve_volume_source(value: str) -> Path:
    supplied = Path(value)
    mount = Path(VOLUME_MOUNT).resolve()
    candidate = supplied if supplied.is_absolute() else mount / supplied
    resolved = candidate.resolve()
    try:
        resolved.relative_to(mount)
    except ValueError as exc:
        raise ValueError(
            "botiot_source must be stored inside the BitGuard Modal Volume"
        ) from exc
    if not resolved.exists():
        raise FileNotFoundError(
            f"BoT-IoT source does not exist in the Modal Volume: {resolved}"
        )
    return resolved


def _stage_volume_source(value: str) -> Path:
    """Copy an optional user source off the Volume before BitGuard touches it."""

    source = _resolve_volume_source(value)
    root = Path(USER_SOURCES_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / source.name
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return destination


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    cpu=8.0,
    memory=32768,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
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
    """Run BitGuard on local Linux storage with durable Volume snapshots."""

    if dataset not in {"all", "nbaiot", "botiot"}:
        raise ValueError("dataset must be one of: all, nbaiot, botiot")
    if dataset in {"all", "botiot"} and not accept_botiot_license:
        raise ValueError(
            "BoT-IoT requires explicit acceptance after reviewing the UNSW "
            "academic-use terms; rerun with --accept-botiot-license"
        )

    volume.reload()
    _restore_snapshot()

    local_botiot_source = (
        str(_stage_volume_source(botiot_source)) if botiot_source else ""
    )

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
    if local_botiot_source:
        command.extend(["--botiot-source", local_botiot_source])

    print("[modal] BitGuard command:", " ".join(command), flush=True)
    print(
        f"[modal] working filesystem: {WORK_ROOT}; durable snapshot: {SNAPSHOT_ROOT}",
        flush=True,
    )

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    stop = threading.Event()
    synchronizer = threading.Thread(
        target=_sync_periodically,
        args=(stop,),
        name="bitguard-snapshot-sync",
        daemon=True,
    )
    synchronizer.start()
    completed: subprocess.CompletedProcess[bytes] | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        )
    finally:
        stop.set()
        synchronizer.join(timeout=SYNC_INTERVAL_SECONDS + 30)
        _sync_snapshot()

    assert completed is not None
    summary = _report_summary(completed.returncode, persistent=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if completed.returncode != 0 or summary.get("status") == "failed":
        raise RuntimeError(
            "BitGuard bootstrap did not complete successfully. Its durable snapshot "
            "was committed. Inspect recovery_command, then rerun with the indicated "
            "--restart-stage only when the report requires it."
        )
    return summary


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOLUME_MOUNT: volume.with_mount_options(read_only=True)},
    timeout=120,
)
def read_status() -> dict[str, object]:
    """Read the latest durable bootstrap status without allocating a GPU."""

    volume.reload()
    return _report_summary(0, persistent=True)


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
