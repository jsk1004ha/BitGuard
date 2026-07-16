from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .metrics import StreamingClassificationMetrics

_EVALUATION_TRANSACTION_VERSION = 4
_WORK_OWNER_FILE = ".bitguard-evaluation-owner.json"
_WORK_OWNER_INITIALIZING_FILE = ".bitguard-evaluation-owner.initializing"


def _path_entry_exists(path: Path) -> bool:
    """Return whether a directory entry exists, including dangling symlinks."""
    return os.path.lexists(path)


def _absolute_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.parent.resolve() / candidate.name


def _entry_instance(path: Path) -> tuple[int, int] | None:
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    return int(observed.st_dev), int(observed.st_ino)


def _partial_path(path: Path, transaction_id: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex if transaction_id is None else transaction_id
    return path.with_name(f".{path.name}.{token}.partial")


def _sync_file(path: Path) -> None:
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _encoded_json(payload)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _identity_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        size = int(expected["size"])
        digest = str(expected["sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    if size < 0 or len(digest) != 64 or path.stat().st_size != size:
        return False
    return _file_identity(path) == {"size": size, "sha256": digest}


def _transaction_path(metrics_path: Path) -> Path:
    return metrics_path.with_name(f".{metrics_path.name}.evaluation-transaction.json")


def _transaction_lock_path(journal: Path) -> Path:
    return journal.with_name(f".{journal.name}.lock")


def _acquire_transaction_lock(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
            )
        return handle
    except (OSError, ImportError) as error:
        handle.close()
        raise RuntimeError(
            f"another evaluation process owns the publish transaction: {path}"
        ) from error


def _release_transaction_lock(handle: BinaryIO) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(), fcntl.LOCK_UN  # type: ignore[attr-defined]
            )
    finally:
        handle.close()


def _write_transaction(path: Path, payload: Mapping[str, Any]) -> None:
    if _path_entry_exists(path):
        raise FileExistsError(f"evaluation publish transaction already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        _write_json(temporary, payload)
        if _path_entry_exists(path):
            raise FileExistsError(
                f"evaluation publish transaction appeared concurrently: {path}"
            )
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_transaction(path: Path, payload: Mapping[str, Any]) -> None:
    if not _path_entry_exists(path):
        raise FileNotFoundError(f"evaluation publish transaction disappeared: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        _write_json(temporary, payload)
        if not _path_entry_exists(path):
            raise FileNotFoundError(
                f"evaluation publish transaction disappeared concurrently: {path}"
            )
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _after_publish_boundary(_index: int, _path: Path) -> None:
    """Test seam representing a process loss immediately after publication."""


def _after_build_boundary(_index: int, _path: Path) -> None:
    """Test seam representing a process loss while artifacts are still private."""


def _before_atomic_publish(_index: int, _path: Path) -> None:
    """Test seam after the fast precheck and before atomic create-if-absent."""


def _after_publish_link_boundary(_index: int, _path: Path) -> None:
    """Test seam after durable linking but before owned-partial removal."""


def _after_work_directory_created(_path: Path) -> None:
    """Test seam representing a process loss immediately after work mkdir."""


def _during_owner_marker_write(_path: Path) -> None:
    """Test seam representing a process loss during owner marker initialization."""


def _write_owner_marker(
    owner_marker: Path,
    initializing_marker: Path,
    payload: Mapping[str, Any],
) -> None:
    encoded = _encoded_json(payload)
    split = max(1, len(encoded) // 2)
    with initializing_marker.open("xb") as handle:
        handle.write(encoded[:split])
        handle.flush()
        os.fsync(handle.fileno())
        _sync_directory(initializing_marker.parent)
        _during_owner_marker_write(initializing_marker)
        handle.write(encoded[split:])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(initializing_marker, owner_marker)
    _sync_directory(owner_marker.parent)


def _artifact_records(
    publish_order: Sequence[Path], partials: Mapping[Path, Path]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for final in publish_order:
        partial = partials[final]
        instance = _entry_instance(partial)
        if instance is None:
            raise RuntimeError(
                f"evaluation partial disappeared before publication: {partial}"
            )
        records.append(
            {
                "final": str(final),
                "partial": str(partial),
                "identity": _file_identity(partial),
                "instance": list(instance),
            }
        )
    return records


def _link_no_replace(partial: Path, final: Path) -> None:
    """Atomically create *final* without ever replacing an existing entry."""
    try:
        os.link(partial, final, follow_symlinks=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"foreign artifact appeared during atomic publication: {final}"
        ) from error
    _sync_directory(final.parent)


def _validate_transaction(
    payload: object,
    *,
    expected_paths: Sequence[Path],
    test_contract: str,
    request_contract: Mapping[str, Any],
    temporary_directory: Path,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RuntimeError("evaluation publish transaction is not a JSON object")
    if payload.get("format_version") != _EVALUATION_TRANSACTION_VERSION:
        raise RuntimeError("evaluation publish transaction version is unsupported")
    state = payload.get("state")
    if state not in {"building", "publishing", "committed"}:
        raise RuntimeError("evaluation publish transaction state is invalid")
    if payload.get("test_contract") != test_contract:
        raise RuntimeError(
            "evaluation publish transaction test_contract mismatch; refusing recovery"
        )
    if payload.get("request_contract") != dict(request_contract):
        raise RuntimeError(
            "evaluation publish transaction request contract mismatch; refusing recovery"
        )
    transaction_id = payload.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise RuntimeError("evaluation publish transaction id is invalid")
    owned_work = payload.get("owned_work")
    if not isinstance(owned_work, dict) or set(owned_work) != {
        "path",
        "owner_marker",
        "owner_initializing_marker",
        "owner_payload",
        "temporary_parent",
        "parent_created",
    }:
        raise RuntimeError("evaluation publish transaction work ownership is invalid")
    expected_parent = temporary_directory.resolve()
    expected_work = expected_parent / f"evaluation-{transaction_id}"
    expected_owner = expected_work / _WORK_OWNER_FILE
    expected_initializing_owner = expected_work / _WORK_OWNER_INITIALIZING_FILE
    if (
        Path(str(owned_work["temporary_parent"])) != expected_parent
        or Path(str(owned_work["path"])) != expected_work
        or Path(str(owned_work["owner_marker"])) != expected_owner
        or Path(str(owned_work["owner_initializing_marker"]))
        != expected_initializing_owner
        or not isinstance(owned_work["parent_created"], bool)
    ):
        raise RuntimeError(
            "evaluation publish transaction work path is outside its expected owner root"
        )
    expected_owner_payload = {
        "format_version": 1,
        "transaction_id": transaction_id,
        "test_contract": test_contract,
    }
    if owned_work["owner_payload"] != expected_owner_payload:
        raise RuntimeError("evaluation publish transaction work owner payload is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_paths):
        raise RuntimeError("evaluation publish transaction artifact list is invalid")
    validated: list[dict[str, Any]] = []
    for expected_final, raw in zip(expected_paths, artifacts, strict=True):
        if not isinstance(raw, dict) or set(raw) != {
            "final",
            "partial",
            "identity",
            "instance",
        }:
            raise RuntimeError("evaluation publish transaction artifact record is invalid")
        final = Path(str(raw["final"]))
        partial = Path(str(raw["partial"]))
        if final != expected_final:
            raise RuntimeError(
                "evaluation publish transaction targets different final artifacts"
            )
        expected_partial = final.with_name(
            f".{final.name}.{transaction_id}.partial"
        )
        if partial != expected_partial:
            raise RuntimeError("evaluation publish transaction partial path is invalid")
        identity = raw["identity"]
        instance = raw["instance"]
        if state == "building":
            if identity is not None or instance is not None:
                raise RuntimeError(
                    "building evaluation transaction must not claim artifact identities or instances"
                )
        elif not isinstance(identity, dict) or set(identity) != {"size", "sha256"}:
            raise RuntimeError("evaluation publish transaction identity is invalid")
        elif (
            not isinstance(instance, list)
            or len(instance) != 2
            or any(not isinstance(value, int) for value in instance)
        ):
            raise RuntimeError("evaluation publish transaction instance is invalid")
        validated.append(
            {
                "final": final,
                "partial": partial,
                "identity": identity,
                "instance": instance,
            }
        )
    return validated


def _cleanup_owned_work(
    payload: Mapping[str, Any], *, temporary_directory: Path
) -> None:
    owned_work = payload["owned_work"]
    if not isinstance(owned_work, dict):
        raise RuntimeError("evaluation work ownership record is invalid")
    work = Path(str(owned_work["path"]))
    owner_marker = Path(str(owned_work["owner_marker"]))
    initializing_marker = Path(str(owned_work["owner_initializing_marker"]))
    expected_parent = temporary_directory.resolve()
    if not _path_entry_exists(work):
        if bool(owned_work["parent_created"]):
            try:
                expected_parent.rmdir()
            except OSError:
                # Missing, non-empty, or replaced parents are not ours to remove.
                pass
        return
    if work.is_symlink() or not work.is_dir() or work.parent != expected_parent:
        raise RuntimeError(
            f"foreign work path blocks evaluation cleanup: {work}"
        )
    if owner_marker.parent != work or owner_marker.name != _WORK_OWNER_FILE:
        raise RuntimeError("evaluation work ownership marker path is invalid")
    if (
        initializing_marker.parent != work
        or initializing_marker.name != _WORK_OWNER_INITIALIZING_FILE
    ):
        raise RuntimeError("evaluation initializing owner marker path is invalid")
    if _path_entry_exists(owner_marker):
        if (
            owner_marker.is_symlink()
            or not owner_marker.is_file()
            or _path_entry_exists(initializing_marker)
        ):
            raise RuntimeError(
                f"foreign or unverifiable work directory blocks evaluation cleanup: {work}"
            )
        try:
            observed_owner = json.loads(owner_marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"evaluation work ownership marker cannot be validated: {owner_marker}"
            ) from error
        if observed_owner != owned_work["owner_payload"]:
            raise RuntimeError(
                f"foreign work ownership marker blocks evaluation cleanup: {work}"
            )
        shutil.rmtree(work)
    else:
        # The durable building journal and transaction lock prove this exact
        # UUID work path is ours, but before atomic marker publication only an
        # empty directory or the journal-declared initializing file is safe to
        # remove. Any other child is preserved as foreign and blocks cleanup.
        entries = list(work.iterdir())
        unexpected = [path for path in entries if path != initializing_marker]
        if unexpected:
            raise RuntimeError(
                f"foreign path blocks evaluation ownership initialization cleanup: {unexpected[0]}"
            )
        if entries:
            if initializing_marker.is_symlink() or not initializing_marker.is_file():
                raise RuntimeError(
                    "foreign initializing marker blocks evaluation cleanup: "
                    f"{initializing_marker}"
                )
            initializing_marker.unlink()
        work.rmdir()
    if bool(owned_work["parent_created"]):
        try:
            expected_parent.rmdir()
        except OSError:
            # A concurrent/foreign sibling is never removed.
            pass


def _recover_transaction(
    journal: Path,
    *,
    publish_order: Sequence[Path],
    test_contract: str,
    request_contract: Mapping[str, Any],
    prediction_path: Path,
    metrics_path: Path,
    plot_manifest_path: Path,
    temporary_directory: Path,
) -> dict[str, Any] | None:
    if not _path_entry_exists(journal):
        return None
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"evaluation publish transaction cannot be validated: {journal}"
        ) from error
    records = _validate_transaction(
        payload,
        expected_paths=publish_order,
        test_contract=test_contract,
        request_contract=request_contract,
        temporary_directory=temporary_directory,
    )
    if payload.get("state") == "building":
        # A building transaction has never published a final path. The external
        # transaction lock guarantees that these exact transaction-id paths do
        # not belong to a live evaluator while recovery removes them.
        foreign_finals = [
            record["final"]
            for record in records
            if _path_entry_exists(record["final"])
        ]
        if foreign_finals:
            raise RuntimeError(
                "foreign artifact blocks building evaluation recovery: "
                f"{foreign_finals[0]}"
            )
        for record in records:
            partial = record["partial"]
            if not _path_entry_exists(partial):
                continue
            if partial.is_symlink() or not partial.is_file():
                raise RuntimeError(
                    f"foreign partial blocks building evaluation recovery: {partial}"
                )
        _cleanup_owned_work(payload, temporary_directory=temporary_directory)
        for record in records:
            record["partial"].unlink(missing_ok=True)
        journal.unlink()
        _sync_directory(journal.parent)
        return None

    # Validate every existing name before changing anything. A path whose bytes
    # differ from the recorded identity is foreign and must never be removed.
    for record in records:
        final = record["final"]
        partial = record["partial"]
        identity = record["identity"]
        instance = tuple(record["instance"])
        if _path_entry_exists(final) and (
            _entry_instance(final) != instance
            or not _identity_matches(final, identity)
        ):
            raise RuntimeError(
                f"foreign artifact or identity mismatch blocks evaluation recovery: {final}"
            )
        if _path_entry_exists(partial) and (
            _entry_instance(partial) != instance
            or not _identity_matches(partial, identity)
        ):
            raise RuntimeError(
                f"foreign partial or identity mismatch blocks evaluation recovery: {partial}"
            )
        if not _path_entry_exists(final) and not _path_entry_exists(partial):
            raise RuntimeError(
                f"evaluation transaction is missing both final and partial artifact: {final}"
            )
    for record in records:
        final = record["final"]
        partial = record["partial"]
        identity = record["identity"]
        instance = tuple(record["instance"])
        if _path_entry_exists(final):
            if _path_entry_exists(partial):
                partial.unlink()
            continue
        _link_no_replace(partial, final)
        if (
            _entry_instance(final) != instance
            or not _identity_matches(final, identity)
        ):
            raise RuntimeError(
                f"published artifact identity changed during recovery: {final}"
            )
        if (
            _entry_instance(partial) != instance
            or not _identity_matches(partial, identity)
        ):
            raise RuntimeError(
                f"partial artifact identity changed during recovery: {partial}"
            )
        partial.unlink()
        _sync_directory(final.parent)
    if payload.get("state") != "committed":
        committed = dict(payload)
        committed["state"] = "committed"
        _replace_transaction(journal, committed)
        payload = committed
    _cleanup_owned_work(payload, temporary_directory=temporary_directory)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    plot_manifest = json.loads(plot_manifest_path.read_text(encoding="utf-8"))
    return {
        "metrics": metrics,
        "plot_manifest": plot_manifest,
        "prediction_path": str(prediction_path),
        "rows": int(plot_manifest["numeric_rows"]),
    }


def _normalized_batch(
    batch: Mapping[str, Any], probability_labels: Sequence[str], storage_start: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "row_uid",
        "true_label",
        "predicted_label",
        "probabilities",
        "exit_stage",
    }
    generated = {
        "storage_position",
        "row_uid",
        "true_label",
        "predicted_label",
        "probabilities",
        "__replay_position",
        *(f"prob_{label}" for label in probability_labels),
    }
    normalized: dict[str, object] = {}
    for raw_name in batch:
        name = str(raw_name)
        if name in normalized:
            raise ValueError(
                f"prediction batch field collision after string normalization: {name!r}"
            )
        normalized[name] = raw_name
        if raw_name not in required and (
            name in generated or name.startswith("prob_")
        ):
            raise ValueError(f"prediction batch field {name!r} is reserved")
    missing = required - set(batch)
    if missing:
        raise ValueError(f"prediction batch missing fields: {sorted(missing)}")
    true_labels = np.asarray(batch["true_label"], dtype=str)
    predicted_labels = np.asarray(batch["predicted_label"], dtype=str)
    row_uids = np.asarray(batch["row_uid"], dtype=str)
    probabilities = np.asarray(batch["probabilities"], dtype=np.float32)
    exit_stage = np.asarray(batch["exit_stage"])
    row_count = len(true_labels)
    if predicted_labels.shape != (row_count,) or row_uids.shape != (row_count,):
        raise ValueError("prediction batch label and UID fields must align")
    if probabilities.shape != (row_count, len(probability_labels)):
        raise ValueError("prediction batch probabilities have an invalid shape")
    if exit_stage.shape != (row_count,) or not np.issubdtype(
        exit_stage.dtype, np.integer
    ):
        raise ValueError("prediction batch exit_stage must be a 1-D integer array")
    if np.any((exit_stage < 0) | (exit_stage > 2)):
        raise ValueError("prediction batch exit_stage values must be in {0, 1, 2}")
    if row_count == 0:
        return {}, true_labels, predicted_labels, probabilities, row_uids

    columns: dict[str, Any] = {
        "storage_position": np.arange(
            storage_start, storage_start + row_count, dtype=np.int64
        ),
        "row_uid": row_uids,
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "exit_stage": exit_stage.astype(np.int16, copy=False),
    }
    for index, label in enumerate(probability_labels):
        # This float32 array is also passed to the metric accumulator below, so
        # numeric reports describe exactly the values persisted for deployment.
        columns[f"prob_{label}"] = probabilities[:, index]
    for name, raw_value in batch.items():
        if name in required:
            continue
        value = np.asarray(raw_value)
        if value.shape != (row_count,):
            raise ValueError(f"prediction batch field {name!r} must align with rows")
        columns[str(name)] = value
    return columns, true_labels, predicted_labels, probabilities, row_uids


def _table(columns: Mapping[str, Any], schema: pa.Schema | None = None) -> pa.Table:
    table = pa.table(dict(columns))
    if schema is not None:
        table = table.cast(schema, safe=True)
    return table


def evaluate_prediction_batches(
    batch_factory: Callable[[], Iterable[Mapping[str, Any]]],
    *,
    probability_labels: Sequence[str],
    high_risk_labels: Sequence[str],
    test_contract: str,
    prediction_path: str | Path,
    metrics_path: str | Path,
    plot_sample_path: str | Path,
    plot_manifest_path: str | Path,
    temporary_directory: str | Path,
    operating_thresholds: Mapping[float, float] | None = None,
    plot_sample_rows: int = 50_000,
    plot_sample_seed: int | str = 2309,
    score_run_rows: int = 131_072,
) -> dict[str, Any]:
    """Stream already-routed prediction batches into exact metrics/artifacts.

    The callback boundary deliberately accepts model-independent batches so the
    full-run orchestrator can release inference objects between phases. Task 4
    owns temporal cascade ordering and scatters routed results back to source
    positions; this writer consumes those source-ordered routed batches once.
    """

    if plot_sample_rows < 0:
        raise ValueError("plot_sample_rows must be non-negative")
    if not isinstance(test_contract, str) or not test_contract.strip():
        raise ValueError("test_contract must be non-empty")
    labels = [str(label) for label in probability_labels]
    high_risk_values = [str(label) for label in high_risk_labels]
    prediction_final = _absolute_output_path(prediction_path)
    metrics_final = _absolute_output_path(metrics_path)
    sample_final = _absolute_output_path(plot_sample_path)
    plot_manifest_final = _absolute_output_path(plot_manifest_path)
    final_paths = [
        prediction_final,
        metrics_final,
        sample_final,
        plot_manifest_final,
    ]
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("evaluation artifact paths must be distinct")
    publish_order = [
        prediction_final,
        sample_final,
        plot_manifest_final,
        metrics_final,
    ]
    request_contract = {
        "probability_labels": labels,
        "high_risk_labels": sorted(high_risk_values),
        "operating_thresholds": [
            [float(target), float(threshold)]
            for target, threshold in sorted((operating_thresholds or {}).items())
        ],
        "plot_sample_rows": int(plot_sample_rows),
        "plot_sample_seed": str(plot_sample_seed),
    }
    journal = _transaction_path(metrics_final)
    temp_parent = Path(temporary_directory).resolve()
    lock_path = _transaction_lock_path(journal)
    if _path_entry_exists(journal):
        recovery_lock = _acquire_transaction_lock(lock_path)
        try:
            recovered = _recover_transaction(
                journal,
                publish_order=publish_order,
                test_contract=test_contract,
                request_contract=request_contract,
                prediction_path=prediction_final,
                metrics_path=metrics_final,
                plot_manifest_path=plot_manifest_final,
                temporary_directory=temp_parent,
            )
        finally:
            _release_transaction_lock(recovery_lock)
        if recovered is not None:
            return recovered
    existing = [str(path) for path in final_paths if _path_entry_exists(path)]
    if existing:
        raise FileExistsError(f"refusing to replace evaluation artifacts: {existing}")

    # Validate the first material batch before creating a writer, UID database,
    # temporary directory, or partial artifact. Every later batch follows the
    # same validate-then-insert/write ordering.
    storage_position = 0
    batch_iterator = iter(batch_factory())
    first_normalized: tuple[
        dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None
    for first_batch in batch_iterator:
        candidate = _normalized_batch(first_batch, labels, storage_position)
        if len(candidate[1]):
            first_normalized = candidate
            break
    if first_normalized is None:
        raise ValueError("prediction evaluation requires at least one row")

    transaction_lock = _acquire_transaction_lock(lock_path)
    try:
        # The first batch was evaluated without the lock to preserve the
        # zero-write validation contract. Re-check state after acquiring it in
        # case another process won the race during that validation.
        recovered = _recover_transaction(
            journal,
            publish_order=publish_order,
            test_contract=test_contract,
            request_contract=request_contract,
            prediction_path=prediction_final,
            metrics_path=metrics_final,
            plot_manifest_path=plot_manifest_final,
            temporary_directory=temp_parent,
        )
        if recovered is not None:
            return recovered
        existing = [str(path) for path in final_paths if _path_entry_exists(path)]
        if existing:
            raise FileExistsError(
                f"refusing to replace evaluation artifacts: {existing}"
            )

        for final in final_paths:
            final.parent.mkdir(parents=True, exist_ok=True)
        transaction_id = uuid.uuid4().hex
        partials = {
            path: _partial_path(path, transaction_id) for path in final_paths
        }
        parent_created = not _path_entry_exists(temp_parent)
        work = temp_parent / f"evaluation-{transaction_id}"
        owner_payload = {
            "format_version": 1,
            "transaction_id": transaction_id,
            "test_contract": test_contract,
        }
        owner_marker = work / _WORK_OWNER_FILE
        initializing_owner_marker = work / _WORK_OWNER_INITIALIZING_FILE
        transaction: dict[str, Any] = {
            "format_version": _EVALUATION_TRANSACTION_VERSION,
            "state": "building",
            "transaction_id": transaction_id,
            "test_contract": test_contract,
            "request_contract": request_contract,
            "owned_work": {
                "path": str(work),
                "owner_marker": str(owner_marker),
                "owner_initializing_marker": str(initializing_owner_marker),
                "owner_payload": owner_payload,
                "temporary_parent": str(temp_parent),
                "parent_created": parent_created,
            },
            "artifacts": [
                {
                    "final": str(final),
                    "partial": str(partials[final]),
                    "identity": None,
                    "instance": None,
                }
                for final in publish_order
            ],
        }
        writer: pq.ParquetWriter | None = None
        accumulator: StreamingClassificationMetrics | None = None
        uid_connection: sqlite3.Connection | None = None
        schema: pa.Schema | None = None
        retained: list[tuple[str, str, int, dict[str, Any]]] = []
        metrics: dict[str, Any] | None = None
        plot_manifest: dict[str, Any] | None = None
        journal_owned = False
        discard_building = False
        try:
            # Persist ownership before creating any private work or partial.
            # A process loss from this point onward is therefore recoverable.
            _write_transaction(journal, transaction)
            journal_owned = True
            temp_parent.mkdir(parents=True, exist_ok=True)
            work.mkdir(exist_ok=False)
            _sync_directory(temp_parent)
            _after_work_directory_created(work)
            _write_owner_marker(
                owner_marker,
                initializing_owner_marker,
                owner_payload,
            )
            accumulator = StreamingClassificationMetrics(
                probability_labels=labels,
                high_risk_labels=high_risk_values,
                temporary_directory=work,
                score_run_rows=score_run_rows,
            )
            uid_connection = sqlite3.connect(work / "row-uids.sqlite")
            uid_connection.execute("CREATE TABLE row_uid (value TEXT PRIMARY KEY)")

            def normalized_batches():
                yield first_normalized
                for batch in batch_iterator:
                    yield _normalized_batch(batch, labels, storage_position)

            build_index = 0
            for normalized in normalized_batches():
                columns, y_true, y_pred, probabilities, row_uids = normalized
                if not len(y_true):
                    continue
                try:
                    uid_connection.executemany(
                        "INSERT INTO row_uid VALUES (?)",
                        ((str(value),) for value in row_uids),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        "prediction row_uid values must be globally unique"
                    ) from error
                table = _table(columns, schema)
                if writer is None:
                    metadata = {
                        b"bitguard_artifact_format": b"prediction_parquet",
                        b"bitguard_artifact_version": b"1",
                        b"bitguard_probability_labels": json.dumps(
                            labels, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8"),
                        b"bitguard_test_contract": test_contract.encode("utf-8"),
                    }
                    table = table.replace_schema_metadata(metadata)
                    schema = table.schema
                    writer = pq.ParquetWriter(
                        partials[prediction_final], schema, compression="zstd"
                    )
                writer.write_table(table, row_group_size=len(table))
                accumulator.update(y_true, y_pred, probabilities, row_uids)

                if plot_sample_rows:
                    candidates = list(retained)
                    for offset in range(len(y_true)):
                        row = {
                            name: table[name][offset].as_py()
                            for name in table.column_names
                        }
                        uid = str(row["row_uid"])
                        priority = hashlib.sha256(
                            f"{plot_sample_seed}\0{uid}".encode("utf-8")
                        ).hexdigest()
                        candidates.append(
                            (priority, uid, storage_position + offset, row)
                        )
                    retained = heapq.nsmallest(
                        plot_sample_rows,
                        candidates,
                        key=lambda item: (item[0], item[1], item[2]),
                    )
                storage_position += len(y_true)
                _after_build_boundary(build_index, partials[prediction_final])
                build_index += 1
            if writer is None or schema is None or storage_position == 0:
                raise ValueError("prediction evaluation requires at least one row")
            writer.close()
            writer = None
            _sync_file(partials[prediction_final])

            metrics = accumulator.finalize(operating_thresholds)
            _write_json(partials[metrics_final], metrics)

            sample_rows = [item[3] for item in retained]
            sample_table = pa.Table.from_pylist(sample_rows, schema=schema)
            pq.write_table(
                sample_table,
                partials[sample_final],
                compression="zstd",
            )
            _sync_file(partials[sample_final])
            plot_manifest = {
                "numeric_metrics_scope": "full_test",
                "numeric_rows": storage_position,
                "plot_rows_scope": "deterministic_sample",
                "plot_rows": len(sample_rows),
                "plot_sample_rows_limit": int(plot_sample_rows),
                "plot_sample_seed": str(plot_sample_seed),
                "plot_sampling_algorithm": "sha256(seed\\0row_uid)-smallest-v1",
                "test_contract": test_contract,
            }
            _write_json(partials[plot_manifest_final], plot_manifest)

            artifact_records = _artifact_records(publish_order, partials)
            transaction["artifacts"] = artifact_records
            transaction["state"] = "publishing"
            _replace_transaction(journal, transaction)
            # The durable journal is the commit intent. Recovery validates all
            # identities before completing any prefix left by a process loss.
            for index, final in enumerate(publish_order):
                if _path_entry_exists(final):
                    raise RuntimeError(
                        "foreign artifact appeared during evaluation publication: "
                        f"{final}"
                    )
                _before_atomic_publish(index, final)
                _link_no_replace(partials[final], final)
                expected_identity = artifact_records[index]["identity"]
                expected_instance = tuple(
                    artifact_records[index]["instance"]
                )
                if (
                    _entry_instance(final) != expected_instance
                    or not _identity_matches(final, expected_identity)
                ):
                    raise RuntimeError(
                        "evaluation artifact identity changed during publication: "
                        f"{final}"
                    )
                _after_publish_link_boundary(index, final)
                if (
                    _entry_instance(partials[final]) != expected_instance
                    or not _identity_matches(partials[final], expected_identity)
                ):
                    raise RuntimeError(
                        "evaluation partial identity changed during publication: "
                        f"{partials[final]}"
                    )
                partials[final].unlink()
                _sync_directory(final.parent)
                _after_publish_boundary(index, final)
            for record in artifact_records:
                final = Path(str(record["final"]))
                expected_instance = tuple(record["instance"])
                if (
                    _entry_instance(final) != expected_instance
                    or not _identity_matches(final, record["identity"])
                ):
                    raise RuntimeError(
                        "evaluation artifact identity changed before commit: "
                        f"{final}"
                    )
            transaction["state"] = "committed"
            _replace_transaction(journal, transaction)
        except BaseException:
            discard_building = (
                journal_owned and transaction.get("state") == "building"
            )
            if writer is not None:
                writer.close()
                writer = None
            raise
        finally:
            if uid_connection is not None:
                uid_connection.close()
            if accumulator is not None:
                accumulator.cleanup()
            if journal_owned:
                if discard_building:
                    _recover_transaction(
                        journal,
                        publish_order=publish_order,
                        test_contract=test_contract,
                        request_contract=request_contract,
                        prediction_path=prediction_final,
                        metrics_path=metrics_final,
                        plot_manifest_path=plot_manifest_final,
                        temporary_directory=temp_parent,
                    )
                else:
                    _cleanup_owned_work(
                        transaction, temporary_directory=temp_parent
                    )

        assert metrics is not None and plot_manifest is not None
        return {
            "metrics": metrics,
            "plot_manifest": plot_manifest,
            "prediction_path": str(prediction_final),
            "rows": storage_position,
        }
    finally:
        _release_transaction_lock(transaction_lock)


__all__ = ["evaluate_prediction_batches"]
