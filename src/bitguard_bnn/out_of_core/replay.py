from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..state import (
    TemporalSecurityStateMachine,
    replay_prediction_row,
    temporal_state_key,
)
from .evaluate import (
    _WORK_OWNER_FILE,
    _WORK_OWNER_INITIALIZING_FILE,
    _cleanup_owned_work as _cleanup_transaction_work,
    _replace_transaction,
    _sync_directory,
    _write_owner_marker,
    _write_transaction,
)

_MAX_MERGE_FAN_IN = 32
_REPLAY_TRANSACTION_VERSION = 1


def _path_entry_exists(path: Path) -> bool:
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


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _identity_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    if not _path_entry_exists(path) or path.is_symlink() or not path.is_file():
        return False
    return _file_identity(path) == dict(expected)


def _replay_lock_path(destination: Path) -> Path:
    lock_root = Path(tempfile.gettempdir()) / "bitguard-replay-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    filesystem_key = os.path.normcase(os.path.normpath(str(destination)))
    digest = hashlib.sha256(filesystem_key.encode("utf-8")).hexdigest()
    return lock_root / f"{digest}.lock"


def _acquire_replay_lock(destination: Path) -> BinaryIO:
    path = _replay_lock_path(destination)
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
            f"another replay process owns the output transaction: {destination}"
        ) from error


def _release_replay_lock(handle: BinaryIO) -> None:
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


def _before_replay_publish(_destination: Path, _partial: Path) -> None:
    """Test seam immediately before the publication no-clobber check."""


def _after_replay_publish(
    _destination: Path, _identity: Mapping[str, Any]
) -> None:
    """Test seam after publication used to verify identity-aware rollback."""


def _after_replay_build_boundary(_index: int, _path: Path) -> None:
    """Test seam representing a process loss during external-sort build."""


def _after_replay_link_boundary(_destination: Path, _partial: Path) -> None:
    """Test seam after durable linking and before partial removal."""


def _link_replay_no_replace(
    partial: Path,
    destination: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_instance: tuple[int, int],
) -> None:
    try:
        os.link(partial, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"foreign replay artifact appeared before publication: {destination}"
        ) from error
    _sync_directory(destination.parent)
    if (
        _entry_instance(destination) != expected_instance
        or not _identity_matches(destination, expected_identity)
    ):
        raise RuntimeError(
            f"replay artifact identity changed during publication: {destination}"
        )
    if (
        _entry_instance(partial) != expected_instance
        or not _identity_matches(partial, expected_identity)
    ):
        raise RuntimeError(
            f"replay partial identity changed during publication: {partial}"
        )
    _after_replay_link_boundary(destination, partial)
    partial.unlink()
    _sync_directory(destination.parent)


def _replay_transaction_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.replay-transaction.json")


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(dict(config)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _recover_replay_transaction(
    journal: Path,
    *,
    source: Path,
    source_identity: Mapping[str, Any],
    config_fingerprint: str,
    destination: Path,
    temporary_parent: Path,
) -> dict[str, Any] | None:
    if not _path_entry_exists(journal):
        return None
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"replay transaction cannot be validated: {journal}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("replay transaction is not a JSON object")
    if payload.get("format_version") != _REPLAY_TRANSACTION_VERSION:
        raise RuntimeError("replay transaction version is unsupported")
    state = payload.get("state")
    if state not in {"building", "publishing", "committed"}:
        raise RuntimeError("replay transaction state is invalid")
    transaction_id = payload.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise RuntimeError("replay transaction id is invalid")
    if (
        payload.get("source_path") != str(source)
        or payload.get("source_identity") != dict(source_identity)
        or payload.get("config_fingerprint") != config_fingerprint
        or payload.get("destination") != str(destination)
    ):
        raise RuntimeError("replay transaction input/config/output contract mismatch")
    if not isinstance(payload.get("output_parent_created"), bool):
        raise RuntimeError("replay transaction output parent ownership is invalid")
    owned_work = payload.get("owned_work")
    if not isinstance(owned_work, dict):
        raise RuntimeError("replay transaction work ownership is invalid")
    expected_work = temporary_parent / f"replay-{transaction_id}"
    expected_owner = expected_work / _WORK_OWNER_FILE
    expected_initializing = expected_work / _WORK_OWNER_INITIALIZING_FILE
    expected_owner_payload = {
        "format_version": 1,
        "transaction_id": transaction_id,
        "source_identity": dict(source_identity),
        "config_fingerprint": config_fingerprint,
    }
    if owned_work != {
        "path": str(expected_work),
        "owner_marker": str(expected_owner),
        "owner_initializing_marker": str(expected_initializing),
        "owner_payload": expected_owner_payload,
        "temporary_parent": str(temporary_parent),
        "parent_created": owned_work.get("parent_created"),
    } or not isinstance(owned_work.get("parent_created"), bool):
        raise RuntimeError("replay transaction work ownership contract mismatch")
    partial = destination.with_name(f".{destination.name}.{transaction_id}.partial")
    if payload.get("partial") != str(partial):
        raise RuntimeError("replay transaction partial path is invalid")

    if state == "building":
        if _path_entry_exists(destination):
            raise RuntimeError(
                f"foreign destination blocks building replay recovery: {destination}"
            )
        if _path_entry_exists(partial) and (
            partial.is_symlink() or not partial.is_file()
        ):
            raise RuntimeError(
                f"foreign partial blocks building replay recovery: {partial}"
            )
        _cleanup_transaction_work(payload, temporary_directory=temporary_parent)
        if _path_entry_exists(partial):
            partial.unlink()
        journal.unlink()
        _sync_directory(journal.parent)
        if bool(payload.get("output_parent_created")):
            try:
                destination.parent.rmdir()
            except OSError:
                pass
        return None

    output_identity = payload.get("output_identity")
    output_instance = payload.get("output_instance")
    metrics = payload.get("metrics")
    if (
        not isinstance(output_identity, dict)
        or set(output_identity) != {"size", "sha256"}
        or not isinstance(output_instance, list)
        or len(output_instance) != 2
        or any(not isinstance(value, int) for value in output_instance)
        or not isinstance(metrics, dict)
    ):
        raise RuntimeError("replay publishing transaction payload is invalid")
    expected_instance = (int(output_instance[0]), int(output_instance[1]))
    if _path_entry_exists(destination) and not _identity_matches(
        destination, output_identity
    ):
        raise RuntimeError(
            f"foreign destination or identity mismatch blocks replay recovery: {destination}"
        )
    if _path_entry_exists(destination) and _entry_instance(
        destination
    ) != expected_instance:
        raise RuntimeError(
            f"foreign destination instance blocks replay recovery: {destination}"
        )
    if _path_entry_exists(partial) and not _identity_matches(
        partial, output_identity
    ):
        raise RuntimeError(
            f"foreign partial or identity mismatch blocks replay recovery: {partial}"
        )
    if _path_entry_exists(partial) and _entry_instance(partial) != expected_instance:
        raise RuntimeError(
            f"foreign partial instance blocks replay recovery: {partial}"
        )
    if not _path_entry_exists(destination) and not _path_entry_exists(partial):
        raise RuntimeError("replay transaction is missing output and partial artifact")
    if not _path_entry_exists(destination):
        _link_replay_no_replace(
            partial,
            destination,
            expected_identity=output_identity,
            expected_instance=expected_instance,
        )
    elif _path_entry_exists(partial):
        partial.unlink()
    if not _identity_matches(destination, output_identity):
        raise RuntimeError("recovered replay artifact identity changed")
    if state != "committed":
        committed = dict(payload)
        committed["state"] = "committed"
        _replace_transaction(journal, committed)
        payload = committed
    _cleanup_transaction_work(payload, temporary_directory=temporary_parent)
    return dict(metrics)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _SortedJsonRuns:
    def __init__(
        self,
        root: Path,
        prefix: str,
        run_rows: int,
        key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    ) -> None:
        if run_rows <= 0:
            raise ValueError("order_run_rows must be positive")
        self.root = root
        self.prefix = prefix
        self.run_rows = int(run_rows)
        self.key = key
        self.buffer: list[dict[str, Any]] = []
        self.paths: list[Path] = []

    def add(self, row: Mapping[str, Any]) -> None:
        self.buffer.append(dict(row))
        if len(self.buffer) >= self.run_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.buffer.sort(key=self.key)
        path = self.root / f"{self.prefix}-{len(self.paths):08d}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in self.buffer:
                handle.write(
                    json.dumps(
                        _json_safe(row),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
        self.paths.append(path)
        self.buffer.clear()

    @staticmethod
    def _read(handle: TextIO) -> dict[str, Any] | None:
        line = handle.readline()
        return None if not line else dict(json.loads(line))

    def _iter_paths(self, paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(path.open("r", encoding="utf-8"))
                for path in paths
            ]
            heap: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
            for index, handle in enumerate(handles):
                row = self._read(handle)
                if row is not None:
                    heapq.heappush(heap, (self.key(row), index, row))
            while heap:
                _key, index, row = heapq.heappop(heap)
                yield row
                following = self._read(handles[index])
                if following is not None:
                    heapq.heappush(
                        heap, (self.key(following), index, following)
                    )

    def _compact(self) -> None:
        generation = 0
        while len(self.paths) > _MAX_MERGE_FAN_IN:
            following: list[Path] = []
            for group_index, start in enumerate(
                range(0, len(self.paths), _MAX_MERGE_FAN_IN)
            ):
                group = self.paths[start : start + _MAX_MERGE_FAN_IN]
                if len(group) == 1:
                    following.append(group[0])
                    continue
                path = self.root / (
                    f"{self.prefix}-merge-{generation:04d}-{group_index:08d}.jsonl"
                )
                with path.open("x", encoding="utf-8", newline="\n") as handle:
                    for row in self._iter_paths(group):
                        handle.write(
                            json.dumps(
                                _json_safe(row),
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                        )
                        handle.write("\n")
                for old in group:
                    old.unlink()
                following.append(path)
            self.paths = following
            generation += 1

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.flush()
        self._compact()
        yield from self._iter_paths(self.paths)


class _OperationalAccumulator:
    """Disk-backed per-episode operational metrics without a full frame."""

    def __init__(self, path: Path, *, timestamps_complete: bool) -> None:
        self.connection = sqlite3.connect(path)
        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.executescript(
                """
                CREATE TABLE episode (
                    episode TEXT PRIMARY KEY,
                    decisions INTEGER NOT NULL,
                    last_attack INTEGER NOT NULL,
                    last_alert INTEGER NOT NULL,
                    onset_decision INTEGER,
                    onset_time REAL,
                    first_alert_decision INTEGER,
                    first_alert_time REAL,
                    first_mitigation_decision INTEGER,
                    first_mitigation_time REAL,
                    minimum_time REAL,
                    maximum_time REAL
                );
                CREATE TABLE delay (kind TEXT NOT NULL, value REAL NOT NULL);
                CREATE INDEX delay_order ON delay(kind, value);
                """
            )
        except BaseException:
            self.connection.close()
            raise
        self.timestamps_complete = timestamps_complete
        self.has_real_time = timestamps_complete
        self.continuous = True
        self.rows = 0
        self.alerts = 0
        self.mitigations = 0
        self.benign_rows = 0
        self.attack_rows = 0
        self.benign_mitigations = 0
        self.attack_mitigations = 0
        self.attack_alerts = 0
        self.stateless_alerts = 0
        self.stateless_attack_alerts = 0
        self.low_rows = 0
        self.low_alerts = 0
        self.false_positive_alert_events = 0
        self.attack_episode_count = 0
        self.missed_alert_episodes = 0
        self.missed_mitigation_episodes = 0
        self.closed = False

    def _finish_episode(self, row: sqlite3.Row | Mapping[str, Any]) -> None:
        self.attack_episode_count += 1
        onset_decision = int(row["onset_decision"])
        onset_time = row["onset_time"]
        first_alert = row["first_alert_decision"]
        if first_alert is None:
            self.missed_alert_episodes += 1
        else:
            self.connection.execute(
                "INSERT INTO delay VALUES (?, ?)",
                ("alert_decisions", float(int(first_alert) - onset_decision)),
            )
            if onset_time is not None and row["first_alert_time"] is not None:
                self.connection.execute(
                    "INSERT INTO delay VALUES (?, ?)",
                    (
                        "alert_seconds",
                        float(row["first_alert_time"]) - float(onset_time),
                    ),
                )
        first_mitigation = row["first_mitigation_decision"]
        if first_mitigation is None:
            self.missed_mitigation_episodes += 1
        else:
            self.connection.execute(
                "INSERT INTO delay VALUES (?, ?)",
                (
                    "mitigation_decisions",
                    float(int(first_mitigation) - onset_decision),
                ),
            )
            if onset_time is not None and row["first_mitigation_time"] is not None:
                self.connection.execute(
                    "INSERT INTO delay VALUES (?, ?)",
                    (
                        "mitigation_seconds",
                        float(row["first_mitigation_time"]) - float(onset_time),
                    ),
                )

    def update(self, row: Mapping[str, Any]) -> None:
        true_label = str(row["true_label"])
        predicted = str(row["predicted_label"])
        action = int(row["action_level"])
        attack = true_label != "benign"
        alert = action >= 2
        mitigation = action >= 3
        self.rows += 1
        self.alerts += int(alert)
        self.mitigations += int(mitigation)
        self.attack_rows += int(attack)
        self.benign_rows += int(not attack)
        self.attack_alerts += int(attack and alert)
        self.attack_mitigations += int(attack and mitigation)
        self.benign_mitigations += int(not attack and mitigation)
        stateless = predicted != "benign"
        self.stateless_alerts += int(stateless)
        self.stateless_attack_alerts += int(stateless and attack)
        raw_attack = str(row.get("raw_attack", "")).lower()
        low_rate = any(token in raw_attack for token in ("low", "slow", "beacon"))
        self.low_rows += int(low_rate)
        self.low_alerts += int(low_rate and alert)
        self.has_real_time = self.has_real_time and bool(
            row.get("has_wall_clock_time", False)
        )
        self.continuous = self.continuous and bool(
            row.get("temporal_continuity", False)
        )

        source = str(row.get("source_file", "default_episode"))
        device = str(row["device_id"])
        key = temporal_state_key(source, device)
        current = self.connection.execute(
            "SELECT * FROM episode WHERE episode = ?", (key,)
        ).fetchone()
        timestamp_raw = row.get("timestamp")
        timestamp = (
            float(timestamp_raw)
            if timestamp_raw is not None and np.isfinite(float(timestamp_raw))
            else None
        )
        if current is None:
            state: dict[str, Any] = {
                "decisions": 0,
                "last_attack": 0,
                "last_alert": 0,
                "onset_decision": None,
                "onset_time": None,
                "first_alert_decision": None,
                "first_alert_time": None,
                "first_mitigation_decision": None,
                "first_mitigation_time": None,
                "minimum_time": timestamp,
                "maximum_time": timestamp,
            }
        else:
            state = dict(current)
        decision = int(state["decisions"])
        if bool(state["last_attack"]) and not attack:
            self._finish_episode(state)
            for name in (
                "onset_decision",
                "onset_time",
                "first_alert_decision",
                "first_alert_time",
                "first_mitigation_decision",
                "first_mitigation_time",
            ):
                state[name] = None
        if attack and not bool(state["last_attack"]):
            state["onset_decision"] = decision
            state["onset_time"] = timestamp
            state["first_alert_decision"] = None
            state["first_alert_time"] = None
            state["first_mitigation_decision"] = None
            state["first_mitigation_time"] = None
        if attack and alert and state["first_alert_decision"] is None:
            state["first_alert_decision"] = decision
            state["first_alert_time"] = timestamp
        if attack and mitigation and state["first_mitigation_decision"] is None:
            state["first_mitigation_decision"] = decision
            state["first_mitigation_time"] = timestamp
        if alert and not bool(state["last_alert"]) and not attack:
            self.false_positive_alert_events += 1
        if timestamp is not None:
            minimum = state["minimum_time"]
            maximum = state["maximum_time"]
            state["minimum_time"] = timestamp if minimum is None else min(float(minimum), timestamp)
            state["maximum_time"] = timestamp if maximum is None else max(float(maximum), timestamp)
        state["decisions"] = decision + 1
        state["last_attack"] = int(attack)
        state["last_alert"] = int(alert)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO episode VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                state["decisions"],
                state["last_attack"],
                state["last_alert"],
                state["onset_decision"],
                state["onset_time"],
                state["first_alert_decision"],
                state["first_alert_time"],
                state["first_mitigation_decision"],
                state["first_mitigation_time"],
                state["minimum_time"],
                state["maximum_time"],
            ),
        )
        if self.rows % 10_000 == 0:
            self.connection.commit()

    def _percentile(self, kind: str, percentile: float) -> float | None:
        count = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM delay WHERE kind = ?", (kind,)
            ).fetchone()[0]
        )
        if count == 0:
            return None
        rank = (count - 1) * percentile / 100.0
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        lower_value = float(
            self.connection.execute(
                "SELECT value FROM delay WHERE kind = ? ORDER BY value LIMIT 1 OFFSET ?",
                (kind, lower),
            ).fetchone()[0]
        )
        if lower == upper:
            return lower_value
        upper_value = float(
            self.connection.execute(
                "SELECT value FROM delay WHERE kind = ? ORDER BY value LIMIT 1 OFFSET ?",
                (kind, upper),
            ).fetchone()[0]
        )
        return lower_value + (upper_value - lower_value) * (rank - lower)

    def finalize(self) -> dict[str, Any]:
        for row in self.connection.execute(
            "SELECT * FROM episode WHERE last_attack = 1"
        ):
            self._finish_episode(row)
        self.connection.commit()
        attack_mitigation_rate = (
            self.attack_mitigations / self.attack_rows if self.attack_rows else None
        )
        effective_real_time = self.has_real_time and self.continuous
        metrics: dict[str, Any] = {
            "rows": self.rows,
            "alerts": self.alerts,
            "mitigation_recommendations": self.mitigations,
            "benign_disruption_rate": self.benign_mitigations / self.benign_rows if self.benign_rows else None,
            "action_recommendation_precision": self.attack_mitigations / self.mitigations if self.mitigations else None,
            "attack_row_recall_at_alert": self.attack_alerts / self.attack_rows if self.attack_rows else None,
            "attack_row_recall_at_mitigation": attack_mitigation_rate,
            "alert_reduction_ratio_vs_alert_every_row": 1.0 - self.alerts / self.rows,
            "packed_counter_state_bytes_per_device_theoretical": 3,
            "counter_count": 5,
            "automatic_blocking_performed": False,
            "attack_reduction_sensitivity": {
                "25_percent_effective_after_level3": attack_mitigation_rate * 0.25 if attack_mitigation_rate is not None else None,
                "50_percent_effective_after_level3": attack_mitigation_rate * 0.50 if attack_mitigation_rate is not None else None,
                "75_percent_effective_after_level3": attack_mitigation_rate * 0.75 if attack_mitigation_rate is not None else None,
            },
            "stateless_alerts": self.stateless_alerts,
            "alert_reduction_ratio_vs_stateless_classifier": 1.0 - self.alerts / self.stateless_alerts if self.stateless_alerts else None,
            "stateless_attack_recall": self.stateless_attack_alerts / self.attack_rows if self.attack_rows else None,
            "stateful_attack_recall_at_level2": self.attack_alerts / self.attack_rows if self.attack_rows else None,
            "low_rate_recall_at_level2": self.low_alerts / self.low_rows if self.low_rows else None,
            "detection_delay_decisions_p50": self._percentile("alert_decisions", 50) if self.continuous else None,
            "detection_delay_decisions_p95": self._percentile("alert_decisions", 95) if self.continuous else None,
            "detection_delay_seconds_p50": self._percentile("alert_seconds", 50) if effective_real_time else None,
            "detection_delay_seconds_p95": self._percentile("alert_seconds", 95) if effective_real_time else None,
            "time_to_mitigation_p50_seconds_or_decisions": self._percentile("mitigation_seconds" if effective_real_time else "mitigation_decisions", 50) if self.continuous else None,
            "time_to_mitigation_unit": "seconds" if effective_real_time else "decisions",
            "attack_episode_count": self.attack_episode_count,
            "missed_alert_episode_count": self.missed_alert_episodes,
            "missed_mitigation_episode_count": self.missed_mitigation_episodes,
            "delay_note": "Delay percentiles are conditional on detection; misses are reported separately.",
            "false_positive_alert_events": self.false_positive_alert_events,
        }
        if effective_real_time:
            hours = 0.0
            for minimum, maximum in self.connection.execute(
                "SELECT minimum_time, maximum_time FROM episode"
            ):
                hours += max((float(maximum) - float(minimum)) / 3600.0, 1.0 / 3600.0)
            metrics["false_positive_alerts_per_device_hour"] = self.false_positive_alert_events / hours
            metrics["observed_device_hours"] = hours
        else:
            metrics["false_positive_alerts_per_device_hour"] = None
            metrics["observed_device_hours"] = None
            metrics["time_metric_note"] = "No verified continuous wall-clock episode; device-hour and time-delay metrics are withheld."
        metrics["temporal_continuity_verified"] = self.continuous
        return metrics

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.connection.close()


def _timestamps_complete(parquet: pq.ParquetFile, batch_rows: int) -> bool:
    if "timestamp" not in parquet.schema_arrow.names:
        return False
    for batch in parquet.iter_batches(columns=["timestamp"], batch_size=batch_rows):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(numeric).all():
            return False
    return True


def replay_parquet_predictions(
    prediction_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    temporary_directory: str | Path | None = None,
    batch_rows: int = 65_536,
    order_run_rows: int = 131_072,
) -> dict[str, Any]:
    """Replay prediction Parquet with external ordering and bounded state."""

    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    source = Path(prediction_path).resolve()
    destination = _absolute_output_path(output_path)
    if source == destination:
        raise ValueError("prediction input and replay output must be distinct")
    parent = (
        Path(temporary_directory)
        if temporary_directory is not None
        else destination.parent / f".{destination.name}.replay-temporary"
    ).resolve()
    journal = _replay_transaction_path(destination)
    source_identity = _file_identity(source)
    config_digest = _config_fingerprint(config)
    timestamps_complete = False
    use_sequence = False

    def chronological_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        position = int(row["__replay_position"])
        uid = str(row["row_uid"])
        if timestamps_complete:
            return (
                float(row["timestamp"]),
                str(row["device_id"]),
                position,
                uid,
            )
        if use_sequence:
            return (
                str(row["device_id"]),
                int(row["sequence_index"]),
                position,
                uid,
            )
        return (position, uid)

    lock = _acquire_replay_lock(destination)
    parent_created = False
    destination_parent_created = False
    work: Path | None = None
    partial: Path | None = None
    published_identity: dict[str, Any] | None = None
    operational: _OperationalAccumulator | None = None
    writer: pq.ParquetWriter | None = None
    parquet: pq.ParquetFile | None = None
    transaction: dict[str, Any] | None = None
    journal_owned = False
    try:
        recovered = _recover_replay_transaction(
            journal,
            source=source,
            source_identity=source_identity,
            config_fingerprint=config_digest,
            destination=destination,
            temporary_parent=parent,
        )
        if recovered is not None:
            return recovered
        if _path_entry_exists(destination):
            raise FileExistsError(
                f"refusing to replace replay artifact: {destination}"
            )
        parquet = pq.ParquetFile(source)
        names = parquet.schema_arrow.names
        required = {"device_id", "true_label", "predicted_label", "row_uid"}
        missing = required - set(names)
        if missing:
            raise ValueError(f"prediction replay missing columns: {sorted(missing)}")
        probability_columns = [name for name in names if name.startswith("prob_")]
        if not probability_columns:
            raise ValueError("prediction replay needs prob_<label> columns")
        timestamps_complete = _timestamps_complete(parquet, batch_rows)
        use_sequence = not timestamps_complete and "sequence_index" in names

        transaction_id = uuid.uuid4().hex
        work = parent / f"replay-{transaction_id}"
        partial = destination.with_name(
            f".{destination.name}.{transaction_id}.partial"
        )
        parent_created = not _path_entry_exists(parent)
        destination_parent_created = not _path_entry_exists(destination.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        owner_payload = {
            "format_version": 1,
            "transaction_id": transaction_id,
            "source_identity": source_identity,
            "config_fingerprint": config_digest,
        }
        owner_marker = work / _WORK_OWNER_FILE
        initializing_owner = work / _WORK_OWNER_INITIALIZING_FILE
        transaction = {
            "format_version": _REPLAY_TRANSACTION_VERSION,
            "state": "building",
            "transaction_id": transaction_id,
            "source_path": str(source),
            "source_identity": source_identity,
            "config_fingerprint": config_digest,
            "destination": str(destination),
            "partial": str(partial),
            "output_parent_created": destination_parent_created,
            "output_identity": None,
            "output_instance": None,
            "metrics": None,
            "owned_work": {
                "path": str(work),
                "owner_marker": str(owner_marker),
                "owner_initializing_marker": str(initializing_owner),
                "owner_payload": owner_payload,
                "temporary_parent": str(parent),
                "parent_created": parent_created,
            },
        }
        _write_transaction(journal, transaction)
        journal_owned = True
        parent.mkdir(parents=True, exist_ok=True)
        work.mkdir(exist_ok=False)
        _sync_directory(parent)
        _write_owner_marker(owner_marker, initializing_owner, owner_payload)
        chronological = _SortedJsonRuns(
            work, "chronological", order_run_rows, chronological_key
        )
        restored = _SortedJsonRuns(
            work,
            "storage",
            order_run_rows,
            lambda row: (
                int(row.get("storage_position", row["__replay_position"])),
                str(row["row_uid"]),
                int(row["__replay_position"]),
            ),
        )
        operational = _OperationalAccumulator(
            work / "operational.sqlite", timestamps_complete=timestamps_complete
        )
        position = 0
        for batch_index, batch in enumerate(
            parquet.iter_batches(batch_size=batch_rows)
        ):
            for row in pa.Table.from_batches([batch]).to_pylist():
                row["__replay_position"] = position
                chronological.add(row)
                position += 1
            _after_replay_build_boundary(batch_index, work)
        if position == 0:
            raise ValueError("prediction replay requires at least one row")

        machine = TemporalSecurityStateMachine(config)
        for row in chronological:
            transformed = replay_prediction_row(
                row, machine, probability_columns
            )
            operational.update(transformed)
            restored.add(transformed)
        metrics = operational.finalize()
        metrics["temporal_state_device_evictions"] = machine.evictions
        metrics["temporal_state_peak_device_capacity"] = machine.max_devices

        output_buffer: list[dict[str, Any]] = []
        schema: pa.Schema | None = None
        for row in restored:
            row.pop("__replay_position", None)
            output_buffer.append(row)
            if len(output_buffer) < batch_rows:
                continue
            table = pa.Table.from_pylist(output_buffer, schema=schema)
            if writer is None:
                source_metadata = parquet.schema_arrow.metadata or {}
                metadata = {
                    b"bitguard_artifact_format": b"temporal_prediction_parquet",
                    b"bitguard_artifact_version": b"1",
                    b"bitguard_probability_labels": json.dumps(
                        [column.removeprefix("prob_") for column in probability_columns],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                }
                if b"bitguard_test_contract" in source_metadata:
                    metadata[b"bitguard_test_contract"] = source_metadata[
                        b"bitguard_test_contract"
                    ]
                table = table.replace_schema_metadata(metadata)
                schema = table.schema
                if _path_entry_exists(partial):
                    raise FileExistsError(
                        f"foreign replay partial blocks output: {partial}"
                    )
                writer = pq.ParquetWriter(partial, schema, compression="zstd")
            writer.write_table(table, row_group_size=len(table))
            output_buffer.clear()
        if output_buffer:
            table = pa.Table.from_pylist(output_buffer, schema=schema)
            if writer is None:
                source_metadata = parquet.schema_arrow.metadata or {}
                metadata = {
                    b"bitguard_artifact_format": b"temporal_prediction_parquet",
                    b"bitguard_artifact_version": b"1",
                    b"bitguard_probability_labels": json.dumps(
                        [column.removeprefix("prob_") for column in probability_columns],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                }
                if b"bitguard_test_contract" in source_metadata:
                    metadata[b"bitguard_test_contract"] = source_metadata[
                        b"bitguard_test_contract"
                    ]
                table = table.replace_schema_metadata(metadata)
                schema = table.schema
                if _path_entry_exists(partial):
                    raise FileExistsError(
                        f"foreign replay partial blocks output: {partial}"
                    )
                writer = pq.ParquetWriter(partial, schema, compression="zstd")
            writer.write_table(table, row_group_size=len(table))
        if writer is None:
            raise RuntimeError("replay output writer made no progress")
        writer.close()
        writer = None
        with partial.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        published_identity = _file_identity(partial)
        published_instance = _entry_instance(partial)
        if published_instance is None:
            raise RuntimeError("replay partial disappeared before publication")
        transaction["output_identity"] = published_identity
        transaction["output_instance"] = list(published_instance)
        transaction["metrics"] = metrics
        transaction["state"] = "publishing"
        _replace_transaction(journal, transaction)
        if _path_entry_exists(destination):
            raise FileExistsError(
                f"foreign replay artifact appeared before publication: {destination}"
            )
        _before_replay_publish(destination, partial)
        _link_replay_no_replace(
            partial,
            destination,
            expected_identity=published_identity,
            expected_instance=published_instance,
        )
        _after_replay_publish(destination, published_identity)
        if (
            _entry_instance(destination) != published_instance
            or not _identity_matches(destination, published_identity)
        ):
            raise RuntimeError(
                f"replay artifact identity changed before commit: {destination}"
            )
        transaction["state"] = "committed"
        _replace_transaction(journal, transaction)
        return metrics
    except BaseException:
        if writer is not None:
            writer.close()
            writer = None
        raise
    finally:
        if writer is not None:
            writer.close()
        if operational is not None:
            operational.close()
        if parquet is not None:
            parquet.close(force=True)
        if journal_owned and transaction is not None:
            if transaction.get("state") == "building":
                _recover_replay_transaction(
                    journal,
                    source=source,
                    source_identity=source_identity,
                    config_fingerprint=config_digest,
                    destination=destination,
                    temporary_parent=parent,
                )
            else:
                _cleanup_transaction_work(
                    transaction, temporary_directory=parent
                )
        else:
            if parent_created:
                try:
                    parent.rmdir()
                except OSError:
                    pass
            if destination_parent_created:
                try:
                    destination.parent.rmdir()
                except OSError:
                    pass
        _release_replay_lock(lock)


__all__ = ["replay_parquet_predictions"]
