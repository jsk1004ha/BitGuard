"""Deterministic bounded-memory row-priority quantile sketches.

The sketch retains complete rows so every feature uses the same deterministic
sample.  Input streams must have globally unique ``row_uid`` values, and
sketches passed to :meth:`PriorityRowSketch.merge` must be built from disjoint
streams.  A retained duplicate can be detected exactly, but a fixed-size
sketch cannot detect a duplicate UID after that UID has been evicted.  This
limitation is included in every serialized snapshot.
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import math
import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


ALGORITHM = "blake2b-128-bottom-k-complete-row"
VERSION = 1
_PERSONALIZATION = b"bg-row-prio-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _priority_for_uid(seed: int, row_uid: str) -> int:
    """Return the version-1 128-bit priority for a canonical seed/UID pair."""

    seed_bytes = str(seed).encode("ascii")
    uid_bytes = row_uid.encode("utf-8")
    message = (
        len(seed_bytes).to_bytes(4, "big")
        + seed_bytes
        + len(uid_bytes).to_bytes(8, "big")
        + uid_bytes
    )
    digest = hashlib.blake2b(
        message,
        digest_size=16,
        person=_PERSONALIZATION,
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


def _float_token(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return value.hex()


def _float_from_token(token: object) -> float:
    if not isinstance(token, str):
        raise ValueError("serialized row value must be a string token")
    if token == "nan":
        return float("nan")
    if token == "+inf":
        return float("inf")
    if token == "-inf":
        return float("-inf")
    try:
        value = float.fromhex(token)
    except ValueError as exc:
        raise ValueError("invalid serialized float token") from exc
    if not math.isfinite(value):
        raise ValueError("non-finite serialized values require canonical tokens")
    return value


def _row_tokens(values: Sequence[float]) -> tuple[str, ...]:
    return tuple(_float_token(value) for value in values)


def _require_plain_int(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class _HeapRow:
    priority: int
    row_uid: str
    values: tuple[float, ...]

    @property
    def key(self) -> tuple[int, str]:
        return self.priority, self.row_uid

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _HeapRow):
            return NotImplemented
        # heapq is a min-heap. Reverse the key so index zero is the worst
        # retained row and can be replaced in O(log capacity).
        return self.key > other.key


class PriorityRowSketch:
    """A deterministic bottom-k sample of complete numeric rows.

    ``NaN`` and infinities are accepted as missing observations.  Quantiles
    use only finite retained values, while finite and missing counts describe
    every accepted input row under the globally-unique-UID precondition.
    """

    algorithm = ALGORITHM
    version = VERSION

    def __init__(self, *, capacity: int, seed: int, width: int) -> None:
        self._capacity = _require_plain_int(capacity, "capacity", positive=True)
        self._seed = _require_plain_int(seed, "seed")
        self._width = _require_plain_int(width, "width", positive=True)
        self._algorithm = self.algorithm
        self._version = self.version
        self._heap: list[_HeapRow] = []
        self._retained_by_uid: dict[str, _HeapRow] = {}
        self._total_rows = 0
        self._finite_counts: NDArray[np.int64] = np.zeros(
            self.width, dtype=np.int64
        )
        self._missing_counts: NDArray[np.int64] = np.zeros(
            self.width, dtype=np.int64
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def width(self) -> int:
        return self._width

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def retained_count(self) -> int:
        return len(self._heap)

    @property
    def finite_counts(self) -> np.ndarray:
        result = self._finite_counts.copy()
        result.flags.writeable = False
        return result

    @property
    def missing_counts(self) -> np.ndarray:
        result = self._missing_counts.copy()
        result.flags.writeable = False
        return result

    def _validate_uid(self, row_uid: object) -> str:
        if not isinstance(row_uid, str) or not row_uid:
            raise ValueError("row_uid must be a non-empty string")
        try:
            row_uid.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("row_uid must be valid UTF-8") from exc
        return row_uid

    def _validate_values(self, values: object) -> tuple[float, ...]:
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError("row values must be one-dimensional")
        if len(raw) != self.width:
            raise ValueError(
                f"row width mismatch: expected {self.width}, received {len(raw)}"
            )
        if raw.dtype.kind not in "iuf" and not (
            raw.dtype.kind == "O"
            and all(
                isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
                for value in raw
            )
        ):
            raise TypeError("row values must be numeric")
        try:
            numeric = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("row values must be numeric") from exc
        result: list[float] = []
        for value in numeric:
            converted = float(value)
            result.append(float("nan") if math.isnan(converted) else converted)
        return tuple(result)

    def _retain(self, row: _HeapRow) -> None:
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, row)
            self._retained_by_uid[row.row_uid] = row
            return
        if row.key >= self._heap[0].key:
            return
        evicted = heapq.heapreplace(self._heap, row)
        del self._retained_by_uid[evicted.row_uid]
        self._retained_by_uid[row.row_uid] = row

    def update(self, row_uid: str, values: Sequence[float] | np.ndarray) -> None:
        """Observe one row.

        An exact duplicate is idempotent only while its UID is retained.  A
        conflicting retained duplicate is rejected.  Callers must enforce
        global UID uniqueness for rows no longer present in the bottom-k set.
        """

        uid = self._validate_uid(row_uid)
        row_values = self._validate_values(values)
        existing = self._retained_by_uid.get(uid)
        if existing is not None:
            if _row_tokens(existing.values) == _row_tokens(row_values):
                return
            raise ValueError(f"conflicting duplicate row_uid: {uid}")

        finite = np.isfinite(np.asarray(row_values, dtype=np.float64))
        self._total_rows += 1
        self._finite_counts += finite.astype(np.int64)
        self._missing_counts += (~finite).astype(np.int64)
        self._retain(_HeapRow(_priority_for_uid(self.seed, uid), uid, row_values))

    def update_many(
        self,
        row_uids: Iterable[str] | np.ndarray,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> None:
        uids = list(row_uids)
        raw = np.asarray(values)
        if raw.ndim != 2:
            raise ValueError("batch values must be two-dimensional")
        if raw.shape != (len(uids), self.width):
            raise ValueError(
                "batch shape mismatch: expected "
                f"({len(uids)}, {self.width}), received {raw.shape}"
            )
        for index, uid in enumerate(uids):
            self.update(uid, raw[index])

    def _compatibility_key(self) -> tuple[object, ...]:
        return (
            self._algorithm,
            self._version,
            self.capacity,
            self.seed,
            self.width,
        )

    def merge(self, other: "PriorityRowSketch") -> "PriorityRowSketch":
        """Merge a compatible sketch built from a disjoint UID stream.

        Overlap among retained rows is detected exactly and identical rows are
        counted once.  Overlap involving an evicted row is not detectable; the
        disjoint-stream precondition is therefore required for exact counts.
        """

        if not isinstance(other, PriorityRowSketch):
            raise TypeError("other must be a PriorityRowSketch")
        if self is other:
            raise ValueError("a sketch cannot merge itself; merge inputs must be disjoint")
        if self._compatibility_key() != other._compatibility_key():
            raise ValueError("incompatible priority sketches")

        overlap = set(self._retained_by_uid).intersection(other._retained_by_uid)
        for uid in overlap:
            left = self._retained_by_uid[uid]
            right = other._retained_by_uid[uid]
            if _row_tokens(left.values) != _row_tokens(right.values):
                raise ValueError(f"conflicting duplicate row_uid: {uid}")

        combined_rows = dict(self._retained_by_uid)
        for uid, row in other._retained_by_uid.items():
            combined_rows.setdefault(uid, row)
        retained = heapq.nsmallest(
            self.capacity,
            combined_rows.values(),
            key=lambda row: row.key,
        )

        overlap_finite: NDArray[np.int64] = np.zeros(
            self.width, dtype=np.int64
        )
        for uid in overlap:
            values = np.asarray(self._retained_by_uid[uid].values, dtype=np.float64)
            overlap_finite += np.isfinite(values).astype(np.int64)
        overlap_missing: NDArray[np.int64] = len(overlap) - overlap_finite

        # Commit only after compatibility and overlap validation have succeeded.
        self._total_rows = self._total_rows + other._total_rows - len(overlap)
        self._finite_counts = (
            self._finite_counts + other._finite_counts - overlap_finite
        )
        self._missing_counts = (
            self._missing_counts + other._missing_counts - overlap_missing
        )
        self._heap = list(retained)
        heapq.heapify(self._heap)
        self._retained_by_uid = {row.row_uid: row for row in retained}
        return self

    def quantile(self, column: int, q: float) -> float:
        if isinstance(column, bool) or not isinstance(column, int):
            raise TypeError("column must be an integer")
        if column < 0 or column >= self.width:
            raise ValueError(f"column must be in [0, {self.width})")
        if isinstance(q, bool) or not isinstance(q, (int, float, np.integer, np.floating)):
            raise TypeError("q must be a finite number in [0, 1]")
        q_value = float(q)
        if not math.isfinite(q_value) or not 0.0 <= q_value <= 1.0:
            raise ValueError("q must be a finite number in [0, 1]")
        finite_values = np.asarray(
            [
                row.values[column]
                for row in self._heap
                if math.isfinite(row.values[column])
            ],
            dtype=np.float64,
        )
        if not len(finite_values):
            raise ValueError(f"column {column} has no finite retained values")
        return float(np.quantile(finite_values, q_value))

    @staticmethod
    def _deduplication_contract() -> dict[str, Any]:
        return {
            "globally_unique_row_uid_required": True,
            "merge_inputs_must_be_disjoint": True,
            "supports_exact_overlap_detection": False,
            "retained_overlap_detection": "exact",
            "evicted_overlap_detection": "unsupported",
        }

    def confidence_metadata(self, *, confidence: float = 0.95) -> dict[str, Any]:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float, np.integer, np.floating))
            or not math.isfinite(float(confidence))
            or not 0.0 < float(confidence) < 1.0
        ):
            raise ValueError("confidence must be a finite number in (0, 1)")
        confidence_value = float(confidence)
        alpha = 1.0 - confidence_value
        columns: list[dict[str, object]] = []
        for column in range(self.width):
            sample_size = sum(
                math.isfinite(row.values[column]) for row in self._heap
            )
            bound = None
            if sample_size:
                bound = min(
                    1.0,
                    math.sqrt(math.log(2.0 / alpha) / (2.0 * sample_size)),
                )
            columns.append(
                {
                    "column": column,
                    "retained_finite_samples": sample_size,
                    "cdf_supremum_bound": bound,
                }
            )
        return {
            "method": "Dvoretzky-Kiefer-Wolfowitz",
            "confidence": confidence_value,
            "columns": columns,
            "deterministic_value_error_bound": None,
            "semantics": (
                "With the stated probability, each finite retained sample's "
                "empirical CDF is within its reported supremum bound of the "
                "corresponding finite-row distribution; no deterministic "
                "quantile value-error bound is claimed."
            ),
            "deduplication": self._deduplication_contract(),
        }

    def snapshot(self) -> dict[str, Any]:
        retained_rows = [
            {
                "priority": f"{row.priority:032x}",
                "row_uid": row.row_uid,
                "values": list(_row_tokens(row.values)),
            }
            for row in sorted(self._heap, key=lambda item: item.key)
        ]
        return {
            "algorithm": self._algorithm,
            "version": self._version,
            "capacity": self.capacity,
            "seed": self.seed,
            "width": self.width,
            "total_rows": self._total_rows,
            "finite_counts": self._finite_counts.tolist(),
            "missing_counts": self._missing_counts.tolist(),
            "retained_rows": retained_rows,
            "deduplication": self._deduplication_contract(),
        }

    def to_bytes(self) -> bytes:
        payload = self.snapshot()
        payload_bytes = _canonical_json(payload)
        envelope = {
            "checksum": hashlib.sha256(payload_bytes).hexdigest(),
            "payload": payload,
        }
        return _canonical_json(envelope)

    @classmethod
    def from_bytes(cls, serialized: bytes | bytearray | memoryview) -> "PriorityRowSketch":
        try:
            raw = bytes(serialized)
            envelope = json.loads(raw.decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid or tampered sketch serialization") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"checksum", "payload"}:
            raise ValueError("invalid sketch serialization envelope")
        checksum = envelope["checksum"]
        payload = envelope["payload"]
        if not isinstance(checksum, str) or not isinstance(payload, dict):
            raise ValueError("invalid sketch serialization envelope")
        payload_bytes = _canonical_json(payload)
        expected_checksum = hashlib.sha256(payload_bytes).hexdigest()
        if not hmac.compare_digest(checksum, expected_checksum):
            raise ValueError("sketch serialization checksum mismatch (tampered payload)")
        return cls._from_snapshot(payload)

    @classmethod
    def _from_snapshot(cls, payload: Mapping[str, Any]) -> "PriorityRowSketch":
        expected_keys = {
            "algorithm",
            "version",
            "capacity",
            "seed",
            "width",
            "total_rows",
            "finite_counts",
            "missing_counts",
            "retained_rows",
            "deduplication",
        }
        if set(payload) != expected_keys:
            raise ValueError("serialized sketch fields are invalid")
        if payload["algorithm"] != ALGORITHM:
            raise ValueError("unsupported sketch algorithm")
        if payload["version"] != VERSION:
            raise ValueError("unsupported sketch version")
        if payload["deduplication"] != cls._deduplication_contract():
            raise ValueError("serialized deduplication contract is invalid")

        sketch = cls(
            capacity=_require_plain_int(payload["capacity"], "capacity", positive=True),
            seed=_require_plain_int(payload["seed"], "seed"),
            width=_require_plain_int(payload["width"], "width", positive=True),
        )
        total_rows = _require_plain_int(payload["total_rows"], "total_rows")
        if total_rows < 0:
            raise ValueError("total_rows must be non-negative")
        finite = cls._parse_counts(payload["finite_counts"], sketch.width, "finite_counts")
        missing = cls._parse_counts(payload["missing_counts"], sketch.width, "missing_counts")
        if np.any(finite + missing != total_rows):
            raise ValueError("serialized counts do not match total_rows")

        retained_payload = payload["retained_rows"]
        if not isinstance(retained_payload, list):
            raise ValueError("retained_rows must be a list")
        expected_retained = min(total_rows, sketch.capacity)
        if len(retained_payload) != expected_retained:
            raise ValueError("serialized retained row count is invalid")

        rows: list[_HeapRow] = []
        seen_uids: set[str] = set()
        previous_key: tuple[int, str] | None = None
        for item in retained_payload:
            if not isinstance(item, dict) or set(item) != {"priority", "row_uid", "values"}:
                raise ValueError("serialized retained row is invalid")
            uid = sketch._validate_uid(item["row_uid"])
            if uid in seen_uids:
                raise ValueError("serialized retained row_uid is duplicated")
            seen_uids.add(uid)
            priority_token = item["priority"]
            if (
                not isinstance(priority_token, str)
                or len(priority_token) != 32
                or any(char not in "0123456789abcdef" for char in priority_token)
            ):
                raise ValueError("serialized row priority is invalid")
            priority = int(priority_token, 16)
            if priority != _priority_for_uid(sketch.seed, uid):
                raise ValueError("serialized row priority does not match row_uid")
            value_tokens = item["values"]
            if not isinstance(value_tokens, list) or len(value_tokens) != sketch.width:
                raise ValueError("serialized row width is invalid")
            values = tuple(_float_from_token(token) for token in value_tokens)
            row = _HeapRow(priority, uid, values)
            if previous_key is not None and row.key <= previous_key:
                raise ValueError("serialized retained rows are not canonically ordered")
            previous_key = row.key
            rows.append(row)

        retained_finite: NDArray[np.int64] = np.zeros(
            sketch.width, dtype=np.int64
        )
        for row in rows:
            retained_finite += np.isfinite(np.asarray(row.values)).astype(np.int64)
        retained_missing: NDArray[np.int64] = len(rows) - retained_finite
        if np.any(retained_finite > finite) or np.any(retained_missing > missing):
            raise ValueError("serialized retained rows contradict aggregate counts")

        sketch._total_rows = total_rows
        sketch._finite_counts = finite
        sketch._missing_counts = missing
        sketch._heap = rows
        heapq.heapify(sketch._heap)
        sketch._retained_by_uid = {row.row_uid: row for row in rows}
        return sketch

    @staticmethod
    def _parse_counts(value: object, width: int, name: str) -> np.ndarray:
        if not isinstance(value, list) or len(value) != width:
            raise ValueError(f"{name} must contain exactly width entries")
        parsed: list[int] = []
        for count in value:
            parsed_count = _require_plain_int(count, name)
            if parsed_count < 0:
                raise ValueError(f"{name} entries must be non-negative")
            parsed.append(parsed_count)
        return np.asarray(parsed, dtype=np.int64)


__all__ = ["PriorityRowSketch"]
