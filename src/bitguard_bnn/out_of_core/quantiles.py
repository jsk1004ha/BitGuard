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
LIMITS_VERSION = 1
# These limits cover the full profiles (capacity 200,000 and up to 115 current
# source features) while bounding allocations from untrusted state. Canonical
# JSON float tokens for that 23-million-value profile fit below 768 MiB.
MAX_SKETCH_WIDTH = 4_096
MAX_SKETCH_CAPACITY = 1_000_000
MAX_RETAINED_ROWS = 1_000_000
MAX_RETAINED_VALUES = 32_000_000
MAX_SERIALIZED_BYTES = 768 * 1024**2
MAX_COUNT = int(np.iinfo(np.int64).max)
MIN_SEED = -(2**63)
MAX_SEED = 2**63 - 1
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


def _require_plain_int(
    value: object,
    name: str,
    *,
    positive: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return value


def _validate_shape_product(capacity: int, width: int) -> None:
    if capacity * width > MAX_RETAINED_VALUES:
        raise ValueError(
            "capacity * width exceeds the "
            f"{MAX_RETAINED_VALUES} retained values limit"
        )


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
        self._capacity = _require_plain_int(
            capacity,
            "capacity",
            positive=True,
            maximum=MAX_SKETCH_CAPACITY,
        )
        self._seed = _require_plain_int(
            seed,
            "seed",
            minimum=MIN_SEED,
            maximum=MAX_SEED,
        )
        self._width = _require_plain_int(
            width,
            "width",
            positive=True,
            maximum=MAX_SKETCH_WIDTH,
        )
        _validate_shape_product(self._capacity, self._width)
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

    def retained_rows(self) -> tuple[tuple[str, NDArray[np.float64]], ...]:
        """Return canonical copies of retained complete rows for bounded consumers."""

        result: list[tuple[str, NDArray[np.float64]]] = []
        for row in sorted(self._heap, key=lambda item: item.key):
            values = np.asarray(row.values, dtype=np.float64)
            values.flags.writeable = False
            result.append((row.row_uid, values))
        return tuple(result)

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

    def _retention_action(self, row: _HeapRow) -> tuple[str, _HeapRow | None]:
        if len(self._heap) < self.capacity:
            return "append", None
        if row.key >= self._heap[0].key:
            return "discard", None
        return "replace", self._heap[0]

    def _apply_retention(
        self,
        row: _HeapRow,
        action: tuple[str, _HeapRow | None],
    ) -> None:
        operation, expected_evicted = action
        if operation == "append":
            heapq.heappush(self._heap, row)
            self._retained_by_uid[row.row_uid] = row
            return
        if operation == "discard":
            return
        evicted = heapq.heapreplace(self._heap, row)
        if evicted is not expected_evicted:
            raise RuntimeError("priority sketch heap changed during update")
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

        priority = _priority_for_uid(self.seed, uid)
        prospective_total = self._total_rows + 1
        if prospective_total > MAX_COUNT:
            raise OverflowError(f"total_rows exceeds the {MAX_COUNT} counter limit")
        finite = np.isfinite(np.asarray(row_values, dtype=np.float64))
        # A valid column count never exceeds total_rows. Checking the
        # prospective total first therefore proves these int64 additions safe.
        next_finite = self._finite_counts + finite.astype(np.int64)
        next_missing = self._missing_counts + (~finite).astype(np.int64)
        row = _HeapRow(priority, uid, row_values)
        retention_action = self._retention_action(row)

        # Priority calculation, counter bounds, arrays, and the heap decision
        # are complete before any observable state is mutated.
        self._apply_retention(row, retention_action)
        self._total_rows = prospective_total
        self._finite_counts = next_finite
        self._missing_counts = next_missing

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

        overlap_finite = [0] * self.width
        for uid in overlap:
            for column, value in enumerate(self._retained_by_uid[uid].values):
                overlap_finite[column] += int(math.isfinite(value))
        overlap_missing = [len(overlap) - count for count in overlap_finite]

        prospective_total = self._total_rows + other._total_rows - len(overlap)
        if not 0 <= prospective_total <= MAX_COUNT:
            raise OverflowError(f"merged total_rows exceeds the {MAX_COUNT} counter limit")
        prospective_finite: list[int] = []
        prospective_missing: list[int] = []
        for column in range(self.width):
            finite_count = (
                int(self._finite_counts[column])
                + int(other._finite_counts[column])
                - overlap_finite[column]
            )
            missing_count = (
                int(self._missing_counts[column])
                + int(other._missing_counts[column])
                - overlap_missing[column]
            )
            if (
                not 0 <= finite_count <= MAX_COUNT
                or not 0 <= missing_count <= MAX_COUNT
                or finite_count + missing_count != prospective_total
            ):
                raise OverflowError("merged column counts overflow or contradict total_rows")
            prospective_finite.append(finite_count)
            prospective_missing.append(missing_count)
        next_finite = np.asarray(prospective_finite, dtype=np.int64)
        next_missing = np.asarray(prospective_missing, dtype=np.int64)
        next_heap = list(retained)
        heapq.heapify(next_heap)
        next_by_uid = {row.row_uid: row for row in retained}

        # Commit only after compatibility and overlap validation have succeeded.
        self._total_rows = prospective_total
        self._finite_counts = next_finite
        self._missing_counts = next_missing
        self._heap = next_heap
        self._retained_by_uid = next_by_uid
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
        serialized = _canonical_json(envelope)
        if len(serialized) > MAX_SERIALIZED_BYTES:
            raise ValueError(
                f"serialized sketch size exceeds {MAX_SERIALIZED_BYTES} bytes"
            )
        return serialized

    @classmethod
    def from_bytes(cls, serialized: bytes | bytearray | memoryview) -> "PriorityRowSketch":
        if isinstance(serialized, memoryview):
            serialized_size = serialized.nbytes
        elif isinstance(serialized, (bytes, bytearray)):
            serialized_size = len(serialized)
        else:
            raise ValueError("serialized sketch must be bytes-like")
        if serialized_size > MAX_SERIALIZED_BYTES:
            raise ValueError(
                f"serialized sketch size exceeds {MAX_SERIALIZED_BYTES} bytes"
            )
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
        (
            capacity,
            seed,
            width,
            total_rows,
            finite_values,
            missing_values,
            retained_payload,
        ) = cls._preflight_snapshot(payload)
        sketch = cls(capacity=capacity, seed=seed, width=width)
        finite = np.asarray(finite_values, dtype=np.int64)
        missing = np.asarray(missing_values, dtype=np.int64)

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

    @classmethod
    def _preflight_snapshot(
        cls,
        payload: Mapping[str, Any],
    ) -> tuple[int, int, int, int, list[int], list[int], list[Any]]:
        """Validate all allocation-driving state using Python scalars only."""

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
        if not isinstance(payload["algorithm"], str) or payload["algorithm"] != ALGORITHM:
            raise ValueError("unsupported sketch algorithm")
        version = _require_plain_int(payload["version"], "version")
        if version != VERSION:
            raise ValueError("unsupported sketch version")
        if payload["deduplication"] != cls._deduplication_contract():
            raise ValueError("serialized deduplication contract is invalid")

        capacity = _require_plain_int(
            payload["capacity"],
            "capacity",
            positive=True,
            maximum=MAX_SKETCH_CAPACITY,
        )
        seed = _require_plain_int(
            payload["seed"],
            "seed",
            minimum=MIN_SEED,
            maximum=MAX_SEED,
        )
        width = _require_plain_int(
            payload["width"],
            "width",
            positive=True,
            maximum=MAX_SKETCH_WIDTH,
        )
        _validate_shape_product(capacity, width)
        total_rows = _require_plain_int(
            payload["total_rows"],
            "total_rows",
            minimum=0,
            maximum=MAX_COUNT,
        )
        finite = cls._parse_count_values(payload["finite_counts"], width, "finite_counts")
        missing = cls._parse_count_values(
            payload["missing_counts"], width, "missing_counts"
        )
        if any(
            finite_count + missing_count != total_rows
            for finite_count, missing_count in zip(finite, missing)
        ):
            raise ValueError("serialized counts do not match total_rows")

        retained_payload = payload["retained_rows"]
        if not isinstance(retained_payload, list):
            raise ValueError("retained_rows must be a list")
        if len(retained_payload) > MAX_RETAINED_ROWS:
            raise ValueError(
                f"retained row count exceeds the {MAX_RETAINED_ROWS} limit"
            )
        expected_retained = min(total_rows, capacity)
        if len(retained_payload) != expected_retained:
            raise ValueError("serialized retained row count is invalid")
        return capacity, seed, width, total_rows, finite, missing, retained_payload

    @staticmethod
    def _parse_count_values(value: object, width: int, name: str) -> list[int]:
        if not isinstance(value, list) or len(value) != width:
            raise ValueError(f"{name} must contain exactly width entries")
        parsed: list[int] = []
        for count in value:
            parsed_count = _require_plain_int(
                count,
                name,
                minimum=0,
                maximum=MAX_COUNT,
            )
            parsed.append(parsed_count)
        return parsed


__all__ = [
    "LIMITS_VERSION",
    "MAX_COUNT",
    "MAX_RETAINED_ROWS",
    "MAX_RETAINED_VALUES",
    "MAX_SEED",
    "MAX_SERIALIZED_BYTES",
    "MAX_SKETCH_CAPACITY",
    "MAX_SKETCH_WIDTH",
    "MIN_SEED",
    "PriorityRowSketch",
]
