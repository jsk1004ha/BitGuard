"""Deterministic, bounded-memory access to verified Parquet training rows.

``ParquetTrainingDataset`` is a worker-facing stream of shuffled row chunks.
Training code must consume it through :func:`iter_ordered_batches`, which restores
the global shard order before forming logical batches.  This keeps the logical
batch sequence identical for zero or many DataLoader workers.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from bitguard_bnn.out_of_core.prepare import PreparedDataset, verify_prepared_dataset
from bitguard_bnn.out_of_core.manifest import stable_fingerprint
from bitguard_bnn.out_of_core.shard import SHARD_MANIFEST_SCHEMA
from bitguard_bnn.preprocess import FeaturePreprocessor


DATASET_ALGORITHM = "bitguard.deterministic-parquet-dataset.v1"
_PARTITIONS = frozenset({"train", "validation", "test"})
_MANIFEST_SEMANTIC_FIELDS = (
    "schema_version",
    "dataset",
    "preprocessing_fingerprint",
    "split_fingerprint",
    "selected_features",
    "materialized_features",
    "boolean_fast_path",
    "counts",
    "class_counts",
    "source_coverage",
    "coverage",
    "shard_contract",
    "schema",
    "schema_fingerprint",
    "algorithm_versions",
    "entries",
    "resource_usage",
)


@dataclass(frozen=True, slots=True)
class DataCursor:
    """Position of the next global logical batch that has not been applied."""

    epoch: int
    shard_position: int
    batch_position: int
    optimizer_step: int

    def __post_init__(self) -> None:
        for name in ("epoch", "shard_position", "batch_position", "optimizer_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"DataCursor.{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class _PinnedFileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class _ShardEntry:
    path: str
    fingerprint: str
    rows: int
    label: str
    row_group_rows: tuple[int, ...] = ()
    identity: _PinnedFileIdentity | None = None


@dataclass(frozen=True, slots=True)
class _RowGroupChunk:
    position: int
    row_groups: tuple[int, ...]
    rows: int


@dataclass(frozen=True, slots=True)
class _BatchSpec:
    shard_position: int
    batch_position: int
    global_start: int
    rows: int


@dataclass(frozen=True, slots=True)
class _BatchLayout:
    """Arithmetic logical-batch layout with constant-size state."""

    row_count: int
    batch_size: int
    allow_singleton: bool

    def __post_init__(self) -> None:
        if self.row_count <= 0:
            raise ValueError("row_count must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least two")
        quotient, remainder = divmod(self.row_count, self.batch_size)
        if remainder == 1 and not self.allow_singleton:
            if quotient == 0:
                raise ValueError("a single-row dataset cannot form a training batch")
            if self.batch_size == 2:
                raise ValueError(
                    "odd row coverage cannot avoid a singleton with batch_size=2"
                )

    @property
    def batch_count(self) -> int:
        quotient, remainder = divmod(self.row_count, self.batch_size)
        return quotient + int(remainder > 0)

    def size_at(self, index: int) -> int:
        if index < 0 or index >= self.batch_count:
            raise IndexError("logical batch index is out of range")
        quotient, remainder = divmod(self.row_count, self.batch_size)
        if remainder != 1:
            return self.batch_size if index < quotient else remainder
        if quotient == 0:
            return 1
        if self.batch_size == 2:
            return self.batch_size if index < quotient else 1
        if index < quotient - 1:
            return self.batch_size
        return self.batch_size - 1 if index == quotient - 1 else 2

    def start_at(self, index: int) -> int:
        if index < 0 or index > self.batch_count:
            raise IndexError("logical batch index is out of range")
        if index == self.batch_count:
            return self.row_count
        quotient, remainder = divmod(self.row_count, self.batch_size)
        if remainder == 1 and quotient > 0 and self.batch_size > 2:
            if index == quotient:
                return quotient * self.batch_size - 1
        return index * self.batch_size

    def first_index_starting_at_or_after(self, row_offset: int) -> int:
        if row_offset < 0 or row_offset > self.row_count:
            raise ValueError("row offset is outside the logical batch layout")
        lower = 0
        upper = self.batch_count
        while lower < upper:
            middle = (lower + upper) // 2
            if self.start_at(middle) < row_offset:
                lower = middle + 1
            else:
                upper = middle
        return lower

    def iter_sizes(self, start_index: int = 0) -> Iterator[int]:
        if start_index < 0 or start_index > self.batch_count:
            raise ValueError("logical batch start index is out of range")
        for index in range(start_index, self.batch_count):
            yield self.size_at(index)


@dataclass(frozen=True, slots=True)
class _ResumeLayout:
    layout: _BatchLayout
    start_index: int
    start_shard_position: int
    skip_rows: int
    start_batch_position: int


def _identity(value: Any) -> Any:
    """Spawn-pickle-safe DataLoader collate function."""

    return value


def _sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _file_identity(value: os.stat_result) -> _PinnedFileIdentity:
    return _PinnedFileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=stat.S_IFMT(value.st_mode),
        size=int(value.st_size),
        modified_ns=int(value.st_mtime_ns),
    )


def _is_reparse_stat(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & flag)


@contextmanager
def _open_pinned_regular(
    path: Path, subject: str
) -> Iterator[tuple[BinaryIO, _PinnedFileIdentity]]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"unable to inspect {subject}: {path}") from exc
    if (
        _is_link_like(path)
        or _is_reparse_stat(before)
        or not stat.S_ISREG(before.st_mode)
    ):
        raise RuntimeError(f"unsafe linked or non-regular {subject}: {path}")
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise RuntimeError(f"unable to open {subject}: {path}") from exc
    try:
        actual = os.fstat(handle.fileno())
        after = os.lstat(path)
        identities = (_file_identity(before), _file_identity(actual), _file_identity(after))
        if (
            not stat.S_ISREG(actual.st_mode)
            or _is_reparse_stat(after)
            or _is_link_like(path)
            or identities[0] != identities[1]
            or identities[1] != identities[2]
        ):
            raise RuntimeError(f"{subject} changed while it was opened: {path}")
        yield handle, identities[1]
    finally:
        handle.close()


def _load_pinned_manifest(prepared: PreparedDataset) -> dict[str, Any]:
    path = Path(prepared.shard_manifest_path)
    root = Path(prepared.output_dir)
    if (
        _is_link_like(root)
        or not root.is_dir()
        or Path(os.path.abspath(path.parent)) != Path(os.path.abspath(root))
    ):
        raise RuntimeError("unsafe prepared shard manifest root")
    with _open_pinned_regular(path, "prepared shard manifest") as (handle, _):
        encoded = handle.read()
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("unable to decode prepared shard manifest") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SHARD_MANIFEST_SCHEMA:
        raise RuntimeError("unsupported prepared shard manifest schema")
    try:
        semantics = {name: payload[name] for name in _MANIFEST_SEMANTIC_FIELDS}
        actual = stable_fingerprint(semantics)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid prepared shard manifest semantics") from exc
    if (
        payload.get("fingerprint") != actual
        or actual != prepared.shard_fingerprint
    ):
        raise RuntimeError("prepared shard manifest fingerprint mismatch")
    return payload


def _inspect_verified_shard(
    path: Path, expected_sha256: str
) -> tuple[_PinnedFileIdentity, tuple[int, ...]]:
    with _open_pinned_regular(path, "prepared shard") as (handle, identity):
        if _sha256_handle(handle) != expected_sha256:
            raise RuntimeError(f"prepared shard checksum mismatch: {path}")
        handle.seek(0)
        parquet = pq.ParquetFile(handle)
        row_groups = tuple(
            int(parquet.metadata.row_group(index).num_rows)
            for index in range(parquet.metadata.num_row_groups)
        )
    if not row_groups or any(rows <= 0 for rows in row_groups):
        raise RuntimeError(f"prepared shard has invalid row groups: {path}")
    return identity, row_groups


def _iter_row_group_chunks(
    entry: _ShardEntry, buffer_rows: int
) -> Iterator[_RowGroupChunk]:
    group: list[int] = []
    rows = 0
    position = 0
    for index, group_rows in enumerate(entry.row_group_rows):
        if group_rows > buffer_rows:
            raise RuntimeError("prepared row group exceeds shuffle_buffer_rows")
        if group and rows + group_rows > buffer_rows:
            yield _RowGroupChunk(position, tuple(group), rows)
            position += 1
            group = []
            rows = 0
        group.append(index)
        rows += group_rows
    if group:
        yield _RowGroupChunk(position, tuple(group), rows)


def _row_group_chunk_count(entry: _ShardEntry, buffer_rows: int) -> int:
    return sum(1 for _ in _iter_row_group_chunks(entry, buffer_rows))


def _buffer_seed(seed: int, epoch: int, fingerprint: str, index: int) -> int:
    material = f"{seed}\0{epoch}\0{fingerprint}\0{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "little")


def _selected_unencoded(
    processor: FeaturePreprocessor, frame: pd.DataFrame
) -> np.ndarray:
    """Apply the frozen preprocessor using only its materialized selected inputs."""

    if not processor.fitted:
        raise RuntimeError("frozen preprocessor is not fitted")
    selected = list(processor.selected_features)
    candidate_positions = {name: index for index, name in enumerate(processor.candidate_features)}
    try:
        positions = np.asarray(
            [candidate_positions[name] for name in selected], dtype=np.int64
        )
    except KeyError as exc:
        raise RuntimeError("selected feature is absent from frozen preprocessing state") from exc
    raw = (
        frame[selected]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32, copy=True)
    )
    statistics = np.asarray(processor.imputer.statistics_, dtype=np.float32)[positions]
    missing = np.isnan(raw)
    if missing.any():
        raw[missing] = np.broadcast_to(statistics, raw.shape)[missing]
    return processor.scaler.transform(raw).astype(np.float32, copy=False)


class ParquetTrainingDataset(torch.utils.data.IterableDataset[dict[str, Any]]):
    """Spawn-safe worker dataset backed only by immutable path metadata."""

    def __init__(
        self,
        descriptor_path: Path | str,
        *,
        split: str = "train",
        batch_size: int,
        seed: int,
        shuffle_buffer_rows: int | None = None,
    ) -> None:
        super().__init__()
        # Verification deliberately precedes all manifest-derived construction.
        prepared = verify_prepared_dataset(descriptor_path)
        self._initialize(
            prepared,
            split=split,
            batch_size=batch_size,
            seed=seed,
            shuffle_buffer_rows=shuffle_buffer_rows,
        )

    def _initialize(
        self,
        prepared: PreparedDataset,
        *,
        split: str,
        batch_size: int,
        seed: int,
        shuffle_buffer_rows: int | None,
    ) -> None:
        if split not in _PARTITIONS:
            raise ValueError(f"unsupported prepared split: {split}")
        if isinstance(batch_size, bool) or int(batch_size) < 2:
            raise ValueError("batch_size must be at least two")
        if (
            shuffle_buffer_rows is not None
            and (
                isinstance(shuffle_buffer_rows, bool)
                or int(shuffle_buffer_rows) <= 0
            )
        ):
            raise ValueError("shuffle_buffer_rows must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        manifest = _load_pinned_manifest(prepared)
        try:
            record_batch_rows = int(manifest["shard_contract"]["record_batch_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid prepared shard record-batch contract") from exc
        if record_batch_rows <= 0:
            raise RuntimeError("invalid prepared shard record-batch contract")
        buffer_rows = (
            max(int(batch_size), record_batch_rows)
            if shuffle_buffer_rows is None
            else int(shuffle_buffer_rows)
        )
        root = Path(prepared.output_dir)
        if _is_link_like(root) or not root.is_dir():
            raise RuntimeError("unsafe prepared shard output root")
        entries: list[_ShardEntry] = []
        for value in manifest["entries"]:
            if not isinstance(value, Mapping) or value.get("split") != split:
                continue
            relative = str(value["path"])
            pure = PurePosixPath(relative)
            if (
                not relative
                or "\\" in relative
                or pure.is_absolute()
                or pure.as_posix() != relative
                or any(part in {"", ".", ".."} for part in pure.parts)
            ):
                raise RuntimeError("non-canonical prepared shard path")
            candidate = root.joinpath(*pure.parts)
            current = root
            for part in pure.parts[:-1]:
                current /= part
                if _is_link_like(current) or not current.is_dir():
                    raise RuntimeError("unsafe prepared shard parent path")
            path = candidate.resolve(strict=True)
            try:
                path.relative_to(root.resolve(strict=True))
            except ValueError as exc:
                raise RuntimeError("prepared shard path escapes its output root") from exc
            identity, row_group_rows = _inspect_verified_shard(
                path, str(value["sha256"])
            )
            if sum(row_group_rows) != int(value["rows"]):
                raise RuntimeError("prepared shard row-group coverage mismatch")
            if max(row_group_rows) > buffer_rows:
                raise ValueError(
                    "shuffle_buffer_rows must be at least the largest prepared row group"
                )
            entries.append(
                _ShardEntry(
                    path=str(path),
                    fingerprint=str(value["sha256"]),
                    rows=int(value["rows"]),
                    label=str(value["label"]),
                    row_group_rows=row_group_rows,
                    identity=identity,
                )
            )
        expected = int(getattr(prepared, f"{split}_count"))
        if not entries or sum(entry.rows for entry in entries) != expected:
            raise RuntimeError("prepared shard entries do not cover the requested split")
        if split == "train" and expected < 2:
            raise ValueError("streaming training requires at least two rows")
        _BatchLayout(
            row_count=expected,
            batch_size=int(batch_size),
            allow_singleton=split != "train",
        )

        self.descriptor_path = prepared.descriptor_path
        self.preprocessor_path = prepared.preprocessor_path
        self.preprocessor_sha256 = prepared.preprocessor_sha256
        self.manifest_fingerprint = prepared.shard_fingerprint
        self.preprocessing_fingerprint = prepared.preprocessing_fingerprint
        self.split = split
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_buffer_rows = int(buffer_rows)
        self.entries = tuple(entries)
        self.selected_features = tuple(str(value) for value in manifest["selected_features"])
        self.materialized_features = tuple(
            str(value) for value in manifest["materialized_features"]
        )
        self.boolean_features = tuple(
            str(value) for value in manifest["boolean_fast_path"]["available_features"]
        )
        self.row_count = expected
        self.epoch = 0
        self.cursor: DataCursor | None = None
        self._max_pending_chunks_observed = 0
        self._worker_ids_observed: set[int] = set()

    @property
    def max_pending_chunks_observed(self) -> int:
        return self._max_pending_chunks_observed

    @property
    def worker_ids_observed(self) -> tuple[int, ...]:
        return tuple(sorted(self._worker_ids_observed))

    def set_epoch(self, epoch: int, cursor: DataCursor | None = None) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if cursor is not None and cursor.epoch != epoch:
            raise ValueError("resume cursor epoch does not match dataset epoch")
        self.epoch = int(epoch)
        self.cursor = cursor

    def permuted_shards(self, epoch: int | None = None) -> tuple[_ShardEntry, ...]:
        """Return a deterministic, class-interleaved PCG64 shard order."""

        selected_epoch = self.epoch if epoch is None else int(epoch)
        generator = np.random.Generator(np.random.PCG64(self.seed + selected_epoch))
        labels = sorted({entry.label for entry in self.entries})
        label_order = [labels[index] for index in generator.permutation(len(labels))]
        groups: dict[str, list[_ShardEntry]] = {}
        for label in label_order:
            values = [entry for entry in self.entries if entry.label == label]
            groups[label] = [values[index] for index in generator.permutation(len(values))]
        ordered: list[_ShardEntry] = []
        position = 0
        while len(ordered) < len(self.entries):
            for label in label_order:
                values = groups[label]
                if position < len(values):
                    ordered.append(values[position])
            position += 1
        return tuple(ordered)

    def _iter_assigned_shard_chunks(
        self,
        processor: FeaturePreprocessor,
        entry: _ShardEntry,
        shard_position: int,
        base_ordinal: int,
        worker_id: int,
        worker_count: int,
    ) -> Iterator[dict[str, Any]]:
        chunk_count = _row_group_chunk_count(entry, self.shuffle_buffer_rows)
        first_owned = (worker_id - (base_ordinal % worker_count)) % worker_count
        if first_owned >= chunk_count:
            return
        path = Path(entry.path)
        metadata_columns = (
            "row_uid",
            "source_file",
            "sequence_index",
            "device_id",
            "raw_attack",
            "behavior_label",
            "timestamp",
        )
        columns = [*metadata_columns, *self.materialized_features]
        with _open_pinned_regular(path, "prepared shard") as (handle, identity):
            if entry.identity is None or identity != entry.identity:
                raise RuntimeError(
                    f"prepared shard identity changed during iteration: {path}"
                )
            parquet = pq.ParquetFile(handle)
            for chunk in _iter_row_group_chunks(entry, self.shuffle_buffer_rows):
                ordinal = base_ordinal + chunk.position
                if ordinal % worker_count != worker_id:
                    continue
                table = parquet.read_row_groups(
                    list(chunk.row_groups),
                    columns=columns,
                    use_threads=False,
                )
                if table.num_rows != chunk.rows:
                    raise RuntimeError("prepared row-group chunk coverage mismatch")
                frame = table.to_pandas()
                order = np.random.Generator(
                    np.random.PCG64(
                        _buffer_seed(
                            self.seed,
                            self.epoch,
                            entry.fingerprint,
                            chunk.position,
                        )
                    )
                ).permutation(len(frame))
                frame = frame.iloc[order].reset_index(drop=True)
                unencoded = _selected_unencoded(processor, frame)
                encoded = processor.encoder.transform(unencoded).astype(
                    np.float32, copy=False
                )
                labels = processor.encode_labels(frame)
                metadata = {
                    name: frame[name].to_numpy(copy=True) for name in metadata_columns[1:]
                }
                boolean_raw = {
                    name: frame[name].to_numpy(dtype=np.float32, copy=True)
                    for name in self.boolean_features
                }
                yield {
                    "_chunk_ordinal": ordinal,
                    "_worker_id": worker_id,
                    "_shard_position": shard_position,
                    "_chunk_position": chunk.position,
                    "_last_chunk": chunk.position == chunk_count - 1,
                    "features": encoded,
                    "unencoded": unencoded,
                    "labels": labels,
                    "row_uid": frame["row_uid"].astype(str).to_numpy(copy=True),
                    "metadata": metadata,
                    "boolean_raw": boolean_raw,
                }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        # joblib opens and closes the artifact inside this worker process.
        preprocessor_path = Path(self.preprocessor_path)
        if _is_link_like(preprocessor_path) or not preprocessor_path.is_file():
            raise RuntimeError("prepared preprocessor checksum mismatch during iteration")
        with preprocessor_path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RuntimeError(
                    "prepared preprocessor is not a regular file during iteration"
                )
            if _sha256_handle(handle) != self.preprocessor_sha256:
                raise RuntimeError(
                    "prepared preprocessor checksum mismatch during iteration"
                )
            handle.seek(0)
            processor = joblib.load(handle)
        if not isinstance(processor, FeaturePreprocessor):
            raise TypeError("artifact is not a FeaturePreprocessor")
        if tuple(processor.selected_features) != self.selected_features:
            raise RuntimeError("frozen preprocessor no longer matches shard features")
        resume = _resume_layout(self)
        start_position = resume.start_shard_position
        ordinal = 0
        for position, entry in enumerate(self.permuted_shards()):
            if position < start_position:
                continue
            chunk_count = _row_group_chunk_count(entry, self.shuffle_buffer_rows)
            yield from self._iter_assigned_shard_chunks(
                processor,
                entry,
                position,
                ordinal,
                worker_id,
                worker_count,
            )
            ordinal += chunk_count


def _total_chunk_count(
    dataset: ParquetTrainingDataset, start_position: int
) -> int:
    return sum(
        _row_group_chunk_count(entry, dataset.shuffle_buffer_rows)
        for entry in dataset.permuted_shards()[start_position:]
    )


def _ordered_chunks(
    dataset: ParquetTrainingDataset, num_workers: int
) -> Iterator[dict[str, Any]]:
    start_position = _resume_layout(dataset).start_shard_position
    if start_position == len(dataset.entries):
        return
    prefetch_factor = 2
    loader_options: dict[str, Any] = {
        "batch_size": None,
        "num_workers": num_workers,
        "collate_fn": _identity,
        "persistent_workers": False,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor
    loader = torch.utils.data.DataLoader(dataset, **loader_options)
    pending: list[tuple[int, int, dict[str, Any]]] = []
    expected = 0
    expected_total = _total_chunk_count(dataset, start_position)
    pending_limit = 1 if num_workers == 0 else num_workers * prefetch_factor
    serial = 0
    for chunk in loader:
        ordinal = int(chunk["_chunk_ordinal"])
        dataset._worker_ids_observed.add(int(chunk["_worker_id"]))
        if ordinal < expected:
            raise RuntimeError("worker stream duplicated an ordered row-group chunk")
        heapq.heappush(pending, (ordinal, serial, chunk))
        serial += 1
        dataset._max_pending_chunks_observed = max(
            dataset._max_pending_chunks_observed,
            len(pending),
        )
        if len(pending) > pending_limit:
            raise RuntimeError("worker reorder payloads exceeded the prefetch bound")
        while pending and pending[0][0] == expected:
            _, _, ready = heapq.heappop(pending)
            yield ready
            expected += 1
    while pending and pending[0][0] == expected:
        _, _, ready = heapq.heappop(pending)
        yield ready
        expected += 1
    if pending or expected != expected_total:
        raise RuntimeError("worker stream omitted or duplicated an ordered shard chunk")


def _logical_batch_sizes(
    row_count: int, batch_size: int, *, allow_singleton: bool
) -> Iterator[int]:
    return _BatchLayout(
        row_count=row_count,
        batch_size=batch_size,
        allow_singleton=allow_singleton,
    ).iter_sizes()


def _batch_layout(dataset: ParquetTrainingDataset) -> _BatchLayout:
    return _BatchLayout(
        dataset.row_count,
        dataset.batch_size,
        allow_singleton=dataset.split != "train",
    )


def _resume_layout(
    dataset: ParquetTrainingDataset,
) -> _ResumeLayout:
    layout = _batch_layout(dataset)
    entries = dataset.permuted_shards()
    if dataset.cursor is None:
        return _ResumeLayout(layout, 0, 0, 0, 0)
    target = (dataset.cursor.shard_position, dataset.cursor.batch_position)
    if target == (len(dataset.entries), 0):
        return _ResumeLayout(
            layout,
            layout.batch_count,
            len(entries),
            0,
            0,
        )
    shard_position = dataset.cursor.shard_position
    if shard_position < 0 or shard_position >= len(entries):
        raise ValueError("resume cursor does not identify a logical batch boundary")
    preceding = sum(entry.rows for entry in entries[:shard_position])
    shard_end = preceding + entries[shard_position].rows
    first_index = layout.first_index_starting_at_or_after(preceding)
    start_index = first_index + dataset.cursor.batch_position
    if start_index >= layout.batch_count:
        raise ValueError("resume cursor does not identify a logical batch boundary")
    global_start = layout.start_at(start_index)
    if global_start < preceding or global_start >= shard_end:
        raise ValueError("resume cursor does not identify a logical batch boundary")
    return _ResumeLayout(
        layout,
        start_index,
        shard_position,
        global_start - preceding,
        dataset.cursor.batch_position,
    )


def _iter_batch_specs(
    dataset: ParquetTrainingDataset,
    resume: _ResumeLayout,
) -> Iterator[_BatchSpec]:
    entries = dataset.permuted_shards()
    shard_position = resume.start_shard_position
    preceding = sum(entry.rows for entry in entries[:shard_position])
    shard_end = (
        dataset.row_count
        if shard_position == len(entries)
        else preceding + entries[shard_position].rows
    )
    batch_position = resume.start_batch_position
    for index in range(resume.start_index, resume.layout.batch_count):
        global_start = resume.layout.start_at(index)
        while global_start >= shard_end:
            preceding = shard_end
            shard_position += 1
            if shard_position >= len(entries):
                raise RuntimeError("logical batch start exceeds prepared shard coverage")
            shard_end = preceding + entries[shard_position].rows
            batch_position = 0
        yield _BatchSpec(
            shard_position=shard_position,
            batch_position=batch_position,
            global_start=global_start,
            rows=resume.layout.size_at(index),
        )
        batch_position += 1


def _take_rows(
    chunks: Iterator[dict[str, Any]],
    state: list[Any],
    count: int,
) -> tuple[int, dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    remaining = count
    start_shard = -1
    while remaining:
        if state[0] is None:
            state[0] = next(chunks)
            state[1] = 0
        chunk = state[0]
        offset = int(state[1])
        available = len(chunk["row_uid"]) - offset
        take = min(remaining, available)
        if start_shard < 0:
            start_shard = int(chunk["_shard_position"])
        stop = offset + take
        parts.append(
            {
                "features": chunk["features"][offset:stop],
                "unencoded": chunk["unencoded"][offset:stop],
                "labels": chunk["labels"][offset:stop],
                "row_uid": chunk["row_uid"][offset:stop],
                "metadata": {
                    name: values[offset:stop]
                    for name, values in chunk["metadata"].items()
                },
                "boolean_raw": {
                    name: values[offset:stop]
                    for name, values in chunk["boolean_raw"].items()
                },
            }
        )
        remaining -= take
        state[1] = stop
        if stop == len(chunk["row_uid"]):
            state[0] = None
            state[1] = 0
    return start_shard, _concatenate_parts(parts)


def _concatenate_parts(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([part[name] for part in parts], axis=0)

    metadata_names = tuple(parts[0]["metadata"])
    boolean_names = tuple(parts[0]["boolean_raw"])
    return {
        "features": concatenate("features"),
        "unencoded": concatenate("unencoded"),
        "labels": concatenate("labels"),
        "row_uid": concatenate("row_uid"),
        "metadata": {
            name: np.concatenate([part["metadata"][name] for part in parts])
            for name in metadata_names
        },
        "boolean_raw": {
            name: np.concatenate([part["boolean_raw"][name] for part in parts])
            for name in boolean_names
        },
    }


def _logical_batches(
    dataset: ParquetTrainingDataset, num_workers: int
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    resume = _resume_layout(dataset)
    if resume.start_index == resume.layout.batch_count:
        return
    chunks = iter(_ordered_chunks(dataset, num_workers))
    state: list[Any] = [None, 0]
    if resume.skip_rows:
        _take_rows(chunks, state, resume.skip_rows)
    for spec in _iter_batch_specs(dataset, resume):
        shard_position, batch = _take_rows(chunks, state, spec.rows)
        if shard_position != spec.shard_position:
            raise RuntimeError("logical batch start does not match its manifest plan")
        yield spec.shard_position, spec.batch_position, batch
    if state[0] is not None or next(chunks, None) is not None:
        raise RuntimeError("logical batching did not consume exact prepared row coverage")


def iter_ordered_batches(
    dataset: ParquetTrainingDataset,
    *,
    num_workers: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield the sole training-facing, globally ordered logical batch stream."""

    if isinstance(num_workers, bool) or int(num_workers) < 0:
        raise ValueError("num_workers must be a non-negative integer")
    requested = dataset.cursor
    optimizer_step = 0 if requested is None else requested.optimizer_step
    source = iter(_logical_batches(dataset, int(num_workers)))
    current = next(source, None)
    while current is not None:
        following = next(source, None)
        shard_position, batch_position, batch = current
        cursor = DataCursor(
            epoch=dataset.epoch,
            shard_position=shard_position,
            batch_position=batch_position,
            optimizer_step=optimizer_step,
        )
        if following is None:
            next_cursor = DataCursor(
                epoch=dataset.epoch,
                shard_position=len(dataset.entries),
                batch_position=0,
                optimizer_step=optimizer_step + 1,
            )
        else:
            next_cursor = DataCursor(
                epoch=dataset.epoch,
                shard_position=following[0],
                batch_position=following[1],
                optimizer_step=optimizer_step + 1,
            )
        yield {**batch, "cursor": cursor, "next_cursor": next_cursor}
        optimizer_step += 1
        current = following


__all__ = [
    "DATASET_ALGORITHM",
    "DataCursor",
    "ParquetTrainingDataset",
    "iter_ordered_batches",
]
