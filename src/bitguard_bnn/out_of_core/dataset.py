"""Deterministic, bounded-memory access to verified Parquet training rows.

``ParquetTrainingDataset`` is a worker-facing stream of shuffled row chunks.
Training code must consume it through :func:`iter_ordered_batches`, which restores
the global shard order before forming logical batches.  This keeps the logical
batch sequence identical for zero or many DataLoader workers.
"""

from __future__ import annotations

import hashlib
import heapq
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from bitguard_bnn.out_of_core.prepare import PreparedDataset, verify_prepared_dataset
from bitguard_bnn.out_of_core.shard import load_shard_manifest
from bitguard_bnn.preprocess import FeaturePreprocessor


DATASET_ALGORITHM = "bitguard.deterministic-parquet-dataset.v1"
_PARTITIONS = frozenset({"train", "validation", "test"})


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
class _ShardEntry:
    path: str
    fingerprint: str
    rows: int
    label: str


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
        buffer_rows = int(batch_size) if shuffle_buffer_rows is None else shuffle_buffer_rows
        if isinstance(buffer_rows, bool) or int(buffer_rows) <= 0:
            raise ValueError("shuffle_buffer_rows must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        manifest = load_shard_manifest(prepared.shard_manifest_path)
        root = Path(prepared.shard_manifest_path).resolve().parent
        entries: list[_ShardEntry] = []
        for value in manifest["entries"]:
            if not isinstance(value, Mapping) or value.get("split") != split:
                continue
            relative = str(value["path"])
            pure = PurePosixPath(relative)
            path = root.joinpath(*pure.parts).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise RuntimeError("prepared shard path escapes its output root") from exc
            entries.append(
                _ShardEntry(
                    path=str(path),
                    fingerprint=str(value["sha256"]),
                    rows=int(value["rows"]),
                    label=str(value["label"]),
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

    def _iter_shard_chunks(
        self,
        processor: FeaturePreprocessor,
        entry: _ShardEntry,
        shard_position: int,
    ) -> Iterator[dict[str, Any]]:
        path = Path(entry.path)
        if _is_link_like(path) or not path.is_file():
            raise RuntimeError(f"unsafe prepared shard during iteration: {path}")
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
        with path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RuntimeError(f"prepared shard is not a regular file: {path}")
            if _sha256_handle(handle) != entry.fingerprint:
                raise RuntimeError(
                    f"prepared shard checksum mismatch during iteration: {path}"
                )
            handle.seek(0)
            parquet = pq.ParquetFile(handle)
            batches = iter(
                parquet.iter_batches(
                    batch_size=self.shuffle_buffer_rows,
                    columns=columns,
                    use_threads=False,
                )
            )
            current = next(batches, None)
            chunk_position = 0
            while current is not None:
                following = next(batches, None)
                frame = current.to_pandas()
                order = np.random.Generator(
                    np.random.PCG64(
                        _buffer_seed(
                            self.seed,
                            self.epoch,
                            entry.fingerprint,
                            chunk_position,
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
                    "_shard_position": shard_position,
                    "_chunk_position": chunk_position,
                    "_last_chunk": following is None,
                    "features": encoded,
                    "unencoded": unencoded,
                    "labels": labels,
                    "row_uid": frame["row_uid"].astype(str).to_numpy(copy=True),
                    "metadata": metadata,
                    "boolean_raw": boolean_raw,
                }
                current = following
                chunk_position += 1

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
        for position, entry in enumerate(self.permuted_shards()):
            if position < start_position:
                continue
            if position % worker_count != worker_id:
                continue
            yield from self._iter_shard_chunks(processor, entry, position)


def _ordered_chunks(
    dataset: ParquetTrainingDataset, num_workers: int
) -> Iterator[dict[str, Any]]:
    start_position = _resume_layout(dataset).start_shard_position
    if start_position == len(dataset.entries):
        return
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=_identity,
        persistent_workers=False,
    )
    pending: list[tuple[int, int, int, dict[str, Any]]] = []
    expected = (start_position, 0)
    serial = 0
    for chunk in loader:
        key = (int(chunk["_shard_position"]), int(chunk["_chunk_position"]))
        heapq.heappush(pending, (*key, serial, chunk))
        serial += 1
        while pending and pending[0][:2] == expected:
            shard_position, chunk_position, _, ready = heapq.heappop(pending)
            yield ready
            expected = (
                (shard_position + 1, 0)
                if bool(ready["_last_chunk"])
                else (shard_position, chunk_position + 1)
            )
    while pending and pending[0][:2] == expected:
        shard_position, chunk_position, _, ready = heapq.heappop(pending)
        yield ready
        expected = (
            (shard_position + 1, 0)
            if bool(ready["_last_chunk"])
            else (shard_position, chunk_position + 1)
        )
    if pending or expected != (len(dataset.entries), 0):
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
