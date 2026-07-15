"""Verified, restart-safe orchestration for complete out-of-core datasets."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from bitguard_bnn.bootstrap.inspect import inspect_csv_dataset
from bitguard_bnn.bootstrap.manifest import SourceManifest, build_source_manifest
from bitguard_bnn.bootstrap.registry import load_registry
from bitguard_bnn.config import load_config, resolve_path
from bitguard_bnn.out_of_core.manifest import (
    SplitPlan,
    canonical_json_bytes,
    manifest_path_for_membership,
    read_split_manifest,
    stable_fingerprint,
    write_json_atomic,
)
from bitguard_bnn.out_of_core.preprocess import StreamingFeaturePreprocessor
from bitguard_bnn.out_of_core.shard import (
    verify_shard_manifest,
    write_parquet_shards,
)
from bitguard_bnn.out_of_core.split import build_split_plan
from bitguard_bnn.out_of_core.source import (
    NormalizedSource,
    NormalizedSourceProof,
    open_normalized_source,
)
from bitguard_bnn.preprocess import FeaturePreprocessor


PREPARED_DATASET_SCHEMA = "bitguard.prepared-dataset.v1"
FEATURE_ARTIFACT_SCHEMA = "bitguard.streaming-feature-artifact.v1"
PREPARATION_ALGORITHM = "bitguard.full-dataset-preparation.v1"

_PARTITIONS = ("train", "validation", "test")
_MEMBERSHIP_COLUMNS = ("row_uid", "split", "behavior_label")


@dataclass(frozen=True, slots=True)
class PreparationDiskEstimate:
    source_snapshot_bytes: int
    membership_sqlite_bytes: int
    audit_sqlite_bytes: int
    external_merge_bytes: int
    staging_bytes: int
    final_shard_bytes: int

    @property
    def total_bytes(self) -> int:
        return sum(asdict(self).values())

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "total_bytes": self.total_bytes}


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    descriptor_path: str
    dataset: str
    resolved_config_path: str
    config_sha256: str
    raw_root: str
    source_manifest_path: str
    source_manifest_fingerprint: str
    schema_report_path: str
    schema_report_fingerprint: str
    normalized_source_fingerprint: str
    split_membership_path: str
    split_membership_sha256: str
    split_manifest_path: str
    split_fingerprint: str
    preprocessor_path: str
    preprocessor_sha256: str
    feature_manifest_path: str
    preprocessing_fingerprint: str
    shard_manifest_path: str
    shard_fingerprint: str
    train_count: int
    validation_count: int
    test_count: int
    total_count: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("descriptor_path")
        payload["schema_version"] = PREPARED_DATASET_SCHEMA
        payload["algorithm"] = PREPARATION_ALGORITHM
        payload["fingerprint"] = stable_fingerprint(payload)
        return payload

    @classmethod
    def from_dict(cls, path: Path, payload: Mapping[str, object]) -> PreparedDataset:
        semantic = dict(payload)
        fingerprint = semantic.pop("fingerprint", None)
        if (
            semantic.get("schema_version") != PREPARED_DATASET_SCHEMA
            or semantic.get("algorithm") != PREPARATION_ALGORITHM
            or fingerprint != stable_fingerprint(semantic)
        ):
            raise RuntimeError("prepared dataset descriptor fingerprint mismatch")
        semantic.pop("schema_version")
        semantic.pop("algorithm")
        expected = {
            field for field in cls.__dataclass_fields__ if field != "descriptor_path"
        }
        if set(semantic) != expected:
            raise RuntimeError("prepared dataset descriptor fields are invalid")
        string_fields = expected - {
            "train_count",
            "validation_count",
            "test_count",
            "total_count",
        }
        if any(
            not isinstance(semantic[name], str) or not str(semantic[name])
            for name in string_fields
        ):
            raise RuntimeError("prepared dataset descriptor contains an invalid string")
        counts = {name: semantic[name] for name in expected - string_fields}
        if any(type(value) is not int or int(value) <= 0 for value in counts.values()):
            raise RuntimeError("prepared dataset descriptor contains an invalid count")
        return cls(descriptor_path=str(path.resolve()), **semantic)  # type: ignore[arg-type]


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, subject: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read {subject}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{subject} root must be an object: {path}")
    return value


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
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


def _publish_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"immutable JSON artifact conflict: {path}")
        return
    write_json_atomic(path, payload)


def _clean_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in config.items() if not name.startswith("_")}


def _validate_full_config(config: Mapping[str, Any]) -> None:
    dataset = config["dataset"]
    if dataset.get("storage") != "parquet":
        raise ValueError("full preparation requires dataset.storage=parquet")
    for name in ("max_rows_per_file", "max_rows_per_class", "max_loaded_rows"):
        if dataset.get(name) is not None:
            raise ValueError(f"full preparation requires dataset.{name}=null")


def _load_source_contract(
    source_manifest_path: Path,
    schema_report_path: Path,
) -> tuple[SourceManifest, dict[str, Any], str]:
    try:
        source_manifest = SourceManifest.from_dict(
            _read_json(source_manifest_path, "source manifest")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("source manifest validation failed") from exc
    schema = _read_json(schema_report_path, "schema report")
    schema_fingerprint = stable_fingerprint(schema)
    return source_manifest, schema, schema_fingerprint


def _verify_source_proof(
    proof: NormalizedSourceProof,
    source_manifest: SourceManifest,
    schema: Mapping[str, Any],
) -> None:
    if proof.dataset != source_manifest.dataset_name:
        raise RuntimeError("normalized source dataset does not match source manifest")
    if schema.get("dataset") != proof.dataset:
        raise RuntimeError("schema report dataset does not match normalized source")
    if int(schema.get("rejected_rows", -1)) != 0:
        raise RuntimeError("schema report contains rejected rows")
    if int(schema.get("accepted_rows", -1)) != proof.row_count:
        raise RuntimeError("schema report row count does not match normalized source")
    if tuple(str(name) for name in schema.get("feature_columns", ())) != proof.feature_names:
        raise RuntimeError("schema feature order does not match normalized source")

    manifest_files = {record.relative_path: record for record in source_manifest.files}
    schema_files = {
        str(value["relative_path"]): value
        for value in schema.get("files", ())
        if isinstance(value, Mapping) and "relative_path" in value
    }
    if len(schema_files) != len(proof.files):
        raise RuntimeError("schema file coverage does not match normalized source")
    for file_proof in proof.files:
        record = manifest_files.get(file_proof.relative_path)
        report = schema_files.get(file_proof.relative_path)
        if record is None or report is None:
            raise RuntimeError(
                f"normalized source file is absent from provenance: {file_proof.relative_path}"
            )
        if (
            record.sha256 != file_proof.source_fingerprint.sha256
            or record.byte_size != file_proof.source_fingerprint.byte_size
            or int(report.get("accepted_rows", -1)) != file_proof.normalized_row_count
            or int(report.get("rows", -1)) != file_proof.raw_row_count
            or file_proof.normalized_row_count != file_proof.raw_row_count
        ):
            raise RuntimeError(
                f"normalized source proof mismatch: {file_proof.relative_path}"
            )


class _MembershipIndex:
    """Bounded disk-backed lookup for split membership during train-only passes."""

    def __init__(self, plan: SplitPlan, work_dir: Path, *, batch_rows: int) -> None:
        work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = work_dir / f"membership-{uuid.uuid4().hex}.sqlite3"
        self._connection = sqlite3.connect(self.path)
        self._closed = False
        try:
            self._connection.executescript(
                "PRAGMA journal_mode=DELETE;"
                "PRAGMA synchronous=FULL;"
                "CREATE TABLE membership ("
                "row_uid TEXT PRIMARY KEY, split TEXT NOT NULL, label TEXT NOT NULL"
                ") WITHOUT ROWID;"
            )
            rows = 0
            with plan.membership_path.open("rb") as handle:
                parquet = pq.ParquetFile(handle)
                if tuple(parquet.schema_arrow.names) != _MEMBERSHIP_COLUMNS:
                    raise RuntimeError("split membership schema drift")
                for batch in parquet.iter_batches(batch_size=batch_rows):
                    records = [
                        (str(uid), str(split), str(label))
                        for uid, split, label in zip(
                            batch.column(0).to_pylist(),
                            batch.column(1).to_pylist(),
                            batch.column(2).to_pylist(),
                        )
                    ]
                    self._connection.executemany(
                        "INSERT INTO membership VALUES (?, ?, ?)", records
                    )
                    rows += len(records)
                self._connection.commit()
            expected = plan.train_count + plan.validation_count + plan.test_count
            if rows != expected:
                raise RuntimeError("split membership row count changed while indexing")
        except sqlite3.IntegrityError as exc:
            self.close()
            raise RuntimeError("duplicate split membership row_uid") from exc
        except BaseException:
            self.close()
            raise

    def lookup(self, row_uids: Sequence[str]) -> list[tuple[str, str]]:
        result: dict[str, tuple[str, str]] = {}
        for start in range(0, len(row_uids), 500):
            values = list(row_uids[start : start + 500])
            placeholders = ",".join("?" for _ in values)
            result.update(
                {
                    str(uid): (str(split), str(label))
                    for uid, split, label in self._connection.execute(
                        "SELECT row_uid, split, label FROM membership "
                        f"WHERE row_uid IN ({placeholders})",
                        values,
                    )
                }
            )
        if len(result) != len(row_uids):
            missing = next(uid for uid in row_uids if uid not in result)
            raise RuntimeError(f"normalized row is missing split membership: {missing}")
        return [result[uid] for uid in row_uids]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def __enter__(self) -> _MembershipIndex:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _iter_train_batches(
    source: NormalizedSource,
    membership: _MembershipIndex,
    features: Sequence[str],
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]]:
    iterator = source.iter_chunks()
    try:
        for chunk in iterator:
            frame = chunk.frame
            missing = [name for name in ("row_uid", *features) if name not in frame.columns]
            if missing:
                raise RuntimeError(f"normalized source lost feature columns: {missing}")
            uids = frame["row_uid"].astype(str).tolist()
            assignments = membership.lookup(uids)
            train_positions = [
                index for index, (split, _label) in enumerate(assignments) if split == "train"
            ]
            if not train_positions:
                continue
            train_frame = frame.iloc[train_positions]
            train_uids = train_frame["row_uid"].astype(str).to_numpy()
            labels = np.asarray(
                [assignments[index][1] for index in train_positions], dtype=object
            )
            values = train_frame[list(features)].to_numpy()
            partitions: np.ndarray = np.full(
                len(train_positions), "train", dtype=object
            )
            yield train_uids, values, labels, partitions, train_frame
    finally:
        iterator.close()


def _fit_preprocessor(
    source: NormalizedSource,
    plan: SplitPlan,
    membership: _MembershipIndex,
    config: dict[str, Any],
) -> tuple[FeaturePreprocessor, Any]:
    features = list(source.proof.feature_names)
    builder = StreamingFeaturePreprocessor(
        config,
        candidate_features=features,
        split_fingerprint=plan.fingerprint,
        expected_train_rows=plan.train_count,
        quantile_capacity=int(config["dataset"]["quantile_sketch_capacity"]),
        quantile_seed=int(config["split"]["seed"]),
    )
    sample = None
    try:
        for uids, values, labels, partitions, _frame in _iter_train_batches(
            source, membership, features
        ):
            builder.inspect_batch(
                uids,
                values,
                labels,
                split_fingerprint=plan.fingerprint,
                feature_names=features,
                membership=partitions,
            )
        builder.finalize_imputation()
        for uids, values, labels, partitions, _frame in _iter_train_batches(
            source, membership, features
        ):
            builder.accumulate_anova_batch(
                uids,
                values,
                labels,
                split_fingerprint=plan.fingerprint,
                feature_names=features,
                membership=partitions,
            )
        builder.finalize_selection()
        for uids, values, labels, partitions, frame in _iter_train_batches(
            source, membership, features
        ):
            builder.calibrate_selected_batch(
                uids,
                values,
                labels,
                split_fingerprint=plan.fingerprint,
                feature_names=features,
                membership=partitions,
            )
            if sample is None:
                sample = frame.iloc[: min(len(frame), 64)].copy()
        result = builder.finalize()
        if sample is None:
            raise RuntimeError("no train sample was available for artifact parity")
        return result, sample
    except BaseException as primary:
        try:
            builder._close_audit()
        except BaseException as cleanup:
            if hasattr(primary, "add_note"):
                primary.add_note(
                    f"preprocessing audit cleanup failed: {type(cleanup).__name__}: {cleanup}"
                )
            raise primary from cleanup
        raise


def _publish_preprocessor(
    processor: FeaturePreprocessor,
    sample: Any,
    path: Path,
) -> tuple[FeaturePreprocessor, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        processor.save(temporary)
        _fsync_file(temporary)
        candidate = FeaturePreprocessor.load(temporary)
        expected = processor.transform(sample)
        if not np.array_equal(candidate.transform(sample), expected):
            raise RuntimeError("preprocessor joblib reload transform parity failed")
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("existing preprocessor is not a regular file")
            existing = FeaturePreprocessor.load(path)
            if (
                canonical_json_bytes(existing.feature_manifest())
                != canonical_json_bytes(processor.feature_manifest())
                or not np.array_equal(existing.transform(sample), expected)
            ):
                raise RuntimeError("immutable preprocessor artifact conflict")
            return existing, _sha256_file(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return candidate, _sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _feature_artifact(
    processor: FeaturePreprocessor,
    *,
    source_proof_fingerprint: str,
    split_fingerprint: str,
    preprocessor_sha256: str,
) -> dict[str, Any]:
    def array_state(value: object, name: str) -> dict[str, Any]:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise RuntimeError(f"preprocessor scientific state is non-finite: {name}")
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.tolist(),
        }

    science: dict[str, Any] = {
        "candidate_features": list(processor.candidate_features),
        "selected_features": list(processor.selected_features),
        "selected_indices": array_state(processor.selected_indices, "selected_indices"),
        "selection_scores": array_state(
            processor.selection_scores, "selection_scores"
        ),
        "feature_costs": array_state(processor.feature_costs, "feature_costs"),
        "active_labels": list(processor.active_labels),
        "imputer_statistics": array_state(
            processor.imputer.statistics_, "imputer_statistics"
        ),
        "imputer_n_features_in": int(processor.imputer.n_features_in_),
        "encoder_kind": processor.encoder.kind,
        "encoder_bits": processor.encoder.bits,
        "encoder_thresholds": (
            None
            if processor.encoder.thresholds is None
            else array_state(processor.encoder.thresholds, "encoder_thresholds")
        ),
        "benign_center": array_state(processor.benign_center, "benign_center"),
        "open_distance_threshold": processor.open_distance_threshold,
    }
    if not np.isfinite(float(processor.open_distance_threshold)):
        raise RuntimeError("preprocessor scientific state is non-finite: distance")
    for name in (
        "center_",
        "mean_",
        "var_",
        "scale_",
        "n_features_in_",
        "n_samples_seen_",
    ):
        value = getattr(processor.scaler, name, None)
        science[f"scaler_{name}"] = (
            None if value is None else array_state(value, f"scaler_{name}")
        )
    payload: dict[str, Any] = {
        "schema_version": FEATURE_ARTIFACT_SCHEMA,
        "source_proof_fingerprint": source_proof_fingerprint,
        "split_fingerprint": split_fingerprint,
        "preprocessor_sha256": preprocessor_sha256,
        "feature_manifest": processor.feature_manifest(),
        "scientific_fingerprint": stable_fingerprint(science),
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    return payload


def estimate_preparation_disk(
    source_manifest_path: Path | str,
    schema_report_path: Path | str,
    *,
    train_fraction: float = 0.70,
) -> PreparationDiskEstimate:
    source, schema, _fingerprint = _load_source_contract(
        Path(source_manifest_path), Path(schema_report_path)
    )
    rows = int(schema.get("accepted_rows", 0))
    if rows <= 0:
        raise RuntimeError("cannot estimate preparation disk for an empty source")
    train_rows = max(1, int(rows * float(train_fraction)))
    source_bytes = int(source.total_bytes)
    # SQLite B-trees and three audit phases carry UID/label/digest overhead.
    membership_bytes = max(4096, rows * 160)
    audit_bytes = max(4096, train_rows * 3 * 192)
    # External source/join/coverage runs may coexist with staged and final shards.
    merge_bytes = max(source_bytes * 3, rows * 256)
    final_bytes = max(source_bytes, rows * 128)
    return PreparationDiskEstimate(
        source_snapshot_bytes=source_bytes,
        membership_sqlite_bytes=membership_bytes,
        audit_sqlite_bytes=audit_bytes,
        external_merge_bytes=merge_bytes,
        staging_bytes=final_bytes,
        final_shard_bytes=final_bytes,
    )


def _validate_source_contract_against_disk(
    prepared: PreparedDataset,
) -> tuple[SourceManifest, dict[str, Any]]:
    manifest, schema, schema_fingerprint = _load_source_contract(
        Path(prepared.source_manifest_path), Path(prepared.schema_report_path)
    )
    if (
        manifest.content_sha256 != prepared.source_manifest_fingerprint
        or schema_fingerprint != prepared.schema_report_fingerprint
        or manifest.dataset_name != prepared.dataset
    ):
        raise RuntimeError("prepared source provenance fingerprint mismatch")
    spec = load_registry()[prepared.dataset]
    rebuilt = build_source_manifest(
        Path(prepared.raw_root),
        spec,
        acquisition_method=(
            "official-download" if prepared.dataset == "nbaiot" else "manual-local-source"
        ),
        acquisition_url=spec.download_url if prepared.dataset == "nbaiot" else None,
    )
    if rebuilt.to_dict() != manifest.to_dict():
        raise RuntimeError("raw source no longer matches its source manifest")
    inspected = inspect_csv_dataset(
        prepared.dataset,
        Path(prepared.raw_root),
        required_columns=spec.required_columns,
    ).as_dict()
    if inspected != schema:
        raise RuntimeError("raw source no longer matches its schema report")
    return manifest, schema


def verify_prepared_dataset(descriptor_path: Path | str) -> PreparedDataset:
    path = Path(descriptor_path).resolve(strict=True)
    prepared = PreparedDataset.from_dict(
        path, _read_json(path, "prepared dataset descriptor")
    )
    config_path = Path(prepared.resolved_config_path)
    if _sha256_file(config_path) != prepared.config_sha256:
        raise RuntimeError("prepared dataset config checksum mismatch")
    config = load_config(config_path)
    _validate_full_config(config)
    _validate_source_contract_against_disk(prepared)

    split = SplitPlan(
        strategy=str(config["split"]["strategy"]),
        train_count=prepared.train_count,
        validation_count=prepared.validation_count,
        test_count=prepared.test_count,
        membership_path=Path(prepared.split_membership_path),
        fingerprint=prepared.split_fingerprint,
    )
    split_manifest_path = manifest_path_for_membership(split.membership_path)
    if split_manifest_path.resolve() != Path(prepared.split_manifest_path).resolve():
        raise RuntimeError("prepared split manifest path mismatch")
    split_manifest = read_split_manifest(split)
    if (
        split_manifest.get("fingerprint") != prepared.split_fingerprint
        or _sha256_file(split.membership_path) != prepared.split_membership_sha256
        or split_manifest.get("counts")
        != {
            "train": prepared.train_count,
            "validation": prepared.validation_count,
            "test": prepared.test_count,
        }
    ):
        raise RuntimeError("prepared split fingerprint mismatch")

    preprocessor_path = Path(prepared.preprocessor_path)
    if _sha256_file(preprocessor_path) != prepared.preprocessor_sha256:
        raise RuntimeError("prepared preprocessor checksum mismatch")
    processor = FeaturePreprocessor.load(preprocessor_path)
    feature_payload = _read_json(
        Path(prepared.feature_manifest_path), "feature manifest"
    )
    semantic_feature = dict(feature_payload)
    feature_fingerprint = semantic_feature.pop("fingerprint", None)
    if (
        feature_payload.get("schema_version") != FEATURE_ARTIFACT_SCHEMA
        or feature_fingerprint != stable_fingerprint(semantic_feature)
        or feature_fingerprint != prepared.preprocessing_fingerprint
        or feature_payload.get("source_proof_fingerprint")
        != prepared.normalized_source_fingerprint
        or feature_payload.get("split_fingerprint") != prepared.split_fingerprint
        or feature_payload.get("preprocessor_sha256") != prepared.preprocessor_sha256
        or canonical_json_bytes(feature_payload.get("feature_manifest"))
        != canonical_json_bytes(processor.feature_manifest())
        or feature_payload.get("scientific_fingerprint")
        != _feature_artifact(
            processor,
            source_proof_fingerprint=prepared.normalized_source_fingerprint,
            split_fingerprint=prepared.split_fingerprint,
            preprocessor_sha256=prepared.preprocessor_sha256,
        ).get("scientific_fingerprint")
    ):
        raise RuntimeError("prepared feature artifact fingerprint mismatch")

    shard = verify_shard_manifest(
        prepared.shard_manifest_path,
        split_plan=split,
        preprocessing_fingerprint=prepared.preprocessing_fingerprint,
        max_rows_per_run=int(config["dataset"]["record_batch_rows"]),
    )
    counts = shard["counts"]
    if (
        shard.get("fingerprint") != prepared.shard_fingerprint
        or counts
        != {
            "train": prepared.train_count,
            "validation": prepared.validation_count,
            "test": prepared.test_count,
        }
        or prepared.total_count
        != prepared.train_count + prepared.validation_count + prepared.test_count
    ):
        raise RuntimeError("prepared shard coverage descriptor mismatch")
    return prepared


def prepare_full_dataset(
    config_path: Path | str,
    *,
    raw_root: Path | str,
    source_manifest_path: Path | str,
    schema_report_path: Path | str,
    output_dir: Path | str | None = None,
    descriptor_path: Path | str | None = None,
    work_dir: Path | str | None = None,
) -> PreparedDataset:
    """Prepare all verified source rows into immutable Parquet shards."""

    resolved_config = Path(config_path).expanduser().resolve(strict=True)
    config = load_config(resolved_config)
    _validate_full_config(config)
    dataset = str(config["dataset"]["type"]).lower()
    if dataset not in {"nbaiot", "botiot"}:
        raise ValueError("full preparation supports only nbaiot and botiot")
    source_root = Path(raw_root).expanduser().resolve(strict=True)
    source_manifest_path = Path(source_manifest_path).expanduser().resolve(strict=True)
    schema_report_path = Path(schema_report_path).expanduser().resolve(strict=True)
    configured_manifest = resolve_path(config, config["dataset"]["shard_manifest"])
    assert configured_manifest is not None
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else configured_manifest.parent.resolve()
    )
    descriptor = (
        Path(descriptor_path).expanduser().resolve()
        if descriptor_path is not None
        else output / "prepared_dataset.json"
    )
    work = (
        Path(work_dir).expanduser().resolve()
        if work_dir is not None
        else output / ".work"
    )
    if descriptor.exists():
        existing = verify_prepared_dataset(descriptor)
        requested_paths = {
            "resolved_config_path": resolved_config,
            "raw_root": source_root,
            "source_manifest_path": source_manifest_path,
            "schema_report_path": schema_report_path,
        }
        for field, requested in requested_paths.items():
            if Path(str(getattr(existing, field))).resolve() != requested:
                raise RuntimeError(
                    f"prepared descriptor {field} does not match requested input"
                )
        if Path(existing.shard_manifest_path).resolve().parent != output:
            raise RuntimeError("prepared descriptor output does not match requested output")
        return existing

    output.mkdir(parents=True, exist_ok=True)
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_manifest, schema, schema_fingerprint = _load_source_contract(
        source_manifest_path, schema_report_path
    )
    if source_manifest.dataset_name != dataset:
        raise RuntimeError("source manifest dataset does not match full config")

    split_dir = output / "split"
    preprocessor_path = output / "preprocessor.joblib"
    feature_manifest_path = output / "feature_manifest.json"
    source: NormalizedSource | None = None
    try:
        source = open_normalized_source(
            config,
            path_override=source_root,
            apply_sampling_caps=False,
            work_dir=work,
        )
        with source:
            _verify_source_proof(source.proof, source_manifest, schema)
            split = build_split_plan(
                source.iter_chunks(),
                config,
                split_dir,
                max_rows_per_run=int(config["dataset"]["record_batch_rows"]),
                source_manifest_fingerprint=source_manifest.content_sha256,
            )
            with _MembershipIndex(
                split,
                work,
                batch_rows=int(config["dataset"]["record_batch_rows"]),
            ) as membership:
                processor, sample = _fit_preprocessor(
                    source, split, membership, config
                )
            processor, preprocessor_sha = _publish_preprocessor(
                processor, sample, preprocessor_path
            )
            feature_payload = _feature_artifact(
                processor,
                source_proof_fingerprint=source.proof.fingerprint,
                split_fingerprint=split.fingerprint,
                preprocessor_sha256=preprocessor_sha,
            )
            _publish_json_immutable(feature_manifest_path, feature_payload)
            preprocessing_fingerprint = str(feature_payload["fingerprint"])
            shards = write_parquet_shards(
                source.iter_chunks(),
                split,
                processor.selected_features,
                output,
                dataset_name=dataset,
                preprocessing_fingerprint=preprocessing_fingerprint,
                shard_target_rows=int(config["dataset"]["shard_target_rows"]),
                record_batch_rows=int(config["dataset"]["record_batch_rows"]),
                max_rows_per_run=int(config["dataset"]["record_batch_rows"]),
            )
            verified_shard = verify_shard_manifest(
                shards.manifest_path,
                split_plan=split,
                preprocessing_fingerprint=preprocessing_fingerprint,
                max_rows_per_run=int(config["dataset"]["record_batch_rows"]),
            )
            split_manifest = read_split_manifest(split)
            prepared = PreparedDataset(
                descriptor_path=str(descriptor),
                dataset=dataset,
                resolved_config_path=str(resolved_config),
                config_sha256=_sha256_file(resolved_config),
                raw_root=str(source_root),
                source_manifest_path=str(source_manifest_path),
                source_manifest_fingerprint=source_manifest.content_sha256,
                schema_report_path=str(schema_report_path),
                schema_report_fingerprint=schema_fingerprint,
                normalized_source_fingerprint=source.proof.fingerprint,
                split_membership_path=str(split.membership_path.resolve()),
                split_membership_sha256=str(split_manifest["membership"]["sha256"]),
                split_manifest_path=str(
                    manifest_path_for_membership(split.membership_path).resolve()
                ),
                split_fingerprint=split.fingerprint,
                preprocessor_path=str(preprocessor_path.resolve()),
                preprocessor_sha256=preprocessor_sha,
                feature_manifest_path=str(feature_manifest_path.resolve()),
                preprocessing_fingerprint=preprocessing_fingerprint,
                shard_manifest_path=str(shards.manifest_path.resolve()),
                shard_fingerprint=str(verified_shard["fingerprint"]),
                train_count=split.train_count,
                validation_count=split.validation_count,
                test_count=split.test_count,
                total_count=(
                    split.train_count + split.validation_count + split.test_count
                ),
            )
            _publish_json_immutable(descriptor, prepared.to_dict())
    finally:
        if source is not None:
            try:
                source.close()
            except RuntimeError as exc:
                if "closed" not in str(exc):
                    raise
    result = verify_prepared_dataset(descriptor)
    try:
        if work.exists() and not any(work.iterdir()):
            work.rmdir()
    except OSError:
        pass
    return result


__all__ = [
    "FEATURE_ARTIFACT_SCHEMA",
    "PREPARED_DATASET_SCHEMA",
    "PREPARATION_ALGORITHM",
    "PreparedDataset",
    "PreparationDiskEstimate",
    "estimate_preparation_disk",
    "prepare_full_dataset",
    "verify_prepared_dataset",
]
