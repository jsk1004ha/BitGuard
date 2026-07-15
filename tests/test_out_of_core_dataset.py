from __future__ import annotations

import json
import pickle
import tempfile
import tracemalloc
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from tests.test_out_of_core_prepare import _source_contract, _write_botiot
from tests.test_out_of_core_shard import _make_directory_link, _remove_directory_link


class BatchLayoutBoundedMemoryTests(unittest.TestCase):
    def test_huge_resume_layout_does_not_materialize_one_object_per_batch(self) -> None:
        from bitguard_bnn.out_of_core.dataset import (
            DataCursor,
            _ShardEntry,
            _resume_layout,
        )

        row_count = 100_000_001
        batch_size = 2_048
        batch_position = 48_827
        entries = (
            _ShardEntry(
                path="unused.parquet",
                fingerprint="f" * 64,
                rows=row_count,
                label="benign",
            ),
        )
        dataset = SimpleNamespace(
            row_count=row_count,
            batch_size=batch_size,
            split="train",
            epoch=7,
            cursor=DataCursor(7, 0, batch_position, 12),
            entries=entries,
            permuted_shards=lambda: entries,
        )

        tracemalloc.start()
        layout = _resume_layout(dataset)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertFalse(isinstance(layout, (list, tuple)))
        self.assertEqual(layout.start_index, batch_position)
        self.assertLess(peak, 2 * 1024 * 1024)

    def test_odd_training_rows_with_batch_size_two_are_rejected(self) -> None:
        from bitguard_bnn.out_of_core.dataset import _logical_batch_sizes

        with self.assertRaisesRegex(ValueError, "batch_size=2"):
            tuple(
                _logical_batch_sizes(
                    5,
                    2,
                    allow_singleton=False,
                )
            )


class OutOfCoreDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        raw = cls.root / "raw"
        _write_botiot(raw)
        source, schema = _source_contract("botiot", raw, cls.root)
        repository = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load(
            (repository / "configs" / "full" / "botiot.yaml").read_text(
                encoding="utf-8"
            )
        )
        payload["dataset"]["record_batch_rows"] = 3
        payload["dataset"]["shard_target_rows"] = 8
        payload["dataset"]["quantile_sketch_capacity"] = 64
        payload["preprocess"].update(
            {
                "feature_budget": 1,
                "selection": "expert",
                "expert_features": ["bytes"],
            }
        )
        payload["cascade"].update(
            {
                "boolean_fast_path_enabled": True,
                "boolean_fast_path_features": ["rate"],
            }
        )
        cls.config_path = cls.root / "fixture.yaml"
        cls.config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        cls.prepared = prepare_full_dataset(
            cls.config_path,
            raw_root=raw,
            source_manifest_path=source,
            schema_report_path=schema,
            output_dir=cls.root / "prepared",
            descriptor_path=cls.root / "control" / "botiot.json",
            work_dir=cls.root / "work",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _dataset(self, *, epoch: int = 3, batch_size: int = 4):
        from bitguard_bnn.out_of_core.dataset import ParquetTrainingDataset

        dataset = ParquetTrainingDataset(
            self.prepared.descriptor_path,
            split="train",
            batch_size=batch_size,
            seed=17,
            shuffle_buffer_rows=3,
        )
        dataset.set_epoch(epoch)
        return dataset

    @staticmethod
    def _collect(dataset, workers: int):
        from bitguard_bnn.out_of_core.dataset import iter_ordered_batches

        return list(iter_ordered_batches(dataset, num_workers=workers))

    def test_public_contract_and_spawn_pickle_have_no_live_handles(self) -> None:
        from bitguard_bnn.out_of_core.dataset import (
            DATASET_ALGORITHM,
            DataCursor,
            ParquetTrainingDataset,
            iter_ordered_batches,
        )

        dataset = self._dataset()
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertTrue(DATASET_ALGORITHM.startswith("bitguard."))
        self.assertTrue(hasattr(DataCursor, "__dataclass_fields__"))
        self.assertTrue(callable(ParquetTrainingDataset))
        self.assertTrue(callable(iter_ordered_batches))
        self.assertEqual(restored.entries, dataset.entries)
        forbidden = (pq.ParquetFile,)
        self.assertFalse(any(isinstance(value, forbidden) for value in dataset.__dict__.values()))
        self.assertFalse(any(hasattr(value, "read") for value in dataset.__dict__.values()))

    def test_zero_and_two_workers_produce_identical_batches_and_exact_coverage(self) -> None:
        zero = self._collect(self._dataset(), 0)
        two_dataset = self._dataset()
        two = self._collect(two_dataset, 2)
        zero_uids = [list(batch["row_uid"]) for batch in zero]
        two_uids = [list(batch["row_uid"]) for batch in two]
        self.assertEqual(zero_uids, two_uids)
        flat = [uid for batch in zero_uids for uid in batch]
        self.assertEqual(len(flat), self.prepared.train_count)
        self.assertEqual(len(flat), len(set(flat)))
        self.assertTrue(all(2 <= len(batch) <= 4 for batch in zero_uids))

    def test_uneven_shards_keep_pending_payloads_within_prefetch_bound(self) -> None:
        dataset = self._dataset()
        self.assertGreater(len({entry.rows for entry in dataset.entries}), 1)

        self._collect(dataset, 2)

        self.assertEqual(dataset.worker_ids_observed, (0, 1))
        self.assertGreater(dataset.max_pending_chunks_observed, 0)
        self.assertLessEqual(dataset.max_pending_chunks_observed, 4)

    def test_shuffle_buffer_rejects_a_larger_row_group_payload(self) -> None:
        from bitguard_bnn.out_of_core.dataset import ParquetTrainingDataset

        with self.assertRaisesRegex(ValueError, "largest prepared row group"):
            ParquetTrainingDataset(
                self.prepared.descriptor_path,
                split="train",
                batch_size=4,
                seed=17,
                shuffle_buffer_rows=2,
            )

    def test_epoch_order_is_stable_then_changes_and_classes_are_interleaved(self) -> None:
        first_dataset = self._dataset(epoch=3)
        repeat_dataset = self._dataset(epoch=3)
        other_dataset = self._dataset(epoch=4)
        first = [uid for batch in self._collect(first_dataset, 0) for uid in batch["row_uid"]]
        repeat = [uid for batch in self._collect(repeat_dataset, 0) for uid in batch["row_uid"]]
        other = [uid for batch in self._collect(other_dataset, 0) for uid in batch["row_uid"]]
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)
        labels = [entry.label for entry in first_dataset.permuted_shards()]
        if len(set(labels)) > 1:
            self.assertNotEqual(labels[0], labels[1])

    def test_selected_transform_labels_metadata_and_raw_boolean_match_frozen_artifact(self) -> None:
        from bitguard_bnn.preprocess import FeaturePreprocessor

        dataset = self._dataset()
        batches = self._collect(dataset, 0)
        processor = FeaturePreprocessor.load(Path(self.prepared.preprocessor_path))
        manifest = json.loads(Path(self.prepared.shard_manifest_path).read_text(encoding="utf-8"))
        raw = pd.concat(
            [
                pq.read_table(
                    Path(self.prepared.output_dir) / entry["path"],
                    columns=["row_uid", "behavior_label", *processor.candidate_features],
                ).to_pandas()
                for entry in manifest["entries"]
                if entry["split"] == "train"
            ],
            ignore_index=True,
        ).set_index("row_uid")
        for batch in batches:
            ordered = raw.loc[list(batch["row_uid"])].reset_index()
            np.testing.assert_allclose(batch["features"], processor.transform(ordered))
            np.testing.assert_allclose(
                batch["unencoded"], processor.transform_unencoded(ordered)
            )
            np.testing.assert_array_equal(batch["labels"], processor.encode_labels(ordered))
            np.testing.assert_allclose(batch["boolean_raw"]["rate"], ordered["rate"])
            self.assertIn("device_id", batch["metadata"])
            self.assertIn("timestamp", batch["metadata"])

    def test_cursor_resumes_at_exact_suffix_and_rejects_epoch_mismatch(self) -> None:
        from bitguard_bnn.out_of_core.dataset import DataCursor

        dataset = self._dataset(epoch=5)
        full = self._collect(dataset, 0)
        cursor = full[2]["cursor"]
        dataset.set_epoch(5, cursor)
        resumed = self._collect(dataset, 0)
        self.assertEqual(
            [list(batch["row_uid"]) for batch in resumed],
            [list(batch["row_uid"]) for batch in full[2:]],
        )
        self.assertEqual(resumed[0]["cursor"], cursor)
        self.assertEqual(
            resumed[-1]["next_cursor"].shard_position, len(dataset.entries)
        )
        with self.assertRaisesRegex(ValueError, "epoch"):
            dataset.set_epoch(
                6,
                DataCursor(
                    epoch=5,
                    shard_position=0,
                    batch_position=0,
                    optimizer_step=0,
                ),
            )

    def test_singleton_tail_is_redistributed_without_exceeding_maximum(self) -> None:
        batches = self._collect(self._dataset(batch_size=3), 0)
        sizes = [len(batch["row_uid"]) for batch in batches]
        self.assertEqual(sum(sizes), self.prepared.train_count)
        self.assertNotIn(1, sizes)
        self.assertLessEqual(max(sizes), 3)

    def test_training_split_with_fewer_than_two_rows_is_rejected(self) -> None:
        from bitguard_bnn.out_of_core.dataset import (
            ParquetTrainingDataset,
            _PinnedFileIdentity,
            _logical_batch_sizes,
        )
        from bitguard_bnn.out_of_core.shard import load_shard_manifest

        manifest = load_shard_manifest(self.prepared.shard_manifest_path)
        train_entry = next(
            dict(entry) for entry in manifest["entries"] if entry["split"] == "train"
        )
        train_entry["rows"] = 1
        tiny_manifest = {**manifest, "entries": [train_entry]}
        tiny_prepared = replace(self.prepared, train_count=1)
        identity = _PinnedFileIdentity(1, 1, 0, 1, 1)
        with (
            patch(
                "bitguard_bnn.out_of_core.dataset.verify_prepared_dataset",
                return_value=tiny_prepared,
            ) as verify,
            patch(
                "bitguard_bnn.out_of_core.dataset._load_pinned_manifest",
                return_value=tiny_manifest,
            ),
            patch(
                "bitguard_bnn.out_of_core.dataset._inspect_verified_shard",
                return_value=(identity, (1,)),
            ),
            self.assertRaisesRegex(ValueError, "at least two rows"),
        ):
            ParquetTrainingDataset(
                self.prepared.descriptor_path, batch_size=4, seed=1
            )
        verify.assert_called_once()
        self.assertEqual(
            tuple(_logical_batch_sizes(1, 4, allow_singleton=True)),
            (1,),
        )

        odd_entry = dict(train_entry)
        odd_entry["rows"] = 5
        odd_manifest = {**manifest, "entries": [odd_entry]}
        odd_prepared = replace(self.prepared, train_count=5)
        with (
            patch(
                "bitguard_bnn.out_of_core.dataset.verify_prepared_dataset",
                return_value=odd_prepared,
            ),
            patch(
                "bitguard_bnn.out_of_core.dataset._load_pinned_manifest",
                return_value=odd_manifest,
            ),
            patch(
                "bitguard_bnn.out_of_core.dataset._inspect_verified_shard",
                return_value=(identity, (5,)),
            ),
            self.assertRaisesRegex(ValueError, "batch_size=2"),
        ):
            ParquetTrainingDataset(
                self.prepared.descriptor_path,
                batch_size=2,
                seed=1,
                shuffle_buffer_rows=5,
            )

    def test_corrupt_descriptor_and_manifest_are_rejected_before_iteration(self) -> None:
        from bitguard_bnn.out_of_core.dataset import ParquetTrainingDataset

        descriptor = Path(self.prepared.descriptor_path)
        descriptor_bytes = descriptor.read_bytes()
        try:
            payload = json.loads(descriptor_bytes)
            payload["train_count"] += 1
            descriptor.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                ParquetTrainingDataset(descriptor, batch_size=4, seed=1)
        finally:
            descriptor.write_bytes(descriptor_bytes)

        manifest = Path(self.prepared.shard_manifest_path)
        manifest_bytes = manifest.read_bytes()
        try:
            payload = json.loads(manifest_bytes)
            payload["counts"]["train"] += 1
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ParquetTrainingDataset(descriptor, batch_size=4, seed=1)
        finally:
            manifest.write_bytes(manifest_bytes)

    def test_manifest_swap_after_strict_verification_is_rejected(self) -> None:
        from bitguard_bnn.out_of_core.dataset import ParquetTrainingDataset
        from bitguard_bnn.out_of_core.manifest import stable_fingerprint
        from bitguard_bnn.out_of_core.prepare import verify_prepared_dataset
        from bitguard_bnn.out_of_core.shard import _manifest_semantics

        manifest = Path(self.prepared.shard_manifest_path)
        original = manifest.read_bytes()
        replacement = json.loads(original)
        replacement["coverage"]["uid_digest"] = "f" * 64
        replacement["fingerprint"] = stable_fingerprint(
            _manifest_semantics(replacement)
        )

        def verify_then_swap(path):
            verified = verify_prepared_dataset(path)
            manifest.write_text(
                json.dumps(replacement, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            return verified

        try:
            with (
                patch(
                    "bitguard_bnn.out_of_core.dataset.verify_prepared_dataset",
                    side_effect=verify_then_swap,
                ),
                self.assertRaisesRegex(RuntimeError, "manifest.*fingerprint"),
            ):
                ParquetTrainingDataset(
                    self.prepared.descriptor_path,
                    batch_size=4,
                    seed=1,
                    shuffle_buffer_rows=3,
                )
        finally:
            manifest.write_bytes(original)

    def test_linked_manifest_root_after_strict_verification_is_rejected(self) -> None:
        from bitguard_bnn.out_of_core.dataset import ParquetTrainingDataset
        from bitguard_bnn.out_of_core.prepare import verify_prepared_dataset

        output = Path(self.prepared.output_dir)
        backup = self.root / "prepared-before-link"
        attacker = self.root / "attacker-manifest-root"
        attacker.mkdir()
        (attacker / "shard_manifest.json").write_bytes(
            Path(self.prepared.shard_manifest_path).read_bytes()
        )

        def verify_then_link(path):
            verified = verify_prepared_dataset(path)
            output.rename(backup)
            try:
                _make_directory_link(output, attacker)
            except BaseException:
                backup.rename(output)
                raise
            return verified

        try:
            with (
                patch(
                    "bitguard_bnn.out_of_core.dataset.verify_prepared_dataset",
                    side_effect=verify_then_link,
                ),
                self.assertRaisesRegex(RuntimeError, "manifest root"),
            ):
                ParquetTrainingDataset(
                    self.prepared.descriptor_path,
                    batch_size=4,
                    seed=1,
                    shuffle_buffer_rows=3,
                )
        finally:
            if output.exists() or output.is_symlink():
                _remove_directory_link(output)
            if backup.exists():
                backup.rename(output)


if __name__ == "__main__":
    unittest.main()
