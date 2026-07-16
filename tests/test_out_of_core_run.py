from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.bootstrap.orchestrator import BootstrapDependencies, run_bootstrap
from bitguard_bnn.bootstrap.types import BootstrapOptions
from tests.test_out_of_core_prepare import (
    _source_contract,
    _write_botiot,
    _write_nbaiot,
)


def _write_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousCheckpointValue:
    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):
        return (_write_marker, (self.marker,))


class OutOfCoreRunBoundaryTests(unittest.TestCase):
    @staticmethod
    def _config(model_type: str = "vanilla_bnn") -> dict:
        config = copy.deepcopy(DEFAULTS)
        config["dataset"].update(
            {
                "storage": "parquet",
                "shard_manifest": "prepared/shard_manifest.json",
            }
        )
        config["model"]["type"] = model_type
        return config

    def test_run_training_dispatches_parquet_before_csv_allocation(self) -> None:
        from bitguard_bnn import trainer

        config = self._config()
        expected = Path("runs/parquet-run")
        with (
            patch.object(trainer, "load_config", return_value=config),
            patch.object(
                trainer,
                "_load_and_split",
                side_effect=AssertionError("CSV loader must not run"),
            ),
            patch(
                "bitguard_bnn.out_of_core.run.run_out_of_core_training",
                return_value=expected,
            ) as dispatched,
        ):
            actual = trainer.run_training("full.yaml")

        self.assertEqual(actual, expected)
        dispatched.assert_called_once_with(Path("full.yaml"), config=config)

    def test_classical_parquet_model_fails_before_run_directory_or_dataset(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        config = self._config("random_forest")
        with (
            patch.object(run_module, "load_config", return_value=config),
            patch.object(
                run_module,
                "verify_prepared_dataset",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                run_module,
                "create_run_dir",
                side_effect=AssertionError("run directory must not be allocated"),
            ),
            patch.object(
                run_module,
                "ParquetTrainingDataset",
                side_effect=AssertionError("dataset must not be allocated"),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "random_forest does not support exact out-of-core fitting; "
                "choose a neural model or an explicitly supported incremental baseline",
            ):
                run_module.run_out_of_core_training(
                    "full.yaml", config=config, prepared_descriptor_path="prepared.json"
                )

    def test_verified_config_model_guard_runs_before_run_directory_creation(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        caller = self._config("vanilla_bnn")
        verified = self._config("random_forest")
        prepared = SimpleNamespace(resolved_config_path="verified.yaml")
        with (
            patch.object(run_module, "load_config", return_value=verified),
            patch.object(run_module, "create_run_dir") as create_run_dir,
        ):
            with self.assertRaisesRegex(
                ValueError, "random_forest does not support exact out-of-core fitting"
            ):
                run_module._run_verified_neural_training(caller, prepared)

        create_run_dir.assert_not_called()

    def test_verified_config_scientific_mismatch_precedes_run_directory(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        caller = self._config()
        verified = copy.deepcopy(caller)
        verified["training"]["epochs"] += 1
        prepared = SimpleNamespace(resolved_config_path="verified.yaml")
        with (
            patch.object(run_module, "load_config", return_value=verified),
            patch.object(run_module, "create_run_dir") as create_run_dir,
        ):
            with self.assertRaisesRegex(ValueError, "scientific training contract"):
                run_module._run_verified_neural_training(caller, prepared)

        create_run_dir.assert_not_called()

    def test_phase_records_temporary_disk_high_water_sample(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        records: list[dict[str, object]] = []
        calls = 0

        def sample(
            _path: Path, *, cancelled: threading.Event | None = None
        ) -> tuple[int, int]:
            nonlocal calls
            values = ((10, 5), (20, 100), (15, 1))
            value = values[min(calls, len(values) - 1)]
            calls += 1
            return value

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run_module, "_PHASE_SAMPLE_INTERVAL_SECONDS", 0.01),
            patch.object(run_module, "_phase_resource_sample", side_effect=sample),
        ):
            with run_module._phase(records, "external-sort", Path(temporary)):
                deadline = time.monotonic() + 2.0
                while calls < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)

        self.assertGreaterEqual(calls, 2)
        self.assertEqual(records[0]["temporary_disk_sampled_peak_bytes"], 100)

    def test_phase_waits_for_inflight_sample_and_records_its_peak(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        records: list[dict[str, object]] = []
        sample_started = threading.Event()
        release_sample = threading.Event()
        sampling_threads: list[threading.Thread] = []
        phase_entered = threading.Event()

        def sample(
            _path: Path, *, cancelled: threading.Event | None = None
        ) -> tuple[int, int]:
            current = threading.current_thread()
            if current.name.startswith("bitguard-phase-sampler-"):
                sampling_threads.append(current)
                sample_started.set()
                if not release_sample.wait(5.0):
                    raise RuntimeError("test did not release the resource sampler")
                return 20, 100
            return 10, 5

        def run_phase(temporary: Path) -> None:
            with run_module._phase(records, "slow-scan", temporary):
                phase_entered.set()
                self.assertTrue(sample_started.wait(2.0))

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run_module, "_PHASE_SAMPLE_INTERVAL_SECONDS", 0.01),
            patch.object(run_module, "_phase_resource_sample", side_effect=sample),
        ):
            phase_thread = threading.Thread(
                target=run_phase,
                args=(Path(temporary),),
                name="phase-test-worker",
            )
            phase_thread.start()
            try:
                self.assertTrue(phase_entered.wait(2.0))
                self.assertTrue(sample_started.wait(2.0))
                phase_thread.join(timeout=1.2)
                self.assertTrue(
                    phase_thread.is_alive(),
                    "a phase record was published before its in-flight sample ended",
                )
                self.assertEqual(records, [])
            finally:
                release_sample.set()
                phase_thread.join(timeout=2.0)

        self.assertFalse(phase_thread.is_alive())
        self.assertTrue(sampling_threads)
        self.assertTrue(all(not thread.is_alive() for thread in sampling_threads))
        self.assertEqual(records[0]["temporary_disk_sampled_peak_bytes"], 100)

    def test_phase_preserves_body_error_when_resource_sampler_also_fails(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        records: list[dict[str, object]] = []
        sampler_failed = threading.Event()

        def sample(
            _path: Path, *, cancelled: threading.Event | None = None
        ) -> tuple[int, int]:
            if threading.current_thread().name.startswith("bitguard-phase-sampler-"):
                sampler_failed.set()
                raise RuntimeError("sampler failed")
            return 10, 5

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run_module, "_PHASE_SAMPLE_INTERVAL_SECONDS", 0.01),
            patch.object(run_module, "_phase_resource_sample", side_effect=sample),
            self.assertRaisesRegex(ValueError, "primary training failure") as raised,
        ):
            with run_module._phase(records, "dual-failure", Path(temporary)):
                self.assertTrue(sampler_failed.wait(2.0))
                raise ValueError("primary training failure")

        self.assertEqual(records, [])
        self.assertTrue(
            any(
                "sampler failed" in note
                for note in getattr(raised.exception, "__notes__", ())
            )
        )

    def test_phase_preserves_body_error_when_final_sample_also_fails(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        records: list[dict[str, object]] = []
        calls = 0

        def sample(
            _path: Path, *, cancelled: threading.Event | None = None
        ) -> tuple[int, int]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 10, 5
            raise RuntimeError("final sample failed")

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run_module, "_PHASE_SAMPLE_INTERVAL_SECONDS", 3600.0),
            patch.object(run_module, "_phase_resource_sample", side_effect=sample),
            self.assertRaisesRegex(ValueError, "primary training failure") as raised,
        ):
            with run_module._phase(records, "final-dual-failure", Path(temporary)):
                raise ValueError("primary training failure")

        self.assertEqual(records, [])
        self.assertEqual(calls, 2)
        self.assertTrue(
            any(
                "final sample failed" in note
                for note in getattr(raised.exception, "__notes__", ())
            )
        )

    def test_phase_reports_final_sample_failure_without_body_error(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        calls = 0

        def sample(
            _path: Path, *, cancelled: threading.Event | None = None
        ) -> tuple[int, int]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 10, 5
            raise RuntimeError("final sample failed")

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run_module, "_PHASE_SAMPLE_INTERVAL_SECONDS", 3600.0),
            patch.object(run_module, "_phase_resource_sample", side_effect=sample),
            self.assertRaisesRegex(RuntimeError, "phase resource sampling failed"),
        ):
            with run_module._phase([], "final-failure", Path(temporary)):
                pass

    def test_phase_preserves_body_error_when_lifetime_peak_query_fails(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run_module, "_phase_resource_sample", return_value=(10, 5)),
            patch.object(
                run_module,
                "_peak_rss_bytes",
                side_effect=RuntimeError("peak query failed"),
            ),
            self.assertRaisesRegex(ValueError, "primary training failure") as raised,
        ):
            with run_module._phase([], "peak-dual-failure", Path(temporary)):
                raise ValueError("primary training failure")

        self.assertTrue(
            any(
                "peak query failed" in note
                for note in getattr(raised.exception, "__notes__", ())
            )
        )

    def test_phase_reports_lifetime_peak_failure_without_body_error(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run_module, "_phase_resource_sample", return_value=(10, 5)),
            patch.object(
                run_module,
                "_peak_rss_bytes",
                side_effect=RuntimeError("peak query failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "phase resource sampling failed"),
        ):
            with run_module._phase([], "peak-failure", Path(temporary)):
                pass

    def test_exception_note_helper_is_python310_compatible(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        error = RuntimeError("primary")
        with patch.object(error, "add_note", None):
            run_module._add_exception_note(error, "secondary")

        self.assertEqual(str(error), "primary")

    def test_phase_sampling_interval_is_production_safe(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        self.assertGreaterEqual(run_module._PHASE_SAMPLE_INTERVAL_SECONDS, 1.0)

    def test_temporary_disk_sample_tolerates_concurrent_file_cleanup(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            disappearing = root / "disappearing.bin"
            disappearing.write_bytes(b"temporary")
            original_stat = Path.stat

            def racing_stat(path: Path, *args: object, **kwargs: object):
                if path == disappearing:
                    disappearing.unlink(missing_ok=True)
                    raise FileNotFoundError(str(path))
                return original_stat(path, *args, **kwargs)

            with patch.object(Path, "stat", new=racing_stat):
                self.assertEqual(run_module._directory_bytes(root), 0)

    def test_validation_cache_context_removes_memmaps_on_success_and_failure(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import run as run_module
        from bitguard_bnn.out_of_core.cache import CacheLayout

        layout = CacheLayout(
            prepared_descriptor_fingerprint="prepared",
            shard_fingerprint="shards",
            preprocessor_fingerprint="preprocessor",
            source_fingerprint="source",
            main_checkpoint_fingerprint="main",
            tiny_checkpoint_fingerprint=None,
            inference_contract_fingerprint="inference",
            split="validation",
            row_count=1,
            main_class_labels=("benign",),
            routed_class_labels=("benign",),
            true_class_labels=("benign",),
            selected_features=("feature",),
            boolean_features=(),
            device_id_width=1,
            source_id_width=1,
        )
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                cache_root = Path(temporary) / "validation-cache"
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        with run_module._temporary_calibration_cache(
                            cache_root, layout
                        ):
                            raise RuntimeError("injected")
                else:
                    with run_module._temporary_calibration_cache(cache_root, layout):
                        self.assertTrue(cache_root.is_dir())
                self.assertFalse(cache_root.exists())

    def test_validation_cache_context_removes_partial_create_failure(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        def fail_create(root: Path, _layout: object) -> object:
            root.mkdir()
            (root / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("injected create failure")

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "bitguard_bnn.out_of_core.cache.CalibrationCache.create",
                side_effect=fail_create,
            ),
        ):
            cache_root = Path(temporary) / "validation-cache"
            with self.assertRaisesRegex(RuntimeError, "injected create failure"):
                with run_module._temporary_calibration_cache(cache_root, object()):
                    pass
            self.assertFalse(cache_root.exists())

    def test_evaluation_batches_bound_and_preserve_nested_row_order(self) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        class Dataset:
            row_count = 7
            batch_size = 3

            def __init__(self) -> None:
                self.epochs: list[int] = []

            def set_epoch(self, epoch: int) -> None:
                self.epochs.append(epoch)

            def __iter__(self):
                yield {
                    "features": np.arange(14, dtype=np.float32).reshape(7, 2),
                    "unencoded": np.arange(7, dtype=np.float32).reshape(7, 1),
                    "labels": np.arange(7, dtype=np.int64),
                    "row_uid": np.asarray([f"uid-{index}" for index in range(7)]),
                    "metadata": {
                        "sequence_index": np.arange(7, dtype=np.int64),
                        "source_file": np.asarray(["source"] * 7),
                    },
                    "boolean_raw": {
                        "flag": np.arange(7, dtype=np.float32),
                    },
                }

        dataset = Dataset()
        batches = list(run_module._iter_evaluation_batches(dataset))

        self.assertEqual(dataset.epochs, [0])
        self.assertEqual([len(batch["row_uid"]) for batch in batches], [3, 3, 1])
        self.assertLessEqual(max(len(batch["row_uid"]) for batch in batches), 3)
        self.assertEqual(
            [uid for batch in batches for uid in batch["row_uid"]],
            [f"uid-{index}" for index in range(7)],
        )
        np.testing.assert_array_equal(
            np.concatenate([batch["labels"] for batch in batches]), np.arange(7)
        )
        np.testing.assert_array_equal(
            np.concatenate([batch["metadata"]["sequence_index"] for batch in batches]),
            np.arange(7),
        )
        np.testing.assert_array_equal(
            np.concatenate([batch["boolean_raw"]["flag"] for batch in batches]),
            np.arange(7, dtype=np.float32),
        )

    def test_validation_and_test_models_never_receive_shuffle_buffer_batches(
        self,
    ) -> None:
        import torch

        from bitguard_bnn.constants import CANONICAL_LABELS
        from bitguard_bnn.out_of_core import run as run_module

        expected_uids = [f"uid-{index}" for index in range(7)]

        class Dataset:
            row_count = 7
            batch_size = 3
            boolean_features: tuple[str, ...] = ()

            def set_epoch(self, _epoch: int) -> None:
                return None

            def __iter__(self):
                yield {
                    "features": np.arange(14, dtype=np.float32).reshape(7, 2),
                    "unencoded": np.zeros((7, 1), dtype=np.float32),
                    "labels": np.arange(7, dtype=np.int64) % 2,
                    "row_uid": np.asarray(expected_uids),
                    "metadata": {
                        "source_file": np.asarray(["source"] * 7),
                        "device_id": np.asarray(["device"] * 7),
                        "timestamp": np.arange(7, dtype=np.float64),
                        "sequence_index": np.arange(7, dtype=np.int64),
                        "raw_attack": np.asarray([""] * 7),
                        "behavior_label": np.asarray(
                            [
                                "benign",
                                "scan_like",
                                "benign",
                                "scan_like",
                                "benign",
                                "scan_like",
                                "benign",
                            ]
                        ),
                    },
                    "boolean_raw": {},
                }

        observed_batch_sizes: list[int] = []

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, values):
                observed_batch_sizes.append(len(values))
                return torch.stack((values[:, 0], -values[:, 0]), dim=1)

        class Preprocessor:
            active_labels = ["benign", "scan_like"]

            def apply_open_set(self, known, _unencoded):
                return (
                    np.asarray(["benign"] * len(known)),
                    np.zeros(len(known), dtype=bool),
                    np.column_stack((known, np.zeros(len(known), dtype=np.float32))),
                )

        with tempfile.TemporaryDirectory() as temporary:
            validation = run_module._validation_callback(
                Dataset(),
                ["benign", "scan_like"],
                Path(temporary) / "validation",
            )
            validation(Model(), torch.device("cpu"))
            batches = list(
                run_module._prediction_batches(
                    Dataset(),
                    Model(),
                    Preprocessor(),
                    tiny_model=None,
                    tiny_indices=None,
                    calibration=None,
                    boolean_calibration=None,
                    config=copy.deepcopy(DEFAULTS),
                    attack_prior=np.zeros(len(CANONICAL_LABELS)),
                    prediction_metadata={
                        "has_wall_clock_time": True,
                        "temporal_continuity": True,
                    },
                )
            )

        self.assertTrue(observed_batch_sizes)
        self.assertLessEqual(max(observed_batch_sizes), Dataset.batch_size)
        self.assertEqual(
            [uid for batch in batches for uid in batch["row_uid"]], expected_uids
        )

    def test_temporal_prediction_routing_is_chronological_and_restores_storage_order(
        self,
    ) -> None:
        import torch

        from bitguard_bnn.cascade import CascadeCalibration, CascadeStreamRouter
        from bitguard_bnn.constants import CANONICAL_LABELS
        from bitguard_bnn.out_of_core import run as run_module

        storage_uids = ["uid-3", "uid-z", "uid-a", "uid-2"]
        timestamps = [3.0, 1.0, 1.0, 2.0]

        class Dataset:
            row_count = 4
            batch_size = 2

            def set_epoch(self, _epoch: int) -> None:
                return None

            def __iter__(self):
                for start in (0, 2):
                    end = start + 2
                    yield {
                        "features": np.asarray(
                            [[0.2 + index, 0.1] for index in range(start, end)],
                            dtype=np.float32,
                        ),
                        "unencoded": np.zeros((2, 1), dtype=np.float32),
                        "row_uid": np.asarray(storage_uids[start:end], dtype=str),
                        "metadata": {
                            "source_file": np.asarray(["episode"] * 2, dtype=str),
                            "device_id": np.asarray(["device"] * 2, dtype=str),
                            "timestamp": np.asarray(timestamps[start:end]),
                            "sequence_index": np.asarray([3, 1, 0, 2][start:end]),
                            "raw_attack": np.asarray([""] * 2, dtype=str),
                            "behavior_label": np.asarray(["benign"] * 2, dtype=str),
                        },
                        "boolean_raw": {},
                    }

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, values):
                return torch.stack((values[:, 0], -values[:, 0]), dim=1)

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, values):
                benign = torch.full_like(values[:, 0], float(np.log(0.40)))
                attack = torch.full_like(values[:, 0], float(np.log(0.60)))
                return torch.stack((benign, attack), dim=1)

        class Preprocessor:
            active_labels = ["benign", "scan_like"]

            def apply_open_set(self, known, _unencoded):
                unknown = np.zeros(len(known), dtype=bool)
                active = np.column_stack(
                    (known, np.zeros(len(known), dtype=np.float32))
                )
                return np.asarray(["benign"] * len(known)), unknown, active

        observed_keys: list[tuple[float, str, int]] = []

        class AssertingRouter(CascadeStreamRouter):
            def route_batch(self, metadata, *args, **kwargs):
                observed_keys.extend(
                    (
                        float(row.timestamp),
                        str(row.row_uid),
                        int(row.storage_position),
                    )
                    for row in metadata.itertuples()
                )
                return super().route_batch(metadata, *args, **kwargs)

        config = copy.deepcopy(DEFAULTS)
        config["temporal"]["enabled"] = True
        config["cascade"].update(
            {
                "enabled": True,
                "use_temporal_state": True,
                "temporal_penalty": 1.0,
            }
        )
        calibration = CascadeCalibration(-0.31, 1.0, 0.0, 0.0, 4, 0.1)
        attack_prior = np.zeros(len(CANONICAL_LABELS), dtype=np.float64)
        attack_prior[1] = 1.0
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run_module, "CascadeStreamRouter", AssertingRouter),
        ):
            batches = list(
                run_module._prediction_batches(
                    Dataset(),
                    Model(),
                    Preprocessor(),
                    tiny_model=Tiny(),
                    tiny_indices=np.asarray([0], dtype=np.int64),
                    calibration=calibration,
                    boolean_calibration=None,
                    config=config,
                    attack_prior=attack_prior,
                    prediction_metadata={
                        "has_wall_clock_time": True,
                        "temporal_continuity": True,
                    },
                    temporary_directory=Path(temporary),
                )
            )
            self.assertEqual(list(Path(temporary).iterdir()), [])

        self.assertEqual(
            observed_keys,
            [
                (1.0, "uid-a", 2),
                (1.0, "uid-z", 1),
                (2.0, "uid-2", 3),
                (3.0, "uid-3", 0),
            ],
        )
        self.assertEqual(
            [uid for batch in batches for uid in batch["row_uid"]], storage_uids
        )
        actual_probabilities = np.vstack(
            [np.asarray(batch["probabilities"]) for batch in batches]
        )
        actual_stages = np.concatenate(
            [np.asarray(batch["exit_stage"]) for batch in batches]
        )

        features = np.asarray(
            [[0.2 + index, 0.1] for index in range(4)], dtype=np.float32
        )
        with torch.inference_mode():
            known = torch.softmax(Model()(torch.from_numpy(features)), dim=1).numpy()
            tiny_probability = torch.softmax(
                Tiny()(torch.from_numpy(features[:, [0]])), dim=1
            )[:, 0].numpy()
        active = np.column_stack((known, np.zeros(4, dtype=np.float32)))
        main_probability = np.zeros((4, len(CANONICAL_LABELS)), dtype=np.float32)
        for column, label in enumerate(["benign", "scan_like", "unknown_like"]):
            main_probability[:, CANONICAL_LABELS.index(label)] = active[:, column]
        metadata = pd.DataFrame(
            {
                "storage_position": np.arange(4),
                "row_uid": storage_uids,
                "source_file": ["episode"] * 4,
                "device_id": ["device"] * 4,
                "timestamp": timestamps,
                "sequence_index": [3, 1, 0, 2],
            }
        )
        chronological_order = np.asarray([2, 1, 3, 0], dtype=np.int64)
        reference_router = CascadeStreamRouter(
            list(CANONICAL_LABELS), calibration, config, attack_prior
        )
        ordered_probabilities, ordered_stages, _ = reference_router.route_batch(
            metadata.iloc[chronological_order].reset_index(drop=True),
            tiny_probability[chronological_order],
            main_probability[chronological_order],
            np.zeros(4, dtype=bool),
        )
        expected_probabilities = np.empty_like(ordered_probabilities)
        expected_stages = np.empty_like(ordered_stages)
        expected_probabilities[chronological_order] = ordered_probabilities
        expected_stages[chronological_order] = ordered_stages
        np.testing.assert_allclose(actual_probabilities, expected_probabilities)
        np.testing.assert_array_equal(actual_stages, expected_stages)

        old_storage_first_order = np.asarray([1, 2, 3, 0], dtype=np.int64)
        old_order_router = CascadeStreamRouter(
            list(CANONICAL_LABELS), calibration, config, attack_prior
        )
        old_order_probabilities, old_order_stages, _ = old_order_router.route_batch(
            metadata.iloc[old_storage_first_order].reset_index(drop=True),
            tiny_probability[old_storage_first_order],
            main_probability[old_storage_first_order],
            np.zeros(4, dtype=bool),
        )
        restored_old_probabilities = np.empty_like(old_order_probabilities)
        restored_old_stages = np.empty_like(old_order_stages)
        restored_old_probabilities[old_storage_first_order] = old_order_probabilities
        restored_old_stages[old_storage_first_order] = old_order_stages
        self.assertTrue(
            np.any(restored_old_stages != expected_stages)
            or not np.allclose(restored_old_probabilities, expected_probabilities)
        )

    def test_temporal_prediction_order_uses_uid_before_storage_position(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core import run as run_module

        self.assertEqual(
            run_module._temporal_prediction_order(missing_timestamps=0),
            "timestamp, device_id, row_uid, storage_position",
        )
        self.assertEqual(
            run_module._temporal_prediction_order(missing_timestamps=1),
            "device_id, sequence_index, row_uid, storage_position",
        )

    def test_temporal_test_inference_matches_validation_cache_on_uid_ties(
        self,
    ) -> None:
        import torch

        from bitguard_bnn.cascade import CascadeCalibration
        from bitguard_bnn.constants import CANONICAL_LABELS
        from bitguard_bnn.out_of_core import run as run_module
        from bitguard_bnn.out_of_core.cache import CacheLayout, CalibrationCache
        from bitguard_bnn.out_of_core.calibrate import route_validation_cache

        storage_uids = [f"{value:064x}" for value in (3, 9, 1, 2)]
        timestamps = np.asarray([3.0, 1.0, 1.0, 2.0], dtype=np.float64)
        sequences = np.asarray([3, 1, 0, 2], dtype=np.int64)
        features = np.asarray(
            [[0.2 + index, 0.1] for index in range(4)], dtype=np.float32
        )

        class Dataset:
            row_count = 4
            batch_size = 2

            def set_epoch(self, _epoch: int) -> None:
                return None

            def __iter__(self):
                for start in (0, 2):
                    stop = start + 2
                    yield {
                        "features": features[start:stop],
                        "unencoded": np.zeros((2, 1), dtype=np.float32),
                        "labels": np.zeros(2, dtype=np.int64),
                        "row_uid": np.asarray(storage_uids[start:stop]),
                        "metadata": {
                            "source_file": np.asarray(["episode"] * 2),
                            "device_id": np.asarray(["device"] * 2),
                            "timestamp": timestamps[start:stop],
                            "sequence_index": sequences[start:stop],
                            "raw_attack": np.asarray([""] * 2),
                            "behavior_label": np.asarray(["benign"] * 2),
                        },
                        "boolean_raw": {},
                    }

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, values):
                return torch.stack((values[:, 0], -values[:, 0]), dim=1)

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, values):
                benign = torch.full_like(values[:, 0], float(np.log(0.40)))
                attack = torch.full_like(values[:, 0], float(np.log(0.60)))
                return torch.stack((benign, attack), dim=1)

        config = copy.deepcopy(DEFAULTS)
        config["temporal"]["enabled"] = True
        config["cascade"].update(
            {
                "enabled": True,
                "use_temporal_state": True,
                "temporal_penalty": 1.0,
            }
        )

        class Preprocessor:
            active_labels = ["benign", "scan_like"]
            selected_features = ["feature"]
            open_distance_threshold = 1.0

            def __init__(self) -> None:
                self.config = config

            def apply_open_set(self, known, _unencoded):
                return (
                    np.asarray(["benign"] * len(known)),
                    np.zeros(len(known), dtype=bool),
                    np.column_stack((known, np.zeros(len(known), dtype=np.float32))),
                )

        preprocessor = Preprocessor()
        calibration = CascadeCalibration(-0.31, 1.0, 0.0, 0.0, 4, 0.1)
        attack_prior = np.zeros(len(CANONICAL_LABELS), dtype=np.float64)
        attack_prior[CANONICAL_LABELS.index("scan_like")] = 1.0
        main_model = Model()
        tiny_model = Tiny()
        with torch.inference_mode():
            known = torch.softmax(main_model(torch.from_numpy(features)), dim=1).numpy()
            tiny = torch.softmax(tiny_model(torch.from_numpy(features[:, [0]])), dim=1)[
                :, 0
            ].numpy()

        layout = CacheLayout(
            prepared_descriptor_fingerprint="prepared",
            shard_fingerprint="shards",
            preprocessor_fingerprint="preprocessor",
            source_fingerprint="source",
            main_checkpoint_fingerprint="main",
            tiny_checkpoint_fingerprint="tiny",
            inference_contract_fingerprint="contract",
            split="validation",
            row_count=4,
            main_class_labels=("benign", "scan_like"),
            routed_class_labels=tuple(CANONICAL_LABELS),
            true_class_labels=tuple(CANONICAL_LABELS),
            selected_features=("feature",),
            boolean_features=(),
            device_id_width=8,
            source_id_width=8,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with CalibrationCache.create(root / "cache", layout) as cache:
                cache.commit_inference_range(
                    0,
                    {
                        "cache_position": np.arange(4, dtype=np.int64),
                        "uid_digest": np.vstack(
                            [
                                np.frombuffer(bytes.fromhex(uid), dtype=np.uint8)
                                for uid in storage_uids
                            ]
                        ),
                        "true_label": np.zeros(4, dtype=np.int32),
                        "known_probabilities": known.astype(np.float32),
                        "selected_values": np.zeros((4, 1), dtype=np.float32),
                        "tiny_benign_probability": tiny.astype(np.float32),
                        "timestamp": timestamps,
                        "sequence": sequences,
                        "device_id": ["device"] * 4,
                        "source_id": ["episode"] * 4,
                        "boolean_flags": np.zeros((4, 0), dtype=bool),
                    },
                )
                route_validation_cache(
                    cache,
                    preprocessor,
                    calibration,
                    config,
                    attack_prior,
                    validation_contract="contract",
                    work_dir=root / "validation-routing",
                    chunk_rows=2,
                )
                validation_probabilities = np.asarray(
                    cache.arrays["routed_probabilities"]
                ).copy()
                validation_stages = np.asarray(cache.arrays["exit_stage"]).copy()

            prediction_batches = list(
                run_module._prediction_batches(
                    Dataset(),
                    main_model,
                    preprocessor,
                    tiny_model=tiny_model,
                    tiny_indices=np.asarray([0], dtype=np.int64),
                    calibration=calibration,
                    boolean_calibration=None,
                    config=config,
                    attack_prior=attack_prior,
                    prediction_metadata={
                        "has_wall_clock_time": True,
                        "temporal_continuity": True,
                    },
                    temporary_directory=root / "test-routing",
                )
            )

        np.testing.assert_allclose(
            np.vstack([batch["probabilities"] for batch in prediction_batches]),
            validation_probabilities,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            np.concatenate([batch["exit_stage"] for batch in prediction_batches]),
            validation_stages,
        )

    def test_export_checkpoint_load_is_weights_only(self) -> None:
        import torch

        from bitguard_bnn import export

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            checkpoint.write_bytes(b"placeholder")
            with patch.object(torch, "load", return_value={}) as loader:
                self.assertEqual(export._load_checkpoint(checkpoint), {})

        self.assertIs(loader.call_args.kwargs["weights_only"], True)
        self.assertEqual(str(loader.call_args.kwargs["map_location"]), "cpu")
        self.assertTrue(hasattr(loader.call_args.args[0], "read"))

    def test_export_checkpoint_loader_rejects_executable_pickle_payload(self) -> None:
        import torch

        from bitguard_bnn import export

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed.txt"
            checkpoint = root / "best_model.pt"
            torch.save({"payload": _MaliciousCheckpointValue(marker)}, checkpoint)

            with self.assertRaisesRegex(ValueError, "safe tensor payload"):
                export._load_checkpoint(checkpoint)

            self.assertFalse(marker.exists())


class OutOfCoreRunEndToEndTests(unittest.TestCase):
    def _prepare(self, dataset: str, root: Path, *, cascade_enabled: bool = True):
        from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

        repository = Path(__file__).resolve().parents[1]
        raw = root / "raw"
        if dataset == "nbaiot":
            _write_nbaiot(raw)
            expert_features = ["mean", "std"]
        else:
            _write_botiot(raw)
            expert_features = ["bytes", "rate"]
        source, schema = _source_contract(dataset, raw, root)
        payload = yaml.safe_load(
            (repository / "configs" / "full" / f"{dataset}.yaml").read_text(
                encoding="utf-8"
            )
        )
        payload["experiment"]["output_dir"] = str(root / "runs")
        payload["dataset"].update(
            {
                "record_batch_rows": 4,
                "shard_target_rows": 8,
                "quantile_sketch_capacity": 64,
            }
        )
        payload["preprocess"].update(
            {
                "feature_budget": 2,
                "selection": "expert",
                "expert_features": expert_features,
            }
        )
        payload["model"].update(
            {
                "type": "vanilla_bnn",
                "hidden_dims": [4],
                "dropout": 0.0,
            }
        )
        payload["loss"]["distillation_alpha"] = 0.0
        payload["training"].update(
            {
                "epochs": 1,
                "batch_size": 4,
                "patience": 2,
                "num_workers": 0,
                "device": "cpu",
                "amp": False,
                "checkpoint_every_steps": 1,
                "shuffle_buffer_rows": 8,
            }
        )
        payload["cascade"].update(
            {
                "enabled": cascade_enabled,
                "boolean_fast_path_enabled": False,
                "tiny_feature_budget": 1,
                "hidden_dims": [4],
                "min_attack_recall": 0.5,
                "threshold_grid_size": 11,
            }
        )
        # Test inference must exercise stateful routing over the dataset's
        # deliberately shuffled row-group stream for both supported datasets.
        payload["temporal"]["enabled"] = True
        payload["evaluation"].update(
            {
                "fixed_fpr_targets": [0.5],
                "plot_sample_rows": 10,
                "benchmark_warmup": 1,
                "benchmark_repeats": 1,
            }
        )
        config_path = root / f"{dataset}.yaml"
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return prepare_full_dataset(
            config_path,
            raw_root=raw,
            source_manifest_path=source,
            schema_report_path=schema,
            output_dir=root / "prepared",
            descriptor_path=root / "control" / f"{dataset}.json",
            work_dir=root / "work",
        )

    def test_both_dataset_shapes_run_all_rows_calibrate_evaluate_and_export(
        self,
    ) -> None:
        import torch

        from bitguard_bnn.config import load_config
        from bitguard_bnn.out_of_core.manifest import stable_fingerprint
        from bitguard_bnn.out_of_core.replay import replay_parquet_predictions
        from bitguard_bnn.out_of_core.run import run_out_of_core_training

        expected_temporal_contract = {
            "nbaiot": (False, True),
            "botiot": (True, True),
        }
        for dataset in ("nbaiot", "botiot"):
            with (
                self.subTest(dataset=dataset),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                prepared = self._prepare(dataset, root)
                config = load_config(prepared.resolved_config_path)
                config["experiment"]["output_dir"] = str(root / "runs")

                run_dir = run_out_of_core_training(
                    prepared.resolved_config_path,
                    config=config,
                    prepared_descriptor_path=prepared.descriptor_path,
                )

                summary = json.loads(
                    (run_dir / "run_summary.json").read_text(encoding="utf-8")
                )
                main_summary = json.loads(
                    (run_dir / "model_summary.json").read_text(encoding="utf-8")
                )
                tiny_summary = json.loads(
                    (run_dir / "tiny_model_summary.json").read_text(encoding="utf-8")
                )
                metrics = json.loads(
                    (run_dir / "metrics.json").read_text(encoding="utf-8")
                )
                edge = json.loads(
                    (run_dir / "edge" / "bitguard_edge_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                checkpoint = torch.load(
                    run_dir / "last_training_state.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                prediction_contract = json.loads(
                    (run_dir / "inference_contract.json").read_text(encoding="utf-8")
                )
                prediction_file = pq.ParquetFile(run_dir / "predictions.parquet")
                try:
                    prediction_rows = prediction_file.metadata.num_rows
                    prediction_metadata = prediction_file.schema_arrow.metadata or {}
                    temporal_columns = prediction_file.read(
                        columns=["has_wall_clock_time", "temporal_continuity"]
                    ).to_pydict()
                finally:
                    prediction_file.close(force=True)
                expected_wall_clock, expected_continuity = expected_temporal_contract[
                    dataset
                ]

                self.assertEqual(
                    main_summary["train_rows_per_epoch"], prepared.train_count
                )
                self.assertEqual(
                    main_summary["validation_rows_per_epoch"],
                    prepared.validation_count,
                )
                self.assertEqual(
                    tiny_summary["train_rows_per_epoch"], prepared.train_count
                )
                self.assertEqual(summary["validation_rows"], prepared.validation_count)
                self.assertEqual(summary["test_rows"], prepared.test_count)
                self.assertEqual(checkpoint["scientific_signature"]["split"], "train")
                self.assertEqual(
                    checkpoint["scientific_signature"][
                        "prepared_descriptor_fingerprint"
                    ],
                    prepared.to_dict()["fingerprint"],
                )
                self.assertEqual(prediction_rows, prepared.test_count)
                self.assertFalse((run_dir / "validation_cache").exists())
                self.assertEqual(
                    set(temporal_columns["has_wall_clock_time"]),
                    {expected_wall_clock},
                )
                self.assertEqual(
                    set(temporal_columns["temporal_continuity"]),
                    {expected_continuity},
                )
                self.assertEqual(
                    prediction_metadata[b"bitguard_test_contract"].decode("utf-8"),
                    prediction_contract["fingerprint"],
                )
                self.assertEqual(
                    prediction_contract["fingerprint"],
                    stable_fingerprint(prediction_contract["contract"]),
                )
                self.assertEqual(prediction_contract["format_version"], 2)
                self.assertEqual(
                    prediction_contract["contract"]["algorithm"],
                    "bitguard.full-test-inference-contract.v2",
                )
                self.assertEqual(
                    set(prediction_contract["contract"]),
                    {
                        "algorithm",
                        "prepared",
                        "checkpoints",
                        "preprocessor",
                        "cascade",
                        "boolean_fast_path",
                        "routing",
                        "fixed_fpr",
                        "prediction_metadata",
                    },
                )
                self.assertEqual(
                    prediction_contract["contract"]["prediction_metadata"],
                    {
                        "has_wall_clock_time": expected_wall_clock,
                        "temporal_continuity": expected_continuity,
                    },
                )
                contract = prediction_contract["contract"]
                self.assertIn("descriptor_fingerprint", contract["prepared"])
                self.assertIn("main_sha256", contract["checkpoints"])
                self.assertIn("tiny_sha256", contract["checkpoints"])
                self.assertIn("artifact_sha256", contract["preprocessor"])
                self.assertIn("feature_manifest", contract["preprocessor"])
                self.assertIn("preprocess_config", contract["preprocessor"])
                self.assertIn("calibration", contract["cascade"])
                self.assertIn("calibration", contract["boolean_fast_path"])
                self.assertIn("enabled_setting", contract["boolean_fast_path"])
                self.assertIn("temporal_config", contract["routing"])
                self.assertIn("cascade_config", contract["routing"])
                self.assertEqual(
                    contract["routing"]["order_algorithm"],
                    {
                        "wall_clock": [
                            "timestamp",
                            "device_id",
                            "row_uid",
                            "storage_position",
                        ],
                        "sequence_fallback": [
                            "device_id",
                            "sequence_index",
                            "row_uid",
                            "storage_position",
                        ],
                        "timestamp_policy": "wall_clock_only_if_all_rows_are_finite",
                        "restore": ["storage_position"],
                    },
                )
                legacy_contract = copy.deepcopy(contract)
                legacy_contract["algorithm"] = (
                    "bitguard.full-test-inference-contract.v1"
                )
                legacy_contract["routing"].pop("order_algorithm")
                self.assertNotEqual(
                    prediction_contract["fingerprint"],
                    stable_fingerprint(legacy_contract),
                )
                self.assertIn("thresholds", contract["fixed_fpr"])
                self.assertEqual(metrics["fixed_fpr"]["threshold_source"], "validation")
                fixed_fpr = json.loads(
                    (run_dir / "fixed_fpr_thresholds.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    fixed_fpr["score_pipeline"],
                    "routed_attack_score=1-p_benign",
                )
                self.assertTrue(edge["folding_parity_passed"])
                self.assertTrue(edge["end_to_end_logit_parity_passed"])
                self.assertNotIn("storage", edge)

                replay_metrics = replay_parquet_predictions(
                    run_dir / "predictions.parquet",
                    run_dir / "temporal_predictions.parquet",
                    config,
                    temporary_directory=run_dir / "replay-temporary",
                    batch_rows=3,
                    order_run_rows=3,
                )
                replay_table = pq.read_table(
                    run_dir / "temporal_predictions.parquet",
                    columns=["has_wall_clock_time", "temporal_continuity"],
                ).to_pydict()
                self.assertEqual(replay_metrics["rows"], prepared.test_count)
                self.assertEqual(
                    replay_metrics["temporal_continuity_verified"],
                    expected_continuity,
                )
                self.assertEqual(
                    replay_metrics["observed_device_hours"] is not None,
                    expected_wall_clock,
                )
                self.assertEqual(
                    set(replay_table["has_wall_clock_time"]),
                    {expected_wall_clock},
                )
                self.assertEqual(
                    set(replay_table["temporal_continuity"]),
                    {expected_continuity},
                )

    def test_main_only_still_uses_validation_calibration_for_fixed_fpr(self) -> None:
        from bitguard_bnn.config import load_config
        from bitguard_bnn.out_of_core.run import run_out_of_core_training

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self._prepare("botiot", root, cascade_enabled=False)
            config = load_config(prepared.resolved_config_path)
            config["experiment"]["output_dir"] = str(root / "runs")

            run_dir = run_out_of_core_training(
                prepared.resolved_config_path,
                config=config,
                prepared_descriptor_path=prepared.descriptor_path,
            )

            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            operating_points = json.loads(
                (run_dir / "fixed_fpr_thresholds.json").read_text(encoding="utf-8")
            )
            self.assertFalse((run_dir / "tiny_model.pt").exists())
            self.assertFalse((run_dir / "cascade_calibration.json").exists())
            self.assertEqual(metrics["fixed_fpr"]["threshold_source"], "validation")
            self.assertEqual(
                operating_points["score_pipeline"],
                "routed_attack_score=1-p_benign",
            )
            self.assertEqual(
                pq.ParquetFile(run_dir / "predictions.parquet").metadata.num_rows,
                prepared.test_count,
            )


class BootstrapRunStageTests(unittest.TestCase):
    def _fixture(self, root: Path, *, prepare_only: bool, fail_botiot: bool = False):
        archive = root / "nbaiot.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("device_a/benign_traffic.csv", "mean,std\n1,2\n2,3\n")
            output.writestr("device_b/gafgyt_attacks/scan.csv", "mean,std\n8,9\n9,10\n")
        botiot = root / "botiot"
        botiot.mkdir()
        (botiot / "flows.csv").write_text(
            "category,subcategory,saddr,stime,bytes,rate\n"
            "Normal,Normal,10.0.0.1,1.0,100,1.0\n"
            "DDoS,TCP,10.0.0.2,2.0,200,2.0\n",
            encoding="utf-8",
        )
        calls: list[str] = []
        failures_remaining = [1 if fail_botiot else 0]
        generations: dict[str, int] = {}

        def preparer(_config: Path, **kwargs: object) -> object:
            descriptor = Path(str(kwargs["descriptor_path"]))
            dataset = descriptor.parent.name
            descriptor.parent.mkdir(parents=True, exist_ok=True)
            descriptor.write_text(
                json.dumps({"dataset": dataset, "valid": True}) + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(descriptor_path=str(descriptor))

        def verifier(descriptor: Path) -> object:
            payload = json.loads(descriptor.read_text(encoding="utf-8"))
            if payload.get("valid") is not True:
                raise RuntimeError("invalid prepared descriptor")
            return SimpleNamespace(
                to_dict=lambda: {
                    "dataset": payload["dataset"],
                    "descriptor_path": str(descriptor),
                }
            )

        def trainer(*, dataset: str, descriptor_path: Path, runs_root: Path) -> Path:
            calls.append(dataset)
            if dataset == "botiot" and failures_remaining[0]:
                failures_remaining[0] -= 1
                raise RuntimeError("injected second dataset failure")
            generations[dataset] = generations.get(dataset, 0) + 1
            run_dir = runs_root / f"{dataset}-run-{generations[dataset]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            edge_dir = run_dir / "edge"
            edge_dir.mkdir()
            artifacts = {
                "best_checkpoint": run_dir / "best_model.pt",
                "metrics": run_dir / "metrics.json",
                "predictions": run_dir / "predictions.parquet",
                "inference_contract": run_dir / "inference_contract.json",
                "weights": edge_dir / "bitguard_edge_weights.npz",
                "model": edge_dir / "bitguard_edge_model.onnx",
                "metadata": edge_dir / "bitguard_edge_metadata.json",
                "manifest": edge_dir / "bitguard_edge_manifest.json",
            }
            for name, path in artifacts.items():
                if name == "manifest":
                    continue
                path.write_bytes(f"{dataset}:{generations[dataset]}:{name}".encode())
            artifacts["manifest"].write_text(
                json.dumps(
                    {
                        "files": [
                            artifacts["weights"].name,
                            artifacts["model"].name,
                            artifacts["metadata"].name,
                            artifacts["manifest"].name,
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = {
                "prepared_descriptor": str(descriptor_path),
                "best_checkpoint": str(artifacts["best_checkpoint"]),
                "metrics": str(artifacts["metrics"]),
                "predictions": str(artifacts["predictions"]),
                "inference_contract": str(artifacts["inference_contract"]),
                "export": {
                    "output_dir": str(edge_dir),
                    "weights": str(artifacts["weights"]),
                    "manifest": str(artifacts["manifest"]),
                },
            }
            (run_dir / "run_summary.json").write_text(
                json.dumps(summary) + "\n", encoding="utf-8"
            )
            return run_dir

        options = BootstrapOptions(
            datasets=("botiot", "nbaiot"),
            botiot_source=botiot.resolve(),
            data_root=(root / "data").resolve(),
            runs_root=(root / "runs").resolve(),
            compute="cpu",
            prepare_only=prepare_only,
            install_system_tools=False,
            accepted_botiot_license=True,
            restart_stage=None,
        )
        dependencies = BootstrapDependencies(
            nbaiot_archive=archive,
            available_bytes=10**12,
            preparation_available_bytes=10**12,
            compute_resolver=lambda requested: {
                "requested": requested,
                "selected_profile": "cpu",
                "device": "cpu",
            },
            preparer=preparer,
            prepared_verifier=verifier,
            trainer=trainer,
        )
        return options, dependencies, calls

    def test_all_trains_nbaiot_then_botiot_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            options, dependencies, calls = self._fixture(
                Path(temporary), prepare_only=False
            )
            report = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(report["status"], "completed", msg=report.get("error"))
            self.assertEqual(calls, ["nbaiot", "botiot"])
            self.assertEqual(report["last_completed_stage"], "summarize")
            self.assertEqual(set(report["trained_runs"]), {"nbaiot", "botiot"})
            self.assertTrue(Path(report["reports"]["summary"]).is_file())
            self.assertEqual(
                set(report["dataset_statuses"]["nbaiot"]["artifacts"]),
                {
                    "checkpoint",
                    "metrics",
                    "predictions",
                    "inference_contract",
                    "export_manifest",
                    "export_file:bitguard_edge_weights.npz",
                    "export_file:bitguard_edge_model.onnx",
                    "export_file:bitguard_edge_metadata.json",
                },
            )

    def test_second_dataset_failure_keeps_first_completed_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            options, dependencies, calls = self._fixture(
                Path(temporary), prepare_only=False, fail_botiot=True
            )
            report = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed_stage"], "train")
            self.assertEqual(calls, ["nbaiot", "botiot"])
            self.assertEqual(
                report["dataset_statuses"]["nbaiot"]["status"], "completed"
            )
            self.assertEqual(report["dataset_statuses"]["botiot"]["status"], "failed")
            persisted = json.loads(
                Path(report["reports"]["training"]).read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["datasets"]["nbaiot"]["status"], "completed")

    def test_retry_after_second_dataset_failure_reuses_verified_first_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            options, dependencies, calls = self._fixture(
                Path(temporary), prepare_only=False, fail_botiot=True
            )
            failed = run_bootstrap(options, dependencies=dependencies)
            completed = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(failed["status"], "failed")
            self.assertEqual(
                completed["status"], "completed", msg=completed.get("error")
            )
            self.assertEqual(calls, ["nbaiot", "botiot", "botiot"])
            self.assertEqual(
                completed["dataset_statuses"]["nbaiot"]["run"],
                failed["dataset_statuses"]["nbaiot"]["run"],
            )

    def test_malformed_training_report_root_is_not_treated_as_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            options, dependencies, calls = self._fixture(
                Path(temporary), prepare_only=False, fail_botiot=True
            )
            failed = run_bootstrap(options, dependencies=dependencies)
            Path(failed["reports"]["training"]).write_text("[]\n", encoding="utf-8")

            completed = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(
                completed["status"], "completed", msg=completed.get("error")
            )
            self.assertEqual(calls, ["nbaiot", "botiot", "nbaiot", "botiot"])

    def test_changed_completed_artifact_forces_only_that_dataset_to_retrain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            options, dependencies, calls = self._fixture(
                Path(temporary), prepare_only=False
            )
            first = run_bootstrap(options, dependencies=dependencies)
            Path(first["dataset_statuses"]["nbaiot"]["metrics"]).write_text(
                "tampered", encoding="utf-8"
            )
            second = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(second["status"], "completed", msg=second.get("error"))
            self.assertEqual(calls, ["nbaiot", "botiot", "nbaiot"])

    def test_hardlinked_completed_artifact_is_never_certified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies, calls = self._fixture(root, prepare_only=False)
            first = run_bootstrap(options, dependencies=dependencies)
            metrics = Path(first["dataset_statuses"]["nbaiot"]["metrics"])
            foreign = root / "foreign-metrics.json"
            foreign.write_bytes(metrics.read_bytes())
            metrics.unlink()
            try:
                import os

                os.link(foreign, metrics)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"hard links unavailable: {error}")

            second = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(second["status"], "completed", second.get("error"))
            self.assertEqual(calls, ["nbaiot", "botiot", "nbaiot"])

    def test_linked_artifact_parent_is_never_certified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies, calls = self._fixture(root, prepare_only=False)
            first = run_bootstrap(options, dependencies=dependencies)
            run_dir = Path(first["dataset_statuses"]["nbaiot"]["run"])
            edge = run_dir / "edge"
            actual = run_dir / "edge-actual"
            edge.rename(actual)
            try:
                edge.symlink_to(actual, target_is_directory=True)
            except (NotImplementedError, OSError) as error:
                actual.rename(edge)
                self.skipTest(f"directory symlinks unavailable: {error}")

            second = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(second["status"], "completed", second.get("error"))
            self.assertEqual(calls, ["nbaiot", "botiot", "nbaiot"])

    def test_artifact_parent_swap_during_graph_hash_is_rejected(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies, _calls = self._fixture(root, prepare_only=False)
            first = run_bootstrap(options, dependencies=dependencies)
            status = first["dataset_statuses"]["nbaiot"]
            run_dir = Path(status["run"])
            descriptor = Path(status["descriptor"])
            summary = json.loads(
                (run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            edge = run_dir / "edge"
            retired = run_dir / "edge-retired"
            swapped = False
            original_digest = orchestrator._artifact_digest

            def swap_after_digest(graph_root: Path, path: Path):
                nonlocal swapped
                result = original_digest(graph_root, path)
                if not swapped and path.parent == edge:
                    edge.rename(retired)
                    import shutil

                    shutil.copytree(retired, edge)
                    swapped = True
                return result

            with (
                patch.object(
                    orchestrator, "_artifact_digest", side_effect=swap_after_digest
                ),
                self.assertRaisesRegex(
                    RuntimeError, "artifact graph changed|run summary changed"
                ),
            ):
                orchestrator._completed_training_status(
                    "nbaiot", descriptor, run_dir, summary
                )

            self.assertTrue(swapped)

    def test_run_summary_replacement_after_parse_cannot_certify_mixed_graph(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies, _calls = self._fixture(root, prepare_only=False)
            first = run_bootstrap(options, dependencies=dependencies)
            status = first["dataset_statuses"]["nbaiot"]
            run_dir = Path(status["run"])
            descriptor = Path(status["descriptor"])
            summary_path = run_dir / "run_summary.json"
            parsed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            alternate_metrics = run_dir / "metrics-alternate.json"
            alternate_metrics.write_bytes(b"alternate")
            replacement = copy.deepcopy(parsed_summary)
            replacement["metrics"] = str(alternate_metrics)
            original_digest = orchestrator._regular_digest
            replaced = False

            def replace_after_summary_parse(path: Path):
                nonlocal replaced
                if not replaced and path == descriptor:
                    summary_path.write_text(
                        json.dumps(replacement) + "\n", encoding="utf-8"
                    )
                    replaced = True
                return original_digest(path)

            with (
                patch.object(
                    orchestrator,
                    "_regular_digest",
                    side_effect=replace_after_summary_parse,
                ),
                self.assertRaisesRegex(RuntimeError, "summary.*changed|summary.*match"),
            ):
                orchestrator._completed_training_status(
                    "nbaiot", descriptor, run_dir, parsed_summary
                )

            self.assertTrue(replaced)
            self.assertTrue(alternate_metrics.is_file())

    def test_artifact_mutation_during_final_summary_read_is_rejected(self) -> None:
        from bitguard_bnn.bootstrap import orchestrator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies, _calls = self._fixture(root, prepare_only=False)
            first = run_bootstrap(options, dependencies=dependencies)
            status = first["dataset_statuses"]["nbaiot"]
            run_dir = Path(status["run"])
            descriptor = Path(status["descriptor"])
            summary_path = run_dir / "run_summary.json"
            parsed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = Path(parsed_summary["metrics"])
            original_read = orchestrator._read_artifact_json_record
            summary_reads = 0

            def mutate_during_second_summary_read(
                graph_root: Path, path: Path, *, subject: str
            ):
                nonlocal summary_reads
                result = original_read(graph_root, path, subject=subject)
                if path == summary_path:
                    summary_reads += 1
                    if summary_reads == 2:
                        metrics.write_bytes(b"mutated-during-final-summary-read")
                return result

            with (
                patch.object(
                    orchestrator,
                    "_read_artifact_json_record",
                    side_effect=mutate_during_second_summary_read,
                ),
                self.assertRaisesRegex(RuntimeError, "artifact graph changed"),
            ):
                orchestrator._completed_training_status(
                    "nbaiot", descriptor, run_dir, parsed_summary
                )

            self.assertEqual(summary_reads, 2)
            self.assertEqual(metrics.read_bytes(), b"mutated-during-final-summary-read")

    def test_inference_contract_and_export_weight_changes_retrain_only_owner(
        self,
    ) -> None:
        scenarios = (
            ("inference_contract", "tamper"),
            ("inference_contract", "delete"),
            ("export_file:bitguard_edge_weights.npz", "tamper"),
            ("export_file:bitguard_edge_weights.npz", "delete"),
        )
        for artifact_name, operation in scenarios:
            with (
                self.subTest(artifact=artifact_name, operation=operation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                options, dependencies, calls = self._fixture(
                    Path(temporary), prepare_only=False
                )
                first = run_bootstrap(options, dependencies=dependencies)
                artifact = Path(
                    first["dataset_statuses"]["nbaiot"]["artifacts"][artifact_name][
                        "path"
                    ]
                )
                if operation == "delete":
                    artifact.unlink()
                else:
                    artifact.write_bytes(b"tampered")

                second = run_bootstrap(options, dependencies=dependencies)

                self.assertEqual(second["status"], "completed", second.get("error"))
                self.assertEqual(calls, ["nbaiot", "botiot", "nbaiot"])

    def test_training_artifact_graph_rejects_foreign_and_traversal_locators(
        self,
    ) -> None:
        from bitguard_bnn.bootstrap import orchestrator

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            edge_dir = run_dir / "edge"
            edge_dir.mkdir(parents=True)
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            base = {
                "best_checkpoint": run_dir / "best_model.pt",
                "metrics": run_dir / "metrics.json",
                "predictions": run_dir / "predictions.parquet",
                "inference_contract": run_dir / "inference_contract.json",
            }
            for path in base.values():
                path.write_bytes(b"artifact")
            manifest = edge_dir / "manifest.json"
            summary: dict[str, object] = {
                **{name: str(path) for name, path in base.items()},
                "export": {
                    "output_dir": str(edge_dir),
                    "weights": str(edge_dir / "weights.bin"),
                    "manifest": str(manifest),
                },
            }
            (edge_dir / "weights.bin").write_bytes(b"weights")

            foreign_summary = copy.deepcopy(summary)
            foreign_summary["inference_contract"] = str(outside)
            manifest.write_text(
                json.dumps({"files": ["weights.bin", "manifest.json"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "escapes its run directory"):
                orchestrator._training_artifact_paths(foreign_summary, run_dir)

            manifest.write_text(
                json.dumps({"files": ["../outside.bin", "manifest.json"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "canonical relative path"):
                orchestrator._training_artifact_paths(summary, run_dir)

    def test_restart_train_atomically_refreshes_summary_run_locators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            options, dependencies, calls = self._fixture(
                Path(temporary), prepare_only=False
            )
            first = run_bootstrap(options, dependencies=dependencies)
            restarted = run_bootstrap(
                dataclasses.replace(options, restart_stage="train"),
                dependencies=dependencies,
            )
            summary = json.loads(
                Path(restarted["reports"]["summary"]).read_text(encoding="utf-8")
            )

            self.assertEqual(
                restarted["status"], "completed", msg=restarted.get("error")
            )
            self.assertEqual(calls, ["nbaiot", "botiot", "nbaiot", "botiot"])
            self.assertNotEqual(
                first["trained_runs"]["nbaiot"], restarted["trained_runs"]["nbaiot"]
            )
            self.assertEqual(
                summary["runs"]["nbaiot"], restarted["trained_runs"]["nbaiot"]
            )

    def test_prepare_only_skips_train_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            options, dependencies, calls = self._fixture(
                Path(temporary), prepare_only=True
            )
            report = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(report["status"], "prepared", msg=report.get("error"))
            self.assertEqual(calls, [])
            self.assertNotIn("train", report["executed_stages"])
            self.assertNotIn("summarize", report["executed_stages"])


if __name__ == "__main__":
    unittest.main()
