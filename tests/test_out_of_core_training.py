from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bitguard_bnn.config import DEFAULTS, seed_everything, validate_config
from bitguard_bnn.models import build_model
from bitguard_bnn.out_of_core.dataset import (
    DataCursor,
    ParquetTrainingDataset,
    _ShardEntry,
)


_VALIDATION_CONTRACT = {
    "algorithm": "fixture.complete-validation.v1",
    "split_fingerprint": "split-fingerprint",
    "cache_layout_fingerprint": "fixture-cache-layout",
}


def _write_malicious_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):
        return (_write_malicious_marker, (self.marker,))


class _FakeDataset:
    def __init__(self, batches: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.batches = batches
        self.batch_size = max(len(features) for features, _ in batches)
        self.seed = 17
        self.shuffle_buffer_rows = 32
        self.row_count = sum(len(features) for features, _ in batches)
        self.split = "train"
        self.manifest_fingerprint = "shard-fingerprint"
        self.preprocessing_fingerprint = "preprocessor-fingerprint"
        self.descriptor_path = "fixture-prepared.json"
        self.entries = (SimpleNamespace(rows=self.row_count),)
        self.epoch = 0
        self.cursor: DataCursor | None = None

    def set_epoch(self, epoch: int, cursor: DataCursor | None = None) -> None:
        if cursor is not None and cursor.epoch != epoch:
            raise ValueError("resume cursor epoch does not match dataset epoch")
        self.epoch = epoch
        self.cursor = cursor

    def permuted_shards(self, epoch: int | None = None):
        del epoch
        return self.entries


class _ActualLoaderDataset(ParquetTrainingDataset):
    """In-memory rows driven through the production ordered DataLoader path."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._features = features
        self._labels = labels
        self.batch_size = 2
        self.seed = 17
        self.shuffle_buffer_rows = 32
        self.row_count = len(features)
        self.split = "train"
        self.manifest_fingerprint = "shard-fingerprint"
        self.preprocessing_fingerprint = "preprocessor-fingerprint"
        self.descriptor_path = "fixture-prepared.json"
        self.entries = (
            _ShardEntry(
                path="in-memory.parquet",
                fingerprint="f" * 64,
                rows=len(features),
                label="benign",
                row_group_rows=(len(features),),
            ),
        )
        self.epoch = 0
        self.cursor: DataCursor | None = None
        self._max_pending_chunks_observed = 0
        self._worker_ids_observed: set[int] = set()

    def permuted_shards(self, epoch: int | None = None):
        del epoch
        return self.entries

    def __iter__(self):
        import torch

        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        yield {
            "_chunk_ordinal": 0,
            "_worker_id": worker_id,
            "_shard_position": 0,
            "_chunk_position": 0,
            "_last_chunk": True,
            "features": self._features.copy(),
            "unencoded": self._features.copy(),
            "labels": self._labels.copy(),
            "row_uid": np.asarray(
                [f"row-{index}" for index in range(self.row_count)], dtype=object
            ),
            "metadata": {},
            "boolean_raw": {},
        }


def _ordered_fake_batches(dataset: _FakeDataset, *, num_workers: int = 0):
    del num_workers
    start = 0
    optimizer_step = 0
    if dataset.cursor is not None:
        optimizer_step = dataset.cursor.optimizer_step
        start = (
            len(dataset.batches)
            if dataset.cursor.shard_position == len(dataset.entries)
            else dataset.cursor.batch_position
        )
    for index in range(start, len(dataset.batches)):
        features, labels = dataset.batches[index]
        cursor = DataCursor(dataset.epoch, 0, index, optimizer_step)
        if index + 1 == len(dataset.batches):
            next_cursor = DataCursor(
                dataset.epoch, len(dataset.entries), 0, optimizer_step + 1
            )
        else:
            next_cursor = DataCursor(
                dataset.epoch, 0, index + 1, optimizer_step + 1
            )
        yield {
            "features": features,
            "labels": labels,
            "cursor": cursor,
            "next_cursor": next_cursor,
        }
        optimizer_step += 1


def _prepared() -> SimpleNamespace:
    payload = {
        "preparation_fingerprint": "preparation-fingerprint",
        "shard_fingerprint": "shard-fingerprint",
        "preprocessing_fingerprint": "preprocessor-fingerprint",
        "normalized_source_fingerprint": "normalized-source-fingerprint",
        "split_fingerprint": "split-fingerprint",
    }
    return SimpleNamespace(
        **payload,
        to_dict=lambda: {**payload, "fingerprint": "descriptor-fingerprint"},
    )


class OutOfCoreTrainingTests(unittest.TestCase):
    def _config(self, *, epochs: int = 2, dropout: float = 0.25) -> dict:
        config = copy.deepcopy(DEFAULTS)
        config["experiment"]["seed"] = 17
        config["model"].update(
            {
                "type": "vanilla_bnn",
                "hidden_dims": [8],
                "dropout": dropout,
                "binary_first_layer": True,
            }
        )
        config["preprocess"]["open_set"]["enabled"] = False
        config["training"].update(
            {
                "epochs": epochs,
                "batch_size": 2,
                "patience": 10,
                "num_workers": 0,
                "device": "cpu",
                "amp": False,
                "checkpoint_every_steps": 1,
                "shuffle_buffer_rows": 32,
            }
        )
        return config

    @staticmethod
    def _dataset() -> _FakeDataset:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(6, 4)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(np.int64)
        return _FakeDataset(
            [(features[start : start + 2], labels[start : start + 2]) for start in range(0, 6, 2)]
        )

    @staticmethod
    def _actual_loader_dataset() -> _ActualLoaderDataset:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(6, 4)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(np.int64)
        return _ActualLoaderDataset(features, labels)

    @staticmethod
    def _model(config: dict):
        return build_model(
            config,
            input_dim=4,
            output_dim=2,
            input_groups=np.arange(4, dtype=np.int64),
            feature_costs=np.ones(4, dtype=np.float32),
        )

    @staticmethod
    def _validation(_model, _device) -> dict[str, float]:
        return {
            "validation_macro_f1": 0.75,
            "validation_macro_auprc": 0.80,
            "validation_attack_recall": 0.90,
        }

    def test_mid_epoch_next_cursor_resume_matches_dropout_training_exactly(self) -> None:
        import torch

        from bitguard_bnn.out_of_core.trainer import (
            StreamingTrainingInterrupted,
            fit_neural_streaming,
        )

        config = self._config()
        resumed_validation_calls: list[int] = []

        def counted_validation(model, device):
            resumed_validation_calls.append(1)
            return self._validation(model, device)

        with patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ):
            seed_everything(17)
            uninterrupted = fit_neural_streaming(
                self._model(config),
                self._actual_loader_dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                self._validation,
                _VALIDATION_CONTRACT,
            )
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "stream-state.pt"
                seed_everything(17)
                with self.assertRaises(StreamingTrainingInterrupted):
                    fit_neural_streaming(
                        self._model(config),
                        self._actual_loader_dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        counted_validation,
                        _VALIDATION_CONTRACT,
                        checkpoint_path=checkpoint,
                        stop_after_optimizer_step=2,
                    )
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(state["format_version"], 4)
                self.assertEqual(
                    state["cursor"],
                    {
                        "epoch": 1,
                        "shard_position": 0,
                        "batch_position": 2,
                        "optimizer_step": 2,
                    },
                )
                self.assertIsNone(state["best_state_dict"])
                self.assertEqual(resumed_validation_calls, [])

                incompatible = self._config()
                incompatible["model"]["dropout"] = 0.5
                with self.assertRaisesRegex(ValueError, "scientific signature"):
                    fit_neural_streaming(
                        self._model(incompatible),
                        self._actual_loader_dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        incompatible,
                        self._validation,
                        _VALIDATION_CONTRACT,
                        resume_from=checkpoint,
                    )

                changed_validation = {
                    **_VALIDATION_CONTRACT,
                    "split_fingerprint": "different-split",
                }
                with self.assertRaisesRegex(ValueError, "scientific signature"):
                    fit_neural_streaming(
                        self._model(config),
                        self._actual_loader_dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        self._validation,
                        changed_validation,
                        resume_from=checkpoint,
                    )

                tampered = dict(state)
                tampered["global_optimizer_step"] = 1
                torch.save(tampered, checkpoint)
                with self.assertRaisesRegex(ValueError, "optimizer step and cursor"):
                    fit_neural_streaming(
                        self._model(config),
                        self._actual_loader_dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        self._validation,
                        _VALIDATION_CONTRACT,
                        resume_from=checkpoint,
                    )

                def invalid_shard(value):
                    value["cursor"]["shard_position"] = 99

                def invalid_batch(value):
                    value["cursor"]["batch_position"] = 99

                def inconsistent_rows(value):
                    value["partial_seen_rows"] -= 1

                def inconsistent_step(value):
                    value["cursor"]["optimizer_step"] -= 1
                    value["global_optimizer_step"] -= 1

                cursor_corruptions = (
                    (invalid_shard, "logical batch boundary"),
                    (invalid_batch, "logical batch boundary"),
                    (inconsistent_rows, "cursor, rows, and step"),
                    (inconsistent_step, "cursor, rows, and step"),
                )
                for corrupt, message in cursor_corruptions:
                    damaged = copy.deepcopy(state)
                    corrupt(damaged)
                    torch.save(damaged, checkpoint)
                    candidate = self._model(config)
                    resume_dataset = self._actual_loader_dataset()
                    before = {
                        name: value.detach().clone()
                        for name, value in candidate.state_dict().items()
                    }
                    with self.subTest(cursor=message), self.assertRaisesRegex(
                        ValueError, message
                    ):
                        fit_neural_streaming(
                            candidate,
                            resume_dataset,
                            {"benign": 3, "flood_like": 3},
                            ("benign", "flood_like"),
                            config,
                            self._validation,
                            _VALIDATION_CONTRACT,
                            resume_from=checkpoint,
                        )
                    for name, expected in before.items():
                        self.assertTrue(
                            torch.equal(expected, candidate.state_dict()[name])
                        )
                    self.assertEqual(resume_dataset.epoch, 0)
                    self.assertIsNone(resume_dataset.cursor)
                torch.save(state, checkpoint)

                seed_everything(999)
                resumed = fit_neural_streaming(
                    self._model(config),
                    self._actual_loader_dataset(),
                    {"benign": 3, "flood_like": 3},
                    ("benign", "flood_like"),
                    config,
                    counted_validation,
                    _VALIDATION_CONTRACT,
                    checkpoint_path=checkpoint,
                    resume_from=checkpoint,
                )
                final_state = torch.load(
                    checkpoint, map_location="cpu", weights_only=True
                )
                self.assertEqual(final_state["scheduler_state_dict"]["last_epoch"], 2)

        self.assertEqual(
            uninterrupted.history.to_dict(orient="records"),
            resumed.history.to_dict(orient="records"),
        )
        for name, expected in uninterrupted.model.state_dict().items():
            self.assertTrue(
                torch.equal(expected.cpu(), resumed.model.state_dict()[name].cpu()), name
            )
        self.assertEqual(len(resumed_validation_calls), 2)

    def test_one_epoch_streaming_matches_array_fit_with_fixed_batch_order(self) -> None:
        import torch

        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming
        from bitguard_bnn.trainer import (
            _fit_neural,
            _predict_neural_probabilities,
            neural_validation_metrics,
        )

        rng = np.random.default_rng(17)
        train_features = rng.normal(size=(6, 4)).astype(np.float32)
        train_labels = (train_features[:, 0] > 0).astype(np.int64)
        validation_features = rng.normal(size=(4, 4)).astype(np.float32)
        validation_labels = (validation_features[:, 0] > 0).astype(np.int64)
        batches = [
            (train_features[start : start + 2], train_labels[start : start + 2])
            for start in range(0, 6, 2)
        ]
        config = self._config(epochs=1, dropout=0.0)

        class FixedDataLoader:
            def __init__(self, dataset, **_kwargs) -> None:
                self.dataset = dataset
                self._iterator = None

            def __iter__(self):
                features, labels = self.dataset.tensors
                return iter(
                    [
                        (features[start : start + 2], labels[start : start + 2])
                        for start in range(0, len(features), 2)
                    ]
                )

        seed_everything(17)
        with patch("torch.utils.data.DataLoader", FixedDataLoader):
            array_result = _fit_neural(
                self._model(config),
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                np.ones(2, dtype=np.float32),
                config,
            )

        def complete_validation(model, device):
            probabilities = _predict_neural_probabilities(
                model, validation_features, 2, device
            )
            return neural_validation_metrics(
                validation_labels, probabilities, config["training"]
            )

        seed_everything(17)
        with patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ), patch(
            "bitguard_bnn.out_of_core.trainer.iter_ordered_batches",
            side_effect=_ordered_fake_batches,
        ):
            streaming_result = fit_neural_streaming(
                self._model(config),
                _FakeDataset(batches),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                complete_validation,
                _VALIDATION_CONTRACT,
            )

        self.assertEqual(
            array_result.history.to_dict(orient="records"),
            streaming_result.history.to_dict(orient="records"),
        )
        for name, expected in array_result.model.state_dict().items():
            self.assertTrue(
                torch.equal(
                    expected.cpu(), streaming_result.model.state_dict()[name].cpu()
                ),
                name,
            )

    def test_terminal_training_cursor_resumes_before_scheduler_exactly(self) -> None:
        import warnings

        import torch

        from bitguard_bnn.out_of_core.trainer import (
            StreamingTrainingInterrupted,
            fit_neural_streaming,
        )

        config = self._config(epochs=2, dropout=0.25)
        with tempfile.TemporaryDirectory() as directory, patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ):
            checkpoint = Path(directory) / "terminal-training.pt"
            seed_everything(17)
            uninterrupted = fit_neural_streaming(
                self._model(config),
                self._actual_loader_dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                self._validation,
                _VALIDATION_CONTRACT,
            )
            seed_everything(17)
            with self.assertRaises(StreamingTrainingInterrupted):
                fit_neural_streaming(
                    self._model(config),
                    self._actual_loader_dataset(),
                    {"benign": 3, "flood_like": 3},
                    ("benign", "flood_like"),
                    config,
                    self._validation,
                    _VALIDATION_CONTRACT,
                    checkpoint_path=checkpoint,
                    stop_after_optimizer_step=3,
                )
            interrupted = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            self.assertEqual(interrupted["epoch_phase"], "training")
            self.assertEqual(interrupted["partial_seen_rows"], 6)
            self.assertEqual(interrupted["scheduler_state_dict"]["last_epoch"], 0)
            seed_everything(999)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                resumed = fit_neural_streaming(
                    self._model(config),
                    self._actual_loader_dataset(),
                    {"benign": 3, "flood_like": 3},
                    ("benign", "flood_like"),
                    config,
                    self._validation,
                    _VALIDATION_CONTRACT,
                    checkpoint_path=checkpoint,
                    resume_from=checkpoint,
                )
            self.assertFalse(
                any("lr_scheduler.step" in str(item.message) for item in caught)
            )
            completed = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        self.assertEqual(completed["scheduler_state_dict"]["last_epoch"], 2)
        self.assertEqual(
            uninterrupted.history.to_dict(orient="records"),
            resumed.history.to_dict(orient="records"),
        )
        for name, expected in uninterrupted.model.state_dict().items():
            self.assertTrue(
                torch.equal(expected.cpu(), resumed.model.state_dict()[name].cpu()),
                name,
            )

    def test_partial_epoch_totals_are_weighted_by_actual_batch_rows(self) -> None:
        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming

        rng = np.random.default_rng(3)
        batches = [
            (rng.normal(size=(2, 4)).astype(np.float32), np.array([0, 1])),
            (rng.normal(size=(3, 4)).astype(np.float32), np.array([1, 0, 1])),
        ]
        dataset = _FakeDataset(batches)
        config = self._config(epochs=1, dropout=0.0)
        config["training"]["batch_size"] = 3

        def row_metric(*_args, **kwargs):
            rows = len(kwargs["features"])
            kwargs["optimizer"].step()
            return {name: float(rows) for name in ("loss", "detection", "feature_cost", "fn", "fp")}

        with patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ), patch(
            "bitguard_bnn.out_of_core.trainer.iter_ordered_batches",
            side_effect=_ordered_fake_batches,
        ), patch(
            "bitguard_bnn.out_of_core.trainer.neural_train_step", side_effect=row_metric
        ):
            result = fit_neural_streaming(
                self._model(config),
                dataset,
                {"benign": 2, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                self._validation,
                _VALIDATION_CONTRACT,
            )

        self.assertAlmostEqual(result.history.iloc[0]["train_loss"], 13.0 / 5.0)

    def test_tiny_transform_accepts_numpy_integer_feature_indices(self) -> None:
        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming

        config = self._config(epochs=1, dropout=0.0)
        tiny_model = build_model(
            config,
            input_dim=2,
            output_dim=2,
            input_groups=np.arange(2, dtype=np.int64),
            feature_costs=np.ones(2, dtype=np.float32),
        )
        with patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ), patch(
            "bitguard_bnn.out_of_core.trainer.iter_ordered_batches",
            side_effect=_ordered_fake_batches,
        ):
            result = fit_neural_streaming(
                tiny_model,
                self._dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                self._validation,
                _VALIDATION_CONTRACT,
                training_role="tiny",
                feature_indices=np.asarray([0, 2], dtype=np.int64),
                binary_attack_target=True,
            )
        self.assertEqual(result.history["epoch"].tolist(), [1])

    def test_class_counts_must_exactly_describe_the_train_split(self) -> None:
        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming

        config = self._config(epochs=1, dropout=0.0)
        invalid = (
            ({"benign": 6}, "keys"),
            ({"benign": 3, "flood_like": 3, "extra": 1}, "keys"),
            ({"benign": True, "flood_like": 5}, "positive integers"),
            ({"benign": 2, "flood_like": 3}, "total"),
        )
        for counts, message in invalid:
            with self.subTest(counts=counts), self.assertRaisesRegex(
                ValueError, message
            ):
                fit_neural_streaming(
                    self._model(config),
                    self._dataset(),
                    counts,
                    ("benign", "flood_like"),
                    config,
                    self._validation,
                    _VALIDATION_CONTRACT,
                )

    def test_validation_metric_bounds_fail_safely_and_resume(self) -> None:
        import torch

        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming

        config = self._config(epochs=1, dropout=0.0)
        invalid_metrics = {
            "validation_macro_f1": 1.01,
            "validation_macro_auprc": 0.8,
            "validation_attack_recall": 0.9,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ), patch(
            "bitguard_bnn.out_of_core.trainer.iter_ordered_batches",
            side_effect=_ordered_fake_batches,
        ):
            checkpoint = Path(directory) / "invalid-validation.pt"
            with self.assertRaisesRegex(ValueError, r"finite in \[0, 1\]"):
                fit_neural_streaming(
                    self._model(config),
                    self._dataset(),
                    {"benign": 3, "flood_like": 3},
                    ("benign", "flood_like"),
                    config,
                    lambda _model, _device: invalid_metrics,
                    _VALIDATION_CONTRACT,
                    checkpoint_path=checkpoint,
                )
            interrupted = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            self.assertEqual(interrupted["epoch_phase"], "validation")
            self.assertEqual(interrupted["history"], [])
            resumed = fit_neural_streaming(
                self._model(config),
                self._dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                self._validation,
                _VALIDATION_CONTRACT,
                resume_from=checkpoint,
            )
        self.assertEqual(resumed.history["epoch"].tolist(), [1])

    def test_resuming_early_stop_boundary_does_not_repeat_epoch_or_validation(self) -> None:
        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming

        config = self._config(epochs=3, dropout=0.0)
        config["training"]["patience"] = 1
        calls: list[int] = []

        def validation(model, device):
            calls.append(1)
            return self._validation(model, device)

        with tempfile.TemporaryDirectory() as directory, patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ), patch(
            "bitguard_bnn.out_of_core.trainer.iter_ordered_batches",
            side_effect=_ordered_fake_batches,
        ):
            checkpoint = Path(directory) / "early-stop.pt"
            seed_everything(17)
            stopped = fit_neural_streaming(
                self._model(config),
                self._dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                validation,
                _VALIDATION_CONTRACT,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(stopped.history["epoch"].tolist(), [1, 2])
            self.assertEqual(len(calls), 2)
            calls.clear()
            seed_everything(999)
            resumed = fit_neural_streaming(
                self._model(config),
                self._dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                validation,
                _VALIDATION_CONTRACT,
                resume_from=checkpoint,
            )

        self.assertEqual(calls, [])
        self.assertEqual(
            stopped.history.to_dict(orient="records"),
            resumed.history.to_dict(orient="records"),
        )

    def test_streaming_corruption_is_rejected_before_model_mutation(self) -> None:
        import torch

        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming

        config = self._config(epochs=2, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory, patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ), patch(
            "bitguard_bnn.out_of_core.trainer.iter_ordered_batches",
            side_effect=_ordered_fake_batches,
        ):
            checkpoint = Path(directory) / "stream-corruption.pt"
            seed_everything(17)
            fit_neural_streaming(
                self._model(config),
                self._dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                self._validation,
                _VALIDATION_CONTRACT,
                checkpoint_path=checkpoint,
            )
            original = torch.load(checkpoint, map_location="cpu", weights_only=True)

            def best_nan(state):
                state["best_metric"] = float("nan")

            def negative_stale(state):
                state["stale_epochs"] = -1

            def history_nan(state):
                state["history"][0]["train_loss"] = float("nan")

            def scheduler_ahead(state):
                state["scheduler_state_dict"]["last_epoch"] += 1

            def invalid_numpy_rng(state):
                state["numpy_rng_state"]["position"] = 625

            def unknown_field(state):
                state["unknown"] = 1

            def legacy_format(state):
                state["format_version"] = 3

            corruptions = (
                (best_nan, "best"),
                (negative_stale, "best/stale"),
                (history_nan, "non-finite"),
                (scheduler_ahead, "scheduler phase"),
                (invalid_numpy_rng, "NumPy RNG"),
                (unknown_field, "fields"),
                (legacy_format, "unsupported"),
            )
            for corrupt, message in corruptions:
                damaged = copy.deepcopy(original)
                corrupt(damaged)
                torch.save(damaged, checkpoint)
                candidate = self._model(config)
                before = {
                    name: value.detach().clone()
                    for name, value in candidate.state_dict().items()
                }
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    fit_neural_streaming(
                        candidate,
                        self._dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        self._validation,
                        _VALIDATION_CONTRACT,
                        resume_from=checkpoint,
                    )
                for name, expected in before.items():
                    self.assertTrue(torch.equal(expected, candidate.state_dict()[name]))

            tolerated = copy.deepcopy(original)
            tolerated["history"][1]["validation_selection_score"] = (
                tolerated["history"][0]["validation_selection_score"] + 5e-7
            )
            torch.save(tolerated, checkpoint)
            resumed = fit_neural_streaming(
                self._model(config),
                self._dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                self._validation,
                _VALIDATION_CONTRACT,
                resume_from=checkpoint,
            )
            self.assertEqual(resumed.best_epoch, 1)

    def test_validation_failure_resume_does_not_repeat_scheduler_or_optimizer(self) -> None:
        import torch

        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming
        from bitguard_bnn.trainer import neural_train_step as real_train_step

        config = self._config(epochs=1, dropout=0.0)
        with patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ), patch(
            "bitguard_bnn.out_of_core.trainer.iter_ordered_batches",
            side_effect=_ordered_fake_batches,
        ):
            seed_everything(17)
            uninterrupted = fit_neural_streaming(
                self._model(config),
                self._dataset(),
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                config,
                self._validation,
                _VALIDATION_CONTRACT,
            )
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "validation-phase.pt"
                seed_everything(17)
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    fit_neural_streaming(
                        self._model(config),
                        self._dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        lambda _model, _device: (_ for _ in ()).throw(
                            RuntimeError("validation failed")
                        ),
                        _VALIDATION_CONTRACT,
                        checkpoint_path=checkpoint,
                    )
                interrupted = torch.load(
                    checkpoint, map_location="cpu", weights_only=True
                )
                self.assertEqual(interrupted["epoch_phase"], "validation")
                self.assertEqual(interrupted["scheduler_state_dict"]["last_epoch"], 1)

                tampered = dict(interrupted)
                tampered["partial_seen_rows"] = interrupted["partial_seen_rows"] - 1
                torch.save(tampered, checkpoint)
                with self.assertRaisesRegex(
                    ValueError, "cursor, rows, and step"
                ):
                    fit_neural_streaming(
                        self._model(config),
                        self._dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        self._validation,
                        _VALIDATION_CONTRACT,
                        resume_from=checkpoint,
                    )
                torch.save(interrupted, checkpoint)

                seed_everything(999)
                with patch(
                    "bitguard_bnn.out_of_core.trainer.neural_train_step",
                    wraps=real_train_step,
                ) as resumed_step:
                    resumed = fit_neural_streaming(
                        self._model(config),
                        self._dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        self._validation,
                        _VALIDATION_CONTRACT,
                        checkpoint_path=checkpoint,
                        resume_from=checkpoint,
                    )
                resumed_step.assert_not_called()
                final_state = torch.load(
                    checkpoint, map_location="cpu", weights_only=True
                )

        self.assertEqual(final_state["scheduler_state_dict"]["last_epoch"], 1)
        self.assertEqual(
            uninterrupted.history.to_dict(orient="records"),
            resumed.history.to_dict(orient="records"),
        )
        for name, expected in uninterrupted.model.state_dict().items():
            self.assertTrue(
                torch.equal(expected.cpu(), resumed.model.state_dict()[name].cpu()), name
            )

    def test_training_step_failure_closes_stream_iterator(self) -> None:
        from bitguard_bnn.out_of_core.trainer import fit_neural_streaming

        dataset = self._dataset()

        class ClosingIterator:
            def __init__(self) -> None:
                self.closed = False
                self._source = iter(_ordered_fake_batches(dataset))

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._source)

            def close(self) -> None:
                self.closed = True

        iterator = ClosingIterator()
        with patch(
            "bitguard_bnn.out_of_core.trainer.verify_prepared_dataset",
            return_value=_prepared(),
        ), patch(
            "bitguard_bnn.out_of_core.trainer.iter_ordered_batches",
            return_value=iterator,
        ), patch(
            "bitguard_bnn.out_of_core.trainer.neural_train_step",
            side_effect=RuntimeError("objective failed"),
        ), self.assertRaisesRegex(RuntimeError, "objective failed"):
            fit_neural_streaming(
                self._model(self._config(epochs=1)),
                dataset,
                {"benign": 3, "flood_like": 3},
                ("benign", "flood_like"),
                self._config(epochs=1),
                self._validation,
                _VALIDATION_CONTRACT,
            )
        self.assertTrue(iterator.closed)

    def test_array_training_step_failure_closes_its_loader_iterator(self) -> None:
        from bitguard_bnn.trainer import _fit_neural

        rng = np.random.default_rng(9)
        features = rng.normal(size=(6, 4)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(np.int64)
        config = self._config(epochs=1, dropout=0.0)
        with patch(
            "bitguard_bnn.trainer.neural_train_step",
            side_effect=RuntimeError("array objective failed"),
        ), patch(
            "bitguard_bnn.trainer._shutdown_persistent_workers"
        ) as close, self.assertRaisesRegex(RuntimeError, "array objective failed"):
            _fit_neural(
                self._model(config),
                features[:4],
                labels[:4],
                features[4:],
                labels[4:],
                np.ones(2, dtype=np.float32),
                config,
            )
        close.assert_called_once()

    def test_array_resume_binds_data_teacher_and_checkpoint_semantics(self) -> None:
        import torch

        from bitguard_bnn.trainer import _fit_neural

        rng = np.random.default_rng(23)
        arrays = [
            rng.normal(size=(8, 4)).astype(np.float32),
            np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
            rng.normal(size=(4, 4)).astype(np.float32),
            np.asarray([0, 1, 0, 1], dtype=np.int64),
        ]
        weights = np.ones(2, dtype=np.float32)
        config = self._config(epochs=2, dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "array-state.pt"
            seed_everything(23)
            teacher = self._model(config)
            _fit_neural(
                self._model(config),
                *arrays,
                weights,
                config,
                teacher,
                checkpoint_path=checkpoint,
                stop_after_epoch=1,
            )
            original = torch.load(checkpoint, map_location="cpu", weights_only=True)

            for index, name in enumerate(
                ("x_train", "y_train", "x_validation", "y_validation")
            ):
                changed = [value.copy() for value in arrays]
                changed[index].flat[0] = changed[index].flat[0] + 1
                with self.subTest(array=name), self.assertRaisesRegex(
                    ValueError, "does not match"
                ):
                    _fit_neural(
                        self._model(config),
                        *changed,
                        weights,
                        config,
                        teacher,
                        resume_from=checkpoint,
                    )

            changed_weights = weights.copy()
            changed_weights[0] = 2.0
            with self.assertRaisesRegex(ValueError, "does not match"):
                _fit_neural(
                    self._model(config),
                    *arrays,
                    changed_weights,
                    config,
                    teacher,
                    resume_from=checkpoint,
                )

            changed_teacher = copy.deepcopy(teacher)
            with torch.no_grad():
                next(changed_teacher.parameters()).add_(1.0)
            with self.assertRaisesRegex(ValueError, "does not match"):
                _fit_neural(
                    self._model(config),
                    *arrays,
                    weights,
                    config,
                    changed_teacher,
                    resume_from=checkpoint,
                )

            def unsupported_format(state):
                state["format_version"] = 999

            def unknown_field(state):
                state["unknown"] = 1

            def history_nan(state):
                state["history"][0]["train_loss"] = float("nan")

            def best_nan(state):
                state["best_metric"] = float("nan")

            def negative_stale(state):
                state["stale_epochs"] = -1

            def scheduler_ahead(state):
                state["scheduler_state_dict"]["last_epoch"] += 1

            def invalid_rng(state):
                state["numpy_rng_state"]["position"] = 625

            def out_of_range_rng_word(state):
                state["numpy_rng_state"]["values"][0] = 2**32

            corruptions = (
                (unsupported_format, "unsupported"),
                (unknown_field, "fields"),
                (history_nan, "non-finite"),
                (best_nan, "best/stale"),
                (negative_stale, "best/stale"),
                (scheduler_ahead, "scheduler phase"),
                (invalid_rng, "NumPy RNG"),
                (out_of_range_rng_word, "NumPy RNG"),
            )
            for corrupt, message in corruptions:
                damaged = copy.deepcopy(original)
                corrupt(damaged)
                torch.save(damaged, checkpoint)
                candidate = self._model(config)
                before = {
                    name: value.detach().clone()
                    for name, value in candidate.state_dict().items()
                }
                with self.subTest(corruption=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    _fit_neural(
                        candidate,
                        *arrays,
                        weights,
                        config,
                        teacher,
                        resume_from=checkpoint,
                    )
                for name, expected in before.items():
                    self.assertTrue(torch.equal(expected, candidate.state_dict()[name]))

    def test_array_worker_rng_isolated_for_exact_resume(self) -> None:
        import torch

        from bitguard_bnn.trainer import _fit_neural

        rng = np.random.default_rng(31)
        values = rng.normal(size=(12, 4)).astype(np.float32)
        labels = np.asarray([0, 1] * 6, dtype=np.int64)
        arrays = (values[:8], labels[:8], values[8:], labels[8:])
        weights = np.ones(2, dtype=np.float32)
        config = self._config(epochs=2, dropout=0.25)
        config["training"]["num_workers"] = 1

        seed_everything(31)
        uninterrupted = _fit_neural(
            self._model(config), *arrays, weights, config
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "worker-resume.pt"
            seed_everything(31)
            _fit_neural(
                self._model(config),
                *arrays,
                weights,
                config,
                checkpoint_path=checkpoint,
                stop_after_epoch=1,
            )
            seed_everything(999)
            resumed = _fit_neural(
                self._model(config),
                *arrays,
                weights,
                config,
                resume_from=checkpoint,
            )
        self.assertEqual(
            uninterrupted.history.to_dict(orient="records"),
            resumed.history.to_dict(orient="records"),
        )
        for name, expected in uninterrupted.model.state_dict().items():
            self.assertTrue(
                torch.equal(expected.cpu(), resumed.model.state_dict()[name].cpu()),
                name,
            )

    def test_array_terminal_early_stop_resume_is_idempotent(self) -> None:
        import torch

        from bitguard_bnn.trainer import _fit_neural

        rng = np.random.default_rng(37)
        values = rng.normal(size=(12, 4)).astype(np.float32)
        labels = np.asarray([0, 1] * 6, dtype=np.int64)
        arrays = (values[:8], labels[:8], values[8:], labels[8:])
        weights = np.ones(2, dtype=np.float32)
        config = self._config(epochs=3, dropout=0.0)
        config["training"]["patience"] = 1
        fixed_metrics = {
            "validation_macro_f1": 0.7,
            "validation_macro_auprc": 0.8,
            "validation_attack_recall": 0.9,
            "validation_selection_score": 0.77,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "terminal-stop.pt"
            with patch(
                "bitguard_bnn.trainer.neural_validation_metrics",
                return_value=fixed_metrics,
            ):
                stopped = _fit_neural(
                    self._model(config),
                    *arrays,
                    weights,
                    config,
                    checkpoint_path=checkpoint,
                )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["stale_epochs"], 1)
            with patch(
                "bitguard_bnn.trainer.neural_train_step"
            ) as train_step, patch(
                "bitguard_bnn.trainer._predict_neural_probabilities"
            ) as predict, patch(
                "bitguard_bnn.trainer.neural_validation_metrics"
            ) as validate:
                resumed = _fit_neural(
                    self._model(config),
                    *arrays,
                    weights,
                    config,
                    resume_from=checkpoint,
                )
            train_step.assert_not_called()
            predict.assert_not_called()
            validate.assert_not_called()
        self.assertEqual(
            stopped.history.to_dict(orient="records"),
            resumed.history.to_dict(orient="records"),
        )

    def test_checkpoint_publication_cleanup_and_malicious_pickle_rejection(self) -> None:
        import torch

        from bitguard_bnn.trainer import (
            _atomic_torch_save,
            _safe_torch_load,
            _serialized_numpy_rng_state,
            _validated_numpy_rng_state,
        )

        expected_numpy_state = np.random.get_state()
        with patch(
            "torch.from_numpy",
            side_effect=AssertionError("uint32 from_numpy is unsupported on Torch 2.2"),
        ):
            serialized_numpy_state = _serialized_numpy_rng_state()
        self.assertEqual(serialized_numpy_state["values"].dtype, torch.int64)
        restored_numpy_state = _validated_numpy_rng_state(
            torch, {"numpy_rng_state": serialized_numpy_state}
        )
        self.assertEqual(restored_numpy_state[0], expected_numpy_state[0])
        np.testing.assert_array_equal(restored_numpy_state[1], expected_numpy_state[1])
        self.assertEqual(restored_numpy_state[2:], expected_numpy_state[2:])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "state.pt"
            checkpoint.write_bytes(b"stable")

            def failed_save(_payload, target):
                Path(target).write_bytes(b"partial")
                raise OSError("storage failed")

            with patch("torch.save", side_effect=failed_save), self.assertRaisesRegex(
                OSError, "storage failed"
            ):
                _atomic_torch_save(checkpoint, {"value": 1})
            self.assertEqual(checkpoint.read_bytes(), b"stable")
            self.assertFalse((root / "state.pt.tmp").exists())

            with patch.object(
                torch, "__version__", "2.5.1+cu124"
            ), patch("torch.load") as vulnerable_load, self.assertRaisesRegex(
                RuntimeError, "PyTorch 2.6 or newer"
            ):
                _safe_torch_load(checkpoint, torch.device("cpu"))
            vulnerable_load.assert_not_called()
            with patch.object(
                torch, "__version__", "unparseable"
            ), patch("torch.load") as unknown_load, self.assertRaisesRegex(
                RuntimeError, "unable to verify"
            ):
                _safe_torch_load(checkpoint, torch.device("cpu"))
            unknown_load.assert_not_called()
            with patch.object(
                torch, "__version__", "2.6.0+cu124"
            ), patch("torch.load", return_value={}) as patched_load:
                self.assertEqual(
                    _safe_torch_load(checkpoint, torch.device("cpu")), {}
                )
            patched_load.assert_called_once()

            marker = root / "executed.txt"
            malicious = root / "malicious.pt"
            torch.save({"payload": _MaliciousPayload(marker)}, malicious)
            with self.assertRaisesRegex(ValueError, "safe tensor payload"):
                _safe_torch_load(malicious, torch.device("cpu"))
            self.assertFalse(marker.exists())

    def test_streaming_config_defaults_and_strict_positive_validation(self) -> None:
        self.assertEqual(DEFAULTS["training"]["checkpoint_every_steps"], 1000)
        self.assertEqual(DEFAULTS["training"]["shuffle_buffer_rows"], 262144)
        for key in ("checkpoint_every_steps", "shuffle_buffer_rows"):
            for value in (True, 0, -1, 1.5):
                config = copy.deepcopy(DEFAULTS)
                config["training"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_config(config)
        invalid_weights = (
            {"macro_f1": 0.5, "macro_auprc": 0.5},
            {
                "macro_f1": 0.5,
                "macro_auprc": 0.3,
                "attack_recall": 0.2,
                "extra": 0.0,
            },
            {"macro_f1": True, "macro_auprc": 0.0, "attack_recall": 0.0},
            {"macro_f1": float("nan"), "macro_auprc": 0.3, "attack_recall": 0.7},
            {"macro_f1": -0.1, "macro_auprc": 0.4, "attack_recall": 0.7},
            {"macro_f1": 0.5, "macro_auprc": 0.3, "attack_recall": 0.20000001},
        )
        for weights in invalid_weights:
            config = copy.deepcopy(DEFAULTS)
            config["training"]["selection_weights"] = weights
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
