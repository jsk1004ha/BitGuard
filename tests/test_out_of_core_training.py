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
from bitguard_bnn.out_of_core.dataset import DataCursor


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
        self.entries = (object(),)
        self.epoch = 0
        self.cursor: DataCursor | None = None

    def set_epoch(self, epoch: int, cursor: DataCursor | None = None) -> None:
        if cursor is not None and cursor.epoch != epoch:
            raise ValueError("resume cursor epoch does not match dataset epoch")
        self.epoch = epoch
        self.cursor = cursor


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
            )
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "stream-state.pt"
                seed_everything(17)
                with self.assertRaises(StreamingTrainingInterrupted):
                    fit_neural_streaming(
                        self._model(config),
                        self._dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        counted_validation,
                        checkpoint_path=checkpoint,
                        stop_after_optimizer_step=2,
                    )
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(state["format_version"], 3)
                self.assertEqual(state["cursor"], DataCursor(1, 0, 2, 2))
                self.assertIsNone(state["best_state_dict"])
                self.assertEqual(resumed_validation_calls, [])

                incompatible = self._config()
                incompatible["model"]["dropout"] = 0.5
                with self.assertRaisesRegex(ValueError, "scientific signature"):
                    fit_neural_streaming(
                        self._model(incompatible),
                        self._dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        incompatible,
                        self._validation,
                        resume_from=checkpoint,
                    )

                tampered = dict(state)
                tampered["global_optimizer_step"] = 1
                torch.save(tampered, checkpoint)
                with self.assertRaisesRegex(ValueError, "optimizer step and cursor"):
                    fit_neural_streaming(
                        self._model(config),
                        self._dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        self._validation,
                        resume_from=checkpoint,
                    )
                torch.save(state, checkpoint)

                seed_everything(999)
                resumed = fit_neural_streaming(
                    self._model(config),
                    self._dataset(),
                    {"benign": 3, "flood_like": 3},
                    ("benign", "flood_like"),
                    config,
                    counted_validation,
                    checkpoint_path=checkpoint,
                    resume_from=checkpoint,
                )
                final_state = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
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
            )

        self.assertAlmostEqual(result.history.iloc[0]["train_loss"], 13.0 / 5.0)

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
                resume_from=checkpoint,
            )

        self.assertEqual(calls, [])
        self.assertEqual(
            stopped.history.to_dict(orient="records"),
            resumed.history.to_dict(orient="records"),
        )

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
                        checkpoint_path=checkpoint,
                    )
                interrupted = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                self.assertEqual(interrupted["epoch_phase"], "validation")
                self.assertEqual(interrupted["scheduler_state_dict"]["last_epoch"], 1)

                tampered = dict(interrupted)
                tampered["partial_seen_rows"] = interrupted["partial_seen_rows"] - 1
                torch.save(tampered, checkpoint)
                with self.assertRaisesRegex(RuntimeError, "row coverage"):
                    fit_neural_streaming(
                        self._model(config),
                        self._dataset(),
                        {"benign": 3, "flood_like": 3},
                        ("benign", "flood_like"),
                        config,
                        self._validation,
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
                        checkpoint_path=checkpoint,
                        resume_from=checkpoint,
                    )
                resumed_step.assert_not_called()
                final_state = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
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
        ), patch("bitguard_bnn.trainer._shutdown_persistent_workers") as close, self.assertRaisesRegex(
            RuntimeError, "array objective failed"
        ):
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

    def test_streaming_config_defaults_and_strict_positive_validation(self) -> None:
        self.assertEqual(DEFAULTS["training"]["checkpoint_every_steps"], 1000)
        self.assertEqual(DEFAULTS["training"]["shuffle_buffer_rows"], 262144)
        for key in ("checkpoint_every_steps", "shuffle_buffer_rows"):
            for value in (True, 0, -1, 1.5):
                config = copy.deepcopy(DEFAULTS)
                config["training"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
