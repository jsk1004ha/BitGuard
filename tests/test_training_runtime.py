from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from bitguard_bnn.config import DEFAULTS, environment_manifest, seed_everything
from bitguard_bnn.models import build_model
from bitguard_bnn.trainer import _fit_neural, _make_grad_scaler


class TrainingRuntimeTest(unittest.TestCase):
    def _config(self, epochs: int) -> dict:
        config = copy.deepcopy(DEFAULTS)
        config["experiment"]["seed"] = 17
        config["model"].update(
            {
                "type": "vanilla_bnn",
                "hidden_dims": [8],
                "dropout": 0.0,
                "binary_first_layer": True,
            }
        )
        config["preprocess"]["open_set"]["enabled"] = False
        config["training"].update(
            {
                "epochs": epochs,
                "batch_size": 8,
                "patience": 10,
                "num_workers": 0,
                "device": "cpu",
                "amp": False,
            }
        )
        return config

    @staticmethod
    def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(17)
        values = rng.normal(size=(40, 4)).astype(np.float32)
        labels = (values[:, 0] + values[:, 1] > 0).astype(np.int64)
        return values[:32], labels[:32], values[32:], labels[32:]

    @staticmethod
    def _model(config: dict):
        return build_model(
            config,
            input_dim=4,
            output_dim=2,
            input_groups=np.arange(4, dtype=np.int64),
            feature_costs=np.ones(4, dtype=np.float32),
        )

    def test_seed_sets_cuda_determinism_environment(self) -> None:
        previous = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        try:
            seed_everything(17)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        finally:
            if previous is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous

    def test_environment_manifest_records_core_packages_and_determinism(self) -> None:
        seed_everything(17)
        manifest = environment_manifest()
        self.assertTrue(
            {"numpy", "pandas", "scikit-learn", "joblib", "PyYAML", "torch"}
            <= set(manifest["packages"])
        )
        self.assertEqual(manifest["determinism"]["cublas_workspace_config"], ":4096:8")
        self.assertTrue(manifest["cudnn_deterministic"])

    def test_epoch_checkpoint_is_atomic_and_resume_continues_history(self) -> None:
        x_train, y_train, x_validation, y_validation = self._data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last_training_state.pt"
            progress = root / "training_history.partial.csv"
            first_config = self._config(2)
            seed_everything(17)
            first = _fit_neural(
                self._model(first_config),
                x_train,
                y_train,
                x_validation,
                y_validation,
                np.ones(2, dtype=np.float32),
                first_config,
                checkpoint_path=checkpoint,
                progress_path=progress,
                stop_after_epoch=1,
            )
            self.assertEqual(first.history["epoch"].tolist(), [1])
            self.assertTrue(checkpoint.exists())
            self.assertTrue(progress.exists())
            self.assertFalse((root / "last_training_state.pt.tmp").exists())

            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 1)
            self.assertIn("optimizer_state_dict", state)
            self.assertIn("best_state_dict", state)
            self.assertIn("generator_state", state)
            self.assertIn("training_signature", state)

            resumed_config = self._config(2)
            seed_everything(999)
            resumed = _fit_neural(
                self._model(resumed_config),
                x_train,
                y_train,
                x_validation,
                y_validation,
                np.ones(2, dtype=np.float32),
                resumed_config,
                checkpoint_path=checkpoint,
                progress_path=progress,
                resume_from=checkpoint,
            )
            self.assertEqual(resumed.history["epoch"].tolist(), [1, 2])
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 2)

            incompatible_config = self._config(2)
            incompatible_config["model"]["dropout"] = 0.5
            with self.assertRaisesRegex(ValueError, "does not match"):
                _fit_neural(
                    self._model(incompatible_config),
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    np.ones(2, dtype=np.float32),
                    incompatible_config,
                    resume_from=checkpoint,
                )

    def test_resume_restores_dropout_rng_and_matches_uninterrupted_training(self) -> None:
        import torch

        x_train, y_train, x_validation, y_validation = self._data()
        config = self._config(2)
        config["model"]["dropout"] = 0.25
        weights = np.ones(2, dtype=np.float32)

        seed_everything(17)
        uninterrupted = _fit_neural(
            self._model(config),
            x_train,
            y_train,
            x_validation,
            y_validation,
            weights,
            config,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            seed_everything(17)
            _fit_neural(
                self._model(config),
                x_train,
                y_train,
                x_validation,
                y_validation,
                weights,
                config,
                checkpoint_path=checkpoint,
                stop_after_epoch=1,
            )
            seed_everything(999)
            resumed = _fit_neural(
                self._model(config),
                x_train,
                y_train,
                x_validation,
                y_validation,
                weights,
                config,
                resume_from=checkpoint,
            )

        self.assertEqual(
            uninterrupted.history.to_dict(orient="records"),
            resumed.history.to_dict(orient="records"),
        )
        for name, expected in uninterrupted.model.state_dict().items():
            self.assertTrue(torch.equal(expected.cpu(), resumed.model.state_dict()[name].cpu()), name)

    def test_grad_scaler_falls_back_for_declared_torch_22_floor(self) -> None:
        legacy_scaler = Mock(return_value=object())
        torch_22 = SimpleNamespace(
            amp=SimpleNamespace(),
            cuda=SimpleNamespace(amp=SimpleNamespace(GradScaler=legacy_scaler)),
        )
        result = _make_grad_scaler(torch_22, "cuda", enabled=True)
        self.assertIsNotNone(result)
        legacy_scaler.assert_called_once_with(enabled=True)


if __name__ == "__main__":
    unittest.main()
