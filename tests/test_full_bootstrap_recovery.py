from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bitguard_bnn.config import load_config
from bitguard_bnn.bootstrap.orchestrator import BootstrapDependencies, run_bootstrap
from bitguard_bnn.bootstrap.types import BootstrapOptions
from bitguard_bnn.out_of_core.prepare import PreparedDataset


class _ProgressBoundaryReached(RuntimeError):
    pass


def _placeholder_prepared(root: Path, config_path: Path) -> PreparedDataset:
    """Build the smallest descriptor object needed to reach the run-created boundary."""

    values = {
        "descriptor_path": str(root / "prepared.json"),
        "dataset": "nbaiot",
        "template_config_path": str(config_path),
        "template_config_sha256": "template",
        "resolved_config_path": str(config_path),
        "config_sha256": "config",
        "preparation_fingerprint": "preparation",
        "raw_root": str(root / "raw"),
        "output_dir": str(root / "prepared"),
        "work_dir": str(root / "work"),
        "source_manifest_path": str(root / "source.json"),
        "source_manifest_fingerprint": "source-manifest",
        "schema_report_path": str(root / "schema.json"),
        "schema_report_fingerprint": "schema",
        "normalized_source_fingerprint": "source",
        "split_membership_path": str(root / "membership.sqlite"),
        "split_membership_sha256": "membership",
        "split_manifest_path": str(root / "split.json"),
        "split_fingerprint": "split",
        "preprocessor_path": str(root / "preprocessor.joblib"),
        "preprocessor_sha256": "preprocessor-file",
        "feature_manifest_path": str(root / "features.json"),
        "preprocessing_fingerprint": "preprocessor",
        "shard_manifest_path": str(root / "shards.json"),
        "shard_fingerprint": "shards",
        "train_count": 1,
        "validation_count": 1,
        "test_count": 1,
        "total_count": 3,
    }
    return PreparedDataset(**values)


def _write_fake_completed_run(
    run_dir: Path, *, dataset: str, descriptor_path: Path
) -> None:
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
        if name != "manifest":
            path.write_bytes(f"{dataset}:{name}".encode("utf-8"))
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
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
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
        )
        + "\n",
        encoding="utf-8",
    )


def _bootstrap_resume_fixture(
    root: Path,
) -> tuple[BootstrapOptions, BootstrapDependencies]:
    archive = root / "nbaiot.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("device_a/benign_traffic.csv", "mean,std\n1,2\n2,3\n")
        output.writestr("device_b/gafgyt_attacks/scan.csv", "mean,std\n8,9\n9,10\n")

    repository = Path(__file__).resolve().parents[1]
    resolved_config = repository / "configs" / "full" / "nbaiot.yaml"

    def prepare(_config: Path, **kwargs: object) -> object:
        descriptor = Path(str(kwargs["descriptor_path"]))
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text(
            json.dumps({"dataset": "nbaiot", "valid": True}) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(descriptor_path=str(descriptor))

    def verify(descriptor: Path) -> object:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        if payload != {"dataset": "nbaiot", "valid": True}:
            raise RuntimeError("invalid prepared descriptor fixture")
        return SimpleNamespace(
            resolved_config_path=str(resolved_config),
            to_dict=lambda: {
                "dataset": "nbaiot",
                "descriptor_path": str(descriptor),
                "fingerprint": "fixture-prepared-fingerprint",
            },
        )

    options = BootstrapOptions(
        datasets=("nbaiot",),
        botiot_source=None,
        data_root=(root / "data").resolve(),
        runs_root=(root / "runs").resolve(),
        compute="cpu",
        prepare_only=False,
        install_system_tools=False,
        accepted_botiot_license=False,
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
        preparer=prepare,
        prepared_verifier=verify,
    )
    return options, dependencies


def write_nbaiot_training_fixture(archive: Path) -> None:
    with zipfile.ZipFile(archive, "w") as output:
        for device_index, device in enumerate(
            ("Ecobee_Thermostat", "Philips_B120N10_Baby_Monitor")
        ):
            benign = "mean,std\n" + "".join(
                f"{1 + device_index + row},{2 + row / 10}\n" for row in range(8)
            )
            attack = "mean,std\n" + "".join(
                f"{20 + device_index + row},{4 + row / 10}\n" for row in range(8)
            )
            output.writestr(f"{device}/benign_traffic.csv", benign)
            output.writestr(f"{device}/gafgyt_attacks/scan.csv", attack)
        output.writestr(
            "Danmini_Doorbell/benign_traffic.csv",
            "mean,std\n" + "".join(f"{1000 + row},{2000 + row}\n" for row in range(8)),
        )


def write_fast_nbaiot_profile(path: Path) -> None:
    import yaml

    repository = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (repository / "configs" / "full" / "nbaiot.yaml").read_text(encoding="utf-8")
    )
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
            "expert_features": ["mean", "std"],
        }
    )
    payload["model"].update({"type": "vanilla_bnn", "hidden_dims": [4], "dropout": 0.0})
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
    payload["cascade"].update({"enabled": False, "boolean_fast_path_enabled": False})
    payload["temporal"]["enabled"] = False
    payload["evaluation"].update(
        {
            "fixed_fpr_targets": [0.5],
            "plot_sample_rows": 10,
            "benchmark_warmup": 1,
            "benchmark_repeats": 1,
        }
    )
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def write_fast_cuda_nbaiot_profile(path: Path) -> None:
    import yaml

    write_fast_nbaiot_profile(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["training"].update({"device": "cuda", "amp": True})
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def write_botiot_training_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = ["category,subcategory,saddr,stime,bytes,rate\n"]
    for index in range(40):
        benign = index % 2 == 0
        rows.append(
            f"{'Normal' if benign else 'DDoS'},"
            f"{'Normal' if benign else 'TCP'},10.0.0.{index % 4 + 1},"
            f"{index + 0.5},{100 + index},{1.0 + index / 10}\n"
        )
    (root / "flows.csv").write_text("".join(rows), encoding="utf-8")


def write_fast_botiot_profile(path: Path) -> None:
    import yaml

    repository = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (repository / "configs" / "full" / "botiot.yaml").read_text(encoding="utf-8")
    )
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
            "expert_features": ["bytes", "rate"],
        }
    )
    payload["model"].update({"type": "vanilla_bnn", "hidden_dims": [4], "dropout": 0.0})
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
    payload["cascade"].update({"enabled": False, "boolean_fast_path_enabled": False})
    payload["temporal"]["enabled"] = True
    payload["evaluation"].update(
        {
            "fixed_fpr_targets": [0.5],
            "plot_sample_rows": 10,
            "benchmark_warmup": 1,
            "benchmark_repeats": 1,
        }
    )
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def run_optimizer_exit_fixture_child(
    root_value: str,
    *,
    kill_after_uncommitted_step: bool,
    restart_train: bool,
) -> None:
    """Subprocess entry used to prove bootstrap-owned optimizer resume."""

    from unittest.mock import patch as child_patch

    from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

    root = Path(root_value).resolve()
    archive = root / "nbaiot.zip"
    config_path = root / "nbaiot-fast.yaml"
    options = BootstrapOptions(
        datasets=("nbaiot",),
        botiot_source=None,
        data_root=(root / "data").resolve(),
        runs_root=(root / "runs").resolve(),
        compute="cpu",
        prepare_only=False,
        install_system_tools=False,
        accepted_botiot_license=False,
        restart_stage="train" if restart_train else None,
    )

    def prepare(_config: Path, **kwargs: object) -> object:
        return prepare_full_dataset(config_path, **kwargs)

    dependencies = BootstrapDependencies(
        nbaiot_archive=archive,
        available_bytes=10**12,
        preparation_available_bytes=10**12,
        compute_resolver=lambda requested: {
            "requested": requested,
            "selected_profile": "cpu",
            "device": "cpu",
        },
        preparer=prepare,
        preparation_signature_token=(
            "optimizer-exit-fixture:"
            + hashlib.sha256(config_path.read_bytes()).hexdigest()
        ),
    )

    if kill_after_uncommitted_step:
        from bitguard_bnn.out_of_core import trainer as streaming_trainer

        original_step = streaming_trainer.neural_train_step
        calls = 0

        def exit_after_second_optimizer_step(*args, **kwargs):
            nonlocal calls
            result = original_step(*args, **kwargs)
            calls += 1
            if calls == 2:
                os._exit(91)
            return result

        with child_patch.object(
            streaming_trainer,
            "neural_train_step",
            side_effect=exit_after_second_optimizer_step,
        ):
            run_bootstrap(options, dependencies=dependencies)
        raise AssertionError(
            "optimizer interruption seam did not terminate the process"
        )

    report = run_bootstrap(options, dependencies=dependencies)
    if report.get("status") != "completed":
        raise RuntimeError(json.dumps(report, sort_keys=True, default=str))


def run_validation_cache_exit_fixture_child(
    root_value: str,
    *,
    kill_after_cache_commit: bool,
    restart_train: bool = False,
) -> None:
    """Subprocess entry proving bootstrap discovers a hard-exit cache journal."""

    from unittest.mock import patch as child_patch

    from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

    root = Path(root_value).resolve()
    archive = root / "nbaiot.zip"
    config_path = root / "nbaiot-fast.yaml"
    options = BootstrapOptions(
        datasets=("nbaiot",),
        botiot_source=None,
        data_root=(root / "data").resolve(),
        runs_root=(root / "runs").resolve(),
        compute="cpu",
        prepare_only=False,
        install_system_tools=False,
        accepted_botiot_license=False,
        restart_stage="train" if restart_train else None,
    )

    def prepare(_config: Path, **kwargs: object) -> object:
        return prepare_full_dataset(config_path, **kwargs)

    dependencies = BootstrapDependencies(
        nbaiot_archive=archive,
        available_bytes=10**12,
        preparation_available_bytes=10**12,
        compute_resolver=lambda requested: {
            "requested": requested,
            "selected_profile": "cpu",
            "device": "cpu",
        },
        preparer=prepare,
        preparation_signature_token=(
            "validation-cache-exit-fixture:"
            + hashlib.sha256(config_path.read_bytes()).hexdigest()
        ),
    )
    if kill_after_cache_commit:
        from bitguard_bnn.out_of_core.cache import CalibrationCache

        original_commit = CalibrationCache.commit_inference_range
        killed = False

        def exit_after_committed_journal(cache, start, values):
            nonlocal killed
            result = original_commit(cache, start, values)
            if not killed and cache.committed_rows > 0:
                killed = True
                os._exit(94)
            return result

        with child_patch.object(
            CalibrationCache,
            "commit_inference_range",
            new=exit_after_committed_journal,
        ):
            run_bootstrap(options, dependencies=dependencies)
        raise AssertionError("validation cache interruption seam did not terminate")

    report = run_bootstrap(options, dependencies=dependencies)
    if report.get("status") != "completed":
        raise RuntimeError(json.dumps(report, sort_keys=True, default=str))


def run_cuda_resume_fixture_child(
    root_value: str,
    *,
    kill_after_uncommitted_step: bool,
) -> None:
    """Subprocess entry proving CUDA AMP checkpoints resume through bootstrap."""

    from unittest.mock import patch as child_patch

    from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

    root = Path(root_value).resolve()
    archive = root / "nbaiot.zip"
    config_path = root / "nbaiot-cuda-fast.yaml"
    options = BootstrapOptions(
        datasets=("nbaiot",),
        botiot_source=None,
        data_root=(root / "data").resolve(),
        runs_root=(root / "runs").resolve(),
        compute="auto",
        prepare_only=False,
        install_system_tools=False,
        accepted_botiot_license=False,
        restart_stage=None,
    )

    def prepare(_config: Path, **kwargs: object) -> object:
        return prepare_full_dataset(config_path, **kwargs)

    dependencies = BootstrapDependencies(
        nbaiot_archive=archive,
        available_bytes=10**12,
        preparation_available_bytes=10**12,
        preparer=prepare,
        preparation_signature_token=(
            "cuda-resume-fixture:"
            + hashlib.sha256(config_path.read_bytes()).hexdigest()
        ),
    )
    if kill_after_uncommitted_step:
        from bitguard_bnn.out_of_core import trainer as streaming_trainer

        original_step = streaming_trainer.neural_train_step
        calls = 0

        def exit_after_second_optimizer_step(*args, **kwargs):
            nonlocal calls
            result = original_step(*args, **kwargs)
            calls += 1
            if calls == 2:
                os._exit(95)
            return result

        with child_patch.object(
            streaming_trainer,
            "neural_train_step",
            side_effect=exit_after_second_optimizer_step,
        ):
            run_bootstrap(options, dependencies=dependencies)
        raise AssertionError("CUDA optimizer interruption seam did not terminate")

    report = run_bootstrap(options, dependencies=dependencies)
    if report.get("status") != "completed":
        raise RuntimeError(json.dumps(report, sort_keys=True, default=str))


def _subprocess_environment(repository: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source = str(repository / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not existing else source + os.pathsep + existing
    )
    return environment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_layout_fixture(rows: int = 6):
    from bitguard_bnn.out_of_core.cache import CacheLayout

    return CacheLayout(
        prepared_descriptor_fingerprint="prepared",
        shard_fingerprint="shards",
        preprocessor_fingerprint="preprocessor",
        source_fingerprint="source",
        main_checkpoint_fingerprint="main",
        tiny_checkpoint_fingerprint="tiny",
        inference_contract_fingerprint="inference",
        split="validation",
        row_count=rows,
        main_class_labels=("benign", "scan_like"),
        routed_class_labels=("benign", "scan_like", "unknown_like"),
        true_class_labels=("benign", "scan_like", "unknown_like"),
        selected_features=("f1", "f2"),
        boolean_features=("flag_a", "flag_b"),
        device_id_width=8,
        source_id_width=8,
    )


def cache_inference_batch(start: int, end: int) -> dict[str, object]:
    import numpy as np

    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
    known = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.1, 0.9]],
        dtype=np.float32,
    )
    selected = np.asarray(
        [[0.1, 0.2], [1.0, 2.0], [0.2, 0.3], [2.0, 1.0], [0.3, 0.4], [3.0, 1.0]],
        dtype=np.float32,
    )
    tiny = np.asarray([0.95, 0.2, 0.85, 0.3, 0.75, 0.1], dtype=np.float32)
    flags = np.asarray(
        [
            [True, True],
            [False, True],
            [True, False],
            [False, False],
            [True, True],
            [False, False],
        ],
        dtype=np.bool_,
    )
    return {
        "cache_position": np.arange(start, end, dtype=np.int64),
        "uid_digest": np.vstack(
            [
                np.frombuffer(bytes.fromhex(f"{index + 1:064x}"), dtype=np.uint8)
                for index in range(start, end)
            ]
        ),
        "true_label": labels[start:end],
        "known_probabilities": known[start:end],
        "selected_values": selected[start:end],
        "tiny_benign_probability": tiny[start:end],
        "boolean_flags": flags[start:end],
        "timestamp": np.arange(start, end, dtype=np.float64),
        "sequence": np.arange(start, end, dtype=np.int64),
        "device_id": ["dev"] * (end - start),
        "source_id": ["capture"] * (end - start),
    }


def run_cache_journal_exit_child(root_value: str, stage: str) -> None:
    from unittest.mock import patch as child_patch

    from bitguard_bnn.out_of_core import cache as cache_module
    from bitguard_bnn.out_of_core.cache import CalibrationCache

    root = Path(root_value)
    with CalibrationCache.open_resume(root, cache_layout_fixture()) as cache:
        if stage == "before":
            with child_patch.object(
                cache_module,
                "_write_journal",
                side_effect=lambda *_args, **_kwargs: os._exit(92),
            ):
                cache.commit_inference_range(2, cache_inference_batch(2, 4))
        elif stage == "after":
            with child_patch.object(
                cache_module,
                "_fsync_directory",
                side_effect=lambda *_args, **_kwargs: os._exit(93),
            ):
                cache.commit_inference_range(2, cache_inference_batch(2, 4))
        else:
            raise ValueError(stage)
    raise AssertionError("cache journal interruption seam did not terminate")


class FullBootstrapRecoveryTests(unittest.TestCase):
    def test_resume_header_is_rejected_before_run_directory_creation(self) -> None:
        import torch

        from bitguard_bnn.out_of_core.prepare import verify_prepared_dataset
        from bitguard_bnn.out_of_core.run import run_out_of_core_training

        repository = Path(__file__).resolve().parents[1]
        environment = _subprocess_environment(repository)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "nbaiot-fast.yaml"
            write_nbaiot_training_fixture(root / "nbaiot.zip")
            write_fast_nbaiot_profile(config_path)
            expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_optimizer_exit_fixture_child as run; "
                f"run({str(root)!r}, kill_after_uncommitted_step=True, "
                "restart_train=False)"
            )
            interrupted = subprocess.run(
                [sys.executable, "-c", expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                interrupted.returncode,
                91,
                msg=interrupted.stdout + interrupted.stderr,
            )
            status = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )["datasets"]["nbaiot"]
            checkpoint = Path(status["active_checkpoint"]["path"])
            descriptor = Path(status["descriptor"])
            prepared = verify_prepared_dataset(descriptor)
            original_state = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )

            def missing_field(state: dict[str, object]) -> None:
                state.pop("scaler_state_dict")

            def wrong_format(state: dict[str, object]) -> None:
                state["format_version"] = 99

            def wrong_signature(state: dict[str, object]) -> None:
                signature = dict(state["scientific_signature"])  # type: ignore[arg-type]
                signature["fingerprint"] = "tampered"
                state["scientific_signature"] = signature

            def wrong_epochs(state: dict[str, object]) -> None:
                state["target_epochs"] = int(state["target_epochs"]) + 1

            def wrong_device(state: dict[str, object]) -> None:
                state["device_type"] = "cuda"

            def invalid_cursor(state: dict[str, object]) -> None:
                state["cursor"] = "invalid"

            def invalid_optimizer(state: dict[str, object]) -> None:
                state["optimizer_state_dict"] = "invalid"

            def incompatible_optimizer(state: dict[str, object]) -> None:
                optimizer = copy.deepcopy(state["optimizer_state_dict"])
                assert isinstance(optimizer, dict)
                parameter_groups = optimizer["param_groups"]
                assert isinstance(parameter_groups, list)
                first_group = parameter_groups[0]
                assert isinstance(first_group, dict)
                first_group["params"] = []
                state["optimizer_state_dict"] = optimizer

            def incompatible_scheduler(state: dict[str, object]) -> None:
                scheduler = copy.deepcopy(state["scheduler_state_dict"])
                assert isinstance(scheduler, dict)
                removable = next(key for key in scheduler if key != "last_epoch")
                scheduler.pop(removable)
                state["scheduler_state_dict"] = scheduler

            def invalid_scaler(state: dict[str, object]) -> None:
                state["scaler_state_dict"] = "invalid"

            def incompatible_scaler(state: dict[str, object]) -> None:
                scaler = copy.deepcopy(state["scaler_state_dict"])
                assert isinstance(scaler, dict)
                scaler["unexpected"] = 1
                state["scaler_state_dict"] = scaler

            def invalid_rng(state: dict[str, object]) -> None:
                state["torch_rng_state"] = "invalid"

            def invalid_model(state: dict[str, object]) -> None:
                model = copy.deepcopy(state["model_state_dict"])
                assert isinstance(model, dict)
                model.pop(next(iter(model)))
                state["model_state_dict"] = model

            cases = (
                ("fields", missing_field, "streaming checkpoint fields are invalid"),
                ("format", wrong_format, "unsupported streaming checkpoint format"),
                (
                    "signature",
                    wrong_signature,
                    "streaming checkpoint scientific signature mismatch",
                ),
                (
                    "epochs",
                    wrong_epochs,
                    "training.epochs must match the streaming checkpoint target",
                ),
                (
                    "device",
                    wrong_device,
                    "streaming checkpoint device type mismatch",
                ),
                (
                    "cursor",
                    invalid_cursor,
                    "streaming checkpoint cursor is invalid",
                ),
                (
                    "optimizer",
                    invalid_optimizer,
                    "streaming checkpoint optimizer state is invalid",
                ),
                (
                    "optimizer-layout",
                    incompatible_optimizer,
                    "streaming checkpoint optimizer state is incompatible",
                ),
                (
                    "scheduler-layout",
                    incompatible_scheduler,
                    "streaming checkpoint scheduler state is incompatible",
                ),
                (
                    "scaler",
                    invalid_scaler,
                    "streaming checkpoint scaler state is invalid",
                ),
                (
                    "scaler-layout",
                    incompatible_scaler,
                    "streaming checkpoint scaler state is incompatible",
                ),
                (
                    "rng",
                    invalid_rng,
                    "training checkpoint Torch RNG state is invalid",
                ),
                (
                    "model",
                    invalid_model,
                    "training checkpoint streaming model state fields are invalid",
                ),
            )
            for name, mutate, expected in cases:
                with self.subTest(name=name):
                    state = copy.deepcopy(original_state)
                    mutate(state)
                    candidate = root / f"invalid-{name}.pt"
                    torch.save(state, candidate)
                    config = copy.deepcopy(load_config(prepared.resolved_config_path))
                    config["experiment"]["output_dir"] = str(root / f"runs-{name}")
                    config["training"]["resume_from"] = str(candidate)
                    with (
                        patch(
                            "bitguard_bnn.out_of_core.run.create_run_dir",
                            side_effect=AssertionError(
                                "create_run_dir called before resume validation"
                            ),
                        ) as create_run_dir,
                        self.assertRaisesRegex(ValueError, expected) as raised,
                    ):
                        run_out_of_core_training(
                            config_path,
                            config=config,
                            prepared_descriptor_path=descriptor,
                            resume_checkpoint_integrity={
                                "sha256": _sha256(candidate),
                                "bytes": candidate.stat().st_size,
                            },
                        )
                    create_run_dir.assert_not_called()
                    if name in {
                        "optimizer-layout",
                        "scheduler-layout",
                        "scaler-layout",
                    }:
                        from bitguard_bnn.bootstrap.orchestrator import _recovery

                        recovery = _recovery(
                            "train",
                            raised.exception,
                            {"selected_profile": "cpu", "device": "cpu"},
                        )
                        self.assertIn("--restart-stage train", recovery)

            distillation_config = copy.deepcopy(
                load_config(prepared.resolved_config_path)
            )
            distillation_config["experiment"]["output_dir"] = str(
                root / "runs-distillation"
            )
            distillation_config["loss"]["distillation_alpha"] = 0.5
            distillation_config["training"]["resume_from"] = str(checkpoint)
            with (
                patch(
                    "bitguard_bnn.out_of_core.run.load_config",
                    return_value=distillation_config,
                ),
                patch(
                    "bitguard_bnn.out_of_core.run.create_run_dir",
                    side_effect=AssertionError(
                        "create_run_dir called before distillation resume rejection"
                    ),
                ) as create_run_dir,
                self.assertRaisesRegex(
                    ValueError,
                    "distillation resume requires independently verified teacher identity",
                ),
            ):
                run_out_of_core_training(
                    config_path,
                    config=distillation_config,
                    prepared_descriptor_path=descriptor,
                    resume_checkpoint_integrity={
                        "sha256": _sha256(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                    },
                )
            create_run_dir.assert_not_called()

    def test_training_progress_is_published_immediately_after_run_creation(
        self,
    ) -> None:
        from bitguard_bnn.out_of_core.run import _run_verified_neural_training

        repository = Path(__file__).resolve().parents[1]
        config_path = repository / "configs" / "full" / "nbaiot.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = _placeholder_prepared(root, config_path)
            config = copy.deepcopy(load_config(config_path))
            config["experiment"]["output_dir"] = str(root / "runs")
            observed: list[dict[str, object]] = []

            def progress(event: dict[str, object]) -> None:
                observed.append(dict(event))
                raise _ProgressBoundaryReached("durable bootstrap progress recorded")

            with self.assertRaisesRegex(
                _ProgressBoundaryReached, "durable bootstrap progress"
            ):
                _run_verified_neural_training(
                    config,
                    prepared,
                    progress_callback=progress,
                )

            self.assertEqual(len(observed), 1)
            event = observed[0]
            self.assertEqual(event["status"], "run_created")
            run_dir = Path(str(event["run_dir"]))
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(
                Path(str(event["active_checkpoint"])),
                run_dir / "last_training_state.pt",
            )
            self.assertIsNone(event["resume_checkpoint"])
            self.assertEqual(
                event["prepared_descriptor_fingerprint"],
                prepared.to_dict()["fingerprint"],
            )
            self.assertFalse((run_dir / "temporary").exists())

    def test_resume_locator_is_an_operational_override_not_scientific_drift(
        self,
    ) -> None:
        import torch

        from bitguard_bnn.out_of_core.run import _run_verified_neural_training

        repository = Path(__file__).resolve().parents[1]
        config_path = repository / "configs" / "full" / "nbaiot.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = _placeholder_prepared(root, config_path)
            config = copy.deepcopy(load_config(config_path))
            config["experiment"]["output_dir"] = str(root / "runs")
            prior_checkpoint = root / "prior" / "last_training_state.pt"
            prior_checkpoint.parent.mkdir()
            torch.save({"marker": "operational-resume"}, prior_checkpoint)
            config["training"]["resume_from"] = str(prior_checkpoint)
            observed: list[dict[str, object]] = []

            def progress(event: dict[str, object]) -> None:
                observed.append(dict(event))
                raise _ProgressBoundaryReached("resume locator published")

            with (
                patch(
                    "bitguard_bnn.out_of_core.run."
                    "_preflight_streaming_resume_checkpoint"
                ),
                self.assertRaisesRegex(_ProgressBoundaryReached, "resume locator"),
            ):
                _run_verified_neural_training(
                    config,
                    prepared,
                    progress_callback=progress,
                )

            self.assertEqual(
                observed[0]["resume_checkpoint"],
                str(prior_checkpoint.resolve(strict=False)),
            )

    def test_repeated_precheckpoint_exit_preserves_last_committed_resume_locator(
        self,
    ) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies = _bootstrap_resume_fixture(root)
            calls: list[Path | None] = []
            run_directories: list[Path] = []
            descriptor_path = options.data_root / ".bitguard" / "prepared" / "nbaiot"

            def fake_training(
                _config_path: str | Path,
                *,
                config: dict[str, object],
                prepared_descriptor_path: str | Path,
                progress_callback=None,
                **_runtime_options: object,
            ) -> Path:
                invocation = len(calls) + 1
                resume_value = config["training"].get("resume_from")  # type: ignore[index,union-attr]
                resume = None if resume_value is None else Path(str(resume_value))
                calls.append(resume)
                run_dir = options.runs_root / f"nbaiot-attempt-{invocation}"
                run_dir.mkdir(parents=True)
                run_directories.append(run_dir)
                active = run_dir / "last_training_state.pt"
                self.assertIsNotNone(progress_callback)
                progress_callback(
                    {
                        "status": "run_created",
                        "dataset": "nbaiot",
                        "prepared_descriptor": str(prepared_descriptor_path),
                        "prepared_descriptor_fingerprint": (
                            "fixture-prepared-fingerprint"
                        ),
                        "run_dir": str(run_dir.resolve()),
                        "active_checkpoint": str(active.resolve()),
                        "resume_checkpoint": (
                            None if resume is None else str(resume.resolve())
                        ),
                    }
                )
                if invocation == 1:
                    torch.save(
                        {
                            "format_version": 4,
                            "device_type": "cpu",
                            "scientific_signature": {
                                "prepared_descriptor_fingerprint": (
                                    "fixture-prepared-fingerprint"
                                ),
                                "training_transform": {"role": "main"},
                            },
                        },
                        active,
                    )
                    raise RuntimeError(
                        "process exited after committed optimizer cursor"
                    )
                if invocation == 2:
                    raise RuntimeError("process exited before next checkpoint publish")
                _write_fake_completed_run(
                    run_dir,
                    dataset="nbaiot",
                    descriptor_path=Path(prepared_descriptor_path),
                )
                return run_dir

            with patch(
                "bitguard_bnn.out_of_core.run.run_out_of_core_training",
                side_effect=fake_training,
            ):
                first = run_bootstrap(options, dependencies=dependencies)
                self.assertEqual(first["failed_stage"], "train")
                second = run_bootstrap(options, dependencies=dependencies)
                self.assertEqual(second["failed_stage"], "train")
                persisted = json.loads(
                    (options.data_root / ".bitguard" / "training.json").read_text(
                        encoding="utf-8"
                    )
                )["datasets"]["nbaiot"]
                self.assertEqual(persisted["status"], "failed")
                self.assertEqual(
                    Path(persisted["resume_checkpoint"]["path"]),
                    run_directories[0] / "last_training_state.pt",
                )
                self.assertEqual(
                    Path(persisted["active_checkpoint"]["path"]),
                    run_directories[1] / "last_training_state.pt",
                )
                third = run_bootstrap(options, dependencies=dependencies)

            first_checkpoint = run_directories[0] / "last_training_state.pt"
            self.assertEqual(calls, [None, first_checkpoint, first_checkpoint])
            self.assertEqual(third["status"], "completed", msg=third.get("error"))
            self.assertEqual(third["dataset_statuses"]["nbaiot"]["status"], "completed")
            self.assertTrue(descriptor_path.is_dir())

    def test_optimizer_process_exit_resumes_to_exact_uninterrupted_result(self) -> None:
        import torch

        repository = Path(__file__).resolve().parents[1]
        environment = _subprocess_environment(repository)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_nbaiot_training_fixture(root / "nbaiot.zip")
            write_fast_nbaiot_profile(root / "nbaiot-fast.yaml")

            expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_optimizer_exit_fixture_child as run; "
                f"run({str(root)!r}, kill_after_uncommitted_step=True, "
                "restart_train=False)"
            )
            interrupted = subprocess.run(
                [sys.executable, "-c", expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                interrupted.returncode,
                91,
                msg=interrupted.stdout + interrupted.stderr,
            )
            running = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )["datasets"]["nbaiot"]
            active_checkpoint = Path(running["active_checkpoint"]["path"])
            self.assertTrue(active_checkpoint.is_file())
            interrupted_state = torch.load(
                active_checkpoint, map_location="cpu", weights_only=True
            )
            self.assertEqual(interrupted_state["cursor"]["optimizer_step"], 1)

            resume_expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_optimizer_exit_fixture_child as run; "
                f"run({str(root)!r}, kill_after_uncommitted_step=False, "
                "restart_train=False)"
            )
            resumed_process = subprocess.run(
                [sys.executable, "-c", resume_expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
            self.assertEqual(
                resumed_process.returncode,
                0,
                msg=resumed_process.stdout + resumed_process.stderr,
            )
            completed = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )
            resumed_run = Path(completed["runs"]["nbaiot"])

            control_expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_optimizer_exit_fixture_child as run; "
                f"run({str(root)!r}, kill_after_uncommitted_step=False, "
                "restart_train=True)"
            )
            control_process = subprocess.run(
                [sys.executable, "-c", control_expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
            self.assertEqual(
                control_process.returncode,
                0,
                msg=control_process.stdout + control_process.stderr,
            )
            control_report = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )
            control_run = Path(control_report["runs"]["nbaiot"])
            self.assertNotEqual(resumed_run, control_run)

            resumed_state = torch.load(
                resumed_run / "last_training_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            control_state = torch.load(
                control_run / "last_training_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            for field in (
                "cursor",
                "history",
                "scientific_signature",
                "global_optimizer_step",
                "epoch_phase",
                "best_epoch",
            ):
                self.assertEqual(resumed_state[field], control_state[field], field)
            self.assertEqual(
                set(resumed_state["model_state_dict"]),
                set(control_state["model_state_dict"]),
            )
            for name, tensor in resumed_state["model_state_dict"].items():
                self.assertTrue(
                    torch.equal(tensor, control_state["model_state_dict"][name]),
                    msg=name,
                )
            self.assertEqual(
                (resumed_run / "training_history.csv").read_bytes(),
                (control_run / "training_history.csv").read_bytes(),
            )
            self.assertEqual(
                json.loads((resumed_run / "metrics.json").read_text(encoding="utf-8")),
                json.loads((control_run / "metrics.json").read_text(encoding="utf-8")),
            )
            resumed_contract = json.loads(
                (resumed_run / "inference_contract.json").read_text(encoding="utf-8")
            )
            control_contract = json.loads(
                (control_run / "inference_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                resumed_contract["fingerprint"], control_contract["fingerprint"]
            )
            self.assertEqual(
                _sha256(resumed_run / "edge" / "bitguard_edge_weights.npz"),
                _sha256(control_run / "edge" / "bitguard_edge_weights.npz"),
            )

    def test_pin_failure_preserves_fallback_for_unchanged_rerun(self) -> None:
        from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

        repository = Path(__file__).resolve().parents[1]
        environment = _subprocess_environment(repository)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "nbaiot.zip"
            config_path = root / "nbaiot-fast.yaml"
            write_nbaiot_training_fixture(archive)
            write_fast_nbaiot_profile(config_path)
            expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_optimizer_exit_fixture_child as run; "
                f"run({str(root)!r}, kill_after_uncommitted_step=True, "
                "restart_train=False)"
            )
            interrupted = subprocess.run(
                [sys.executable, "-c", expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                interrupted.returncode,
                91,
                msg=interrupted.stdout + interrupted.stderr,
            )
            prior_status = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )["datasets"]["nbaiot"]
            fallback = Path(prior_status["active_checkpoint"]["path"])

            options = BootstrapOptions(
                datasets=("nbaiot",),
                botiot_source=None,
                data_root=(root / "data").resolve(),
                runs_root=(root / "runs").resolve(),
                compute="cpu",
                prepare_only=False,
                install_system_tools=False,
                accepted_botiot_license=False,
                restart_stage=None,
            )

            def prepare(_config: Path, **kwargs: object) -> object:
                return prepare_full_dataset(config_path, **kwargs)

            dependencies = BootstrapDependencies(
                nbaiot_archive=archive,
                available_bytes=10**12,
                preparation_available_bytes=10**12,
                compute_resolver=lambda requested: {
                    "requested": requested,
                    "selected_profile": "cpu",
                    "device": "cpu",
                },
                preparer=prepare,
                preparation_signature_token=(
                    "optimizer-exit-fixture:"
                    + hashlib.sha256(config_path.read_bytes()).hexdigest()
                ),
            )
            with patch(
                "bitguard_bnn.out_of_core.run._pin_verified_resume_checkpoint",
                side_effect=OSError(errno.ENOSPC, "pin destination is full"),
            ):
                failed = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(failed["failed_stage"], "train", failed)
            self.assertIn("pin destination is full", str(failed["error"]))
            failed_status = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )["datasets"]["nbaiot"]
            self.assertEqual(failed_status["status"], "failed")
            self.assertEqual(Path(failed_status["resume_checkpoint"]["path"]), fallback)
            self.assertFalse(Path(failed_status["active_checkpoint"]["path"]).exists())

            completed = run_bootstrap(options, dependencies=dependencies)
            self.assertEqual(completed["status"], "completed", completed.get("error"))
            resumed_run = Path(completed["trained_runs"]["nbaiot"])
            pinned = resumed_run / "resume_training_state.pt"
            self.assertTrue(pinned.is_file())
            self.assertEqual(_sha256(pinned), _sha256(fallback))

    def test_validation_cache_process_exit_before_and_after_journal_replace(
        self,
    ) -> None:
        import numpy as np

        from bitguard_bnn.out_of_core.cache import CalibrationCache
        from bitguard_bnn.out_of_core.calibrate import tune_exit_threshold_from_cache

        repository = Path(__file__).resolve().parents[1]
        environment = _subprocess_environment(repository)
        for stage, exit_code, committed_after_exit in (
            ("before", 92, 2),
            ("after", 93, 4),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                interrupted_root = root / "interrupted"
                control_root = root / "control"
                layout = cache_layout_fixture()
                with CalibrationCache.create(interrupted_root, layout) as cache:
                    cache.commit_inference_range(0, cache_inference_batch(0, 2))

                expression = (
                    "from tests.test_full_bootstrap_recovery import "
                    "run_cache_journal_exit_child as run; "
                    f"run({str(interrupted_root)!r}, {stage!r})"
                )
                child = subprocess.run(
                    [sys.executable, "-c", expression],
                    cwd=repository,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                self.assertEqual(
                    child.returncode,
                    exit_code,
                    msg=child.stdout + child.stderr,
                )

                with CalibrationCache.open_resume(interrupted_root, layout) as resumed:
                    self.assertEqual(resumed.committed_rows, committed_after_exit)
                    resumed.commit_inference_range(
                        committed_after_exit,
                        cache_inference_batch(committed_after_exit, 6),
                    )
                    resumed_calibration = tune_exit_threshold_from_cache(
                        resumed,
                        0.5,
                        11,
                        0.1,
                        root / f"resumed-calibration-{stage}",
                        chunk_rows=2,
                    )
                    resumed_arrays = {
                        name: np.asarray(value).copy()
                        for name, value in resumed.arrays.items()
                    }

                with CalibrationCache.create(control_root, layout) as control:
                    for start in (0, 2, 4):
                        control.commit_inference_range(
                            start, cache_inference_batch(start, start + 2)
                        )
                    control_calibration = tune_exit_threshold_from_cache(
                        control,
                        0.5,
                        11,
                        0.1,
                        root / f"control-calibration-{stage}",
                        chunk_rows=2,
                    )
                    for name, value in control.arrays.items():
                        np.testing.assert_array_equal(resumed_arrays[name], value)

                self.assertEqual(
                    resumed_calibration.to_dict(), control_calibration.to_dict()
                )
                self.assertEqual(list(interrupted_root.glob(".*.tmp")), [])

    def test_low_disk_and_cuda_resolution_fail_before_any_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            low_root = Path(temporary) / "low-disk"
            low_root.mkdir()
            low_options, low_dependencies = _bootstrap_resume_fixture(low_root)
            low_report = run_bootstrap(
                low_options,
                dependencies=replace(low_dependencies, available_bytes=0),
            )
            self.assertEqual(low_report["failed_stage"], "preflight")
            self.assertIn("required=", str(low_report["error"]))
            self.assertIn("available=0 bytes", str(low_report["error"]))
            self.assertIn("shortfall=", str(low_report["error"]))
            self.assertFalse(low_options.runs_root.exists())
            self.assertEqual(list(low_root.rglob("run_summary.json")), [])

            cuda_root = Path(temporary) / "cuda-failure"
            cuda_root.mkdir()
            cuda_options, cuda_dependencies = _bootstrap_resume_fixture(cuda_root)

            def fail_cuda(_requested: str):
                raise RuntimeError(
                    "CUDA profile verification failed: selected_profile=cu124, "
                    "torch_cuda=false"
                )

            cuda_options = replace(cuda_options, compute="cuda")
            cuda_report = run_bootstrap(
                cuda_options,
                dependencies=replace(
                    cuda_dependencies,
                    compute_resolver=fail_cuda,
                ),
            )
            self.assertEqual(cuda_report["failed_stage"], "environment")
            self.assertIn("selected_profile=cu124", str(cuda_report["error"]))
            self.assertIn("--compute cpu", str(cuda_report["recovery_command"]))
            self.assertIn(
                "--restart-stage environment", str(cuda_report["recovery_command"])
            )
            self.assertEqual(list(cuda_options.runs_root.rglob("run_summary.json")), [])

    def test_resume_rejects_foreign_or_scientifically_unbound_checkpoint(self) -> None:
        import torch

        from bitguard_bnn.bootstrap.orchestrator import (
            _checkpoint_locator,
            _validate_checkpoint_locator,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_root = root / "runs"
            runs_root.mkdir()
            foreign_run = root / "foreign-run"
            foreign_run.mkdir()
            foreign_checkpoint = foreign_run / "last_training_state.pt"
            torch.save(
                {
                    "format_version": 4,
                    "scientific_signature": {
                        "prepared_descriptor_fingerprint": "prepared",
                        "training_transform": {"role": "main"},
                    },
                },
                foreign_checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "escapes runs_root"):
                _validate_checkpoint_locator(
                    _checkpoint_locator(foreign_run, foreign_checkpoint),
                    runs_root=runs_root,
                    expected_descriptor_fingerprint="prepared",
                )

            local_run = runs_root / "local-run"
            local_run.mkdir()
            local_checkpoint = local_run / "last_training_state.pt"
            torch.save(
                {
                    "format_version": 4,
                    "scientific_signature": {
                        "prepared_descriptor_fingerprint": "foreign-prepared",
                        "training_transform": {"role": "main"},
                    },
                },
                local_checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                _validate_checkpoint_locator(
                    _checkpoint_locator(local_run, local_checkpoint),
                    runs_root=runs_root,
                    expected_descriptor_fingerprint="prepared",
                )

    def test_resume_checkpoint_is_pinned_by_digest_before_training_load(self) -> None:
        import torch

        from bitguard_bnn.out_of_core.run import _pin_verified_resume_checkpoint
        from bitguard_bnn.trainer import _safe_torch_load

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "foreign" / "last_training_state.pt"
            source.parent.mkdir()
            original = {"marker": "validated", "tensor": torch.tensor([1, 2, 3])}
            replacement = {
                "marker": "replacement",
                "tensor": torch.tensor([9, 9, 9]),
            }
            torch.save(original, source)
            expected_sha256 = _sha256(source)
            expected_bytes = source.stat().st_size

            replacement_path = root / "replacement.pt"
            torch.save(replacement, replacement_path)
            os.replace(replacement_path, source)
            destination = root / "controlled-run" / "resume_training_state.pt"
            destination.parent.mkdir()
            with self.assertRaisesRegex(RuntimeError, "checkpoint.*changed"):
                _pin_verified_resume_checkpoint(
                    source,
                    destination,
                    expected_sha256=expected_sha256,
                    expected_bytes=expected_bytes,
                )
            self.assertFalse(destination.exists())

            torch.save(original, source)
            expected_sha256 = _sha256(source)
            expected_bytes = source.stat().st_size
            pinned = _pin_verified_resume_checkpoint(
                source,
                destination,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
            )
            torch.save(replacement, source)
            loaded = _safe_torch_load(
                pinned,
                torch.device("cpu"),
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
            )
            self.assertEqual(loaded["marker"], "validated")
            torch.testing.assert_close(loaded["tensor"], original["tensor"])
            with self.assertRaises(FileExistsError):
                _pin_verified_resume_checkpoint(
                    source,
                    destination,
                    expected_sha256=_sha256(source),
                    expected_bytes=source.stat().st_size,
                )

    def test_bootstrap_rerun_reopens_verified_validation_cache_journal(self) -> None:
        import torch

        from bitguard_bnn.out_of_core.cache import CalibrationCache
        from bitguard_bnn.out_of_core.run import _validation_calibration_cache

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies = _bootstrap_resume_fixture(root)
            layout = cache_layout_fixture(rows=6)
            attempts: list[Path] = []
            resumed_committed_rows: list[int] = []

            def fake_training(
                _config_path: str | Path,
                *,
                config: dict[str, object],
                prepared_descriptor_path: str | Path,
                progress_callback=None,
                resume_checkpoint_integrity=None,
                **_resources: object,
            ) -> Path:
                run_dir = options.runs_root / f"cache-attempt-{len(attempts) + 1}"
                run_dir.mkdir(parents=True)
                attempts.append(run_dir)
                active = run_dir / "last_training_state.pt"
                resume_value = config["training"].get("resume_from")  # type: ignore[index,union-attr]
                self.assertIsNotNone(progress_callback)
                progress_callback(
                    {
                        "status": "run_created",
                        "dataset": "nbaiot",
                        "prepared_descriptor": str(prepared_descriptor_path),
                        "prepared_descriptor_fingerprint": (
                            "fixture-prepared-fingerprint"
                        ),
                        "run_dir": str(run_dir.resolve()),
                        "active_checkpoint": str(active.resolve()),
                        "resume_checkpoint": (
                            None
                            if resume_value is None
                            else str(Path(str(resume_value)).resolve())
                        ),
                    }
                )
                torch.save(
                    {
                        "format_version": 4,
                        "device_type": "cpu",
                        "scientific_signature": {
                            "prepared_descriptor_fingerprint": (
                                "fixture-prepared-fingerprint"
                            ),
                            "training_transform": {"role": "main"},
                        },
                    },
                    active,
                )
                current_cache = run_dir / "temporary" / "validation-cache"
                current_cache.parent.mkdir()
                if len(attempts) == 1:
                    cache = CalibrationCache.create(current_cache, layout)
                    cache.commit_inference_range(0, cache_inference_batch(0, 2))
                    cache.close()
                    raise RuntimeError("hard exit after committed cache journal")

                self.assertIsNotNone(resume_value)
                with _validation_calibration_cache(
                    current_cache,
                    layout,
                    resume_from=Path(str(resume_value)),
                ) as cache:
                    resumed_committed_rows.append(cache.committed_rows)
                    cache.commit_inference_range(2, cache_inference_batch(2, 6))
                    self.assertEqual(cache.committed_rows, 6)
                _write_fake_completed_run(
                    run_dir,
                    dataset="nbaiot",
                    descriptor_path=Path(prepared_descriptor_path),
                )
                return run_dir

            with patch(
                "bitguard_bnn.out_of_core.run.run_out_of_core_training",
                side_effect=fake_training,
            ):
                first = run_bootstrap(options, dependencies=dependencies)
                second = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(first["failed_stage"], "train")
            self.assertEqual(second["status"], "completed", msg=second.get("error"))
            self.assertEqual(resumed_committed_rows, [2])
            self.assertFalse((attempts[0] / "temporary" / "validation-cache").exists())
            self.assertFalse((attempts[1] / "temporary" / "validation-cache").exists())

    def test_bootstrap_subprocess_hard_exit_resumes_committed_validation_cache(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = _subprocess_environment(repository)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_nbaiot_training_fixture(root / "nbaiot.zip")
            write_fast_nbaiot_profile(root / "nbaiot-fast.yaml")
            interrupted_expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_validation_cache_exit_fixture_child as run; "
                f"run({str(root)!r}, kill_after_cache_commit=True)"
            )
            interrupted = subprocess.run(
                [sys.executable, "-c", interrupted_expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                interrupted.returncode,
                94,
                msg=interrupted.stdout + interrupted.stderr,
            )
            training = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )["datasets"]["nbaiot"]
            interrupted_run = Path(training["active_checkpoint"]["run"])
            cache_root = interrupted_run / "temporary" / "validation-cache"
            journal = json.loads(
                (cache_root / "inference_journal.json").read_text(encoding="utf-8")
            )
            self.assertGreater(journal["committed_rows"], 0)

            resumed_expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_validation_cache_exit_fixture_child as run; "
                f"run({str(root)!r}, kill_after_cache_commit=False)"
            )
            resumed = subprocess.run(
                [sys.executable, "-c", resumed_expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                resumed.returncode,
                0,
                msg=resumed.stdout + resumed.stderr,
            )
            self.assertFalse(cache_root.exists())
            completed = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(completed["status"], "completed")
            resumed_run = Path(completed["runs"]["nbaiot"])

            control_expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_validation_cache_exit_fixture_child as run; "
                f"run({str(root)!r}, kill_after_cache_commit=False, "
                "restart_train=True)"
            )
            control = subprocess.run(
                [sys.executable, "-c", control_expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                control.returncode,
                0,
                msg=control.stdout + control.stderr,
            )
            control_training = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )
            control_run = Path(control_training["runs"]["nbaiot"])
            self.assertNotEqual(resumed_run, control_run)
            self.assertEqual(
                json.loads((resumed_run / "metrics.json").read_text(encoding="utf-8")),
                json.loads((control_run / "metrics.json").read_text(encoding="utf-8")),
            )
            resumed_contract = json.loads(
                (resumed_run / "inference_contract.json").read_text(encoding="utf-8")
            )
            control_contract = json.loads(
                (control_run / "inference_contract.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                resumed_contract["fingerprint"], control_contract["fingerprint"]
            )
            self.assertEqual(resumed_contract["contract"], control_contract["contract"])
            self.assertEqual(
                _sha256(resumed_run / "predictions.parquet"),
                _sha256(control_run / "predictions.parquet"),
            )
            self.assertEqual(
                _sha256(resumed_run / "edge" / "bitguard_edge_weights.npz"),
                _sha256(control_run / "edge" / "bitguard_edge_weights.npz"),
            )
            resumed_export = resumed_run / "edge" / "bitguard_edge_manifest.json"
            control_export = control_run / "edge" / "bitguard_edge_manifest.json"
            self.assertEqual(
                json.loads(resumed_export.read_text(encoding="utf-8")),
                json.loads(control_export.read_text(encoding="utf-8")),
            )
            self.assertEqual(_sha256(resumed_export), _sha256(control_export))

    def test_cuda_amp_checkpoint_resumes_through_bootstrap_when_available(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        repository = Path(__file__).resolve().parents[1]
        environment = _subprocess_environment(repository)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_nbaiot_training_fixture(root / "nbaiot.zip")
            write_fast_cuda_nbaiot_profile(root / "nbaiot-cuda-fast.yaml")
            interrupted_expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_cuda_resume_fixture_child as run; "
                f"run({str(root)!r}, kill_after_uncommitted_step=True)"
            )
            interrupted = subprocess.run(
                [sys.executable, "-c", interrupted_expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
            self.assertEqual(
                interrupted.returncode,
                95,
                msg=interrupted.stdout + interrupted.stderr,
            )
            failed_training = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )["datasets"]["nbaiot"]
            checkpoint = Path(failed_training["active_checkpoint"]["path"])
            self.assertTrue(checkpoint.is_file())

            resumed_expression = (
                "from tests.test_full_bootstrap_recovery import "
                "run_cuda_resume_fixture_child as run; "
                f"run({str(root)!r}, kill_after_uncommitted_step=False)"
            )
            resumed = subprocess.run(
                [sys.executable, "-c", resumed_expression],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
            self.assertEqual(
                resumed.returncode,
                0,
                msg=resumed.stdout + resumed.stderr,
            )
            environment_report = json.loads(
                (root / "data" / ".bitguard" / "environment.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(environment_report["compute"]["device"], "cuda:0")
            self.assertIn(
                environment_report["compute"]["selected_profile"],
                {"cu118", "cu124", "cu128"},
            )
            completed_training = json.loads(
                (root / "data" / ".bitguard" / "training.json").read_text(
                    encoding="utf-8"
                )
            )
            completed_run = Path(completed_training["runs"]["nbaiot"])
            run_environment = json.loads(
                (completed_run / "environment.json").read_text(encoding="utf-8")
            )
            self.assertTrue(run_environment["deterministic_algorithms"])
            self.assertFalse(run_environment["cudnn_benchmark"])
            from bitguard_bnn.trainer import _safe_torch_load

            completed_checkpoint = _safe_torch_load(
                completed_run / "last_training_state.pt", torch.device("cpu")
            )
            self.assertEqual(completed_checkpoint["device_type"], "cuda")
            self.assertIsInstance(completed_checkpoint["scaler_state_dict"], dict)
            self.assertTrue(completed_checkpoint["scaler_state_dict"])

    def test_prepared_reuse_runs_training_disk_and_ram_preflight_before_run_dir(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        config_path = repository / "configs" / "full" / "nbaiot.yaml"
        for constrained, expected in (("disk", "disk"), ("ram", "RAM")):
            with self.subTest(constrained=constrained):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    options, dependencies = _bootstrap_resume_fixture(root)
                    prepared_only = run_bootstrap(
                        replace(options, prepare_only=True),
                        dependencies=dependencies,
                    )
                    self.assertEqual(
                        prepared_only["status"],
                        "prepared",
                        msg=prepared_only.get("error"),
                    )
                    descriptor = options.data_root / ".bitguard" / "prepared" / "nbaiot"
                    prepared = replace(
                        _placeholder_prepared(root, config_path),
                        descriptor_path=str(descriptor.resolve()),
                    )
                    injected = replace(
                        dependencies,
                        prepared_verifier=lambda _path: prepared,
                        training_available_bytes=(
                            1 if constrained == "disk" else 10**12
                        ),
                        training_available_ram_bytes=(
                            1 if constrained == "ram" else 10**12
                        ),
                    )
                    with (
                        patch(
                            "bitguard_bnn.out_of_core.run.verify_prepared_dataset",
                            return_value=prepared,
                        ),
                        patch(
                            "bitguard_bnn.out_of_core.run.create_run_dir"
                        ) as create_run_dir,
                    ):
                        failed = run_bootstrap(options, dependencies=injected)

                    self.assertEqual(failed["failed_stage"], "train")
                    self.assertIn(expected, str(failed["error"]))
                    self.assertIn("required=", str(failed["error"]))
                    self.assertIn("available=1", str(failed["error"]))
                    self.assertIn("shard", failed["reused_stages"])
                    create_run_dir.assert_not_called()

    def test_environment_recovery_only_suggests_cpu_for_cuda_validation(self) -> None:
        from bitguard_bnn.bootstrap.orchestrator import _recovery

        generic = _recovery(
            "environment",
            RuntimeError("environment report publication failed"),
            {"selected_profile": "cpu", "device": "cpu"},
        )
        self.assertNotIn("--compute cpu", generic)
        self.assertIn("--restart-stage environment", generic)
        for message in (
            "CUDA profile verification failed: Torch reports unavailable",
            "Torch tensor allocation verification failed on cuda:0: allocator failed",
            "CUDA synchronization verification failed: sync failed",
            "CUDA device-name verification failed: invalid name",
            "Torch CUDA build version verification failed: mismatch",
        ):
            with self.subTest(message=message):
                cuda = _recovery(
                    "environment",
                    RuntimeError(message),
                    None,
                )
                self.assertIn("--compute cpu", cuda)
                self.assertIn("--restart-stage environment", cuda)

    def test_train_recovery_preserves_resume_unless_reset_is_required(self) -> None:
        from bitguard_bnn.bootstrap.orchestrator import _recovery

        normal = _recovery(
            "train",
            RuntimeError("worker process stopped unexpectedly"),
            {"compute": {"selected_profile": "cpu", "device": "cpu"}},
        )
        self.assertIn("rerun the original command", normal)
        self.assertIn("automatic optimizer/cache resume", normal)
        self.assertNotIn("--restart-stage", normal)
        reset_errors = (
            "streaming checkpoint fields are invalid",
            "unsupported streaming checkpoint format",
            "streaming checkpoint scientific signature mismatch",
            "training.epochs must match the streaming checkpoint target",
            "streaming checkpoint device type mismatch",
            "streaming checkpoint cursor is invalid",
            "streaming checkpoint optimizer state is invalid",
            "streaming checkpoint partial totals are non-finite",
            "streaming checkpoint cursor, rows, and step are inconsistent",
            "training checkpoint is not a safe tensor payload",
            "training checkpoint contains a recursive mapping",
            "training checkpoint optimizer state is invalid",
            "training checkpoint best/stale state is inconsistent",
            "training.epochs must match the checkpoint target; resume an interrupted "
            "run with its original total epoch count",
            "training resume checkpoint is incompatible",
            "streaming checkpoint distillation resume requires independently "
            "verified teacher identity",
        )
        for message in reset_errors:
            with self.subTest(message=message):
                reset = _recovery(
                    "train",
                    RuntimeError(message),
                    {"compute": {"selected_profile": "cpu", "device": "cpu"}},
                )
                self.assertIn("--restart-stage train", reset)

        transient_open_error = ValueError(
            "training checkpoint could not be opened safely"
        )
        transient_open_error.__cause__ = PermissionError(
            "invalid argument while opening checkpoint"
        )
        for error in (
            transient_open_error,
            FileNotFoundError("training checkpoint not found"),
        ):
            with self.subTest(transient=str(error)):
                recovery = _recovery(
                    "train",
                    error,
                    {"compute": {"selected_profile": "cpu", "device": "cpu"}},
                )
                self.assertIn("rerun the original command", recovery)
                self.assertIn("automatic optimizer/cache resume", recovery)
                self.assertNotIn("--restart-stage", recovery)

    def test_cuda_training_failures_require_fresh_cpu_restart(self) -> None:
        import torch

        cuda_compute = {"compute": {"selected_profile": "cu124", "device": "cuda:0"}}
        from bitguard_bnn.bootstrap.orchestrator import _recovery

        failures = (
            RuntimeError("CUDA out of memory while allocating a tensor"),
            RuntimeError("CUDA synchronization verification failed: sync failed"),
            RuntimeError("CUDA device-name verification failed: invalid name"),
            RuntimeError("Torch CUDA build version verification failed: mismatch"),
        )
        for error in failures:
            with self.subTest(error=str(error)):
                recovery = _recovery("train", error, cuda_compute)
                self.assertIn("--compute cpu --restart-stage train", recovery)
                self.assertIn("starts fresh training", recovery)
                self.assertIn("optimizer/cache resume", recovery)
        unrelated_cuda_path = _recovery(
            "train",
            FileNotFoundError(r"C:\datasets\cuda\missing.csv"),
            cuda_compute,
        )
        self.assertNotIn("--compute cpu", unrelated_cuda_path)
        self.assertNotIn("--restart-stage", unrelated_cuda_path)

        cu128_profile_only = _recovery(
            "train",
            RuntimeError("CUDA out of memory while allocating a tensor"),
            {"compute": {"selected_profile": "cu128"}},
        )
        self.assertIn("--compute cpu --restart-stage train", cu128_profile_only)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies = _bootstrap_resume_fixture(root)
            cuda_dependencies = replace(
                dependencies,
                compute_resolver=lambda requested: {
                    "requested": requested,
                    "selected_profile": "cu124" if requested != "cpu" else "cpu",
                    "device": "cuda:0" if requested != "cpu" else "cpu",
                },
            )
            attempts = 0
            observed_resume_values: list[object] = []

            def fake_training(
                _config_path: str | Path,
                *,
                config: dict[str, object],
                prepared_descriptor_path: str | Path,
                progress_callback=None,
                **_runtime_options: object,
            ) -> Path:
                nonlocal attempts
                attempts += 1
                training = config["training"]
                assert isinstance(training, dict)
                observed_resume_values.append(training.get("resume_from"))
                run_dir = options.runs_root / f"cuda-cpu-attempt-{attempts}"
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "last_training_state.pt"
                if attempts == 1:
                    assert progress_callback is not None
                    progress_callback(
                        {
                            "status": "run_created",
                            "dataset": "nbaiot",
                            "prepared_descriptor": str(prepared_descriptor_path),
                            "prepared_descriptor_fingerprint": (
                                "fixture-prepared-fingerprint"
                            ),
                            "run_dir": str(run_dir.resolve()),
                            "active_checkpoint": str(checkpoint.resolve()),
                            "resume_checkpoint": None,
                        }
                    )
                    torch.save(
                        {
                            "format_version": 4,
                            "device_type": "cuda",
                            "scientific_signature": {
                                "prepared_descriptor_fingerprint": (
                                    "fixture-prepared-fingerprint"
                                ),
                                "training_transform": {"role": "main"},
                            },
                        },
                        checkpoint,
                    )
                    raise RuntimeError("CUDA out of memory while allocating a tensor")
                _write_fake_completed_run(
                    run_dir,
                    dataset="nbaiot",
                    descriptor_path=Path(prepared_descriptor_path),
                )
                return run_dir

            with patch(
                "bitguard_bnn.out_of_core.run.run_out_of_core_training",
                side_effect=fake_training,
            ):
                failed = run_bootstrap(
                    replace(options, compute="cu124"),
                    dependencies=cuda_dependencies,
                )
                rejected_cpu_resume = run_bootstrap(
                    replace(options, compute="cpu"),
                    dependencies=cuda_dependencies,
                )
                recovered = run_bootstrap(
                    replace(options, compute="cpu", restart_stage="train"),
                    dependencies=cuda_dependencies,
                )

            self.assertEqual(failed["failed_stage"], "train")
            self.assertIn(
                "--compute cpu --restart-stage train",
                str(failed["recovery_command"]),
            )
            self.assertEqual(rejected_cpu_resume["failed_stage"], "train")
            self.assertIn("device type mismatch", str(rejected_cpu_resume["error"]))
            self.assertIn(
                "--restart-stage train",
                str(rejected_cpu_resume["recovery_command"]),
            )
            self.assertEqual(recovered["status"], "completed", recovered.get("error"))
            self.assertEqual(observed_resume_values, [None, None])

    def test_same_host_absent_pid_lock_is_reclaimed_after_process_exit(self) -> None:
        from bitguard_bnn.bootstrap.orchestrator import _lock_path
        from bitguard_bnn.bootstrap.state import BootstrapWriterLock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options, dependencies = _bootstrap_resume_fixture(root)
            lock_path = _lock_path(options)
            abandoned = BootstrapWriterLock(lock_path, pid=2_147_483_647)
            abandoned.acquire()

            report = run_bootstrap(
                options,
                dependencies=replace(dependencies, available_bytes=0),
            )

            self.assertEqual(report["failed_stage"], "preflight")
            self.assertIn("Insufficient disk space", str(report["error"]))
            self.assertNotIn("appears stale", str(report["error"]))
            self.assertFalse(lock_path.exists())

    def test_one_cpu_bootstrap_command_trains_both_profiles_and_is_idempotent(
        self,
    ) -> None:
        import torch

        from bitguard_bnn.out_of_core.prepare import (
            prepare_full_dataset,
            verify_prepared_dataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "nbaiot.zip"
            botiot_source = root / "botiot"
            write_nbaiot_training_fixture(archive)
            write_botiot_training_fixture(botiot_source)
            profiles = {
                "nbaiot": root / "nbaiot-fast.yaml",
                "botiot": root / "botiot-fast.yaml",
            }
            write_fast_nbaiot_profile(profiles["nbaiot"])
            write_fast_botiot_profile(profiles["botiot"])
            prepare_calls: list[str] = []

            def prepare(_config: Path, **kwargs: object) -> object:
                dataset = Path(str(kwargs["descriptor_path"])).parent.name
                prepare_calls.append(dataset)
                return prepare_full_dataset(profiles[dataset], **kwargs)

            options = BootstrapOptions(
                datasets=("nbaiot", "botiot"),
                botiot_source=botiot_source.resolve(),
                data_root=(root / "data").resolve(),
                runs_root=(root / "runs").resolve(),
                compute="cpu",
                prepare_only=False,
                install_system_tools=False,
                accepted_botiot_license=True,
                restart_stage=None,
            )
            token = ":".join(_sha256(path) for path in profiles.values())
            dependencies = BootstrapDependencies(
                nbaiot_archive=archive,
                available_bytes=10**12,
                preparation_available_bytes=10**12,
                compute_resolver=lambda requested: {
                    "requested": requested,
                    "selected_profile": "cpu",
                    "device": "cpu",
                },
                preparer=prepare,
                preparation_signature_token=f"dual-profile-fixture:{token}",
            )

            first = run_bootstrap(options, dependencies=dependencies)
            self.assertEqual(first["status"], "completed", msg=first.get("error"))
            self.assertEqual(prepare_calls, ["nbaiot", "botiot"])
            first_runs = {
                dataset: Path(value) for dataset, value in first["trained_runs"].items()
            }
            first_artifact_hashes: dict[str, dict[str, str]] = {}
            for dataset in ("nbaiot", "botiot"):
                prepared = verify_prepared_dataset(
                    Path(first["prepared_datasets"][dataset])
                )
                self.assertEqual(
                    prepared.train_count
                    + prepared.validation_count
                    + prepared.test_count,
                    prepared.total_count,
                )
                run_dir = first_runs[dataset]
                history_lines = (
                    (run_dir / "training_history.csv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                self.assertEqual(len(history_lines), 2)
                checkpoint = torch.load(
                    run_dir / "last_training_state.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(len(checkpoint["history"]), 1)
                self.assertEqual(
                    checkpoint["scientific_signature"][
                        "prepared_descriptor_fingerprint"
                    ],
                    prepared.to_dict()["fingerprint"],
                )
                model_summary = json.loads(
                    (run_dir / "model_summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    model_summary["train_rows_per_epoch"], prepared.train_count
                )
                metrics = json.loads(
                    (run_dir / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metrics["fixed_fpr"]["threshold_source"], "validation")
                contract = json.loads(
                    (run_dir / "inference_contract.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    contract["contract"]["prepared"]["descriptor_fingerprint"],
                    prepared.to_dict()["fingerprint"],
                )
                manifest = json.loads(
                    (run_dir / "edge" / "bitguard_edge_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(manifest["folding_parity_passed"])
                first_artifact_hashes[dataset] = {
                    name: _sha256(run_dir / name)
                    for name in (
                        "run_summary.json",
                        "metrics.json",
                        "inference_contract.json",
                        "best_model.pt",
                    )
                }

            second = run_bootstrap(options, dependencies=dependencies)
            self.assertEqual(second["status"], "completed", msg=second.get("error"))
            self.assertEqual(prepare_calls, ["nbaiot", "botiot"])
            self.assertEqual(
                {
                    dataset: Path(value)
                    for dataset, value in second["trained_runs"].items()
                },
                first_runs,
            )
            self.assertEqual(
                {
                    dataset: {
                        name: _sha256(first_runs[dataset] / name) for name in hashes
                    }
                    for dataset, hashes in first_artifact_hashes.items()
                },
                first_artifact_hashes,
            )
            self.assertEqual(
                {
                    path.resolve()
                    for path in options.runs_root.rglob("run_summary.json")
                },
                {path / "run_summary.json" for path in first_runs.values()},
            )


if __name__ == "__main__":
    unittest.main()
