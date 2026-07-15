from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from bitguard_bnn.bootstrap.inspect import inspect_csv_dataset
from bitguard_bnn.bootstrap.manifest import build_source_manifest, write_source_manifest
from bitguard_bnn.bootstrap.orchestrator import BootstrapDependencies, run_bootstrap
from bitguard_bnn.bootstrap.registry import load_registry
from bitguard_bnn.bootstrap.types import BootstrapOptions
from bitguard_bnn.config import load_config


def _write_nbaiot(root: Path, *, heldout_offset: float = 0.0) -> None:
    devices = ("Ecobee_Thermostat", "Philips_B120N10_Baby_Monitor")
    for device_index, device in enumerate(devices):
        benign = root / device / "benign_traffic.csv"
        attack = root / device / "gafgyt_attacks" / "scan.csv"
        benign.parent.mkdir(parents=True, exist_ok=True)
        attack.parent.mkdir(parents=True, exist_ok=True)
        benign.write_text(
            "mean,std\n"
            + "".join(
                f"{1 + device_index + row},{2 + row / 10}\n" for row in range(8)
            ),
            encoding="utf-8",
        )
        attack.write_text(
            "mean,std\n"
            + "".join(
                f"{20 + device_index + row},{4 + row / 10}\n" for row in range(8)
            ),
            encoding="utf-8",
        )

    heldout = root / "Danmini_Doorbell" / "benign_traffic.csv"
    heldout.parent.mkdir(parents=True, exist_ok=True)
    heldout.write_text(
        "mean,std\n"
        + "".join(
            f"{1000 + heldout_offset + row},{2000 + heldout_offset + row}\n"
            for row in range(8)
        ),
        encoding="utf-8",
    )


def _write_botiot(root: Path) -> None:
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


def _source_contract(dataset: str, raw_root: Path, root: Path) -> tuple[Path, Path]:
    spec = load_registry()[dataset]
    manifest = build_source_manifest(
        raw_root,
        spec,
        acquisition_method=(
            "official-download" if dataset == "nbaiot" else "manual-local-source"
        ),
        acquisition_url=spec.download_url if dataset == "nbaiot" else None,
    )
    contract = root / "contract"
    contract.mkdir(parents=True, exist_ok=True)
    manifest_path = contract / "source.json"
    write_source_manifest(manifest_path, manifest)
    schema = inspect_csv_dataset(
        dataset,
        raw_root,
        required_columns=spec.required_columns,
    )
    schema_path = contract / "schema.json"
    schema_path.write_text(
        json.dumps(schema.as_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest_path, schema_path


class FullProfileConfigTests(unittest.TestCase):
    def test_nested_configs_directory_resolves_project_root_and_parquet_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "configs" / "full"
            nested.mkdir(parents=True)
            payload = {
                "dataset": {
                    "type": "csv",
                    "path": "data/source.csv",
                    "storage": "parquet",
                    "shard_manifest": "data/prepared/fixture/shard_manifest.json",
                    "record_batch_rows": 32,
                    "shard_target_rows": 64,
                    "quantile_sketch_capacity": 128,
                }
            }
            path = nested / "fixture.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            config = load_config(path)

            self.assertEqual(Path(config["_project_root"]), root.resolve())
            self.assertEqual(config["dataset"]["storage"], "parquet")

    def test_repository_full_profiles_are_uncapped_and_separate(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for dataset in ("nbaiot", "botiot"):
            config = load_config(repository / "configs" / "full" / f"{dataset}.yaml")
            values = config["dataset"]
            self.assertEqual(values["storage"], "parquet")
            self.assertIsNone(values["max_rows_per_file"])
            self.assertIsNone(values["max_rows_per_class"])
            self.assertIsNone(values["max_loaded_rows"])
            self.assertEqual(values["record_batch_rows"], 65_536)
            self.assertEqual(values["shard_target_rows"], 1_000_000)
            self.assertEqual(values["quantile_sketch_capacity"], 200_000)
            self.assertIn(f"data/prepared/{dataset}/", values["shard_manifest"])


class FullDatasetPreparationTests(unittest.TestCase):
    def _prepare(
        self, dataset: str, root: Path, *, heldout_offset: float = 0.0
    ):
        from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

        raw_root = root / "raw"
        source_manifest = root / "contract" / "source.json"
        schema_report = root / "contract" / "schema.json"
        if not raw_root.exists():
            if dataset == "nbaiot":
                _write_nbaiot(raw_root, heldout_offset=heldout_offset)
            else:
                _write_botiot(raw_root)
            source_manifest, schema_report = _source_contract(dataset, raw_root, root)
        repository = Path(__file__).resolve().parents[1]
        return prepare_full_dataset(
            repository / "configs" / "full" / f"{dataset}.yaml",
            raw_root=raw_root,
            source_manifest_path=source_manifest,
            schema_report_path=schema_report,
            output_dir=root / "prepared",
            descriptor_path=root / "control" / f"{dataset}.json",
            work_dir=root / "work",
        )

    def test_nbaiot_and_botiot_prepare_every_source_row_and_reverify(self) -> None:
        from bitguard_bnn.out_of_core.prepare import (
            PreparedDataset,
            verify_prepared_dataset,
        )

        for dataset, expected_rows in (("nbaiot", 40), ("botiot", 40)):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                first = self._prepare(dataset, root)
                self.assertIsInstance(first, PreparedDataset)
                self.assertEqual(first.total_count, expected_rows)
                self.assertEqual(
                    first.train_count + first.validation_count + first.test_count,
                    expected_rows,
                )
                self.assertEqual(
                    first.source_manifest_fingerprint,
                    json.loads(
                        Path(first.source_manifest_path).read_text(encoding="utf-8")
                    )["content_sha256"],
                )
                self.assertEqual(first, verify_prepared_dataset(first.descriptor_path))

                second = self._prepare(dataset, root)
                self.assertEqual(first, second)

                shard_manifest = json.loads(
                    Path(first.shard_manifest_path).read_text(encoding="utf-8")
                )
                shard = Path(first.shard_manifest_path).parent / shard_manifest["entries"][0]["path"]
                original = shard.read_bytes()
                shard.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    verify_prepared_dataset(first.descriptor_path)

    def test_preprocessor_joblib_and_feature_manifest_are_bound_together(self) -> None:
        from bitguard_bnn.preprocess import FeaturePreprocessor

        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepare("nbaiot", Path(temporary))
            joblib_path = Path(prepared.preprocessor_path)
            digest = hashlib.sha256(joblib_path.read_bytes()).hexdigest()
            self.assertEqual(digest, prepared.preprocessor_sha256)
            processor = FeaturePreprocessor.load(joblib_path)
            feature_payload = json.loads(
                Path(prepared.feature_manifest_path).read_text(encoding="utf-8")
            )
            self.assertEqual(
                feature_payload["feature_manifest"], processor.feature_manifest()
            )
            self.assertEqual(
                processor.fit_provenance["rows_considered"], prepared.train_count
            )
            self.assertFalse(processor.fit_provenance["validation_calibration_used"])

    def test_heldout_device_values_do_not_change_train_scientific_artifact(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_temporary,
            tempfile.TemporaryDirectory() as second_temporary,
        ):
            first = self._prepare("nbaiot", Path(first_temporary), heldout_offset=0.0)
            second = self._prepare(
                "nbaiot", Path(second_temporary), heldout_offset=1_000_000.0
            )
            first_feature = json.loads(
                Path(first.feature_manifest_path).read_text(encoding="utf-8")
            )
            second_feature = json.loads(
                Path(second.feature_manifest_path).read_text(encoding="utf-8")
            )
            self.assertNotEqual(first.split_fingerprint, second.split_fingerprint)
            self.assertEqual(
                first_feature["scientific_fingerprint"],
                second_feature["scientific_fingerprint"],
            )


class BootstrapPreparationStageTests(unittest.TestCase):
    def test_shard_and_validate_publish_control_state_and_reverify_on_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "official-botiot"
            _write_botiot(source)
            data_root = (root / "data").resolve()
            verification_calls: list[Path] = []

            def prepare(_config: Path, **kwargs: object) -> object:
                descriptor = Path(str(kwargs["descriptor_path"]))
                descriptor.parent.mkdir(parents=True, exist_ok=True)
                descriptor.write_text(
                    json.dumps({"dataset": "botiot"}, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(descriptor_path=str(descriptor))

            def verify(descriptor: Path) -> object:
                self.assertEqual(
                    json.loads(descriptor.read_text(encoding="utf-8")),
                    {"dataset": "botiot"},
                )
                verification_calls.append(descriptor)
                return SimpleNamespace(
                    to_dict=lambda: {
                        "dataset": "botiot",
                        "descriptor_path": str(descriptor),
                    }
                )

            dependencies = BootstrapDependencies(
                available_bytes=10**12,
                compute_resolver=lambda requested: {
                    "requested": requested,
                    "selected_profile": "cpu",
                    "device": "cpu",
                },
                preparer=prepare,
                prepared_verifier=verify,
            )
            options = BootstrapOptions(
                datasets=("botiot",),
                botiot_source=source.resolve(),
                data_root=data_root,
                runs_root=(root / "runs").resolve(),
                compute="cpu",
                prepare_only=True,
                install_system_tools=False,
                accepted_botiot_license=True,
                restart_stage=None,
            )

            first = run_bootstrap(options, dependencies=dependencies)
            self.assertEqual(first["status"], "prepared", msg=first.get("error"))
            self.assertEqual(first["last_completed_stage"], "validate")
            self.assertEqual(first["next_stage"], "train")
            self.assertEqual(
                first["executed_stages"],
                [
                    "preflight",
                    "environment",
                    "acquire",
                    "extract",
                    "inspect",
                    "shard",
                    "validate",
                ],
            )

            second = run_bootstrap(options, dependencies=dependencies)
            self.assertEqual(second["status"], "prepared", msg=second.get("error"))
            self.assertEqual(
                second["reused_stages"],
                ["preflight", "environment", "acquire", "extract", "inspect", "shard"],
            )
            self.assertEqual(second["executed_stages"], ["validate"])
            self.assertEqual(len(verification_calls), 2)

            changed = run_bootstrap(
                options,
                dependencies=replace(
                    dependencies, preparation_signature_token="new-science-code"
                ),
            )
            self.assertEqual(changed["status"], "prepared", msg=changed.get("error"))
            self.assertEqual(
                changed["reused_stages"],
                ["preflight", "environment", "acquire", "extract", "inspect"],
            )
            self.assertEqual(changed["executed_stages"], ["shard", "validate"])


if __name__ == "__main__":
    unittest.main()
