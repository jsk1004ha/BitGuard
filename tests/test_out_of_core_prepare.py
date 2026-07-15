from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from os import chdir, getcwd
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from bitguard_bnn.bootstrap.inspect import inspect_csv_dataset
from bitguard_bnn.bootstrap.manifest import build_source_manifest, write_source_manifest
from bitguard_bnn.bootstrap.orchestrator import BootstrapDependencies, run_bootstrap
from bitguard_bnn.bootstrap.registry import load_registry
from bitguard_bnn.bootstrap.types import BootstrapOptions
from bitguard_bnn.config import load_config, resolve_path


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


def _write_botiot_with_validation_only_boolean(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = ["category,subcategory,saddr,stime,bytes,rate\n"]
    for index in range(40):
        benign = index % 2 == 0
        rate = "" if index < 28 else str(1000.0 + index)
        rows.append(
            f"{'Normal' if benign else 'DDoS'},"
            f"{'Normal' if benign else 'TCP'},10.0.0.{index % 4 + 1},"
            f"{index + 0.5},{100 + index},{rate}\n"
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
                resolved = load_config(first.resolved_config_path)
                self.assertEqual(
                    resolve_path(resolved, resolved["dataset"]["path"]),
                    (root / "raw").resolve(),
                )
                self.assertEqual(
                    resolve_path(resolved, resolved["dataset"]["shard_manifest"]),
                    Path(first.shard_manifest_path).resolve(),
                )
                self.assertEqual(
                    Path(first.template_config_path),
                    (
                        Path(__file__).resolve().parents[1]
                        / "configs"
                        / "full"
                        / f"{dataset}.yaml"
                    ).resolve(),
                )
                self.assertEqual(
                    hashlib.sha256(
                        Path(first.resolved_config_path).read_bytes()
                    ).hexdigest(),
                    first.config_sha256,
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

    def test_boolean_availability_comes_from_source_not_train_usable_features(self) -> None:
        from bitguard_bnn.out_of_core.prepare import (
            prepare_full_dataset,
            verify_prepared_dataset,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_botiot_with_validation_only_boolean(raw)
            source_manifest, schema_report = _source_contract("botiot", raw, root)
            template = root / "configs" / "full" / "botiot.yaml"
            template.parent.mkdir(parents=True)
            payload = yaml.safe_load(
                (
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "full"
                    / "botiot.yaml"
                ).read_text(encoding="utf-8")
            )
            payload["dataset"]["record_batch_rows"] = 4
            payload["dataset"]["shard_target_rows"] = 3
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
                    "boolean_fast_path_features": ["rate", "absent_feature"],
                }
            )
            template.write_text(yaml.safe_dump(payload), encoding="utf-8")

            prepared = prepare_full_dataset(
                template,
                raw_root=raw,
                source_manifest_path=source_manifest,
                schema_report_path=schema_report,
                output_dir=root / "prepared",
                descriptor_path=root / "control" / "botiot.json",
                work_dir=root / "work",
            )
            manifest = json.loads(
                Path(prepared.shard_manifest_path).read_text(encoding="utf-8")
            )
            expected_boolean = {
                "configured_features": ["rate", "absent_feature"],
                "available_features": ["rate"],
                "missing_features": ["absent_feature"],
            }
            self.assertEqual(manifest["selected_features"], ["bytes"])
            self.assertEqual(manifest["materialized_features"], ["bytes", "rate"])
            self.assertEqual(manifest["boolean_fast_path"], expected_boolean)
            feature = json.loads(
                Path(prepared.feature_manifest_path).read_text(encoding="utf-8")
            )
            self.assertEqual(feature["source_feature_order"], ["bytes", "rate"])
            self.assertEqual(feature["boolean_fast_path"], expected_boolean)
            from bitguard_bnn.preprocess import FeaturePreprocessor

            processor = FeaturePreprocessor.load(prepared.preprocessor_path)
            self.assertNotIn("rate", processor.candidate_features)
            heldout = [
                Path(prepared.shard_manifest_path).parent / entry["path"]
                for entry in manifest["entries"]
                if entry["split"] in {"validation", "test"}
            ]
            rates = np.concatenate(
                [
                    pq.read_table(path, columns=["rate"])["rate"].to_numpy()
                    for path in heldout
                ]
            )
            self.assertTrue(np.isfinite(rates).all())
            self.assertEqual(prepared, verify_prepared_dataset(prepared.descriptor_path))

    def test_fixed_descriptor_rejects_conflicting_work_identity(self) -> None:
        from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare("botiot", root)
            with self.assertRaisesRegex(RuntimeError, "work_dir"):
                prepare_full_dataset(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "full"
                    / "botiot.yaml",
                    raw_root=root / "raw",
                    source_manifest_path=root / "contract" / "source.json",
                    schema_report_path=root / "contract" / "schema.json",
                    output_dir=root / "prepared",
                    descriptor_path=root / "control" / "botiot.json",
                    work_dir=root / "different-work",
                )

    def test_generated_config_materializes_template_relative_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            config_dir = project / "configs" / "full"
            config_dir.mkdir(parents=True)
            payload = yaml.safe_load(
                (
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "full"
                    / "botiot.yaml"
                ).read_text(encoding="utf-8")
            )
            payload["preprocess"]["feature_cost_csv"] = "costs/feature-costs.csv"
            payload["experiment"]["output_dir"] = "runs/full"
            template = config_dir / "botiot.yaml"
            template.write_text(yaml.safe_dump(payload), encoding="utf-8")
            costs = project / "costs" / "feature-costs.csv"
            costs.parent.mkdir(parents=True)
            costs.write_text("feature,cost\nbytes,2\nrate,1\n", encoding="utf-8")
            raw = root / "raw"
            _write_botiot(raw)
            manifest, schema = _source_contract("botiot", raw, root)
            original_cwd = getcwd()
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            try:
                chdir(elsewhere)
                from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

                prepared = prepare_full_dataset(
                    template,
                    raw_root=raw,
                    source_manifest_path=manifest,
                    schema_report_path=schema,
                    output_dir=root / "prepared",
                    descriptor_path=root / "control" / "botiot.json",
                    work_dir=root / "work",
                )
            finally:
                chdir(original_cwd)
            resolved = load_config(prepared.resolved_config_path)
            self.assertEqual(
                Path(resolved["preprocess"]["feature_cost_csv"]), costs.resolve()
            )
            self.assertEqual(
                Path(resolved["experiment"]["output_dir"]),
                (project / "runs" / "full").resolve(),
            )

    def test_supplied_descriptor_link_is_rejected_before_resolution(self) -> None:
        from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            _write_botiot(raw)
            manifest, schema = _source_contract("botiot", raw, root)
            target = root / "foreign.json"
            target.write_text("{}\n", encoding="utf-8")
            descriptor = root / "descriptor.json"
            try:
                descriptor.symlink_to(target)
            except OSError as error:
                self.skipTest(f"file symlinks unavailable: {error}")
            with self.assertRaisesRegex(RuntimeError, "descriptor.*link|link.*descriptor"):
                prepare_full_dataset(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "full"
                    / "botiot.yaml",
                    raw_root=raw,
                    source_manifest_path=manifest,
                    schema_report_path=schema,
                    output_dir=root / "prepared",
                    descriptor_path=descriptor,
                    work_dir=root / "work",
                )
            self.assertEqual(target.read_text(encoding="utf-8"), "{}\n")

    def test_standalone_fingerprint_binds_local_preparation_contract(self) -> None:
        import bitguard_bnn.out_of_core.prepare as prepare_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._prepare("botiot", root)
            contract = prepare_module.preparation_implementation_contract()
            changed = {**contract, "algorithm": "injected-new-science"}
            with (
                patch.object(
                    prepare_module,
                    "preparation_implementation_contract",
                    return_value=changed,
                ),
                self.assertRaisesRegex(RuntimeError, "preparation_fingerprint"),
            ):
                self._prepare("botiot", root)
            self.assertEqual(first, prepare_module.verify_prepared_dataset(first.descriptor_path))

    def test_preparation_contract_invalidates_on_transitive_science_change(self) -> None:
        import bitguard_bnn.constants as constants_module
        import bitguard_bnn.out_of_core.prepare as prepare_module

        baseline = prepare_module.preparation_implementation_contract()
        implementations = baseline["implementations"]
        self.assertIsInstance(implementations, dict)
        self.assertTrue(
            {
                "bitguard_bnn.constants",
                "bitguard_bnn.preprocess",
                "bitguard_bnn.config",
                "bitguard_bnn.out_of_core.common",
                "bitguard_bnn.bootstrap.registry",
                "bitguard_bnn.bootstrap.inspect",
                "bitguard_bnn.bootstrap.manifest",
            }.issubset(implementations)
        )
        constants_path = Path(str(constants_module.__file__)).resolve()
        real_digest = prepare_module._sha256_file

        def changed_digest(path: Path) -> str:
            if path.resolve() == constants_path:
                return "0" * 64
            return real_digest(path)

        with patch.object(
            prepare_module, "_sha256_file", side_effect=changed_digest
        ):
            changed = prepare_module.preparation_implementation_contract()
        self.assertNotEqual(changed, baseline)

    def test_json_publication_never_replaces_concurrent_conflict(self) -> None:
        import bitguard_bnn.out_of_core.prepare as prepare_module

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "artifact.json"
            real_link = prepare_module.os.link

            def collide(source: object, destination: object, *args: object, **kwargs: object) -> None:
                Path(destination).write_text('{"winner":"other"}\n', encoding="utf-8")
                raise FileExistsError(str(destination))

            with (
                patch.object(prepare_module.os, "link", side_effect=collide),
                self.assertRaisesRegex(RuntimeError, "immutable JSON artifact conflict"),
            ):
                prepare_module._publish_json_immutable(target, {"winner": "ours"})
            self.assertEqual(target.read_text(encoding="utf-8"), '{"winner":"other"}\n')
            self.assertIsNotNone(real_link)

    def test_preprocessor_publication_reuses_concurrent_equivalent_without_replace(
        self,
    ) -> None:
        import bitguard_bnn.out_of_core.prepare as prepare_module
        from bitguard_bnn.preprocess import FeaturePreprocessor

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self._prepare("botiot", root)
            original = Path(prepared.preprocessor_path).read_bytes()
            processor = FeaturePreprocessor.load(prepared.preprocessor_path)
            sample = pd.DataFrame(
                np.zeros((2, len(processor.selected_features))),
                columns=processor.selected_features,
            )
            target = root / "concurrent-preprocessor.joblib"

            def collide(
                _source: object,
                destination: object,
                *args: object,
                **kwargs: object,
            ) -> None:
                Path(destination).write_bytes(original)
                raise FileExistsError(str(destination))

            with patch.object(prepare_module.os, "link", side_effect=collide):
                published, digest = prepare_module._publish_preprocessor(
                    processor, sample, target
                )
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(
                published.feature_manifest(), processor.feature_manifest()
            )
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())

    def test_disk_layout_groups_work_and_output_bytes_by_actual_device(self) -> None:
        from bitguard_bnn.bootstrap.orchestrator import (
            _group_preparation_disk_requirements,
        )
        from bitguard_bnn.out_of_core.prepare import PreparationDiskEstimate

        estimate = PreparationDiskEstimate(10, 20, 30, 40, 50, 60)
        groups = _group_preparation_disk_requirements(
            {"botiot": estimate},
            work_paths={"botiot": Path("work-device")},
            output_paths={"botiot": Path("output-device")},
            device_for=lambda path: 1 if path.name == "work-device" else 2,
        )

        self.assertEqual(groups[1]["required_bytes"], 60)
        self.assertEqual(groups[1]["datasets"]["botiot"]["work"], 60)
        self.assertEqual(groups[2]["required_bytes"], 150)
        self.assertEqual(groups[2]["datasets"]["botiot"]["output"], 150)

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
            preparation_calls: list[tuple[Path, Path, Path]] = []

            def prepare(_config: Path, **kwargs: object) -> object:
                descriptor = Path(str(kwargs["descriptor_path"]))
                output = Path(str(kwargs["output_dir"]))
                work = Path(str(kwargs["work_dir"]))
                preparation_calls.append((descriptor, output, work))
                descriptor.parent.mkdir(parents=True, exist_ok=True)
                payload = json.dumps(
                    {"dataset": "botiot", "generation": descriptor.stem},
                    sort_keys=True,
                ) + "\n"
                if descriptor.exists():
                    self.assertEqual(descriptor.read_text(encoding="utf-8"), payload)
                else:
                    descriptor.write_text(payload, encoding="utf-8")
                return SimpleNamespace(descriptor_path=str(descriptor))

            def verify(descriptor: Path) -> object:
                payload = json.loads(descriptor.read_text(encoding="utf-8"))
                self.assertEqual(payload["dataset"], "botiot")
                self.assertEqual(payload["generation"], descriptor.stem)
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

            import bitguard_bnn.bootstrap.orchestrator as orchestrator_module

            full_template = (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "full"
                / "botiot.yaml"
            ).resolve()
            real_regular_digest = orchestrator_module._regular_digest
            template_digest_calls = 0

            def counting_regular_digest(path: Path) -> tuple[str, int]:
                nonlocal template_digest_calls
                if path.resolve() == full_template:
                    template_digest_calls += 1
                return real_regular_digest(path)

            with patch.object(
                orchestrator_module,
                "_regular_digest",
                side_effect=counting_regular_digest,
            ):
                first = run_bootstrap(options, dependencies=dependencies)
            self.assertEqual(template_digest_calls, 2)
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
            self.assertEqual(len(verification_calls), 3)
            self.assertEqual(len(preparation_calls), 1)
            first_descriptor, first_output, first_work = preparation_calls[0]
            first_bytes = first_descriptor.read_bytes()

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
            self.assertEqual(len(preparation_calls), 2)
            changed_descriptor, changed_output, changed_work = preparation_calls[1]
            self.assertNotEqual(changed_descriptor, first_descriptor)
            self.assertNotEqual(changed_output, first_output)
            self.assertNotEqual(changed_work, first_work)
            self.assertEqual(first_descriptor.read_bytes(), first_bytes)
            self.assertTrue(changed_descriptor.is_file())

            restarted = run_bootstrap(
                replace(options, restart_stage="shard"),
                dependencies=replace(
                    dependencies, preparation_signature_token="new-science-code"
                ),
            )
            self.assertEqual(
                restarted["status"], "prepared", msg=restarted.get("error")
            )
            self.assertEqual(preparation_calls[-1][0], changed_descriptor)
            self.assertEqual(first_descriptor.read_bytes(), first_bytes)

    def test_shard_stage_rehash_rejects_mid_run_template_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "official-botiot"
            _write_botiot(source)

            def prepare(_config: Path, **kwargs: object) -> object:
                descriptor = Path(str(kwargs["descriptor_path"]))
                descriptor.parent.mkdir(parents=True, exist_ok=True)
                descriptor.write_text(
                    json.dumps(
                        {"dataset": "botiot", "generation": descriptor.stem},
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(descriptor_path=str(descriptor))

            dependencies = BootstrapDependencies(
                available_bytes=10**12,
                compute_resolver=lambda requested: {
                    "requested": requested,
                    "selected_profile": "cpu",
                    "device": "cpu",
                },
                preparer=prepare,
                prepared_verifier=lambda descriptor: SimpleNamespace(
                    descriptor_path=str(descriptor)
                ),
            )
            options = BootstrapOptions(
                datasets=("botiot",),
                botiot_source=source.resolve(),
                data_root=(root / "data").resolve(),
                runs_root=(root / "runs").resolve(),
                compute="cpu",
                prepare_only=True,
                install_system_tools=False,
                accepted_botiot_license=True,
                restart_stage=None,
            )
            import bitguard_bnn.bootstrap.orchestrator as orchestrator_module

            template = (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "full"
                / "botiot.yaml"
            ).resolve()
            real_digest = orchestrator_module._regular_digest
            template_calls = 0

            def drifting_digest(path: Path) -> tuple[str, int]:
                nonlocal template_calls
                digest, size = real_digest(path)
                if path.resolve() == template:
                    template_calls += 1
                    if template_calls >= 2:
                        return "f" * 64, size
                return digest, size

            with patch.object(
                orchestrator_module, "_regular_digest", side_effect=drifting_digest
            ):
                report = run_bootstrap(options, dependencies=dependencies)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed_stage"], "shard")
            self.assertIn("inputs changed", str(report["error"]))
            state = json.loads(
                (root / "data" / ".bitguard" / "bootstrap-state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("shard", state["stages"])


if __name__ == "__main__":
    unittest.main()
