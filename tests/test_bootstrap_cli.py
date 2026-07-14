from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path

from bitguard_bnn.bootstrap.cli import parse_bootstrap_options
from bitguard_bnn.bootstrap.registry import load_registry
from bitguard_bnn.cli import _build_parser, main


class BootstrapRegistryTest(unittest.TestCase):
    def test_registry_contains_only_official_sources(self):
        registry = load_registry()

        self.assertEqual(registry["nbaiot"].doi, "10.24432/C5RC8J")
        self.assertIn("archive.ics.uci.edu", registry["nbaiot"].download_url)
        self.assertIsNone(registry["botiot"].download_url)
        self.assertIn("research.unsw.edu.au", registry["botiot"].project_url)

    def test_registry_is_typed_immutable_and_json_safe(self):
        registry = load_registry()
        nbaiot = registry["nbaiot"]

        with self.assertRaises(FrozenInstanceError):
            nbaiot.doi = "replacement"  # type: ignore[misc]
        serialized = nbaiot.to_dict()
        self.assertEqual(serialized["expected_patterns"], ["**/*.csv"])
        self.assertEqual(serialized["required_columns"], [])
        self.assertEqual(json.loads(json.dumps(serialized)), serialized)

    def test_registry_has_exact_supported_keys_and_no_botiot_automation_secrets(self):
        registry = load_registry()

        self.assertEqual(tuple(registry), ("nbaiot", "botiot"))
        self.assertIsNone(registry["botiot"].download_url)
        self.assertEqual(
            registry["botiot"].required_columns,
            ("category", "subcategory", "saddr", "stime"),
        )
        serialized = json.dumps(
            {name: spec.to_dict() for name, spec in registry.items()}
        ).lower()
        self.assertNotIn("password", serialized)
        self.assertNotIn("credential", serialized)
        self.assertNotIn("sha256", serialized)


class BootstrapOptionsTest(unittest.TestCase):
    def test_full_all_requires_botiot_source_and_license(self):
        with self.assertRaisesRegex(ValueError, "botiot-source"):
            parse_bootstrap_options(["--full", "--dataset", "all"])

    def test_full_all_reports_missing_license_after_source(self):
        with self.assertRaisesRegex(ValueError, "accept-botiot-academic-license"):
            parse_bootstrap_options(
                ["--full", "--dataset", "all", "--botiot-source", "official.zip"]
            )

    def test_nbaiot_only_does_not_require_botiot_source_or_license(self):
        options = parse_bootstrap_options(["--dataset", "nbaiot"])

        self.assertEqual(options.datasets, ("nbaiot",))
        self.assertIsNone(options.botiot_source)
        self.assertFalse(options.accepted_botiot_license)

    def test_full_without_explicit_dataset_selects_all(self):
        options = parse_bootstrap_options(
            [
                "--full",
                "--botiot-source",
                "official.zip",
                "--accept-botiot-academic-license",
            ]
        )

        self.assertEqual(options.datasets, ("nbaiot", "botiot"))

    def test_duplicate_dataset_selections_normalize_in_registry_order(self):
        options = parse_bootstrap_options(
            ["--dataset", "nbaiot", "--dataset", "nbaiot"]
        )

        self.assertEqual(options.datasets, ("nbaiot",))

    def test_paths_resolve_and_parsing_does_not_create_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            original = Path.cwd()
            os.chdir(base)
            try:
                options = parse_bootstrap_options(
                    [
                        "--dataset",
                        "botiot",
                        "--botiot-source",
                        "incoming/official.zip",
                        "--accept-botiot-academic-license",
                        "--data-root",
                        "new-data",
                        "--runs-root",
                        "new-runs",
                    ]
                )
            finally:
                os.chdir(original)

            self.assertEqual(options.botiot_source, (base / "incoming/official.zip").resolve())
            self.assertEqual(options.data_root, (base / "new-data").resolve())
            self.assertEqual(options.runs_root, (base / "new-runs").resolve())
            self.assertFalse((base / "incoming").exists())
            self.assertFalse((base / "new-data").exists())
            self.assertFalse((base / "new-runs").exists())
            self.assertEqual(
                json.loads(json.dumps(options.to_dict()))["datasets"], ["botiot"]
            )
            with self.assertRaises(FrozenInstanceError):
                options.compute = "cpu"  # type: ignore[misc]


class BootstrapDispatchTest(unittest.TestCase):
    def test_existing_train_parser_remains_available(self):
        args = _build_parser().parse_args(["train", "--config", "config.yaml"])

        self.assertEqual(args.command, "train")
        self.assertEqual(args.config, Path("config.yaml"))

    def test_bootstrap_help_is_available(self):
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                _build_parser().parse_args(["bootstrap", "--help"])

        self.assertEqual(raised.exception.code, 0)

    def test_valid_bootstrap_dispatch_reports_validation_without_mutation(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "bootstrap",
                    "--dataset",
                    "nbaiot",
                    "--compute",
                    "cpu",
                    "--prepare-only",
                    "--install-system-tools",
                    "--restart-stage",
                    "inspect",
                ]
            )

        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "validated")
        self.assertEqual(report["scope"], "bootstrap-options")
        self.assertEqual(report["options"]["datasets"], ["nbaiot"])
        self.assertEqual(report["options"]["restart_stage"], "inspect")


if __name__ == "__main__":
    unittest.main()
