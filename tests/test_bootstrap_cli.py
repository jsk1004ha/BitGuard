from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path

from bitguard_bnn.bootstrap.cli import (
    add_bootstrap_arguments,
    options_from_namespace,
    parse_bootstrap_options,
)
from bitguard_bnn.bootstrap.registry import load_registry
from bitguard_bnn.bootstrap.state import STAGE_ORDER
from bitguard_bnn.cli import _build_parser, main

ROOT = Path(__file__).resolve().parents[1]


class BootstrapRegistryTest(unittest.TestCase):
    def assert_registry_rejected(
        self, dataset: str, field: str, value: str, message: str
    ) -> None:
        payload = {
            name: spec.to_dict() for name, spec in load_registry().items()
        }
        payload = copy.deepcopy(payload)
        payload[dataset][field] = value
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "datasets.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_registry(path)

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

    def test_registry_rejects_non_https_and_credential_bearing_urls(self):
        cases = (
            ("project_url", "javascript:alert(1)", "HTTPS"),
            ("download_url", "file:///tmp/archive.zip", "HTTPS"),
            ("project_url", "http://archive.ics.uci.edu/dataset/442", "HTTPS"),
            (
                "download_url",
                "https://user:secret@archive.ics.uci.edu/static/public/442/archive.zip",
                "credentials",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                self.assert_registry_rejected("nbaiot", field, value, message)

    def test_registry_rejects_wrong_official_identities_and_botiot_download(self):
        cases = (
            (
                "nbaiot",
                "project_url",
                "https://archive.ics.uci.edu/dataset/441/not-nbaiot",
                "official UCI",
            ),
            (
                "nbaiot",
                "download_url",
                "https://archive.ics.uci.edu/static/public/441/not-nbaiot.zip",
                "official UCI",
            ),
            ("nbaiot", "doi", "10.0000/WRONG", r"10\.24432/C5RC8J"),
            (
                "botiot",
                "project_url",
                "https://research.unsw.edu.au/projects/not-bot-iot",
                "official UNSW",
            ),
            (
                "botiot",
                "download_url",
                "https://research.unsw.edu.au/downloads/bot-iot.zip",
                "must not define download_url",
            ),
        )
        for dataset, field, value, message in cases:
            with self.subTest(dataset=dataset, field=field):
                self.assert_registry_rejected(dataset, field, value, message)


class BootstrapOptionsTest(unittest.TestCase):
    def test_restart_stage_choices_use_canonical_state_order(self):
        parser = argparse.ArgumentParser()
        add_bootstrap_arguments(parser)
        restart_action = next(
            action for action in parser._actions if action.dest == "restart_stage"
        )

        self.assertIs(restart_action.choices, STAGE_ORDER)

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
    @staticmethod
    def _noncanonical_paths() -> tuple[str, str, str]:
        if os.name == "nt":
            return (
                r".\incoming\..\_review_absent_source.zip",
                r".\new-data\.\payload",
                r".\new-runs\..\run-output",
            )
        return (
            "./incoming/../_review_absent_source.zip",
            "./new-data/./payload",
            "./new-runs/../run-output",
        )

    def test_existing_train_parser_remains_available(self):
        args = _build_parser().parse_args(["train", "--config", "config.yaml"])

        self.assertEqual(args.command, "train")
        self.assertEqual(args.config, Path("config.yaml"))

    def test_bootstrap_help_is_available(self):
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                _build_parser().parse_args(["bootstrap", "--help"])

        self.assertEqual(raised.exception.code, 0)

    def test_parser_retains_raw_path_spellings_while_options_resolve(self):
        source, data_root, runs_root = self._noncanonical_paths()
        args = _build_parser().parse_args(
            [
                "bootstrap",
                "--dataset",
                "botiot",
                "--botiot-source",
                source,
                "--accept-botiot-academic-license",
                "--data-root",
                data_root,
                "--runs-root",
                runs_root,
            ]
        )

        self.assertEqual(args.botiot_source, source)
        self.assertEqual(args.data_root, data_root)
        self.assertEqual(args.runs_root, runs_root)
        options = options_from_namespace(args)
        self.assertEqual(options.botiot_source, Path(source).expanduser().resolve())
        self.assertEqual(options.data_root, Path(data_root).expanduser().resolve())
        self.assertEqual(options.runs_root, Path(runs_root).expanduser().resolve())

    def test_bootstrap_report_preserves_raw_path_spellings_byte_for_byte(self):
        source, data_root, runs_root = self._noncanonical_paths()
        output = StringIO()
        with redirect_stdout(output):
            main(
                [
                    "bootstrap",
                    "--dataset",
                    "botiot",
                    "--botiot-source",
                    source,
                    "--accept-botiot-academic-license",
                    "--data-root",
                    data_root,
                    "--runs-root",
                    runs_root,
                ]
            )

        report = json.loads(output.getvalue())
        self.assertEqual(
            report["inputs"],
            {
                "botiot_source": source,
                "data_root": data_root,
                "runs_root": runs_root,
            },
        )

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

    def test_module_entrypoint_reports_bootstrap_semantic_error_without_traceback(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bitguard_bnn",
                "bootstrap",
                "--dataset",
                "botiot",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertIn("botiot-source", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")


class BootstrapPackagingTest(unittest.TestCase):
    def test_built_wheel_contains_and_loads_dataset_registry_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            shutil.copy2(ROOT / "pyproject.toml", project / "pyproject.toml")
            shutil.copy2(ROOT / "README.md", project / "README.md")
            shutil.copytree(
                ROOT / "src",
                project / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
            )
            wheelhouse = root / "wheelhouse"
            environment = os.environ.copy()
            environment.update(
                {
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "PIP_NO_INDEX": "1",
                    "PYTHONPATH": "",
                }
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    str(project),
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheelhouse),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            wheels = list(wheelhouse.glob("*.whl"))
            self.assertEqual(len(wheels), 1)

            with zipfile.ZipFile(wheels[0]) as archive:
                names = archive.namelist()
                self.assertIn("bitguard_bnn/bootstrap/datasets.json", names)
                extracted = root / "extracted"
                archive.extractall(extracted)

            environment["PYTHONPATH"] = str(extracted)
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from bitguard_bnn.bootstrap.registry import load_registry; "
                        "print(load_registry()['nbaiot'].doi)"
                    ),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)
            self.assertEqual(probe.stdout.strip(), "10.24432/C5RC8J")


if __name__ == "__main__":
    unittest.main()
