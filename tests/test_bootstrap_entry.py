import unittest
from pathlib import Path

from scripts.bootstrap import build_package_command, validate_python_version


class BootstrapEntryTest(unittest.TestCase):
    def test_build_command_forwards_full_source_and_license(self):
        command = build_package_command(
            Path(".venv"),
            ["--full", "--botiot-source", "input.zip", "--accept-botiot-academic-license"],
        )
        self.assertEqual(
            command[-5:],
            [
                "bootstrap",
                "--full",
                "--botiot-source",
                "input.zip",
                "--accept-botiot-academic-license",
            ],
        )

    def test_python_outside_supported_range_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Python 3.10 through 3.12"):
            validate_python_version((3, 13, 0))


if __name__ == "__main__":
    unittest.main()
