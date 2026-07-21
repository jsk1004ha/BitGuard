from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SUPPORTED_PYTHON = ((3, 10), (3, 11), (3, 12))
TORCH_PROFILES = ("cpu", "cu118", "cu124", "cu128")


def validate_python_version(version: tuple[int, int, int]) -> None:
    if version[:2] not in SUPPORTED_PYTHON:
        raise RuntimeError("bootstrap requires Python 3.10 through 3.12")


def venv_python(root: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return root / relative


def build_package_command(environment: Path, forwarded: list[str]) -> list[str]:
    return [
        str(venv_python(environment)),
        "-m",
        "bitguard_bnn",
        "bootstrap",
        *forwarded,
    ]


def validate_virtual_environment(environment: Path) -> None:
    environment_python = venv_python(environment)
    probe = (
        "import json, sys\n"
        "print(json.dumps({"
        "'version': list(sys.version_info[:3]), "
        "'is_venv': sys.prefix != sys.base_prefix"
        "}))\n"
    )
    try:
        result = subprocess.run(
            [str(environment_python), "-c", probe],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"unable to run virtual environment Python at {environment_python}: {exc}"
        ) from exc

    if result.returncode != 0:
        detail = (
            result.stderr.strip() or result.stdout.strip() or "interpreter probe failed"
        )
        raise RuntimeError(
            f"virtual environment Python at {environment_python} is unusable: {detail}"
        )

    try:
        metadata = json.loads(result.stdout)
        raw_version = metadata["version"]
        is_venv = metadata["is_venv"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"virtual environment Python at {environment_python} returned invalid metadata"
        ) from exc

    if (
        not isinstance(raw_version, list)
        or len(raw_version) < 3
        or any(type(component) is not int for component in raw_version[:3])
    ):
        raise RuntimeError(
            f"virtual environment Python at {environment_python} returned an invalid version"
        )

    version = tuple(raw_version[:3])
    if version[:2] not in SUPPORTED_PYTHON:
        raise RuntimeError(
            f"virtual environment at {environment} uses Python {version[0]}.{version[1]}; "
            "bootstrap requires Python 3.10 through 3.12"
        )
    if is_venv is not True:
        raise RuntimeError(
            f"interpreter at {environment_python} is not a virtual environment; "
            "remove .venv and rerun bootstrap"
        )


def _detect_torch_profile() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return "cpu"
    except OSError as exc:
        raise RuntimeError(
            f"CUDA detection failed while running nvidia-smi: {exc}"
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or "nvidia-smi returned a non-zero exit code"
        raise RuntimeError(
            f"CUDA detection failed; refusing to downgrade to CPU: {detail}"
        )

    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if match is None:
        raise RuntimeError(
            "nvidia-smi did not report a CUDA version; refusing to downgrade to CPU"
        )

    cuda_version = (int(match.group(1)), int(match.group(2)))
    if cuda_version >= (12, 8):
        return "cu128"
    if cuda_version >= (12, 4):
        return "cu124"
    if cuda_version >= (11, 8):
        return "cu118"
    raise RuntimeError(
        f"CUDA {cuda_version[0]}.{cuda_version[1]} is below the supported cu118 profile"
    )


def _verify_torch_profile(environment: Path, profile: str) -> None:
    expected_cuda = {
        "cpu": None,
        "cu118": "11.8",
        "cu124": "12.4",
        "cu128": "12.8",
    }[profile]
    verification = (
        "import torch\n"
        f"expected = {expected_cuda!r}\n"
        "actual = torch.version.cuda\n"
        "if actual != expected:\n"
        "    raise SystemExit(f'expected Torch CUDA {expected!r}, found {actual!r}')\n"
        "if expected is not None and not torch.cuda.is_available():\n"
        "    raise SystemExit('the selected CUDA profile cannot access a CUDA device')\n"
        "if expected is not None:\n"
        "    probe = torch.ones(1, device='cuda')\n"
        "    result = probe + 1\n"
        "    torch.cuda.synchronize()\n"
        "    if result.cpu().item() != 2.0:\n"
        "        raise SystemExit('the selected CUDA profile produced an invalid result')\n"
    )
    result = subprocess.run(
        [str(venv_python(environment)), "-c", verification],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or "Torch verification failed"
        )
        raise RuntimeError(
            f"Torch profile {profile} is unusable; refusing to downgrade to CPU: {detail}"
        )


def _parse_arguments(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compute", choices=("auto", *TORCH_PROFILES), default="auto")
    options, forwarded = parser.parse_known_args(argv)
    return options.compute, forwarded


def main(argv: list[str] | None = None) -> int:
    validate_python_version(sys.version_info[:3])
    requested_profile, forwarded = _parse_arguments(
        list(sys.argv[1:] if argv is None else argv)
    )
    profile = (
        _detect_torch_profile() if requested_profile == "auto" else requested_profile
    )

    repository = Path(__file__).resolve().parents[1]
    environment = repository / ".venv"
    environment_python = venv_python(environment)
    if not environment_python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    validate_virtual_environment(environment)

    pip_install = [str(environment_python), "-m", "pip", "install"]
    locks = repository / "requirements" / "locks"
    subprocess.run(
        [*pip_install, "--requirement", str(locks / f"torch-{profile}.txt")],
        check=True,
        cwd=repository,
    )
    _verify_torch_profile(environment, profile)
    subprocess.run(
        [
            *pip_install,
            "--no-build-isolation",
            "--editable",
            str(repository),
            "--no-deps",
        ],
        check=True,
        cwd=repository,
    )
    subprocess.run(
        [*pip_install, "--requirement", str(locks / "full-base.txt")],
        check=True,
        cwd=repository,
    )

    # argparse consumes the installer-level flag. Hand the resolved profile to
    # the package bootstrap exactly once so installation and runtime recovery
    # cannot disagree (including when the user selected ``auto``).
    package_arguments = ["--compute", profile, *forwarded]
    return subprocess.call(
        build_package_command(environment, package_arguments), cwd=repository
    )


if __name__ == "__main__":
    raise SystemExit(main())
