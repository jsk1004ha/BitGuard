from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from .state import STAGE_ORDER
from .types import BootstrapOptions

DATASETS = ("nbaiot", "botiot")
COMPUTE_PROFILES = ("auto", "cpu", "cu118", "cu124", "cu128")
_URLISH_PATH = re.compile(r"https?:[\\/]+", re.IGNORECASE)


def _resolve_local_path(value: str, option: str) -> Path:
    if _URLISH_PATH.search(value):
        raise ValueError(
            f"{option} must be a local filesystem path; URL-looking values are not allowed"
        )
    return Path(value).expanduser().resolve()


def add_bootstrap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--full", action="store_true", help="select the complete bootstrap scope")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("all", *DATASETS),
        help="dataset to prepare; may be repeated",
    )
    parser.add_argument(
        "--botiot-source",
        help="optional local BoT-IoT directory, ZIP, or RAR override",
    )
    parser.add_argument(
        "--accept-botiot-academic-license",
        action="store_true",
        help="confirm review of the official UNSW academic-use terms",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--compute", choices=COMPUTE_PROFILES, default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    system_tools = parser.add_mutually_exclusive_group()
    system_tools.add_argument(
        "--install-system-tools",
        dest="install_system_tools",
        action="store_true",
        help="allow installation of required operating-system tools",
    )
    system_tools.add_argument(
        "--no-install-system-tools",
        dest="install_system_tools",
        action="store_false",
        help="disable the automatic system-tool installation enabled by --full",
    )
    parser.set_defaults(install_system_tools=None)
    parser.add_argument("--restart-stage", choices=STAGE_ORDER)


def _selected_datasets(full: bool, requested: list[str] | None) -> tuple[str, ...]:
    selections = list(requested or ())
    if full and not selections:
        selections = ["all"]
    if not selections:
        raise ValueError("bootstrap requires --full or at least one --dataset")
    expanded = set(DATASETS if "all" in selections else selections)
    return tuple(name for name in DATASETS if name in expanded)


def options_from_namespace(args: argparse.Namespace) -> BootstrapOptions:
    datasets = _selected_datasets(args.full, args.dataset)
    if "botiot" in datasets and not args.accept_botiot_academic_license:
        raise ValueError(
            "BoT-IoT selection requires --accept-botiot-academic-license after reviewing its terms"
        )

    return BootstrapOptions(
        datasets=datasets,
        botiot_source=_resolve_local_path(args.botiot_source, "--botiot-source")
        if args.botiot_source is not None
        else None,
        data_root=_resolve_local_path(args.data_root, "--data-root"),
        runs_root=_resolve_local_path(args.runs_root, "--runs-root"),
        compute=args.compute,
        prepare_only=args.prepare_only,
        install_system_tools=(
            args.full if args.install_system_tools is None else args.install_system_tools
        ),
        accepted_botiot_license=args.accept_botiot_academic_license,
        restart_stage=args.restart_stage,
    )


def parse_bootstrap_options(argv: list[str]) -> BootstrapOptions:
    parser = argparse.ArgumentParser(prog="bitguard bootstrap")
    add_bootstrap_arguments(parser)
    return options_from_namespace(parser.parse_args(argv))


def run_from_namespace(
    args: argparse.Namespace,
    *,
    runner: Callable[..., Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Resolve validated options while preserving original path spellings."""

    options = options_from_namespace(args)
    if runner is None:
        from .orchestrator import run_bootstrap

        runner = run_bootstrap
    result = runner(
        options,
        raw_inputs={
            "botiot_source": args.botiot_source,
            "data_root": args.data_root,
            "runs_root": args.runs_root,
        },
    )
    return dict(result)
