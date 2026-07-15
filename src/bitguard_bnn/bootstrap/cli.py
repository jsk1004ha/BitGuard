from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .state import STAGE_ORDER
from .types import BootstrapOptions

DATASETS = ("nbaiot", "botiot")
COMPUTE_PROFILES = ("auto", "cpu", "cu118", "cu124")


def add_bootstrap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--full", action="store_true", help="select the complete bootstrap scope")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("all", *DATASETS),
        help="dataset to prepare; may be repeated",
    )
    parser.add_argument("--botiot-source")
    parser.add_argument("--accept-botiot-academic-license", action="store_true")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--compute", choices=COMPUTE_PROFILES, default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--install-system-tools", action="store_true")
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
    if "botiot" in datasets and args.botiot_source is None:
        raise ValueError("BoT-IoT selection requires --botiot-source from the official project")
    if "botiot" in datasets and not args.accept_botiot_academic_license:
        raise ValueError(
            "BoT-IoT selection requires --accept-botiot-academic-license after reviewing its terms"
        )

    return BootstrapOptions(
        datasets=datasets,
        botiot_source=Path(args.botiot_source).expanduser().resolve()
        if args.botiot_source is not None
        else None,
        data_root=Path(args.data_root).expanduser().resolve(),
        runs_root=Path(args.runs_root).expanduser().resolve(),
        compute=args.compute,
        prepare_only=args.prepare_only,
        install_system_tools=args.install_system_tools,
        accepted_botiot_license=args.accept_botiot_academic_license,
        restart_stage=args.restart_stage,
    )


def parse_bootstrap_options(argv: list[str]) -> BootstrapOptions:
    parser = argparse.ArgumentParser(prog="bitguard bootstrap")
    add_bootstrap_arguments(parser)
    return options_from_namespace(parser.parse_args(argv))


def validation_report(args: argparse.Namespace, options: BootstrapOptions) -> dict[str, Any]:
    return {
        "status": "validated",
        "scope": "bootstrap-options",
        "message": "Bootstrap stages are not implemented yet; no data was acquired or trained.",
        "inputs": {
            "botiot_source": str(args.botiot_source)
            if args.botiot_source is not None
            else None,
            "data_root": str(args.data_root),
            "runs_root": str(args.runs_root),
        },
        "options": options.to_dict(),
    }
