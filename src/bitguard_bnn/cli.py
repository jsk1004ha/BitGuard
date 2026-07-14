from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bitguard", description="BitGuard-BNN research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("make-demo", help="generate a labelled metadata-feature demo CSV")
    demo.add_argument("--output", type=Path, default=Path("data/demo.csv"))
    demo.add_argument("--rows", type=int, default=12_000)
    demo.add_argument("--seed", type=int, default=2309)
    stream = subparsers.add_parser(
        "stream-features", help="convert ordered payload-free packet metadata to the shared feature CSV"
    )
    stream.add_argument("--input", type=Path, required=True)
    stream.add_argument("--output", type=Path, required=True)
    stream.add_argument("--window-seconds", type=float, default=60.0)
    stream.add_argument("--max-events-per-device", type=int, default=2048)
    stream.add_argument("--max-devices", type=int, default=4096)
    stream.add_argument("--chunk-size", type=int, default=100000)
    train = subparsers.add_parser("train", help="run preprocessing, training, and evaluation")
    train.add_argument("--config", type=Path, required=True)
    export = subparsers.add_parser("export", help="export a trained BNN for packed edge inference")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    replay = subparsers.add_parser("replay", help="rerun temporal/action simulation from predictions")
    replay.add_argument("--run", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "make-demo":
        from .demo import generate_demo

        frame = generate_demo(args.output.resolve(), args.rows, args.seed)
        print(json.dumps({"path": str(args.output.resolve()), "rows": len(frame)}, ensure_ascii=False))
        return
    if args.command == "stream-features":
        from .streaming import process_metadata_csv

        result = process_metadata_csv(
            args.input.resolve(),
            args.output.resolve(),
            window_seconds=args.window_seconds,
            max_events_per_device=args.max_events_per_device,
            max_devices=args.max_devices,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.command == "train":
        from .trainer import run_training

        run_dir = run_training(args.config)
        print(str(run_dir.resolve()))
        return
    if args.command == "export":
        from .export import export_run

        result = export_run(args.run.resolve(), args.output.resolve())
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.command == "replay":
        import pandas as pd

        from .config import load_config, save_json
        from .state import replay_predictions

        config = load_config(args.run / "resolved_config.yaml")
        predictions = pd.read_csv(args.run / "predictions.csv")
        temporal, metrics = replay_predictions(predictions, config)
        temporal.to_csv(args.run / "temporal_predictions.csv", index=False)
        save_json(metrics, args.run / "operational_metrics.json")
        print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
