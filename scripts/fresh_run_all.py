from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blueberry_multitask.config import load_config
from blueberry_multitask.engine import run_all
from blueberry_multitask.summarize import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fresh multitask benchmark methods.")
    parser.add_argument("--config", default="configs/fresh_benchmark.yaml")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    failures = run_all(
        config=config,
        tasks=args.tasks,
        methods=args.methods,
        seed=args.seed,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        limit=args.limit,
        pretrained=args.pretrained,
        stop_on_error=args.stop_on_error,
    )
    summarize(config)
    if failures:
        print("\nFailures:")
        for key, error in failures.items():
            print(f"- {key}: {error}")


if __name__ == "__main__":
    main()

