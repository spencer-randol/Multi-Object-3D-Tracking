""" main file to run track pipeline"""
from cli import CLI

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tracking pipeline.")
    p.add_argument(
        "input_folder",
        type=str,
        help="Path to dataset folder containing detections.csv",
    )
    p.add_argument(
        "--gate",
        type=float,
        default=1.0,
        help="Gating threshold (distance units of detections).",
    )
    p.add_argument(
        "--model",
        choices=("p3", "v6", "a9"),
        default="v6",
        help="Motion model used for newly created tracks.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    folder = Path(args.input_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    runner = CLI(gate=args.gate, model=args.model)
    runner.run(str(folder))


if __name__ == "__main__":
    main()
