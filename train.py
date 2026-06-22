#!/usr/bin/env python3
"""Train a YOLO model for oil spill segmentation.

This script is a thin wrapper around the `ultralytics` training API.
It expects a dataset YAML file describing the training and validation
data and optionally accepts an initial set of weights.  You can pass
any extra arguments supported by the underlying Ultralytics CLI via
``--yolo-args``.

Note: In the in‑progress version of this project the detection
logic does not rely on YOLO.  This training script is provided for
completeness and to support future work when integrating a custom
model.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import List

try:
    from ultralytics import YOLO  # type: ignore[import]
except ImportError as exc:
    YOLO = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model on an oil spill dataset")
    parser.add_argument("data", type=str, help="Path to dataset YAML file")
    parser.add_argument("--weights", type=str, default="yolov8n-seg.pt", help="Initial weights (optional)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--yolo-args", type=str, default="", help="Additional arguments to pass to YOLO as a string")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if YOLO is None:
        print(
            "ultralytics is not installed.  Install it with 'pip install ultralytics' to use the training script.",
            file=sys.stderr,
        )
        sys.exit(1)
    # Instantiate model
    model = YOLO(args.weights)
    # Parse additional arguments to pass to YOLO.train()
    extra_args: List[str] = []
    if args.yolo_args:
        extra_args = shlex.split(args.yolo_args)
    # Run training
    model.train(data=args.data, epochs=args.epochs, *extra_args)


if __name__ == "__main__":
    main()