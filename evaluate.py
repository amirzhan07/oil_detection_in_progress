#!/usr/bin/env python3
"""Evaluate a trained YOLO model on a validation set.

This script wraps the Ultralytics evaluation API to compute metrics
on a given dataset using a specified model.  It is intended for
users who wish to measure the performance of their segmentation
models on the oil spill dataset.  Metrics such as mAP and
segmentation quality will be printed upon completion.
"""

from __future__ import annotations

import argparse
import sys

try:
    from ultralytics import YOLO  # type: ignore[import]
except ImportError as exc:
    YOLO = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on an oil spill dataset")
    parser.add_argument("weights", type=str, help="Path to trained model weights (e.g. best.pt)")
    parser.add_argument("data", type=str, help="Path to dataset YAML file")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if YOLO is None:
        print(
            "ultralytics is not installed.  Install it with 'pip install ultralytics' to use the evaluation script.",
            file=sys.stderr,
        )
        sys.exit(1)
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, batch=args.batch)
    # Print metrics dictionary
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()