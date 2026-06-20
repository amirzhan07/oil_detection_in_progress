"""Utilities for logging detection results to CSV and saving screenshots.

This module defines the :class:`DetectionLogger` class which can be
used to keep a record of detection summaries and corresponding
screenshots.  Each call to :meth:`log` appends a row to a CSV file
and optionally writes an image to a screenshot directory.  The CSV
file includes a timestamp in ISO 8601 format, the chosen command,
basic statistics about the detection and the path to the saved
image.

The logger will create its output directories if they do not already
exist.
"""

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .navigator import DetectionSummary


class DetectionLogger:
    """Record detection summaries and save annotated screenshots."""

    def __init__(self, csv_path: str | Path, screenshot_dir: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.screenshot_dir = Path(screenshot_dir)
        # Ensure parent directories exist
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        # Initialise the CSV file with headers if it doesn't exist
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "command",
                    "num_regions",
                    "max_area",
                    "centroid_x",
                    "centroid_y",
                    "screenshot_path",
                ])

    def log(self, command: str, summary: DetectionSummary, frame: Optional[np.ndarray] = None) -> None:
        """Append a detection summary and optional screenshot to the log.

        Parameters
        ----------
        command : str
            The navigation command issued for this frame.
        summary : DetectionSummary
            Summary statistics about the detection.
        frame : ndarray, optional
            Annotated frame to save as a screenshot.  If omitted
            ``screenshot_path`` in the CSV will be empty.
        """
        timestamp = _dt.datetime.now().isoformat(timespec="seconds")
        screenshot_path = ""
        if frame is not None:
            # Save image using a timestamped filename.  Replace colon in ISO
            # format with hyphens to make the filename portable.
            filename = timestamp.replace(":", "-") + ".jpg"
            path = self.screenshot_dir / filename
            cv2.imwrite(str(path), frame)
            screenshot_path = str(path)
        # Write a row to the CSV
        row = [
            timestamp,
            command,
            summary.num_regions,
            summary.max_area,
            summary.centroid_x,
            summary.centroid_y,
            screenshot_path,
        ]
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
