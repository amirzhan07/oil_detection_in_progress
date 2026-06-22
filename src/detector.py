"""Simple oil slick detector based on intensity thresholding.

The original repository relied on YOLOv8 segmentation models to
identify oil spills.  Training and distributing such models is
outside the scope of this in‑progress version, so the detector
implemented here instead uses basic image processing to locate dark
regions in the frame.  Although less sophisticated, this approach
provides a working demonstration without requiring external model
files.

If you wish to plug in a YOLO model, modify the :class:`OilDetector`
implementation accordingly.  Typically you would load a trained
``ultralytics.YOLO`` model and call ``model.predict()`` on the
preprocessed frames to obtain segmentation masks.  From there you
could construct :class:`OilRegion` objects as shown below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class OilRegion:
    """Representation of an individual oil slick region detected in an image."""

    bbox: Tuple[int, int, int, int]
    """Bounding box of the region given as ``(x, y, w, h)`` in pixels."""

    mask: np.ndarray
    """Binary mask of the region with the same height and width as the frame."""

    area: float
    """Area of the region in pixels."""

    centroid: Tuple[int, int]
    """Centroid of the region given as ``(x, y)`` in pixels."""


@dataclass
class DetectionResult:
    """Holds the results of a detection run."""
    regions: List[OilRegion]
    frame: np.ndarray


class OilDetector:
    """Detect dark regions in an image that may correspond to oil spills.

    Parameters
    ----------
    threshold : int, optional
        Threshold value (0–255) used when binarising the image.  Pixels
        darker than this value are considered part of a potential oil
        slick.
    min_area : int, optional
        Minimum area (in pixels) for a region to be considered valid.
        Regions smaller than this value are discarded.
    """

    def __init__(self, threshold: int = 30, min_area: int = 500) -> None:
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single frame.

        The frame is converted to grayscale and thresholded.  Contours
        are then extracted from the binary mask.  Each sufficiently
        large contour becomes an :class:`OilRegion`.

        Parameters
        ----------
        frame : ndarray
            Input BGR image.

        Returns
        -------
        DetectionResult
            Object containing a list of regions and the original frame.
        """
        # Convert to grayscale and threshold.  Invert the mask so that
        # dark pixels (oil) become white (foreground).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        # Find contours of the binary mask.  OpenCV returns a list of
        # contour arrays and a hierarchy (unused here).
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[OilRegion] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue  # Skip small noise blobs
            x, y, w, h = cv2.boundingRect(cnt)
            # Compute centroid using image moments
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            # Create a mask for this contour
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            regions.append(OilRegion(bbox=(x, y, w, h), mask=mask, area=area, centroid=(cx, cy)))
        return DetectionResult(regions=regions, frame=frame)

    def draw_overlay(self, frame: np.ndarray, detection_result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and centroids of detected regions on a copy of the frame.

        Parameters
        ----------
        frame : ndarray
            The frame on which to draw.  It is not modified in place.
        detection_result : DetectionResult
            Result from :meth:`detect` containing the regions to draw.

        Returns
        -------
        ndarray
            Annotated image.
        """
        output = frame.copy()
        for region in detection_result.regions:
            x, y, w, h = region.bbox
            # Draw bounding box (green)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            # Draw centroid (red)
            cx, cy = region.centroid
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), thickness=-1)
        return output