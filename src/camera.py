"""Simple wrapper around OpenCV's video capture.

The original repository defined a minimal camera helper that raised
custom exceptions when failing to open or read from the camera.  This
module provides similar functionality while encapsulating resource
management and adding context manager support.  It can be used both
for live detection and for reading from video files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2


class VideoCaptureError(Exception):
    """Raised when the camera cannot be opened."""


class ReadFrameError(Exception):
    """Raised when a frame cannot be read from the camera."""


class VideoCamera:
    """Context manager for reading frames from a video device or file.

    Parameters
    ----------
    source : int | str
        Identifier for the video source.  An integer opens a live
        camera (e.g. ``0`` for the default webcam) and a string opens
        a video file.
    """

    def __init__(self, source: Union[int, str] = 0) -> None:
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.open(source)

    def open(self, source: Union[int, str]) -> None:
        """Open the video source.

        Raises
        ------
        VideoCaptureError
            If the source cannot be opened.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise VideoCaptureError(f"Failed to open video source: {source}")

    def read(self) -> 'tuple[bool, cv2.typing.MatLike]':
        """Read a frame from the video source.

        Returns
        -------
        success, frame : bool, ndarray
            The return value from :func:`cv2.VideoCapture.read`.

        Raises
        ------
        ReadFrameError
            If no frame can be read (e.g. end of file).
        """
        if self.cap is None:
            raise VideoCaptureError("Video source is not opened")
        success, frame = self.cap.read()
        if not success:
            raise ReadFrameError("Failed to read frame from video source")
        return success, frame

    def release(self) -> None:
        """Release the underlying video capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> 'VideoCamera':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
