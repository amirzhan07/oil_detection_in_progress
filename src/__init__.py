"""Top‑level package for the oil detection project.

This file exposes the most important classes so that users can simply
import them from the package root instead of drilling into internal
modules.  Each component is documented in its own module – see
``camera.py``, ``detector.py``, ``navigator.py``, ``preprocessing.py``,
``logger.py`` and ``serial_comm.py`` for details.

Example::

    from oil_detection_in_progress import OilDetector, Navigator

    detector = OilDetector(threshold=50)
    navigator = Navigator(config=NavigatorConfig(frame_width=640))

"""

from .camera import VideoCamera, VideoCaptureError, ReadFrameError
from .preprocessing import preprocess_frame
from .detector import OilDetector, OilRegion, DetectionResult
from .navigator import Navigator, NavigatorConfig, DetectionSummary
from .logger import DetectionLogger
from .serial_comm import SerialConfig, SerialCommandSender

__all__ = [
    "VideoCamera",
    "VideoCaptureError",
    "ReadFrameError",
    "preprocess_frame",
    "OilDetector",
    "OilRegion",
    "DetectionResult",
    "Navigator",
    "NavigatorConfig",
    "DetectionSummary",
    "DetectionLogger",
    "SerialConfig",
    "SerialCommandSender",
]