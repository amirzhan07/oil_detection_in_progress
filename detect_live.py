#!/usr/bin/env python3
"""Run oil detection on a live camera feed.

This script opens a video capture device, preprocesses each frame,
runs oil detection on it and decides a navigation command.  The
results are displayed in a window with overlays and logged to a CSV
file.  When provided, navigation commands can also be sent over a
serial connection.  Press ``q`` to quit.

Usage::

    python detect_live.py --config config.yaml --device 0 --serial /dev/ttyUSB0

The configuration file controls threshold values, navigator
thresholds and logging paths.  Command line arguments override
values defined in the YAML file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2

from src.camera import VideoCamera, VideoCaptureError, ReadFrameError
from src.detector import OilDetector
from src.preprocessing import preprocess_frame
from src.navigator import Navigator, NavigatorConfig
from src.logger import DetectionLogger
from src.serial_comm import SerialConfig, SerialCommandSender
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live oil detection")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Video device index or file path",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Serial port to send commands to (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load configuration from YAML
    config_dict = load_config(args.config)
    # Extract detector parameters
    det_conf = config_dict.get("detector", {})
    threshold = int(det_conf.get("threshold", 30))
    min_area = int(det_conf.get("min_area", 500))
    detector = OilDetector(threshold=threshold, min_area=min_area)
    # Create camera
    device = args.device
    # Parse device as int if possible
    try:
        device_index: int | str = int(device)
    except ValueError:
        device_index = device
    # Create logger
    log_conf = config_dict.get("logging", {})
    csv_path = log_conf.get("csv_path", "logs/detections.csv")
    screenshot_dir = log_conf.get("screenshot_dir", "logs/screenshots")
    logger = DetectionLogger(csv_path=csv_path, screenshot_dir=screenshot_dir)
    # Navigator configuration
    nav_conf = config_dict.get("navigator", {})
    frame_width = int(nav_conf.get("frame_width", 0))
    nav_config = NavigatorConfig(
        frame_width=frame_width,
        left_threshold=float(nav_conf.get("left_threshold", 0.4)),
        right_threshold=float(nav_conf.get("right_threshold", 0.6)),
        left_command=str(nav_conf.get("left_command", "left")),
        right_command=str(nav_conf.get("right_command", "right")),
        forward_command=str(nav_conf.get("forward_command", "forward")),
        stop_command=str(nav_conf.get("stop_command", "stop")),
        area_threshold=float(nav_conf.get("area_threshold", 5000)),
    )
    navigator = Navigator(nav_config)
    # Optionally open serial port
    serial_sender: Optional[SerialCommandSender] = None
    if args.serial:
        serial_sender = SerialCommandSender(SerialConfig(port=args.serial))
    # Open video source
    try:
        with VideoCamera(device_index) as camera:
            # If frame width wasn't set in config, read it from the camera
            if nav_config.frame_width <= 0:
                # Query the frame width property if available
                width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if camera.cap else 0
                nav_config.frame_width = width or 1  # avoid divide by zero
            print("Live detection started... Press 'q' to exit.")
            while True:
                try:
                    _, frame = camera.read()
                except ReadFrameError:
                    print("End of video feed or unable to read frame.")
                    break
                # Preprocess frame
                processed = preprocess_frame(frame)
                # Detect oil regions
                detection_result = detector.detect(processed)
                # Summarise and decide navigation
                summary = navigator.summarise(detection_result)
                command = navigator.decide(summary)
                # Draw overlay and display
                overlay = detector.draw_overlay(frame, detection_result)
                cv2.imshow("Oil Detection", overlay)
                # Send command over serial if configured
                if serial_sender is not None:
                    serial_sender.send(command)
                # Log the result
                logger.log(command, summary, overlay)
                # Exit on 'q'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    except VideoCaptureError as exc:
        print(f"Failed to open video source: {exc}")
    finally:
        if serial_sender is not None:
            serial_sender.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()