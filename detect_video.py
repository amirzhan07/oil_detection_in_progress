#!/usr/bin/env python3
"""Run oil detection on a video file.

This script processes an input video, frame by frame, applying
preprocessing and oil detection, deciding navigation commands, and
writing an annotated output video.  It also logs detection summaries
to a CSV file and optionally sends commands over a serial port.  The
output video will have the same dimensions and frame rate as the
input.
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
    parser = argparse.ArgumentParser(description="Run oil detection on a video file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to write annotated video (optional)")
    parser.add_argument("--serial", type=str, default=None, help="Serial port to send commands to (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dict = load_config(args.config)
    # Detector parameters
    det_conf = config_dict.get("detector", {})
    threshold = int(det_conf.get("threshold", 30))
    min_area = int(det_conf.get("min_area", 500))
    detector = OilDetector(threshold=threshold, min_area=min_area)
    # Logger
    log_conf = config_dict.get("logging", {})
    csv_path = log_conf.get("csv_path", "logs/detections.csv")
    screenshot_dir = log_conf.get("screenshot_dir", "logs/screenshots")
    logger = DetectionLogger(csv_path=csv_path, screenshot_dir=screenshot_dir)
    # Navigator
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
    # Serial sender (optional)
    serial_sender: Optional[SerialCommandSender] = None
    if args.serial:
        serial_sender = SerialCommandSender(SerialConfig(port=args.serial))
    # Open input video
    input_path = args.input
    try:
        with VideoCamera(input_path) as camera:
            # Determine frame properties
            width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if camera.cap else 0
            height = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if camera.cap else 0
            fps = camera.cap.get(cv2.CAP_PROP_FPS) if camera.cap else 30.0
            # Update navigator width if not set
            if nav_config.frame_width <= 0:
                nav_config.frame_width = width or 1
            # Prepare video writer if output specified
            writer: Optional[cv2.VideoWriter] = None
            if args.output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
            print(f"Processing video: {input_path}")
            while True:
                try:
                    _, frame = camera.read()
                except ReadFrameError:
                    break
                processed = preprocess_frame(frame)
                detection_result = detector.detect(processed)
                summary = navigator.summarise(detection_result)
                command = navigator.decide(summary)
                overlay = detector.draw_overlay(frame, detection_result)
                # Write frame to output video
                if writer is not None:
                    writer.write(overlay)
                # Send command over serial
                if serial_sender is not None:
                    serial_sender.send(command)
                # Log
                logger.log(command, summary, overlay)
                # Display the frame
                cv2.imshow("Oil Detection Video", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            # Clean up
            if writer is not None:
                writer.release()
    except VideoCaptureError as exc:
        print(f"Failed to open video file: {exc}")
    finally:
        if serial_sender is not None:
            serial_sender.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
