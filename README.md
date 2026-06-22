# Oil Detection Camera (In Progress)

Oil Detection Camera is a work‑in‑progress project that aims to detect and
track oil slicks on the surface of water.  The original version relied on a
YOLOv8 segmentation model.  For the sake of a working prototype this
in‑progress version implements a simpler detector based on intensity
thresholding and contour analysis.  It still provides modules for camera
capture, navigation logic, preprocessing, logging and optional serial
communication.  The code has been reorganised into a ``src/`` package to fix
broken imports.  This prototype is not production ready and is intended as
a foundation for future work.

## Project structure

The code has been reorganised into a proper package.  All library modules live
under the `src` package and top‑level scripts import from this package:

```
oil_detection_in_progress/
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── detector.py
│   ├── logger.py
│   ├── navigator.py
│   ├── preprocessing.py
│   ├── serial_comm.py
│   └── utils.py
├── config.yaml
├── detect_live.py
├── detect_video.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

Scripts such as `detect_live.py` and `detect_video.py` can be executed from
the repository root.  They import from the `src` package (e.g.
`from src.detector import OilDetector`) which works because `src` contains an
`__init__.py` file making it a package.

## Installation

1. Make sure you have Python 3.10 or newer installed.  Creating a virtual
   environment is recommended:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install the dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. (Optional) If you intend to train or evaluate a YOLO model you will
   need the Ultralytics package and a set of weights.  Follow the
   instructions under *Training and evaluation* below.  The simple
   threshold‑based detector does not require any external model files.

## Usage
### Live camera detection

To run the detector on a webcam or other live video source use
``detect_live.py``.  The ``--device`` argument accepts either an integer
camera index (``0`` is usually the default webcam) or a file path to a
video stream:

```sh
python detect_live.py --device 0 --config config.yaml
```

To transmit the chosen navigation commands to an external device over a
serial link, specify the serial port.  For example, on Linux you might
use:

```sh
python detect_live.py --device 0 --config config.yaml --serial /dev/ttyUSB0
```

While running, a window will display the camera feed with annotated
bounding boxes and centroids.  Press ``q`` to quit.

### Video file detection

``detect_video.py`` processes a video file frame by frame.  It can also
write an annotated output file.  Only the ``--input`` argument is
required:

```sh
python detect_video.py --input path/to/input.mp4 --config config.yaml --output path/to/out.mp4
```

Omit ``--output`` if you do not wish to save the annotated video.  The
script logs detection summaries and screenshots to the locations
specified in ``config.yaml``.

### Training and evaluation (optional)

The ``train.py`` and ``evaluate.py`` scripts are thin wrappers around
Ultralytics' YOLO API.  They are provided for future work and require
the ``ultralytics`` package to be installed.  Use them only if you plan
to train or evaluate a YOLO model on your own dataset.

To train a model:

```sh
pip install ultralytics
python train.py data.yaml --weights yolov8n-seg.pt --epochs 50
```

To evaluate a model:

```sh
python evaluate.py best.pt data.yaml
```

## Configuration

The project reads its settings from ``config.yaml``.  Thresholds for
detection, navigation behaviour and logging paths can all be adjusted
there.  Inline comments explain the purpose of each field.  Feel free
to tune the values to suit your environment and camera setup.

## Status

This repository is labelled **“in progress”** because it is still under
active development.  The code has been cleaned up and reorganised, but there
may still be missing functionality (e.g. trained models, datasets).  Please
open issues or pull requests if you encounter problems or would like to
contribute improvements.

## License

This project is released under the MIT License.
