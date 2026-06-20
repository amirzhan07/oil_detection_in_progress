# Oil Detection Camera (In Progress)
<img width="585" height="321" alt="Screenshot 2026-06-20 at 11 53 49 PM" src="https://github.com/user-attachments/assets/02eec75c-6baf-40bd-bee6-6df68a8e37f7" />

Oil Detection Camera is a work‑in‑progress project that aims to detect and
track oil slicks on the surface of water.  The original version relied on a
YOLOv8 segmentation model.  For the sake of a working prototype this
in‑progress version implements a simpler detector based on intensity
thresholding and contour analysis.  It still provides modules for camera
capture, navigation logic, preprocessing, logging and optional serial
communication.  The code has been reorganised into a ``src/`` package to fix
broken imports.  This prototype is not production ready and is intended as
a foundation for future work with quick Fusion 360 design.

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
