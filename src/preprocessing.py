"""Image preprocessing utilities.

This module contains simple functions for preparing frames before
running oil detection.  They include glare reduction, adaptive
histogram equalisation (CLAHE), and colour balancing based on the
Gray–World assumption.  The functions are designed to be
lightweight and have no external dependencies beyond NumPy and
OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np


def reduce_glare(image: np.ndarray) -> np.ndarray:
    """Reduce specular highlights in an image using median filtering.

    Parameters
    ----------
    image : ndarray
        Input BGR image.

    Returns
    -------
    ndarray
        Image with reduced glare.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Apply a median blur to the value channel to smooth bright spots
    v = cv2.medianBlur(v, 5)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to improve local contrast.

    Parameters
    ----------
    image : ndarray
        Input BGR image.
    clip_limit : float, optional
        Threshold for contrast limiting.
    tile_grid_size : tuple[int, int], optional
        Size of the grid for the histogram equalisation.

    Returns
    -------
    ndarray
        Image after CLAHE is applied.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gray_world_normalization(image: np.ndarray) -> np.ndarray:
    """Perform simple Gray-World colour balancing.

    This normalises the colour channels so that their averages become
    equal.  It is a common technique to remove colour casts.

    Parameters
    ----------
    image : ndarray
        Input BGR image.

    Returns
    -------
    ndarray
        Colour balanced image.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    avg = (avg_b + avg_g + avg_r) / 3.0
    # Avoid division by zero
    b = b * (avg / max(avg_b, 1e-6))
    g = g * (avg / max(avg_g, 1e-6))
    r = r * (avg / max(avg_r, 1e-6))
    # Clip values to 0–255 and convert back to uint8
    balanced = cv2.merge([b, g, r])
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Run a sequence of preprocessing steps on a frame.

    The function reduces glare, applies CLAHE and performs
    Gray-World normalisation.  It returns the processed image
    without modifying the input.

    Parameters
    ----------
    frame : ndarray
        Input BGR image.

    Returns
    -------
    ndarray
        Preprocessed image ready for detection.
    """
    # Operate on a copy to avoid modifying the original frame
    result = frame.copy()
    result = reduce_glare(result)
    result = apply_clahe(result)
    result = gray_world_normalization(result)
    return result
