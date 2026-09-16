"""Shared wafer-map preprocessing used by training and inference.

The source notebook works with categorical wafer maps whose values are:

* ``0`` - background
* ``1`` - valid die/wafer area
* ``2`` - defect

Keeping this module independent from TensorFlow makes it usable by both the
offline exporter and the lightweight API validation layer.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

TARGET_SIZE = (64, 64)
ALLOWED_VALUES = (0, 1, 2)
NORMALIZATION_DIVISOR = 2.0
MAX_INPUT_SIDE = 512


class WaferMapValidationError(ValueError):
    """Raised when an input cannot be interpreted as a wafer map."""


def _as_numeric_array(wafer_map: Any) -> np.ndarray:
    try:
        array = np.asarray(wafer_map)
    except Exception as exc:  # pragma: no cover - defensive for unusual objects
        raise WaferMapValidationError("wafer_map must be a 2D numeric matrix") from exc

    if array.ndim != 2 or array.size == 0:
        raise WaferMapValidationError("wafer_map must be a non-empty 2D matrix")
    if array.shape[0] > MAX_INPUT_SIDE or array.shape[1] > MAX_INPUT_SIDE:
        raise WaferMapValidationError(
            f"wafer_map dimensions cannot exceed {MAX_INPUT_SIDE}x{MAX_INPUT_SIDE}"
        )

    try:
        numeric = array.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise WaferMapValidationError("wafer_map values must be numeric") from exc

    if not np.isfinite(numeric).all():
        raise WaferMapValidationError("wafer_map cannot contain NaN or infinite values")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise WaferMapValidationError("wafer_map values must be integers 0, 1, or 2")

    integer = numeric.astype(np.int16)
    if not np.isin(integer, ALLOWED_VALUES).all():
        raise WaferMapValidationError("wafer_map values must be limited to 0, 1, and 2")
    return integer


def validate_wafer_map(wafer_map: Any) -> np.ndarray:
    """Validate and return a compact integer wafer-map array."""

    return _as_numeric_array(wafer_map)


def resize_wafer_map(
    wafer_map: Any,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Resize a categorical map without introducing new category values."""

    source = _as_numeric_array(wafer_map).astype(np.uint8)
    resized = cv2.resize(source, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8, copy=False)


def normalize_wafer_map(
    wafer_map: Any,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Return a CNN-ready tensor with shape ``(height, width, 1)``."""

    resized = resize_wafer_map(wafer_map, target_size=target_size)
    normalized = resized.astype(np.float32) / NORMALIZATION_DIVISOR
    return np.expand_dims(normalized, axis=-1)


def calculate_defect_ratio(wafer_map: Any) -> float:
    """Calculate defect pixels divided by non-background pixels."""

    array = _as_numeric_array(wafer_map)
    valid_pixels = np.sum(array > 0)
    defect_pixels = np.sum(array == 2)
    if valid_pixels == 0:
        return 0.0
    return float(defect_pixels / valid_pixels)


def stack_cnn_inputs(
    wafer_maps: Sequence[Any],
    target_size: tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Preprocess a sequence of maps into a batch for Keras."""

    if not wafer_maps:
        raise WaferMapValidationError("at least one wafer map is required")
    return np.stack(
        [normalize_wafer_map(wafer_map, target_size=target_size) for wafer_map in wafer_maps]
    ).astype(np.float32)

