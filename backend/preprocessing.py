"""Input validation and preprocessing for categorical wafer maps.

The notebook represents a wafer map as a 2-D categorical matrix:

* ``0`` – background
* ``1`` – valid wafer/die pixels
* ``2`` – defect pixels

Nearest-neighbour resizing is important here: interpolation must never create
new category values such as ``0.5`` or ``1.7``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import Settings


ALLOWED_VALUES: tuple[int, ...] = (0, 1, 2)


class WaferMapValidationError(ValueError):
    """Raised when an input wafer map does not match the API contract."""


def _is_row(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_integer(value: Any) -> bool:
    # bool is a subclass of int, but True/False are not valid categorical
    # wafer-map values for this API.
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def validate_wafer_map(
    wafer_map: Any,
    *,
    max_input_size: int = 512,
    allowed_values: Sequence[int] = ALLOWED_VALUES,
) -> tuple[int, int]:
    """Validate a rectangular, non-empty categorical matrix.

    Returns ``(height, width)`` so callers do not need to materialize another
    array just to inspect the input shape.
    """

    if isinstance(wafer_map, np.ndarray):
        if wafer_map.ndim != 2:
            raise WaferMapValidationError("wafer_map must be a 2-D matrix")
        rows = wafer_map.tolist()
    elif isinstance(wafer_map, Sequence) and not isinstance(
        wafer_map, (str, bytes, bytearray)
    ):
        rows = list(wafer_map)
    else:
        raise WaferMapValidationError("wafer_map must be a 2-D matrix")

    if not rows:
        raise WaferMapValidationError("wafer_map must not be empty")
    if len(rows) > max_input_size:
        raise WaferMapValidationError(
            f"wafer_map height must be <= {max_input_size} pixels"
        )
    if not all(_is_row(row) for row in rows):
        raise WaferMapValidationError("wafer_map must contain only rows of values")

    width = len(rows[0])
    if width == 0:
        raise WaferMapValidationError("wafer_map rows must not be empty")
    if width > max_input_size:
        raise WaferMapValidationError(
            f"wafer_map width must be <= {max_input_size} pixels"
        )
    if any(len(row) != width for row in rows):
        raise WaferMapValidationError("wafer_map must be rectangular")

    allowed = set(int(item) for item in allowed_values)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if not _is_integer(value):
                raise WaferMapValidationError(
                    "wafer_map values must be integers in {0, 1, 2}; "
                    f"found non-integer at [{row_index}][{column_index}]"
                )
            integer_value = int(value)
            if integer_value not in allowed:
                allowed_text = ", ".join(str(item) for item in sorted(allowed))
                raise WaferMapValidationError(
                    f"wafer_map values must be one of {{{allowed_text}}}; "
                    f"found {integer_value} at [{row_index}][{column_index}]"
                )

    return len(rows), width


def resize_nearest(
    wafer_map: Any,
    *,
    target_size: tuple[int, int] = (64, 64),
    max_input_size: int = 512,
    allowed_values: Sequence[int] = ALLOWED_VALUES,
) -> np.ndarray:
    """Validate and resize a wafer map with nearest-neighbour sampling.

    A small NumPy implementation keeps the API image-processing dependency
    light while matching the categorical semantics of OpenCV's
    ``INTER_NEAREST`` operation.
    """

    validate_wafer_map(
        wafer_map,
        max_input_size=max_input_size,
        allowed_values=allowed_values,
    )
    source = np.asarray(wafer_map, dtype=np.uint8)
    target_height, target_width = target_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_size dimensions must be positive")

    # Floor-based source indices preserve categories and are equivalent to
    # nearest-neighbour sampling for the integer categorical grid used here.
    source_rows = np.floor(
        np.arange(target_height, dtype=np.float64) * source.shape[0] / target_height
    ).astype(np.int64)
    source_columns = np.floor(
        np.arange(target_width, dtype=np.float64) * source.shape[1] / target_width
    ).astype(np.int64)
    source_rows = np.clip(source_rows, 0, source.shape[0] - 1)
    source_columns = np.clip(source_columns, 0, source.shape[1] - 1)
    return source[np.ix_(source_rows, source_columns)].astype(np.uint8, copy=False)


def normalize_for_model(
    processed_map: np.ndarray,
    *,
    normalization_divisor: float = 2.0,
) -> np.ndarray:
    """Convert a processed map to the CNN input shape ``(1, H, W, 1)``."""

    if processed_map.ndim != 2:
        raise ValueError("processed_map must be a 2-D array")
    if normalization_divisor <= 0:
        raise ValueError("normalization_divisor must be positive")
    normalized = processed_map.astype(np.float32) / float(normalization_divisor)
    return normalized[np.newaxis, ..., np.newaxis]


def calculate_defect_ratio(processed_map: np.ndarray) -> float:
    """Return defect pixels divided by non-background pixels.

    This follows the notebook's ``calculate_defect_ratio`` definition:
    ``2`` values are defects, and ``0`` background values are excluded from
    the denominator.  An empty valid wafer is reported as zero rather than
    producing a division-by-zero error.
    """

    if processed_map.ndim != 2:
        raise ValueError("processed_map must be a 2-D array")
    valid_pixels = int(np.count_nonzero(processed_map > 0))
    if valid_pixels == 0:
        return 0.0
    defect_pixels = int(np.count_nonzero(processed_map == 2))
    return float(defect_pixels / valid_pixels)


def preprocess_wafer_map(wafer_map: Any, settings: Settings) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate, resize, normalize, and calculate the defect ratio."""

    processed_map = resize_nearest(
        wafer_map,
        target_size=settings.target_size,
        max_input_size=settings.max_input_size,
        allowed_values=ALLOWED_VALUES,
    )
    model_input = normalize_for_model(
        processed_map,
        normalization_divisor=settings.normalization_divisor,
    )
    defect_ratio = calculate_defect_ratio(processed_map)
    return processed_map, model_input, defect_ratio

