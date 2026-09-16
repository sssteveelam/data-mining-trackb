from __future__ import annotations

import numpy as np
import pytest

from backend.preprocessing import (
    WaferMapValidationError,
    calculate_defect_ratio,
    normalize_for_model,
    resize_nearest,
    validate_wafer_map,
)


def test_resize_preserves_categorical_values_and_shape() -> None:
    processed = resize_nearest([[0, 1], [2, 1]], target_size=(64, 64))
    assert processed.shape == (64, 64)
    assert set(np.unique(processed).tolist()) <= {0, 1, 2}
    assert processed.dtype == np.uint8


def test_resize_uses_nearest_neighbour_for_a_small_grid() -> None:
    processed = resize_nearest([[0, 1], [2, 1]], target_size=(4, 4))
    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(processed, expected)


@pytest.mark.parametrize(
    "wafer_map, message",
    [
        ([], "empty"),
        ([[0, 1], [1]], "rectangular"),
        ([[0, 3]], "one of"),
        ([[0.0, 1]], "integers"),
        ([[True, 1]], "integers"),
        ([["0", 1]], "integers"),
    ],
)
def test_validate_rejects_invalid_maps(wafer_map, message: str) -> None:
    with pytest.raises(WaferMapValidationError, match=message):
        validate_wafer_map(wafer_map)


def test_validate_rejects_oversized_maps() -> None:
    with pytest.raises(WaferMapValidationError, match="height"):
        validate_wafer_map([[0] * 2 for _ in range(513)])


def test_defect_ratio_excludes_background() -> None:
    processed = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.uint8)
    assert calculate_defect_ratio(processed) == pytest.approx(2 / 4)
    assert calculate_defect_ratio(np.zeros((4, 4), dtype=np.uint8)) == 0.0


def test_normalize_adds_batch_and_channel_dimensions() -> None:
    processed = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    normalized = normalize_for_model(processed)
    assert normalized.shape == (1, 2, 2, 1)
    assert normalized.dtype == np.float32
    assert normalized[0, 1, 0, 0] == pytest.approx(1.0)

