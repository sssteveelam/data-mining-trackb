import numpy as np
import pytest

from training.preprocess import (
    calculate_defect_ratio,
    normalize_wafer_map,
    resize_wafer_map,
    validate_wafer_map,
)


def test_resize_preserves_categorical_values():
    matrix = [[0, 1, 2], [2, 1, 0]]
    resized = resize_wafer_map(matrix)

    assert resized.shape == (64, 64)
    assert set(np.unique(resized)).issubset({0, 1, 2})


def test_normalize_adds_channel_and_scales():
    tensor = normalize_wafer_map([[0, 1], [2, 0]])

    assert tensor.shape == (64, 64, 1)
    assert tensor.dtype == np.float32
    assert float(tensor.max()) <= 1.0


def test_defect_ratio_ignores_background():
    assert calculate_defect_ratio([[0, 1, 2, 2]]) == pytest.approx(2 / 3)
    assert calculate_defect_ratio([[0, 0]]) == 0.0


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        [[0, 1], [1]],
        [[0, 3]],
        [["bad"]],
        [[float("nan")]],
    ],
)
def test_invalid_maps_are_rejected(matrix):
    with pytest.raises(ValueError):
        validate_wafer_map(matrix)


def test_oversized_map_is_rejected():
    with pytest.raises(ValueError):
        validate_wafer_map(np.zeros((513, 1), dtype=np.uint8))

