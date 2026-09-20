from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from training.export_model import (  # noqa: E402
    DEFAULT_MIN_SAMPLES_PER_CLASS,
    _load_dataset,
)


BASE_LABELS = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Normal",
    "Random",
    "Scratch",
]


def _write_dataset(path: Path, horizontal_stripes_count: int) -> None:
    rows = []
    for label in BASE_LABELS:
        for index in range(30):
            rows.append(
                {
                    "waferMap_resized": np.zeros((64, 64), dtype=np.uint8),
                    "failureType": label,
                }
            )
    for _ in range(horizontal_stripes_count):
        rows.append(
            {
                "waferMap_resized": np.ones((64, 64), dtype=np.uint8),
                "failureType": "Horizontal_Stripes",
            }
        )
    pd.DataFrame(rows).to_pickle(path)


def test_two_sample_discovery_candidate_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "candidate.pkl"
    _write_dataset(dataset, horizontal_stripes_count=2)

    with pytest.raises(ValueError, match="Horizontal_Stripes"):
        _load_dataset(dataset)


def test_reviewed_new_label_dataset_is_loadable(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed.pkl"
    _write_dataset(
        dataset,
        horizontal_stripes_count=DEFAULT_MIN_SAMPLES_PER_CLASS,
    )

    loaded = _load_dataset(dataset)

    assert "Horizontal_Stripes" in set(loaded["failureType"])
    assert len(loaded) > 0
