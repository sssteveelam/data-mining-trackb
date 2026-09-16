"""Runtime configuration for the wafer-map API.

The backend intentionally keeps training-only paths out of runtime code.  A
trained model can be mounted through ``ARTIFACT_DIR``; when it is not mounted,
the API remains available in an explicitly degraded mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _float_between(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


@dataclass(frozen=True)
class Settings:
    """Settings used by preprocessing, model loading, and API responses."""

    artifact_dir: Path
    target_height: int = 64
    target_width: int = 64
    normalization_divisor: float = 2.0
    max_input_size: int = 512
    review_threshold: float = 0.60
    service_name: str = "wafer-map-api"
    frontend_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )

    @property
    def target_size(self) -> tuple[int, int]:
        return (self.target_height, self.target_width)

    @classmethod
    def from_env(cls) -> "Settings":
        default_artifact_dir = PROJECT_ROOT / "artifacts"
        configured_dir = os.getenv("ARTIFACT_DIR")
        artifact_dir = Path(configured_dir).expanduser() if configured_dir else default_artifact_dir
        configured_origins = os.getenv("FRONTEND_ORIGINS")
        if configured_origins:
            frontend_origins = tuple(
                origin.strip()
                for origin in configured_origins.split(",")
                if origin.strip()
            )
        else:
            frontend_origins = cls.frontend_origins
        return cls(
            artifact_dir=artifact_dir,
            target_height=_positive_int("TARGET_HEIGHT", 64),
            target_width=_positive_int("TARGET_WIDTH", 64),
            normalization_divisor=_float_between("NORMALIZATION_DIVISOR", 2.0, 0.000001, 1_000_000.0),
            max_input_size=_positive_int("MAX_INPUT_SIZE", 512),
            review_threshold=_float_between("REVIEW_THRESHOLD", 0.60, 0.0, 1.0),
            frontend_origins=frontend_origins,
        )
