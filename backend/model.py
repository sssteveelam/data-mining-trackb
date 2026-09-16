"""Model artifact loading and prediction.

TensorFlow is imported lazily.  This keeps the API importable for validation
and exposes a clear degraded state when a trained ``model.keras`` artifact has
not yet been exported; prediction requests return HTTP 503 in that state.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)

DEFAULT_LABELS: tuple[str, ...] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Normal",
    "Random",
    "Scratch",
)
FALLBACK_MODEL_VERSION = "cnn-9class-unavailable"
EXPECTED_CLASS_COUNT = 9


class ModelUnavailableError(RuntimeError):
    """Raised when a request cannot be served by a trained model artifact."""


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path.name} must contain a JSON object or array")
    return value


def _load_labels(path: Path) -> list[str]:
    raw = _read_json(path)
    if raw is None:
        return list(DEFAULT_LABELS)
    if isinstance(raw, dict):
        raw = raw.get("labels", raw.get("classes"))
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError("labels.json must contain a non-empty string list or {'labels': [...]}")
    labels = [item.strip() for item in raw]
    if (
        len(labels) != EXPECTED_CLASS_COUNT
        or "Horizontal_Stripes" in labels
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(
            "production artifact must contain exactly the nine trained classes "
            "and must not contain Horizontal_Stripes"
        )
    return labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_probabilities(values: Any, class_count: int) -> np.ndarray:
    """Validate model output and normalize it to a probability vector."""

    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim == 2:
        if probabilities.shape[0] < 1:
            raise ValueError("model returned an empty batch")
        probabilities = probabilities[0]
    probabilities = probabilities.reshape(-1)
    if probabilities.size != class_count:
        raise ValueError(
            f"model returned {probabilities.size} scores; expected {class_count}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model returned non-finite scores")
    if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
        # Some exported models may expose logits.  Convert those safely rather
        # than returning invalid probabilities from the API.
        shifted = probabilities - np.max(probabilities)
        exponentials = np.exp(np.clip(shifted, -80.0, 80.0))
        denominator = float(exponentials.sum())
        if denominator <= 0.0:
            raise ValueError("model returned scores that cannot be normalized")
        probabilities = exponentials / denominator
    else:
        total = float(probabilities.sum())
        probabilities = probabilities / total if total else probabilities
    return probabilities.astype(np.float64, copy=False)


@dataclass
class ArtifactStatus:
    model_version: str
    labels: list[str]
    loaded: bool
    degraded: bool
    load_error: str | None
    metrics: dict[str, Any] | None
    manifest: dict[str, Any] | None
    preprocess: dict[str, Any] | None
    model_sha256: str | None


class ModelLoader:
    """Load a Keras artifact once and provide a safe unavailable state."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        labels: list[str] | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._model: Any | None = None
        self._fallback_labels = labels or list(DEFAULT_LABELS)
        self._status = self._load()

    @property
    def status(self) -> ArtifactStatus:
        return self._status

    def _load(self) -> ArtifactStatus:
        labels = list(self._fallback_labels)
        manifest: dict[str, Any] | None = None
        metrics: dict[str, Any] | None = None
        preprocess: dict[str, Any] | None = None
        model_path = self.artifact_dir / "model.keras"

        try:
            loaded_labels = _load_labels(self.artifact_dir / "labels.json")
            labels = loaded_labels
            raw_manifest = _read_json(self.artifact_dir / "manifest.json")
            if isinstance(raw_manifest, dict):
                manifest = raw_manifest
            raw_metrics = _read_json(self.artifact_dir / "metrics.json")
            if isinstance(raw_metrics, dict):
                metrics = raw_metrics
            raw_preprocess = _read_json(self.artifact_dir / "preprocess.json")
            if isinstance(raw_preprocess, dict):
                preprocess = raw_preprocess

            if not model_path.exists():
                return ArtifactStatus(
                    model_version=str((manifest or {}).get("model_version") or FALLBACK_MODEL_VERSION),
                    labels=labels,
                    loaded=False,
                    degraded=True,
                    load_error=f"Model artifact not found: {model_path}",
                    metrics=metrics,
                    manifest=manifest,
                    preprocess=preprocess,
                    model_sha256=None,
                )

            try:
                import tensorflow as tf  # type: ignore
            except Exception as exc:  # pragma: no cover - depends on runtime image
                return ArtifactStatus(
                    model_version=str((manifest or {}).get("model_version") or FALLBACK_MODEL_VERSION),
                    labels=labels,
                    loaded=False,
                    degraded=True,
                    load_error=f"TensorFlow is unavailable: {exc}",
                    metrics=metrics,
                    manifest=manifest,
                    preprocess=preprocess,
                    model_sha256=_sha256(model_path),
                )

            self._model = tf.keras.models.load_model(model_path, compile=False)
            return ArtifactStatus(
                model_version=str((manifest or {}).get("model_version") or "cnn-9class-v1"),
                labels=labels,
                loaded=True,
                degraded=False,
                load_error=None,
                metrics=metrics,
                manifest=manifest,
                preprocess=preprocess,
                model_sha256=_sha256(model_path),
            )
        except Exception as exc:
            LOGGER.exception("Unable to load model artifacts from %s", self.artifact_dir)
            self._model = None
            return ArtifactStatus(
                model_version=str((manifest or {}).get("model_version") or FALLBACK_MODEL_VERSION),
                labels=labels,
                loaded=False,
                degraded=True,
                load_error=str(exc),
                metrics=metrics,
                manifest=manifest,
                preprocess=preprocess,
                model_sha256=_sha256(model_path) if model_path.exists() else None,
            )

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        if self._model is None:
            message = self._status.load_error or "No trained model artifact is loaded"
            raise ModelUnavailableError(message)
        try:
            raw = self._model.predict(model_input, verbose=0)
            return _normalise_probabilities(raw, len(self._status.labels))
        except Exception as exc:
            # Keep the process alive if a mounted artifact is corrupt or has
            # an incompatible input signature.  The status becomes degraded so
            # health/model-info expose the incident.
            LOGGER.exception("Model prediction failed; switching to fallback")
            self._model = None
            self._status = ArtifactStatus(
                model_version=self._status.model_version,
                labels=self._status.labels,
                loaded=False,
                degraded=True,
                load_error=f"Model prediction failed: {exc}",
                metrics=self._status.metrics,
                manifest=self._status.manifest,
                preprocess=self._status.preprocess,
                model_sha256=self._status.model_sha256,
            )
            raise ModelUnavailableError(str(self._status.load_error)) from exc
