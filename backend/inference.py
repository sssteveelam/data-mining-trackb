"""Inference service combining preprocessing and model prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings
from .model import ModelLoader
from .preprocessing import preprocess_wafer_map


@dataclass(frozen=True)
class Prediction:
    model_version: str
    predicted_label: str
    confidence: float
    review_required: bool
    top_k: list[dict[str, float | str]]
    input_shape: tuple[int, int]
    processed_shape: tuple[int, int]
    defect_ratio: float
    processed_map: list[list[int]]
    degraded: bool
    notice: str | None


class InferenceService:
    """Long-lived service; the model loader is initialized once per process."""

    def __init__(self, settings: Settings, loader: ModelLoader | None = None) -> None:
        self.settings = settings
        self.loader = loader or ModelLoader(settings.artifact_dir)

    @property
    def status(self):
        return self.loader.status

    def health(self) -> dict[str, Any]:
        status = self.loader.status
        return {
            "service": self.settings.service_name,
            "status": "degraded" if status.degraded else "ok",
            "model_status": "ready" if status.loaded else "degraded",
            "model_loaded": status.loaded,
            "model_version": status.model_version,
            "detail": status.load_error,
        }

    def model_info(self) -> dict[str, Any]:
        status = self.loader.status
        preprocess = status.preprocess or {}
        target_size_raw = preprocess.get("target_size", self.settings.target_size)
        try:
            target_size = tuple(int(item) for item in target_size_raw)
            if len(target_size) != 2 or any(item <= 0 for item in target_size):
                raise ValueError
        except (TypeError, ValueError):
            target_size = self.settings.target_size
        return {
            "model_version": status.model_version,
            "model_status": "ready" if status.loaded else "degraded",
            "model_loaded": status.loaded,
            "degraded": status.degraded,
            "labels": status.labels,
            "class_count": len(status.labels),
            "target_size": target_size,
            "normalization_divisor": float(
                preprocess.get("normalization_divisor", self.settings.normalization_divisor)
            ),
            "allowed_values": [0, 1, 2],
            "max_input_size": self.settings.max_input_size,
            "review_threshold": self.settings.review_threshold,
            "artifact_dir": str(self.settings.artifact_dir),
            "metrics": status.metrics,
            "manifest": status.manifest,
            "load_error": status.load_error,
        }

    def predict(self, wafer_map: Any) -> Prediction:
        # Keep the public preprocessing settings authoritative.  Artifact
        # metadata is reported through /model-info and should be generated with
        # the same 64x64 + /2 contract by the training package.
        processed_map, model_input, defect_ratio = preprocess_wafer_map(
            wafer_map, self.settings
        )
        probabilities = np.asarray(self.loader.predict(model_input), dtype=np.float64)
        labels = self.loader.status.labels
        if probabilities.ndim != 1 or probabilities.size != len(labels):
            raise RuntimeError(
                f"model returned {probabilities.size} probabilities for {len(labels)} labels"
            )
        probabilities = np.clip(probabilities, 0.0, 1.0)
        total = float(probabilities.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise RuntimeError("model returned an invalid probability vector")
        probabilities /= total

        order = np.argsort(-probabilities, kind="stable")
        top_k = [
            {
                "label": labels[int(index)],
                "probability": float(probabilities[int(index)]),
            }
            for index in order[: min(3, len(labels))]
        ]
        top_index = int(order[0])
        confidence = float(probabilities[top_index])
        return Prediction(
            model_version=self.loader.status.model_version,
            predicted_label=labels[top_index],
            confidence=confidence,
            review_required=confidence < self.settings.review_threshold
            or self.loader.status.degraded,
            top_k=top_k,
            input_shape=(len(wafer_map), len(wafer_map[0])),
            processed_shape=(int(processed_map.shape[0]), int(processed_map.shape[1])),
            defect_ratio=float(defect_ratio),
            processed_map=processed_map.astype(int).tolist(),
            degraded=self.loader.status.degraded,
            notice=None,
        )
