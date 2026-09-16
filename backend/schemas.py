"""Pydantic request and response schemas for the public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from .preprocessing import WaferMapValidationError, validate_wafer_map


class PredictionMetadata(BaseModel):
    """Optional metadata accepted alongside a wafer map.

    The fields used by the notebook are explicit, while extra fields are
    preserved to make the demo convenient for lot-level integrations.
    """

    model_config = ConfigDict(extra="allow")

    lot_name: str | None = Field(default=None, max_length=256)
    wafer_index: int | None = Field(default=None, ge=0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wafer_map: list[list[StrictInt]]
    metadata: PredictionMetadata | dict[str, Any] | None = None

    @field_validator("wafer_map")
    @classmethod
    def wafer_map_must_be_valid(cls, value: list[list[int]]) -> list[list[int]]:
        try:
            validate_wafer_map(value)
        except WaferMapValidationError as exc:
            raise ValueError(str(exc)) from exc
        return value


class TopKPrediction(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool
    top_k: list[TopKPrediction]
    input_shape: tuple[int, int]
    processed_shape: tuple[int, int]
    defect_ratio: float = Field(ge=0.0, le=1.0)
    processed_map: list[list[int]]
    degraded: bool = False
    notice: str | None = None


class HealthResponse(BaseModel):
    service: str
    status: str
    model_status: str
    model_loaded: bool
    model_version: str
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_status: str
    model_loaded: bool
    degraded: bool
    labels: list[str]
    class_count: int
    target_size: tuple[int, int]
    normalization_divisor: float
    allowed_values: list[int]
    max_input_size: int
    review_threshold: float
    artifact_dir: str
    metrics: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    load_error: str | None = None
