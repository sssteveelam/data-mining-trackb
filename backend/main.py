"""FastAPI application for wafer-map classification."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .inference import InferenceService
from .model import ModelUnavailableError
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    TopKPrediction,
)


def create_app(
    *,
    settings: Settings | None = None,
    service: InferenceService | None = None,
) -> FastAPI:
    """Create an app, allowing tests to inject an in-memory service."""

    resolved_settings = settings or Settings.from_env()
    inference_service = service or InferenceService(resolved_settings)
    app = FastAPI(
        title="Wafer Map Classifier API",
        description=(
            "CNN inference for categorical wafer maps. "
            "The runtime accepts JSON matrices containing 0, 1, and 2."
        ),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.frontend_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.settings = resolved_settings
    app.state.inference_service = inference_service

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    def health() -> dict[str, Any]:
        return inference_service.health()

    @app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["system"])
    def model_info() -> dict[str, Any]:
        return inference_service.model_info()

    @app.post("/api/v1/predict", response_model=PredictResponse, tags=["inference"])
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            result = inference_service.predict(request.wafer_map)
        except ModelUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MODEL_UNAVAILABLE",
                    "message": (
                        "Chưa có model đã train. "
                        "Hãy export và mount model.keras cùng các artifact trước khi dự đoán."
                    ),
                    "reason": str(exc),
                },
            ) from exc
        return PredictResponse(
            model_version=result.model_version,
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            review_required=result.review_required,
            top_k=[TopKPrediction(**item) for item in result.top_k],
            input_shape=result.input_shape,
            processed_shape=result.processed_shape,
            defect_ratio=result.defect_ratio,
            processed_map=result.processed_map,
            degraded=result.degraded,
            notice=result.notice,
        )

    return app


app = create_app()
